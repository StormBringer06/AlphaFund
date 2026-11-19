"""Reusable chronological walk-forward backtest for pairs-trading models."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Callable, Protocol

import pandas as pd

from base_model import NaivePairsTradingModel


_REQUIRED_DATA_COLUMNS = {
    "date",
    "pair_id",
    "adjusted_close_a",
    "adjusted_close_b",
    "return_a",
    "return_b",
}
_TARGET_COLUMNS = ("target_weight_a", "target_weight_b")
_GROUP_COLUMNS = ("model_name", "pair_id", "fold_id", "test_year")


class TradingModel(Protocol):
    """Interface required by the walk-forward runner."""

    def fit(self, train_data: pd.DataFrame) -> object | None: ...

    def predict(self, test_data: pd.DataFrame) -> pd.DataFrame: ...


@dataclass(frozen=True)
class WalkForwardFold:
    fold_id: int
    train_start: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp

    @property
    def test_year(self) -> int:
        return self.test_start.year


def build_walk_forward_folds(
    start_year: int = 2014,
    end_year: int = 2023,
    train_years: int = 4,
    test_years: int = 1,
    step_years: int = 1,
) -> list[WalkForwardFold]:
    """Create exclusive-end calendar folds with expanding calendar boundaries."""
    if start_year > end_year:
        raise ValueError("start_year must be less than or equal to end_year.")
    if train_years < 1 or test_years < 1 or step_years < 1:
        raise ValueError("train_years, test_years, and step_years must be positive.")

    folds: list[WalkForwardFold] = []
    last_start_year = end_year - test_years + 1
    for fold_id, test_year in enumerate(
        range(start_year, last_start_year + 1, step_years), start=1
    ):
        test_start = pd.Timestamp(year=test_year, month=1, day=1)
        folds.append(
            WalkForwardFold(
                fold_id=fold_id,
                train_start=test_start - pd.DateOffset(years=train_years),
                test_start=test_start,
                test_end=test_start + pd.DateOffset(years=test_years),
            )
        )
    if not folds:
        raise ValueError("The requested years do not contain a complete test window.")
    return folds


def run_walk_forward_backtest(
    data: pd.DataFrame,
    model_factory: Callable[[], TradingModel] | None = None,
    start_year: int = 2014,
    end_year: int = 2023,
    train_years: int = 4,
    test_years: int = 1,
    step_years: int = 1,
    transaction_cost_bps: float = 0.0,
    model_name: str | None = None,
) -> pd.DataFrame:
    """Fit a fresh model per pair/fold and return daily out-of-sample results.

    A model's close-of-day target weights are lagged one pair observation before
    applying returns. Each fold starts flat and closes its position at the end.
    """
    if not math.isfinite(transaction_cost_bps) or transaction_cost_bps < 0:
        raise ValueError("transaction_cost_bps must be finite and non-negative.")
    pair_prices = _prepare_pair_prices(data)
    folds = build_walk_forward_folds(
        start_year=start_year,
        end_year=end_year,
        train_years=train_years,
        test_years=test_years,
        step_years=step_years,
    )
    factory = model_factory or NaivePairsTradingModel
    result_frames: list[pd.DataFrame] = []

    for pair_id, pair in pair_prices.groupby("pair_id", sort=True):
        pair = pair.sort_values("date").reset_index(drop=True)
        for fold in folds:
            train = pair.loc[
                (pair["date"] >= fold.train_start)
                & (pair["date"] < fold.test_start)
            ].copy()
            test = pair.loc[
                (pair["date"] >= fold.test_start) & (pair["date"] < fold.test_end)
            ].copy()
            if train.empty:
                raise ValueError(
                    f"No training data for pair {pair_id} in fold {fold.fold_id}."
                )
            if test.empty:
                raise ValueError(
                    f"No test data for pair {pair_id} in fold {fold.fold_id}."
                )
            if test[["return_a", "return_b"]].isna().any().any():
                raise ValueError(
                    f"Test returns contain missing values for pair {pair_id} "
                    f"in fold {fold.fold_id}."
                )

            model = factory()
            model.fit(train)
            predictions = model.predict(test.copy())
            if isinstance(predictions, pd.DataFrame) and "date" in predictions.columns:
                predictions = predictions.copy()
                predictions["date"] = pd.to_datetime(
                    predictions["date"], errors="coerce"
                )
            _validate_predictions(predictions, test, pair_id, fold)
            forecast_columns = [
                column
                for column in predictions.columns
                if column == "date" or column not in test.columns
            ]
            forecast = predictions[forecast_columns].copy()
            results = test.merge(
                forecast,
                on="date",
                how="left",
                validate="one_to_one",
            )
            results["fold_id"] = fold.fold_id
            results["train_start"] = fold.train_start
            results["test_start"] = fold.test_start
            results["test_end"] = fold.test_end
            results["test_year"] = fold.test_year
            results["model_name"] = model_name or type(model).__name__
            result_frames.append(results)

    daily = pd.concat(result_frames, ignore_index=True)
    return apply_positions_to_returns(
        daily, transaction_cost_bps=transaction_cost_bps
    )


def apply_positions_to_returns(
    predictions: pd.DataFrame, transaction_cost_bps: float = 0.0
) -> pd.DataFrame:
    """Apply lagged targets to returns, optional costs, and fold-level liquidation."""
    if not math.isfinite(transaction_cost_bps) or transaction_cost_bps < 0:
        raise ValueError("transaction_cost_bps must be finite and non-negative.")
    required = {
        "date",
        "pair_id",
        "fold_id",
        "return_a",
        "return_b",
        *_TARGET_COLUMNS,
    }
    missing = required.difference(predictions.columns)
    if missing:
        raise ValueError(
            "Predictions are missing columns: " + ", ".join(sorted(missing)) + "."
        )
    if predictions.empty:
        raise ValueError("Predictions must not be empty.")

    daily = predictions.copy().sort_values(
        ["pair_id", "fold_id", "date"]
    ).reset_index(drop=True)
    for column in ("return_a", "return_b", *_TARGET_COLUMNS):
        values = pd.to_numeric(daily[column], errors="coerce")
        if values.isna().any() or not values.map(math.isfinite).all():
            raise ValueError(f"{column} must contain finite values.")
        daily[column] = values

    group_columns = ["pair_id", "fold_id"]
    if "model_name" in daily.columns:
        group_columns.insert(0, "model_name")
    grouped = daily.groupby(group_columns, sort=False)
    daily["position_weight_a"] = grouped["target_weight_a"].shift(1).fillna(0.0)
    daily["position_weight_b"] = grouped["target_weight_b"].shift(1).fillna(0.0)
    daily["position_gross_exposure"] = (
        daily["position_weight_a"].abs() + daily["position_weight_b"].abs()
    )

    prior_target_a = grouped["target_weight_a"].shift(1).fillna(0.0)
    prior_target_b = grouped["target_weight_b"].shift(1).fillna(0.0)
    rebalance_turnover = (
        (daily["target_weight_a"] - prior_target_a).abs()
        + (daily["target_weight_b"] - prior_target_b).abs()
    )
    daily["turnover"] = rebalance_turnover
    last_row_in_fold = ~daily.duplicated(group_columns, keep="last")
    daily.loc[last_row_in_fold, "turnover"] = daily.loc[
        last_row_in_fold, "position_gross_exposure"
    ]

    daily["transaction_cost"] = daily["turnover"] * (
        transaction_cost_bps / 10_000.0
    )
    daily["gross_strategy_return"] = (
        daily["position_weight_a"] * daily["return_a"]
        + daily["position_weight_b"] * daily["return_b"]
    )
    daily["strategy_return"] = (
        daily["gross_strategy_return"] - daily["transaction_cost"]
    )
    wealth = (1.0 + daily["strategy_return"]).groupby(
        [daily[column] for column in group_columns], sort=False
    ).cumprod()
    daily["cumulative_return"] = wealth - 1.0
    return daily


def summarize_backtest(daily_results: pd.DataFrame) -> pd.DataFrame:
    """Summarize compounded return, risk, exposure, and turnover per pair/fold."""
    required = set(_GROUP_COLUMNS) | {
        "strategy_return",
        "position_gross_exposure",
        "turnover",
    }
    missing = required.difference(daily_results.columns)
    if missing:
        raise ValueError(
            "Backtest results are missing columns: "
            + ", ".join(sorted(missing))
            + "."
        )
    if daily_results.empty:
        raise ValueError("Backtest results must not be empty.")

    summaries: list[dict[str, object]] = []
    for keys, group in daily_results.groupby(list(_GROUP_COLUMNS), sort=True):
        returns = group["strategy_return"]
        total_return = float((1.0 + returns).prod() - 1.0)
        daily_volatility = float(returns.std(ddof=1))
        annualized_return = (
            (1.0 + total_return) ** (252.0 / len(returns)) - 1.0
            if total_return > -1.0
            else float("nan")
        )
        wealth = (1.0 + returns).cumprod()
        drawdown = wealth / wealth.cummax().clip(lower=1.0) - 1.0
        summaries.append(
            {
                **dict(zip(_GROUP_COLUMNS, keys)),
                "observations": len(group),
                "total_return": total_return,
                "annualized_return": annualized_return,
                "annualized_volatility": daily_volatility * math.sqrt(252.0),
                "sharpe_ratio": (
                    float(returns.mean() / daily_volatility * math.sqrt(252.0))
                    if daily_volatility > 0
                    else float("nan")
                ),
                "max_drawdown": float(drawdown.min()),
                "average_gross_exposure": float(
                    group["position_gross_exposure"].mean()
                ),
                "average_daily_turnover": float(group["turnover"].mean()),
                "total_transaction_cost": float(group["transaction_cost"].sum()),
            }
        )
    return pd.DataFrame(summaries)


def _prepare_pair_prices(data: pd.DataFrame) -> pd.DataFrame:
    missing = _REQUIRED_DATA_COLUMNS.difference(data.columns)
    if missing:
        raise ValueError(
            "Pair data is missing columns: " + ", ".join(sorted(missing)) + "."
        )
    if data.empty:
        raise ValueError("Pair data must not be empty.")

    prepared = data.copy()
    prepared["date"] = pd.to_datetime(prepared["date"], errors="coerce")
    if prepared["date"].isna().any():
        raise ValueError("Pair data contains invalid dates.")
    if prepared["pair_id"].isna().any():
        raise ValueError("Pair data contains missing pair_id values.")
    if prepared.duplicated(["pair_id", "date"]).any():
        raise ValueError("Pair data contains duplicate pair/date rows.")

    for column in ("return_a", "return_b"):
        numeric = pd.to_numeric(prepared[column], errors="coerce")
        invalid = prepared[column].notna() & numeric.isna()
        if invalid.any() or not numeric.dropna().map(math.isfinite).all():
            raise ValueError(f"{column} contains invalid return values.")
        prepared[column] = numeric
    return prepared.sort_values(["pair_id", "date"]).reset_index(drop=True)


def _validate_predictions(
    predictions: pd.DataFrame,
    test_data: pd.DataFrame,
    pair_id: object,
    fold: WalkForwardFold,
) -> None:
    if not isinstance(predictions, pd.DataFrame):
        raise TypeError("Model predict must return a pandas DataFrame.")
    required = {"date", *_TARGET_COLUMNS}
    missing = required.difference(predictions.columns)
    if missing:
        raise ValueError(
            f"Model predictions for pair {pair_id}, fold {fold.fold_id} are "
            "missing columns: "
            + ", ".join(sorted(missing))
            + "."
        )
    if predictions.empty:
        raise ValueError(
            f"Model returned no predictions for pair {pair_id}, fold {fold.fold_id}."
        )

    dates = pd.to_datetime(predictions["date"], errors="coerce")
    if dates.isna().any() or dates.duplicated().any():
        raise ValueError("Model prediction dates must be valid and unique.")
    predictions["date"] = dates
    if set(dates) != set(test_data["date"]):
        raise ValueError(
            f"Model must return exactly one prediction per test date for pair "
            f"{pair_id}, fold {fold.fold_id}."
        )
    for column in _TARGET_COLUMNS:
        values = pd.to_numeric(predictions[column], errors="coerce")
        if values.isna().any() or not values.map(math.isfinite).all():
            raise ValueError(f"Model prediction {column} must contain finite values.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data") / "pair_prices.csv",
        help="Input pair-prices CSV (default: data/pair_prices.csv).",
    )
    parser.add_argument("--start-year", type=int, default=2014)
    parser.add_argument("--end-year", type=int, default=2023)
    parser.add_argument("--train-years", type=int, default=4)
    parser.add_argument("--test-years", type=int, default=1)
    parser.add_argument("--step-years", type=int, default=1)
    parser.add_argument(
        "--transaction-cost-bps",
        type=float,
        default=0.0,
        help="One-way transaction cost in basis points per dollar traded.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("backtest_results"),
        help="Directory for daily results and summary CSVs.",
    )
    args = parser.parse_args()

    data = pd.read_csv(args.input, parse_dates=["date"])
    daily = run_walk_forward_backtest(
        data,
        start_year=args.start_year,
        end_year=args.end_year,
        train_years=args.train_years,
        test_years=args.test_years,
        step_years=args.step_years,
        transaction_cost_bps=args.transaction_cost_bps,
    )
    summary = summarize_backtest(daily)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    daily.to_csv(args.output_dir / "daily_results.csv", index=False)
    summary.to_csv(args.output_dir / "summary.csv", index=False)
    print(
        f"Wrote {len(daily)} daily rows and {len(summary)} pair/fold summaries "
        f"to {args.output_dir}."
    )


if __name__ == "__main__":
    main()
