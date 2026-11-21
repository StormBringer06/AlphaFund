"""V4: the V3 signal and pairs, levered to a portfolio volatility target.

Signal and pairs: identical to V3 (EWMA z-score, VIX size overlay with
``vix_scale`` chosen yearly from {0, 0.25}, fixed no-trade band 0.2; top 5 pairs
from the 10-year point-in-time screen). Trading up to 10 pairs
(``--input data/pit10_pair_prices.csv --selection data/pit10_pair_selection.csv``)
was tried: the screen only found 3-8 per year and the extra pairs lowered the
unlevered Sharpe (0.53 -> 0.45), so the default is back to the top 5.

Change from V3:

* **Portfolio volatility target.** All pair weights are multiplied by a
  portfolio leverage ``L = min(target_vol / estimated_vol, max_leverage)``,
  where ``estimated_vol`` is the annualized volatility of the *unlevered*
  equal-weight portfolio. Two estimators are reported:

  - ``v4_static_vol``: one ``L`` per test year, from the volatility of the
    inner walk-forward (the two years before the test year, chosen
    parameters). Uses only pre-year data.
  - ``v4_dynamic_vol``: daily ``L``, from an EWMA (``--vol-halflife``) of the
    unlevered portfolio's realized daily returns up to and including that
    day's close; applied to the next day's returns like every other target.
    Seeded with the first year's inner walk-forward returns.

Leverage changes count as turnover. Results are gross of fees first; the net
table adds ``--net-cost-bps`` trading costs and ``--borrow-bps`` per year on the
short leg (taken as half of gross exposure).
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

from backtest import apply_positions_to_returns, run_walk_forward_backtest, summarize_backtest
from base_model import NaivePairsTradingModel
from improved_models.regression_model import portfolio_metrics
from improved_models.vol_scaled_naive_model import (
    VolScaledNaivePairsTradingModel,
    robustness_table,
    select_parameters,
)
from improved_models.vol_scaled_naive_model_v3 import V3_GRID


def portfolio_returns(daily: pd.DataFrame) -> pd.Series:
    """Equal-weight daily portfolio return (mean across pairs)."""
    return daily.groupby("date")["strategy_return"].mean().sort_index()


def annualized_vol(returns: pd.Series) -> float:
    return float(returns.std() * math.sqrt(252.0))


def lever(
    daily: pd.DataFrame,
    leverage: pd.Series,
    model_name: str,
    transaction_cost_bps: float,
) -> pd.DataFrame:
    """Scale every pair's target weights by that date's leverage and re-cost."""
    out = daily.copy()
    factor = out["date"].map(leverage)
    if factor.isna().any():
        raise ValueError("Leverage is missing for some dates.")
    out["leverage"] = factor
    out["target_weight_a"] = out["target_weight_a"] * factor
    out["target_weight_b"] = out["target_weight_b"] * factor
    out["model_name"] = model_name
    return apply_positions_to_returns(out, transaction_cost_bps=transaction_cost_bps)


def dynamic_leverage(
    returns: pd.Series,
    target_vol: float,
    max_leverage: float,
    halflife: float,
    min_vol: float = 0.005,
) -> pd.Series:
    """Leverage from an EWMA vol of returns up to and including each date."""
    vol = returns.ewm(halflife=halflife, min_periods=20).std() * math.sqrt(252.0)
    return (target_vol / vol.clip(lower=min_vol)).clip(upper=max_leverage)


def net_of_costs(daily: pd.DataFrame, cost_bps: float, borrow_bps: float) -> pd.DataFrame:
    """Re-cost the same positions; add borrow on the short leg (half of gross)."""
    out = apply_positions_to_returns(daily, transaction_cost_bps=cost_bps)
    borrow = out["position_gross_exposure"] * 0.5 * borrow_bps / 10_000.0 / 252.0
    out["strategy_return"] = out["strategy_return"] - borrow
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--input", type=Path, default=Path("data") / "pit_pair_prices.csv")
    parser.add_argument("--selection", type=Path, default=Path("data") / "pit_pair_selection.csv")
    parser.add_argument("--start-year", type=int, default=2014)
    parser.add_argument("--end-year", type=int, default=2023)
    parser.add_argument("--train-years", type=int, default=4)
    parser.add_argument("--inner-test-years", type=int, default=2)
    parser.add_argument("--target-vol", type=float, default=0.08)
    parser.add_argument("--max-leverage", type=float, default=4.0)
    parser.add_argument("--vol-halflife", type=float, default=60.0)
    parser.add_argument(
        "--transaction-cost-bps", type=float, default=0.0,
        help="Cost for the parameter search, leverage estimates and main (gross) results.",
    )
    parser.add_argument("--net-cost-bps", type=float, default=5.0)
    parser.add_argument("--borrow-bps", type=float, default=50.0, help="Per year, on the short leg.")
    parser.add_argument("--output-dir", type=Path, default=Path("backtest_results") / "vol_scaled_naive_v4")
    args = parser.parse_args()

    data = pd.read_csv(args.input, parse_dates=["date"])
    selection = pd.read_csv(args.selection)
    cost = args.transaction_cost_bps
    inner_train_years = args.train_years - args.inner_test_years

    frames, unlevered_frames, chosen = [], [], []
    static_by_year: dict[int, float] = {}
    seed_returns: pd.Series | None = None
    for fold_id, year in enumerate(range(args.start_year, args.end_year + 1), start=1):
        pairs = selection.loc[selection["test_year"] == year, "pair_id"].unique()
        year_data = data.loc[data["pair_id"].isin(pairs)]
        params, _ = select_parameters(
            year_data, year, cost, args.train_years, args.inner_test_years, grid=V3_GRID
        )
        signal = lambda p=params: VolScaledNaivePairsTradingModel(**p)

        # Inner walk-forward with the chosen parameters: pre-year vol estimate.
        training = year_data.loc[
            (year_data["date"] >= pd.Timestamp(year=year - args.train_years, month=1, day=1))
            & (year_data["date"] < pd.Timestamp(year=year, month=1, day=1))
        ]
        inner = run_walk_forward_backtest(
            training,
            model_factory=signal,
            start_year=year - args.inner_test_years,
            end_year=year - 1,
            train_years=inner_train_years,
            transaction_cost_bps=cost,
        )
        inner_returns = portfolio_returns(inner)
        if seed_returns is None:
            seed_returns = inner_returns
        inner_vol = annualized_vol(inner_returns)
        static_by_year[year] = min(args.target_vol / inner_vol, args.max_leverage)
        chosen.append({
            "test_year": year, **params, "n_pairs": len(pairs),
            "inner_vol": inner_vol, "static_leverage": static_by_year[year],
            "pairs": ",".join(pairs),
        })
        print(
            f"{year}: {len(pairs)} pairs, vix_scale={params['vix_scale']}, "
            f"inner vol {inner_vol:.2%}, static leverage {static_by_year[year]:.2f}",
            flush=True,
        )

        models = {
            "naive_zscore": NaivePairsTradingModel,
            "naive_ewma_z": VolScaledNaivePairsTradingModel,
            "v4_unlevered": signal,
        }
        for name, factory in models.items():
            daily = run_walk_forward_backtest(
                year_data,
                model_factory=factory,
                start_year=year,
                end_year=year,
                train_years=args.train_years,
                transaction_cost_bps=cost,
                model_name=name,
            )
            daily["fold_id"] = fold_id
            frames.append(daily)
            if name == "v4_unlevered":
                unlevered_frames.append(daily)

    unlevered = pd.concat(unlevered_frames, ignore_index=True)
    test_dates = pd.Index(sorted(unlevered["date"].unique()))

    static = pd.Series(test_dates.year.map(static_by_year), index=test_dates, dtype=float)
    history = pd.concat([seed_returns, portfolio_returns(unlevered)])
    history = history[~history.index.duplicated(keep="last")].sort_index()
    dynamic = dynamic_leverage(
        history, args.target_vol, args.max_leverage, args.vol_halflife
    ).reindex(test_dates)

    frames.append(lever(unlevered, static, "v4_static_vol", cost))
    frames.append(lever(unlevered, dynamic, "v4_dynamic_vol", cost))
    daily = pd.concat(frames, ignore_index=True)

    comparison = portfolio_metrics(daily)
    leverage_used = daily.groupby("model_name")["leverage"].mean() if "leverage" in daily else None
    if leverage_used is not None:
        comparison["average_leverage"] = comparison["model_name"].map(leverage_used).fillna(1.0)
    robustness = robustness_table(daily, baseline="naive_zscore")
    net_comparison = portfolio_metrics(net_of_costs(daily, args.net_cost_bps, args.borrow_bps))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{cost:g}bps"
    daily.to_csv(args.output_dir / f"daily_results{suffix}.csv", index=False)
    summarize_backtest(daily).to_csv(args.output_dir / f"summary{suffix}.csv", index=False)
    comparison.to_csv(args.output_dir / f"comparison{suffix}.csv", index=False)
    robustness.to_csv(args.output_dir / f"robustness{suffix}.csv", index=False)
    net_comparison.to_csv(
        args.output_dir / f"comparison_net_{args.net_cost_bps:g}bps_borrow_{args.borrow_bps:g}.csv",
        index=False,
    )
    pd.DataFrame(chosen).to_csv(args.output_dir / f"selected_parameters{suffix}.csv", index=False)
    pd.DataFrame({"static_leverage": static, "dynamic_leverage": dynamic}).to_csv(
        args.output_dir / f"leverage{suffix}.csv", index_label="date"
    )
    with pd.option_context("display.width", 240, "display.precision", 3):
        print(f"\nGross ({cost:g} bps), target vol {args.target_vol:.0%}:")
        print(comparison.to_string(index=False))
        print()
        print(robustness.to_string(index=False))
        print(f"\nNet of {args.net_cost_bps:g} bps + {args.borrow_bps:g} bps/yr borrow (same positions):")
        print(net_comparison.to_string(index=False))


if __name__ == "__main__":
    main()
