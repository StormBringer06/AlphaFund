"""Naive pairs sizing with an EWMA z-score, VIX size overlay and VIX no-trade band.

Per pair, each trading day (close-of-day targets):

1. **Spread / z-score.** OLS hedge ratio of log A on log B over the trailing
   ``lookback_years``, refit every ``refit_interval_months`` from data strictly
   before the refit date. ``z_t = (spread_t - m_{t-1}) / s_{t-1}`` with ``m`` /
   ``s`` the EWMA mean / std of the spread (half-life ``zscore_halflife``).
2. **Nonlinear base signal.** The signed squared z-score is ``sign(z) * z**2``;
   sizing is ``-tanh(signed_squared_z / full_exposure_zscore)`` so the signal
   remains mean-reverting on both sides of the spread.
3. **VIX size overlay.** ``vix_pct`` is today's VIX percentile within the
   trailing ``vix_window`` days. The signal is multiplied by
   ``clip(1 + 2 * vix_scale * (vix_pct - 0.5), 0.5, 1.5)`` - larger in
   high-VIX periods, where mean reversion was historically stronger - then
   capped at ``max_gross_exposure``. ``vix_scale = 0`` turns it off.
4. **VIX-dependent no-trade band.** The held position only moves to the new
   target when ``|target - position| > band_t`` with
   ``band_t = band * clip(1 + 2 * band_vix_scale * (0.5 - vix_pct), 0, 2)``,
   i.e. a wider band (fewer trades) in calm markets, where the edge is
   weakest, and a narrower one in stressed markets. ``band = 0`` trades daily.

The position is in gross-exposure units and split across legs with the current
hedge ratio. ``select_parameters`` fits ``vix_scale``, ``band`` and
``band_vix_scale`` on a training window, pooled across pairs.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import itertools
import math
from pathlib import Path

import numpy as np
import pandas as pd

from backtest import run_walk_forward_backtest, summarize_backtest
from base_model import NaivePairsTradingModel
from improved_models.regression_model import RegressionPairsTradingModel, portfolio_metrics


_INPUT_COLUMNS = ("date", "adjusted_close_a", "adjusted_close_b", "vix")


@dataclass(frozen=True)
class _Hedge:
    intercept: float
    hedge_ratio: float


class VolScaledNaivePairsTradingModel:
    """Signed nonlinear z-score sizing with VIX overlay and no-trade band."""

    def __init__(
        self,
        vix_scale: float = 0.0,
        band: float = 0.0,
        band_vix_scale: float = 0.0,
        zscore_halflife: float = 120.0,
        full_exposure_zscore: float = 4.0,
        max_gross_exposure: float = 1.0,
        vix_window: int = 504,
        lookback_years: int = 2,
        refit_interval_months: int = 6,
        min_train_observations: int = 120,
    ) -> None:
        for name, value in (("vix_scale", vix_scale), ("band", band),
                            ("band_vix_scale", band_vix_scale)):
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative.")
        if vix_scale > 0.5 or band_vix_scale > 0.5:
            raise ValueError("vix_scale and band_vix_scale must be at most 0.5.")
        if zscore_halflife <= 0 or full_exposure_zscore <= 0 or max_gross_exposure <= 0:
            raise ValueError(
                "zscore_halflife, full_exposure_zscore and max_gross_exposure "
                "must be positive."
            )
        if vix_window < 20:
            raise ValueError("vix_window must be at least 20.")
        if lookback_years < 1 or refit_interval_months < 1:
            raise ValueError("lookback_years and refit_interval_months must be >= 1.")

        self.vix_scale = float(vix_scale)
        self.band = float(band)
        self.band_vix_scale = float(band_vix_scale)
        self.zscore_halflife = float(zscore_halflife)
        self.full_exposure_zscore = float(full_exposure_zscore)
        self.max_gross_exposure = float(max_gross_exposure)
        self.vix_window = vix_window
        self.lookback_years = lookback_years
        self.refit_interval_months = refit_interval_months
        self.min_train_observations = min_train_observations

        self._hedge: _Hedge | None = None
        self._history: pd.DataFrame | None = None
        self._next_refit_date: pd.Timestamp | None = None
        self._last_refit_date: pd.Timestamp | None = None
        self._position = 0.0

    # ------------------------------------------------------------------ API

    def fit(self, train_data: pd.DataFrame) -> VolScaledNaivePairsTradingModel:
        train = RegressionPairsTradingModel._prepare(train_data, "training")
        last_date = train["date"].iloc[-1]
        self._hedge = self._estimate(train, last_date + pd.Timedelta(days=1))
        self._history = train[list(_INPUT_COLUMNS)].reset_index(drop=True)
        self._last_refit_date = last_date
        self._next_refit_date = last_date + pd.DateOffset(
            months=self.refit_interval_months
        )
        self._position = 0.0
        return self

    def predict(self, test_data: pd.DataFrame) -> pd.DataFrame:
        if self._hedge is None or self._history is None:
            raise RuntimeError("Call fit with training data before predict.")
        test = RegressionPairsTradingModel._prepare(test_data, "test")
        if test["date"].iloc[0] <= self._history["date"].iloc[-1]:
            raise ValueError("Test dates must be strictly after previously seen dates.")

        combined = pd.concat(
            [self._history, test[list(_INPUT_COLUMNS)]], ignore_index=True
        )
        hedge = self._hedge
        last_refit = self._last_refit_date
        frames: list[pd.DataFrame] = []
        for start, end, refit in self._segments(test["date"]):
            if refit:
                hedge = self._estimate(combined.loc[combined["date"] < start], start)
                last_refit = start
            features = self._features(combined.loc[combined["date"] <= end], hedge)
            segment = features.loc[features["date"] >= start].copy()
            segment["beta_refit"] = (segment["date"] == start) & refit
            segment["beta_refit_date"] = last_refit
            frames.append(self._positions(segment, hedge))
        signals = pd.concat(frames, ignore_index=True)

        self._hedge = hedge
        self._last_refit_date = last_refit
        self._history = self._trim(combined, test["date"].iloc[-1])
        return test.merge(
            signals.drop(columns=list(_INPUT_COLUMNS[1:])),
            on="date",
            how="left",
            validate="one_to_one",
        )

    # ------------------------------------------------------------ internals

    def _estimate(self, known: pd.DataFrame, as_of: pd.Timestamp) -> _Hedge:
        window = known.loc[
            known["date"] >= as_of - pd.DateOffset(years=self.lookback_years)
        ]
        if len(window) < self.min_train_observations:
            raise ValueError(
                f"Estimation window has {len(window)} rows; "
                f"need {self.min_train_observations}."
            )
        log_a = np.log(window["adjusted_close_a"])
        log_b = np.log(window["adjusted_close_b"])
        hedge_ratio = float(log_b.cov(log_a) / log_b.var())
        if not math.isfinite(hedge_ratio) or abs(hedge_ratio) < 1e-12:
            raise ValueError("Estimation window produced an invalid hedge ratio.")
        return _Hedge(float(log_a.mean() - hedge_ratio * log_b.mean()), hedge_ratio)

    def _features(self, frame: pd.DataFrame, hedge: _Hedge) -> pd.DataFrame:
        out = frame.reset_index(drop=True).copy()
        spread = np.log(out["adjusted_close_a"]) - (
            hedge.intercept + hedge.hedge_ratio * np.log(out["adjusted_close_b"])
        )
        ewm = spread.ewm(
            halflife=self.zscore_halflife, min_periods=int(self.zscore_halflife)
        )
        mean, std = ewm.mean().shift(1), ewm.std().shift(1)
        out["hedge_ratio"] = hedge.hedge_ratio
        out["spread"] = spread
        out["spread_zscore"] = (spread - mean) / std.where(std > 0)
        out["spread_zscore1"] = (
            np.sign(out["spread_zscore"]) * out["spread_zscore"].pow(2)
        )
        out["vix_pct"] = (
            out["vix"]
            .rolling(self.vix_window, min_periods=self.vix_window // 4)
            .rank(pct=True)
        )
        return out

    def _positions(self, segment: pd.DataFrame, hedge: _Hedge) -> pd.DataFrame:
        z = segment["spread_zscore1"].to_numpy(dtype=float)
        vix_pct = segment["vix_pct"].fillna(0.5).to_numpy(dtype=float)
        base = -np.tanh(z / self.full_exposure_zscore)
        multiplier = np.clip(1.0 + 2.0 * self.vix_scale * (vix_pct - 0.5), 0.5, 1.5)
        target = np.clip(
            base * multiplier, -self.max_gross_exposure, self.max_gross_exposure
        )
        target = np.nan_to_num(target, nan=0.0)  # flat during EWMA warm-up
        band = self.band * np.clip(
            1.0 + 2.0 * self.band_vix_scale * (0.5 - vix_pct), 0.0, 2.0
        )

        position = np.empty(len(target))
        traded = np.zeros(len(target), dtype=bool)
        current = self._position
        for index, (wanted, width) in enumerate(zip(target, band)):
            if abs(wanted - current) > width:
                current = wanted
                traded[index] = True
            position[index] = current
        self._position = current

        scale = 1.0 + abs(hedge.hedge_ratio)
        segment["vix_multiplier"] = multiplier
        segment["band_width"] = band
        segment["target_signal"] = target
        segment["rebalanced"] = traded
        segment["signal"] = position
        segment["target_weight_a"] = position / scale
        segment["target_weight_b"] = -hedge.hedge_ratio * position / scale
        segment["target_gross_exposure"] = np.abs(position)
        return segment

    def _segments(
        self, dates: pd.Series
    ) -> list[tuple[pd.Timestamp, pd.Timestamp, bool]]:
        segments: list[list[object]] = []
        for date in dates:
            if date >= self._next_refit_date:
                segments.append([date, date, True])
                while self._next_refit_date <= date:
                    self._next_refit_date += pd.DateOffset(
                        months=self.refit_interval_months
                    )
            elif not segments:
                segments.append([date, date, False])
            else:
                segments[-1][1] = date
        return [(s, e, r) for s, e, r in segments]

    def _trim(self, data: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
        warmup_days = max(
            math.ceil(5 * self.zscore_halflife * 7 / 5),
            math.ceil(self.vix_window * 7 / 5),
        )
        cutoff = (
            as_of
            - pd.DateOffset(years=self.lookback_years)
            - pd.Timedelta(days=warmup_days)
        )
        return data.loc[data["date"] >= cutoff].reset_index(drop=True)


# ------------------------------------------------------ parameter selection

DEFAULT_GRID = {
    "vix_scale": (0.0, 0.25, 0.5),
    "band": (0.0, 0.1, 0.2, 0.3, 0.4),
    "band_vix_scale": (0.0, 0.5),
}


def _portfolio_sharpe(daily: pd.DataFrame) -> float:
    returns = daily.groupby("date")["strategy_return"].mean()
    vol = float(returns.std())
    return float(returns.mean() / vol * math.sqrt(252.0)) if vol > 0 else -math.inf


def select_parameters(
    data: pd.DataFrame,
    test_year: int,
    transaction_cost_bps: float,
    train_years: int = 4,
    inner_test_years: int = 2,
    grid: dict[str, tuple[float, ...]] = DEFAULT_GRID,
) -> tuple[dict[str, float], pd.DataFrame]:
    """Pick overlay/band parameters on the training window before ``test_year``.

    Runs an inner walk-forward over the last ``inner_test_years`` of the
    ``train_years`` before ``test_year`` (each inner year trained on the years
    before it) and maximizes the net Sharpe of the equal-weight portfolio of all
    pairs in ``data``. Only data before ``test_year`` is used.
    """
    inner_train_years = train_years - inner_test_years
    data = data.loc[
        (data["date"] >= pd.Timestamp(year=test_year - train_years, month=1, day=1))
        & (data["date"] < pd.Timestamp(year=test_year, month=1, day=1))
    ]
    rows = []
    for values in itertools.product(*grid.values()):
        params = dict(zip(grid, values))
        if params["band"] == 0 and params["band_vix_scale"] > 0:
            continue  # same as band_vix_scale = 0
        daily = run_walk_forward_backtest(
            data,
            model_factory=lambda p=params: VolScaledNaivePairsTradingModel(**p),
            start_year=test_year - inner_test_years,
            end_year=test_year - 1,
            train_years=inner_train_years,
            transaction_cost_bps=transaction_cost_bps,
        )
        rows.append({**params, "inner_sharpe": _portfolio_sharpe(daily)})
    scores = pd.DataFrame(rows)
    best = scores.loc[scores["inner_sharpe"].idxmax(), list(grid)].to_dict()
    return {k: float(v) for k, v in best.items()}, scores


# ---------------------------------------------------------------- evaluation


def _year_bootstrap(
    returns: pd.DataFrame, models: list[str], baseline: str, draws: int = 2000
) -> pd.DataFrame:
    """Resample whole years; Sharpe CI per model and of its gap to baseline."""
    rng = np.random.default_rng(0)
    years = returns.index.year.unique().to_numpy()
    by_year = {y: returns.loc[returns.index.year == y] for y in years}
    samples = {m: [] for m in models}
    for _ in range(draws):
        sample = pd.concat([by_year[y] for y in rng.choice(years, len(years))])
        sharpe = sample.mean() / sample.std() * math.sqrt(252.0)
        for m in models:
            samples[m].append(sharpe[m])
    rows = []
    for m in models:
        s = np.array(samples[m])
        gap = s - np.array(samples[baseline])
        rows.append({
            "model_name": m,
            "sharpe_p05": np.quantile(s, 0.05),
            "sharpe_p95": np.quantile(s, 0.95),
            "gap_vs_baseline_p05": np.quantile(gap, 0.05),
            "gap_vs_baseline_p95": np.quantile(gap, 0.95),
            "prob_beats_baseline": float((gap > 0).mean()),
        })
    return pd.DataFrame(rows)


def robustness_table(daily: pd.DataFrame, baseline: str = "naive_zscore") -> pd.DataFrame:
    returns = daily.pivot_table(
        index="date", columns="model_name", values="strategy_return", aggfunc="mean"
    )
    models = list(returns.columns)
    ann = math.sqrt(252.0)
    sharpe = lambda r: r.mean() / r.std() * ann
    calm = returns.loc[~returns.index.year.isin([2020, 2022])]
    target_vol = returns[baseline].std()
    table = pd.DataFrame({
        "model_name": models,
        "sharpe": [sharpe(returns[m]) for m in models],
        "sharpe_ex_2020_2022": [sharpe(calm[m]) for m in models],
        "ann_return_at_baseline_vol": [
            returns[m].mean() * target_vol / returns[m].std() * 252 for m in models
        ],
    })
    return table.merge(_year_bootstrap(returns, models, baseline), on="model_name")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--input", type=Path, default=Path("data") / "pit_pair_prices.csv")
    parser.add_argument(
        "--selection",
        type=Path,
        default=Path("data") / "pit_pair_selection.csv",
        help="Per-year pair list; pass 'none' to trade every pair in --input each year.",
    )
    parser.add_argument("--start-year", type=int, default=2014)
    parser.add_argument("--end-year", type=int, default=2023)
    parser.add_argument("--train-years", type=int, default=4)
    parser.add_argument("--transaction-cost-bps", type=float, default=5.0)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("backtest_results") / "vol_scaled_naive"
    )
    args = parser.parse_args()

    data = pd.read_csv(args.input, parse_dates=["date"])
    selection = (
        None if str(args.selection).lower() == "none"
        else pd.read_csv(args.selection)
    )
    cost = args.transaction_cost_bps

    daily_frames, chosen = [], []
    for fold_id, year in enumerate(range(args.start_year, args.end_year + 1), start=1):
        pairs = (
            data["pair_id"].unique() if selection is None
            else selection.loc[selection["test_year"] == year, "pair_id"].unique()
        )
        year_data = data.loc[data["pair_id"].isin(pairs)]
        params, _ = select_parameters(year_data, year, cost, args.train_years)
        chosen.append({"test_year": year, **params, "pairs": ",".join(pairs)})
        print(f"{year}: {params}", flush=True)
        models = {
            "naive_zscore": NaivePairsTradingModel,
            "naive_ewma_z": VolScaledNaivePairsTradingModel,
            "vol_scaled_naive": lambda p=params: VolScaledNaivePairsTradingModel(**p),
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
            daily_frames.append(daily)

    daily = pd.concat(daily_frames, ignore_index=True)
    summary = summarize_backtest(daily)
    comparison = portfolio_metrics(daily)
    robustness = robustness_table(daily)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{cost:g}bps"
    daily.to_csv(args.output_dir / f"daily_results{suffix}.csv", index=False)
    summary.to_csv(args.output_dir / f"summary{suffix}.csv", index=False)
    comparison.to_csv(args.output_dir / f"comparison{suffix}.csv", index=False)
    robustness.to_csv(args.output_dir / f"robustness{suffix}.csv", index=False)
    pd.DataFrame(chosen).to_csv(
        args.output_dir / f"selected_parameters{suffix}.csv", index=False
    )
    with pd.option_context("display.width", 220, "display.precision", 3):
        print(comparison.to_string(index=False))
        print()
        print(robustness.to_string(index=False))


if __name__ == "__main__":
    main()
