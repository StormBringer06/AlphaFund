"""Trade-level sizing from the market state at entry (spread vol and VIX).

Diagnostics on point-in-time pairs showed that spread deviations which build
slowly while both the spread's own volatility and VIX are normal ("calm
drift") did not revert, whereas deviations that arrive with a spread-volatility
spike, or while VIX is high, did. Daily rescaling by VIX had failed because it
cut trades short, so this model decides size once per trade.

Per pair, each trading day (close-of-day targets):

1. Spread, hedge ratio and EWMA z-score exactly as
   ``VolScaledNaivePairsTradingModel``.
2. ``vol_ratio`` = std of daily spread changes over ``short_vol_window`` days
   divided by the same over ``long_vol_window`` days; ``vix_pct`` = VIX
   percentile in its trailing ``vix_window`` days.
3. **Trades.** A trade opens when ``|z| > entry_zscore`` and closes when z
   crosses zero. At entry it is *favourable* if ``vix_pct > vix_threshold`` or
   ``vol_ratio > vol_ratio_threshold``, otherwise *calm drift*.
4. **Size.** While a trade is open the target is the naive
   ``clip(-z / full_exposure_zscore, -1, 1)``, multiplied by 1 if favourable or
   ``calm_size`` if calm drift. A calm-drift trade is upgraded to full size if
   the state becomes favourable while it is open (``allow_upgrade``), but a
   trade is never downgraded. Flat when no trade is open.
5. **Execution.** The held position only moves when the target differs from it
   by more than ``band`` (gross exposure units), except that a closed trade is
   always flattened.

Thresholds are fixed, not fitted. ``calm_size=1`` keeps the trade logic but
removes the state filter, isolating its effect.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

from backtest import run_walk_forward_backtest, summarize_backtest
from base_model import NaivePairsTradingModel
from improved_models.regression_model import portfolio_metrics
from improved_models.vol_scaled_naive_model import (
    VolScaledNaivePairsTradingModel,
    _Hedge,
    robustness_table,
)


class EntryStatePairsTradingModel(VolScaledNaivePairsTradingModel):
    """EWMA z-score trades sized by spread-vol / VIX state at entry."""

    def __init__(
        self,
        entry_zscore: float = 1.0,
        vix_threshold: float = 0.75,
        vol_ratio_threshold: float = 1.3,
        calm_size: float = 0.5,
        band: float = 0.2,
        short_vol_window: int = 20,
        long_vol_window: int = 120,
        allow_upgrade: bool = True,
        **kwargs: object,
    ) -> None:
        super().__init__(band=band, **kwargs)
        if not math.isfinite(entry_zscore) or entry_zscore < 0:
            raise ValueError("entry_zscore must be finite and non-negative.")
        if not 0 < vix_threshold < 1:
            raise ValueError("vix_threshold must be between 0 and 1.")
        if not math.isfinite(vol_ratio_threshold) or vol_ratio_threshold <= 0:
            raise ValueError("vol_ratio_threshold must be finite and positive.")
        if not 0 <= calm_size <= 1:
            raise ValueError("calm_size must be between 0 and 1.")
        if not 2 <= short_vol_window < long_vol_window:
            raise ValueError("Need 2 <= short_vol_window < long_vol_window.")
        self.entry_zscore = float(entry_zscore)
        self.vix_threshold = float(vix_threshold)
        self.vol_ratio_threshold = float(vol_ratio_threshold)
        self.calm_size = float(calm_size)
        self.short_vol_window = short_vol_window
        self.long_vol_window = long_vol_window
        self.allow_upgrade = allow_upgrade
        self._trade_side = 0
        self._trade_full = False

    def fit(self, train_data: pd.DataFrame) -> EntryStatePairsTradingModel:
        super().fit(train_data)
        self._trade_side = 0
        self._trade_full = False
        return self

    def _features(self, frame: pd.DataFrame, hedge: _Hedge) -> pd.DataFrame:
        out = super()._features(frame, hedge)
        change = out["spread"].diff()
        out["spread_vol_short"] = change.rolling(self.short_vol_window).std()
        out["spread_vol_long"] = change.rolling(
            self.long_vol_window, min_periods=self.long_vol_window // 2
        ).std()
        out["vol_ratio"] = out["spread_vol_short"] / out["spread_vol_long"]
        return out

    def _positions(self, segment: pd.DataFrame, hedge: _Hedge) -> pd.DataFrame:
        z = segment["spread_zscore"].to_numpy(dtype=float)
        vix_pct = segment["vix_pct"].to_numpy(dtype=float)
        vol_ratio = segment["vol_ratio"].to_numpy(dtype=float)
        favourable = (np.nan_to_num(vix_pct, nan=0.0) > self.vix_threshold) | (
            np.nan_to_num(vol_ratio, nan=0.0) > self.vol_ratio_threshold
        )

        n = len(z)
        target = np.zeros(n)
        position = np.zeros(n)
        side_out = np.zeros(n, dtype=int)
        full_out = np.zeros(n, dtype=bool)
        entries = np.zeros(n, dtype=bool)
        exits = np.zeros(n, dtype=bool)
        traded = np.zeros(n, dtype=bool)
        side, full, held = self._trade_side, self._trade_full, self._position

        for i in range(n):
            if not math.isfinite(z[i]):
                if side:
                    exits[i] = True
                side, full = 0, False
            else:
                # side = direction of the trade: +1 long spread (z < 0).
                if side and side * z[i] >= 0:
                    side, full = 0, False
                    exits[i] = True
                if side == 0 and abs(z[i]) > self.entry_zscore:
                    side = -1 if z[i] > 0 else 1
                    full = bool(favourable[i])
                    entries[i] = True
                elif self.allow_upgrade and side and not full and favourable[i]:
                    full = True  # upgrade only, never downgrade
            if side:
                size = 1.0 if full else self.calm_size
                base = min(1.0, abs(z[i]) / self.full_exposure_zscore)
                target[i] = side * min(base * size, self.max_gross_exposure)
            if side == 0 or abs(target[i] - held) > self.band:
                traded[i] = held != target[i]
                held = target[i]
            position[i] = held
            side_out[i], full_out[i] = side, full

        self._trade_side, self._trade_full, self._position = side, full, held
        scale = 1.0 + abs(hedge.hedge_ratio)
        segment["favourable_state"] = favourable
        segment["trade_side"] = side_out
        segment["trade_full_size"] = full_out
        segment["trade_entry"] = entries
        segment["trade_exit"] = exits
        segment["target_signal"] = target
        segment["rebalanced"] = traded
        segment["signal"] = position
        segment["target_weight_a"] = position / scale
        segment["target_weight_b"] = -hedge.hedge_ratio * position / scale
        segment["target_gross_exposure"] = np.abs(position)
        return segment


# ---------------------------------------------------------------- comparison


def model_variants() -> dict[str, object]:
    return {
        "naive_zscore": NaivePairsTradingModel,
        "ewma_z_band": lambda: VolScaledNaivePairsTradingModel(band=0.2),
        "entry_state_calm_1": lambda: EntryStatePairsTradingModel(calm_size=1.0),
        "entry_state_calm_0.5": lambda: EntryStatePairsTradingModel(calm_size=0.5),
        "entry_state_calm_0": lambda: EntryStatePairsTradingModel(calm_size=0.0),
        "entry_state_calm_0.5_no_upgrade": lambda: EntryStatePairsTradingModel(
            calm_size=0.5, allow_upgrade=False
        ),
        "entry_state_calm_0_no_upgrade": lambda: EntryStatePairsTradingModel(
            calm_size=0.0, allow_upgrade=False
        ),
    }


def run_by_year(
    data: pd.DataFrame,
    selection: pd.DataFrame | None,
    models: dict[str, object],
    start_year: int,
    end_year: int,
    train_years: int,
    transaction_cost_bps: float,
) -> pd.DataFrame:
    """Walk-forward one test year at a time, trading that year's selected pairs."""
    frames = []
    for fold_id, year in enumerate(range(start_year, end_year + 1), start=1):
        pairs = (
            data["pair_id"].unique() if selection is None
            else selection.loc[selection["test_year"] == year, "pair_id"].unique()
        )
        year_data = data.loc[data["pair_id"].isin(pairs)]
        for name, factory in models.items():
            daily = run_walk_forward_backtest(
                year_data,
                model_factory=factory,
                start_year=year,
                end_year=year,
                train_years=train_years,
                transaction_cost_bps=transaction_cost_bps,
                model_name=name,
            )
            daily["fold_id"] = fold_id
            frames.append(daily)
    return pd.concat(frames, ignore_index=True)


def trade_statistics(daily: pd.DataFrame) -> pd.DataFrame:
    """Per-model count of trades and share opened in a favourable state."""
    trades = daily.loc[daily.get("trade_entry", False) == True]  # noqa: E712
    if trades.empty:
        return pd.DataFrame()
    return trades.groupby("model_name").agg(
        trades=("trade_entry", "size"),
        favourable_at_entry=("trade_full_size", "mean"),
    )


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
        "--output-dir", type=Path, default=Path("backtest_results") / "entry_state"
    )
    args = parser.parse_args()

    data = pd.read_csv(args.input, parse_dates=["date"])
    selection = (
        None if str(args.selection).lower() == "none" else pd.read_csv(args.selection)
    )
    daily = run_by_year(
        data, selection, model_variants(), args.start_year, args.end_year,
        args.train_years, args.transaction_cost_bps,
    )
    summary = summarize_backtest(daily)
    comparison = portfolio_metrics(daily)
    robustness = robustness_table(daily)
    trades = trade_statistics(daily)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{args.transaction_cost_bps:g}bps"
    daily.to_csv(args.output_dir / f"daily_results{suffix}.csv", index=False)
    summary.to_csv(args.output_dir / f"summary{suffix}.csv", index=False)
    comparison.to_csv(args.output_dir / f"comparison{suffix}.csv", index=False)
    robustness.to_csv(args.output_dir / f"robustness{suffix}.csv", index=False)
    trades.to_csv(args.output_dir / f"trades{suffix}.csv")
    with pd.option_context("display.width", 220, "display.precision", 3):
        print(comparison.to_string(index=False))
        print()
        print(robustness.to_string(index=False))
        print()
        print(trades.to_string())


if __name__ == "__main__":
    main()
