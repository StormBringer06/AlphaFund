"""Ablation of the V2 features on point-in-time pairs, with fixed parameters.

Starts from V2 with every feature off (equivalent to ``naive_ewma_z``), then:

* ``+feature``: switches on one feature at a time;
* ``full-feature``: switches off one feature at a time from the full V2.

No parameter search, so every variant is evaluated with the same fixed values
(those of ``V2_DEFAULT_GRID``, with the VIX overlay and band at 0.25 / 0.2).
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd

from backtest import run_walk_forward_backtest
from base_model import NaivePairsTradingModel
from improved_models.regression_model import portfolio_metrics
from improved_models.vol_scaled_naive_model import (
    VolScaledNaivePairsTradingModel,
    robustness_table,
)
from improved_models.vol_scaled_naive_model_v2 import VolScaledNaivePairsTradingModelV2


ALL_OFF = {
    "zscore_halflife": 120.0,
    "vix_scale": 0.0,
    "band": 0.0,
    "band_vix_scale": 0.0,
    "target_pair_volatility": None,
    "max_spread_half_life": math.inf,
    "min_hedge_stability": 0.0,
}
FEATURES = {
    "halflife_60": {"zscore_halflife": 60.0},
    "gate": {"max_spread_half_life": 60.0, "min_hedge_stability": 0.75},
    "vol_target": {"target_pair_volatility": 0.12},
    "band": {"band": 0.2},
    "vix_overlay": {"vix_scale": 0.25},
}


def variants() -> dict[str, dict[str, object]]:
    full = {**ALL_OFF}
    for changes in FEATURES.values():
        full.update(changes)
    out = {"v2_all_off": dict(ALL_OFF)}
    out.update({f"+{name}": {**ALL_OFF, **changes} for name, changes in FEATURES.items()})
    out["v2_full_fixed"] = full
    for name, changes in FEATURES.items():
        out[f"full-{name}"] = {**full, **{key: ALL_OFF[key] for key in changes}}
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--input", type=Path, default=Path("data") / "pit_pair_prices.csv")
    parser.add_argument("--selection", type=Path, default=Path("data") / "pit_pair_selection.csv")
    parser.add_argument("--start-year", type=int, default=2014)
    parser.add_argument("--end-year", type=int, default=2023)
    parser.add_argument("--train-years", type=int, default=4)
    parser.add_argument("--transaction-cost-bps", type=float, default=5.0)
    parser.add_argument("--output-dir", type=Path, default=Path("backtest_results") / "v2_ablation")
    args = parser.parse_args()

    data = pd.read_csv(args.input, parse_dates=["date"])
    selection = pd.read_csv(args.selection)
    models = {
        "naive_zscore": NaivePairsTradingModel,
        "naive_ewma_z": VolScaledNaivePairsTradingModel,
        **{
            name: (lambda p=params: VolScaledNaivePairsTradingModelV2(**p))
            for name, params in variants().items()
        },
    }

    frames = []
    for fold_id, year in enumerate(range(args.start_year, args.end_year + 1), start=1):
        pairs = selection.loc[selection["test_year"] == year, "pair_id"].unique()
        year_data = data.loc[data["pair_id"].isin(pairs)]
        print(f"{year}: {', '.join(pairs)}", flush=True)
        for name, factory in models.items():
            daily = run_walk_forward_backtest(
                year_data,
                model_factory=factory,
                start_year=year,
                end_year=year,
                train_years=args.train_years,
                transaction_cost_bps=args.transaction_cost_bps,
                model_name=name,
            )
            daily["fold_id"] = fold_id
            frames.append(daily)

    daily = pd.concat(frames, ignore_index=True)
    comparison = portfolio_metrics(daily)
    robustness = robustness_table(daily, baseline="v2_all_off")
    table = comparison.merge(
        robustness[["model_name", "sharpe_ex_2020_2022", "gap_vs_baseline_p05",
                    "gap_vs_baseline_p95", "prob_beats_baseline"]],
        on="model_name",
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{args.transaction_cost_bps:g}bps"
    table.to_csv(args.output_dir / f"ablation{suffix}.csv", index=False)
    daily.to_csv(args.output_dir / f"daily_results{suffix}.csv", index=False)
    pd.DataFrame(
        [{"model_name": name, **params} for name, params in variants().items()]
    ).to_csv(args.output_dir / f"variants{suffix}.csv", index=False)
    with pd.option_context("display.width", 220, "display.precision", 3):
        print(table.to_string(index=False))


if __name__ == "__main__":
    main()
