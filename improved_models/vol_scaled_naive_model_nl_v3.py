"""Nonlinear V3 on point-in-time screened pairs.

This variant applies signed-squared z-score sizing to the EWMA spread z-score
(half-life 120), while retaining the no-trade band and VIX size overlay:

* ``band = 0.2`` fixed (no band VIX scaling) - it mainly cuts turnover;
* ``vix_scale`` chosen each year from {0, 0.25} by an inner walk-forward on the
  four years before the test year.

By default pairs come from ``build_point_in_time_dataset.py`` (``pit_*`` files):
each test year's pairs are screened on the ten years before it. Shorter screens
were tested: 2 years (``pit2y_*``) roughly halved Sharpe for every model, and
5 years (``pit5y_*``) was about level with 10 years; pass ``--input`` /
``--selection`` to use them.

Results are reported gross of fees (the default ``--transaction-cost-bps 0``,
also used for the parameter search). A secondary table re-costs the same
positions at ``--net-cost-bps`` for reference.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from backtest import apply_positions_to_returns, run_walk_forward_backtest, summarize_backtest
from base_model import NaivePairsTradingModel
from improved_models.regression_model import portfolio_metrics
from improved_models.vol_scaled_naive_model_nl import (
    VolScaledNaivePairsTradingModel,
    robustness_table,
    select_parameters,
)


V3_GRID = {
    "vix_scale": (0.0, 0.25),
    "band": (0.2,),
    "band_vix_scale": (0.0,),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--input", type=Path, default=Path("data") / "pit_pair_prices.csv")
    parser.add_argument("--selection", type=Path, default=Path("data") / "pit_pair_selection.csv")
    parser.add_argument("--start-year", type=int, default=2014)
    parser.add_argument("--end-year", type=int, default=2023)
    parser.add_argument("--train-years", type=int, default=4)
    parser.add_argument(
        "--transaction-cost-bps", type=float, default=0.0,
        help="Cost used for the parameter search and the main (gross) results.",
    )
    parser.add_argument(
        "--net-cost-bps", type=float, default=5.0,
        help="Cost for the secondary net-of-fees table (same positions).",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("backtest_results") / "vol_scaled_naive_v3")
    args = parser.parse_args()

    data = pd.read_csv(args.input, parse_dates=["date"])
    selection = pd.read_csv(args.selection)
    cost = args.transaction_cost_bps

    frames, chosen = [], []
    for fold_id, year in enumerate(range(args.start_year, args.end_year + 1), start=1):
        pairs = selection.loc[selection["test_year"] == year, "pair_id"].unique()
        year_data = data.loc[data["pair_id"].isin(pairs)]
        params, _ = select_parameters(year_data, year, cost, args.train_years, grid=V3_GRID)
        chosen.append({"test_year": year, **params, "pairs": ",".join(pairs)})
        print(f"{year}: {params}  pairs: {', '.join(pairs)}", flush=True)
        models = {
            "naive_zscore": NaivePairsTradingModel,
            "naive_ewma_z": VolScaledNaivePairsTradingModel,
            "vol_scaled_naive_v3": lambda p=params: VolScaledNaivePairsTradingModel(**p),
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

    daily = pd.concat(frames, ignore_index=True)
    comparison = portfolio_metrics(daily)
    robustness = robustness_table(daily, baseline="naive_zscore")
    net_daily = apply_positions_to_returns(daily, transaction_cost_bps=args.net_cost_bps)
    net_comparison = portfolio_metrics(net_daily)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{cost:g}bps"
    daily.to_csv(args.output_dir / f"daily_results{suffix}.csv", index=False)
    summarize_backtest(daily).to_csv(args.output_dir / f"summary{suffix}.csv", index=False)
    comparison.to_csv(args.output_dir / f"comparison{suffix}.csv", index=False)
    robustness.to_csv(args.output_dir / f"robustness{suffix}.csv", index=False)
    net_comparison.to_csv(
        args.output_dir / f"comparison_net_{args.net_cost_bps:g}bps.csv", index=False
    )
    pd.DataFrame(chosen).to_csv(args.output_dir / f"selected_parameters{suffix}.csv", index=False)
    with pd.option_context("display.width", 220, "display.precision", 3):
        print(f"\nGross ({cost:g} bps):")
        print(comparison.to_string(index=False))
        print()
        print(robustness.to_string(index=False))
        print(f"\nNet of {args.net_cost_bps:g} bps (same positions):")
        print(net_comparison.to_string(index=False))


if __name__ == "__main__":
    main()
