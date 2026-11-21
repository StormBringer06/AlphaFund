"""AlphaFund V5: run the pairs-trading model and write results to ``results/``.

V5 = BD's V3 signal + Ciaran's exit classifier + dynamic leverage:

1. **Pairs** - each test year trades the top 5 same-sector pairs from a screen
   run on the ten years before it (``data/pit_pair_*.csv``).
2. **Signal (V3)** - EWMA z-score of the spread, sized ``clip(-z/2, -1, 1)``
   with a VIX overlay and a 0.2 no-trade band; ``vix_scale`` is re-chosen each
   year on prior data (``pairs_model/signal.py``).
3. **Exit classifier** - at days 5-120 of each open position a ridge regression
   predicts the remaining P&L; if negative, the position is closed until the
   spread crosses zero. One model per year, trained only on earlier data and
   loaded from ``models/exit_ridge_weights.json`` (``pairs_model/exit_classifier.py``).
4. **Leverage** - the book is scaled to an 8% volatility target (max 4x) from
   its own trailing volatility (``pairs_model/leverage.py``).

Results are reported gross and net of trading costs plus short-borrow cost,
next to plain V3 for reference.

    python main.py                          # 2014-2023 with saved exit models
    python main.py --train-exit-model       # refit the yearly exit models first
    python main.py --start-year 2024 --end-year 2025 --freeze-exit-model 2023  # holdout
"""

from __future__ import annotations

import argparse
from pathlib import Path
import warnings

import pandas as pd

from backtest import apply_positions_to_returns
from pairs_model.exit_classifier import (
    EXIT_RULE, RIDGE_WEIGHTS_PATH, apply_exit_targets, load_context, load_ridge_weights,
    pool_checkpoints, ridge_decisions, save_ridge_weights, traded_checkpoints, train_ridge_weights,
)
from pairs_model.leverage import levered, seed_returns, with_borrow
from pairs_model.metrics import portfolio_metrics, robustness_table
from pairs_model.signal import run_v3


def exit_models(years: range, ctx: dict, train: bool, freeze: int | None) -> dict:
    """Yearly exit models: saved weights, fitting any that are missing (or all with ``train``)."""
    models = {} if train else load_ridge_weights()
    needed = [freeze] if freeze is not None else list(years)
    missing = [y for y in needed if str(y) not in models]
    if missing:
        print(f"Training exit models for {missing} ...", flush=True)
        models.update(train_ridge_weights(pool_checkpoints(ctx), missing))
        save_ridge_weights(dict(sorted(models.items())))
        print(f"  saved to {RIDGE_WEIGHTS_PATH}", flush=True)
    if freeze is not None:
        if min(years) < freeze:
            raise ValueError(f"--freeze-exit-model {freeze} would use future data for {min(years)}.")
        print(f"Using the {freeze} exit model for every year", flush=True)
        return {str(y): models[str(freeze)] for y in years}
    return models


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--start-year", type=int, default=2014)
    parser.add_argument("--end-year", type=int, default=2023)
    parser.add_argument("--train-years", type=int, default=4)
    parser.add_argument("--inner-test-years", type=int, default=2)
    parser.add_argument("--target-vol", type=float, default=0.08)
    parser.add_argument("--max-leverage", type=float, default=4.0)
    parser.add_argument("--vol-halflife", type=float, default=60.0)
    parser.add_argument("--cost-bps", type=float, default=5.0, help="Trading cost for the net results.")
    parser.add_argument("--borrow-bps", type=float, default=50.0, help="Per year, on the short leg.")
    parser.add_argument("--train-exit-model", action="store_true",
                        help="Refit every yearly exit model instead of loading saved weights.")
    parser.add_argument("--freeze-exit-model", type=int, default=None, metavar="YEAR",
                        help="Use the exit model of YEAR for every test year (test years must not be before YEAR).")
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    args = parser.parse_args()
    warnings.simplefilter("ignore")

    data = pd.read_csv("data/pit_pair_prices.csv", parse_dates=["date"])
    selection = pd.read_csv("data/pit_pair_selection.csv")
    years = range(args.start_year, args.end_year + 1)

    print("Running the V3 signal ...", flush=True)
    v3_gross, params = run_v3(data, selection, years, args.train_years)
    first = params.iloc[0][["vix_scale", "band", "band_vix_scale"]].astype(float).to_dict()
    seed = seed_returns(data, selection, first, args.start_year, args.train_years, args.inner_test_years)

    ctx = load_context()
    models = exit_models(years, ctx, args.train_exit_model, args.freeze_exit_model)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    params.to_csv(args.output_dir / "v3_selected_parameters.csv", index=False)
    tables = {}
    for label, cost, borrow in (("gross", 0.0, 0.0),
                                (f"net_{args.cost_bps:g}bps_borrow_{args.borrow_bps:g}",
                                 args.cost_bps, args.borrow_bps)):
        print(f"Exits and leverage ({label}) ...", flush=True)
        v3 = apply_positions_to_returns(v3_gross, transaction_cost_bps=cost)
        checkups, _ = traded_checkpoints(ctx, v3)
        decisions = ridge_decisions(models, checkups, years)
        with_exits = apply_positions_to_returns(apply_exit_targets(v3, decisions, EXIT_RULE),
                                                transaction_cost_bps=cost)
        daily = pd.concat([
            v3.assign(model_name="v3"),
            levered(with_exits, seed, "v5", cost, args.target_vol, args.max_leverage, args.vol_halflife),
        ], ignore_index=True)
        daily = with_borrow(daily, borrow)

        table = portfolio_metrics(daily)
        table["average_leverage"] = table["model_name"].map(
            daily.groupby("model_name")["leverage"].mean()).fillna(1.0)
        robust = robustness_table(daily, baseline="v3")
        table = table.merge(robust[["model_name", "sharpe_ex_2020_2022", "sharpe_p05", "sharpe_p95"]],
                            on="model_name")
        port = daily.groupby(["model_name", "date"])["strategy_return"].mean().unstack(0)
        yearly = port.groupby(port.index.year).apply(lambda r: (1 + r).prod() - 1)

        tables[label] = table
        table.to_csv(args.output_dir / f"summary_{label}.csv", index=False)
        yearly.to_csv(args.output_dir / f"yearly_{label}.csv")
        decisions.to_csv(args.output_dir / f"exit_decisions_{label}.csv", index=False)
        daily.to_csv(args.output_dir / f"daily_{label}.csv", index=False)

    with pd.option_context("display.width", 220, "display.precision", 3):
        for label, table in tables.items():
            print(f"\n== {label} ({args.start_year}-{args.end_year}) ==")
            print(table.to_string(index=False))


if __name__ == "__main__":
    main()
