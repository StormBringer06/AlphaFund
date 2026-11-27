"""AlphaFund V7: run the pairs-trading model and write results to ``results/``.

V7 = vol-scaled signal with a low-volatility boost + exit classifier + VIX floor
+ dynamic leverage:

1. **Pairs** - each test year trades the top 5 same-sector pairs from a screen
   run on the ten years before it (``data/pit_pair_*.csv``).
2. **Signal** - EWMA z-score of the spread, sized ``clip(-z/2, -1, 1)`` with a
   VIX overlay, a low-volatility boost (up to 1.5x while the spread is calmer
   than its two-year norm) and a 0.2 no-trade band; ``vix_scale`` is re-chosen
   each year on prior data (``pairs_model/signal.py``).
3. **Exit classifier** - at days 5-120 of each open position a ridge regression
   predicts the remaining P&L; if negative, the position is closed until the
   spread crosses zero. One model per year, trained only on earlier data and
   loaded from ``models/exit_ridge_weights.json`` (``pairs_model/exit_classifier.py``).
4. **VIX floor** - no positions while VIX closes below 12 (``pairs_model/vix_floor.py``).
5. **Leverage** - the book is scaled to an 8% volatility target (max 4x) from
   its own trailing volatility (``pairs_model/leverage.py``).

By default two separate walk-forward runs are made: the development period
2008-2017 and the holdout 2018-2026. Results are reported gross and net of
trading costs plus short-borrow cost.

    python main.py                                    # development + holdout
    python main.py --start-year 2018 --end-year 2026  # one period only
    python main.py --train-exit-model                 # refit the yearly exit models first
"""

from __future__ import annotations

import argparse
from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from backtest import apply_positions_to_returns
from pairs_model.exit_classifier import (
    EXIT_RULE, RIDGE_WEIGHTS_PATH, apply_exit_targets, load_context, load_ridge_weights,
    pool_checkpoints, ridge_decisions, save_ridge_weights, traded_checkpoints, train_ridge_weights,
)
from pairs_model.leverage import levered, seed_returns, with_borrow
from pairs_model.metrics import portfolio_metrics
from pairs_model.signal import run_signal
from pairs_model.vix_floor import VIX_FLOOR, apply_vix_floor

MODEL_NAME = "v7"
PERIODS = (("development", 2008, 2017), ("holdout", 2018, 2026))
SIGNAL_PARAMETERS = ("vix_scale", "band", "band_vix_scale")


def exit_models(years: list[int], ctx: dict, train: bool) -> dict:
    """Yearly exit models: saved weights, fitting any that are missing (or all with ``train``)."""
    models = {} if train else load_ridge_weights()
    missing = [y for y in years if str(y) not in models]
    if missing:
        print(f"Training exit models for {missing} ...", flush=True)
        models.update(train_ridge_weights(pool_checkpoints(ctx), missing))
        save_ridge_weights(dict(sorted(models.items())))
        print(f"  saved to {RIDGE_WEIGHTS_PATH}", flush=True)
    return models


def run_period(data: pd.DataFrame, selection: pd.DataFrame, ctx: dict, models: dict,
               years: range, args: argparse.Namespace) -> tuple[dict, pd.DataFrame]:
    """One walk-forward run over ``years``.

    Returns ``{cost label: (daily results, exit decisions)}`` and the yearly
    signal parameters.
    """
    print(f"Signal {years.start}-{years.stop - 1} ...", flush=True)
    signal_gross, params = run_signal(data, selection, years, args.train_years)
    first = params.iloc[0][list(SIGNAL_PARAMETERS)].astype(float).to_dict()
    seed = seed_returns(data, selection, first, years.start, args.train_years, args.inner_test_years)
    seed = apply_vix_floor(seed, args.vix_floor)

    out = {}
    for label, cost, borrow in (("gross", 0.0, 0.0),
                                (f"net_{args.cost_bps:g}bps_borrow_{args.borrow_bps:g}",
                                 args.cost_bps, args.borrow_bps)):
        signal = apply_positions_to_returns(signal_gross, transaction_cost_bps=cost)
        checkups, _ = traded_checkpoints(ctx, signal)
        decisions = ridge_decisions(models, checkups, years)
        gated = apply_vix_floor(apply_exit_targets(signal, decisions, EXIT_RULE), args.vix_floor)
        daily = levered(apply_positions_to_returns(gated, transaction_cost_bps=cost), seed,
                        MODEL_NAME, cost, args.target_vol, args.max_leverage, args.vol_halflife)
        out[label] = (with_borrow(daily, borrow), decisions)
    return out, params


def summarize(daily: pd.DataFrame, period: str) -> dict:
    row = portfolio_metrics(daily).iloc[0].drop("model_name").to_dict()
    return {"period": period, "start": daily["date"].min().date(), "end": daily["date"].max().date(),
            **row, "average_leverage": float(daily["leverage"].mean()),
            "days_below_vix_floor": int(daily.loc[~daily["vix_gate_open"], "date"].nunique())}


def yearly(daily: pd.DataFrame, period: str) -> pd.DataFrame:
    returns = daily.groupby("date")["strategy_return"].mean()
    by_year = returns.groupby(returns.index.year)
    return pd.DataFrame({
        "period": period,
        "total_return": by_year.apply(lambda r: (1 + r).prod() - 1),
        "sharpe_ratio": by_year.apply(lambda r: r.mean() / r.std() * np.sqrt(252.0)),
    }).rename_axis("year").reset_index()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--start-year", type=int, default=None,
                        help="Run one period from this test year (default: development + holdout).")
    parser.add_argument("--end-year", type=int, default=None)
    parser.add_argument("--train-years", type=int, default=4)
    parser.add_argument("--inner-test-years", type=int, default=2)
    parser.add_argument("--vix-floor", type=float, default=VIX_FLOOR)
    parser.add_argument("--target-vol", type=float, default=0.08)
    parser.add_argument("--max-leverage", type=float, default=4.0)
    parser.add_argument("--vol-halflife", type=float, default=60.0)
    parser.add_argument("--cost-bps", type=float, default=5.0, help="Trading cost for the net results.")
    parser.add_argument("--borrow-bps", type=float, default=50.0, help="Per year, on the short leg.")
    parser.add_argument("--train-exit-model", action="store_true",
                        help="Refit every yearly exit model instead of loading saved weights.")
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    args = parser.parse_args()
    warnings.simplefilter("ignore")

    if (args.start_year is None) != (args.end_year is None):
        parser.error("Give both --start-year and --end-year, or neither.")
    periods = (PERIODS if args.start_year is None
               else (("custom", args.start_year, args.end_year),))

    data = pd.read_csv("data/pit_pair_prices.csv", parse_dates=["date"])
    selection = pd.read_csv("data/pit_pair_selection.csv")
    ctx = load_context()
    all_years = sorted({y for _, a, b in periods for y in range(a, b + 1)})
    models = exit_models(all_years, ctx, args.train_exit_model)

    results: dict[str, list] = {}
    parameters = []
    for period, start, end in periods:
        runs, params = run_period(data, selection, ctx, models, range(start, end + 1), args)
        parameters.append(params)
        for label, (daily, decisions) in runs.items():
            results.setdefault(label, []).append((period, daily, decisions))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.concat(parameters).to_csv(args.output_dir / "selected_parameters.csv", index=False)
    for label, runs in results.items():
        summary = pd.DataFrame([summarize(daily, period) for period, daily, _ in runs])
        summary.to_csv(args.output_dir / f"summary_{label}.csv", index=False)
        pd.concat([yearly(daily, period) for period, daily, _ in runs]).to_csv(
            args.output_dir / f"yearly_{label}.csv", index=False)
        pd.concat([d.assign(period=p) for p, _, d in runs]).to_csv(
            args.output_dir / f"exit_decisions_{label}.csv", index=False)
        pd.concat([d.assign(period=p) for p, d, _ in runs]).to_csv(
            args.output_dir / f"daily_{label}.csv", index=False)
        with pd.option_context("display.width", 220, "display.precision", 3):
            print(f"\n== {label} ==")
            print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
