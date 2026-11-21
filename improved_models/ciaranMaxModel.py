"""ciaranMaxModel: BD's V3 vol-scaled model plus a walk-forward exit classifier.

1. V3 (``improved_models/vol_scaled_naive_model_v3.py``): EWMA z-score
   (half-life 120), naive sizing ``clip(-z/2, -1, 1)``, no-trade band 0.2 and a
   VIX size overlay whose ``vix_scale`` is chosen each year from {0, 0.25} by an
   inner walk-forward on the four prior years. Pairs are re-screened every year
   on the ten years before it (``data/pit_pair_*``).

2. Exit classifier (``improved_models/exit_model.py``): for each open episode (a
   run of same-sign position) at days 5, 10, 20, 40, 60, 90 and 120, a ridge
   regression predicts the episode's remaining P&L from information known at
   that close: days held, |z| now and its change since entry, P&L so far,
   earnings since entry (signed change in the TTM EPS ratio, aligned
   surprises), hedge-ratio shift since entry, spread trend since entry, VIX
   level and change, spread vol ratio and half-life ratio. If the prediction is
   negative the position is closed and stays flat for the rest of the episode.

No look-ahead: the classifier for year Y is trained only on checkpoints from
~110 same-sector pairs' episodes resolved before 1 Jan Y (it never sees V3's
own trades); features use data up to the decision close; the exit is a zero
target at that close, so the position is flat from the next day. Verified by
``python -m tests.test_exit_model_lookahead`` (every decision unchanged when
all data after the test year is removed).

Accounting: exits edit V3's target weights and the result is re-run through
``backtest.apply_positions_to_returns``, so fees follow the backtester.

    python -m improved_models.ciaranMaxModel                    # 10-year screen
    python -m improved_models.ciaranMaxModel --screen 5y        # BD's 5-year screen
    python -m improved_models.ciaranMaxModel --retrain          # refit the exit models

The yearly exit models are saved in ``models/exit_ridge_weights.json`` (13
weights, intercept and input scaling per year) and loaded on later runs; any
year missing from the file is fitted on the fly and added.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from backtest import apply_positions_to_returns, run_walk_forward_backtest
from improved_models.exit_model import (
    RIDGE_WEIGHTS_PATH, load_context, load_ridge_weights, pool_checkpoints, ridge_decisions,
    save_ridge_weights, traded_checkpoints, train_ridge_weights,
)
from improved_models.regression_model import portfolio_metrics
from improved_models.vol_scaled_naive_model import VolScaledNaivePairsTradingModel, select_parameters
from improved_models.vol_scaled_naive_model_v3 import V3_GRID

EXIT_RULE = "ridge<0"
SCREENS = {"10y": "pit", "5y": "pit5y", "2y": "pit2y"}


def run_v3(data: pd.DataFrame, selection: pd.DataFrame, years: range, train_years: int) -> tuple:
    """BD's V3: yearly parameter search at 0 bps, then the walk-forward backtest."""
    frames, chosen = [], []
    for fold_id, year in enumerate(years, start=1):
        pairs = selection.loc[selection["test_year"] == year, "pair_id"].unique()
        year_data = data.loc[data["pair_id"].isin(pairs)]
        params, _ = select_parameters(year_data, year, 0.0, train_years, grid=V3_GRID)
        chosen.append({"test_year": year, **params, "pairs": ",".join(pairs)})
        print(f"  {year}: {params}", flush=True)
        daily = run_walk_forward_backtest(
            year_data, model_factory=lambda p=params: VolScaledNaivePairsTradingModel(**p),
            start_year=year, end_year=year, train_years=train_years,
            transaction_cost_bps=0.0, model_name="vol_scaled_naive_v3",
        )
        daily["fold_id"] = fold_id
        frames.append(daily)
    return pd.concat(frames, ignore_index=True), pd.DataFrame(chosen)


def apply_exit_targets(daily: pd.DataFrame, decisions: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Zero V3's targets from each exit close to the end of that episode."""
    from improved_models.loss_forensics import build_episodes

    d = build_episodes(daily).sort_values(["pair_id", "fold_id", "date"]).reset_index(drop=True)
    d["k"] = d.groupby("ep").cumcount()
    hit = decisions.loc[decisions[rule]].sort_values("checkpoint").drop_duplicates("ep_key")
    exit_k = dict(zip(hit["ep_key"], hit["checkpoint"] - 1))
    held = d["side"] != 0
    d["exit_k"] = ("t" + d["ep"].astype(str)).map(exit_k)
    # target at row r drives the position at row r+1: zero targets from the exit
    # close through the episode's second-to-last held row (its last row's
    # position comes from the previous target, which is now zero).
    last_k = d.loc[held].groupby("ep")["k"].transform("max").reindex(d.index)
    zero = held & d["exit_k"].notna() & (d["k"] >= d["exit_k"]) & (d["k"] < last_k)
    out = daily.copy().sort_values(["pair_id", "fold_id", "date"]).reset_index(drop=True)
    out.loc[zero.to_numpy(), ["target_weight_a", "target_weight_b"]] = 0.0
    out["exited_by_classifier"] = zero.to_numpy()
    keep = [c for c in out.columns if c not in {
        "position_weight_a", "position_weight_b", "position_gross_exposure", "turnover",
        "transaction_cost", "gross_strategy_return", "strategy_return", "cumulative_return"}]
    return out[keep]


def metrics(daily: pd.DataFrame, name: str) -> dict:
    row = portfolio_metrics(daily.assign(model_name=name)).iloc[0].to_dict()
    port = daily.groupby("date")["strategy_return"].mean()
    calm = port[~port.index.year.isin([2020, 2022])]
    row["sharpe_ex_2020_2022"] = calm.mean() / calm.std() * math.sqrt(252)
    row["total_return"] = (1 + port).prod() - 1
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--screen", default="10y", choices=list(SCREENS))
    parser.add_argument("--start-year", type=int, default=2014)
    parser.add_argument("--end-year", type=int, default=2023)
    parser.add_argument("--train-years", type=int, default=4)
    parser.add_argument("--cost-bps", type=float, default=5.0, help="Net-of-fees cost.")
    parser.add_argument("--retrain", action="store_true",
                        help="Refit the yearly exit models instead of loading saved weights.")
    parser.add_argument("--output-dir", type=Path, default=Path("backtest_results") / "ciaranMaxModel")
    args = parser.parse_args()
    warnings.simplefilter("ignore")
    out = args.output_dir / f"screen_{args.screen}"
    out.mkdir(parents=True, exist_ok=True)
    prefix = SCREENS[args.screen]
    data = pd.read_csv(f"data/{prefix}_pair_prices.csv", parse_dates=["date"])
    selection = pd.read_csv(f"data/{prefix}_pair_selection.csv")
    years = range(args.start_year, args.end_year + 1)

    print("Running BD's V3 ...", flush=True)
    v3_gross, params = run_v3(data, selection, years, args.train_years)

    ctx = load_context()
    models = {} if args.retrain else load_ridge_weights()
    missing = [y for y in years if str(y) not in models]
    if missing:
        print(f"Training exit classifier for {missing} (pool data, ~30s) ...", flush=True)
        models.update(train_ridge_weights(pool_checkpoints(ctx), missing))
        save_ridge_weights(dict(sorted(models.items())))
        print(f"  saved to {RIDGE_WEIGHTS_PATH}", flush=True)
    else:
        print(f"Loaded saved exit-classifier weights from {RIDGE_WEIGHTS_PATH}", flush=True)

    rows, outputs = [], {}
    for label, cost in (("gross", 0.0), (f"net_{args.cost_bps:g}bps", args.cost_bps)):
        v3 = apply_positions_to_returns(v3_gross, transaction_cost_bps=cost)
        traded_cp, _ = traded_checkpoints(ctx, v3)
        decisions = ridge_decisions(models, traded_cp, years)
        maxed = apply_positions_to_returns(apply_exit_targets(v3, decisions, EXIT_RULE),
                                           transaction_cost_bps=cost)
        rows.append({"costs": label, **metrics(v3, "vol_scaled_naive_v3 (BD)")})
        rows.append({"costs": label, **metrics(maxed, "ciaranMaxModel"),
                     "episodes_exited": int(decisions.loc[decisions[EXIT_RULE]]["ep_key"].nunique())})
        outputs[label] = (maxed, decisions)

    summary = pd.DataFrame(rows)
    net_label = f"net_{args.cost_bps:g}bps"
    maxed, decisions = outputs[net_label]
    summary.to_csv(out / "summary.csv", index=False)
    params.to_csv(out / "v3_selected_parameters.csv", index=False)
    decisions.to_csv(out / f"exit_decisions_{net_label}.csv", index=False)
    maxed.to_csv(out / f"daily_results_{net_label}.csv", index=False)
    port = pd.DataFrame({
        "v3": apply_positions_to_returns(v3_gross, transaction_cost_bps=args.cost_bps)
        .groupby("date")["strategy_return"].mean(),
        "ciaranMaxModel": maxed.groupby("date")["strategy_return"].mean(),
    })
    yearly = port.groupby(port.index.year).apply(lambda x: (1 + x).prod() - 1)
    yearly.to_csv(out / f"yearly_{net_label}.csv")
    cols = ["model_name", "annualized_return", "annualized_volatility", "sharpe_ratio",
            "max_drawdown", "total_return", "sharpe_ex_2020_2022", "average_daily_turnover"]
    with pd.option_context("display.width", 220, "display.precision", 4):
        for label in ("gross", net_label):
            print(f"\n== {label} ({args.start_year}-{args.end_year}, screen {args.screen}) ==")
            print(summary.loc[summary["costs"] == label, cols].to_string(index=False))
        print(f"\nYearly returns, {net_label} (%):")
        print((yearly * 100).round(2).to_string())


if __name__ == "__main__":
    main()
