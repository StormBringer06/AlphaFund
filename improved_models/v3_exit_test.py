"""Walk-forward exit classifier (improved_models.exit_model) on BD's V3 model.

V3 = vol_scaled_naive with band 0.2 and vix_scale chosen yearly from {0, 0.25},
on point-in-time 10-year-screened pairs (BD's committed run). Results are shown
gross (BD's headline convention) and net of 5 bps (same positions re-costed).
The exit models for year Y are trained only on pool checkpoints resolved
before 1 Jan Y; they never see V3's own trades.

    python -m improved_models.v3_exit_test [--screen 10y|5y|2y]
"""

from __future__ import annotations

import math
from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from backtest import apply_positions_to_returns
from improved_models.exit_model import (
    RULE_COLS, apply_exits, load_context, pool_checkpoints, traded_checkpoints,
    walk_forward_decisions,
)

V3_DAILY = Path("backtest_results/vol_scaled_naive_v3/screen_10y/daily_results_0bps.csv")
OUT = Path("backtest_results") / "v3_exit_test"


def stats(r: pd.Series) -> dict:
    w = (1 + r).cumprod()
    return {"sharpe": r.mean() / r.std() * math.sqrt(252), "ann_return": w.iloc[-1] ** (252 / len(r)) - 1,
            "ann_vol": r.std() * math.sqrt(252), "max_drawdown": float((w / w.cummax() - 1).min()),
            "sharpe_ex_2020_2022": (lambda x: x.mean() / x.std() * math.sqrt(252))(
                r[~r.index.year.isin([2020, 2022])])}


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--screen", default="10y", choices=["10y", "5y", "2y"],
                        help="Which of BD's V3 runs to use (pair screen length).")
    args = parser.parse_args()
    warnings.simplefilter("ignore")
    out = OUT / f"screen_{args.screen}"
    out.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(V3_DAILY.parent.parent / f"screen_{args.screen}" / V3_DAILY.name,
                      parse_dates=["date"], low_memory=False)
    v3 = raw.loc[raw["model_name"] == "vol_scaled_naive_v3"]
    versions = {"gross": v3, "net 5bps": apply_positions_to_returns(v3, transaction_cost_bps=5.0)}

    print("Building pool checkpoints and walk-forward exit models ...", flush=True)
    ctx = load_context()
    pool_cp = pool_checkpoints(ctx)

    rows, episode_rows = [], []
    for label, daily in versions.items():
        cost = 0.0 if label == "gross" else 5.0
        traded_cp, ep_daily = traded_checkpoints(ctx, daily)
        decisions = walk_forward_decisions(pool_cp, traded_cp, range(2014, 2024))
        base = ep_daily.groupby("date")["strategy_return"].mean()
        rows.append({"costs": label, "model": "vol_scaled_naive_v3 (BD)", **stats(base),
                     "episodes exited": 0, "worst exited": "-"})
        worst = set(traded_cp.loc[traded_cp["worst"], "ep_key"])
        for rule in RULE_COLS:
            hit = decisions.loc[decisions[rule]].sort_values("checkpoint").drop_duplicates("ep_key")
            r = apply_exits(ep_daily, decisions, rule, cost)
            rows.append({"costs": label, "model": f"+ exit: {rule}", **stats(r),
                         "episodes exited": len(hit),
                         "worst exited": f"{int(hit['ep_key'].isin(worst).sum())}/{len(worst)}"})
        if label == "net 5bps":
            decisions.to_csv(out / "exit_decisions_net5bps.csv", index=False)
    table = pd.DataFrame(rows)
    table.to_csv(out / "summary.csv", index=False)
    with pd.option_context("display.width", 220, "display.precision", 4):
        for label in versions:
            print(f"\n== V3 {label} (2014-2023) ==")
            print(table.loc[table["costs"] == label].drop(columns="costs").to_string(index=False))


if __name__ == "__main__":
    main()
