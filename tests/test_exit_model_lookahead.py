"""End-to-end look-ahead test for the walk-forward exit classifier on V3.

For a test year Y, rebuild everything (prices, earnings, pool training data,
V3 episodes and checkpoints) from data truncated at 31 Dec Y and check that
every exit decision dated in Y matches the full-data run exactly. Any feature,
label or scaling that peeked past Y would change at least one decision.

Slow (~1 minute per year); run directly:

    python -m tests.test_exit_model_lookahead
"""

from __future__ import annotations

import warnings

import pandas as pd

from backtest import apply_positions_to_returns
from improved_models.exit_model import (
    FEATURES, RULE_COLS, load_context, pool_checkpoints, traded_checkpoints,
    walk_forward_decisions,
)

V3 = "backtest_results/vol_scaled_naive_v3/screen_10y/daily_results_0bps.csv"
YEARS = (2016, 2019, 2021)


def decisions_for(year: int, end: str | None, v3: pd.DataFrame) -> pd.DataFrame:
    ctx = load_context(end)
    daily = v3 if end is None else v3.loc[v3["date"] <= pd.Timestamp(end)]
    cp, ep_daily = traded_checkpoints(ctx, daily)
    dec = walk_forward_decisions(pool_checkpoints(ctx), cp, range(year, year + 1))
    # key episodes by (pair, first held date) - episode ids depend on row order
    first = ep_daily.loc[ep_daily["side"] != 0].groupby("ep").agg(pair=("pair_id", "first"),
                                                                    start=("date", "first"))
    key = ("t" + first.index.astype(str)).to_series(index=first.index)
    lookup = pd.DataFrame({"ep_key": key.to_numpy(), "pair": first["pair"].to_numpy(),
                           "start": first["start"].to_numpy()})
    feats = cp[["ep_key", "checkpoint", *FEATURES]]
    out = dec.merge(lookup, on="ep_key").merge(feats, on=["ep_key", "checkpoint"])
    return out.drop(columns=["ep_key", "remaining_pnl"]).sort_values(
        ["pair", "start", "checkpoint"]).reset_index(drop=True)


def main() -> None:
    warnings.simplefilter("ignore")
    raw = pd.read_csv(V3, parse_dates=["date"], low_memory=False)
    v3 = apply_positions_to_returns(raw.loc[raw["model_name"] == "vol_scaled_naive_v3"],
                                    transaction_cost_bps=5.0)
    all_ok = True
    for year in YEARS:
        full = decisions_for(year, None, v3)
        trunc = decisions_for(year, f"{year}-12-31", v3)
        same_rows = full[["pair", "start", "checkpoint"]].equals(trunc[["pair", "start", "checkpoint"]])
        same_rules = same_rows and full[RULE_COLS].equals(trunc[RULE_COLS])
        max_feat_diff = (float((full[FEATURES] - trunc[FEATURES]).abs().max().max())
                         if same_rows else float("inf"))
        ok = same_rows and same_rules and max_feat_diff <= 1e-9
        all_ok &= ok
        print(f"{year}: checkpoints {len(full)} vs {len(trunc)} | identical decisions: {same_rules} | "
              f"max feature diff {max_feat_diff:.2e} | {'PASS' if ok else 'FAIL'}", flush=True)
    print("ALL PASS" if all_ok else "FAILURE")
    raise SystemExit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
