"""End-to-end look-ahead check for the exit classifier.

For a test year Y, rebuild everything (prices, earnings, pool training data,
signal episodes and checkpoints) from data truncated at 31 Dec Y and check that
every exit decision dated in Y matches the full-data run exactly. Any feature,
label or scaling that peeked past Y would change at least one decision.

Slow (~1 minute per year), so it is not part of the unit tests; run directly:

    python -m tests.lookahead_check
"""

from __future__ import annotations

import warnings

import pandas as pd

from backtest import apply_positions_to_returns
from pairs_model.exit_classifier import (
    EXIT_RULE, FEATURES, load_context, pool_checkpoints, ridge_decisions, traded_checkpoints,
    train_ridge_weights,
)
from pairs_model.signal import run_signal

YEARS = (2016, 2019, 2021)


def decisions_for(year: int, end: str | None, signal: pd.DataFrame) -> pd.DataFrame:
    ctx = load_context(end)
    daily = signal if end is None else signal.loc[signal["date"] <= pd.Timestamp(end)]
    cp, ep_daily = traded_checkpoints(ctx, daily)
    models = train_ridge_weights(pool_checkpoints(ctx), [year])
    dec = ridge_decisions(models, cp, [year])
    # key episodes by (pair, first held date) - episode ids depend on row order
    first = ep_daily.loc[ep_daily["side"] != 0].groupby("ep").agg(pair=("pair_id", "first"),
                                                                    start=("date", "first"))
    key = ("t" + first.index.astype(str)).to_series(index=first.index)
    lookup = pd.DataFrame({"ep_key": key.to_numpy(), "pair": first["pair"].to_numpy(),
                           "start": first["start"].to_numpy()})
    feats = cp[["ep_key", "checkpoint", *FEATURES]]
    out = dec.merge(lookup, on="ep_key").merge(feats, on=["ep_key", "checkpoint"])
    return out.drop(columns=["ep_key", "remaining_pnl", "predicted_remaining"]).sort_values(
        ["pair", "start", "checkpoint"]).reset_index(drop=True)


def main() -> None:
    warnings.simplefilter("ignore")
    data = pd.read_csv("data/pit_pair_prices.csv", parse_dates=["date"])
    selection = pd.read_csv("data/pit_pair_selection.csv")
    signal_gross, _ = run_signal(data, selection, range(2008, max(YEARS) + 1), 4)
    signal = apply_positions_to_returns(signal_gross, transaction_cost_bps=5.0)
    all_ok = True
    for year in YEARS:
        full = decisions_for(year, None, signal)
        trunc = decisions_for(year, f"{year}-12-31", signal)
        same_rows = full[["pair", "start", "checkpoint"]].equals(trunc[["pair", "start", "checkpoint"]])
        same_rules = same_rows and full[[EXIT_RULE]].equals(trunc[[EXIT_RULE]])
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
