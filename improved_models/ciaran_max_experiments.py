"""Five candidate changes to ciaranMaxModel, scored on the 2014-2023 walk-forward.

Holdout: every input is truncated at 2023-12-31 before anything runs, so the
2024+ holdout is never read.

Baseline = ciaranMaxModel (V3 + ridge exit classifier) on the 10-year screen.
Each change is applied on its own to the baseline's close-of-day targets (so
the backtester's one-day lag still applies), and then the ones that helped are
combined. Every change uses fixed, a-priori settings (no grid search) and only
information available at the decision close:

1. ``more_pairs``   - diversify: the same screen with up to 10 pairs a year
                      (``pit10_*``) instead of 5.
1b. ``relaxed_screen`` - wider screen: p <= 0.10, 3 pairs per sector, up to 12.
2. ``earnings_blackout`` - flat over each reaction day of either stock's
                      earnings report (dates are scheduled in advance).
3. ``inverse_vol``  - size pairs by 1 / trailing 63-day vol of the unit spread
                      return, normalised to the same average size each day.
4. ``vol_target``   - scale the whole book by clip(vol_252 / vol_63, 0.5, 2) of
                      its own trailing returns (de-risk when vol jumps).
5. ``z_stop``       - close the episode and stay flat once |z| > 3.

    python -m improved_models.ciaran_max_experiments
"""

from __future__ import annotations

from pathlib import Path
import warnings

import numpy as np
import pandas as pd

import screen_pairs
from backtest import apply_positions_to_returns
from build_point_in_time_dataset import pair_id as make_pair_id, select_by_year
from build_screened_dataset import make_pair_prices
from improved_models.ciaranMaxModel import EXIT_RULE, apply_exit_targets, run_v3
from improved_models.exit_model import (
    load_context, load_ridge_weights, ridge_decisions, traded_checkpoints,
)

END = pd.Timestamp("2023-12-31")
YEARS = range(2014, 2024)
OUT = Path("backtest_results") / "ciaran_max_experiments"
TARGETS = ["target_weight_a", "target_weight_b"]


# ----------------------------------------------------------------- pipeline

def load_screen(prefix: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = pd.read_csv(f"data/{prefix}_pair_prices.csv", parse_dates=["date"])
    selection = pd.read_csv(f"data/{prefix}_pair_selection.csv")
    return data.loc[data["date"] <= END], selection.loc[selection["test_year"] <= END.year]


def relaxed_screen() -> tuple[pd.DataFrame, pd.DataFrame]:
    closes = pd.read_csv("data/universe_closes.csv", index_col="date", parse_dates=["date"])
    closes = closes.loc[closes.index <= END]
    vix, stocks = closes["^VIX"].dropna(), closes.drop(columns="^VIX")
    saved = screen_pairs.MAX_FULL_PERIOD_PVALUE, screen_pairs.MAX_PAIRS_PER_SECTOR
    screen_pairs.MAX_FULL_PERIOD_PVALUE, screen_pairs.MAX_PAIRS_PER_SECTOR = 0.10, 3
    try:
        selection = select_by_year(stocks, YEARS, 10, 12)
    finally:
        screen_pairs.MAX_FULL_PERIOD_PVALUE, screen_pairs.MAX_PAIRS_PER_SECTOR = saved
    pairs = tuple((p, f"{a} / {b}", a, b) for p, a, b in
                  selection[["pair_id", "ticker_a", "ticker_b"]].drop_duplicates().itertuples(index=False))
    return make_pair_prices(stocks, vix, pairs), selection


def baseline_targets(data, selection, ctx, models) -> pd.DataFrame:
    """ciaranMaxModel's gross targets: V3, then the ridge exit classifier."""
    v3_raw, _ = run_v3(data, selection, YEARS, 4)
    v3 = apply_positions_to_returns(v3_raw, transaction_cost_bps=0.0)
    traded_cp, _ = traded_checkpoints(ctx, v3)
    decisions = ridge_decisions(models, traded_cp, YEARS)
    t = apply_exit_targets(v3, decisions, EXIT_RULE)
    return t.sort_values(["pair_id", "fold_id", "date"]).reset_index(drop=True)


def score(targets: pd.DataFrame, cost_bps: float = 0.0) -> tuple[dict, pd.Series]:
    d = apply_positions_to_returns(targets, transaction_cost_bps=cost_bps)
    port = d.groupby("date")["strategy_return"].mean()
    wealth = (1 + port).cumprod()
    yearly = port.groupby(port.index.year).apply(lambda r: r.mean() / r.std() * np.sqrt(252))
    return {
        "sharpe": port.mean() / port.std() * np.sqrt(252),
        "ann_return": port.mean() * 252,
        "ann_vol": port.std() * np.sqrt(252),
        "max_dd": (wealth / wealth.cummax() - 1).min(),
        "avg_gross": d["position_gross_exposure"].mean(),
        "turnover": d["turnover"].mean(),
    }, yearly


# ------------------------------------------------------------------ changes

def reaction_days(ctx_earn: pd.DataFrame, trading_dates: pd.DatetimeIndex) -> dict[str, set]:
    """First trading day whose return reflects each report (pre-noon -> same day)."""
    out = {}
    for t, e in ctx_earn.groupby("ticker"):
        day = e["ts"].dt.tz_localize(None).dt.normalize()
        react = pd.DatetimeIndex(np.where(e["ts"].dt.hour < 12, day, day + pd.Timedelta(days=1)))
        pos = trading_dates.searchsorted(react)
        out[t] = {trading_dates[p] for p in pos if p < len(trading_dates)}
    return out


def earnings_blackout(t: pd.DataFrame, react: dict[str, set]) -> pd.DataFrame:
    t = t.copy()
    nxt = t.groupby(["pair_id", "fold_id"])["date"].shift(-1)
    hit = np.array([(n in react.get(a, ())) or (n in react.get(b, ()))
                    for n, a, b in zip(nxt, t["ticker_a"], t["ticker_b"])])
    t.loc[hit, TARGETS] = 0.0
    return t


def inverse_vol(t: pd.DataFrame, data: pd.DataFrame, window: int = 63) -> pd.DataFrame:
    t = t.copy()
    vols = []
    for pid, g in t.groupby("pair_id"):
        px = data.loc[data["pair_id"] == pid].set_index("date").sort_index()
        h = g.set_index("date")["hedge_ratio"].groupby(level=0).last()
        h = h.reindex(px.index).ffill().bfill()  # pre-test rows use the first fold's (ex-ante) ratio
        unit = (px["return_a"] - h * px["return_b"]) / (1 + h.abs())
        vol = unit.rolling(window, min_periods=20).std()  # through today's close
        vols.append(pd.DataFrame({"pair_id": pid, "date": vol.index, "unit_vol": vol.to_numpy()}))
    t = t.merge(pd.concat(vols), on=["pair_id", "date"], how="left")
    inv = 1 / t["unit_vol"]
    scale = inv / inv.groupby(t["date"]).transform("mean")
    t[TARGETS] = t[TARGETS].mul(scale.fillna(1.0), axis=0)
    return t.drop(columns="unit_vol")


def vol_target(t: pd.DataFrame) -> pd.DataFrame:
    t = t.copy()
    d = apply_positions_to_returns(t, transaction_cost_bps=0.0)
    port = d.groupby("date")["strategy_return"].mean()
    v63, v252 = port.rolling(63, min_periods=40).std(), port.rolling(252, min_periods=63).std()
    scale = (v252 / v63).clip(0.5, 2.0).fillna(1.0)  # known at each close
    t[TARGETS] = t[TARGETS].mul(t["date"].map(scale).fillna(1.0).to_numpy(), axis=0)
    return t


def z_stop(t: pd.DataFrame, threshold: float = 3.0) -> pd.DataFrame:
    t = t.copy()
    side = np.sign(t["target_weight_a"].round(12))
    run = (side != side.groupby([t["pair_id"], t["fold_id"]]).shift()).cumsum()
    breach = (t["spread_zscore"].abs() > threshold) & (side != 0)
    stopped = breach.groupby(run).cummax()
    t.loc[stopped & (side != 0), TARGETS] = 0.0
    return t


# --------------------------------------------------------------------- main

def main() -> None:
    warnings.simplefilter("ignore")
    OUT.mkdir(parents=True, exist_ok=True)
    ctx = load_context(end=END)
    models = load_ridge_weights()
    assert all(str(y) in models for y in YEARS), "missing saved exit weights"
    react = reaction_days(ctx["earn"] if "earn" in ctx else _earnings(), _trading_dates())

    screens = {"base (5 pairs)": load_screen("pit"), "more_pairs (10)": load_screen("pit10")}
    print("Running relaxed screen ...", flush=True)
    screens["relaxed_screen (12)"] = relaxed_screen()

    base_t = {}
    for name, (data, sel) in screens.items():
        print(f"\n== {name}", flush=True)
        base_t[name] = (baseline_targets(data, sel, ctx, models), data)

    t0, data0 = base_t["base (5 pairs)"]
    variants = {
        "0. baseline ciaranMaxModel": t0,
        "1. more_pairs (10-pair screen)": base_t["more_pairs (10)"][0],
        "1b. relaxed_screen (12, p<=.10)": base_t["relaxed_screen (12)"][0],
        "2. earnings_blackout": earnings_blackout(t0, react),
        "3. inverse_vol sizing": inverse_vol(t0, data0),
        "4. portfolio vol_target": vol_target(t0),
        "5. z_stop |z|>3": z_stop(t0),
    }
    rows, yearly = [], {}
    for name, t in variants.items():
        g, y = score(t, 0.0)
        n, _ = score(t, 5.0)
        rows.append({"variant": name, **g, "net5_sharpe": n["sharpe"]})
        yearly[name] = y
    res = pd.DataFrame(rows).set_index("variant")
    yr = pd.DataFrame(yearly).T
    base_sr = res.loc["0. baseline ciaranMaxModel", "sharpe"]
    res["d_sharpe"] = res["sharpe"] - base_sr
    res["years_better"] = (yr.sub(yr.loc["0. baseline ciaranMaxModel"], axis=1) > 0).sum(1)

    # Combine the changes that helped (screen change first, then target overlays).
    helped = [k for k in res.index[1:] if res.loc[k, "d_sharpe"] > 0]
    print("\nHelped on their own:", helped, flush=True)
    screen_keys = [k for k in helped if k.startswith("1")]
    best_screen = max(screen_keys, key=lambda k: res.loc[k, "sharpe"]) if screen_keys else None
    sk = {"1. more_pairs (10-pair screen)": "more_pairs (10)",
          "1b. relaxed_screen (12, p<=.10)": "relaxed_screen (12)"}.get(best_screen, "base (5 pairs)")
    t, data = base_t[sk]
    steps = [sk]
    if "2. earnings_blackout" in helped:
        t = earnings_blackout(t, react); steps.append("earnings")
    if "5. z_stop |z|>3" in helped:
        t = z_stop(t); steps.append("z_stop")
    if "3. inverse_vol sizing" in helped:
        t = inverse_vol(t, data); steps.append("inverse_vol")
    if "4. portfolio vol_target" in helped:
        t = vol_target(t); steps.append("vol_target")
    g, y = score(t, 0.0)
    n, _ = score(t, 5.0)
    name = "6. combined: " + " + ".join(steps)
    res.loc[name] = {**g, "net5_sharpe": n["sharpe"], "d_sharpe": g["sharpe"] - base_sr,
                     "years_better": int((y - yr.loc["0. baseline ciaranMaxModel"] > 0).sum())}
    yr.loc[name] = y

    res.to_csv(OUT / "summary.csv")
    yr.to_csv(OUT / "yearly_sharpe.csv")
    with pd.option_context("display.width", 250, "display.precision", 3):
        print("\n== 2014-2023, gross (net5_sharpe = same positions at 5 bps)")
        print(res.to_string())
        print("\n== Sharpe by year (gross)")
        print(yr.round(2).to_string())


def _earnings() -> pd.DataFrame:
    e = pd.read_csv("data/ciaran_extra/earnings_dates.csv")
    e["ts"] = pd.to_datetime(e["ts"], utc=True).dt.tz_convert("America/New_York")
    return e.loc[e["ts"].dt.tz_localize(None) <= END + pd.Timedelta(days=1)]


def _trading_dates() -> pd.DatetimeIndex:
    c = pd.read_csv("data/universe_closes.csv", index_col="date", parse_dates=["date"])
    return c.index[c.index <= END]


if __name__ == "__main__":
    main()
