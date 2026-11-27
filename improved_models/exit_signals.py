"""Mid-trade exit signals for vol_scaled_naive: once a position is open, can we
tell it is diverging for good and close it?

The right target for an exit is the *remaining* P&L after the exit decision,
not whether the episode ends up a loser. Each rule is checked at every close
while a position is held; at the first trigger the position is flattened
(from the next day) for the rest of the episode. A rule helps only if the P&L
it removes is negative overall.

Rules (fixed thresholds, all information known at the trigger close):
  earnings_confirm   a report from either stock moves log(TTM EPS A / B) since
                     entry in the direction of the price divergence
  surprise_confirm   the outperforming stock beats (or the lagging one misses)
                     its consensus by more than 5%
  hedge_shift        after 40+ days, the hedge ratio fitted since entry differs
                     from the traded one by more than 30%
  trend_since_entry  after 20+ days, the spread's trend since entry has a
                     t-stat above 2 against the position
  new_extreme        |z| has risen 1.5 above its value at entry
  earnings_or_hedge  earnings_confirm or hedge_shift

Replication: every rule is also scored on ~13,000 vol_scaled_naive-style
pseudo-episodes from ~110 same-sector pairs (2006-2023).

    python -m improved_models.exit_signals
"""

from __future__ import annotations

from pathlib import Path
import math
import warnings

import numpy as np
import pandas as pd

from improved_models.drift_guard import raw_features
from improved_models.loss_forensics import build_episodes
from improved_models.loss_oracle import ticker_frames
from screen_pairs import UNIVERSE

OUT = Path("backtest_results") / "exit_signals"
RULES = ["earnings_confirm", "surprise_confirm", "hedge_shift", "trend_since_entry",
         "new_extreme", "earnings_or_hedge"]


def triggers(i0: int, days: np.ndarray, z: np.ndarray, beta_entry: float, la: np.ndarray,
             lb: np.ndarray, spread: np.ndarray, A: dict, B: dict) -> dict:
    """First trigger index (into ``days``) for every rule, or -1.

    ``i0`` is the decision index; ``days`` are the calendar indices at which a
    position is held and an exit could be decided."""
    z_sign = np.sign(z[i0])
    ratio0 = (math.log(A["ttm"][i0] / B["ttm"][i0])
              if A["ttm"][i0] > 0 and B["ttm"][i0] > 0 else np.nan)
    first = {r: -1 for r in RULES}
    # cumulative sums for the since-entry hedge and trend regressions
    sx = sy = sxx = sxy = 0.0
    st = ss = stt = sts = sss = 0.0
    n = 0
    for k, t in enumerate(days):
        x, y = lb[t], la[t]
        s = y - beta_entry * x  # spread at the traded (entry) hedge, no refit jumps
        n += 1
        sx += x; sy += y; sxx += x * x; sxy += x * y
        st += n; ss += s; stt += n * n; sts += n * s; sss += s * s
        fired = {}
        report_a, report_b = A["is_report"][t], B["is_report"][t]
        if report_a or report_b:
            if A["ttm"][t] > 0 and B["ttm"][t] > 0 and math.isfinite(ratio0):
                fired["earnings_confirm"] = z_sign * (math.log(A["ttm"][t] / B["ttm"][t]) - ratio0) > 0
            sa, sb = A["surprise"][t], B["surprise"][t]
            fired["surprise_confirm"] = bool(
                (report_a and math.isfinite(sa) and z_sign * sa > 5)
                or (report_b and math.isfinite(sb) and -z_sign * sb > 5))
        if n >= 40:
            vx = sxx / n - (sx / n) ** 2
            if vx > 0:
                b_since = (sxy / n - sx / n * sy / n) / vx
                fired["hedge_shift"] = abs(b_since - beta_entry) > 0.3 * abs(beta_entry)
        if n >= 20:
            vt = stt / n - (st / n) ** 2
            cov = sts / n - st / n * ss / n
            slope = cov / vt
            resid = max(sss / n - (ss / n) ** 2 - slope * cov, 1e-14) * n / (n - 2)
            tstat = slope / math.sqrt(resid / (n * vt))
            fired["trend_since_entry"] = z_sign * tstat > 2
        if math.isfinite(z[t]) and math.isfinite(z[i0]):
            fired["new_extreme"] = abs(z[t]) > abs(z[i0]) + 1.5
        fired["earnings_or_hedge"] = fired.get("earnings_confirm", False) or fired.get("hedge_shift", False)
        for r in RULES:
            if first[r] < 0 and fired.get(r, False):
                first[r] = k
    return first


def evaluate(pnl_after: np.ndarray, first: dict) -> dict:
    """P&L change of each rule: minus the P&L after the trigger close."""
    return {r: (-float(pnl_after[k]) if k >= 0 else 0.0) for r, k in first.items()}


def pair_arrays(raw: pd.DataFrame, A: pd.DataFrame, B: pd.DataFrame, a: str, b: str,
                stocks: pd.DataFrame) -> tuple:
    return (raw["z"].to_numpy(), raw["beta"].to_numpy(), np.log(stocks[a].to_numpy()),
            np.log(stocks[b].to_numpy()), raw["spread"].to_numpy() if "spread" in raw else None,
            {c: A[c].to_numpy() for c in ("ttm", "surprise", "is_report")},
            {c: B[c].to_numpy() for c in ("ttm", "surprise", "is_report")})


def main() -> None:
    warnings.simplefilter("ignore")
    OUT.mkdir(parents=True, exist_ok=True)
    closes = pd.read_csv("data/universe_closes.csv", index_col="date", parse_dates=["date"])
    vix_pct = closes["^VIX"].ffill().rolling(504, min_periods=126).rank(pct=True)
    stocks = closes.drop(columns=["^VIX"])
    macro = pd.read_csv("data/ciaran_extra/macro_etf_closes.csv", index_col=0, parse_dates=True)
    earn = pd.read_csv("data/ciaran_extra/earnings_dates.csv")
    earn["ts"] = pd.to_datetime(earn["ts"], utc=True).dt.tz_convert("America/New_York")
    tf = ticker_frames(stocks, macro, earn)
    dates = stocks.index

    def arrays(a, b):
        raw = raw_features(stocks, vix_pct, earn, a, b)
        raw["spread"] = np.log(stocks[a]) - raw["beta"] * np.log(stocks[b])
        return pair_arrays(raw, tf[a], tf[b], a, b, stocks), raw

    # ---------------- traded episodes (BD's vol_scaled_naive backtest)
    bd = pd.read_csv("backtest_results/vol_scaled_naive/daily_results_5bps.csv",
                     parse_dates=["date"], low_memory=False)
    d = build_episodes(bd.loc[bd["model_name"] == "vol_scaled_naive"])
    held = d.loc[d["side"] != 0]
    cache, rows = {}, []
    for ep, g in held.groupby("ep"):
        a, b = g["ticker_a"].iloc[0], g["ticker_b"].iloc[0]
        if (a, b) not in cache:
            cache[(a, b)] = arrays(a, b)
        (z, beta, la, lb, spread, A, B), _ = cache[(a, b)]
        idx = dates.get_indexer(g["date"])
        i0 = idx[0] - 1
        pnl = g["strategy_return"].to_numpy()
        after = np.concatenate([np.cumsum(pnl[::-1])[::-1][1:], [0.0]])  # P&L after each day
        first = triggers(i0, idx, z, beta[i0], la, lb, spread, A, B)
        rows.append({"ep": ep, "pnl": pnl.sum(), **evaluate(after, first),
                     **{f"day_{r}": first[r] for r in RULES}})
    traded = pd.DataFrame(rows).sort_values("pnl").reset_index(drop=True)
    traded["worst"] = False
    traded.loc[: int(0.1 * len(traded)) - 1, "worst"] = True

    # ---------------- pool pseudo-episodes
    pool = [(a, b) for ts in UNIVERSE.values() for i, a in enumerate(sorted(ts))
            for b in sorted(ts)[i + 1:] if a in stocks and b in stocks]
    rows = []
    for a, b in pool:
        (z, beta, la, lb, spread, A, B), raw = arrays(a, b)
        r = raw["spread_ret"].to_numpy()
        n = len(z)
        pos = np.zeros(n)
        cur = 0.0
        for t in range(n):
            target = 0.0 if not math.isfinite(z[t]) else float(np.clip(-z[t] / 2, -1, 1))
            if abs(target - cur) > 0.2 or (target == 0.0 and cur != 0.0):
                cur = target
            pos[t] = cur
        pnl = np.zeros(n)
        pnl[1:] = pos[:-1] * np.nan_to_num(r[1:]) - 5e-4 * np.abs(np.diff(pos))
        sgn = np.sign(pos)
        t = 0
        while t < n:
            if sgn[t] != 0 and (t == 0 or sgn[t - 1] != sgn[t]):
                u = t
                while u + 1 < n and sgn[u + 1] == sgn[t]:
                    u += 1
                if u + 1 < n and dates[t] >= pd.Timestamp("2006-01-01"):
                    ep_pnl = pnl[t + 1:u + 2]  # day t+1 .. u+1
                    held_days = np.arange(t + 1, u + 1)  # closes where an exit can be decided
                    after = np.array([ep_pnl[k + 1:].sum() for k in range(len(held_days))])
                    first = triggers(t, held_days, z, beta[t], la, lb, spread, A, B)
                    rows.append({"pair": f"{a}_{b}", "pnl": ep_pnl.sum(), **evaluate(after, first)})
                t = u + 1
            else:
                t += 1
    pool_df = pd.DataFrame(rows)
    pool_df["worst"] = pool_df["pnl"] <= pool_df["pnl"].quantile(0.1)

    # ---------------- summary
    out = []
    for r in RULES:
        tw, to = traded.loc[traded.worst], traded.loc[~traded.worst]
        pw, po = pool_df.loc[pool_df.worst], pool_df.loc[~pool_df.worst]
        out.append({
            "rule": r,
            "traded: worst exited": f"{int((tw[f'day_{r}'] >= 0).sum())}/{len(tw)}",
            "traded: loss saved on worst": tw[r].sum(),
            "traded: others exited": f"{int((to[f'day_{r}'] >= 0).sum())}/{len(to)}",
            "traded: change on others": to[r].sum(),
            "traded: NET change": traded[r].sum(),
            "traded: net as % of total P&L": traded[r].sum() / traded["pnl"].sum(),
            "pool: loss saved on worst": pw[r].sum(),
            "pool: change on others": po[r].sum(),
            "pool: NET change": pool_df[r].sum(),
            "pool: net as % of total P&L": pool_df[r].sum() / pool_df["pnl"].sum(),
        })
    out = pd.DataFrame(out)
    out.to_csv(OUT / "exit_rules.csv", index=False)
    traded.to_csv(OUT / "traded_episode_exits.csv", index=False)
    print(f"Traded episodes {len(traded)} (total P&L {traded['pnl'].sum():.3f}); "
          f"pool episodes {len(pool_df)} (total P&L {pool_df['pnl'].sum():.2f})")
    with pd.option_context("display.width", 260, "display.precision", 3):
        print(out.to_string(index=False))


if __name__ == "__main__":
    main()
