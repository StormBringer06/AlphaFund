"""Exit classifier: closes positions whose remaining P&L is predicted negative.

An *episode* is a run of same-sign position in one pair. At check-ups on days
5, 10, 20, 40, 60, 90 and 120 of an open episode, a ridge regression predicts
the episode's remaining P&L from 13 inputs known at that close:

* where the spread is: |z| now, and how much |z| has widened since entry;
* how the trade is going: days held, P&L so far;
* is the relationship breaking: hedge-ratio shift since entry, spread trend
  since entry (t-stat, against the position);
* fundamentals: change in the TTM-EPS ratio of the two stocks since entry
  (signed with the divergence), divergence-aligned earnings surprises, number
  of reports since entry;
* market state: VIX percentile and its change since entry, spread vol ratio
  (20d / 120d), half-life ratio (120d / 504d).

If the prediction is below zero the position is closed and stays flat until
the episode ends (z crosses zero).

Training (no look-ahead): one model per test year Y, fitted on check-ups from
vol-scaled-style episodes on every same-sector pair in the universe that were
*resolved before 1 Jan Y*. The yearly models are stored as plain numbers in
``models/exit_ridge_weights.json`` and loaded at run time.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from pairs_model.universe import UNIVERSE

DATA_DIR = Path("data")


EXIT_RULE = "ridge<0"
CHECKPOINTS = (5, 10, 20, 40, 60, 90, 120)


FEATURES = ["log_days", "abs_z_now", "z_widening", "pnl_so_far", "eps_confirm", "n_reports",
            "surprise_confirm", "hedge_shift", "trend_tstat", "vix_pct", "vix_change",
            "vol_ratio", "hl_ratio"]

# ------------------------------------------------------------------ features


def _ewm_z(spread: pd.Series, halflife: float) -> pd.Series:
    """z-score against the EWMA mean/std known at the previous close."""
    ewm = spread.ewm(halflife=halflife, min_periods=int(halflife))
    mean, std = ewm.mean().shift(1), ewm.std().shift(1)
    return (spread - mean) / std.where(std > 0)


def _half_life(phi: pd.Series) -> pd.Series:
    return (-math.log(2) / np.log(phi.where((phi > 0) & (phi < 1)))).fillna(1000.0).clip(1, 1000)


def spread_features(closes: pd.DataFrame, vix_pct: pd.Series, a: str, b: str) -> pd.DataFrame:
    """The signal's spread for one pair, rebuilt causally, plus the classifier's spread inputs.

    Hedge: OLS of log A on log B over the two years before each 1 Jan / 1 Jul
    refit; the spread history is recomputed with each new hedge and z-scored
    with a 120-day EWMA (as in the signal). All columns use data up to that close.
    """
    px = closes[[a, b]].dropna()
    la, lb = np.log(px[a]), np.log(px[b])
    beta = pd.Series(np.nan, index=px.index)
    spread = pd.Series(np.nan, index=px.index)
    z = pd.Series(np.nan, index=px.index)
    refits = pd.date_range(f"{px.index[0].year}-01-01", px.index[-1], freq="6MS")
    for k, start in enumerate(refits):
        stop = refits[k + 1] if k + 1 < len(refits) else px.index[-1] + pd.Timedelta(days=1)
        known = (px.index < start) & (px.index >= start - pd.DateOffset(years=2))
        seg = (px.index >= start) & (px.index < stop)
        if known.sum() < 120 or not seg.any():
            continue
        b_k = float(lb[known].cov(la[known]) / lb[known].var())
        a_k = float(la[known].mean() - b_k * lb[known].mean())
        upto = px.index < stop
        s_full = la[upto] - (a_k + b_k * lb[upto])
        beta[seg] = b_k
        spread[seg] = s_full[seg[upto]]
        z[seg] = _ewm_z(s_full, 120.0)[seg[upto]]
    change = spread.diff()
    vol_ratio = change.rolling(20).std() / change.rolling(120, min_periods=60).std()
    out = pd.DataFrame({"beta": beta, "spread": spread, "z": z, "vol_ratio": vol_ratio}).reindex(closes.index)

    s, s_lag = out["spread"], out["spread"].shift(1)
    phi = lambda w: s.rolling(w).cov(s_lag) / s_lag.rolling(w).var()
    ra = closes[a].pct_change(fill_method=None)
    rb = closes[b].pct_change(fill_method=None)
    beta_prev = out["beta"].shift(1)
    out["hl_ratio"] = np.log(_half_life(phi(120)) / _half_life(phi(504)))
    out["vix_pct"] = vix_pct
    out["spread_ret"] = (ra - beta_prev * rb) / (1 + beta_prev.abs())
    return out


def earnings_features(earnings: pd.DataFrame, ticker: str, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """TTM EPS, latest surprise and report-day flag, as known at each close.

    A report before noon (New York) counts that day; after the close, the next day.
    """
    e = earnings.loc[earnings["ticker"] == ticker].dropna(subset=["Reported EPS"]).copy()
    day = e["ts"].dt.tz_localize(None).dt.normalize()
    e["react"] = np.where(e["ts"].dt.hour < 12, day, day + pd.Timedelta(days=1))
    e = e.sort_values("react").drop_duplicates("react", keep="last")
    react = pd.DatetimeIndex(e["react"])
    ttm = pd.Series(e["Reported EPS"].rolling(4).sum().to_numpy(), index=react)
    surprise = pd.Series(e["Surprise(%)"].clip(-50, 50).to_numpy(), index=react)
    return pd.DataFrame({
        "ttm": ttm.reindex(dates.union(react)).ffill().reindex(dates),
        "surprise": surprise.reindex(dates.union(react)).ffill().reindex(dates),
        "is_report": dates.isin(react),
    }, index=dates)


def load_context(end: str | pd.Timestamp | None = None) -> dict:
    """Prices, earnings and a cached per-pair input builder.

    ``end`` truncates every input (used by the look-ahead test)."""
    closes = pd.read_csv(DATA_DIR / "universe_closes.csv", index_col="date", parse_dates=["date"])
    earnings = pd.read_csv(DATA_DIR / "earnings_dates.csv")
    earnings["ts"] = pd.to_datetime(earnings["ts"], utc=True).dt.tz_convert("America/New_York")
    if end is not None:
        end = pd.Timestamp(end)
        closes = closes.loc[closes.index <= end]
        earnings = earnings.loc[earnings["ts"].dt.tz_localize(None) <= end + pd.Timedelta(days=1)]
    vix_pct = closes["^VIX"].ffill().rolling(504, min_periods=126).rank(pct=True)
    stocks = closes.drop(columns=["^VIX"])
    dates = stocks.index
    tickers = {t: earnings_features(earnings, t, dates) for t in stocks.columns}
    cache: dict = {}

    def arrays(a: str, b: str) -> dict:
        if (a, b) not in cache:
            f = spread_features(stocks, vix_pct, a, b)
            cols = ("ttm", "surprise", "is_report")
            cache[(a, b)] = {
                **{c: f[c].to_numpy() for c in ("z", "beta", "vix_pct", "vol_ratio", "hl_ratio", "spread_ret")},
                "la": np.log(stocks[a].to_numpy()), "lb": np.log(stocks[b].to_numpy()), "dates": dates,
                "A": {c: tickers[a][c].to_numpy() for c in cols},
                "B": {c: tickers[b][c].to_numpy() for c in cols},
            }
        return cache[(a, b)]

    return {"stocks": stocks, "dates": dates, "arrays": arrays}


# ------------------------------------------------------------------ episodes and check-ups


def build_episodes(daily: pd.DataFrame) -> pd.DataFrame:
    d = daily.sort_values(["pair_id", "fold_id", "date"]).copy()
    side = np.sign(d["position_weight_a"].round(12))
    d["side"] = side
    d["ep"] = (side != side.groupby([d["pair_id"], d["fold_id"]]).shift()).cumsum()
    return d


def checkpoint_rows(i0: int, held: np.ndarray, pnl: np.ndarray, P: dict, resolved, meta: dict) -> list:
    """``held``: calendar indices of held days; ``pnl[k]`` accrues on held[k].

    An exit decided at the close of held[k] avoids pnl[k+1:]."""
    z, la, lb = P["z"], P["la"], P["lb"]
    z_sign = np.sign(z[i0])
    beta0 = P["beta"][i0]
    A, B = P["A"], P["B"]
    r0 = (math.log(A["ttm"][i0] / B["ttm"][i0]) if A["ttm"][i0] > 0 and B["ttm"][i0] > 0 else np.nan)
    rows = []
    for c in CHECKPOINTS:
        k = c - 1
        if k >= len(held) - 1:
            break
        t = held[k]
        window = np.r_[i0 + 1: t + 1]
        x, y = lb[window], la[window]
        reports_a, reports_b = A["is_report"][window], B["is_report"][window]
        eps = 0.0
        if (reports_a.any() or reports_b.any()) and A["ttm"][t] > 0 and B["ttm"][t] > 0 and math.isfinite(r0):
            eps = z_sign * (math.log(A["ttm"][t] / B["ttm"][t]) - r0)
        sur = 0.0
        if reports_a.any():
            sur += z_sign * np.nansum(A["surprise"][window][reports_a])
        if reports_b.any():
            sur -= z_sign * np.nansum(B["surprise"][window][reports_b])
        hedge = np.nan
        if len(window) >= 20 and np.var(x) > 0:
            hedge = abs(np.cov(x, y, bias=True)[0, 1] / np.var(x) - beta0) / abs(beta0)
        s = y - beta0 * x
        tt = np.arange(len(s), dtype=float)
        trend = np.nan
        if len(s) >= 10:
            slope, icpt = np.polyfit(tt, s, 1)
            resid = s - (slope * tt + icpt)
            se = math.sqrt(max(resid.var() * len(s) / (len(s) - 2), 1e-14) / (len(s) * tt.var()))
            trend = z_sign * slope / se
        rows.append({
            **meta, "checkpoint": c, "date": P["dates"][t], "resolved": resolved,
            "remaining_pnl": float(pnl[k + 1:].sum()),
            "log_days": math.log(c), "abs_z_now": abs(z[t]), "z_widening": abs(z[t]) - abs(z[i0]),
            "pnl_so_far": float(pnl[: k + 1].sum()), "eps_confirm": eps,
            "n_reports": int(reports_a.sum() + reports_b.sum()),
            "surprise_confirm": float(np.clip(sur, -100, 100)), "hedge_shift": hedge,
            "trend_tstat": trend, "vix_pct": P["vix_pct"][t],
            "vix_change": P["vix_pct"][t] - P["vix_pct"][i0],
            "vol_ratio": P["vol_ratio"][t], "hl_ratio": P["hl_ratio"][t],
        })
    return rows


def pool_checkpoints(ctx: dict, band: float = 0.2) -> pd.DataFrame:
    """Checkpoints of vol_scaled_naive-style pseudo-episodes on all same-sector pairs."""
    stocks, dates = ctx["stocks"], ctx["dates"]
    pool = [(a, b) for ts in UNIVERSE.values() for i, a in enumerate(sorted(ts))
            for b in sorted(ts)[i + 1:] if a in stocks and b in stocks]
    rows = []
    for a, b in pool:
        P = ctx["arrays"](a, b)
        z, r = P["z"], P["spread_ret"]
        n = len(z)
        pos = np.zeros(n)
        cur = 0.0
        for t in range(n):
            target = 0.0 if not math.isfinite(z[t]) else float(np.clip(-z[t] / 2, -1, 1))
            if abs(target - cur) > band or (target == 0.0 and cur != 0.0):
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
                    rows += checkpoint_rows(t, np.arange(t + 1, u + 2), pnl[t + 1:u + 2], P, dates[u + 1],
                                            {"ep_key": f"{a}_{b}_{t}", "source": "pool",
                                             "episode_pnl": float(pnl[t + 1:u + 2].sum())})
                t = u + 1
            else:
                t += 1
    return pd.DataFrame(rows)


def traded_checkpoints(ctx: dict, model_daily: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Checkpoints for a backtest's episodes; returns (checkpoints, episode-tagged daily)."""
    d = build_episodes(model_daily)
    held = d.loc[d["side"] != 0]
    ep_pnl = held.groupby("ep")["strategy_return"].sum().sort_values()
    worst = set(ep_pnl.index[: int(0.1 * len(ep_pnl))])
    rows = []
    for ep, g in held.groupby("ep"):
        idx = ctx["dates"].get_indexer(g["date"])
        rows += checkpoint_rows(idx[0] - 1, idx, g["strategy_return"].to_numpy(),
                                ctx["arrays"](g["ticker_a"].iloc[0], g["ticker_b"].iloc[0]),
                                g["date"].iloc[-1], {"ep_key": f"t{ep}", "source": "traded",
                                                     "episode_pnl": float(g["strategy_return"].sum()),
                                                     "worst": ep in worst})
    return pd.DataFrame(rows), d


# ------------------------------------------------------------------ yearly ridge models


RIDGE_WEIGHTS_PATH = Path("models") / "exit_ridge_weights.json"


def fit_ridge_weights(train: pd.DataFrame) -> dict:
    """The ridge part of ExitModels as plain numbers (identical fit and scaling)."""
    x = train[FEATURES].to_numpy(float)
    mu = np.nanmedian(x, 0)
    x = np.where(np.isfinite(x), x, mu)
    sd = x.std(0)
    sd[sd == 0] = 1.0
    y = train["remaining_pnl"].to_numpy() / train["remaining_pnl"].std()
    lo, hi = np.quantile(y, [0.01, 0.99])
    ridge = Ridge(alpha=10.0).fit((x - mu) / sd, np.clip(y, lo, hi))
    return {"fill_median": mu.tolist(), "scale_sd": sd.tolist(),
            "coef": ridge.coef_.tolist(), "intercept": float(ridge.intercept_),
            "n_train_checkpoints": int(len(train))}


def train_ridge_weights(pool_cp: pd.DataFrame, years) -> dict:
    """One model per year, each trained on pool checkpoints resolved before 1 Jan."""
    return {str(y): {"trained_on_resolved_before": f"{y}-01-01",
                     **fit_ridge_weights(pool_cp.loc[pool_cp["resolved"] < pd.Timestamp(year=y, month=1, day=1)])}
            for y in years}


def save_ridge_weights(models: dict, path: Path = RIDGE_WEIGHTS_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"features": FEATURES, "checkpoints": list(CHECKPOINTS), "ridge_alpha": 10.0,
               "rule": "exit when predicted remaining P&L < 0", "models": models}
    path.write_text(json.dumps(payload, indent=1))


def load_ridge_weights(path: Path = RIDGE_WEIGHTS_PATH) -> dict:
    """Saved yearly models, or {} if missing or built for different features."""
    if not path.exists():
        return {}
    payload = json.loads(path.read_text())
    if payload.get("features") != FEATURES or payload.get("checkpoints") != list(CHECKPOINTS):
        return {}
    return payload["models"]


def ridge_decisions(models: dict, traded_cp: pd.DataFrame, years) -> pd.DataFrame:
    """"ridge<0" decisions for checkpoints dated in each year, from saved weights."""
    out = []
    for year in years:
        test = traded_cp.loc[traded_cp["date"].dt.year == year]
        if not len(test):
            continue
        m = models[str(year)]
        mu, sd = np.array(m["fill_median"]), np.array(m["scale_sd"])
        x = test[FEATURES].to_numpy(float)
        x = np.where(np.isfinite(x), x, mu)
        pred = ((x - mu) / sd) @ np.array(m["coef"]) + m["intercept"]
        dec = test[["ep_key", "checkpoint", "remaining_pnl"]].copy()
        dec["predicted_remaining"] = pred
        dec["ridge<0"] = pred < 0
        out.append(dec)
    return pd.concat(out, ignore_index=True)


# ------------------------------------------------------------------ applying exits


def apply_exit_targets(daily: pd.DataFrame, decisions: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Zero the signal's targets from each exit close to the end of that episode."""
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
