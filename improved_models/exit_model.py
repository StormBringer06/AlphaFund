"""Combined mid-trade exit model for vol_scaled_naive.

At checkpoints (days 5, 10, 20, 40, 60, 90, 120 of a held episode) we predict
the episode's *remaining* P&L from everything known at that close, and exit
if the prediction says the rest of the trade is expected to lose.

Features (signed so positive = evidence the divergence is real):
  log_days, abs_z_now, z_widening (|z| now - |z| at entry), pnl_so_far,
  eps_confirm (signed change in log TTM EPS ratio since entry, 0 if no report),
  n_reports, surprise_confirm (sum of divergence-aligned surprises since entry),
  hedge_shift (|beta since entry - traded beta| / |traded beta|),
  trend_tstat (spread trend since entry against the position),
  vix_pct, vix_change, vol_ratio, hl_ratio

Models: ridge regression on remaining P&L (winsorised), logistic regression on
P(remaining P&L < 0), and gradient boosting on remaining P&L. Exit thresholds:
"predicted remaining P&L < 0" / "P(loss) > 0.5", and a stricter threshold
chosen on the training data to maximise the P&L change.

Walk-forward, no look-ahead: the model for year Y is trained on pool
pseudo-episodes (~110 same-sector pairs) resolved before 1 Jan Y and applied
to checkpoints dated in year Y - for BD's traded episodes (2014-2023) and for
the pool itself (2010-2023, out of sample).

    python -m improved_models.exit_model
"""

from __future__ import annotations

import math
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge

from improved_models.drift_guard import raw_features
from improved_models.loss_forensics import build_episodes
from improved_models.loss_oracle import ticker_frames
from screen_pairs import UNIVERSE

OUT = Path("backtest_results") / "exit_model"
CHECKPOINTS = (5, 10, 20, 40, 60, 90, 120)
FEATURES = ["log_days", "abs_z_now", "z_widening", "pnl_so_far", "eps_confirm", "n_reports",
            "surprise_confirm", "hedge_shift", "trend_tstat", "vix_pct", "vix_change",
            "vol_ratio", "hl_ratio"]


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


class ExitModels:
    def __init__(self, train: pd.DataFrame, pnl_scale: float) -> None:
        x = train[FEATURES].to_numpy(float)
        self.mu = np.nanmedian(x, 0)
        x = self._fill(x)
        self.sd = x.std(0)
        self.sd[self.sd == 0] = 1.0
        xs = (x - self.mu) / self.sd
        y = train["remaining_pnl"].to_numpy() / pnl_scale
        lo, hi = np.quantile(y, [0.01, 0.99])
        self.ridge = Ridge(alpha=10.0).fit(xs, np.clip(y, lo, hi))
        self.logit = LogisticRegression(C=0.5, max_iter=3000).fit(xs, y < 0)
        self.gbm = HistGradientBoostingRegressor(max_depth=3, max_iter=200, learning_rate=0.05,
                                                 min_samples_leaf=200, random_state=0).fit(x, np.clip(y, lo, hi))
        # stricter thresholds chosen in-sample to maximise the exit P&L change
        rem = train["remaining_pnl"].to_numpy()
        self.thresholds = {}
        for name, score in self.scores(train).items():
            best, best_thr = 0.0, -np.inf if name != "logit" else np.inf
            for q in (0.01, 0.02, 0.05, 0.1, 0.2, 0.3):
                thr = np.quantile(score, 1 - q) if name == "logit" else np.quantile(score, q)
                exit_mask = score > thr if name == "logit" else score < thr
                gain = first_exit_gain(train.assign(exit=exit_mask))
                if gain > best:
                    best, best_thr = gain, thr
            self.thresholds[name] = best_thr
        self.coef = dict(zip(FEATURES, self.ridge.coef_))

    def _fill(self, x: np.ndarray) -> np.ndarray:
        return np.where(np.isfinite(x), x, self.mu)

    def scores(self, df: pd.DataFrame) -> dict:
        x = self._fill(df[FEATURES].to_numpy(float))
        xs = (x - self.mu) / self.sd
        return {"ridge": self.ridge.predict(xs), "logit": self.logit.predict_proba(xs)[:, 1],
                "gbm": self.gbm.predict(x)}


def first_exit_gain(df: pd.DataFrame) -> float:
    """P&L change if each episode exits at its first flagged checkpoint."""
    hit = df.loc[df["exit"]].sort_values("checkpoint").drop_duplicates("ep_key")
    return float(-hit["remaining_pnl"].sum())


def apply_rules(test: pd.DataFrame, models: ExitModels) -> pd.DataFrame:
    s = models.scores(test)
    out = test[["ep_key", "checkpoint", "remaining_pnl"]].copy()
    out["ridge<0"] = s["ridge"] < 0
    out["logit>0.5"] = s["logit"] > 0.5
    out["gbm<0"] = s["gbm"] < 0
    out["ridge strict"] = s["ridge"] < models.thresholds["ridge"]
    out["logit strict"] = s["logit"] > models.thresholds["logit"]
    out["gbm strict"] = s["gbm"] < models.thresholds["gbm"]
    return out


RULE_COLS = ["ridge<0", "logit>0.5", "gbm<0", "ridge strict", "logit strict", "gbm strict"]


# ------------------------------------------------- reusable pipeline pieces


def load_context(end: str | pd.Timestamp | None = None) -> dict:
    """Prices, earnings and a per-pair feature-array builder.

    ``end`` truncates every input (prices, macro, earnings announced after it),
    which the look-ahead test uses."""
    closes = pd.read_csv("data/universe_closes.csv", index_col="date", parse_dates=["date"])
    macro = pd.read_csv("data/ciaran_extra/macro_etf_closes.csv", index_col=0, parse_dates=True)
    earn = pd.read_csv("data/ciaran_extra/earnings_dates.csv")
    earn["ts"] = pd.to_datetime(earn["ts"], utc=True).dt.tz_convert("America/New_York")
    if end is not None:
        end = pd.Timestamp(end)
        closes, macro = closes.loc[closes.index <= end], macro.loc[macro.index <= end]
        earn = earn.loc[earn["ts"].dt.tz_localize(None) <= end + pd.Timedelta(days=1)]
    vix_pct = closes["^VIX"].ffill().rolling(504, min_periods=126).rank(pct=True)
    stocks = closes.drop(columns=["^VIX"])
    tf = ticker_frames(stocks, macro, earn)
    dates = stocks.index
    cache: dict = {}

    def arrays(a: str, b: str) -> dict:
        if (a, b) not in cache:
            raw = raw_features(stocks, vix_pct, earn, a, b)
            cols = ("ttm", "surprise", "is_report")
            cache[(a, b)] = {
                "z": raw["z"].to_numpy(), "beta": raw["beta"].to_numpy(),
                "la": np.log(stocks[a].to_numpy()), "lb": np.log(stocks[b].to_numpy()),
                "vix_pct": raw["vix_pct"].to_numpy(), "vol_ratio": raw["vol_ratio"].to_numpy(),
                "hl_ratio": raw["hl_ratio"].to_numpy(), "spread_ret": raw["spread_ret"].to_numpy(),
                "dates": dates, "A": {c: tf[a][c].to_numpy() for c in cols},
                "B": {c: tf[b][c].to_numpy() for c in cols}}
        return cache[(a, b)]

    return {"stocks": stocks, "dates": dates, "arrays": arrays}


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


def walk_forward_decisions(pool_cp: pd.DataFrame, traded_cp: pd.DataFrame,
                           years: range) -> pd.DataFrame:
    """Model for year Y trained on pool checkpoints resolved before 1 Jan Y."""
    out = []
    for year in years:
        test = traded_cp.loc[traded_cp["date"].dt.year == year]
        if len(test):
            train = pool_cp.loc[pool_cp["resolved"] < pd.Timestamp(year=year, month=1, day=1)]
            out.append(apply_rules(test, ExitModels(train, train["remaining_pnl"].std())))
    return pd.concat(out, ignore_index=True)


def apply_exits(episode_daily: pd.DataFrame, decisions: pd.DataFrame, rule: str,
                cost_bps: float) -> pd.Series:
    """Equal-weight portfolio daily returns after flattening at each first exit.

    An exit decided at the close of held day k zeroes P&L from day k+1 and pays
    one closing cost on that day."""
    x = episode_daily.sort_values(["ep", "date"]).copy()
    x["k"] = x.groupby("ep").cumcount()
    hit = decisions.loc[decisions[rule]].sort_values("checkpoint").drop_duplicates("ep_key")
    kexit = ("t" + x["ep"].astype(str)).map(dict(zip(hit["ep_key"], hit["checkpoint"] - 1)))
    after = kexit.notna() & (x["side"] != 0) & (x["k"] > kexit)
    first_after = after & ~after.groupby(x["ep"]).shift(fill_value=False)
    x.loc[after, "strategy_return"] = 0.0
    x.loc[first_after, "strategy_return"] = -cost_bps / 1e4 * x.loc[first_after, "position_gross_exposure"]
    return x.groupby("date")["strategy_return"].mean()


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
        cols = ("ttm", "surprise", "is_report")
        return {"z": raw["z"].to_numpy(), "beta": raw["beta"].to_numpy(),
                "la": np.log(stocks[a].to_numpy()), "lb": np.log(stocks[b].to_numpy()),
                "vix_pct": raw["vix_pct"].to_numpy(), "vol_ratio": raw["vol_ratio"].to_numpy(),
                "hl_ratio": raw["hl_ratio"].to_numpy(), "spread_ret": raw["spread_ret"].to_numpy(),
                "dates": dates, "A": {c: tf[a][c].to_numpy() for c in cols},
                "B": {c: tf[b][c].to_numpy() for c in cols}}

    # ---------------- pool checkpoints
    print("Building pool checkpoints ...", flush=True)
    pool = [(a, b) for ts in UNIVERSE.values() for i, a in enumerate(sorted(ts))
            for b in sorted(ts)[i + 1:] if a in stocks and b in stocks]
    prow = []
    for a, b in pool:
        P = arrays(a, b)
        z, r = P["z"], P["spread_ret"]
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
                    held = np.arange(t + 1, u + 2)  # days with P&L from the position
                    prow += checkpoint_rows(t, held, pnl[t + 1:u + 2], P, dates[u + 1],
                                            {"ep_key": f"{a}_{b}_{t}", "source": "pool",
                                             "episode_pnl": float(pnl[t + 1:u + 2].sum())})
                t = u + 1
            else:
                t += 1
    pool_cp = pd.DataFrame(prow)

    # ---------------- traded checkpoints (BD vol_scaled_naive)
    bd = pd.read_csv("backtest_results/vol_scaled_naive/daily_results_5bps.csv",
                     parse_dates=["date"], low_memory=False)
    d = build_episodes(bd.loc[bd["model_name"] == "vol_scaled_naive"])
    held_df = d.loc[d["side"] != 0]
    cache, trow = {}, []
    ep_pnl = held_df.groupby("ep")["strategy_return"].sum().sort_values()
    worst = set(ep_pnl.index[: int(0.1 * len(ep_pnl))])
    for ep, g in held_df.groupby("ep"):
        a, b = g["ticker_a"].iloc[0], g["ticker_b"].iloc[0]
        if (a, b) not in cache:
            cache[(a, b)] = arrays(a, b)
        idx = dates.get_indexer(g["date"])
        trow += checkpoint_rows(idx[0] - 1, idx, g["strategy_return"].to_numpy(), cache[(a, b)],
                                g["date"].iloc[-1], {"ep_key": f"t{ep}", "source": "traded",
                                                     "episode_pnl": float(g["strategy_return"].sum()),
                                                     "worst": ep in worst})
    traded_cp = pd.DataFrame(trow)
    print(f"pool checkpoints {len(pool_cp)} from {pool_cp.ep_key.nunique()} episodes; "
          f"traded checkpoints {len(traded_cp)} from {traded_cp.ep_key.nunique()} episodes", flush=True)

    # ---------------- walk-forward
    scale = pool_cp["remaining_pnl"].std()
    traded_dec, pool_dec, coefs = [], [], {}
    for year in range(2010, 2024):
        cut = pd.Timestamp(year=year, month=1, day=1)
        train = pool_cp.loc[pool_cp["resolved"] < cut]
        m = ExitModels(train, train["remaining_pnl"].std())
        coefs[year] = m.coef
        te_pool = pool_cp.loc[pool_cp["date"].dt.year == year]
        pool_dec.append(apply_rules(te_pool, m))
        if year >= 2014:
            te_tr = traded_cp.loc[traded_cp["date"].dt.year == year]
            if len(te_tr):
                traded_dec.append(apply_rules(te_tr, m))
    traded_dec, pool_dec = pd.concat(traded_dec), pd.concat(pool_dec)

    def score(dec: pd.DataFrame, episodes: pd.DataFrame, worst_keys: set) -> pd.DataFrame:
        total = episodes["episode_pnl"].sum()
        rows = []
        for rule in RULE_COLS:
            hit = dec.loc[dec[rule]].sort_values("checkpoint").drop_duplicates("ep_key")
            change = -hit["remaining_pnl"]
            is_worst = hit["ep_key"].isin(worst_keys)
            rows.append({"rule": rule, "episodes exited": len(hit),
                         "worst exited": f"{int(is_worst.sum())}/{len(worst_keys)}",
                         "saved on worst": change[is_worst].sum(),
                         "change on others": change[~is_worst].sum(),
                         "NET change": change.sum(), "net % of total P&L": change.sum() / total})
        return pd.DataFrame(rows)

    tr_eps = traded_cp.drop_duplicates("ep_key")
    tr_worst = set(tr_eps.loc[tr_eps["worst"], "ep_key"])
    pool_eps = pool_cp.loc[pool_cp["date"].dt.year >= 2010].drop_duplicates("ep_key")
    pool_worst = set(pool_eps.loc[pool_eps["episode_pnl"] <= pool_eps["episode_pnl"].quantile(0.1), "ep_key"])
    traded_score = score(traded_dec, tr_eps, tr_worst)
    pool_score = score(pool_dec, pool_eps, pool_worst)
    coefs = pd.DataFrame(coefs).T

    traded_score.to_csv(OUT / "traded_walk_forward.csv", index=False)
    traded_dec.to_csv(OUT / "traded_exit_decisions.csv", index=False)
    pool_score.to_csv(OUT / "pool_walk_forward.csv", index=False)
    coefs.to_csv(OUT / "ridge_coefficients.csv")
    with pd.option_context("display.width", 220, "display.precision", 3):
        print("\nTRADED episodes 2014-2023 (walk-forward, out of sample):")
        print(traded_score.to_string(index=False))
        print("\nPOOL episodes 2010-2023 (walk-forward, out of sample):")
        print(pool_score.to_string(index=False))
        print("\nRidge coefficients (standardised; negative = predicts lower remaining P&L):")
        print(coefs.round(4).to_string())


if __name__ == "__main__":
    main()
