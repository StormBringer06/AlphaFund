"""Drift guard: an entry gate for vol_scaled_naive aimed at the slow-drift
episodes behind most of its losses.

Loss forensics showed the worst vol_scaled_naive episodes are not news shocks
but slow drifts (median 184 days held, worst 5 days only ~24% of losses). This
module builds features of a spread that is *trending* rather than reverting,
measured at the close where the entry decision is taken:

  trend_against    t-stat of the spread's 60-day linear trend, signed so that
                   positive = trending in the direction it has diverged
  dir_consistency  share of the last 20 daily spread moves in that direction
  build_days       trading days since z last crossed zero (slow build-up)
  vr10             10-day variance ratio of spread changes over 120 days (>1 trend)
  hl_ratio         log(half-life last 120d / half-life last 504d)
  eps_support      latest EPS surprise of A minus B (<= 95 days old), signed so
                   positive = fundamentals agree with the divergence
  vix_pct, vol_ratio, beta_drift, abs_z

Protocol (no look-ahead):
  A. Descriptive: how the 24 worst traded episodes differ (in-sample, for insight).
  B. Discovery: logistic model of "worst-decile episode" + blocking threshold
     chosen to maximise P&L, using only episodes that ended before 2019 (traded
     2014-2018 episodes plus pseudo-episodes of ~110 same-sector pairs from 2006).
  C. Locked test: the frozen gate on 2019-2023 through backtest.py.
  D. Walk-forward: refit before every year 2014-2023 on episodes resolved
     before that year.
  E. In-sample reference: gate fitted on the 2014-2023 traded episodes themselves.
     This uses look-ahead and is reported only to show the gap.

    python -m improved_models.drift_guard
"""

from __future__ import annotations

import math
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from backtest import run_walk_forward_backtest
from improved_models.CiaranTestmodel import pair_features
from improved_models.loss_forensics import build_episodes
from improved_models.regression_model import portfolio_metrics
from improved_models.vol_scaled_naive_model import VolScaledNaivePairsTradingModel, _Hedge
from screen_pairs import UNIVERSE

OUT = Path("backtest_results") / "drift_guard"
FEATURES = ["trend_against", "dir_consistency", "build_days", "vr10", "hl_ratio",
            "eps_support", "vix_pct", "vol_ratio", "beta_drift", "abs_z"]
SPLIT_YEAR = 2019
THRESHOLD_QUANTILES = (0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.975)


# ----------------------------------------------------------------- features


def surprise_series(earn: pd.DataFrame, ticker: str, dates: pd.DatetimeIndex) -> pd.Series:
    """Latest EPS surprise (%) known at each close, if reported within 95 days."""
    e = earn.loc[(earn["ticker"] == ticker) & earn["Surprise(%)"].notna()].copy()
    if e.empty:
        return pd.Series(0.0, index=dates)
    day = e["ts"].dt.tz_localize(None).dt.normalize()
    e["react"] = np.where(e["ts"].dt.hour < 12, day, day + pd.Timedelta(days=1))
    e = e.sort_values("react")
    s = pd.Series(e["Surprise(%)"].clip(-50, 50).to_numpy(), index=pd.DatetimeIndex(e["react"]))
    s = s[~s.index.duplicated(keep="last")]
    latest = s.reindex(dates.union(s.index)).ffill().reindex(dates)
    known_at = pd.Series(s.index, index=s.index).reindex(dates.union(s.index)).ffill().reindex(dates)
    fresh = (dates - pd.DatetimeIndex(known_at)).days <= 95
    return latest.where(fresh, 0.0).fillna(0.0)


def _half_life(phi: pd.Series) -> pd.Series:
    return (-math.log(2) / np.log(phi.where((phi > 0) & (phi < 1)))).fillna(1000.0).clip(1, 1000)


def raw_features(closes: pd.DataFrame, vix_pct: pd.Series, earn: pd.DataFrame,
                 a: str, b: str) -> pd.DataFrame:
    """Unsigned trailing features, plus z, beta and the unit spread return."""
    f = pair_features(closes, a, b, hedge="ols6m")  # vol_scaled_naive's hedge and z
    z, s = f["z120"], f["spread"]
    ds = s.diff()
    tt = pd.Series(np.arange(len(s), dtype=float), index=s.index)
    n = 60
    var_t, cov = tt.rolling(n).var(), s.rolling(n).cov(tt)
    slope = cov / var_t
    resid_var = ((s.rolling(n).var() - slope * cov) * (n - 1) / (n - 2)).clip(lower=1e-14)
    tstat = slope / np.sqrt(resid_var / ((n - 1) * var_t))
    s_lag = s.shift(1)
    phi = lambda w: s.rolling(w).cov(s_lag) / s_lag.rolling(w).var()
    px = closes[[a, b]]
    la, lb = np.log(px[a]), np.log(px[b])
    ra, rb = px[a].pct_change(fill_method=None), px[b].pct_change(fill_method=None)
    b60 = la.rolling(60).cov(lb) / lb.rolling(60).var()
    b252 = la.rolling(252).cov(lb) / lb.rolling(252).var()
    sign = np.sign(z.to_numpy())
    build = np.zeros(len(z))
    for i in range(1, len(z)):
        build[i] = build[i - 1] + 1 if sign[i] == sign[i - 1] and sign[i] != 0 else 0
    beta_prev = f["beta"].shift(1)
    return pd.DataFrame({
        "z": z,
        "slope_tstat": tstat,
        "up_frac": (ds > 0).astype(float).rolling(20).mean(),
        "build_days": build,
        "vr10": s.diff(10).rolling(120).var() / (10 * ds.rolling(120).var()),
        "hl_ratio": np.log(_half_life(phi(120)) / _half_life(phi(504))),
        "eps_diff": surprise_series(earn, a, closes.index) - surprise_series(earn, b, closes.index),
        "vix_pct": vix_pct,
        "vol_ratio": f["vol_ratio"],
        "beta_drift": (b60 - b252).abs() / b252.abs(),
        "abs_z": z.abs(),
        "beta": f["beta"],
        "spread_ret": (ra - beta_prev * rb) / (1 + beta_prev.abs()),
    }, index=closes.index)


def signed(row: pd.Series | pd.DataFrame, z_sign) -> dict | pd.DataFrame:
    """Orient features so that positive = supports further divergence."""
    out = {
        "trend_against": z_sign * row["slope_tstat"],
        "dir_consistency": 0.5 + z_sign * (row["up_frac"] - 0.5),
        "eps_support": z_sign * row["eps_diff"],
    }
    for c in ("build_days", "vr10", "hl_ratio", "vix_pct", "vol_ratio", "beta_drift", "abs_z"):
        out[c] = row[c]
    return out


# ----------------------------------------------------------------- episodes


def pseudo_episodes(raw: pd.DataFrame, pair: str, band: float = 0.2, cost: float = 5e-4) -> pd.DataFrame:
    """vol_scaled_naive-style positions (clip(-z/2), no-trade band) on any pair."""
    z = raw["z"].to_numpy()
    r = raw["spread_ret"].to_numpy()
    n = len(z)
    pos = np.zeros(n)
    cur = 0.0
    for t in range(n):
        target = 0.0 if not math.isfinite(z[t]) else float(np.clip(-z[t] / 2, -1, 1))
        if abs(target - cur) > band or (target == 0.0 and cur != 0.0):
            cur = target
        pos[t] = cur
    pnl = np.zeros(n)
    pnl[1:] = pos[:-1] * np.nan_to_num(r[1:]) - cost * np.abs(np.diff(pos))
    sgn = np.sign(pos)
    rows, t = [], 0
    while t < n:
        if sgn[t] != 0 and (t == 0 or sgn[t - 1] != sgn[t]):
            u = t
            while u + 1 < n and sgn[u + 1] == sgn[t]:
                u += 1
            if u + 1 < n:  # resolved
                feats = signed(raw.iloc[t], np.sign(z[t]))
                rows.append({"pair": pair, "decision": raw.index[t], "resolved": raw.index[u + 1],
                             "pnl": float(pnl[t + 1:u + 2].sum()), "days": u - t + 1,
                             "source": "pool", **feats})
            t = u + 1
        else:
            t += 1
    return pd.DataFrame(rows)


def traded_episodes(daily: pd.DataFrame, raws: dict) -> pd.DataFrame:
    d = build_episodes(daily)
    held = d.loc[d["side"] != 0]
    eps = held.groupby("ep").agg(pair=("pair_id", "first"), a=("ticker_a", "first"), b=("ticker_b", "first"),
                                 start=("date", "first"), end=("date", "last"),
                                 pnl=("strategy_return", "sum"), days=("date", "size"))
    rows = []
    for ep, e in eps.iterrows():
        raw = raws[(e.a, e.b)]
        i = raw.index.searchsorted(e.start) - 1  # decision taken at the previous close
        row = raw.iloc[i]
        rows.append({"ep": ep, "pair": e.pair, "decision": raw.index[i], "resolved": e.end,
                     "start": e.start, "pnl": e.pnl, "days": e.days, "source": "traded",
                     **signed(row, np.sign(row["z"]))})
    out = pd.DataFrame(rows).sort_values("pnl").reset_index(drop=True)
    out["worst"] = False
    out.loc[: int(0.1 * len(out)) - 1, "worst"] = True
    return out


# ------------------------------------------------------------------- gate


class DriftGate:
    """Logistic P(worst-decile episode); blocks when p exceeds a threshold
    chosen to maximise training-set P&L (no blocking if nothing helps)."""

    def __init__(self, train: pd.DataFrame) -> None:
        x = train[FEATURES].to_numpy(float)
        # worst decile within each source (units differ slightly between sources)
        q = train.groupby("source")["pnl"].transform(lambda p: p.quantile(0.1))
        y = (train["pnl"] <= q).to_numpy()
        self.mu, self.sd = np.nanmean(x, 0), np.nanstd(x, 0)
        self.sd[~(self.sd > 0)] = 1.0
        self.model = LogisticRegression(C=1.0, max_iter=3000).fit(self._x(x), y)
        p = self.model.predict_proba(self._x(x))[:, 1]
        self.coef = dict(zip(FEATURES, self.model.coef_[0]))
        self.train_auc = roc_auc_score(y, p)
        pnl = train["pnl"].to_numpy()
        best_gain, self.threshold = 0.0, np.inf
        for qtl in THRESHOLD_QUANTILES:
            thr = float(np.quantile(p, qtl))
            gain = -pnl[p > thr].sum()
            if gain > best_gain:
                best_gain, self.threshold = gain, thr
        self.train_gain = best_gain

    def _x(self, x: np.ndarray) -> np.ndarray:
        x = np.where(np.isfinite(x), x, self.mu)
        return (x - self.mu) / self.sd

    def prob(self, frame: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(self._x(frame[FEATURES].to_numpy(float)))[:, 1]


class DriftGuardModel(VolScaledNaivePairsTradingModel):
    """vol_scaled_naive that refuses to open a new episode the gate flags; a
    blocked divergence stays flat until z crosses zero."""

    def __init__(self, gate: DriftGate | None, raws: dict, **params: float) -> None:
        super().__init__(**params)
        self.gate, self.raws = gate, raws
        self._raw, self._blocked = None, 0
        self.decisions: list[dict] = []

    def fit(self, train_data: pd.DataFrame) -> DriftGuardModel:
        self._raw = self.raws[(train_data["ticker_a"].iloc[0], train_data["ticker_b"].iloc[0])]
        self._blocked = 0
        return super().fit(train_data)

    def _blocks(self, date: pd.Timestamp, z_sign: float) -> bool:
        if self.gate is None or not np.isfinite(self.gate.threshold):
            return False
        row = self._raw.loc[[date]]
        feats = pd.DataFrame(signed(row, z_sign), index=row.index)
        return bool(self.gate.prob(feats)[0] > self.gate.threshold)

    def _positions(self, segment: pd.DataFrame, hedge: _Hedge) -> pd.DataFrame:
        z = segment["spread_zscore"].to_numpy(float)
        vix_pct = segment["vix_pct"].fillna(0.5).to_numpy(float)
        base = np.clip(-z / self.full_exposure_zscore, -1.0, 1.0)
        mult = np.clip(1.0 + 2.0 * self.vix_scale * (vix_pct - 0.5), 0.5, 1.5)
        target = np.nan_to_num(np.clip(base * mult, -self.max_gross_exposure,
                                       self.max_gross_exposure), nan=0.0)
        band = self.band * np.clip(1.0 + 2.0 * self.band_vix_scale * (0.5 - vix_pct), 0.0, 2.0)
        position = np.empty(len(target))
        blocked_flag = np.zeros(len(target), dtype=bool)
        current = self._position
        for i, (date, wanted, width) in enumerate(zip(segment["date"], target, band)):
            zs = np.sign(z[i]) if math.isfinite(z[i]) else 0.0
            if self._blocked and zs != self._blocked:
                self._blocked = 0  # z crossed zero: the blocked divergence is over
            if self._blocked:
                wanted = 0.0
            if abs(wanted - current) > width or (wanted == 0.0 and current != 0.0 and self._blocked):
                opens = wanted != 0.0 and np.sign(wanted) != np.sign(current)
                if opens and self._blocks(date, zs):
                    self._blocked = int(zs)
                    wanted = 0.0
                    self.decisions.append({"date": date, "blocked": True})
                current = wanted
            position[i] = current
            blocked_flag[i] = bool(self._blocked)
        self._position = current
        scale = 1.0 + abs(hedge.hedge_ratio)
        segment["gate_blocked"] = blocked_flag
        segment["signal"] = position
        segment["target_weight_a"] = position / scale
        segment["target_weight_b"] = -hedge.hedge_ratio * position / scale
        segment["target_gross_exposure"] = np.abs(position)
        return segment


# -------------------------------------------------------------------- main


def run_backtest(data, selection, params, raws, gates: dict, years) -> pd.DataFrame:
    frames = []
    for year in years:
        pairs = selection.loc[selection["test_year"] == year, "pair_id"].unique()
        p = params.loc[params["test_year"] == year].iloc[0]
        kw = {"vix_scale": p.vix_scale, "band": p.band, "band_vix_scale": p.band_vix_scale}
        year_data = data.loc[data["pair_id"].isin(pairs)]
        models = {"vol_scaled_naive (BD)": lambda: VolScaledNaivePairsTradingModel(**kw)}
        for name, gate_by_year in gates.items():
            models[name] = (lambda g=gate_by_year[year]: DriftGuardModel(g, raws, **kw))
        for name, factory in models.items():
            frames.append(run_walk_forward_backtest(year_data, factory, start_year=year, end_year=year,
                                                    train_years=4, transaction_cost_bps=5.0,
                                                    model_name=name))
    return pd.concat(frames, ignore_index=True)


def summarise(daily: pd.DataFrame) -> pd.DataFrame:
    s = portfolio_metrics(daily)
    port = daily.groupby(["model_name", "date"])["strategy_return"].mean().unstack(0)
    s["total_return"] = [(1 + port[m]).prod() - 1 for m in s["model_name"]]
    s["daily_cvar_5pct"] = [port[m][port[m] <= port[m].quantile(0.05)].mean() for m in s["model_name"]]
    return s


def main() -> None:
    warnings.simplefilter("ignore")
    OUT.mkdir(parents=True, exist_ok=True)
    closes = pd.read_csv("data/universe_closes.csv", index_col="date", parse_dates=["date"])
    vix_pct = closes["^VIX"].ffill().rolling(504, min_periods=126).rank(pct=True)
    stocks = closes.drop(columns=["^VIX"])
    earn = pd.read_csv("data/ciaran_extra/earnings_dates.csv")
    earn["ts"] = pd.to_datetime(earn["ts"], utc=True).dt.tz_convert("America/New_York")
    data = pd.read_csv("data/pit_pair_prices.csv", parse_dates=["date"])
    selection = pd.read_csv("data/pit_pair_selection.csv")
    params = pd.read_csv("backtest_results/vol_scaled_naive/selected_parameters_5bps.csv")
    bd = pd.read_csv("backtest_results/vol_scaled_naive/daily_results_5bps.csv",
                     parse_dates=["date"], low_memory=False)
    bd = bd.loc[bd["model_name"] == "vol_scaled_naive"]

    # pool: same-sector pairs; traded orientations added
    pool = [(a, b) for ts in UNIVERSE.values() for i, a in enumerate(sorted(ts))
            for b in sorted(ts)[i + 1:] if a in stocks and b in stocks]
    traded = list(data[["ticker_a", "ticker_b"]].drop_duplicates().itertuples(index=False, name=None))
    print(f"Computing features for {len(set(pool) | set(traded))} pairs ...", flush=True)
    raws = {k: raw_features(stocks, vix_pct, earn, *k) for k in set(pool) | set(traded)}
    rets = np.log(stocks).diff()
    pseudo = []
    for a, b in pool:
        e = pseudo_episodes(raws[(a, b)], f"{a}_{b}".lower())
        if not e.empty:
            pseudo.append(e)
    pseudo = pd.concat(pseudo, ignore_index=True)
    pseudo = pseudo.loc[pseudo["decision"] >= "2006-01-01"]

    def pool_pairs_as_of(year: int) -> set:
        cut = pd.Timestamp(year=year, month=1, day=1)
        w = rets.loc[(rets.index >= cut - pd.DateOffset(years=10)) & (rets.index < cut)]
        return {f"{a}_{b}".lower() for a, b in pool
                if w[a].notna().mean() > 0.95 and w[b].notna().mean() > 0.95 and w[a].corr(w[b]) >= 0.6}

    tr = traded_episodes(bd, raws)

    # ---------------- A. descriptive (in-sample)
    desc = []
    for f in FEATURES:
        desc.append({"feature": f, "median_worst24": tr.loc[tr.worst, f].median(),
                     "median_other": tr.loc[~tr.worst, f].median(),
                     "auc_worst_vs_other": roc_auc_score(tr["worst"], tr[f].fillna(tr[f].median()))})
    desc = pd.DataFrame(desc)
    print("\nA. What distinguishes the 24 worst traded episodes (in-sample, descriptive):")
    print(desc.round(3).to_string(index=False))

    # ---------------- B. discovery on pre-2019 data, C. locked test
    split = pd.Timestamp(year=SPLIT_YEAR, month=1, day=1)
    disc = pd.concat([
        pseudo.loc[pseudo["pair"].isin(pool_pairs_as_of(SPLIT_YEAR)) & (pseudo["resolved"] < split)],
        tr.loc[tr["resolved"] < split].assign(source="traded"),
    ], ignore_index=True)
    locked = DriftGate(disc)
    print(f"\nB. Discovery gate (episodes resolved before {SPLIT_YEAR}: {len(disc)}): train AUC "
          f"{locked.train_auc:.3f}, blocks p > {locked.threshold:.3f}")
    print("   coefficients:", {k: round(v, 2) for k, v in locked.coef.items()})

    late = tr.loc[tr["start"] >= split]
    p_late = locked.prob(late)
    print(f"   Out-of-sample on 2019-2023 traded episodes: AUC {roc_auc_score(late['worst'], p_late):.3f}; "
          f"flags {int((p_late > locked.threshold).sum())}/{len(late)} episodes, of which "
          f"{int((late['worst'] & (p_late > locked.threshold)).sum())}/{int(late['worst'].sum())} worst")

    # ---------------- D. walk-forward, E. in-sample reference
    wf = {}
    for year in range(2014, 2024):
        cut = pd.Timestamp(year=year, month=1, day=1)
        train = pd.concat([
            pseudo.loc[pseudo["pair"].isin(pool_pairs_as_of(year)) & (pseudo["resolved"] < cut)],
            tr.loc[tr["resolved"] < cut].assign(source="traded"),
        ], ignore_index=True)
        wf[year] = DriftGate(train)
    insample = DriftGate(tr.assign(source="traded"))
    years = range(2014, 2024)
    gates = {
        "+ drift guard, locked pre-2019 gate": {y: locked for y in years},
        "+ drift guard, walk-forward": wf,
        "+ drift guard, IN-SAMPLE (look-ahead)": {y: insample for y in years},
    }
    print("\nBacktesting through backtest.py (5 bps) ...", flush=True)
    daily = run_backtest(data, selection, params, raws, gates, years)

    late_daily = daily.loc[daily["test_year"] >= SPLIT_YEAR]
    early_daily = daily.loc[daily["test_year"] < SPLIT_YEAR]
    results = {
        "full_2014_2023": summarise(daily),
        f"test_{SPLIT_YEAR}_2023 (locked gate out-of-sample)": summarise(late_daily),
        f"discovery_2014_{SPLIT_YEAR - 1} (locked gate in-sample)": summarise(early_daily),
    }
    port = daily.groupby(["model_name", "date"])["strategy_return"].mean().unstack(0)
    yearly = port.groupby(port.index.year).apply(lambda x: (1 + x).prod() - 1)
    thresholds = pd.DataFrame({y: {"threshold": g.threshold, "train_auc": g.train_auc,
                                   **{f"coef_{k}": v for k, v in g.coef.items()}}
                               for y, g in wf.items()}).T

    desc.to_csv(OUT / "A_descriptive.csv", index=False)
    tr.to_csv(OUT / "traded_episodes_features.csv", index=False)
    thresholds.to_csv(OUT / "walk_forward_gates.csv")
    yearly.to_csv(OUT / "yearly_returns.csv")
    with pd.option_context("display.width", 250, "display.precision", 4):
        for label, s in results.items():
            s.to_csv(OUT / f"summary_{label.split(' ')[0]}.csv", index=False)
            print(f"\n== {label} ==")
            print(s[["model_name", "annualized_return", "annualized_volatility", "sharpe_ratio",
                     "max_drawdown", "total_return", "average_gross_exposure", "daily_cvar_5pct"]]
                  .to_string(index=False))
        print("\nYearly returns (%):")
        print((yearly * 100).round(2).to_string())
        print("\nWalk-forward gates:")
        print(thresholds[["threshold", "train_auc"]].round(3).to_string())


if __name__ == "__main__":
    main()
