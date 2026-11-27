"""Quick test: does a hedge-ratio-aware divergence classifier help vol_scaled_naive?

Not an official model - a fast check of whether the idea is promising.

Stage 1 (hedge ratio): a Kalman filter on log prices with a random-walk
[beta, alpha] state, the standard way to track a rolling hedge ratio. Its
estimate is compared with the 2-year OLS hedge vol_scaled_naive trades; a large
gap means the relationship is moving away from the traded hedge.

Stage 2 (divergence): logistic regression (linear in the log-odds) predicting
whether a divergence episode fails to revert. An episode starts when |z|
crosses 1; it is labelled 1 if |z| reaches 3 or it has not crossed zero within
126 trading days, else 0 (z crossed zero first). Features at the episode start:
Kalman-vs-OLS hedge gap, short/long hedge drift, correlation drop, spread AR(1)
coefficient, divergence speed, idiosyncratic jump, leg momentum gap, VIX
percentile and |z|.

No look-ahead: features are trailing; the classifier for test year Y is
trained only on episodes resolved before 1 Jan Y, from pairs whose 10-year
return correlation before Y is >= 0.6; trading pairs and vol_scaled_naive
parameters are BD's point-in-time selections.

    python -m improved_models.divergence_filter_test
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
from improved_models.regression_model import portfolio_metrics
from improved_models.vol_scaled_naive_model import VolScaledNaivePairsTradingModel, _Hedge
from screen_pairs import UNIVERSE

OUT = Path("backtest_results") / "divergence_filter_test"
FEATURES = ["hedge_gap", "beta_drift", "corr_drop", "ar_phi", "div_speed",
            "jump10", "mom_gap", "vix_pct", "abs_z"]
ENTRY_Z, DIVERGE_Z, MAX_DAYS = 1.0, 3.0, 126


# ------------------------------------------------------------------ stage 1


def kalman_beta(log_a: np.ndarray, log_b: np.ndarray, delta: float = 1e-5,
                obs_var: float = 1e-3) -> np.ndarray:
    """Posterior beta at each close from log_a = alpha + beta * log_b + e."""
    n = len(log_a)
    beta_out = np.full(n, np.nan)
    b, a = 1.0, 0.0
    p11, p12, p22 = 1.0, 0.0, 1.0
    q = delta / (1 - delta)
    for t in range(n):
        x, y = log_b[t], log_a[t]
        if not (math.isfinite(x) and math.isfinite(y)):
            continue
        r11, r12, r22 = p11 + q, p12, p22 + q
        e = y - (b * x + a)
        s = x * x * r11 + 2 * x * r12 + r22 + obs_var
        k1, k2 = (x * r11 + r12) / s, (x * r12 + r22) / s
        b, a = b + k1 * e, a + k2 * e
        f1, f2 = x * r11 + r12, x * r12 + r22  # F R
        p11, p12, p22 = r11 - k1 * f1, r12 - k1 * f2, r22 - k2 * f2
        beta_out[t] = b
    return beta_out


def features_for(closes: pd.DataFrame, vix_pct: pd.Series, a: str, b: str) -> pd.DataFrame:
    """All trailing; indexed by date on the closes calendar."""
    f = pair_features(closes, a, b, hedge="ols6m")  # vol_scaled_naive's hedge and z
    px = closes[[a, b]]
    la, lb = np.log(px[a]), np.log(px[b])
    ra, rb = px[a].pct_change(fill_method=None), px[b].pct_change(fill_method=None)
    kb = pd.Series(kalman_beta(la.to_numpy(), lb.to_numpy()), index=closes.index)
    roll_beta = lambda w: la.rolling(w).cov(lb) / lb.rolling(w).var()
    b60, b252 = roll_beta(60), roll_beta(252)
    s = f["spread"]
    s_lag = s.shift(1)
    resid = ra - f["beta"] * rb
    resid_sd = resid.rolling(120, min_periods=60).std().shift(1)
    z = f["z120"]
    out = pd.DataFrame({
        "z": z,
        "hedge_gap": (kb - f["beta"]).abs() / f["beta"].abs(),
        "beta_drift": (b60 - b252).abs() / b252.abs(),
        "corr_drop": ra.rolling(252).corr(rb) - ra.rolling(60).corr(rb),
        "ar_phi": s.rolling(120).cov(s_lag) / s_lag.rolling(120).var(),
        "div_speed": (z - z.shift(5)).abs(),
        "jump10": (resid.abs() / resid_sd).rolling(10).max(),
        "mom_gap": (la.diff(126) - lb.diff(126)).abs(),
        "vix_pct": vix_pct,
        "abs_z": z.abs(),
    }, index=closes.index)
    return out


# ------------------------------------------------------------------ stage 2


def episodes(feat: pd.DataFrame) -> pd.DataFrame:
    """Triple-barrier episodes from the z-score."""
    z = feat["z"].to_numpy()
    dates = feat.index
    rows, t, n = [], 1, len(z)
    while t < n:
        if math.isfinite(z[t]) and abs(z[t]) > ENTRY_Z and not abs(z[t - 1]) > ENTRY_Z:
            side = np.sign(z[t])
            label, end = None, None
            for u in range(t + 1, min(t + MAX_DAYS, n - 1) + 1):
                if not math.isfinite(z[u]):
                    break
                if side * z[u] <= 0:
                    label, end = 0, u
                    break
                if abs(z[u]) >= DIVERGE_Z:
                    label, end = 1, u
                    break
            else:
                if t + MAX_DAYS <= n - 1:
                    label, end = 1, t + MAX_DAYS
            if label is not None:
                rows.append({"start": dates[t], "resolved": dates[end], "label": label,
                             **feat.iloc[t][FEATURES].to_dict()})
                t = end
        t += 1
    return pd.DataFrame(rows)


class DivergenceClassifier:
    def __init__(self, train: pd.DataFrame, features: list[str]) -> None:
        self.features = features
        x = train[features].to_numpy(float)
        self.mu, self.sd = np.nanmean(x, 0), np.nanstd(x, 0)
        self.sd[~(self.sd > 0)] = 1.0
        self.model = LogisticRegression(C=1.0, max_iter=2000).fit(self._x(x), train["label"])
        p = self.model.predict_proba(self._x(x))[:, 1]
        self.base_rate = float(train["label"].mean())
        self.gate = float(np.quantile(p, 0.8))  # riskiest 20% of training episodes
        self.coef = dict(zip(features, self.model.coef_[0]))

    def _x(self, x: np.ndarray) -> np.ndarray:
        x = np.where(np.isfinite(x), x, self.mu)
        return (x - self.mu) / self.sd

    def prob(self, row: pd.Series | pd.DataFrame) -> np.ndarray:
        x = np.atleast_2d(np.asarray(row[self.features], dtype=float))
        return self.model.predict_proba(self._x(x))[:, 1]


# ------------------------------------------------------------------- model


class DivergenceFilteredModel(VolScaledNaivePairsTradingModel):
    """vol_scaled_naive; within an episode (|z| > 1 until z crosses zero) the
    target is scaled by a multiplier fixed when the episode starts.

    mode: "gate" (0 if p is in the riskiest training quintile), "soft"
    (scale by (1 - p) / (1 - base rate), capped at 1), or "stop" (simple rule:
    flat for the rest of the episode once |z| >= 3; no classifier)."""

    def __init__(self, mode: str, classifier: DivergenceClassifier | None,
                 feature_table: dict, **params: float) -> None:
        super().__init__(**params)
        self.mode, self.clf, self.tables = mode, classifier, feature_table
        self._ep_side, self._ep_mult, self._feat = 0, 1.0, None

    def fit(self, train_data: pd.DataFrame) -> DivergenceFilteredModel:
        a, b = train_data["ticker_a"].iloc[0], train_data["ticker_b"].iloc[0]
        self._feat = self.tables[(a, b)]
        self._ep_side, self._ep_mult = 0, 1.0
        return super().fit(train_data)

    def _multiplier(self, date: pd.Timestamp) -> float:
        if self.mode == "stop" or self.clf is None:
            return 1.0
        p = float(self.clf.prob(self._feat.loc[[date]])[0])
        if self.mode == "gate":
            return 0.0 if p > self.clf.gate else 1.0
        return float(np.clip((1 - p) / (1 - self.clf.base_rate), 0.0, 1.0))

    def _positions(self, segment: pd.DataFrame, hedge: _Hedge) -> pd.DataFrame:
        z = segment["spread_zscore"].to_numpy(float)
        vix_pct = segment["vix_pct"].fillna(0.5).to_numpy(float)
        base = np.clip(-z / self.full_exposure_zscore, -1.0, 1.0)
        mult_vix = np.clip(1.0 + 2.0 * self.vix_scale * (vix_pct - 0.5), 0.5, 1.5)
        target = np.nan_to_num(np.clip(base * mult_vix, -self.max_gross_exposure,
                                       self.max_gross_exposure), nan=0.0)
        band = self.band * np.clip(1.0 + 2.0 * self.band_vix_scale * (0.5 - vix_pct), 0.0, 2.0)
        ep_mult = np.ones(len(z))
        for i, date in enumerate(segment["date"]):
            zi = z[i]
            if not math.isfinite(zi):
                self._ep_side, self._ep_mult = 0, 1.0
            else:
                if self._ep_side and self._ep_side * zi <= 0:
                    self._ep_side, self._ep_mult = 0, 1.0
                if not self._ep_side and abs(zi) > ENTRY_Z:
                    self._ep_side = int(np.sign(zi))
                    self._ep_mult = self._multiplier(date)
                if self.mode == "stop" and self._ep_side and abs(zi) >= DIVERGE_Z:
                    self._ep_mult = 0.0
            ep_mult[i] = self._ep_mult if self._ep_side else 1.0
        target = target * ep_mult

        position = np.empty(len(target))
        current = self._position
        for i, (wanted, width) in enumerate(zip(target, band)):
            if abs(wanted - current) > width or (wanted == 0.0 and current != 0.0 and ep_mult[i] == 0):
                current = wanted
            position[i] = current
        self._position = current
        scale = 1.0 + abs(hedge.hedge_ratio)
        segment["episode_multiplier"] = ep_mult
        segment["signal"] = position
        segment["target_weight_a"] = position / scale
        segment["target_weight_b"] = -hedge.hedge_ratio * position / scale
        segment["target_gross_exposure"] = np.abs(position)
        return segment


# -------------------------------------------------------------------- main


def main() -> None:
    warnings.simplefilter("ignore")
    closes = pd.read_csv("data/universe_closes.csv", index_col="date", parse_dates=["date"])
    vix_pct = closes["^VIX"].ffill().rolling(504, min_periods=126).rank(pct=True)
    stocks = closes.drop(columns=["^VIX"])
    data = pd.read_csv("data/pit_pair_prices.csv", parse_dates=["date"])
    selection = pd.read_csv("data/pit_pair_selection.csv")
    params = pd.read_csv("backtest_results/vol_scaled_naive/selected_parameters_5bps.csv")

    # training pool: all same-sector pairs (canonical orientation)
    pool = [(s, a, b) for s, ts in UNIVERSE.items() for i, a in enumerate(sorted(ts))
            for b in sorted(ts)[i + 1:] if a in stocks and b in stocks]
    print(f"Building features and episodes for {len(pool)} pool pairs ...", flush=True)
    tables, eps = {}, []
    for _, a, b in pool:
        tables[(a, b)] = features_for(stocks, vix_pct, a, b)
        e = episodes(tables[(a, b)])
        if not e.empty:
            eps.append(e.assign(pair=f"{a}_{b}"))
    for a, b in data[["ticker_a", "ticker_b"]].drop_duplicates().itertuples(index=False):
        if (a, b) not in tables:  # traded orientation differs from canonical
            tables[(a, b)] = features_for(stocks, vix_pct, a, b)
    eps = pd.concat(eps, ignore_index=True)
    eps = eps[eps["start"] >= "2006-01-01"]  # feature warm-up

    rets = np.log(stocks).diff()
    clfs, oos = {}, []
    for year in range(2014, 2024):
        cut = pd.Timestamp(year=year, month=1, day=1)
        window = rets.loc[(rets.index >= cut - pd.DateOffset(years=10)) & (rets.index < cut)]
        eligible = {f"{a}_{b}" for _, a, b in pool
                    if window[a].notna().mean() > 0.95 and window[b].notna().mean() > 0.95
                    and window[a].corr(window[b]) >= 0.6}
        in_pool = eps["pair"].isin(eligible)
        train = eps[in_pool & (eps["resolved"] < cut)]
        test = eps[in_pool & (eps["start"].dt.year == year)]
        clfs[year] = {name: DivergenceClassifier(train, feats) for name, feats in
                      (("all", FEATURES), ("no_hedge_gap", [f for f in FEATURES if f != "hedge_gap"]))}
        for name, clf in clfs[year].items():
            if len(test):
                oos.append(test.assign(year=year, variant=name, p=clf.prob(test)))
        print(f"  {year}: pool pairs {len(eligible)}, train episodes {len(train)} "
              f"(diverge rate {train['label'].mean():.2f}), test episodes {len(test)}", flush=True)
    oos = pd.concat(oos, ignore_index=True)

    print("\nOut-of-sample classification (episodes starting in each test year):")
    for name, g in oos.groupby("variant"):
        top = g[g["p"] >= g.groupby("year")["p"].transform(lambda p: p.quantile(0.8))]
        print(f"  {name:13s} AUC {roc_auc_score(g['label'], g['p']):.3f} | diverge rate "
              f"overall {g['label'].mean():.2f}, riskiest 20% {top['label'].mean():.2f}")
    coefs = pd.DataFrame({y: c["all"].coef for y, c in clfs.items()}).T
    print("\nCoefficients (standardised, 'all' model):")
    print(coefs.round(2).to_string())

    print("\nBacktesting through backtest.py (5 bps) ...", flush=True)
    frames = []
    for year in range(2014, 2024):
        pairs = selection.loc[selection["test_year"] == year, "pair_id"].unique()
        p = params.loc[params["test_year"] == year].iloc[0]
        kw = {"vix_scale": p.vix_scale, "band": p.band, "band_vix_scale": p.band_vix_scale}
        c = clfs[year]
        models = {
            "vol_scaled_naive (BD)": lambda: VolScaledNaivePairsTradingModel(**kw),
            "+ stop at |z|>=3": lambda: DivergenceFilteredModel("stop", None, tables, **kw),
            "+ classifier gate": lambda: DivergenceFilteredModel("gate", c["all"], tables, **kw),
            "+ classifier soft": lambda: DivergenceFilteredModel("soft", c["all"], tables, **kw),
            "+ gate, no Kalman feature": lambda: DivergenceFilteredModel("gate", c["no_hedge_gap"], tables, **kw),
        }
        year_data = data.loc[data["pair_id"].isin(pairs)]
        for name, factory in models.items():
            frames.append(run_walk_forward_backtest(year_data, factory, start_year=year, end_year=year,
                                                    train_years=4, transaction_cost_bps=5.0,
                                                    model_name=name))
    daily = pd.concat(frames, ignore_index=True)
    summary = portfolio_metrics(daily)
    port = daily.groupby(["model_name", "date"])["strategy_return"].mean().unstack(0)
    summary["daily_cvar_5pct"] = [port[m].loc[port[m] <= port[m].quantile(0.05)].mean()
                                  for m in summary["model_name"]]
    summary["sharpe_ex_2020_2022"] = [
        (lambda r: r.mean() / r.std() * math.sqrt(252))(port[m][~port.index.year.isin([2020, 2022])])
        for m in summary["model_name"]]
    yearly = port.groupby(port.index.year).apply(lambda x: (1 + x).prod() - 1)

    OUT.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUT / "summary_5bps.csv", index=False)
    yearly.to_csv(OUT / "yearly_5bps.csv")
    oos.to_csv(OUT / "oos_episode_predictions.csv", index=False)
    coefs.to_csv(OUT / "coefficients.csv")
    with pd.option_context("display.width", 200, "display.precision", 4):
        print(summary.to_string(index=False))
        print("\nYearly returns (%):")
        print((yearly * 100).round(2).to_string())


if __name__ == "__main__":
    main()
