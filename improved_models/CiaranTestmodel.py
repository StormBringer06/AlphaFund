"""CiaranTestmodel: point-in-time pairs strategy with ten improvements over the
BD-dev entry-state model and yearly screen.

No look-ahead rule
------------------
Everything decided for test year Y (screen, pair ranking, trade classifier,
entry/stop thresholds) uses only data dated before 1 January Y. Daily signals
use data up to that day's close and, by default, are executed one trading day
later. Every rolling/EWMA statistic is trailing. ``--leakage-test`` checks the
whole pipeline end to end by rerunning it on data truncated at a cutoff and
checking that every position up to the cutoff is unchanged.

Improvements (numbering follows the review)
-------------------------------------------
 1. Screen metrics on the spread the model trades (EW-RLS hedge): its half-life
    and an in-window simulated Sharpe, plus last-two-year cointegration.
 2. Pair ranking learned from how well each screen metric predicted the next
    year's pair performance in earlier years (Spearman IC, expanding window).
 3. Up to 10 pairs (max 2 per sector) instead of 5.
 4. Canonical (alphabetical) pair orientation with a symmetric cointegration
    test; positions carry across year ends; dropped pairs only wind down.
 5. Stop-loss on |z| and a time stop in multiples of the pair's half-life; a
    stopped pair is blocked for the rest of the year.
 6. Hedge ratio by exponentially weighted recursive least squares (a Kalman
    filter with exponential forgetting), updated daily - no 6-month jumps.
 7. z-score EWMA half-life tied to each pair's estimated half-life.
 8. Trade size from a logistic model of P(trade is profitable) fitted on
    earlier, fully resolved trades (embargoed): VIX percentile, spread-vol
    ratio, |z|, divergence speed, idiosyncratic jump (news proxy), market
    return and pair screen statistics.
 9. Inverse-volatility pair sizing, an ex-ante risk budget targeting 10%
    volatility when all pairs are fully sized (4x gross cap), and an SPY hedge
    of residual market beta.
10. Nested walk-forward choice of thresholds, 1-day execution lag, short
    borrow cost, 2008 start, untouched 2024-2025 holdout, year bootstrap,
    deflated Sharpe ratio and the truncation leakage test.

Run from the repository root::

    python -m improved_models.CiaranTestmodel                 # 2008-2023 dev run
    python -m improved_models.CiaranTestmodel --all-variants --baselines  # + ablations, BD
    python -m improved_models.CiaranTestmodel --leakage-test  # look-ahead check
    python -m improved_models.CiaranTestmodel --holdout       # 2024-2025, once
"""

from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")  # one BLAS thread per process; we parallelise by process

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field, replace
from itertools import combinations, product
import math
from pathlib import Path
import time
import warnings

import numpy as np
import pandas as pd
from scipy.stats import norm

from screen_pairs import UNIVERSE, download_closes


VIX, SPY = "^VIX", "SPY"
DATA_PATH = Path("data") / "ciaran_universe_closes.csv"
OUTPUT_DIR = Path("backtest_results") / "ciaran_test_model"
DATA_START = pd.Timestamp("1993-01-01")
Z_HALFLIVES = np.array([20, 30, 40, 60, 90, 120, 180, 250], dtype=float)
ANN = math.sqrt(252.0)
INF = math.inf

DEFAULT_PARAMS = {"entry_z": 1.0, "stop_z": INF, "time_mult": INF}
PARAM_GRID = [
    {"entry_z": e, "stop_z": s, "time_mult": m}
    for e, s, m in product((1.0, 1.5), (INF, 3.5), (INF, 3.0))
]
RANK_METRICS = (
    "return_correlation",
    "log_full_p",
    "window_coint_rate",
    "log_recent_p",
    "half_life",
    "model_half_life",
    "insample_sharpe",
    "beta_stability",
)
CLF_FEATURES = (
    "vix_pct",
    "log_vol_ratio",
    "abs_z",
    "div_speed",
    "jump5",
    "spy_ret20",
    "half_life",
    "window_coint_rate",
    "log_recent_p",
)


@dataclass(frozen=True)
class Config:
    # screening (BD filters kept; ranking and pair count changed)
    screen_years: int = 10
    min_corr: float = 0.6
    max_full_p: float = 0.05
    min_window_beta: float = 0.2
    half_life_range: tuple[float, float] = (5.0, 75.0)
    n_pairs: int = 10
    max_per_sector: int = 2
    learned_ranking: bool = True
    ic_min_years: int = 3
    # spread and signal
    hedge: str = "rls"  # "rls" (daily EW least squares) or "ols6m" (BD style)
    hedge_halflife: float = 252.0
    adaptive_z: bool = True
    z_halflife_mult: float = 3.0
    fixed_z_halflife: float = 120.0
    full_z: float = 2.0
    band: float = 0.2
    use_stops: bool = True
    nested: bool = True
    nested_years: int = 3
    classifier: bool = True
    clf_min_trades: int = 60
    continuity: bool = True
    # sizing
    risk_parity: bool = True
    pair_vol_ref: float = 0.005
    vol_target: float = 0.10
    max_gross: float = 4.0
    beta_hedge: bool = True
    # frictions
    lag: int = 1
    cost_bps: float = 5.0
    borrow_bps: float = 50.0


# --------------------------------------------------------------------- data


def load_closes(path: Path = DATA_PATH, end: str | pd.Timestamp | None = None) -> pd.DataFrame:
    """Adjusted closes on the SPY (NYSE) calendar; VIX forward-filled."""
    if not path.exists():
        tickers = sorted({t for g in UNIVERSE.values() for t in g}) + [VIX, SPY]
        closes = download_closes(tickers, DATA_START, pd.Timestamp("2026-01-01"))
        path.parent.mkdir(parents=True, exist_ok=True)
        closes.to_csv(path, date_format="%Y-%m-%d")
    closes = pd.read_csv(path, index_col="date", parse_dates=["date"]).sort_index()
    closes = closes.loc[closes[SPY].notna()].copy()
    closes[VIX] = closes[VIX].ffill()  # trailing fill only
    if end is not None:
        closes = closes.loc[closes.index <= pd.Timestamp(end)]
    return closes


class Market:
    """Calendar-aligned market arrays (all trailing statistics)."""

    def __init__(self, closes: pd.DataFrame) -> None:
        self.closes = closes
        self.dates = closes.index
        self.stocks = [c for c in closes.columns if c not in (VIX, SPY)]
        self.tickers = self.stocks + [SPY]
        self.col = {t: i for i, t in enumerate(self.tickers)}
        returns = closes[self.tickers].pct_change(fill_method=None)
        self.R = returns.fillna(0.0).to_numpy()
        self.vix_pct = (
            closes[VIX].rolling(504, min_periods=126).rank(pct=True).to_numpy()
        )
        self.spy_ret20 = closes[SPY].pct_change(20, fill_method=None).to_numpy()
        spy = returns[SPY]
        spy_var = spy.ewm(halflife=126, min_periods=63).var()
        self.stock_beta = pd.DataFrame(
            {s: returns[s].ewm(halflife=126, min_periods=63).cov(spy) / spy_var
             for s in self.stocks}
        ).to_numpy()
        years = self.dates.year.to_numpy()
        self.year_first = {y: int(np.flatnonzero(years == y)[0]) for y in np.unique(years)}
        self.year_last = {y: int(np.flatnonzero(years == y)[-1]) for y in np.unique(years)}


def _ewm_z(spread: pd.Series, halflife: float) -> pd.Series:
    ewm = spread.ewm(halflife=halflife, min_periods=int(halflife))
    mean, std = ewm.mean().shift(1), ewm.std().shift(1)
    return (spread - mean) / std.where(std > 0)


def pair_features(
    closes: pd.DataFrame, a: str, b: str, hedge: str = "rls", hedge_halflife: float = 252.0
) -> pd.DataFrame:
    """Causal per-pair series on the closes calendar (NaN where either is missing).

    ``beta`` is the hedge known at close t (used for weights traded then);
    ``spread`` at t uses the hedge known at t-1, so it is a genuine forecast error.
    """
    px = closes[[a, b]].dropna()
    la, lb = np.log(px[a]), np.log(px[b])
    ra, rb = px[a].pct_change(), px[b].pct_change()
    out = pd.DataFrame(index=px.index)

    if hedge == "rls":
        ew = {"halflife": hedge_halflife, "min_periods": 126}
        beta = la.ewm(**ew).cov(lb) / lb.ewm(**ew).var()
        alpha = la.ewm(**ew).mean() - beta * lb.ewm(**ew).mean()
        spread = la - (alpha.shift(1) + beta.shift(1) * lb)
        prior_beta = beta.shift(1)
        for h in Z_HALFLIVES:
            out[f"z{int(h)}"] = _ewm_z(spread, h)
    elif hedge == "ols6m":
        # BD style: OLS on the prior 2 years, refit 1 Jan / 1 Jul from data before
        # the refit date; the spread history is recomputed with each new hedge.
        beta = pd.Series(np.nan, index=px.index)
        spread = pd.Series(np.nan, index=px.index)
        zs = {h: pd.Series(np.nan, index=px.index) for h in Z_HALFLIVES}
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
            for h in Z_HALFLIVES:
                zs[h][seg] = _ewm_z(s_full, h)[seg[upto]]
        prior_beta = beta
        for h in Z_HALFLIVES:
            out[f"z{int(h)}"] = zs[h]
    else:
        raise ValueError(f"Unknown hedge {hedge!r}.")

    hedged = (ra - prior_beta * rb) / (1.0 + prior_beta.abs())
    change = spread.diff()
    resid = ra - prior_beta * rb
    resid_sd = resid.ewm(halflife=60, min_periods=40).std().shift(1)
    out["beta"] = beta
    out["spread"] = spread
    out["sigma"] = hedged.ewm(halflife=63, min_periods=40).std()
    out["vol_ratio"] = change.rolling(20).std() / change.rolling(120, min_periods=60).std()
    out["jump5"] = ((resid.abs() / resid_sd) > 3.0).astype(float).rolling(5, min_periods=1).max()
    return out.reindex(closes.index)


class Context:
    """Market plus a per-process cache of pair feature arrays."""

    def __init__(self, closes: pd.DataFrame) -> None:
        self.market = Market(closes)
        self._cache: dict[tuple, dict[str, np.ndarray]] = {}

    def features(self, a: str, b: str, hedge: str = "rls", halflife: float = 252.0) -> dict:
        key = (a, b, hedge, halflife)
        if key not in self._cache:
            frame = pair_features(self.market.closes, a, b, hedge, halflife)
            self._cache[key] = {c: frame[c].to_numpy() for c in frame.columns}
        return self._cache[key]


# ------------------------------------------------------------------ screen


def _hedge_ratio(la: pd.Series, lb: pd.Series) -> float:
    return float(lb.cov(la) / lb.var())


def half_life(spread: pd.Series) -> float:
    s = spread.dropna()
    if len(s) < 30:
        return INF
    slope = float(np.polyfit(s.shift().dropna().to_numpy(), s.diff().dropna().to_numpy(), 1)[0])
    return -math.log(2) / slope if slope < 0 else INF


def _sym_p(la: pd.Series, lb: pd.Series) -> float:
    """Engle-Granger p-value, minimised over both orientations (symmetric)."""
    from statsmodels.tsa.stattools import coint

    return min(float(coint(la, lb)[1]), float(coint(lb, la)[1]))


def screen_year(window: pd.DataFrame, year: int, cfg: Config) -> pd.DataFrame:
    """Score every same-sector pair on ``window`` (the years before ``year``)."""
    end = pd.Timestamp(year=year, month=1, day=1)
    rows = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for sector, group in UNIVERSE.items():
            tickers = sorted(t for t in group if t in window and window[t].notna().any())
            for a, b in combinations(tickers, 2):  # a < b: canonical orientation
                px = window[[a, b]].dropna()
                if len(px) < 0.95 * len(window):
                    continue
                la, lb = np.log(px[a]), np.log(px[b])
                row = {
                    "test_year": year, "pair_id": f"{a}_{b}".lower(), "sector": sector,
                    "ticker_a": a, "ticker_b": b,
                    "return_correlation": float(la.diff().corr(lb.diff())),
                }
                if row["return_correlation"] < cfg.min_corr:
                    rows.append(row)
                    continue
                p_w, b_w = [], []
                start, last = la.index[0], la.index[-1] - pd.DateOffset(years=2)
                while start <= last:
                    stop = start + pd.DateOffset(years=2)
                    p_w.append(_sym_p(la.loc[start:stop], lb.loc[start:stop]))
                    b_w.append(_hedge_ratio(la.loc[start:stop], lb.loc[start:stop]))
                    start += pd.DateOffset(months=6)
                recent = la.index >= end - pd.DateOffset(years=2)
                beta = _hedge_ratio(la, lb)
                row.update(
                    full_p=_sym_p(la, lb),
                    hedge_ratio=beta,
                    min_window_beta=min(b_w),
                    max_window_beta=max(b_w),
                    window_coint_rate=float(np.mean(np.array(p_w) < 0.05)),
                    recent_p=_sym_p(la[recent], lb[recent]),
                    half_life=half_life(la - beta * lb),
                )
                rows.append(row)
    return apply_filters(pd.DataFrame(rows), cfg)


def apply_filters(screens: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    out = screens.copy()
    for col in ("full_p", "recent_p", "min_window_beta", "max_window_beta",
                "window_coint_rate", "half_life", "hedge_ratio"):
        if col not in out:
            out[col] = np.nan
    low, high = cfg.half_life_range
    out["passes"] = (
        (out["return_correlation"] >= cfg.min_corr)
        & (out["full_p"] <= cfg.max_full_p)
        & (out["min_window_beta"] >= cfg.min_window_beta)
        & out["half_life"].between(low, high)
    )
    return out


def _screen_task(args: tuple) -> pd.DataFrame:
    window, year, cfg = args
    return screen_year(window, year, cfg)


def _bd_screen_task(args: tuple) -> pd.DataFrame:
    from screen_pairs import screen, select_pairs

    window, year, count = args
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        selected = select_pairs(screen(window), count)
    selected.insert(0, "test_year", year)
    selected.insert(1, "pair_id", [f"{a}_{b}".lower() for a, b in
                                   zip(selected["ticker_a"], selected["ticker_b"])])
    return selected


def screen_window(closes: pd.DataFrame, year: int, years: int) -> pd.DataFrame:
    end = pd.Timestamp(year=year, month=1, day=1)
    start = end - pd.DateOffset(years=years)
    stocks = closes.drop(columns=[VIX, SPY])
    return stocks.loc[(stocks.index >= start) & (stocks.index < end)]


def compute_screens(closes: pd.DataFrame, years: list[int], cfg: Config, workers: int) -> pd.DataFrame:
    tasks = [(screen_window(closes, y, cfg.screen_years), y, cfg) for y in years]
    with ProcessPoolExecutor(workers) as pool:
        frames = list(pool.map(_screen_task, tasks))
    return pd.concat(frames, ignore_index=True)


def add_model_metrics(screens: pd.DataFrame, ctx: Context) -> pd.DataFrame:
    """Improvement 1: statistics of the spread the model actually trades,
    measured on the last two years of each screening window (all before Y)."""
    out = screens.copy()
    out["model_half_life"] = np.nan
    out["insample_sharpe"] = np.nan
    m = ctx.market
    for i, row in out.loc[out["passes"]].iterrows():
        arr = ctx.features(row.ticker_a, row.ticker_b)
        end = pd.Timestamp(year=row.test_year, month=1, day=1)
        idx = np.flatnonzero((m.dates >= end - pd.DateOffset(years=2)) & (m.dates < end))
        if len(idx) < 100:
            continue
        out.at[i, "model_half_life"] = half_life(pd.Series(arr["spread"][idx]))
        ret, _ = simulate_pair(arr, m.R[:, m.col[row.ticker_a]], m.R[:, m.col[row.ticker_b]],
                               arr["z120"], idx[0], idx[-1], idx[-1])
        out.at[i, "insample_sharpe"] = _sharpe(ret)
    out["log_full_p"] = np.log10(out["full_p"].clip(lower=1e-12))
    out["log_recent_p"] = np.log10(out["recent_p"].clip(lower=1e-12))
    out["beta_stability"] = out["min_window_beta"] / out["max_window_beta"]
    return out


# ---------------------------------------------------------- single-pair sim


def _sharpe(r: np.ndarray) -> float:
    sd = float(np.std(r, ddof=1)) if len(r) > 1 else 0.0
    return float(np.mean(r) / sd * ANN) if sd > 0 else 0.0


def simulate_pair(
    arr: dict, ra: np.ndarray, rb: np.ndarray, z: np.ndarray,
    t0: int, t_entry_end: int, t_end: int,
    entry_z: float = 1.0, full_z: float = 2.0, band: float = 0.2, cost: float = 5e-4,
) -> tuple[np.ndarray, list[dict]]:
    """Standalone unit-gross sim (no stops, lag 0) used for screen metrics, the
    ranking target and classifier labels. New trades only up to ``t_entry_end``;
    open trades run until they revert or ``t_end``."""
    beta = arr["beta"]
    ret = np.zeros(t_end - t0 + 1)
    side, held, wa, wb = 0, 0.0, 0.0, 0.0
    trades: list[dict] = []
    cur: dict | None = None
    for t in range(t0, t_end + 1):
        k = t - t0
        if k:
            r = wa * ra[t] + wb * rb[t]
            ret[k] += r
            if cur is not None:
                cur["pnl"] += r
        zt = z[t]
        if side:
            reason = None
            if not math.isfinite(zt):
                reason = "no_signal"
            elif side * zt >= 0:
                reason = "revert"
            elif t == t_end:
                reason = "limit"
            if reason:
                c = cost * (abs(wa) + abs(wb))
                ret[k] -= c
                cur["pnl"] -= c
                cur.update(exit_t=t, reason=reason)
                trades.append(cur)
                cur, side, held, wa, wb = None, 0, 0.0, 0.0, 0.0
        if (not side and t <= t_entry_end and t < t_end and math.isfinite(zt)
                and abs(zt) > entry_z and math.isfinite(beta[t])):
            side = -1 if zt > 0 else 1
            cur = {"entry_t": t, "side": side, "pnl": 0.0}
        if side:
            target = side * min(abs(zt) / full_z, 1.0)
            if abs(target - held) > band:
                b = beta[t]
                nwa, nwb = target / (1 + abs(b)), -b * target / (1 + abs(b))
                c = cost * (abs(nwa - wa) + abs(nwb - wb))
                ret[k] -= c
                cur["pnl"] -= c
                wa, wb, held = nwa, nwb, target
        elif t > t_entry_end:
            return ret[: k + 1], trades
    return ret, trades


def entry_features(arr: dict, z: np.ndarray, t: int, m: Market, srow) -> dict:
    zt = z[t]
    z5 = z[t - 5] if t >= 5 else np.nan
    vr = arr["vol_ratio"][t]
    return {
        "vix_pct": m.vix_pct[t],
        "log_vol_ratio": math.log(min(max(vr, 1e-3), 1e3)) if math.isfinite(vr) else np.nan,
        "abs_z": abs(zt),
        "div_speed": abs(zt - z5),
        "jump5": arr["jump5"][t],
        "spy_ret20": m.spy_ret20[t],
        "half_life": srow.half_life,
        "window_coint_rate": srow.window_coint_rate,
        "log_recent_p": srow.log_recent_p,
    }


def z_key(cfg: Config, hl: float) -> str:
    if not cfg.adaptive_z:
        return f"z{int(cfg.fixed_z_halflife)}"
    target = float(np.clip(cfg.z_halflife_mult * hl, 20, 250))
    return f"z{int(Z_HALFLIVES[np.argmin(np.abs(np.log(Z_HALFLIVES / target)))])}"


def candidate_sims(cfg: Config, ctx: Context, screens: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per passing candidate and year: that year's standalone Sharpe (ranking
    target, known at year end) and its trades with entry features and exit date
    (classifier data, used only once the exit date is in the past)."""
    m = ctx.market
    perf, trade_rows = [], []
    for row in screens.loc[screens["passes"]].itertuples():
        if row.test_year not in m.year_first:
            continue
        arr = ctx.features(row.ticker_a, row.ticker_b, cfg.hedge, cfg.hedge_halflife)
        z = arr[z_key(cfg, row.half_life)]
        t0, t1 = m.year_first[row.test_year], m.year_last[row.test_year]
        t_end = m.year_last.get(row.test_year + 1, len(m.dates) - 1)
        ret, trades = simulate_pair(arr, m.R[:, m.col[row.ticker_a]], m.R[:, m.col[row.ticker_b]],
                                    z, t0, t1, t_end)
        perf.append({"test_year": row.test_year, "pair_id": row.pair_id,
                     "pair_sharpe": _sharpe(ret[: t1 - t0 + 1])})
        for tr in trades:
            trade_rows.append({
                "test_year": row.test_year, "pair_id": row.pair_id,
                "entry_date": m.dates[tr["entry_t"]], "exit_date": m.dates[tr["exit_t"]],
                "pnl": tr["pnl"], "reason": tr["reason"],
                **entry_features(arr, z, tr["entry_t"], m, row),
            })
    return pd.DataFrame(perf), pd.DataFrame(trade_rows)


# ------------------------------------------------- ranking and classifier


def rank_and_select(cfg: Config, screens: pd.DataFrame, perf: pd.DataFrame, year: int) -> tuple[pd.DataFrame, dict]:
    """Improvement 2: weight screen metrics by their past information coefficient."""
    cand = screens.loc[(screens["test_year"] == year) & screens["passes"]].copy()
    weights: dict = {}
    if cfg.learned_ranking and not perf.empty:
        ics = []
        for y in sorted(perf["test_year"].unique()):
            if y >= year:  # year y's performance is only known after y ends
                continue
            joined = screens.loc[(screens["test_year"] == y) & screens["passes"]].merge(
                perf.loc[perf["test_year"] == y], on=["test_year", "pair_id"])
            if len(joined) >= 8:
                ics.append({mname: joined[mname].corr(joined["pair_sharpe"], method="spearman")
                            for mname in RANK_METRICS})
        if len(ics) >= cfg.ic_min_years:
            mean_ic = pd.DataFrame(ics).mean()
            if mean_ic.abs().sum() > 0:
                weights = mean_ic.fillna(0.0).to_dict()
    if weights and not cand.empty:
        score = sum(w * cand[mname].rank(pct=True).fillna(0.5) for mname, w in weights.items())
        cand = cand.assign(score=score).sort_values("score", ascending=False)
    else:
        cand = cand.sort_values(["window_coint_rate", "full_p"], ascending=[False, True])
    chosen, used, per_sector = [], set(), {}
    for idx, row in cand.iterrows():
        if row.ticker_a in used or row.ticker_b in used:
            continue
        if per_sector.get(row.sector, 0) >= cfg.max_per_sector:
            continue
        chosen.append(idx)
        used.update((row.ticker_a, row.ticker_b))
        per_sector[row.sector] = per_sector.get(row.sector, 0) + 1
        if len(chosen) == cfg.n_pairs:
            break
    return cand.loc[chosen].reset_index(drop=True), weights


class TradeClassifier:
    """Improvement 8: logistic P(win) -> expected edge -> size multiplier in [0, 1]."""

    def __init__(self, trades: pd.DataFrame) -> None:
        from sklearn.linear_model import LogisticRegression

        x = trades[list(CLF_FEATURES)].to_numpy(dtype=float)
        y = (trades["pnl"] > 0).to_numpy()
        self.mu = np.nanmean(x, axis=0)
        self.sd = np.nanstd(x, axis=0)
        self.sd[~(self.sd > 0)] = 1.0
        xs = self._standardise(x)
        self.model = LogisticRegression(C=1.0, max_iter=2000).fit(xs, y)
        self.win = float(trades.loc[y, "pnl"].mean())
        self.loss = float(-trades.loc[~y, "pnl"].mean())
        self.ref = self._edge(float(np.median(self.model.predict_proba(xs)[:, 1])))
        self.n = len(trades)
        self.coef = dict(zip(CLF_FEATURES, self.model.coef_[0]))

    def _standardise(self, x: np.ndarray) -> np.ndarray:
        x = np.where(np.isfinite(x), x, self.mu)
        return (x - self.mu) / self.sd

    def _edge(self, p: float) -> float:
        return p * self.win - (1 - p) * self.loss

    def multiplier(self, feats: dict) -> float:
        x = self._standardise(np.array([[feats[f] for f in CLF_FEATURES]], dtype=float))
        edge = self._edge(float(self.model.predict_proba(x)[0, 1]))
        if self.ref <= 0:
            return float(edge > 0)
        return float(np.clip(edge / self.ref, 0.0, 1.0))


def build_classifier(cfg: Config, trades: pd.DataFrame, year: int) -> TradeClassifier | None:
    if not cfg.classifier or trades.empty:
        return None
    # Embargo: only trades whose outcome was known before 1 January of ``year``.
    train = trades.loc[trades["exit_date"] < pd.Timestamp(year=year, month=1, day=1)]
    wins = int((train["pnl"] > 0).sum())
    if len(train) < cfg.clf_min_trades or wins < 10 or len(train) - wins < 10:
        return None
    return TradeClassifier(train)


# ---------------------------------------------------------- portfolio sim


@dataclass
class YearPlan:
    year: int
    pairs: pd.DataFrame
    params: dict = field(default_factory=lambda: dict(DEFAULT_PARAMS))
    classifier: TradeClassifier | None = None
    weights: dict = field(default_factory=dict)


class _PairState:
    __slots__ = ("pid", "ia", "ib", "arr", "z", "hl", "row", "active", "blocked",
                 "side", "mult", "entry_t", "entry_z", "entry_year", "held", "w")

    def __init__(self, pid, ia, ib, arr):
        self.pid, self.ia, self.ib, self.arr = pid, ia, ib, arr
        self.active = self.blocked = False
        self.side, self.mult, self.entry_t, self.entry_z, self.entry_year = 0, 1.0, -1, 0.0, 0
        self.held, self.w = 0.0, (0.0, 0.0)
        self.z = self.hl = self.row = None


def run_portfolio(ctx: Context, cfg: Config, plans: dict[int, YearPlan], y0: int, y1: int,
                  vol_target: bool = True) -> dict:
    """Daily portfolio simulation. Targets are set at close t from data <= t;
    positions are the targets ``cfg.lag`` days earlier; returns from t-1 to t
    accrue to positions held at t-1."""
    m = ctx.market
    T, n_cols, spy_col = len(m.dates), len(m.tickers), m.col[SPY]
    t_idx = np.flatnonzero((m.dates.year >= y0) & (m.dates.year <= y1))
    lag = cfg.lag
    U = np.zeros((T, n_cols))  # unlevered targets
    lev = np.zeros(T)
    pair_w: dict[str, np.ndarray] = {}
    states: dict[str, _PairState] = {}
    trades: list[dict] = []
    lev_year, spy_w = 1.0, 0.0
    year, params, clf, n_sel = None, DEFAULT_PARAMS, None, 1
    use_vt = vol_target and cfg.vol_target > 0

    for pos, t in enumerate(t_idx):
        d = m.dates[t]
        if d.year != year:
            year = d.year
            plan = plans[year]
            params, clf = plan.params, plan.classifier
            n_sel = max(len(plan.pairs), 1)
            # Ex-ante risk budget: a fully sized pair is scaled to daily vol
            # pair_vol_ref / n_sel, so n_sel uncorrelated full-size pairs give
            # pair_vol_ref / sqrt(n_sel); lever that up to the target.
            if use_vt:
                lev_year = (cfg.vol_target / ANN) / (cfg.pair_vol_ref / math.sqrt(n_sel))
            for st in states.values():
                st.active, st.blocked = False, False
            for row in plan.pairs.itertuples():
                st = states.get(row.pair_id)
                if st is None:
                    arr = ctx.features(row.ticker_a, row.ticker_b, cfg.hedge, cfg.hedge_halflife)
                    st = _PairState(row.pair_id, m.col[row.ticker_a], m.col[row.ticker_b], arr)
                    states[row.pair_id] = st
                    pair_w[row.pair_id] = np.zeros((T, 2))
                st.active, st.row, st.hl = True, row, row.half_life
                st.z = st.arr[z_key(cfg, row.half_life)]
        month_start = pos == 0 or m.dates[t_idx[pos - 1]].month != d.month
        last_day = pos == len(t_idx) - 1 or m.dates[t_idx[pos + 1]].year != year
        force_flat = last_day and not cfg.continuity
        any_rebalance = False

        for pid, st in states.items():
            if not st.active and st.side == 0:
                continue
            z = st.z[t]
            if st.side:
                reason = None
                if not math.isfinite(z):
                    reason = "no_signal"
                elif st.side * z >= 0:
                    reason = "revert"
                elif abs(z) >= params["stop_z"]:
                    reason = "stop"
                elif t - st.entry_t >= params["time_mult"] * st.hl:
                    reason = "time"
                elif force_flat:
                    reason = "year_end"
                if reason:
                    trades.append({"pair_id": pid, "entry_t": st.entry_t, "exit_t": t,
                                   "reason": reason, "side": st.side, "mult": st.mult,
                                   "entry_z": st.entry_z, "entry_year": st.entry_year})
                    st.side = 0
                    if reason in ("stop", "time"):
                        st.blocked = True
            if (not st.side and st.active and not st.blocked and not force_flat
                    and math.isfinite(z) and abs(z) > params["entry_z"]
                    and math.isfinite(st.arr["beta"][t]) and math.isfinite(st.arr["sigma"][t])):
                st.side = -1 if z > 0 else 1
                st.entry_t, st.entry_z, st.entry_year = t, z, year
                st.mult = clf.multiplier(entry_features(st.arr, st.z, t, m, st.row)) if clf else 1.0
            target = st.side * min(abs(z) / cfg.full_z, 1.0) * st.mult if st.side else 0.0
            if (st.side == 0 or abs(target - st.held) > cfg.band) and target != st.held:
                st.held = target
                b = st.arr["beta"][t]
                if st.held == 0.0 or not math.isfinite(b):
                    st.w = (0.0, 0.0)
                else:
                    k = 1.0 / n_sel
                    if cfg.risk_parity:
                        k *= min(cfg.pair_vol_ref / max(st.arr["sigma"][t], 0.002), 3.0)
                    st.w = (k * target / (1 + abs(b)), -b * k * target / (1 + abs(b)))
                any_rebalance = True
            U[t, st.ia] += st.w[0]
            U[t, st.ib] += st.w[1]
            pair_w[pid][t] = st.w

        if cfg.beta_hedge:
            if any_rebalance or month_start:
                betas = m.stock_beta[t]
                spy_w = -float(np.nansum(U[t, :spy_col] * betas))
            U[t, spy_col] = spy_w
        gross = float(np.abs(U[t]).sum())
        L = lev_year if use_vt else 1.0
        lev[t] = min(L, cfg.max_gross / gross) if gross > 0 else L

    # positions, returns, costs
    W = U[t_idx] * lev[t_idx, None]
    P = np.zeros_like(W)
    P[lag:] = W[: len(W) - lag] if lag else W
    prev = np.vstack([np.zeros((1, n_cols)), P[:-1]])
    R = m.R[t_idx]
    gross_ret = (prev * R).sum(axis=1)
    turnover = np.abs(P - prev).sum(axis=1)
    cost = turnover * cfg.cost_bps / 1e4
    borrow = np.clip(-prev, 0, None).sum(axis=1) * cfg.borrow_bps / 1e4 / 252
    daily = pd.DataFrame({
        "date": m.dates[t_idx], "gross_return": gross_ret, "cost": cost, "borrow": borrow,
        "net_return": gross_ret - cost - borrow, "gross_exposure": np.abs(P).sum(axis=1),
        "turnover": turnover, "leverage": lev[t_idx],
    })

    # per-pair attribution (gross of costs) and trade P&L
    pos_of = {t: i for i, t in enumerate(t_idx)}
    pair_pnl = {}
    for pid, w in pair_w.items():
        st = states[pid]
        lw = w[t_idx] * lev[t_idx, None]
        lp = np.zeros_like(lw)
        lp[lag:] = lw[: len(lw) - lag] if lag else lw
        lp_prev = np.vstack([np.zeros((1, 2)), lp[:-1]])
        pair_pnl[pid] = lp_prev[:, 0] * R[:, st.ia] + lp_prev[:, 1] * R[:, st.ib]
    for tr in trades:
        i0, i1 = pos_of[tr["entry_t"]] + 1 + lag, pos_of[tr["exit_t"]] + 1 + lag
        tr["pnl"] = float(pair_pnl[tr["pair_id"]][i0:i1 + 1].sum())
        tr["entry_date"], tr["exit_date"] = m.dates[tr["entry_t"]], m.dates[tr["exit_t"]]
        tr["days"] = tr["exit_t"] - tr["entry_t"]
    pnl = pd.DataFrame(pair_pnl, index=m.dates[t_idx]) if pair_pnl else pd.DataFrame()
    return {"daily": daily, "trades": pd.DataFrame(trades), "pair_pnl": pnl,
            "U": U, "lev": lev, "t_idx": t_idx}


def choose_params(cfg: Config, ctx: Context, plans: dict[int, YearPlan], year: int) -> dict:
    """Improvement 10: nested walk-forward over the ``nested_years`` before ``year``."""
    if not cfg.nested:
        return dict(DEFAULT_PARAMS)
    inner = [y for y in range(year - cfg.nested_years, year) if y in plans]
    if len(inner) < 2:
        return dict(DEFAULT_PARAMS)
    grid = PARAM_GRID if cfg.use_stops else [p for p in PARAM_GRID
                                             if p["stop_z"] == INF and p["time_mult"] == INF]
    best, best_sharpe = dict(DEFAULT_PARAMS), -INF
    for combo in grid:
        inner_plans = {y: replace(plans[y], params=combo, classifier=None) for y in inner}
        res = run_portfolio(ctx, cfg, inner_plans, inner[0], inner[-1], vol_target=False)
        s = _sharpe(res["daily"]["net_return"].to_numpy())
        if s > best_sharpe + 1e-12:
            best, best_sharpe = dict(combo), s
    return best


def run_config(cfg: Config, ctx: Context, screens: pd.DataFrame, y0: int, y1: int) -> dict:
    perf, cand_trades = candidate_sims(cfg, ctx, screens)
    plans: dict[int, YearPlan] = {}
    for year in range(y0, y1 + 1):
        pairs, weights = rank_and_select(cfg, screens, perf, year)
        plan = YearPlan(year, pairs, classifier=build_classifier(cfg, cand_trades, year),
                        weights=weights)
        plan.params = choose_params(cfg, ctx, plans, year)
        plans[year] = plan
    result = run_portfolio(ctx, cfg, plans, y0, y1)
    result["plans"] = plans
    return result


# ------------------------------------------------------------- evaluation


def perf_stats(r: pd.Series) -> dict:
    r = r.dropna()
    vol = float(r.std())
    wealth = (1 + r).cumprod()
    dd = wealth / wealth.cummax().clip(lower=1.0) - 1
    return {
        "ann_return": float(wealth.iloc[-1] ** (252 / len(r)) - 1),
        "ann_vol": vol * ANN,
        "sharpe": float(r.mean() / vol * ANN) if vol > 0 else np.nan,
        "max_drawdown": float(dd.min()),
    }


def year_bootstrap(r: pd.Series, n: int = 5000, seed: int = 0) -> dict:
    groups = [g.to_numpy() for _, g in r.groupby(r.index.year)]
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n):
        x = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        draws.append(x.mean() / x.std(ddof=1) * ANN)
    draws = np.array(draws)
    return {"sharpe_p05": float(np.quantile(draws, 0.05)),
            "sharpe_p95": float(np.quantile(draws, 0.95)),
            "prob_sharpe_gt_0": float(np.mean(draws > 0))}


def deflated_sharpe(r: pd.Series, trial_sharpes_daily: np.ndarray, n_trials: int) -> dict:
    """Bailey & Lopez de Prado (2014). Sharpe in daily units throughout."""
    x = r.dropna().to_numpy()
    sr = x.mean() / x.std(ddof=1)
    skew = float(pd.Series(x).skew())
    kurt = float(pd.Series(x).kurt()) + 3.0
    denom = math.sqrt(max(1 - skew * sr + (kurt - 1) / 4 * sr ** 2, 1e-12))
    gamma = 0.5772156649
    sd_trials = float(np.std(trial_sharpes_daily, ddof=1))
    sr0 = sd_trials * ((1 - gamma) * norm.ppf(1 - 1 / n_trials)
                       + gamma * norm.ppf(1 - 1 / (n_trials * math.e)))
    stat = lambda bench: float(norm.cdf((sr - bench) * math.sqrt(len(x) - 1) / denom))
    return {"psr_vs_0": stat(0.0), "dsr": stat(sr0), "dsr_benchmark_sharpe": sr0 * ANN}


def reprice_bd(daily: pd.DataFrame, lag: int, cost_bps: float, borrow_bps: float) -> pd.DataFrame:
    """Re-run BD target weights through our execution/friction conventions."""
    out = []
    for (model, _pid, _fold), g in daily.groupby(["model_name", "pair_id", "fold_id"], sort=False):
        g = g.sort_values("date")
        tw = g[["target_weight_a", "target_weight_b"]].to_numpy(dtype=float)
        P = np.zeros_like(tw)
        P[lag:] = tw[: len(tw) - lag] if lag else tw
        P[-1] = 0.0  # fold liquidation, as in BD's backtest
        prev = np.vstack([np.zeros((1, 2)), P[:-1]])
        rets = g[["return_a", "return_b"]].to_numpy(dtype=float)
        net = ((prev * rets).sum(1) - np.abs(P - prev).sum(1) * cost_bps / 1e4
               - np.clip(-prev, 0, None).sum(1) * borrow_bps / 1e4 / 252)
        out.append(pd.DataFrame({"date": g["date"].to_numpy(), "model_name": model, "net": net}))
    frame = pd.concat(out)
    return frame.groupby(["model_name", "date"])["net"].mean().unstack(0)


def bd_baselines(closes: pd.DataFrame, y0: int, y1: int, workers: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    from base_model import NaivePairsTradingModel
    from build_screened_dataset import make_pair_prices
    from improved_models.entry_state_model import EntryStatePairsTradingModel, run_by_year
    from improved_models.vol_scaled_naive_model import VolScaledNaivePairsTradingModel

    tasks = [(screen_window(closes, y, 10), y, 5) for y in range(y0, y1 + 1)]
    with ProcessPoolExecutor(workers) as pool:
        selection = pd.concat(list(pool.map(_bd_screen_task, tasks)), ignore_index=True)
    stocks = closes.drop(columns=[VIX, SPY])
    pairs = tuple((pid, f"{a} / {b}", a, b) for pid, a, b in
                  selection[["pair_id", "ticker_a", "ticker_b"]].drop_duplicates().itertuples(index=False))
    prices = make_pair_prices(stocks, closes[VIX].dropna(), pairs)
    models = {
        "bd_naive_zscore": NaivePairsTradingModel,
        "bd_ewma_z_band": lambda: VolScaledNaivePairsTradingModel(band=0.2),
        "bd_entry_state_calm_0.5": lambda: EntryStatePairsTradingModel(calm_size=0.5),
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        daily = run_by_year(prices, selection, models, y0, y1, 4, 5.0)
    return daily, selection


# -------------------------------------------------------------------- main


VARIANTS: dict[str, dict] = {
    "ciaran_full": {},
    "no_learned_ranking": {"learned_ranking": False},
    "five_pairs": {"n_pairs": 5},
    "no_continuity": {"continuity": False},
    "no_stops": {"use_stops": False},
    "ols6m_hedge": {"hedge": "ols6m"},
    "fixed_z120": {"adaptive_z": False},
    "no_classifier": {"classifier": False},
    "no_risk_parity": {"risk_parity": False},
    "no_beta_hedge": {"beta_hedge": False},
    "full_cost_10bps": {"cost_bps": 10.0},
    "full_bd_frictions": {"lag": 0, "borrow_bps": 0.0},
}

_WORKER: dict = {}


def _init_worker(closes: pd.DataFrame, screens: pd.DataFrame) -> None:
    _WORKER["ctx"] = Context(closes)
    _WORKER["screens"] = screens


def _variant_task(args: tuple) -> tuple[str, dict]:
    name, overrides, y0, y1 = args
    t = time.time()
    res = run_config(replace(Config(), **overrides), _WORKER["ctx"], _WORKER["screens"], y0, y1)
    res.pop("U"), res.pop("lev")
    print(f"  {name} done in {time.time() - t:.0f}s", flush=True)
    return name, res


def get_screens(closes: pd.DataFrame, years: list[int], cfg: Config, workers: int,
                cache: Path | None) -> pd.DataFrame:
    if cache is not None and cache.exists():
        screens = pd.read_csv(cache)
        if set(years) <= set(screens["test_year"]):
            screens = screens.loc[screens["test_year"].isin(years)].reset_index(drop=True)
            return apply_filters(screens, cfg)
    screens = compute_screens(closes, years, cfg, workers)
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        screens.to_csv(cache, index=False)
    return screens


def plans_table(plans: dict[int, YearPlan]) -> tuple[pd.DataFrame, pd.DataFrame]:
    sel, par = [], []
    for y, p in plans.items():
        for row in p.pairs.itertuples():
            sel.append({"test_year": y, "pair_id": row.pair_id, "sector": row.sector,
                        "half_life": row.half_life, "window_coint_rate": row.window_coint_rate,
                        "recent_p": row.recent_p, "score": getattr(row, "score", np.nan)})
        clf = p.classifier
        par.append({"test_year": y, **p.params, "ranking": "learned" if p.weights else "bd_fallback",
                    "classifier_trades": clf.n if clf else 0,
                    **{f"ic_{k}": v for k, v in p.weights.items()},
                    **({f"coef_{k}": v for k, v in clf.coef.items()} if clf else {})})
    return pd.DataFrame(sel), pd.DataFrame(par)


def leakage_test(closes: pd.DataFrame, cutoff: str, workers: int) -> bool:
    """Rerun everything on data truncated at ``cutoff``; positions up to the
    cutoff must equal those from the full-data run."""
    cfg = Config()
    cut = pd.Timestamp(cutoff)
    y1 = cut.year
    years = list(range(2003, y1 + 1))
    print(f"Leakage test: full data vs data truncated at {cutoff} ...", flush=True)
    results = {}
    for label, data in (("full", closes), ("truncated", closes.loc[closes.index <= cut])):
        screens = compute_screens(data, years, cfg, workers)
        ctx = Context(data)
        screens = add_model_metrics(screens, ctx)
        res = run_config(cfg, ctx, screens, 2005, y1)
        keep = res["t_idx"][ctx.market.dates[res["t_idx"]] <= cut]
        results[label] = {
            "W": res["U"][keep] * res["lev"][keep, None],
            "dates": ctx.market.dates[keep],
            "screens": screens.loc[screens["test_year"] <= y1],
            "plans": res["plans"],
        }
    full, trunc = results["full"], results["truncated"]
    same_dates = full["dates"].equals(trunc["dates"])
    max_diff = float(np.max(np.abs(full["W"] - trunc["W"]))) if same_dates else INF
    cols = ["test_year", "pair_id", "passes", "full_p", "recent_p", "insample_sharpe"]
    screens_equal = full["screens"][cols].reset_index(drop=True).equals(
        trunc["screens"][cols].reset_index(drop=True))
    plans_equal = all(
        full["plans"][y].pairs["pair_id"].tolist() == trunc["plans"][y].pairs["pair_id"].tolist()
        and full["plans"][y].params == trunc["plans"][y].params
        for y in range(2005, y1 + 1)
    )
    ok = same_dates and max_diff <= 1e-12 and screens_equal and plans_equal
    lines = [
        f"cutoff: {cutoff}",
        f"dates compared: {len(full['dates'])} ({full['dates'][0].date()} .. {full['dates'][-1].date()})",
        f"screens identical: {screens_equal}",
        f"pair selections and thresholds identical: {plans_equal}",
        f"max |position difference|: {max_diff:.3e}",
        f"PASS: {ok}",
    ]
    print("\n".join(lines))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "leakage_test.txt").write_text("\n".join(lines) + "\n")
    return ok


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--start-year", type=int, default=2008)
    parser.add_argument("--holdout", action="store_true",
                        help="Extend to 2024-2025 (run once, after the dev results are final).")
    parser.add_argument("--leakage-test", action="store_true")
    parser.add_argument("--all-variants", action="store_true",
                        help="Also run the ablation / friction variants.")
    parser.add_argument("--baselines", action="store_true",
                        help="Also re-run BD's models on the same years for comparison.")
    parser.add_argument("--cutoff", default="2015-06-30")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    closes = load_closes()
    if args.leakage_test:
        raise SystemExit(0 if leakage_test(closes, args.cutoff, args.workers) else 1)

    y1 = 2025 if args.holdout else 2023
    burn_in = 2005  # simulation starts here; results reported from --start-year
    cfg = Config()
    t = time.time()
    print("Screening universe year by year ...", flush=True)
    screens = get_screens(closes, list(range(2003, y1 + 1)), cfg, args.workers,
                          OUTPUT_DIR / "screens_raw.csv")
    screens = add_model_metrics(screens, Context(closes))
    print(f"  screens ready ({time.time() - t:.0f}s); running variants ...", flush=True)

    variants = VARIANTS if args.all_variants else {"ciaran_full": {}}
    tasks = [(name, ov, burn_in, y1) for name, ov in variants.items()]
    if len(tasks) == 1:
        _init_worker(closes, screens)
        results = dict([_variant_task(tasks[0])])
    else:
        with ProcessPoolExecutor(args.workers, initializer=_init_worker,
                                 initargs=(closes, screens)) as pool:
            results = dict(pool.map(_variant_task, tasks))

    returns = pd.DataFrame({n: r["daily"].set_index("date")["net_return"] for n, r in results.items()})
    returns = returns.loc[returns.index.year >= args.start_year]
    if args.baselines:
        print("Running BD baselines on the same years ...", flush=True)
        bd_daily, bd_selection = bd_baselines(closes, args.start_year, y1, args.workers)
        bd_native = bd_daily.groupby(["model_name", "date"])["strategy_return"].mean().unstack(0)
        bd_strict = reprice_bd(bd_daily, lag=1, cost_bps=5.0, borrow_bps=50.0)
        for col in bd_native:
            returns[f"{col} (BD frictions)"] = bd_native[col]
            returns[f"{col} (strict frictions)"] = bd_strict[col]
        bd_selection.to_csv(OUTPUT_DIR / "bd_selection_rerun.csv", index=False)

    periods = {f"dev_{args.start_year}_2023": (args.start_year, 2023), "dev_2014_2023": (2014, 2023)}
    if args.holdout:
        periods["holdout_2024_2025"] = (2024, 2025)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for label, (a, b) in periods.items():
        sub = returns.loc[(returns.index.year >= a) & (returns.index.year <= b)]
        trial = np.array([sub[c].mean() / sub[c].std() for c in variants])
        rows = []
        for col in sub.columns:
            r = sub[col].dropna()
            row = {"model": col, **perf_stats(r)}
            if col in results:
                d = results[col]["daily"].set_index("date").loc[r.index]
                row.update(avg_gross=d["gross_exposure"].mean(), avg_daily_turnover=d["turnover"].mean(),
                           cost_drag=d["cost"].sum() / len(d) * 252,
                           borrow_drag=d["borrow"].sum() / len(d) * 252)
            if label.startswith("dev"):
                calm = r.loc[~r.index.year.isin([2008, 2020, 2022])]
                row["sharpe_ex_2008_2020_2022"] = perf_stats(calm)["sharpe"]
                row.update(year_bootstrap(r))
                if len(trial) >= 2:  # deflation needs the spread of Sharpe across trials
                    row.update(deflated_sharpe(r, trial, len(variants) + 25))
                else:
                    row["psr_vs_0"] = deflated_sharpe(r, np.array([0.0, 0.0]), 2)["psr_vs_0"]
            rows.append(row)
        summary = pd.DataFrame(rows)
        summary.to_csv(OUTPUT_DIR / f"summary_{label}.csv", index=False)
        yearly = sub.groupby(sub.index.year).apply(lambda x: (1 + x).prod() - 1)
        yearly.to_csv(OUTPUT_DIR / f"yearly_{label}.csv")
        with pd.option_context("display.width", 200, "display.precision", 4):
            print(f"\n== {label} ==")
            print(summary.set_index("model").T.to_string())

    full = results["ciaran_full"]
    sel, par = plans_table(full["plans"])
    sel.to_csv(OUTPUT_DIR / "selections.csv", index=False)
    par.to_csv(OUTPUT_DIR / "year_parameters.csv", index=False)
    full["trades"].to_csv(OUTPUT_DIR / "trades.csv", index=False)
    full["daily"].to_csv(OUTPUT_DIR / "daily_ciaran_full.csv", index=False)
    full["pair_pnl"].to_csv(OUTPUT_DIR / "pair_pnl_ciaran_full.csv")
    pd.Series(asdict(cfg)).to_csv(OUTPUT_DIR / "config.csv")
    print(f"Done in {time.time() - t:.0f}s. Results in {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
