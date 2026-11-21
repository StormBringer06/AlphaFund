"""Oracle search: which properties separate vol_scaled_naive's worst episodes?

Exploratory, not a trading rule. For every episode we compute ~25 candidate
properties, each tagged by when it is knowable:

  entry      - known at the close where the position is opened
  during     - known part-way through the episode (first 20 trading days)
  hindsight  - measured over the whole episode; explains, cannot be traded

Directional properties are signed so that positive = favours the divergence
continuing (stock A keeps beating B when z > 0, and vice versa).

Replication guard: with 24 losers, something always separates them by chance.
A property only counts if it also separates the worst decile of ~7,000
pseudo-episodes on ~110 same-sector pairs (2006-2023).

    python -m improved_models.loss_oracle
"""

from __future__ import annotations

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from improved_models.drift_guard import pseudo_episodes, raw_features, traded_episodes
from screen_pairs import UNIVERSE

OUT = Path("backtest_results") / "loss_oracle"
SECTOR_ETF = {"banks": "XLF", "insurance": "XLF", "payments": "XLF", "utilities": "XLU",
              "staples": "XLP", "energy": "XLE", "reits": "VNQ", "rails": "IYT",
              "defense": "ITA", "pharma": "XLV", "retail": "XLY"}
SECTOR = {t: s for s, ts in UNIVERSE.items() for t in ts}

PROPS = {  # name: (timing, description)
    "eps_growth_gap": ("entry", "TTM EPS growth A - B (1y)"),
    "eps_ratio_change_1y": ("entry", "change in log(TTM EPS A / B) over 1y"),
    "last_surprise_gap": ("entry", "latest EPS surprise A - B"),
    "days_to_next_earnings": ("entry", "trading days to the next report of either stock (schedule known ahead)"),
    "earnings_within_10d": ("entry", "either stock reports within 10 days"),
    "mkt_beta_gap": ("entry", "|market beta A - B|"),
    "rate_beta_gap": ("entry", "|10y-yield beta A - B|"),
    "idio_vol_gap": ("entry", "|idiosyncratic vol A - B| vs sector ETF"),
    "mom_12_1_gap": ("entry", "12-1 month momentum A - B"),
    "mom_1m_gap": ("entry", "1-month return A - B"),
    "spy_above_200d": ("entry", "SPY above its 200-day average"),
    "spy_ret_3m": ("entry", "SPY 3-month return"),
    "tnx_change_3m": ("entry", "change in 10y yield over 3 months"),
    "credit_ret_3m": ("entry", "HYG 3-month return"),
    "vix_pct": ("entry", "VIX percentile"),
    "hl_ratio": ("entry", "log(half-life 120d / 504d)"),
    "abs_z": ("entry", "|z| at entry"),
    "z_after_20d": ("during", "|z| 20 trading days in, minus |z| at entry"),
    "spread_pnl_20d": ("during", "episode P&L over the first 20 days (unit spread)"),
    "earnings_in_first_20d": ("during", "either stock reported in the first 20 days"),
    "beta_shift": ("hindsight", "|hedge ratio over the episode - hedge ratio before it|"),
    "eps_ratio_change_hold": ("hindsight", "change in log(TTM EPS A / B) during the episode"),
    "rate_push_hold": ("hindsight", "(rate beta A - B) x yield change during episode"),
    "mkt_push_hold": ("hindsight", "(market beta A - B) x SPY return during episode"),
    "sector_rel_move_hold": ("hindsight", "|A minus B return| during episode vs typical"),
}
SIGNED = {"eps_growth_gap", "eps_ratio_change_1y", "last_surprise_gap", "mom_12_1_gap",
          "mom_1m_gap", "eps_ratio_change_hold", "rate_push_hold", "mkt_push_hold"}


def ticker_frames(stocks: pd.DataFrame, macro: pd.DataFrame, earn: pd.DataFrame) -> dict:
    ret = stocks.pct_change(fill_method=None)
    spy = macro["SPY"].reindex(stocks.index).pct_change(fill_method=None)
    dy = (macro["^TNX"].reindex(stocks.index) / 10).diff()
    out = {}
    for t in stocks.columns:
        etf = macro[SECTOR_ETF[SECTOR[t]]].reindex(stocks.index).pct_change(fill_method=None)
        r = ret[t]
        mkt_beta = r.rolling(252, min_periods=120).cov(spy) / spy.rolling(252, min_periods=120).var()
        rate_beta = r.rolling(252, min_periods=120).cov(dy) / dy.rolling(252, min_periods=120).var()
        sb = r.rolling(252, min_periods=120).cov(etf) / etf.rolling(252, min_periods=120).var()
        idio_vol = (r - sb * etf).rolling(120, min_periods=60).std()
        e = earn.loc[earn["ticker"] == t].dropna(subset=["Reported EPS"]).copy()
        day = e["ts"].dt.tz_localize(None).dt.normalize()
        e["react"] = np.where(e["ts"].dt.hour < 12, day, day + pd.Timedelta(days=1))
        e = e.sort_values("react").drop_duplicates("react", keep="last")
        e["ttm"] = e["Reported EPS"].rolling(4).sum()
        idx = stocks.index
        ttm = pd.Series(e["ttm"].to_numpy(), index=pd.DatetimeIndex(e["react"]))
        ttm = ttm.reindex(idx.union(ttm.index)).ffill().reindex(idx)
        surprise = pd.Series(e["Surprise(%)"].clip(-50, 50).to_numpy(), index=pd.DatetimeIndex(e["react"]))
        surprise = surprise.reindex(idx.union(surprise.index)).ffill().reindex(idx)
        react = pd.DatetimeIndex(e["react"])
        pos = react.searchsorted(idx)  # next report on/after each date
        next_report = pd.Series([react[p] if p < len(react) else pd.NaT for p in pos], index=idx)
        out[t] = pd.DataFrame({
            "price": stocks[t], "ret": r, "mkt_beta": mkt_beta, "rate_beta": rate_beta,
            "idio_vol": idio_vol, "ttm": ttm, "surprise": surprise, "next_report": next_report,
            "mom_12_1": stocks[t].shift(21) / stocks[t].shift(252) - 1,
            "mom_1m": stocks[t] / stocks[t].shift(21) - 1,
            "is_report": idx.isin(react),
        }, index=idx)
    return out


def episode_props(ep: pd.Series, a: str, b: str, z_sign: float, raw: pd.DataFrame,
                  tf: dict, macro: pd.DataFrame, end_idx: int | None = None) -> dict:
    dates = raw.index
    i = dates.get_loc(ep["decision"])
    j = dates.get_loc(ep["resolved"]) if end_idx is None else end_idx
    A, B = tf[a], tf[b]
    d0 = dates[i]

    def log_ttm_ratio(k):
        ta, tb = A["ttm"].iloc[k], B["ttm"].iloc[k]
        return np.log(ta / tb) if ta > 0 and tb > 0 else np.nan

    def growth(F, k):
        now, prev = F["ttm"].iloc[k], F["ttm"].iloc[max(k - 252, 0)]
        return np.clip((now - prev) / abs(prev), -1, 1) if prev and np.isfinite(prev) else np.nan

    nxt = min(A["next_report"].iloc[i], B["next_report"].iloc[i])
    days_next = dates.searchsorted(nxt) - i if pd.notna(nxt) else np.nan
    k20 = min(i + 20, j)
    spy, tnx, hyg = macro["SPY"].reindex(dates), macro["^TNX"].reindex(dates) / 10, macro["HYG"].reindex(dates)
    la, lb = np.log(A["price"]), np.log(B["price"])
    hold = slice(i, j + 1)
    pre = slice(max(i - 252, 0), i + 1)
    beta_hold = la.iloc[hold].cov(lb.iloc[hold]) / lb.iloc[hold].var() if j - i >= 10 else np.nan
    beta_pre = la.iloc[pre].cov(lb.iloc[pre]) / lb.iloc[pre].var()
    rel = (A["ret"] - B["ret"])
    typical = rel.iloc[pre].std() * np.sqrt(max(j - i, 1))
    p = {
        "eps_growth_gap": growth(A, i) - growth(B, i),
        "eps_ratio_change_1y": log_ttm_ratio(i) - log_ttm_ratio(max(i - 252, 0)),
        "last_surprise_gap": A["surprise"].iloc[i] - B["surprise"].iloc[i],
        "days_to_next_earnings": days_next,
        "earnings_within_10d": float(days_next <= 10) if np.isfinite(days_next) else np.nan,
        "mkt_beta_gap": abs(A["mkt_beta"].iloc[i] - B["mkt_beta"].iloc[i]),
        "rate_beta_gap": abs(A["rate_beta"].iloc[i] - B["rate_beta"].iloc[i]),
        "idio_vol_gap": abs(A["idio_vol"].iloc[i] - B["idio_vol"].iloc[i]),
        "mom_12_1_gap": A["mom_12_1"].iloc[i] - B["mom_12_1"].iloc[i],
        "mom_1m_gap": A["mom_1m"].iloc[i] - B["mom_1m"].iloc[i],
        "spy_above_200d": float(spy.iloc[i] > spy.iloc[max(i - 200, 0): i + 1].mean()),
        "spy_ret_3m": spy.iloc[i] / spy.iloc[max(i - 63, 0)] - 1,
        "tnx_change_3m": tnx.iloc[i] - tnx.iloc[max(i - 63, 0)],
        "credit_ret_3m": hyg.iloc[i] / hyg.iloc[max(i - 63, 0)] - 1,
        "vix_pct": raw["vix_pct"].iloc[i],
        "hl_ratio": raw["hl_ratio"].iloc[i],
        "abs_z": raw["abs_z"].iloc[i],
        "z_after_20d": abs(raw["z"].iloc[k20]) - abs(raw["z"].iloc[i]),
        "spread_pnl_20d": float((-np.sign(raw["z"].iloc[i]) * raw["spread_ret"].iloc[i + 1:k20 + 1]).sum()),
        "earnings_in_first_20d": float(A["is_report"].iloc[i + 1:k20 + 1].any() or B["is_report"].iloc[i + 1:k20 + 1].any()),
        "beta_shift": abs(beta_hold - beta_pre),
        "eps_ratio_change_hold": log_ttm_ratio(j) - log_ttm_ratio(i),
        "rate_push_hold": (A["rate_beta"].iloc[i] - B["rate_beta"].iloc[i]) * (tnx.iloc[j] - tnx.iloc[i]),
        "mkt_push_hold": (A["mkt_beta"].iloc[i] - B["mkt_beta"].iloc[i]) * (spy.iloc[j] / spy.iloc[i] - 1),
        "sector_rel_move_hold": abs(rel.iloc[i + 1:j + 1].sum()) / typical if typical > 0 else np.nan,
    }
    for k in SIGNED:
        p[k] = z_sign * p[k]
    return p


def best_cut(x: pd.Series, worst: pd.Series) -> dict:
    """Single threshold that screens the most losers per winner screened."""
    best = {"cut": np.nan, "side": "", "losers_screened": 0, "others_screened": 0, "score": -1}
    v = x.dropna()
    w = worst.loc[v.index]
    for q in np.linspace(0.05, 0.95, 19):
        c = v.quantile(q)
        for side, mask in ((">", v > c), ("<", v < c)):
            lo, ot = int((mask & w).sum()), int((mask & ~w).sum())
            score = lo / max(w.sum(), 1) - ot / max((~w).sum(), 1)
            if score > best["score"]:
                best = {"cut": c, "side": side, "losers_screened": lo, "others_screened": ot, "score": score}
    return best


def main() -> None:
    warnings.simplefilter("ignore")
    OUT.mkdir(parents=True, exist_ok=True)
    closes = pd.read_csv("data/universe_closes.csv", index_col="date", parse_dates=["date"])
    vix_pct = closes["^VIX"].ffill().rolling(504, min_periods=126).rank(pct=True)
    stocks = closes.drop(columns=["^VIX"])
    macro = pd.read_csv("data/ciaran_extra/macro_etf_closes.csv", index_col=0, parse_dates=True)
    earn = pd.read_csv("data/ciaran_extra/earnings_dates.csv")
    earn["ts"] = pd.to_datetime(earn["ts"], utc=True).dt.tz_convert("America/New_York")
    data = pd.read_csv("data/pit_pair_prices.csv", parse_dates=["date"])
    bd = pd.read_csv("backtest_results/vol_scaled_naive/daily_results_5bps.csv",
                     parse_dates=["date"], low_memory=False)
    bd = bd.loc[bd["model_name"] == "vol_scaled_naive"]

    pool = [(a, b) for ts in UNIVERSE.values() for i, a in enumerate(sorted(ts))
            for b in sorted(ts)[i + 1:] if a in stocks and b in stocks]
    traded = list(data[["ticker_a", "ticker_b"]].drop_duplicates().itertuples(index=False, name=None))
    print("Building features ...", flush=True)
    raws = {k: raw_features(stocks, vix_pct, earn, *k) for k in set(pool) | set(traded)}
    tf = ticker_frames(stocks, macro, earn)

    tr = traded_episodes(bd, raws)
    tickers = bd.drop_duplicates("pair_id").set_index("pair_id")[["ticker_a", "ticker_b"]]
    rows = []
    for _, e in tr.iterrows():
        a, b = tickers.loc[e.pair]
        z_sign = np.sign(raws[(a, b)].loc[e.decision, "z"])
        rows.append({**episode_props(e, a, b, z_sign, raws[(a, b)], tf, macro), "worst": e.worst, "pnl": e.pnl})
    traded_props = pd.DataFrame(rows)

    rows = []
    for a, b in pool:
        raw = raws[(a, b)]
        eps = pseudo_episodes(raw, f"{a}_{b}")
        eps = eps.loc[eps["decision"] >= "2006-01-01"]
        for _, e in eps.iterrows():
            rows.append({**episode_props(e, a, b, np.sign(raw.loc[e.decision, "z"]), raw, tf, macro),
                         "pnl": e.pnl, "days": e.days})
    pool_props = pd.DataFrame(rows)
    pool_props["worst"] = pool_props["pnl"] <= pool_props["pnl"].quantile(0.1)
    print(f"Traded episodes: {len(traded_props)} (worst {int(traded_props.worst.sum())}); "
          f"pool pseudo-episodes: {len(pool_props)} (worst {int(pool_props.worst.sum())})", flush=True)

    table = []
    for name, (timing, desc) in PROPS.items():
        def auc(df):
            v = df[[name, "worst"]].dropna()
            return roc_auc_score(v["worst"], v[name]) if v["worst"].nunique() == 2 else np.nan
        cut = best_cut(traded_props[name], traded_props["worst"])
        # apply the same cut to the pool
        pv = pool_props[name]
        pmask = (pv > cut["cut"]) if cut["side"] == ">" else (pv < cut["cut"])
        table.append({
            "property": name, "timing": timing, "description": desc,
            "median_worst": traded_props.loc[traded_props.worst, name].median(),
            "median_other": traded_props.loc[~traded_props.worst, name].median(),
            "auc_traded": auc(traded_props), "auc_pool": auc(pool_props),
            "cut": f"{cut['side']} {cut['cut']:.3g}",
            "traded_losers_screened": f"{cut['losers_screened']}/{int(traded_props.worst.sum())}",
            "traded_others_screened": f"{cut['others_screened']}/{int((~traded_props.worst).sum())}",
            "pool_worst_screened": f"{(pmask & pool_props.worst).mean() / pool_props.worst.mean():.0%}",
            "pool_others_screened": f"{(pmask & ~pool_props.worst).mean() / (~pool_props.worst).mean():.0%}",
        })
    table = pd.DataFrame(table)
    table["separation_traded"] = (table["auc_traded"] - 0.5).abs()
    table["replicates"] = (np.sign(table["auc_traded"] - 0.5) == np.sign(table["auc_pool"] - 0.5)) & \
                          ((table["auc_pool"] - 0.5).abs() >= 0.05)
    table = table.sort_values(["timing", "separation_traded"], ascending=[True, False])
    table.to_csv(OUT / "property_table.csv", index=False)
    traded_props.to_csv(OUT / "traded_episode_properties.csv", index=False)
    with pd.option_context("display.width", 260, "display.max_colwidth", 60, "display.precision", 3):
        print(table.drop(columns=["description", "separation_traded"]).to_string(index=False))


if __name__ == "__main__":
    main()
