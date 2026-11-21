"""Loss forensics for vol_scaled_naive: would news/volume indicators have
avoided the episodes behind most of its losses?

An episode is a run of same-sign position for one pair (BD's point-in-time
vol_scaled_naive backtest, 5 bps). For every episode we measure indicators
from the literature with information available at each close:

* earnings announcements (Yahoo dates; pre-open -> same day, after close ->
  next day) - firm-specific news makes divergence persist
  (Engelberg, Gao & Jagannathan 2009; Chan 2003);
* abnormal volume (volume / trailing 60-day median) with a price move;
* overnight gaps (|open/prev close - 1| vs trailing gap volatility);
* idiosyncratic moves vs the sector ETF (return - beta * ETF, z-scored);
* time without convergence (Engelberg et al.: exit after ~10 days).

Entry rules block a whole episode if the trigger fired in the 10 days up to
entry. Exit rules flatten from the day after the first trigger in the hold.
Each rule's effect is the P&L it removes, reported for the worst episodes
(89% of losses) and for all other episodes (profit given up).

    python -m improved_models.loss_forensics
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from screen_pairs import UNIVERSE

OUT = Path("backtest_results") / "loss_forensics"
SECTOR_ETF = {"banks": "XLF", "insurance": "XLF", "payments": "XLF", "utilities": "XLU",
              "staples": "XLP", "energy": "XLE", "reits": "VNQ", "rails": "IYT",
              "defense": "ITA"}
SECTOR = {t: s for s, ts in UNIVERSE.items() for t in ts}
BUILDUP = 10


def ticker_signals(ohlcv: pd.DataFrame, earnings: pd.DataFrame, tickers: list[str]) -> dict:
    close, adj = ohlcv["Close"], ohlcv["Adj Close"]
    ret = adj.pct_change(fill_method=None)
    gap = ohlcv["Open"] / close.shift(1) - 1
    dates = ohlcv.index
    out = {}
    for t in tickers:
        etf = SECTOR_ETF[SECTOR[t]]
        beta = (ret[t].rolling(252, min_periods=120).cov(ret[etf])
                / ret[etf].rolling(252, min_periods=120).var()).shift(1)
        idio = ret[t] - beta * ret[etf]
        e = earnings.loc[earnings["ticker"] == t]
        react = []
        for ts in e["ts"]:
            day = ts.tz_localize(None).normalize()
            i = dates.searchsorted(day if ts.hour < 12 else day + pd.Timedelta(days=1))
            if i < len(dates):
                react.append(dates[i])
        out[t] = pd.DataFrame({
            "ret": ret[t],
            "abvol": ohlcv["Volume"][t] / ohlcv["Volume"][t].rolling(60, min_periods=20).median().shift(1),
            "gap_z": gap[t].abs() / gap[t].rolling(120, min_periods=60).std().shift(1),
            "idio_z": idio / idio.rolling(120, min_periods=60).std().shift(1),
            "earnings": dates.isin(react),
        }, index=dates)
    return out


def build_episodes(daily: pd.DataFrame) -> pd.DataFrame:
    d = daily.sort_values(["pair_id", "fold_id", "date"]).copy()
    side = np.sign(d["position_weight_a"].round(12))
    d["side"] = side
    d["ep"] = (side != side.groupby([d["pair_id"], d["fold_id"]]).shift()).cumsum()
    return d


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    daily = pd.read_csv("backtest_results/vol_scaled_naive/daily_results_5bps.csv",
                        parse_dates=["date"], low_memory=False)
    daily = build_episodes(daily.loc[daily["model_name"] == "vol_scaled_naive"])
    ohlcv = pd.read_pickle("data/ciaran_extra/ohlcv.pkl")
    earnings = pd.read_csv("data/ciaran_extra/earnings_dates.csv")
    earnings["ts"] = pd.to_datetime(earnings["ts"], utc=True).dt.tz_convert("America/New_York")
    tickers = sorted(set(daily["ticker_a"]) | set(daily["ticker_b"]))
    sig = ticker_signals(ohlcv, earnings, tickers)
    dates = ohlcv.index

    # unit spread return and its trailing vol per pair (for "adverse move" tests)
    beta = daily["hedge_ratio"]
    daily["spread_ret"] = (daily["return_a"] - beta * daily["return_b"]) / (1 + beta.abs())
    daily["spread_sd"] = daily.groupby("pair_id")["spread_ret"].transform(
        lambda s: s.rolling(60, min_periods=20).std().shift(1))

    held = daily.loc[daily["side"] != 0]
    eps = held.groupby("ep").agg(pair=("pair_id", "first"), a=("ticker_a", "first"),
                                 b=("ticker_b", "first"), side=("side", "first"),
                                 start=("date", "first"), end=("date", "last"),
                                 days=("date", "size"), pnl=("strategy_return", "sum"),
                                 max_abs_z=("spread_zscore", lambda z: z.abs().max()))
    eps = eps.sort_values("pnl")
    total_loss = eps.loc[eps["pnl"] < 0, "pnl"].sum()
    n_worst = int(0.1 * len(eps))
    eps["worst"] = False
    eps.iloc[:n_worst, eps.columns.get_loc("worst")] = True
    print(f"{len(eps)} episodes; worst {n_worst} = {eps.loc[eps.worst, 'pnl'].sum() / total_loss:.0%} "
          f"of losses ({eps.loc[eps.worst, 'pnl'].sum():.3f} of {total_loss:.3f}); "
          f"total P&L {eps['pnl'].sum():.3f}")

    # ---------------- per-episode indicator evaluation
    entry_rules = {
        "earnings (either leg) in 10d before entry":
            lambda s, w: s["earnings"].loc[w].any(),
        "abnormal volume >=2.5x with |idio| >=2 in 10d before entry":
            lambda s, w: ((s["abvol"] >= 2.5) & (s["idio_z"].abs() >= 2)).loc[w].any(),
        "overnight gap >=3 sd in 10d before entry":
            lambda s, w: (s["gap_z"] >= 3).loc[w].any(),
        "idiosyncratic move >=3 sd in 10d before entry":
            lambda s, w: (s["idio_z"].abs() >= 3).loc[w].any(),
    }
    rows, exit_rows = [], []
    for ep_id, e in eps.iterrows():
        path = held.loc[held["ep"] == ep_id].set_index("date")
        i0 = dates.searchsorted(e.start)
        window = dates[max(0, i0 - BUILDUP): i0 + 1]
        legs = [sig[e.a], sig[e.b]]
        row = {"ep": ep_id, **e.to_dict()}
        for name, rule in entry_rules.items():
            row[name] = any(bool(rule(s, window)) for s in legs)
        # earnings timing detail
        hold_dates = path.index
        row["earnings_in_hold"] = any(s["earnings"].loc[hold_dates].any() for s in legs)
        rows.append(row)

        # exit triggers evaluated at each close in the hold (act from next day)
        adverse = (e.side * path["spread_ret"]) < -2 * path["spread_sd"]
        any_leg = lambda f: pd.concat([f(s).reindex(hold_dates) for s in legs], axis=1).any(axis=1)
        k = np.arange(len(path))
        triggers = {
            "earnings day with adverse spread move (>2sd)":
                any_leg(lambda s: s["earnings"]) & adverse,
            "abnormal volume >=3x with adverse spread move":
                any_leg(lambda s: s["abvol"] >= 3) & adverse,
            "idiosyncratic jump >=3sd with adverse spread move":
                any_leg(lambda s: s["idio_z"].abs() >= 3) & adverse,
            "overnight gap >=3sd with adverse spread move":
                any_leg(lambda s: s["gap_z"] >= 3) & adverse,
            "not converged after 20 days": pd.Series(k >= 19, index=hold_dates),
            "not converged after 40 days": pd.Series(k >= 39, index=hold_dates),
            "|z| >= 3 stop (benchmark)": path["spread_zscore"].abs() >= 3,
        }
        for name, trig in triggers.items():
            hits = np.flatnonzero(trig.to_numpy())
            if len(hits):
                first = hits[0]
                saved = -float(path["strategy_return"].iloc[first + 1:].sum())
                exit_rows.append({"ep": ep_id, "rule": name, "worst": e.worst, "trigger_day": first,
                                  "trigger_date": hold_dates[first], "pnl_change": saved,
                                  "episode_pnl": e.pnl})
    ev = pd.DataFrame(rows).set_index("ep")
    ex = pd.DataFrame(exit_rows)

    # ---------------- summaries
    def summarise_entry(name):
        f = ev[name]
        w, o = ev.loc[f & ev.worst], ev.loc[f & ~ev.worst]
        return {"rule": name,
                "worst caught": f"{len(w)}/{n_worst}",
                "worst loss avoided": -w["pnl"].sum(),
                "other episodes blocked": f"{len(o)}/{len(ev) - n_worst}",
                "profit given up": o["pnl"].sum(),
                "net P&L change": -ev.loc[f, "pnl"].sum(),
                "share of all episodes flagged": f.mean()}

    def summarise_exit(name):
        g = ex.loc[ex["rule"] == name]
        w, o = g.loc[g["worst"]], g.loc[~g["worst"]]
        return {"rule": name,
                "worst caught": f"{len(w)}/{n_worst}",
                "worst loss avoided": w["pnl_change"].sum(),
                "other episodes cut": f"{len(o)}/{len(ev) - n_worst}",
                "profit given up": -o["pnl_change"].sum(),
                "net P&L change": g["pnl_change"].sum(),
                "median trigger day (worst)": w["trigger_day"].median()}

    entry = pd.DataFrame([summarise_entry(n) for n in entry_rules])
    exits = pd.DataFrame([summarise_exit(n) for n in ex["rule"].unique()])
    base_rate = ev.loc[~ev.worst]
    news = pd.DataFrame({
        "group": ["worst episodes", "other episodes"],
        "earnings in 10d before entry": [ev.loc[ev.worst, list(entry_rules)[0]].mean(),
                                          base_rate[list(entry_rules)[0]].mean()],
        "earnings during hold": [ev.loc[ev.worst, "earnings_in_hold"].mean(),
                                 base_rate["earnings_in_hold"].mean()],
        "median days held": [ev.loc[ev.worst, "days"].median(), base_rate["days"].median()],
    })

    # per-episode table for the worst ones
    worst_tbl = ev.loc[ev.worst, ["pair", "start", "end", "days", "pnl", "max_abs_z", "earnings_in_hold",
                                  *entry_rules]].copy()
    for name in ex["rule"].unique():
        g = ex.loc[(ex["rule"] == name)].set_index("ep")
        worst_tbl[f"exit: {name} (day / saved)"] = [
            f"d{int(g.at[i, 'trigger_day'])} / {g.at[i, 'pnl_change']:+.3f}" if i in g.index else "-"
            for i in worst_tbl.index]

    entry.to_csv(OUT / "entry_rules.csv", index=False)
    exits.to_csv(OUT / "exit_rules.csv", index=False)
    news.to_csv(OUT / "news_incidence.csv", index=False)
    worst_tbl.to_csv(OUT / "worst_episodes.csv")
    ev.to_csv(OUT / "all_episodes.csv")
    ex.to_csv(OUT / "exit_triggers.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 30, "display.precision", 3):
        print("\nIncidence:\n", news.to_string(index=False))
        print("\nENTRY rules (block whole episode):\n", entry.to_string(index=False))
        print("\nEXIT rules (flatten after first trigger):\n", exits.to_string(index=False))
        cols = ["pair", "start", "days", "pnl", "max_abs_z", "earnings_in_hold"] + \
            [c for c in worst_tbl.columns if c.startswith("exit: earnings") or c.startswith("exit: idio")
             or c.startswith("exit: not converged after 20")]
        print("\nWorst episodes:\n", worst_tbl[cols].to_string())


if __name__ == "__main__":
    main()
