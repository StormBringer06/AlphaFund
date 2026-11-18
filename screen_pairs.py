"""Rough screen of same-sector US stock pairs for spread mean reversion.

Downloads about ten years of adjusted closes for a fixed candidate universe,
scores every same-sector pair, and picks a small, diversified set that suits a
log-price spread model refit on rolling two-year windows. A pair is kept when:

* daily returns are strongly correlated,
* log prices are cointegrated over the full period (Engle-Granger),
* the rolling two-year hedge ratio stays clearly positive, and
* the spread's mean-reversion half-life is short enough to trade daily.

Survivors are ranked by how often the rolling two-year windows are themselves
cointegrated, which is the property a periodically refit model relies on.

This is a screen, not a backtest. Selecting pairs on the same period you later
backtest overstates performance; pass ``--end`` to screen on an earlier period
and evaluate the chosen pairs afterwards.
"""

from __future__ import annotations

import argparse
from itertools import combinations
import math
from pathlib import Path
import time
import warnings

import numpy as np
import pandas as pd


UNIVERSE: dict[str, tuple[str, ...]] = {
    "banks": ("JPM", "BAC", "C", "WFC", "USB", "PNC", "TFC", "GS", "MS"),
    "payments": ("V", "MA", "AXP"),
    "energy": ("XOM", "CVX", "COP", "EOG", "OXY", "PSX", "VLO", "MPC", "SLB", "HAL"),
    "staples": ("KO", "PEP", "PG", "CL", "KMB", "GIS", "MDLZ", "HSY"),
    "retail": ("WMT", "COST", "TGT", "HD", "LOW"),
    "utilities": ("DUK", "SO", "AEP", "XEL", "ED", "D", "NEE", "WEC"),
    "reits": ("PSA", "ESS", "MAA", "UDR"),
    "rails": ("UNP", "CSX", "NSC"),
    "insurance": ("CB", "TRV", "ALL"),
    "pharma": ("JNJ", "PFE", "MRK", "BMY", "ABT"),
    "defense": ("LMT", "GD", "NOC", "RTX"),
}

MIN_RETURN_CORRELATION = 0.6
MAX_FULL_PERIOD_PVALUE = 0.05
MIN_ROLLING_HEDGE_RATIO = 0.2
HALF_LIFE_RANGE_DAYS = (5.0, 75.0)  # Well inside a 2-year window / 6-month refit.
WINDOW_YEARS = 2
WINDOW_STEP_MONTHS = 6
MAX_PAIRS_PER_SECTOR = 2


def download_closes(
    tickers: list[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
    chunk_size: int = 10,
    request_delay: float = 5.0,
    retry: bool = True,
) -> pd.DataFrame:
    """Fetch adjusted closes in small batches to stay polite to Yahoo Finance."""
    import yfinance as yf

    frames: list[pd.DataFrame] = []
    for index in range(0, len(tickers), chunk_size):
        if index:
            time.sleep(request_delay)
        chunk = tickers[index : index + chunk_size]
        downloaded = yf.download(
            tickers=chunk,
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            interval="1d",
            auto_adjust=True,
            actions=False,
            progress=False,
            threads=False,
        )
        if downloaded.empty:
            continue  # Unavailable tickers are skipped; screen() ignores them.
        closes = downloaded["Close"]
        if isinstance(closes, pd.Series):
            closes = closes.to_frame(chunk[0])
        frames.append(closes)

    if not frames:
        return pd.DataFrame(index=pd.DatetimeIndex([], name="date"))
    closes = pd.concat(frames, axis=1)
    # Yahoo reports rate-limited tickers as "possibly delisted"; retry them once.
    missing = [t for t in tickers if t not in closes or closes[t].isna().all()]
    if missing and retry:
        time.sleep(request_delay * 4)
        retried = download_closes(missing, start, end, chunk_size, request_delay, False)
        closes = closes.drop(columns=missing, errors="ignore").join(retried, how="outer")
        closes.index.name = "date"
    dates = pd.to_datetime(closes.index)
    if dates.tz is not None:
        dates = dates.tz_localize(None)
    closes.index = dates.normalize()
    closes.index.name = "date"
    return closes.sort_index()


def hedge_ratio(log_a: pd.Series, log_b: pd.Series) -> float:
    """OLS slope of log price A on log price B, as in NaivePairsTradingModel."""
    return float(log_b.cov(log_a) / log_b.var())


def half_life(spread: pd.Series) -> float:
    """Days for a spread deviation to halve, from an AR(1) fit of its changes."""
    lagged = spread.shift().dropna()
    change = spread.diff().dropna()
    slope = float(np.polyfit(lagged.to_numpy(), change.to_numpy(), 1)[0])
    return -math.log(2) / slope if slope < 0 else math.inf


def score_pair(log_a: pd.Series, log_b: pd.Series) -> dict[str, float]:
    """Full-period and rolling-window diagnostics for one oriented pair."""
    from statsmodels.tsa.stattools import coint

    full_beta = hedge_ratio(log_a, log_b)
    spread = log_a - full_beta * log_b
    window_pvalues: list[float] = []
    window_betas: list[float] = []
    window_start = log_a.index[0]
    last_start = log_a.index[-1] - pd.DateOffset(years=WINDOW_YEARS)
    while window_start <= last_start:
        window_end = window_start + pd.DateOffset(years=WINDOW_YEARS)
        a = log_a.loc[window_start:window_end]
        b = log_b.loc[window_start:window_end]
        window_pvalues.append(float(coint(a, b)[1]))
        window_betas.append(hedge_ratio(a, b))
        window_start += pd.DateOffset(months=WINDOW_STEP_MONTHS)

    return {
        "full_pvalue": float(coint(log_a, log_b)[1]),
        "hedge_ratio": full_beta,
        "min_window_hedge_ratio": min(window_betas),
        "max_window_hedge_ratio": max(window_betas),
        "window_coint_rate": float(np.mean(np.array(window_pvalues) < 0.05)),
        "half_life_days": half_life(spread),
    }


def screen(closes: pd.DataFrame) -> pd.DataFrame:
    """Score every same-sector pair that has complete data."""
    rows: list[dict[str, object]] = []
    for sector, tickers in UNIVERSE.items():
        available = [ticker for ticker in tickers if ticker in closes.columns]
        for first, second in combinations(available, 2):
            prices = closes[[first, second]].dropna()
            if len(prices) < 0.95 * len(closes):
                continue  # Listed late or patchy history; skip rather than guess.
            logs = np.log(prices)
            correlation = float(logs[first].diff().corr(logs[second].diff()))
            row: dict[str, object] = {
                "sector": sector,
                "return_correlation": correlation,
            }
            if correlation < MIN_RETURN_CORRELATION:
                rows.append({**row, "ticker_a": first, "ticker_b": second})
                continue

            # Engle-Granger is not symmetric; keep the orientation it prefers.
            forward = score_pair(logs[first], logs[second])
            backward = score_pair(logs[second], logs[first])
            if backward["full_pvalue"] < forward["full_pvalue"]:
                rows.append({**row, "ticker_a": second, "ticker_b": first, **backward})
            else:
                rows.append({**row, "ticker_a": first, "ticker_b": second, **forward})

    results = pd.DataFrame(rows)
    low, high = HALF_LIFE_RANGE_DAYS
    results["passes"] = (
        (results["return_correlation"] >= MIN_RETURN_CORRELATION)
        & (results["full_pvalue"] <= MAX_FULL_PERIOD_PVALUE)
        & (results["min_window_hedge_ratio"] >= MIN_ROLLING_HEDGE_RATIO)
        & results["half_life_days"].between(low, high)
    )
    return results.sort_values(
        ["passes", "window_coint_rate", "full_pvalue"],
        ascending=[False, False, True],
    ).reset_index(drop=True)


def select_pairs(results: pd.DataFrame, count: int) -> pd.DataFrame:
    """Greedily take the best passing pairs, reusing no ticker."""
    chosen: list[int] = []
    used_tickers: set[str] = set()
    sector_counts: dict[str, int] = {}
    for index, row in results[results["passes"]].iterrows():
        if row["ticker_a"] in used_tickers or row["ticker_b"] in used_tickers:
            continue
        if sector_counts.get(row["sector"], 0) >= MAX_PAIRS_PER_SECTOR:
            continue
        chosen.append(index)
        used_tickers.update((row["ticker_a"], row["ticker_b"]))
        sector_counts[row["sector"]] = sector_counts.get(row["sector"], 0) + 1
        if len(chosen) == count:
            break
    return results.loc[chosen].reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--end", help="Exclusive end date (default: today).")
    parser.add_argument("--years", type=int, default=10, help="Years to screen.")
    parser.add_argument("--count", type=int, default=5, help="Pairs to select.")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data"), help="Where to write CSVs."
    )
    parser.add_argument(
        "--refresh", action="store_true", help="Redownload even if prices are cached."
    )
    args = parser.parse_args()

    end = pd.Timestamp(args.end) if args.end else pd.Timestamp.today().normalize()
    start = end - pd.DateOffset(years=args.years)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache = args.output_dir / "screen_closes.csv"
    tickers = sorted({ticker for group in UNIVERSE.values() for ticker in group})

    if cache.exists() and not args.refresh:
        closes = pd.read_csv(cache, index_col="date", parse_dates=["date"])
    else:
        closes = download_closes(tickers, start, end)
        closes.to_csv(cache, date_format="%Y-%m-%d")
    closes = closes.loc[(closes.index >= start) & (closes.index < end)]
    print(f"Screening {closes.shape[1]} tickers, {closes.index[0].date()} to "
          f"{closes.index[-1].date()} ({len(closes)} trading days).")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        results = screen(closes)
    results.to_csv(args.output_dir / "pair_screen.csv", index=False)
    selected = select_pairs(results, args.count)

    print(f"{int(results['passes'].sum())} of {len(results)} same-sector pairs "
          "passed every filter.\n")
    columns = [
        "sector", "ticker_a", "ticker_b", "return_correlation", "full_pvalue",
        "window_coint_rate", "half_life_days", "min_window_hedge_ratio",
        "max_window_hedge_ratio",
    ]
    with pd.option_context("display.width", 200, "display.precision", 3):
        print(selected[columns].to_string(index=False))
    if len(selected) < args.count:
        print(f"\nOnly {len(selected)} pairs met the criteria.")
    print(f"\nFull results: {args.output_dir / 'pair_screen.csv'}")


if __name__ == "__main__":
    main()
