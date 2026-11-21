"""Build the model's input data (all saved under ``data/``).

* ``universe_closes.csv`` - adjusted closes for the universe plus ``^VIX``;
* ``pit_pair_selection.csv`` - the pairs traded in each test year, screened
  on the ten years *before* that year (point-in-time, no look-ahead);
* ``pit_pair_prices.csv`` - prices, returns and VIX for every selected pair;
* ``earnings_dates.csv`` - quarterly report times, reported EPS, consensus
  and surprise for every universe ticker (inputs to the exit classifier).

The committed files cover 2014-2023. To extend, e.g. for a 2024-2025 test::

    python -m pairs_model.dataset --end-year 2025 --refresh --earnings

Remaining bias: the universe is today's list of large caps, so survivorship
bias is reduced (pairs are re-screened yearly) but not removed.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import time
import warnings

import pandas as pd

from pairs_model.universe import UNIVERSE, download_closes, screen, select_pairs

VIX_TICKER = "^VIX"
DATA_DIR = Path("data")


def pair_id(ticker_a: str, ticker_b: str) -> str:
    return f"{ticker_a}_{ticker_b}".lower()


def select_by_year(
    closes: pd.DataFrame,
    test_years: range,
    screen_years: int,
    count: int,
    window_months: int = 24,
    step_months: int = 6,
) -> pd.DataFrame:
    """Run the screen on the ``screen_years`` before each test year."""
    rows: list[pd.DataFrame] = []
    for year in test_years:
        end = pd.Timestamp(year=year, month=1, day=1)
        start = end - pd.DateOffset(years=screen_years)
        window = closes.loc[(closes.index >= start) & (closes.index < end)]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            selected = select_pairs(screen(window, window_months, step_months), count)
        selected.insert(0, "test_year", year)
        selected.insert(1, "pair_id", [
            pair_id(a, b) for a, b in zip(selected["ticker_a"], selected["ticker_b"])
        ])
        print(f"{year}: " + ", ".join(selected["pair_id"]), flush=True)
        rows.append(selected)
    return pd.concat(rows, ignore_index=True)


def make_pair_prices(
    closes: pd.DataFrame,
    vix: pd.Series,
    pairs: tuple[tuple[str, str, str, str], ...],
) -> pd.DataFrame:
    """Pair-aligned prices/returns with VIX; dates without a VIX close are dropped.

    ``pairs`` holds ``(pair_id, pair_name, ticker_a, ticker_b)`` tuples.
    """
    frames: list[pd.DataFrame] = []
    for pair_id, pair_name, ticker_a, ticker_b in pairs:
        pair = pd.DataFrame(
            {
                "adjusted_close_a": closes[ticker_a],
                "adjusted_close_b": closes[ticker_b],
            }
        ).dropna()
        pair["return_a"] = pair["adjusted_close_a"].pct_change(fill_method=None)
        pair["return_b"] = pair["adjusted_close_b"].pct_change(fill_method=None)
        pair["vix"] = vix.reindex(pair.index)
        pair = pair.dropna(subset=["vix"])
        pair.insert(0, "ticker_b", ticker_b)
        pair.insert(0, "ticker_a", ticker_a)
        pair.insert(0, "pair_name", pair_name)
        pair.insert(0, "pair_id", pair_id)
        frames.append(pair.reset_index())
    return pd.concat(frames, ignore_index=True)


def download_earnings(tickers: list[str], request_delay: float = 3.0) -> pd.DataFrame:
    """Yahoo earnings history per ticker (report timestamp, EPS, consensus, surprise)."""
    import yfinance as yf

    frames = []
    for ticker in tickers:
        for attempt in range(3):
            try:
                frame = yf.Ticker(ticker).get_earnings_dates(limit=100)
                break
            except Exception:  # Yahoo rate limits; back off and retry
                frame = None
                time.sleep(10 * (attempt + 1))
        if frame is not None and not frame.empty:
            frame = frame.reset_index().rename(columns={"Earnings Date": "ts"})
            frame["ticker"] = ticker
            frames.append(frame)
        time.sleep(request_delay)
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--start-year", type=int, default=2014, help="First test year.")
    parser.add_argument("--end-year", type=int, default=2023, help="Last test year.")
    parser.add_argument("--screen-years", type=int, default=10)
    parser.add_argument("--count", type=int, default=5, help="Pairs per year.")
    parser.add_argument("--request-delay", type=float, default=5.0)
    parser.add_argument("--refresh", action="store_true", help="Redownload universe prices.")
    parser.add_argument("--earnings", action="store_true", help="Redownload earnings dates.")
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    tickers = sorted({t for group in UNIVERSE.values() for t in group})
    cache = DATA_DIR / "universe_closes.csv"
    if cache.exists() and not args.refresh:
        closes = pd.read_csv(cache, index_col="date", parse_dates=["date"])
    else:
        start = pd.Timestamp(year=args.start_year - args.screen_years, month=1, day=1)
        end = pd.Timestamp(year=args.end_year + 1, month=1, day=1)
        closes = download_closes([*tickers, VIX_TICKER], start, end, request_delay=args.request_delay)
        closes.to_csv(cache, date_format="%Y-%m-%d")
    if VIX_TICKER not in closes or closes[VIX_TICKER].isna().all():
        raise RuntimeError("No VIX prices; rerun with --refresh.")

    stocks = closes.drop(columns=[VIX_TICKER])
    selection = select_by_year(stocks, range(args.start_year, args.end_year + 1),
                               args.screen_years, args.count)
    pairs = tuple(
        (pid, f"{a} / {b}", a, b)
        for pid, a, b in selection[["pair_id", "ticker_a", "ticker_b"]].drop_duplicates()
        .itertuples(index=False)
    )
    prices = make_pair_prices(stocks, closes[VIX_TICKER].dropna(), pairs)
    selection.to_csv(DATA_DIR / "pit_pair_selection.csv", index=False)
    prices.to_csv(DATA_DIR / "pit_pair_prices.csv", index=False, date_format="%Y-%m-%d")
    print(f"Selected {len(pairs)} distinct pairs over {selection['test_year'].nunique()} years.")

    if args.earnings or not (DATA_DIR / "earnings_dates.csv").exists():
        download_earnings(tickers).to_csv(DATA_DIR / "earnings_dates.csv", index=False)
        print("Wrote data/earnings_dates.csv")


if __name__ == "__main__":
    main()
