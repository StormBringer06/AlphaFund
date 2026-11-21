"""Point-in-time pair selection: re-screen the universe before every test year.

``screen_pairs.py`` picked the five screened pairs using the latest ten years,
which overlap the 2014-2023 backtest, so every backtest on those pairs is
flattered by look-ahead. This script removes that bias: for each test year Y it
runs the same screen (``screen_pairs.screen`` / ``select_pairs``) on the ten
years ending the day before Y, using only data available then.

Outputs (in ``--output-dir``):

* ``universe_closes.csv`` - cached adjusted closes for the screen universe and
  ``^VIX`` (delete it or pass ``--refresh`` to redownload),
* ``pit_pair_selection.csv`` - one row per test year and selected pair,
* ``pit_pair_prices.csv`` - prices/returns/VIX for every pair selected in any
  year, in the ``screened_pair_prices.csv`` layout.

Remaining bias: the candidate universe itself is today's list of large caps,
so survivorship bias is reduced, not removed.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import warnings

import pandas as pd

from build_screened_dataset import VIX_TICKER, make_pair_prices
from screen_pairs import UNIVERSE, download_closes, screen, select_pairs


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--start-year", type=int, default=2014)
    parser.add_argument("--end-year", type=int, default=2023)
    parser.add_argument("--screen-years", type=int, default=10)
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument(
        "--window-months", type=int, default=24,
        help="Rolling-window length used to rank pairs (must fit in --screen-years).",
    )
    parser.add_argument("--step-months", type=int, default=6)
    parser.add_argument(
        "--output-prefix", default="pit",
        help="Writes <prefix>_pair_selection.csv and <prefix>_pair_prices.csv.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    parser.add_argument("--request-delay", type=float, default=5.0)
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()

    # Prices must cover the earliest screen window and the last test year.
    start = pd.Timestamp(year=args.start_year - args.screen_years, month=1, day=1)
    end = pd.Timestamp(year=args.end_year + 1, month=1, day=1)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache = args.output_dir / "universe_closes.csv"
    tickers = sorted({t for group in UNIVERSE.values() for t in group})

    if cache.exists() and not args.refresh:
        closes = pd.read_csv(cache, index_col="date", parse_dates=["date"])
    else:
        closes = download_closes(
            [*tickers, VIX_TICKER], start, end, request_delay=args.request_delay
        )
        closes.to_csv(cache, date_format="%Y-%m-%d")
    if VIX_TICKER not in closes or closes[VIX_TICKER].isna().all():
        raise RuntimeError("No VIX prices; rerun with --refresh.")

    vix = closes[VIX_TICKER].dropna()
    stocks = closes.drop(columns=[VIX_TICKER])
    selection = select_by_year(
        stocks,
        range(args.start_year, args.end_year + 1),
        args.screen_years,
        args.count,
        args.window_months,
        args.step_months,
    )
    pairs = tuple(
        (pid, f"{a} / {b}", a, b)
        for pid, a, b in selection[["pair_id", "ticker_a", "ticker_b"]]
        .drop_duplicates()
        .itertuples(index=False)
    )
    prices = make_pair_prices(stocks, vix, pairs)

    prefix = args.output_prefix
    selection.to_csv(args.output_dir / f"{prefix}_pair_selection.csv", index=False)
    prices.to_csv(
        args.output_dir / f"{prefix}_pair_prices.csv", index=False, date_format="%Y-%m-%d"
    )
    print(
        f"Selected {len(pairs)} distinct pairs across {selection['test_year'].nunique()} "
        f"years; wrote {len(prices)} pair-date rows to {args.output_dir}."
    )


if __name__ == "__main__":
    main()
