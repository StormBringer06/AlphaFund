"""Download adjusted daily prices for the selected pairs-trading universe."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
import yfinance as yf


PAIRS = (
    ("xel_wec", "Xcel Energy / WEC Energy", "XEL", "WEC"),
    ("bac_pnc", "Bank of America / PNC Financial", "BAC", "PNC"),
    ("all_trv", "Allstate / Travelers", "ALL", "TRV"),
    ("tfc_usb", "Truist Financial / U.S. Bancorp", "TFC", "USB"),
    ("hd_low", "Home Depot / Lowe's", "HD", "LOW"),
)
TICKERS = tuple(dict.fromkeys(ticker for _, _, a, b in PAIRS for ticker in (a, b)))


def download_adjusted_closes(
    start: str | None, end: str | None, request_delay: float = 10.0
) -> pd.DataFrame:
    """Fetch each ticker's adjusted daily closing prices sequentially."""
    options: dict[str, object] = {
        "interval": "1d",
        "auto_adjust": True,
        "actions": False,
        "progress": False,
        "threads": False,
    }
    if start or end:
        options["start"] = start
        options["end"] = end
    else:
        options["period"] = "max"

    ticker_closes: dict[str, pd.Series] = {}
    for index, ticker in enumerate(TICKERS):
        if index:
            time.sleep(request_delay)
        downloaded = yf.download(tickers=ticker, **options)
        if downloaded.empty:
            raise RuntimeError(f"Yahoo Finance returned no price data for {ticker}.")

        if isinstance(downloaded.columns, pd.MultiIndex):
            if "Close" in downloaded.columns.get_level_values(0):
                prices = downloaded["Close"]
            elif "Close" in downloaded.columns.get_level_values(1):
                prices = downloaded.xs("Close", axis=1, level=1)
            else:
                raise RuntimeError(
                    f"Yahoo Finance response for {ticker} had no closing prices."
                )
            if ticker not in prices.columns:
                raise RuntimeError(
                    f"Yahoo Finance response omitted requested ticker {ticker}."
                )
            series = prices[ticker]
        elif "Close" in downloaded.columns:
            series = downloaded["Close"]
        else:
            raise RuntimeError(
                f"Yahoo Finance response for {ticker} had no closing prices."
            )

        if series.dropna().empty:
            raise RuntimeError(f"Yahoo Finance returned no prices for {ticker}.")
        ticker_closes[ticker] = series

    closes = pd.DataFrame(ticker_closes)
    dates = pd.to_datetime(closes.index)
    if dates.tz is not None:
        dates = dates.tz_localize(None)
    closes.index = dates.normalize()
    closes.index.name = "date"
    if closes.index.has_duplicates:
        raise RuntimeError("Yahoo Finance returned duplicate trading dates.")

    return closes.sort_index()


def make_pair_prices(closes: pd.DataFrame) -> pd.DataFrame:
    """Build pair-aligned prices and returns without filling market holidays."""
    frames: list[pd.DataFrame] = []
    for pair_id, pair_name, ticker_a, ticker_b in PAIRS:
        pair = pd.DataFrame(
            {
                "adjusted_close_a": closes[ticker_a],
                "adjusted_close_b": closes[ticker_b],
            }
        ).dropna()
        pair["return_a"] = pair["adjusted_close_a"].pct_change(fill_method=None)
        pair["return_b"] = pair["adjusted_close_b"].pct_change(fill_method=None)
        pair.insert(0, "ticker_b", ticker_b)
        pair.insert(0, "ticker_a", ticker_a)
        pair.insert(0, "pair_name", pair_name)
        pair.insert(0, "pair_id", pair_id)
        frames.append(pair.reset_index())

    return pd.concat(frames, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", help="First date to download (YYYY-MM-DD).")
    parser.add_argument("--end", help="Exclusive end date (YYYY-MM-DD).")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data"),
        help="Directory for CSV outputs (default: data).",
    )
    parser.add_argument(
        "--request-delay",
        type=float,
        default=10.0,
        help="Seconds to wait between ticker requests (default: 10).",
    )
    args = parser.parse_args()
    if args.request_delay < 0:
        parser.error("--request-delay must be non-negative.")

    closes = download_adjusted_closes(args.start, args.end, args.request_delay)
    pair_prices = make_pair_prices(closes)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    closes.to_csv(args.output_dir / "adjusted_close_wide.csv", date_format="%Y-%m-%d")
    pair_prices.to_csv(
        args.output_dir / "pair_prices.csv",
        index=False,
        date_format="%Y-%m-%d",
    )
    print(
        f"Wrote {len(closes)} dates for {len(TICKERS)} tickers and "
        f"{len(pair_prices)} pair-date rows to {args.output_dir}."
    )


if __name__ == "__main__":
    main()
