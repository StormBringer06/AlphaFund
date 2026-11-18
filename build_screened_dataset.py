"""Download long histories for the screened pairs plus the VIX index.

Writes, in the same layout as ``build_dataset.py``:

* ``data/screened_pair_prices.csv`` - one row per pair and shared trading date
  with adjusted prices and daily returns, plus that date's VIX close.
* ``data/vix.csv`` - daily VIX closes.

The pairs are the ones selected by ``screen_pairs.py`` (see README).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from screen_pairs import download_closes


SCREENED_PAIRS = (
    ("xel_wec", "Xcel Energy / WEC Energy", "XEL", "WEC"),
    ("bac_pnc", "Bank of America / PNC Financial", "BAC", "PNC"),
    ("all_trv", "Allstate / Travelers", "ALL", "TRV"),
    ("tfc_usb", "Truist Financial / U.S. Bancorp", "TFC", "USB"),
    ("hd_low", "Home Depot / Lowe's", "HD", "LOW"),
)
VIX_TICKER = "^VIX"


def make_pair_prices(closes: pd.DataFrame, vix: pd.Series) -> pd.DataFrame:
    """Pair-aligned prices/returns with VIX; dates without a VIX close are dropped."""
    frames: list[pd.DataFrame] = []
    for pair_id, pair_name, ticker_a, ticker_b in SCREENED_PAIRS:
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2005-01-01")
    parser.add_argument("--end", default=None, help="Exclusive end (default: today).")
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    parser.add_argument("--request-delay", type=float, default=5.0)
    args = parser.parse_args()

    start = pd.Timestamp(args.start)
    end = pd.Timestamp(args.end) if args.end else pd.Timestamp.today().normalize()
    tickers = [t for _, _, a, b in SCREENED_PAIRS for t in (a, b)] + [VIX_TICKER]
    closes = download_closes(tickers, start, end, request_delay=args.request_delay)
    missing = [t for t in tickers if t not in closes or closes[t].isna().all()]
    if missing:
        raise RuntimeError(f"Yahoo Finance returned no prices for: {missing}")

    vix = closes[VIX_TICKER].dropna()
    pair_prices = make_pair_prices(closes, vix)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    vix.rename("vix").to_csv(args.output_dir / "vix.csv", date_format="%Y-%m-%d")
    pair_prices.to_csv(
        args.output_dir / "screened_pair_prices.csv",
        index=False,
        date_format="%Y-%m-%d",
    )
    print(
        f"Wrote {len(vix)} VIX closes and {len(pair_prices)} pair-date rows "
        f"to {args.output_dir}."
    )


if __name__ == "__main__":
    main()
