# AlphaFund

## Building the pairs-trading dataset

The dataset builder downloads daily, dividend- and split-adjusted closing prices
from Yahoo Finance with `yfinance` for these pairs:

| Pair | Yahoo Finance tickers |
|---|---|
| JPMorgan Chase / Bank of America | `JPM` / `BAC` |
| Coca-Cola / PepsiCo | `KO` / `PEP` |
| Costco / Walmart | `COST` / `WMT` |
| Exxon Mobil / Chevron | `XOM` / `CVX` |
| Visa / Mastercard | `V` / `MA` |

Install the requirements and run the builder from the repository root:

```powershell
python -m pip install -r requirements.txt
python build_dataset.py
```

The builder downloads one ticker at a time and waits ten seconds between
requests by default. Adjust the pause if needed, for example:

```powershell
python build_dataset.py --request-delay 10
```

By default, the script requests all available daily history and writes:

- `data/adjusted_close_wide.csv` — one row per trading date and one column per
  ticker. Dates without a quote for a ticker remain blank.
- `data/pair_prices.csv` — one row per pair and date on which both tickers have
  a quote, with both adjusted prices and their daily percentage returns.

Pair returns are calculated between consecutive dates shared by both tickers;
the first shared date has blank returns. Dates are not forward-filled across
market holidays. Yahoo Finance data availability can change, so rerunning the
script may update the dataset.

## Naive pairs-trading baseline

`base_model.py` provides a per-pair `NaivePairsTradingModel`. Fit it on a
chronological training/formation period, then call `predict` on a later test
period:

```python
import pandas as pd

from base_model import NaivePairsTradingModel

data = pd.read_csv("data/pair_prices.csv", parse_dates=["date"])
pair = data.loc[data["pair_id"] == "jpm_bac"].sort_values("date")
split_date = pd.Timestamp("2010-01-01")
train = pair.loc[pair["date"] < split_date]
test = pair.loc[pair["date"] >= split_date]

model = NaivePairsTradingModel(full_exposure_zscore=2.0)
model.fit(train)
predictions = model.predict(test)
```

The model initially fits `log(price_a) = intercept + hedge_ratio *
log(price_b) + spread` using ordinary least squares on the latest two years of
training data. Every six calendar months, it re-estimates the hedge ratio,
intercept, and spread statistics using the prior two years of observations.
The scheduled refit uses only prices before that prediction date; the current
close is then scored against the updated model. For each test date, the
contrarian signal is `clip(-spread_zscore / 2, -1, 1)` by default: it creates a
position even for small nonzero divergences and reaches full gross exposure at
an absolute z-score of 2. The signal is split between the legs according to
the current hedge ratio, with gross exposure normalized to at most 1.

Predicted weights are close-of-day targets; `beta_refit` and `beta_refit_date`
indicate when the hedge ratio was last updated. A backtest should apply the
weights to the following period's returns. This baseline has no entry/exit
thresholds, volatility filter, cointegration screen, transaction costs, or
borrow costs; those can be evaluated separately.
