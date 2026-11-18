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

## Screened pairs

`screen_pairs.py` screens same-sector US large caps over the last 10 years for
cointegration, a stable positive hedge ratio, and a tradable half-life. It
selected:

| Pair | Sector | Yahoo Finance tickers |
|---|---|---|
| Xcel Energy / WEC Energy | Utilities | `XEL` / `WEC` |
| Bank of America / PNC Financial | Banks | `BAC` / `PNC` |
| Allstate / Travelers | Insurance | `ALL` / `TRV` |
| Truist Financial / U.S. Bancorp | Banks | `TFC` / `USB` |
| Home Depot / Lowe's | Retail | `HD` / `LOW` |

```powershell
python screen_pairs.py
```

Results are written to `data/pair_screen.csv`. Pairs are chosen in-sample, so
use `--end` to screen an earlier period before backtesting later years.

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

## Walk-forward backtest

`backtest.py` uses the same `fit(train_data)` / `predict(test_data)` interface
for the baseline or another model. The default configuration tests each year
from 2014 through 2023 using the preceding four calendar years for training,
moving the window forward one year each fold:

```powershell
python backtest.py
```

By default, it writes `backtest_results/daily_results.csv` and
`backtest_results/summary.csv`. To plug in another model from Python, pass its
class as `model_factory` to `run_walk_forward_backtest`; it must fit on a
training DataFrame and return one prediction per test date with
`target_weight_a` and `target_weight_b` columns.

Targets are close-of-day signals, so the backtest applies each target to the
next shared trading date's pair returns. The first test-date return has no
market exposure (but includes the opening cost if costs are enabled). Each fold
starts flat and liquidates at its end. Transaction costs are optional and
default to zero; configure them with `transaction_cost_bps`. Summary statistics
are reported per model, pair, and test-year fold, including compounded and
annualized returns, annualized volatility, zero-risk-free-rate Sharpe ratio,
maximum drawdown, exposure, turnover, and costs.

## Regression pairs model with volatility regime

`build_screened_dataset.py` downloads the five screened pairs and `^VIX` from
2005 onward into `data/screened_pair_prices.csv` (pair prices, returns, and VIX)
and `data/vix.csv`.

`regression_model.py` defines `RegressionPairsTradingModel`, which fits OLS of
the next-day spread change on the spread z-score and its interactions with VIX,
each stock's rolling 20-day volatility, and the spread's rolling volatility.
A heuristic `regime` (+1 / 0 / -1 for VIX below / between / above the trailing
window's terciles) can be ignored (`regime_mode="none"`), used as a regression
input (`"input"`), used as a position gate of 0 / 0.5 / 1 (`"gate"`), or both
(`"both"`). Override `compute_regime` to try other regime definitions.

```powershell
python build_screened_dataset.py
python regression_model.py --transaction-cost-bps 5
```

This runs the walk-forward backtest for the naive baseline and every regime
mode and writes results to `backtest_results/regression/`.
