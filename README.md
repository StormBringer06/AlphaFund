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

`improved_models/regression_model.py` defines `RegressionPairsTradingModel`,
which fits OLS of the next-day spread change on the spread z-score and its
interactions with VIX, each stock's rolling 20-day volatility, and the spread's
rolling volatility.
A heuristic `regime` (+1 / 0 / -1 for VIX below / between / above the trailing
window's terciles) can be ignored (`regime_mode="none"`), used as a regression
input (`"input"`), used as a position gate of 0 / 0.5 / 1 (`"gate"`), or both
(`"both"`). Override `compute_regime` to try other regime definitions.

```powershell
python build_screened_dataset.py
python -m improved_models.regression_model --transaction-cost-bps 5
```

This runs the walk-forward backtest for the naive baseline and every regime
mode and writes results to `backtest_results/regression/`.

## Sharpe-optimized dynamic-threshold model

`improved_models/dynamic_threshold_model.py` defines
`SharpeOptimizedThresholdModel`. For each training fold it fits linear entry
and exit threshold surfaces using standardized VIX and pair-volatility
features. Differential evolution selects the threshold coefficients that
maximize training-period net Sharpe (including the configured transaction
cost). At each test timestamp those surfaces produce dynamic z-score
thresholds. Entries fade the spread, and an active position scales with
divergence until reaching the configured gross exposure cap; it exits when the
spread returns within the exit threshold or crosses its mean. The hedge ratio
and rolling spread features continue to update every six months.

The default per-pair exposure cap is 1.0 gross, and full size is reached at an
absolute z-score of 3.0. Run the same 2014–2023 walk-forward on screened-pair
data with:

```powershell
python -m improved_models.dynamic_threshold_model --transaction-cost-bps 5
```

Outputs, including daily predicted thresholds and entry/exit diagnostics, are
written to `backtest_results/dynamic_threshold/`. The exposure cap applies to
each pair individually; portfolio-wide allocation is handled outside this
model.

## Regime regression with EWMA z-score

`improved_models/regime_regression_model.py` defines
`RegimeRegressionPairsTradingModel`, a subclass of the regression model with:

- **EWMA z-score**: `z_t = (spread_t - m_{t-1}) / s_{t-1}`, using an
  exponentially weighted mean and std of the spread (`zscore_halflife`, default
  120 trading days), lagged one day. The static window statistics gave test
  z-scores with std ~1.9 and a persistent bias; the EWMA version is ~1.05.
  `zscore_method="static"` restores the old behaviour.
- **Regimes as linear inputs only** (no gating): the VIX tercile `regime`
  (+1 / 0 / -1) and a `vix_extreme` state (+1 below the 10th / -1 above the
  90th percentile of the estimation window), each entered as a level term and
  an interaction with z. `use_extreme_regime=False` drops the extreme state.

```powershell
python -m improved_models.regime_regression_model --transaction-cost-bps 5
```

This compares the naive baseline, the original `lr_regime_input` model, the
new model and two ablations, writing results to
`backtest_results/regime_regression/`.

## Point-in-time pair selection

The screened pairs above were chosen with data overlapping the backtest.
`build_point_in_time_dataset.py` removes that look-ahead: for each test year it
runs the same screen on the ten years before that year and keeps the top five
pairs, using a cached download of the whole universe plus VIX.

```powershell
python build_point_in_time_dataset.py
```

It writes `data/universe_closes.csv` (cache), `data/pit_pair_selection.csv`
(pairs per test year) and `data/pit_pair_prices.csv` (prices for every pair
selected in any year). The candidate universe is still today's large caps, so
survivorship bias is reduced, not removed.

## Vol-scaled naive model

`improved_models/vol_scaled_naive_model.py` defines
`VolScaledNaivePairsTradingModel`, which keeps the naive sizing rule and adds:

1. **EWMA z-score**: `z = (spread - EWMA mean) / EWMA std`, with a 120-day
   half-life and statistics lagged one day.
2. **Naive sizing**: `clip(-z / 2, -1, 1)`.
3. **VIX size overlay**: the position is multiplied by
   `clip(1 + 2 * vix_scale * (vix_pct - 0.5), 0.5, 1.5)`, where `vix_pct` is
   VIX's percentile in the trailing 504 days. Gross exposure is capped at 1.
4. **No-trade band that depends on VIX**: the position only moves to the new
   target when the gap exceeds
   `band * clip(1 + 2 * band_vix_scale * (0.5 - vix_pct), 0, 2)`. The band is
   wider when VIX is low and narrower when it is high.

`vix_scale`, `band` and `band_vix_scale` are chosen separately for each test
year by a grid search on an inner walk-forward over the last two training
years, maximizing the net Sharpe of the portfolio of all pairs.

```powershell
python -m improved_models.vol_scaled_naive_model --transaction-cost-bps 5
python -m improved_models.vol_scaled_naive_model --input data/screened_pair_prices.csv --selection none --output-dir backtest_results/vol_scaled_naive/original_pairs
```

Each run compares `naive_zscore`, `naive_ewma_z` (EWMA z-score only) and
`vol_scaled_naive`. It writes daily results, summaries and the portfolio
comparison, plus `robustness_*.csv` (Sharpe excluding 2020 and 2022, return at
the naive model's volatility, and a bootstrap of the Sharpe gap to naive that
resamples whole years) and `selected_parameters_*.csv`.
