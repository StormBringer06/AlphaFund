# AlphaFund

A pairs-trading strategy on same-sector US large caps. The current model,
**V5**, combines four parts:

1. **Point-in-time pair selection:** each year trades the top 5 pairs from a
   screen run only on the ten years before it.
2. **V3 signal:** a vol-scaled mean-reversion signal on each pair's spread.
3. **Exit classifier:** closes positions whose remaining P&L is predicted to be
   negative.
4. **Dynamic leverage:** scales the whole book to an 8% volatility target.

## Results (2014–2023, out of sample)

| | V3 (signal only) | **V5** |
|---|---|---|
| Sharpe, gross | 0.53 | **0.80** |
| Sharpe, net of 5 bps costs + 0.5%/yr borrow | 0.35 | **0.54** |
| Annual return, net | 1.1% | **3.6%** |
| Volatility | 3.2% | 7.0% |
| Max drawdown, net | −6.9% | −15.9% |
| Average leverage | 1.0x | 3.5x |

Net annual returns for V5: 2014 +3.4%, 2015 +4.9%, 2016 +2.5%, 2017 −7.1%,
2018 −0.3%, 2019 +6.4%, 2020 +3.3%, 2021 +1.0%, 2022 +19.9%, 2023 +3.6%.

Outputs are written to `results/`: summaries, yearly returns, exit decisions
and the yearly V3 parameters.

## Quick start

```bash
python -m pip install -r requirements.txt
python main.py                           # reproduce the table above (~10 s)
python -m pytest tests                   # unit tests
python -m tests.lookahead_check          # end-to-end look-ahead check (~3 min)
```

Useful options for `main.py`:

| Option | Effect |
|---|---|
| `--cost-bps`, `--borrow-bps` | Trading and short-borrow costs for the net results (default 5 and 50) |
| `--target-vol`, `--max-leverage` | Leverage settings (default 8% and 4x) |
| `--train-exit-model` | Refit every yearly exit model instead of loading `models/exit_ridge_weights.json` |
| `--freeze-exit-model YEAR` | Use one fixed exit model for every test year (test years must not be before YEAR, e.g. `2023` for 2024+) |
| `--start-year`, `--end-year` | Test period (data must cover it) |

## How it works

### 1. Pairs (`pairs_model/universe.py`, `pairs_model/dataset.py`)

The universe is about 60 large caps in 11 sectors. Each test year Y, every
same-sector pair is screened on the ten years before Y. A pair passes if:

- daily return correlation is at least 0.6
- the Engle-Granger cointegration test gives p ≤ 0.05
- the rolling two-year hedge ratio stays at or above 0.2
- the spread's half-life is between 5 and 75 days

The top 5 passing pairs are picked, with no ticker reused and at most two per
sector. The screen never sees year Y's data.

### 2. Signal: V3 (`pairs_model/signal.py`)

1. **Spread:** log A minus a hedge ratio times log B. The hedge ratio is an OLS
   fit on the prior two years, refit every six months.
2. **z-score:** the spread against its 120-day EWMA mean and std, lagged one day.
3. **Target position:** `clip(-z/2, -1, 1)`, times a VIX overlay that sizes up
   when VIX is high relative to its last two years.
4. **No-trade band:** the position only changes when the target moves by more
   than 0.2.

The VIX overlay strength is re-chosen each year on prior data only.

### 3. Exit classifier (`pairs_model/exit_classifier.py`)

An *episode* is a run of same-direction position in one pair. At days 5, 10,
20, 40, 60, 90 and 120 of an open episode, a ridge regression predicts the
episode's **remaining** P&L from 13 inputs known at that close:

| Group | Inputs |
|---|---|
| Where the spread is | \|z\| now; how much \|z\| has widened since entry |
| How the trade is going | days held; P&L so far |
| Is the relationship breaking | hedge-ratio shift since entry; spread trend since entry |
| Fundamentals | change in the two stocks' TTM-EPS ratio since entry; earnings surprises that agree with the divergence; number of reports |
| Market state | VIX percentile and its change since entry; spread vol ratio (20d / 120d); half-life ratio |

If the prediction is below zero, the position is closed and stays flat until
the spread crosses zero.

In practice:

- **What pushes it to hold:** a spread stretched far from fair value, high VIX
  and sharp spikes, all signs the spread should snap back.
- **What pushes it to exit:** a gap that keeps widening and a steady trend
  against the position, signs the relationship has broken.

The classifier is retrained **once a year**: the model for year Y is fitted on
check-ups from episodes, on every same-sector pair in the universe, that
**finished before 1 January Y**. It never trains on the strategy's own trades.
The ten yearly models (2014–2023) are saved as plain numbers in
`models/exit_ridge_weights.json` and loaded at run time. A missing year is
fitted automatically.

### 4. Leverage (`pairs_model/leverage.py`)

All weights are scaled by `min(8% / trailing vol, 4)`. The trailing vol is the
60-day-half-life EWMA volatility of the unlevered portfolio up to that close,
applied from the next day. Leverage changes are charged as turnover.

### Backtest conventions (`backtest.py`)

- **Timing:** each target is set at a close and earns the next day's return.
- **Costs:** trading cost is bps × change in weights. Borrow is charged on the
  short leg.
- **Year ends:** each year starts flat and closes every position at year end.
- **Portfolio:** the equal-weight average of the pairs.

## No look-ahead

- **Pair choice:** pairs, V3 parameters and the exit classifier for year Y use
  only data from before 1 January Y.
- **Daily inputs:** every input to a daily decision uses data up to that close.
- **Earnings timing:** a report counts from when the market saw it. A report
  before the open counts that day; one after the close counts the next day.
- **Unit tests** (`tests/`) check that features and model outputs are unchanged
  when future data is removed.
- **End-to-end check:** `python -m tests.lookahead_check` rebuilds everything
  from data cut off at the end of 2016, 2019 and 2021, and confirms every exit
  decision in those years is identical.

## Caveats

- **Survivorship bias:** the universe is today's large caps, so the bias is
  reduced (pairs are re-screened yearly) but not removed.
- **Not yet confirmed on unseen data:** the classifier design was chosen while
  looking at 2014–2023, so the untouched **2024–2025 holdout** is the real test.
  To run it, extend the data with
  `python -m pairs_model.dataset --end-year 2025 --refresh --earnings`, then run
  `python main.py --start-year 2024 --end-year 2025`. The 2024 and 2025 exit
  models are fitted automatically from data before each year; add
  `--freeze-exit-model 2023` to use only the existing 2023 model.
- **Leverage cost:** leverage raises return and volatility together; the net
  max drawdown is about −16%.

## Repository layout

```
main.py                      run V5
backtest.py                  walk-forward backtester
pairs_model/
  universe.py                stock universe and pair screen
  dataset.py                 builds data/ (yearly screen, pair prices, earnings)
  signal.py                  V3 signal and its yearly parameter search
  exit_classifier.py         exit classifier: features, training, decisions
  leverage.py                dynamic volatility-targeted leverage
  metrics.py                 portfolio statistics
models/exit_ridge_weights.json   yearly exit models (2014-2023)
data/                        universe prices, yearly pairs, pair prices, earnings
results/                     outputs of main.py
tests/                       unit tests and the end-to-end look-ahead check
pairs_trading_hac/           Newey-West hypothesis test (see its README)
```

`pairs_trading_hac/` is a separate, self-contained hypothesis-testing package:
a Newey-West test of whether one strategy's mean daily return beats another's.
See `pairs_trading_hac/README.md`.
