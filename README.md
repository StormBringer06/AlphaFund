# AlphaFund

A pairs-trading strategy on same-sector US large caps. The current model,
**V7**, combines five parts:

1. **Point-in-time pair selection:** each year trades the top 5 pairs from a
   screen run only on the ten years before it.
2. **Signal:** a vol-scaled mean-reversion signal on each pair's spread, with a
   VIX overlay and a low-volatility size boost.
3. **Exit classifier:** closes positions whose remaining P&L is predicted to be
   negative.
4. **VIX floor:** holds no positions while VIX is below 12.
5. **Dynamic leverage:** scales the whole book to an 8% volatility target.

## Results

Two separate walk-forward runs: **development 2008–2017** and **holdout
2018–2026** (to 2026-11-20). Every yearly choice (pairs, signal parameters, exit
model) uses only data from before that year.

| | Development 2008–2017 | **Holdout 2018–2026** |
|---|---|---|
| Sharpe, gross | 0.61 | **0.76** |
| Sharpe, net of 5 bps costs + 0.5%/yr borrow | 0.27 | **0.50** |
| Annual return, gross | 4.2% | 5.8% |
| Annual return, net | 1.7% | 3.7% |
| Volatility | 7.1% | 7.8% |
| Max drawdown, net | −15.1% | −16.9% |
| Average leverage | 3.4x | 3.4x |
| Days flat because VIX < 12 | 275 | 36 |

Net annual returns in the holdout: 2018 +1.7%, 2019 +4.9%, 2020 +5.0%,
2021 +1.7%, 2022 +19.9%, 2023 +4.6%, 2024 −4.0%, 2025 −7.7%, 2026 +8.5%
(to 20 Nov).

Outputs are written to `results/`: summaries, yearly returns and Sharpe, exit
decisions and the yearly signal parameters (daily results are written too but
not committed).

## Quick start

```bash
python -m pip install -r requirements.txt
python main.py                           # reproduce the table above (~15 s)
python -m pytest tests                   # unit tests
python -m tests.lookahead_check          # end-to-end look-ahead check (~3 min)
```

Useful options for `main.py`:

| Option | Effect |
|---|---|
| `--start-year`, `--end-year` | Run one period instead of development + holdout (data must cover it) |
| `--cost-bps`, `--borrow-bps` | Trading and short-borrow costs for the net results (default 5 and 50) |
| `--vix-floor` | VIX level below which the book is flat (default 12) |
| `--target-vol`, `--max-leverage` | Leverage settings (default 8% and 4x) |
| `--train-exit-model` | Refit every yearly exit model instead of loading `models/exit_ridge_weights.json` |

## How it works

### 1. Pairs (`pairs_model/universe.py`, `pairs_model/dataset.py`)

The universe is about 60 large caps in 11 sectors. Each test year Y, every
same-sector pair is screened on the ten years before Y. A pair passes if:

- daily return correlation is at least 0.6
- the Engle-Granger cointegration test gives p ≤ 0.05
- the rolling two-year hedge ratio stays at or above 0.2
- the spread's half-life is between 5 and 75 days

The top 5 passing pairs are picked, with no ticker reused and at most two per
sector (2008–2011 have only 1–4 passing pairs). The screen never sees year Y's
data.

### 2. Signal (`pairs_model/signal.py`)

1. **Spread:** log A minus a hedge ratio times log B. The hedge ratio is an OLS
   fit on the prior two years, refit every six months.
2. **z-score:** the spread against its 120-day EWMA mean and std, lagged one day.
3. **Target position:** `clip(-z/2, -1, 1)` times two multipliers, capped at 1:
   - **VIX overlay:** sizes up when VIX is high relative to its last two years.
   - **Low-volatility boost:** `clip(reference vol / spread vol, 1, 1.5)`, where
     spread vol is the 60-day std of the pair's daily spread return and the
     reference is its median over the previous 504 days (both lagged one day).
     A pair is sized up, never down, while its spread is calmer than usual.
4. **No-trade band:** the position only changes when the target moves by more
   than 0.2.

The VIX overlay strength (0 or 0.25) is re-chosen each year by an inner
walk-forward on the four years before it, run on the signal without the boost.

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

The classifier is retrained **once a year**: the model for year Y is fitted on
check-ups from episodes, on every same-sector pair in the universe, that
**finished before 1 January Y**. It never trains on the strategy's own trades.
The yearly models (2008–2026) are saved as plain numbers in
`models/exit_ridge_weights.json` and loaded at run time. A missing year is
fitted automatically.

### 4. VIX floor (`pairs_model/vix_floor.py`)

After the exit classifier, every target is set to zero on any close where VIX
is below 12, so the book is flat the next day. Very calm markets have been poor
for the strategy, and the leverage would otherwise scale up into them.

### 5. Leverage (`pairs_model/leverage.py`)

All weights are scaled by `min(8% / trailing vol, 4)`. The trailing vol is the
60-day-half-life EWMA volatility of the unlevered portfolio (after exits and
the VIX floor) up to that close, applied from the next day. Leverage changes
are charged as turnover.

### Backtest conventions (`backtest.py`)

- **Timing:** each target is set at a close and earns the next day's return.
- **Costs:** trading cost is bps × change in weights. Borrow is charged on the
  short leg.
- **Year ends:** each year starts flat and closes every position at year end.
- **Portfolio:** the equal-weight average of the pairs.

## No look-ahead

- **Yearly choices:** pairs, signal parameters and the exit classifier for year
  Y use only data from before 1 January Y.
- **Daily inputs:** every input to a daily decision (z-score, both multipliers,
  the VIX floor, leverage) uses data up to that close.
- **Earnings timing:** a report counts from when the market saw it. A report
  before the open counts that day; one after the close counts the next day.
- **Unit tests** (`tests/`) check that features and model outputs are unchanged
  when future data is removed.
- **End-to-end check:** `python -m tests.lookahead_check` rebuilds everything
  from data cut off at the end of 2016, 2019 and 2021, and confirms every exit
  decision in those years is identical.

## Caveats

- **The holdout is not untouched.** The low-volatility boost, the VIX floor
  level (tried at 0, 10, 12 and 13) and the exit classifier's design were all
  compared on 2018–2026 results during development. The 0.76 / 0.50 holdout
  Sharpe is therefore optimistic; only data after 2026-11-20 is truly unseen.
- **Survivorship bias:** the universe is today's large caps, so the bias is
  reduced (pairs are re-screened yearly) but not removed.
- **Recent years are weak:** the strategy lost money in 2024 and 2025 (net
  −4.0% and −7.7%); most of the holdout's return comes from 2022 and 2026.
- **Leverage cost:** leverage raises return and volatility together; the net
  max drawdown is about −17%.
- **Saved exit weights:** `models/exit_ridge_weights.json` was fitted on
  universe prices starting in 2004. `--train-exit-model` refits on the
  committed prices (from 1998), which gives slightly different predictions
  but identical exit decisions and returns.

## Repository layout

```
main.py                      run V7
backtest.py                  walk-forward backtester
pairs_model/
  universe.py                stock universe and pair screen
  dataset.py                 builds data/ (yearly screen, pair prices, earnings)
  signal.py                  signal (VIX overlay, low-vol boost) and its yearly parameter search
  exit_classifier.py         exit classifier: features, training, decisions
  vix_floor.py               flat book while VIX is below the floor
  leverage.py                dynamic volatility-targeted leverage
  metrics.py                 portfolio statistics
models/exit_ridge_weights.json   yearly exit models (2008-2026)
data/                        universe prices, yearly pairs, pair prices, earnings
results/                     outputs of main.py
tests/                       unit tests and the end-to-end look-ahead check
pairs_trading_hac/           Newey-West hypothesis test (see its README)
```

`pairs_trading_hac/` is a separate, self-contained hypothesis-testing package:
a Newey-West test of whether one strategy's mean daily return beats another's.
See `pairs_trading_hac/README.md`.
