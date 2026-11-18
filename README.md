# AlphaFund

## Building the pairs-trading dataset

The dataset builder downloads daily, dividend- and split-adjusted closing prices
from Yahoo Finance with `yfinance` for these pairs:

| Pair | Yahoo Finance tickers |
|---|---|
| JPMorgan Chase / Bank of America | `JPM` / `BAC` |
| Coca-Cola / PepsiCo | `KO` / `PEP` |
| Barclays / Lloyds | `BARC.L` / `LLOY.L` |
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

## Plan

Can volatility be used as a signal for entry and exit conditions in pairs trading?

Motivation: Traditional pairs trading strategies use fixed entry and exit conditions, however, the risks and opportunities associated with such a strategy may significantly vary based on both the market volatility and the volatility of the spread.

Backtest on 5 stock pairs.

Quantify volatility using standardised VIX or rolling volatility

Metrics: Sharpe, annualised returns, return per trade for better model, frequency/number of trades for better model, Sortino ratio, maximum drawdown to quantify downside risk

Compare a naive model and a more advanced model:
- Naive model: Trading s.t. position proportional to Z score, ignoring fees. Dynamic hedge ratio (recalculate every 6 months?) since we will be examining long time periods. This doesn’t really have an entry threshold.
- Better model: Use an entry condition, function of Z and vol. Test with various entry thresholds. Idea: Use regime-dependent entry conditions, quantified by volatility as a piecewise function. Another idea: Z score is a function of volatility (stocks and spread?), VIX and a discrete input (-1, 0 or 1) which represents if volatility is in top 10% or bottom 10% over past 2 years (we will fit a linear function which may break down at upper and lower volatility ends, so we introduce this term)
- Use equivalent exit conditions for both models
- Compare log returns of models (which are normal) using a hypothesis test: null hypothesis that means are equal, alternative hypothesis that mean of better model > mean of naive model. 5% significance level.

Could add potential future improvements section?
