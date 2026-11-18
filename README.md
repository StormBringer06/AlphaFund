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

Does waiting for a larger price divergence improve pairs trading returns, and does this depend on the volatility of the market?

Backtest on 5 stock pairs:
JPM, BAC
KO, PEP
XOM, CUX
V, MA
WMT, CST

Quantify volatility using standardised VIX or rolling volatility

Metrics: Sharpe, annualised returns, return per trade for better model, frequency/number of trades for better model, Sortino ratio, maximum drawdown to quantify downside risk

Compare a naive model and a more advanced model:
- Naive model: Trading s.t. position proportional to Z score, ignoring fees. Dynamic hedge ratio (recalculate every 6 months?) since we will be examining long time periods. This doesn’t really have an entry threshold.
- Better model: Use an entry condition, function of Z and vol. Test with various entry thresholds. Idea: Use regime-dependent entry conditions, quantified by volatility as a piecewise function.
- Use equivalent exit conditions for both models
- Compare log returns of models (which are normal) using a hypothesis test: null hypothesis that means are equal, alternative hypothesis that mean of better model > mean of naive model. 5% significance level.
