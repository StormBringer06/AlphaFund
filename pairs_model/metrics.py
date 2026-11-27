"""Portfolio statistics for daily backtest results (equal weight across pairs)."""

from __future__ import annotations

import math

import pandas as pd


def portfolio_metrics(daily: pd.DataFrame) -> pd.DataFrame:
    """Equal-weight daily portfolio across pairs, summarized per model."""
    rows = []
    for model_name, group in daily.groupby("model_name", sort=False):
        returns = group.groupby("date")["strategy_return"].mean()
        vol = float(returns.std())
        wealth = (1.0 + returns).cumprod()
        years = len(returns) / 252.0
        rows.append(
            {
                "model_name": model_name,
                "annualized_return": float(wealth.iloc[-1] ** (1 / years) - 1),
                "annualized_volatility": vol * math.sqrt(252.0),
                "sharpe_ratio": float(returns.mean() / vol * math.sqrt(252.0)),
                "max_drawdown": float((wealth / wealth.cummax() - 1).min()),
                "average_gross_exposure": float(group["position_gross_exposure"].mean()),
                "average_daily_turnover": float(group["turnover"].mean()),
            }
        )
    return pd.DataFrame(rows)
