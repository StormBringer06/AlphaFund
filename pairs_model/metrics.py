"""Portfolio statistics for daily backtest results (equal weight across pairs)."""

from __future__ import annotations

import math

import numpy as np
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


def _year_bootstrap(
    returns: pd.DataFrame, models: list[str], baseline: str, draws: int = 2000
) -> pd.DataFrame:
    """Resample whole years; Sharpe CI per model and of its gap to baseline."""
    rng = np.random.default_rng(0)
    years = returns.index.year.unique().to_numpy()
    by_year = {y: returns.loc[returns.index.year == y] for y in years}
    samples = {m: [] for m in models}
    for _ in range(draws):
        sample = pd.concat([by_year[y] for y in rng.choice(years, len(years))])
        sharpe = sample.mean() / sample.std() * math.sqrt(252.0)
        for m in models:
            samples[m].append(sharpe[m])
    rows = []
    for m in models:
        s = np.array(samples[m])
        gap = s - np.array(samples[baseline])
        rows.append({
            "model_name": m,
            "sharpe_p05": np.quantile(s, 0.05),
            "sharpe_p95": np.quantile(s, 0.95),
            "gap_vs_baseline_p05": np.quantile(gap, 0.05),
            "gap_vs_baseline_p95": np.quantile(gap, 0.95),
            "prob_beats_baseline": float((gap > 0).mean()),
        })
    return pd.DataFrame(rows)


def robustness_table(daily: pd.DataFrame, baseline: str = "v3") -> pd.DataFrame:
    returns = daily.pivot_table(
        index="date", columns="model_name", values="strategy_return", aggfunc="mean"
    )
    models = list(returns.columns)
    ann = math.sqrt(252.0)
    sharpe = lambda r: r.mean() / r.std() * ann
    calm = returns.loc[~returns.index.year.isin([2020, 2022])]
    target_vol = returns[baseline].std()
    table = pd.DataFrame({
        "model_name": models,
        "sharpe": [sharpe(returns[m]) for m in models],
        "sharpe_ex_2020_2022": [sharpe(calm[m]) for m in models],
        "ann_return_at_baseline_vol": [
            returns[m].mean() * target_vol / returns[m].std() * 252 for m in models
        ],
    })
    return table.merge(_year_bootstrap(returns, models, baseline), on="model_name")
