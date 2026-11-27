"""Dynamic leverage: scale the whole book to a volatility target.

Every pair weight is multiplied by ``min(target_vol / ewma_vol, max_leverage)``,
where ``ewma_vol`` is the EWMA volatility of the strategy's own *unlevered*
equal-weight portfolio returns up to that close (applied to the next day, like
any other target). The EWMA is seeded with the signal's inner walk-forward
returns for the two years before the first test year. Leverage changes count
as turnover.

In ``main.py`` the returns fed in are those after the exit classifier and the
VIX floor (the seed has the VIX floor applied), so leverage targets the
volatility of what is actually held.
"""

from __future__ import annotations

import math

import pandas as pd

from backtest import apply_positions_to_returns, run_walk_forward_backtest
from pairs_model.signal import VolScaledNaivePairsTradingModel


def portfolio_returns(daily: pd.DataFrame) -> pd.Series:
    """Equal-weight daily portfolio return (mean across pairs)."""
    return daily.groupby("date")["strategy_return"].mean().sort_index()


def dynamic_leverage(
    returns: pd.Series,
    target_vol: float,
    max_leverage: float,
    halflife: float,
    min_vol: float = 0.005,
) -> pd.Series:
    """Leverage from an EWMA vol of returns up to and including each date."""
    vol = returns.ewm(halflife=halflife, min_periods=20).std() * math.sqrt(252.0)
    return (target_vol / vol.clip(lower=min_vol)).clip(upper=max_leverage)


def lever(
    daily: pd.DataFrame,
    leverage: pd.Series,
    model_name: str,
    transaction_cost_bps: float,
) -> pd.DataFrame:
    """Scale every pair's target weights by that date's leverage and re-cost."""
    out = daily.copy()
    factor = out["date"].map(leverage)
    if factor.isna().any():
        raise ValueError("Leverage is missing for some dates.")
    out["leverage"] = factor
    out["target_weight_a"] = out["target_weight_a"] * factor
    out["target_weight_b"] = out["target_weight_b"] * factor
    out["model_name"] = model_name
    return apply_positions_to_returns(out, transaction_cost_bps=transaction_cost_bps)


def seed_returns(
    data: pd.DataFrame, selection: pd.DataFrame, params: dict, first_year: int,
    train_years: int, inner_test_years: int,
) -> pd.DataFrame:
    """The signal's inner walk-forward daily results for the years before ``first_year``."""
    pairs = selection.loc[selection["test_year"] == first_year, "pair_id"].unique()
    year_data = data.loc[data["pair_id"].isin(pairs)]
    training = year_data.loc[
        (year_data["date"] >= pd.Timestamp(year=first_year - train_years, month=1, day=1))
        & (year_data["date"] < pd.Timestamp(year=first_year, month=1, day=1))
    ]
    return run_walk_forward_backtest(
        training,
        model_factory=lambda: VolScaledNaivePairsTradingModel(**params),
        start_year=first_year - inner_test_years,
        end_year=first_year - 1,
        train_years=train_years - inner_test_years,
        transaction_cost_bps=0.0,
    )


def levered(daily: pd.DataFrame, seed: pd.DataFrame, name: str, cost: float,
            target_vol: float, max_leverage: float, vol_halflife: float) -> pd.DataFrame:
    """Apply dynamic leverage computed from ``daily``'s own unlevered returns."""
    history = pd.concat([
        portfolio_returns(apply_positions_to_returns(seed, transaction_cost_bps=cost)),
        portfolio_returns(daily),
    ]).sort_index()
    history = history[~history.index.duplicated(keep="last")]
    dates = pd.Index(sorted(daily["date"].unique()))
    leverage = dynamic_leverage(
        history, target_vol, max_leverage, vol_halflife
    ).reindex(dates)
    if leverage.isna().any():
        raise ValueError("Leverage is unavailable for some test dates.")
    return lever(daily, leverage, name, cost)


def with_borrow(daily: pd.DataFrame, borrow_bps: float) -> pd.DataFrame:
    out = daily.copy()
    borrow = out["position_gross_exposure"] * 0.5 * borrow_bps / 10_000.0 / 252.0
    out["strategy_return"] = out["strategy_return"] - borrow
    return out
