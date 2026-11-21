"""No-look-ahead tests for improved_models.CiaranTestmodel (synthetic data)."""

import numpy as np
import pandas as pd

from improved_models.CiaranTestmodel import (
    SPY,
    VIX,
    Config,
    Context,
    Market,
    YearPlan,
    build_classifier,
    pair_features,
    rank_and_select,
    run_portfolio,
)


def synthetic_closes(n: int = 1600, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2010-01-01", periods=n)
    market = np.cumsum(rng.normal(0, 0.01, n))
    common = market + np.cumsum(rng.normal(0, 0.005, n))
    spread = np.zeros(n)
    for t in range(1, n):  # mean-reverting spread
        spread[t] = 0.97 * spread[t - 1] + rng.normal(0, 0.01)
    return pd.DataFrame({
        "AAA": 50 * np.exp(common + spread),
        "BBB": 40 * np.exp(common),
        VIX: 20 * np.exp(np.cumsum(rng.normal(0, 0.03, n)) * 0.1),
        SPY: 100 * np.exp(market),
    }, index=pd.DatetimeIndex(dates, name="date"))


def test_pair_features_are_causal():
    closes = synthetic_closes()
    cut = closes.index[1000]
    for hedge in ("rls", "ols6m"):
        full = pair_features(closes, "AAA", "BBB", hedge)
        part = pair_features(closes.loc[:cut], "AAA", "BBB", hedge)
        pd.testing.assert_frame_equal(full.loc[:cut], part, check_exact=False, rtol=0, atol=1e-12)


def test_market_statistics_are_causal():
    closes = synthetic_closes()
    cut = 1000
    full, part = Market(closes), Market(closes.iloc[: cut + 1])
    np.testing.assert_allclose(full.vix_pct[: cut + 1], part.vix_pct, atol=1e-12, equal_nan=True)
    np.testing.assert_allclose(full.stock_beta[: cut + 1], part.stock_beta, atol=1e-12, equal_nan=True)


def test_classifier_only_uses_trades_resolved_before_year():
    rng = np.random.default_rng(1)
    n = 200
    trades = pd.DataFrame({
        "exit_date": pd.date_range("2014-06-01", periods=n, freq="3D"),
        "pnl": rng.normal(0.001, 0.01, n),
        **{f: rng.normal(size=n) for f in (
            "vix_pct", "log_vol_ratio", "abs_z", "div_speed", "jump5", "spy_ret20",
            "half_life", "window_coint_rate", "log_recent_p")},
    })
    clf = build_classifier(Config(clf_min_trades=20), trades, 2015)
    assert clf.n == int((trades["exit_date"] < "2015-01-01").sum())
    # Changing outcomes of trades that resolve later must not change the model.
    later = trades["exit_date"] >= "2015-01-01"
    altered = trades.copy()
    altered.loc[later, "pnl"] = -1.0
    clf2 = build_classifier(Config(clf_min_trades=20), altered, 2015)
    assert clf.coef == clf2.coef


def test_ranking_ignores_current_and_future_performance():
    rng = np.random.default_rng(2)
    rows = []
    for year in range(2010, 2016):
        for i in range(12):
            rows.append({
                "test_year": year, "pair_id": f"p{i}", "sector": f"s{i}",
                "ticker_a": f"A{i}", "ticker_b": f"B{i}", "passes": True,
                "full_p": rng.uniform(0, 0.05), "window_coint_rate": rng.uniform(),
                **{m: rng.normal() for m in (
                    "return_correlation", "log_full_p", "log_recent_p", "half_life",
                    "model_half_life", "insample_sharpe", "beta_stability")},
            })
    screens = pd.DataFrame(rows)
    perf = screens[["test_year", "pair_id"]].assign(pair_sharpe=rng.normal(size=len(screens)))
    chosen, _ = rank_and_select(Config(n_pairs=5), screens, perf, 2014)
    perf_future = perf.copy()
    perf_future.loc[perf_future["test_year"] >= 2014, "pair_sharpe"] = rng.normal(size=24)
    chosen2, _ = rank_and_select(Config(n_pairs=5), screens, perf_future, 2014)
    assert chosen["pair_id"].tolist() == chosen2["pair_id"].tolist()


def test_execution_lag_and_truncation_invariance():
    closes = synthetic_closes()
    pairs = pd.DataFrame([{
        "pair_id": "aaa_bbb", "ticker_a": "AAA", "ticker_b": "BBB", "sector": "x",
        "half_life": 20.0, "window_coint_rate": 0.5, "recent_p": 0.01, "log_recent_p": -2.0,
    }])
    plans = {y: YearPlan(y, pairs) for y in (2012, 2013, 2014, 2015)}
    cfg = Config(classifier=False, beta_hedge=False)
    full = run_portfolio(Context(closes), cfg, plans, 2012, 2015)
    cut = pd.Timestamp("2014-03-31")
    part = run_portfolio(Context(closes.loc[:cut]), cfg, plans, 2012, 2015)
    n = len(part["daily"])
    pd.testing.assert_frame_equal(full["daily"].iloc[:n].reset_index(drop=True),
                                  part["daily"].reset_index(drop=True))
    # Lag 1: day t's P&L uses targets set at close t-2 (position since t-1).
    lag0 = run_portfolio(Context(closes), Config(classifier=False, beta_hedge=False, lag=0),
                         plans, 2012, 2015)
    g0, g1 = lag0["daily"]["gross_return"].to_numpy(), full["daily"]["gross_return"].to_numpy()
    assert not np.allclose(g0, g1)
    assert np.abs(full["daily"]["gross_exposure"].to_numpy()[1:]
                  - lag0["daily"]["gross_exposure"].to_numpy()[:-1]).max() < 1e-12
