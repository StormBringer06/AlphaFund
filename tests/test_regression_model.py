import unittest

import numpy as np
import pandas as pd

from backtest import run_walk_forward_backtest
from improved_models.dynamic_threshold_model import SharpeOptimizedThresholdModel
from improved_models.regression_model import RegressionPairsTradingModel


def make_data(start="2015-01-01", days=1600, seed=0) -> pd.DataFrame:
    """Synthetic cointegrated pair: A = 1.1 * B + mean-reverting spread."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, periods=days)
    log_b = 4.0 + np.cumsum(rng.normal(0, 0.01, days))
    spread = np.zeros(days)
    for i in range(1, days):
        spread[i] = 0.95 * spread[i - 1] + rng.normal(0, 0.005)
    log_a = 0.2 + 1.1 * log_b + spread
    vix = 15 + 10 * np.abs(np.sin(np.arange(days) / 50)) + rng.normal(0, 1, days)
    frame = pd.DataFrame(
        {
            "date": dates,
            "pair_id": "a_b",
            "adjusted_close_a": np.exp(log_a),
            "adjusted_close_b": np.exp(log_b),
            "vix": vix,
        }
    )
    frame["return_a"] = frame["adjusted_close_a"].pct_change()
    frame["return_b"] = frame["adjusted_close_b"].pct_change()
    return frame


class RegressionPairsTradingModelTests(unittest.TestCase):
    def setUp(self):
        self.data = make_data()
        split = pd.Timestamp("2018-01-01")
        self.train = self.data[self.data["date"] < split]
        self.test = self.data[self.data["date"] >= split]

    def test_predictions_do_not_use_future_data(self):
        full = RegressionPairsTradingModel().fit(self.train).predict(self.test)
        prefix = self.test.iloc[:200]  # crosses the six-month refit
        partial = RegressionPairsTradingModel().fit(self.train).predict(prefix)
        self.assertTrue(full["beta_refit"].iloc[:200].any())
        pd.testing.assert_frame_equal(
            full.iloc[:200][["target_weight_a", "target_weight_b"]],
            partial[["target_weight_a", "target_weight_b"]],
        )

    def test_weights_follow_forecast_and_hedge_ratio(self):
        out = RegressionPairsTradingModel(regime_mode="none").fit(self.train).predict(
            self.test
        )
        active = out[out["signal"] != 0]
        self.assertFalse(active.empty)
        np.testing.assert_array_equal(
            np.sign(active["target_weight_a"]), np.sign(active["predicted_spread_change"])
        )
        np.testing.assert_allclose(
            active["target_weight_b"], -active["hedge_ratio"] * active["target_weight_a"]
        )
        self.assertTrue((out["target_gross_exposure"] <= 1 + 1e-12).all())
        # Mean reversion: high spread z-scores should forecast a falling spread.
        self.assertLess(np.corrcoef(out["spread_zscore"], out["signal"])[0, 1], 0)

    def test_regime_gating(self):
        gate = RegressionPairsTradingModel(regime_mode="gate").fit(self.train)
        gate = gate.predict(self.test)
        both = RegressionPairsTradingModel(regime_mode="both").fit(self.train)
        both = both.predict(self.test)
        self.assertEqual(set(gate["regime"]), {-1, 0, 1})
        for out in (gate, both):
            self.assertTrue((out.loc[out["regime"] == -1, "target_weight_a"] == 0).all())
        neutral = gate[gate["regime"] == 0]
        self.assertLessEqual(neutral["target_gross_exposure"].max(), 0.5 + 1e-12)

    def test_requires_vix(self):
        with self.assertRaises(ValueError):
            RegressionPairsTradingModel().fit(self.train.drop(columns="vix"))

    def test_rejects_unknown_regime_mode(self):
        with self.assertRaises(ValueError):
            RegressionPairsTradingModel(regime_mode="bogus")

    def test_runs_in_walk_forward_backtest(self):
        daily = run_walk_forward_backtest(
            self.data,
            model_factory=RegressionPairsTradingModel,
            start_year=2019,
            end_year=2020,
            train_years=3,
        )
        self.assertEqual(set(daily["test_year"]), {2019, 2020})
        self.assertTrue(np.isfinite(daily["strategy_return"]).all())


class SharpeOptimizedThresholdModelTests(unittest.TestCase):
    def test_thresholds_are_dynamic_and_exit_stays_below_entry(self):
        design = np.array([[1.0, -1.0, 0.0], [1.0, 1.0, 1.0]])
        coefficients = np.array([2.0, 0.3, 0.2, 0.5, 0.0, 0.0])

        entry, exit_threshold = (
            SharpeOptimizedThresholdModel._predict_thresholds(
                design, coefficients
            )
        )

        self.assertNotEqual(entry[0], entry[1])
        self.assertTrue((entry >= 0.5).all())
        self.assertTrue((entry <= 2.75).all())
        self.assertTrue((exit_threshold >= 0.0).all())
        self.assertTrue((exit_threshold < entry).all())

    def test_positions_scale_with_divergence_and_exit_on_reversion(self):
        model = SharpeOptimizedThresholdModel(max_gross_exposure=1.0)
        target_a, target_b, entries, exits = model._make_targets(
            zscore=np.array([2.0, 2.5, 1.5, 0.4, -2.5]),
            entry=np.full(5, 2.0),
            exit_threshold=np.full(5, 0.5),
            hedge_ratio=1.0,
            initial_side=0,
        )
        gross = np.abs(target_a) + np.abs(target_b)

        np.testing.assert_allclose(gross, [0.6, 0.8, 0.4, 0.0, 0.8])
        np.testing.assert_array_equal(entries, [True, False, False, False, True])
        np.testing.assert_array_equal(exits, [False, False, False, True, False])
        self.assertLess(target_a[0], 0.0)
        self.assertGreater(target_a[-1], 0.0)
        np.testing.assert_allclose(target_b, -target_a)

    def test_fits_and_predicts_dynamic_thresholds(self):
        data = make_data()
        train = data[data["date"] < "2018-01-01"]
        test = data[data["date"] >= "2018-01-01"]
        model = SharpeOptimizedThresholdModel(
            lookback_years=1,
            min_train_observations=60,
            optimizer_maxiter=1,
            optimizer_popsize=4,
        ).fit(train)

        predictions = model.predict(test)

        self.assertEqual(len(predictions), len(test))
        self.assertTrue(
            {
                "entry_z_threshold",
                "exit_z_threshold",
                "trade_entry",
                "trade_exit",
                "target_gross_exposure",
            }.issubset(predictions.columns)
        )
        self.assertTrue(
            (predictions["exit_z_threshold"] < predictions["entry_z_threshold"]).all()
        )
        self.assertTrue((predictions["target_gross_exposure"] <= 1.0 + 1e-12).all())
        self.assertTrue(np.isfinite(model._threshold_fit_sharpe))


if __name__ == "__main__":
    unittest.main()
