import unittest

import numpy as np
import pandas as pd

from backtest import run_walk_forward_backtest
from improved_models.regime_regression_model import RegimeRegressionPairsTradingModel
from test_regression_model import make_data


class RegimeRegressionPairsTradingModelTests(unittest.TestCase):
    def setUp(self):
        self.data = make_data()
        split = pd.Timestamp("2018-01-01")
        self.train = self.data[self.data["date"] < split]
        self.test = self.data[self.data["date"] >= split]

    def test_predictions_do_not_use_future_data(self):
        full = RegimeRegressionPairsTradingModel().fit(self.train).predict(self.test)
        partial = (
            RegimeRegressionPairsTradingModel()
            .fit(self.train)
            .predict(self.test.iloc[:200])  # crosses the six-month refit
        )
        self.assertTrue(full["beta_refit"].iloc[:200].any())
        pd.testing.assert_frame_equal(
            full.iloc[:200][["spread_zscore", "target_weight_a", "target_weight_b"]],
            partial[["spread_zscore", "target_weight_a", "target_weight_b"]],
        )

    def test_ewma_zscore_is_lagged_and_calibrated(self):
        model = RegimeRegressionPairsTradingModel().fit(self.train)
        out = model.predict(self.test)
        self.assertFalse(out["spread_zscore"].isna().any())
        self.assertAlmostEqual(out["spread_zscore"].std(), 1.0, delta=0.4)
        # Changing today's price must not change the standardization used today.
        bumped = self.test.copy()
        bumped.iloc[-1, bumped.columns.get_loc("adjusted_close_a")] *= 1.05
        model2 = RegimeRegressionPairsTradingModel().fit(self.train)
        out2 = model2.predict(bumped)
        spread_jump = out2["spread"].iloc[-1] - out["spread"].iloc[-1]
        z_jump = out2["spread_zscore"].iloc[-1] - out["spread_zscore"].iloc[-1]
        implied_std = spread_jump / z_jump
        prev = out["spread"].iloc[:-1]
        self.assertGreater(implied_std, 0)
        self.assertLess(implied_std, prev.std() * 3)

    def test_extreme_regime_is_discrete_and_in_regression(self):
        model = RegimeRegressionPairsTradingModel().fit(self.train)
        out = model.predict(self.test)
        self.assertTrue(set(out["vix_extreme"]).issubset({-1, 0, 1}))
        self.assertIn(-1, set(out["vix_extreme"]))
        # Intercept, z, 4 vol interactions, regime (2), extreme (2).
        self.assertEqual(len(model._params.coefficients), 10)
        base = RegimeRegressionPairsTradingModel(use_extreme_regime=False)
        self.assertEqual(len(base.fit(self.train)._params.coefficients), 8)

    def test_no_gating(self):
        out = RegimeRegressionPairsTradingModel().fit(self.train).predict(self.test)
        stressed = out[out["regime"] == -1]
        self.assertTrue((stressed["target_gross_exposure"] > 0).any())
        self.assertTrue((out["target_gross_exposure"] <= 1 + 1e-12).all())

    def test_runs_in_walk_forward_backtest(self):
        daily = run_walk_forward_backtest(
            self.data,
            model_factory=RegimeRegressionPairsTradingModel,
            start_year=2018,
            end_year=2020,
            train_years=3,
        )
        self.assertFalse(daily["strategy_return"].isna().any())


if __name__ == "__main__":
    unittest.main()
