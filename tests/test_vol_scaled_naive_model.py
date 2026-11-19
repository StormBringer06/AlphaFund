import unittest

import numpy as np
import pandas as pd

from backtest import run_walk_forward_backtest
from improved_models.vol_scaled_naive_model import (
    VolScaledNaivePairsTradingModel as Model,
    select_parameters,
)
from test_regression_model import make_data


class VolScaledNaiveModelTests(unittest.TestCase):
    def setUp(self):
        self.data = make_data()
        split = pd.Timestamp("2018-01-01")
        self.train = self.data[self.data["date"] < split]
        self.test = self.data[self.data["date"] >= split]

    def test_predictions_do_not_use_future_data(self):
        kwargs = dict(vix_scale=0.5, band=0.1, band_vix_scale=0.5)
        full = Model(**kwargs).fit(self.train).predict(self.test)
        partial = Model(**kwargs).fit(self.train).predict(self.test.iloc[:200])
        self.assertTrue(full["beta_refit"].iloc[:200].any())
        cols = ["spread_zscore", "vix_pct", "target_weight_a", "target_weight_b"]
        pd.testing.assert_frame_equal(full.iloc[:200][cols], partial[cols])

    def test_defaults_match_naive_sizing_on_ewma_z(self):
        out = Model().fit(self.train).predict(self.test)
        expected = np.clip(-out["spread_zscore"] / 2.0, -1, 1)
        np.testing.assert_allclose(out["signal"], expected)
        np.testing.assert_allclose(
            out["target_weight_b"], -out["hedge_ratio"] * out["target_weight_a"]
        )
        self.assertAlmostEqual(out["spread_zscore"].std(), 1.0, delta=0.4)

    def test_vix_overlay_scales_with_percentile(self):
        out = Model(vix_scale=0.5).fit(self.train).predict(self.test)
        np.testing.assert_allclose(out["vix_multiplier"], 0.5 + out["vix_pct"])
        self.assertTrue((out["target_gross_exposure"] <= 1 + 1e-12).all())

    def test_band_reduces_trading_and_widens_when_calm(self):
        daily = Model().fit(self.train).predict(self.test)
        banded = Model(band=0.2, band_vix_scale=0.5).fit(self.train).predict(self.test)
        self.assertLess(banded["rebalanced"].sum(), len(banded) / 2)
        self.assertLess(
            np.abs(np.diff(banded["signal"])).sum(),
            np.abs(np.diff(daily["signal"])).sum(),
        )
        calm = banded["vix_pct"] < 0.25
        stressed = banded["vix_pct"] > 0.75
        self.assertGreater(
            banded.loc[calm, "band_width"].min(), banded.loc[stressed, "band_width"].max()
        )
        # Held position never strays from target by more than the band.
        gap = (banded["signal"] - banded["target_signal"]).abs()
        self.assertTrue((gap <= banded["band_width"] + 1e-12).all())

    def test_rejects_bad_parameters(self):
        with self.assertRaises(ValueError):
            Model(vix_scale=0.6)
        with self.assertRaises(ValueError):
            Model(band=-0.1)

    def test_select_parameters_uses_only_training_window(self):
        params, scores = select_parameters(
            self.data, test_year=2020, transaction_cost_bps=5.0, train_years=4
        )
        self.assertEqual(set(params), {"vix_scale", "band", "band_vix_scale"})
        self.assertEqual(len(scores), 3 * (1 + 4 * 2))
        # Changing data from the test year onward must not change the choice.
        altered = self.data.copy()
        late = altered["date"] >= pd.Timestamp("2020-01-01")
        altered.loc[late, "adjusted_close_a"] *= 1.3
        altered.loc[late, "vix"] *= 2
        params2, _ = select_parameters(
            altered, test_year=2020, transaction_cost_bps=5.0, train_years=4
        )
        self.assertEqual(params, params2)

    def test_runs_in_walk_forward_backtest(self):
        daily = run_walk_forward_backtest(
            self.data,
            model_factory=lambda: Model(vix_scale=0.25, band=0.1),
            start_year=2019,
            end_year=2020,
            train_years=4,
            transaction_cost_bps=5.0,
        )
        self.assertFalse(daily["strategy_return"].isna().any())


if __name__ == "__main__":
    unittest.main()
