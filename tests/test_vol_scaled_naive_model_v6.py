import unittest

import numpy as np
import pandas as pd

from improved_models.vol_scaled_naive_model import VolScaledNaivePairsTradingModel
from improved_models.vol_scaled_naive_model_v6 import (
    VolScaledNaivePairsTradingModelV6,
    add_pair_vol_features,
    select_low_vol_boost_exponent,
    select_low_vol_boost_vix_scale,
)
from test_regression_model import make_data


class VolScaledNaiveModelV6Tests(unittest.TestCase):
    def setUp(self):
        data = make_data()
        self.data = data
        split = pd.Timestamp("2018-01-01")
        self.train = data[data["date"] < split]
        self.test = data[data["date"] >= split]

    def test_zero_pair_vol_scale_preserves_v5_positions(self):
        kwargs = dict(vix_scale=0.25, band=0.2, band_vix_scale=0.0)
        v5 = VolScaledNaivePairsTradingModel(**kwargs).fit(self.train).predict(self.test)
        v6 = VolScaledNaivePairsTradingModelV6(
            **kwargs, pair_vol_scale=0.0
        ).fit(self.train).predict(self.test)
        columns = [
            "spread_zscore",
            "vix_multiplier",
            "band_width",
            "target_signal",
            "signal",
            "target_weight_a",
            "target_weight_b",
        ]
        pd.testing.assert_frame_equal(v5[columns], v6[columns])

    def test_pair_vol_features_do_not_use_future_prices(self):
        kwargs = dict(vix_scale=0.25, band=0.2, pair_vol_scale=0.25)
        full = VolScaledNaivePairsTradingModelV6(**kwargs).fit(self.train).predict(self.test)
        partial = VolScaledNaivePairsTradingModelV6(**kwargs).fit(self.train).predict(
            self.test.iloc[:200]
        )
        columns = ["pair_vol_ratio", "pair_vol_percentile", "target_signal"]
        pd.testing.assert_frame_equal(full.iloc[:200][columns], partial[columns])

    def test_signed_scale_moves_size_with_pair_volatility_rank(self):
        predictions = {}
        for scale in (0.5, -0.5):
            predictions[scale] = (
                VolScaledNaivePairsTradingModelV6(
                    vix_scale=0.0, band=0.0, pair_vol_scale=scale
                )
                .fit(self.train)
                .predict(self.test)
            )
        positive = predictions[0.5]
        negative = predictions[-0.5]
        high = positive["pair_vol_percentile"] > 0.8
        low = positive["pair_vol_percentile"] < 0.2
        self.assertTrue(high.any() and low.any())
        self.assertGreater(
            positive.loc[high, "pair_vol_multiplier"].mean(),
            positive.loc[low, "pair_vol_multiplier"].mean(),
        )
        self.assertGreater(
            negative.loc[low, "pair_vol_multiplier"].mean(),
            negative.loc[high, "pair_vol_multiplier"].mean(),
        )

    def test_risk_target_uses_lagged_inverse_forecast_volatility(self):
        kwargs = dict(
            vix_scale=0.25,
            band=0.2,
            pair_vol_mode="risk_target",
            risk_exponent=0.5,
        )
        full = VolScaledNaivePairsTradingModelV6(**kwargs).fit(self.train).predict(self.test)
        partial = VolScaledNaivePairsTradingModelV6(**kwargs).fit(self.train).predict(
            self.test.iloc[:200]
        )
        columns = [
            "pair_risk_per_unit",
            "pair_risk_reference",
            "pair_risk_multiplier",
            "target_signal",
        ]
        pd.testing.assert_frame_equal(full.iloc[:200][columns], partial[columns])

        valid = full["pair_risk_per_unit"].gt(0) & full["pair_risk_reference"].notna()
        expected = (
            full.loc[valid, "pair_risk_reference"]
            / full.loc[valid, "pair_risk_per_unit"]
        ).pow(0.5).clip(0.5, 1.5)
        np.testing.assert_allclose(
            full.loc[valid, "pair_risk_multiplier"], expected
        )

    def test_low_vol_boost_never_reduces_the_base_position(self):
        prediction = (
            VolScaledNaivePairsTradingModelV6(
                vix_scale=0.0,
                band=0.0,
                pair_vol_mode="low_vol_boost",
                risk_exponent=0.5,
            )
            .fit(self.train)
            .predict(self.test)
        )
        self.assertTrue((prediction["pair_vol_multiplier"] >= 1.0).all())
        self.assertTrue((prediction["pair_vol_multiplier"] <= 1.5).all())

    def test_low_vol_boost_respects_configured_cap(self):
        prediction = (
            VolScaledNaivePairsTradingModelV6(
                vix_scale=0.0,
                band=0.0,
                pair_vol_mode="low_vol_boost",
                risk_exponent=1.0,
                risk_multiplier_cap=1.75,
            )
            .fit(self.train)
            .predict(self.test)
        )
        self.assertTrue((prediction["pair_vol_multiplier"] <= 1.75).all())

    def test_low_vol_vix_tuning_ignores_post_test_year_data(self):
        settings = {"vix_scale": 0.0, "band": 0.2, "band_vix_scale": 0.0}
        kwargs = dict(
            train_years=4,
            inner_test_years=2,
            risk_exponent=1.0,
            risk_multiplier_cap=1.5,
        )
        selected, scores = select_low_vol_boost_vix_scale(
            self.data, 2021, settings, **kwargs
        )
        truncated = self.data.loc[self.data["date"] < "2021-01-01"]
        selected_truncated, scores_truncated = select_low_vol_boost_vix_scale(
            truncated, 2021, settings, **kwargs
        )
        self.assertIn(selected, (0.0, 0.25, 0.5))
        self.assertEqual(selected, selected_truncated)
        pd.testing.assert_frame_equal(scores, scores_truncated)

    def test_low_vol_exponent_tuning_ignores_post_test_year_data(self):
        settings = {"vix_scale": 0.0, "band": 0.2, "band_vix_scale": 0.0}
        kwargs = dict(
            train_years=4,
            inner_test_years=2,
            risk_multiplier_cap=1.5,
        )
        selected, scores = select_low_vol_boost_exponent(
            self.data, 2021, settings, **kwargs
        )
        truncated = self.data.loc[self.data["date"] < "2021-01-01"]
        selected_truncated, scores_truncated = select_low_vol_boost_exponent(
            truncated, 2021, settings, **kwargs
        )
        self.assertIn(selected, (0.0, 0.25, 0.5, 1.0))
        self.assertEqual(selected, selected_truncated)
        pd.testing.assert_frame_equal(scores, scores_truncated)

    def test_pair_vol_ratio_uses_leg_and_spread_volatility(self):
        frame = pd.DataFrame(
            {
                "adjusted_close_a": np.exp(np.cumsum(np.linspace(-0.01, 0.01, 40))),
                "adjusted_close_b": np.exp(np.cumsum(np.linspace(0.01, -0.01, 40))),
            }
        )
        out = add_pair_vol_features(frame, hedge_ratio=1.0, vol_window=10, rank_window=20)
        self.assertIn("pair_vol_ratio", out)
        self.assertIn("pair_vol_percentile", out)
        self.assertTrue(out["pair_vol_ratio"].dropna().ge(0.0).all())

    def test_rejects_bad_pair_volatility_parameters(self):
        with self.assertRaises(ValueError):
            VolScaledNaivePairsTradingModelV6(pair_vol_scale=0.6)
        with self.assertRaises(ValueError):
            VolScaledNaivePairsTradingModelV6(pair_vol_window=60, pair_vol_rank_window=30)
        with self.assertRaises(ValueError):
            VolScaledNaivePairsTradingModelV6(pair_vol_mode="unknown")
        with self.assertRaises(ValueError):
            VolScaledNaivePairsTradingModelV6(
                pair_vol_mode="risk_target", pair_vol_scale=0.25
            )
        with self.assertRaises(ValueError):
            VolScaledNaivePairsTradingModelV6(
                pair_vol_mode="low_vol_boost", pair_vol_scale=0.25
            )
        with self.assertRaises(ValueError):
            VolScaledNaivePairsTradingModelV6(risk_exponent=1.5)
        with self.assertRaises(ValueError):
            VolScaledNaivePairsTradingModelV6(risk_multiplier_cap=2.5)


if __name__ == "__main__":
    unittest.main()
