import math
import unittest

import pandas as pd

from base_model import NaivePairsTradingModel


def make_pair_data(
    dates: pd.DatetimeIndex,
    log_price_b: list[float],
    residual: list[float],
) -> pd.DataFrame:
    intercept = 0.5
    hedge_ratio = 1.2
    return pd.DataFrame(
        {
            "date": dates,
            "pair_id": "jpm_bac",
            "ticker_a": "JPM",
            "ticker_b": "BAC",
            "adjusted_close_a": [
                math.exp(intercept + hedge_ratio * log_b + error)
                for log_b, error in zip(log_price_b, residual)
            ],
            "adjusted_close_b": [math.exp(value) for value in log_price_b],
        }
    )


class NaivePairsTradingModelTests(unittest.TestCase):
    def setUp(self):
        train_dates = pd.date_range("2024-01-02", periods=40, freq="D")
        train_log_b = [4.0 + (index % 20) * 0.01 for index in range(40)]
        train_residual = [0.01] * 20 + [-0.01] * 20
        self.train = make_pair_data(train_dates, train_log_b, train_residual)

        test_dates = pd.date_range("2024-02-11", periods=3, freq="D")
        test_log_b = [4.25, 4.35, 4.45]
        test_residual = [-0.005, 0.04, -0.005]
        self.test = make_pair_data(test_dates, test_log_b, test_residual)

    def test_fit_predict_outputs_bounded_mean_reversion_weights(self):
        predictions = NaivePairsTradingModel().fit(self.train).predict(self.test)

        self.assertAlmostEqual(predictions["hedge_ratio"].iloc[0], 1.2)
        self.assertLess(predictions["spread_zscore"].iloc[0], 0)
        self.assertGreater(predictions["spread_zscore"].iloc[1], 0)
        self.assertLess(predictions["spread_zscore"].iloc[2], 0)

        self.assertGreater(predictions["target_weight_a"].iloc[0], 0)
        self.assertLess(predictions["target_weight_b"].iloc[0], 0)
        self.assertLess(predictions["target_weight_a"].iloc[1], 0)
        self.assertGreater(predictions["target_weight_b"].iloc[1], 0)
        self.assertTrue((predictions["target_gross_exposure"] <= 1.0).all())
        self.assertGreater(predictions["target_gross_exposure"].iloc[0], 0)

    def test_rejects_test_dates_that_overlap_training_period(self):
        model = NaivePairsTradingModel().fit(self.train)

        with self.assertRaisesRegex(ValueError, "strictly after"):
            model.predict(self.train.tail(1))

    def test_rejects_test_data_for_a_different_pair(self):
        model = NaivePairsTradingModel().fit(self.train)
        wrong_pair = self.test.assign(pair_id="ko_pep")

        with self.assertRaisesRegex(ValueError, "must match the fitted pair"):
            model.predict(wrong_pair)


if __name__ == "__main__":
    unittest.main()
