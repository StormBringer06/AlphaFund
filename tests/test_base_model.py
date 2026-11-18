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

    def test_refits_beta_every_six_months_using_a_trailing_two_year_window(self):
        train_dates = pd.date_range("2021-01-01", periods=1200, freq="D")
        train_log_b = [4.0 + (index % 10) * 0.01 for index in range(1200)]
        train_residual = [
            0.01 if (index // 10) % 2 == 0 else -0.01
            for index in range(1200)
        ]
        train = make_pair_data(train_dates, train_log_b, train_residual)
        train["adjusted_close_a"] = [
            math.exp(
                0.5
                + (2.0 if index < 400 else 1.2) * log_b
                + residual
            )
            for index, (log_b, residual) in enumerate(
                zip(train_log_b, train_residual)
            )
        ]

        test_dates = pd.date_range(train_dates[-1] + pd.Timedelta(days=1), periods=250)
        test_log_b = [4.0 + ((index + 1200) % 10) * 0.01 for index in range(250)]
        test_residual = [
            0.01 if (index // 10) % 2 == 0 else -0.01
            for index in range(250)
        ]
        test = make_pair_data(test_dates, test_log_b, test_residual)
        test["adjusted_close_a"] = [
            math.exp(
                0.5
                + (1.2 if index < 90 else 2.0) * log_b
                + residual
            )
            for index, (log_b, residual) in enumerate(
                zip(test_log_b, test_residual)
            )
        ]

        predictions = NaivePairsTradingModel().fit(train).predict(test)
        scheduled_refit = train_dates[-1] + pd.DateOffset(months=6)
        refit_rows = predictions[predictions["beta_refit"]]

        self.assertAlmostEqual(predictions["hedge_ratio"].iloc[0], 1.2, places=2)
        self.assertEqual(len(refit_rows), 1)
        self.assertEqual(
            refit_rows["date"].iloc[0],
            test.loc[test["date"] >= scheduled_refit, "date"].iloc[0],
        )
        self.assertNotAlmostEqual(refit_rows["hedge_ratio"].iloc[0], 1.2, places=2)
        self.assertEqual(
            refit_rows["beta_refit_date"].iloc[0], refit_rows["date"].iloc[0]
        )

    def test_refits_are_consistent_when_test_data_arrives_in_chunks(self):
        one_batch = NaivePairsTradingModel().fit(self.train).predict(self.test)
        chunked_model = NaivePairsTradingModel().fit(self.train)
        first_chunk = chunked_model.predict(self.test.iloc[:2])
        second_chunk = chunked_model.predict(self.test.iloc[2:])
        chunked = pd.concat([first_chunk, second_chunk], ignore_index=True)

        pd.testing.assert_series_equal(
            one_batch["hedge_ratio"], chunked["hedge_ratio"], check_names=False
        )
        pd.testing.assert_series_equal(
            one_batch["beta_refit"], chunked["beta_refit"], check_names=False
        )


if __name__ == "__main__":
    unittest.main()
