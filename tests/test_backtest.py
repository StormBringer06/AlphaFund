import unittest

import pandas as pd

from backtest import (
    apply_positions_to_returns,
    build_walk_forward_folds,
    run_walk_forward_backtest,
    summarize_backtest,
)


class ConstantWeightModel:
    def fit(self, train_data):
        self.pair_id = train_data["pair_id"].iloc[0]
        return self

    def predict(self, test_data):
        predictions = test_data.copy()
        predictions["target_weight_a"] = 0.5
        predictions["target_weight_b"] = -0.5
        return predictions


class WalkForwardFoldTests(unittest.TestCase):
    def test_builds_ten_calendar_year_folds_with_prior_four_years(self):
        folds = build_walk_forward_folds()

        self.assertEqual(len(folds), 10)
        self.assertEqual(folds[0].train_start, pd.Timestamp("2010-01-01"))
        self.assertEqual(folds[0].test_start, pd.Timestamp("2014-01-01"))
        self.assertEqual(folds[0].test_end, pd.Timestamp("2015-01-01"))
        self.assertEqual(folds[-1].train_start, pd.Timestamp("2019-01-01"))
        self.assertEqual(folds[-1].test_start, pd.Timestamp("2023-01-01"))
        self.assertEqual(folds[-1].test_end, pd.Timestamp("2024-01-01"))


class PositionApplicationTests(unittest.TestCase):
    def test_lags_targets_and_charges_costs_and_fold_liquidation(self):
        predictions = pd.DataFrame(
            {
                "date": pd.date_range("2024-01-02", periods=3, freq="D"),
                "pair_id": "jpm_bac",
                "fold_id": 1,
                "return_a": [0.1, 0.1, 0.1],
                "return_b": [0.0, 0.0, 0.0],
                "target_weight_a": [0.5, -0.5, 0.0],
                "target_weight_b": [0.0, 0.0, 0.0],
            }
        )

        result = apply_positions_to_returns(predictions, transaction_cost_bps=100)

        self.assertEqual(list(result["position_weight_a"]), [0.0, 0.5, -0.5])
        self.assertAlmostEqual(result["strategy_return"].iloc[0], -0.005)
        self.assertAlmostEqual(result["strategy_return"].iloc[1], 0.04)
        self.assertAlmostEqual(result["strategy_return"].iloc[2], -0.055)


class WalkForwardBacktestTests(unittest.TestCase):
    def test_runner_uses_model_factory_and_returns_fold_summary(self):
        dates = pd.bdate_range("2010-01-01", "2015-01-01")
        data = pd.DataFrame(
            {
                "date": dates,
                "pair_id": "jpm_bac",
                "adjusted_close_a": [100.0 + index for index in range(len(dates))],
                "adjusted_close_b": [50.0 + index for index in range(len(dates))],
                "return_a": 0.001,
                "return_b": -0.0005,
            }
        )

        daily = run_walk_forward_backtest(
            data,
            model_factory=ConstantWeightModel,
            start_year=2014,
            end_year=2014,
            transaction_cost_bps=0,
            model_name="constant",
        )
        summary = summarize_backtest(daily)
        test_dates = dates[(dates >= "2014-01-01") & (dates < "2015-01-01")]

        self.assertEqual(len(daily), len(test_dates))
        self.assertEqual(set(daily["model_name"]), {"constant"})
        self.assertEqual(daily["position_weight_a"].iloc[0], 0.0)
        self.assertAlmostEqual(daily["strategy_return"].iloc[1], 0.00075)
        self.assertEqual(len(summary), 1)
        self.assertEqual(summary.loc[0, "test_year"], 2014)


if __name__ == "__main__":
    unittest.main()
