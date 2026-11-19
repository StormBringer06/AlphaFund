import unittest

import numpy as np
import pandas as pd

from backtest import run_walk_forward_backtest
from improved_models.entry_state_model import EntryStatePairsTradingModel as Model
from test_regression_model import make_data


class EntryStateModelTests(unittest.TestCase):
    def setUp(self):
        self.data = make_data()
        split = pd.Timestamp("2018-01-01")
        self.train = self.data[self.data["date"] < split]
        self.test = self.data[self.data["date"] >= split]

    def test_predictions_do_not_use_future_data(self):
        full = Model().fit(self.train).predict(self.test)
        partial = Model().fit(self.train).predict(self.test.iloc[:200])
        self.assertTrue(full["beta_refit"].iloc[:200].any())
        cols = ["vol_ratio", "trade_side", "target_weight_a", "target_weight_b"]
        pd.testing.assert_frame_equal(full.iloc[:200][cols], partial[cols])

    def test_trades_open_beyond_entry_and_close_at_zero_cross(self):
        out = Model(band=0.0).fit(self.train).predict(self.test)
        z = out["spread_zscore"]
        entries = out[out["trade_entry"]]
        self.assertFalse(entries.empty)
        self.assertTrue((entries["spread_zscore"].abs() > 1.0).all())
        # Trades fade the spread.
        np.testing.assert_array_equal(entries["trade_side"], -np.sign(entries["spread_zscore"]))
        flat = out["trade_side"] == 0
        self.assertTrue((out.loc[flat, "signal"] == 0).all())
        # While a trade is open, z has not crossed zero against the trade.
        open_ = out[~flat]
        self.assertTrue((open_["trade_side"] * z[~flat] < 0).all())

    def test_calm_trades_are_scaled_and_only_upgraded(self):
        out = Model(calm_size=0.5, band=0.0).fit(self.train).predict(self.test)
        open_ = out[out["trade_side"] != 0]
        base = np.minimum(1.0, open_["spread_zscore"].abs() / 2.0)
        expected = np.where(open_["trade_full_size"], base, 0.5 * base)
        np.testing.assert_allclose(open_["signal"].abs(), expected)
        # Within a trade, full size never reverts to calm size.
        trade_id = out["trade_entry"].cumsum()
        for _, trade in out[out["trade_side"] != 0].groupby(trade_id):
            flags = trade["trade_full_size"].to_numpy()
            self.assertTrue((np.diff(flags.astype(int)) >= 0).all())

    def test_calm_size_zero_skips_calm_entries(self):
        out = Model(calm_size=0.0, band=0.0).fit(self.train).predict(self.test)
        calm_open = out[(out["trade_side"] != 0) & ~out["trade_full_size"]]
        self.assertTrue((calm_open["signal"] == 0).all())
        self.assertTrue((out.loc[out["trade_full_size"], "signal"] != 0).any())

    def test_no_upgrade_keeps_calm_trades_calm(self):
        out = Model(allow_upgrade=False, band=0.0).fit(self.train).predict(self.test)
        trade_id = out["trade_entry"].cumsum()
        for _, trade in out[out["trade_side"] != 0].groupby(trade_id):
            self.assertEqual(trade["trade_full_size"].nunique(), 1)

    def test_band_limits_rebalancing(self):
        no_band = Model(band=0.0).fit(self.train).predict(self.test)
        banded = Model(band=0.2).fit(self.train).predict(self.test)
        self.assertLess(banded["rebalanced"].sum(), no_band["rebalanced"].sum())

    def test_runs_in_walk_forward_backtest(self):
        daily = run_walk_forward_backtest(
            self.data, model_factory=Model, start_year=2019, end_year=2020,
            train_years=4, transaction_cost_bps=5.0,
        )
        self.assertFalse(daily["strategy_return"].isna().any())


if __name__ == "__main__":
    unittest.main()
