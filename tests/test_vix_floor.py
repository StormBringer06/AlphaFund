"""Tests for the VIX floor (pairs_model.vix_floor)."""

import unittest

import numpy as np
import pandas as pd

from backtest import apply_positions_to_returns
from pairs_model.vix_floor import apply_vix_floor


def daily(vix) -> pd.DataFrame:
    n = len(vix)
    return pd.DataFrame({
        "date": pd.bdate_range("2020-01-01", periods=n), "pair_id": "a_b", "fold_id": 1,
        "test_year": 2020, "vix": vix, "return_a": 0.01, "return_b": -0.01,
        "target_weight_a": 0.5, "target_weight_b": -0.5, "signal": 1.0,
    })


class VixFloorTests(unittest.TestCase):
    def test_zeroes_targets_only_below_the_floor(self):
        out = apply_vix_floor(daily([15.0, 11.9, 12.0, 30.0]), floor=12.0)
        np.testing.assert_array_equal(out["vix_gate_open"], [True, False, True, True])
        np.testing.assert_allclose(out["target_weight_a"], [0.5, 0.0, 0.5, 0.5])
        np.testing.assert_allclose(out["target_weight_b"], [-0.5, 0.0, -0.5, -0.5])
        np.testing.assert_allclose(out["signal"], [1.0, 0.0, 1.0, 1.0])

    def test_low_vix_close_flattens_the_next_day(self):
        # Targets set at close t earn return t+1, so VIX at t decides day t+1's position.
        out = apply_positions_to_returns(apply_vix_floor(daily([15.0, 10.0, 15.0, 15.0])), 0.0)
        np.testing.assert_allclose(out["position_weight_a"], [0.0, 0.5, 0.0, 0.5])

    def test_rejects_bad_input(self):
        with self.assertRaises(ValueError):
            apply_vix_floor(daily([15.0, np.nan]))
        with self.assertRaises(ValueError):
            apply_vix_floor(daily([15.0]), floor=-1.0)


if __name__ == "__main__":
    unittest.main()
