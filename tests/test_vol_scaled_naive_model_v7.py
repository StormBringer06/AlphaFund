import unittest

import numpy as np
import pandas as pd

from backtest import apply_positions_to_returns
from improved_models.vol_scaled_naive_model_v7 import apply_vix_floor_to_targets


class VixFloorTests(unittest.TestCase):
    def test_floor_masks_targets_and_backtest_lags_the_gate(self):
        daily = pd.DataFrame(
            {
                "date": pd.date_range("2020-01-01", periods=4, freq="D"),
                "pair_id": ["a_b"] * 4,
                "fold_id": [1] * 4,
                "vix": [11.0, 9.0, 10.0, 12.0],
                "return_a": [0.01] * 4,
                "return_b": [-0.01] * 4,
                "target_weight_a": [0.4] * 4,
                "target_weight_b": [-0.4] * 4,
                "leverage": [2.0] * 4,
                "target_signal": [0.8] * 4,
                "signal": [0.8] * 4,
                "target_gross_exposure": [0.8] * 4,
            }
        )

        gated = apply_vix_floor_to_targets(
            daily,
            vix_floor=10.0,
            remove_leverage=True,
        )
        np.testing.assert_allclose(
            gated["target_weight_a"],
            [0.2, 0.0, 0.2, 0.2],
        )
        self.assertEqual(
            gated["vix_gate_open"].tolist(),
            [True, False, True, True],
        )

        applied = apply_positions_to_returns(gated, transaction_cost_bps=0.0)
        np.testing.assert_allclose(
            applied["position_weight_a"],
            [0.0, 0.2, 0.0, 0.2],
        )

    def test_rejects_nonfinite_vix_and_nonpositive_leverage(self):
        daily = pd.DataFrame(
            {
                "vix": [float("nan")],
                "target_weight_a": [0.2],
                "target_weight_b": [-0.2],
                "leverage": [1.0],
            }
        )
        with self.assertRaisesRegex(ValueError, "VIX values must be finite"):
            apply_vix_floor_to_targets(daily)

        daily["vix"] = 12.0
        daily["leverage"] = 0.0
        with self.assertRaisesRegex(ValueError, "leverage values must be positive"):
            apply_vix_floor_to_targets(daily, remove_leverage=True)


if __name__ == "__main__":
    unittest.main()
