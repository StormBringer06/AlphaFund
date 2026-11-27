import unittest

import numpy as np
import pandas as pd

from improved_models.vol_scaled_naive_model_nl import (
    VolScaledNaivePairsTradingModel,
    _Hedge,
)


class NonlinearVolScaledNaiveModelTests(unittest.TestCase):
    def test_squared_z_score_preserves_direction(self):
        model = VolScaledNaivePairsTradingModel(
            zscore_halflife=2,
            vix_window=20,
            band=0.0,
        )
        prices_b = np.linspace(100.0, 110.0, 40)
        spread = np.sin(np.arange(40, dtype=float)) * 0.01
        data = pd.DataFrame({
            "date": pd.bdate_range("2020-01-01", periods=40),
            "adjusted_close_a": prices_b * np.exp(spread),
            "adjusted_close_b": prices_b,
            "vix": 20.0,
        })

        features = model._features(data, _Hedge(0.0, 1.0))
        valid = features["spread_zscore"].notna()
        z = features.loc[valid, "spread_zscore"]
        expected = np.sign(z) * z.pow(2)
        np.testing.assert_allclose(
            features.loc[valid, "spread_zscore1"], expected
        )

        signal = model._positions(
            pd.DataFrame({
                "spread_zscore": [2.0, -2.0],
                "spread_zscore1": [4.0, -4.0],
                "vix_pct": [0.5, 0.5],
            }),
            _Hedge(0.0, 1.0),
        )["target_signal"]
        self.assertLess(signal.iloc[0], 0.0)
        self.assertGreater(signal.iloc[1], 0.0)
        self.assertAlmostEqual(abs(signal.iloc[0]), abs(signal.iloc[1]))


if __name__ == "__main__":
    unittest.main()
