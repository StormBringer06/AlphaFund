"""Fast no-look-ahead tests for the exit classifier (pairs_model.exit_classifier).

The full end-to-end check (rebuild everything on truncated data) is slower:
``python -m tests.lookahead_check``.
"""

import unittest

import numpy as np
import pandas as pd

from pairs_model.exit_classifier import (
    CHECKPOINTS, FEATURES, earnings_features, ridge_decisions, spread_features,
)


def synthetic_closes(days: int = 1500, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2010-01-01", periods=days)
    log_b = 4.0 + np.cumsum(rng.normal(0, 0.01, days))
    spread = np.zeros(days)
    for i in range(1, days):
        spread[i] = 0.97 * spread[i - 1] + rng.normal(0, 0.01)
    return pd.DataFrame({"AAA": np.exp(0.1 + 0.9 * log_b + spread), "BBB": np.exp(log_b)},
                        index=pd.DatetimeIndex(dates, name="date"))


class ExitClassifierTests(unittest.TestCase):
    def test_spread_features_use_only_past_data(self):
        closes = synthetic_closes()
        vix_pct = pd.Series(0.5, index=closes.index)
        cut = closes.index[1000]
        full = spread_features(closes, vix_pct, "AAA", "BBB")
        part = spread_features(closes.loc[:cut], vix_pct.loc[:cut], "AAA", "BBB")
        pd.testing.assert_frame_equal(full.loc[:cut], part, check_exact=False, rtol=0, atol=1e-12)

    def test_earnings_count_from_when_the_market_sees_them(self):
        dates = pd.bdate_range("2020-01-01", periods=30)
        ts = pd.to_datetime(["2020-01-07 07:00", "2020-01-14 16:30"]).tz_localize("America/New_York")
        earnings = pd.DataFrame({"ticker": "AAA", "ts": ts, "Reported EPS": [1.0, 1.1],
                                 "Surprise(%)": [5.0, -3.0]})
        f = earnings_features(earnings, "AAA", dates)
        reports = list(f.index[f["is_report"]].strftime("%Y-%m-%d"))
        # pre-open report counts that day; after-close report counts the next day
        self.assertEqual(reports, ["2020-01-07", "2020-01-15"])
        self.assertEqual(f.loc["2020-01-14", "surprise"], 5.0)
        self.assertEqual(f.loc["2020-01-15", "surprise"], -3.0)

    def test_saved_weights_give_the_ridge_prediction(self):
        rng = np.random.default_rng(1)
        n = len(FEATURES)
        model = {"fill_median": rng.normal(size=n).tolist(), "scale_sd": (1 + rng.random(n)).tolist(),
                 "coef": rng.normal(size=n).tolist(), "intercept": 0.1}
        checkups = pd.DataFrame(rng.normal(size=(5, n)), columns=FEATURES)
        checkups.iloc[0, 3] = np.nan  # missing inputs are filled with the training median
        checkups = checkups.assign(ep_key=[f"t{i}" for i in range(5)], checkpoint=CHECKPOINTS[0],
                                   remaining_pnl=0.0, date=pd.Timestamp("2020-06-01"))
        dec = ridge_decisions({"2020": model}, checkups, [2020])
        x = checkups[FEATURES].to_numpy(float)
        x = np.where(np.isfinite(x), x, np.array(model["fill_median"]))
        expected = ((x - model["fill_median"]) / model["scale_sd"]) @ model["coef"] + model["intercept"]
        np.testing.assert_allclose(dec["predicted_remaining"], expected)
        np.testing.assert_array_equal(dec["ridge<0"], expected < 0)


if __name__ == "__main__":
    unittest.main()
