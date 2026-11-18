import unittest
from unittest.mock import patch

import pandas as pd

import build_dataset


class DownloadAdjustedClosesTests(unittest.TestCase):
    @patch("build_dataset.time.sleep")
    @patch("build_dataset.yf.download")
    def test_requests_tickers_sequentially_with_delay(self, download, sleep):
        dates = pd.to_datetime(["2024-01-02", "2024-01-03"])
        download.side_effect = [
            pd.DataFrame({"Close": [10.0 + i, 11.0 + i]}, index=dates)
            for i, _ in enumerate(build_dataset.TICKERS)
        ]

        result = build_dataset.download_adjusted_closes(None, None, request_delay=4)

        self.assertEqual(list(result.columns), list(build_dataset.TICKERS))
        self.assertEqual(result.index.name, "date")
        self.assertEqual(download.call_count, len(build_dataset.TICKERS))
        self.assertEqual(
            [call.kwargs["tickers"] for call in download.call_args_list],
            list(build_dataset.TICKERS),
        )
        self.assertTrue(all(call.kwargs["auto_adjust"] for call in download.call_args_list))
        self.assertTrue(all(call.kwargs["period"] == "max" for call in download.call_args_list))
        self.assertEqual(sleep.call_count, len(build_dataset.TICKERS) - 1)
        sleep.assert_called_with(4)


class MakePairPricesTests(unittest.TestCase):
    def test_returns_use_shared_dates_and_leave_first_return_missing(self):
        dates = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
        closes = pd.DataFrame(
            {
                "JPM": [100.0, 110.0, 121.0],
                "BAC": [50.0, None, 55.0],
                "KO": [60.0, 61.0, 62.0],
                "PEP": [None, 70.0, 71.0],
                "BARC.L": [1.0, 1.1, 1.2],
                "LLOY.L": [2.0, 2.1, 2.2],
                "XOM": [100.0, 101.0, 102.0],
                "CVX": [90.0, 91.0, 92.0],
                "V": [200.0, 201.0, 202.0],
                "MA": [300.0, 301.0, 302.0],
            },
            index=dates,
        )
        closes.index.name = "date"

        result = build_dataset.make_pair_prices(closes)
        jpm_bac = result[result["pair_id"] == "jpm_bac"].reset_index(drop=True)
        ko_pep = result[result["pair_id"] == "ko_pep"].reset_index(drop=True)

        self.assertEqual(list(jpm_bac["date"]), [dates[0], dates[2]])
        self.assertTrue(pd.isna(jpm_bac.loc[0, "return_a"]))
        self.assertAlmostEqual(jpm_bac.loc[1, "return_a"], 0.21)
        self.assertEqual(list(ko_pep["date"]), [dates[1], dates[2]])
        self.assertTrue(pd.isna(ko_pep.loc[0, "return_b"]))


if __name__ == "__main__":
    unittest.main()
