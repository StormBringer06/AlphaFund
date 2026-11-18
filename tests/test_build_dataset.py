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
                "XEL": [100.0, 110.0, 121.0],
                "WEC": [50.0, None, 55.0],
                "BAC": [60.0, 61.0, 62.0],
                "PNC": [None, 70.0, 71.0],
                "ALL": [80.0, 81.0, 82.0],
                "TRV": [90.0, 91.0, 92.0],
                "TFC": [30.0, 31.0, 32.0],
                "USB": [40.0, 41.0, 42.0],
                "HD": [200.0, 201.0, 202.0],
                "LOW": [100.0, 101.0, 102.0],
            },
            index=dates,
        )
        closes.index.name = "date"

        self.assertEqual(
            list(build_dataset.TICKERS),
            ["XEL", "WEC", "BAC", "PNC", "ALL", "TRV", "TFC", "USB", "HD", "LOW"],
        )
        result = build_dataset.make_pair_prices(closes)
        xel_wec = result[result["pair_id"] == "xel_wec"].reset_index(drop=True)
        bac_pnc = result[result["pair_id"] == "bac_pnc"].reset_index(drop=True)

        self.assertEqual(list(xel_wec["date"]), [dates[0], dates[2]])
        self.assertTrue(pd.isna(xel_wec.loc[0, "return_a"]))
        self.assertAlmostEqual(xel_wec.loc[1, "return_a"], 0.21)
        self.assertEqual(list(bac_pnc["date"]), [dates[1], dates[2]])
        self.assertTrue(pd.isna(bac_pnc.loc[0, "return_b"]))
        self.assertEqual(
            set(result["pair_id"]),
            {"xel_wec", "bac_pnc", "all_trv", "tfc_usb", "hd_low"},
        )


if __name__ == "__main__":
    unittest.main()
