import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

import pandas as pd

from app.us_market_data import fetch_us_daily_ohlcv, save_us_daily_csv


class USMarketDataTest(unittest.TestCase):
    @patch("app.us_market_data.requests.get")
    def test_fetch_us_daily_ohlcv_normalizes_and_filters_dates(self, mock_get):
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.text = (
            "Date,Open,High,Low,Close,Volume\n"
            "2024-01-02,100,110,95,108,1000\n"
            "2024-01-03,108,112,107,111,1200\n"
            "2024-01-04,111,115,109,114,1300\n"
        )
        mock_get.return_value = mock_response

        df = fetch_us_daily_ohlcv(
            symbol="xlk",
            start_date="2024-01-03",
            end_date="2024-01-04",
            provider="stooq_csv",
        )

        self.assertEqual(list(df.columns), ["date", "open", "high", "low", "close", "volume"])
        self.assertEqual(len(df), 2)
        self.assertEqual(df.iloc[0]["date"], pd.Timestamp("2024-01-03"))
        self.assertEqual(float(df.iloc[0]["close"]), 111.0)

    def test_save_us_daily_csv_writes_leadlag_compatible_filename_and_columns(self):
        df = pd.DataFrame(
            {
                "date": ["2024-01-02", "2024-01-03"],
                "open": [10.0, 11.0],
                "high": [11.0, 12.0],
                "low": [9.0, 10.0],
                "close": [10.5, 11.5],
                "volume": [100, 110],
            }
        )

        with TemporaryDirectory() as tmp_dir:
            output_path = save_us_daily_csv("xlk", df, Path(tmp_dir))
            self.assertEqual(output_path.name, "XLK.csv")
            saved = pd.read_csv(output_path)

        self.assertEqual(list(saved.columns), ["date", "open", "high", "low", "close", "volume"])
        self.assertEqual(saved.iloc[0]["date"], "2024-01-02")


if __name__ == "__main__":
    unittest.main()
