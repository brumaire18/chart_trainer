import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

import pandas as pd

from app.us_market_data import (
    US_PROVIDER_LABELS,
    fetch_us_daily_ohlcv,
    save_us_daily_csv,
)


class FetchUsDailyOhlcvTest(unittest.TestCase):
    @patch("app.us_market_data.requests.get")
    def test_fetches_and_filters_stooq_csv(self, mock_get):
        response = Mock()
        response.raise_for_status = Mock()
        response.text = (
            "Date,Open,High,Low,Close,Volume\n"
            "2024-01-02,100,110,99,105,1000\n"
            "2024-01-03,106,112,104,108,1200\n"
            "2024-01-04,109,115,108,114,1300\n"
        )
        mock_get.return_value = response

        df = fetch_us_daily_ohlcv(
            symbol="xlk",
            start_date="2024-01-03",
            end_date="2024-01-04",
            provider="stooq_csv",
        )

        self.assertEqual(list(df["date"].dt.strftime("%Y-%m-%d")), ["2024-01-03", "2024-01-04"])
        self.assertEqual(list(df.columns), ["date", "open", "high", "low", "close", "volume"])

    @patch("app.us_market_data.requests.get")
    def test_raises_error_when_provider_is_not_supported(self, mock_get):
        with self.assertRaises(ValueError):
            fetch_us_daily_ohlcv(symbol="XLK", provider="unknown")
        mock_get.assert_not_called()


class SaveUsDailyCsvTest(unittest.TestCase):
    def test_save_uses_normalized_symbol_filename(self):
        df = pd.DataFrame(
            {
                "date": ["2024-01-02", "2024-01-03"],
                "open": [100.0, 101.0],
                "high": [110.0, 111.0],
                "low": [99.0, 100.0],
                "close": [105.0, 108.0],
                "volume": [1000, 1100],
            }
        )

        with TemporaryDirectory() as tmp_dir:
            output = save_us_daily_csv(symbol=" xlk ", df=df, target_dir=Path(tmp_dir))
            self.assertEqual(output.name, "XLK.csv")
            saved_df = pd.read_csv(output)
            self.assertEqual(
                list(saved_df.columns),
                ["date", "open", "high", "low", "close", "volume"],
            )


class UsProviderLabelTest(unittest.TestCase):
    def test_has_label_for_each_supported_provider(self):
        self.assertIn("stooq_csv", US_PROVIDER_LABELS)


if __name__ == "__main__":
    unittest.main()
