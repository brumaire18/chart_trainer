import unittest
from datetime import date
from unittest.mock import patch

import pandas as pd

from app.jquants_fetcher import (
    JQuantsError,
    _get_id_token,
    _normalize_daily_quotes,
    _normalize_topix,
    update_symbol,
)


class UpdateSymbolReturnTest(unittest.TestCase):
    @patch("app.jquants_fetcher._merge_and_save")
    @patch("app.jquants_fetcher._normalize_daily_quotes")
    @patch("app.jquants_fetcher._request_with_token")
    @patch("app.jquants_fetcher._get_latest_trading_day")
    @patch("app.jquants_fetcher._light_plan_window")
    @patch("app.jquants_fetcher._load_existing_csv")
    @patch("app.jquants_fetcher._load_meta")
    @patch("app.jquants_fetcher._get_client")
    def test_returns_merged_dataframe(
        self,
        mock_get_client,
        mock_load_meta,
        mock_load_existing_csv,
        mock_light_plan_window,
        mock_get_latest_trading_day,
        mock_request_with_token,
        mock_normalize_daily_quotes,
        mock_merge_and_save,
    ):
        mock_get_client.return_value = object()
        mock_load_meta.return_value = {}
        mock_load_existing_csv.return_value = None
        mock_light_plan_window.return_value = ("2024-01-01", "2024-01-31")
        mock_get_latest_trading_day.return_value = (date(2024, 1, 31), None)
        mock_request_with_token.return_value = {"dailyQuotes": [{"date": "2024-01-30"}]}

        normalized_df = pd.DataFrame(
            {"date": ["2024-01-30"], "code": ["7203"], "market": ["プライム"]}
        )
        mock_normalize_daily_quotes.return_value = normalized_df

        merged_df = pd.DataFrame({"date": ["2024-01-30"], "close": [1000]})
        mock_merge_and_save.return_value = merged_df

        result = update_symbol("7203")

        self.assertIs(result, merged_df)
        self.assertIn("datetime", result.columns)
        self.assertEqual(result.loc[0, "datetime"], "2024-01-30T00:00:00")


if __name__ == "__main__":
    unittest.main()


class GetIdTokenTests(unittest.TestCase):
    def test_get_id_token_success(self):
        class DummyClient:
            def authenticate(self):
                return "new-id-token"

        token = _get_id_token(DummyClient())
        self.assertEqual(token, "new-id-token")

    def test_get_id_token_failure(self):
        class DummyClient:
            def authenticate(self):
                raise ValueError("invalid grant")

        with self.assertRaises(JQuantsError):
            _get_id_token(DummyClient())


class NormalizeDailyQuotesTests(unittest.TestCase):
    def test_accepts_new_column_names(self):
        df_raw = pd.DataFrame(
            [
                {
                    "date": "2024-01-02",
                    "symbol": "7203",
                    "marketCode": "0111",
                    "adjustmentOpen": 100,
                    "adjustmentHigh": 110,
                    "adjustmentLow": 90,
                    "adjustmentClose": 105,
                    "adjustmentVolume": 1000,
                }
            ]
        )

        normalized = _normalize_daily_quotes(df_raw, "7203")

        self.assertEqual(normalized.loc[0, "code"], "7203")
        self.assertEqual(normalized.loc[0, "market"], "0111")
        self.assertEqual(normalized.loc[0, "open"], 100)
        self.assertEqual(normalized.loc[0, "close"], 105)


class NormalizeTopixTests(unittest.TestCase):
    def test_accepts_lowercase_columns(self):
        df_raw = pd.DataFrame(
            [
                {
                    "date": "2024-01-02",
                    "open": 2000,
                    "high": 2010,
                    "low": 1990,
                    "close": 2005,
                }
            ]
        )

        normalized = _normalize_topix(df_raw)

        self.assertEqual(normalized.loc[0, "code"], "TOPIX")
        self.assertEqual(normalized.loc[0, "close"], 2005)
