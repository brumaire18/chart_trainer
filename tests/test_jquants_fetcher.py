import tempfile
import unittest
from datetime import date, datetime
from unittest.mock import patch

import pandas as pd
from pathlib import Path

from app.jquants_fetcher import (
    JST,
    CLOSE_TIME_DEFAULT,
    JQuantsError,
    _get_id_token,
    _normalize_daily_quotes,
    _normalize_topix,
    fetch_listed_master,
    get_growth_universe,
    should_run_after_close,
    sync_universe_after_close,
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

    def test_applies_adjustment_factor_when_adjusted_columns_missing(self):
        df_raw = pd.DataFrame(
            [
                {
                    "date": "2024-01-02",
                    "Open": 100,
                    "High": 110,
                    "Low": 90,
                    "Close": 105,
                    "Volume": 1000,
                    "AdjustmentFactor": 0.5,
                }
            ]
        )

        normalized = _normalize_daily_quotes(df_raw, "7203")

        self.assertEqual(normalized.loc[0, "open"], 50)
        self.assertEqual(normalized.loc[0, "high"], 55)
        self.assertEqual(normalized.loc[0, "low"], 45)
        self.assertEqual(normalized.loc[0, "close"], 52.5)
        self.assertEqual(normalized.loc[0, "volume"], 2000)


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


class FetchListedMasterTests(unittest.TestCase):
    @patch("app.jquants_fetcher._request_with_token")
    @patch("app.jquants_fetcher._get_client")
    def test_fetch_listed_master_pagination_and_mapping(
        self,
        mock_get_client,
        mock_request_with_token,
    ):
        mock_get_client.return_value = object()
        mock_request_with_token.side_effect = [
            {
                "listedInfo": [
                    {
                        "LocalCode": "7203",
                        "CompanyName": "トヨタ自動車",
                        "MarketCode": "0111",
                        "MarketCodeName": "プライム",
                    }
                ],
                "pagination_key": "next",
            },
            {
                "listedInfo": [
                    {
                        "LocalCode": "8306",
                        "Name": "三菱UFJフィナンシャル・グループ",
                        "MarketCode": "0112",
                        "MarketCodeName": "スタンダード",
                    }
                ]
            },
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("app.jquants_fetcher.META_DIR", Path(tmpdir)):
                df = fetch_listed_master()

        self.assertEqual(df.loc[0, "code"], "7203")
        self.assertEqual(df.loc[0, "name"], "トヨタ自動車")
        self.assertEqual(df.loc[0, "market"], "PRIME")
        self.assertEqual(df.loc[1, "code"], "8306")
        self.assertEqual(df.loc[1, "market"], "STANDARD")


class GetGrowthUniverseTests(unittest.TestCase):
    @patch("app.jquants_fetcher.load_listed_master")
    def test_growth_only_filter(self, mock_load_listed_master):
        mock_load_listed_master.return_value = pd.DataFrame(
            {
                "code": ["1", "2", "3"],
                "market": ["PRIME", "GROWTH", "STANDARD"],
            }
        )

        result = get_growth_universe()

        self.assertEqual(result, ["0002"])


class AfterCloseSyncTests(unittest.TestCase):
    def test_should_run_after_close_true_on_weekday_after_16(self):
        now = datetime(2024, 1, 9, 16, 1, tzinfo=JST)  # Tuesday
        self.assertTrue(should_run_after_close(now=now, close_time=CLOSE_TIME_DEFAULT))

    def test_should_run_after_close_false_before_16(self):
        now = datetime(2024, 1, 9, 15, 59, tzinfo=JST)  # Tuesday
        self.assertFalse(should_run_after_close(now=now, close_time=CLOSE_TIME_DEFAULT))

    def test_should_run_after_close_false_on_weekend(self):
        now = datetime(2024, 1, 13, 16, 30, tzinfo=JST)  # Saturday
        self.assertFalse(should_run_after_close(now=now, close_time=CLOSE_TIME_DEFAULT))

    @patch("app.jquants_fetcher.append_quotes_for_date")
    @patch("app.jquants_fetcher.build_universe")
    def test_sync_universe_after_close_runs_append(self, mock_build_universe, mock_append):
        mock_build_universe.return_value = ["7203", "8306"]

        executed = sync_universe_after_close(
            now=datetime(2024, 1, 9, 16, 5, tzinfo=JST),
            include_custom=True,
            use_listed_master=False,
            market_filter="prime_standard",
        )

        self.assertTrue(executed)
        mock_build_universe.assert_called_once()
        mock_append.assert_called_once_with("2024-01-09", codes=["7203", "8306"])

    @patch("app.jquants_fetcher.append_quotes_for_date")
    @patch("app.jquants_fetcher.build_universe")
    def test_sync_universe_after_close_skips_before_close(self, mock_build_universe, mock_append):
        executed = sync_universe_after_close(
            now=datetime(2024, 1, 9, 15, 30, tzinfo=JST),
        )

        self.assertFalse(executed)
        mock_build_universe.assert_not_called()
        mock_append.assert_not_called()
