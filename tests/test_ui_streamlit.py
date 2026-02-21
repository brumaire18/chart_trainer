import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from app.pair_trading import generate_pairs_by_sector_candidates
from ui_streamlit import (
    _apply_checked_codes_to_groups,
    _build_search_result_df,
    _merge_checked_codes_with_display_selection,
    _calculate_minimum_data_length,
    _has_macd_cross,
    _latest_has_required_data,
    _append_pair_grid_search_history,
    _load_pair_grid_search_history,
    _parse_depth_range_grid_values,
    _parse_numeric_grid_values,
    _filter_symbols_by_search,
    _apply_breadth_exclusions,
    _build_swing_point_series,
    _build_sector_group_symbol_map,
    _group_matches_sector_filter,
)


class HasMacdCrossTest(unittest.TestCase):
    def test_detects_golden_cross(self):
        df = pd.DataFrame(
            [
                {"macd": -0.4, "macd_signal": -0.35, "macd_hist": -0.05},
                {"macd": -0.3, "macd_signal": -0.32, "macd_hist": 0.02},
                {"macd": 0.1, "macd_signal": -0.05, "macd_hist": 0.15},
            ]
        )

        self.assertTrue(_has_macd_cross(df, "golden", lookback=2))

    def test_detects_dead_cross(self):
        df = pd.DataFrame(
            [
                {"macd": 0.35, "macd_signal": 0.3, "macd_hist": 0.05},
                {"macd": 0.2, "macd_signal": 0.22, "macd_hist": -0.02},
                {"macd": 0.1, "macd_signal": 0.12, "macd_hist": -0.02},
            ]
        )

        self.assertTrue(_has_macd_cross(df, "dead", lookback=2))

    def test_no_cross_when_hist_sign_mismatches(self):
        df = pd.DataFrame(
            [
                {"macd": -0.1, "macd_signal": -0.2, "macd_hist": 0.1},
                {"macd": 0.05, "macd_signal": -0.05, "macd_hist": -0.1},
            ]
        )

        self.assertFalse(_has_macd_cross(df, "golden", lookback=2))

    def test_lookback_limits_detection(self):
        df = pd.DataFrame(
            [
                {"macd": -0.3, "macd_signal": -0.2, "macd_hist": -0.1},
                {"macd": 0.1, "macd_signal": -0.05, "macd_hist": 0.15},
                {"macd": 0.2, "macd_signal": 0.15, "macd_hist": 0.05},
                {"macd": 0.25, "macd_signal": 0.2, "macd_hist": 0.05},
            ]
        )

        self.assertFalse(_has_macd_cross(df, "golden", lookback=1))

    def test_cross_detected_after_nan_rows_are_skipped(self):
        df = pd.DataFrame(
            [
                {"macd": -0.2, "macd_signal": -0.1, "macd_hist": -0.1},
                {"macd": float("nan"), "macd_signal": -0.05, "macd_hist": float("nan")},
                {"macd": -0.15, "macd_signal": -0.1, "macd_hist": -0.05},
                {"macd": 0.05, "macd_signal": -0.05, "macd_hist": 0.1},
            ]
        )

        self.assertTrue(_has_macd_cross(df, "golden", lookback=4))


class LatestHasRequiredDataTest(unittest.TestCase):
    def test_allows_nan_when_condition_disabled(self):
        latest = pd.Series(
            {
                "date": pd.Timestamp("2024-01-01"),
                "close": 1000,
                "volume": 12345,
                "rsi14": float("nan"),
                "macd": float("nan"),
                "macd_signal": float("nan"),
                "macd_hist": float("nan"),
                "sma20": float("nan"),
                "sma50": float("nan"),
            }
        )

        self.assertTrue(
            _latest_has_required_data(
                latest,
                apply_rsi_condition=False,
                macd_condition="none",
                require_sma20_trend=False,
                apply_topix_rs_condition=False,
                apply_ma_approach_condition=False,
                ma_target="sma50",
            )
        )

    def test_detects_nan_on_required_columns(self):
        latest = pd.Series(
            {
                "date": pd.Timestamp("2024-01-01"),
                "close": 1000,
                "volume": float("nan"),
                "rsi14": 55.0,
                "macd": 0.1,
                "macd_signal": 0.05,
                "macd_hist": 0.05,
                "sma20": 990.0,
                "sma50": 995.0,
            }
        )

        self.assertFalse(
            _latest_has_required_data(
                latest,
                apply_rsi_condition=True,
                macd_condition="golden",
                require_sma20_trend=True,
                apply_topix_rs_condition=False,
                apply_ma_approach_condition=False,
                ma_target="sma50",
            )
        )


    def test_requires_ma_target_column_when_ma_approach_enabled(self):
        latest = pd.Series(
            {
                "date": pd.Timestamp("2024-01-01"),
                "close": 1000,
                "volume": 12345,
                "rsi14": 55.0,
                "macd": 0.1,
                "macd_signal": 0.05,
                "macd_hist": 0.05,
                "sma20": 990.0,
            }
        )

        self.assertFalse(
            _latest_has_required_data(
                latest,
                apply_rsi_condition=False,
                macd_condition="none",
                require_sma20_trend=False,
                apply_topix_rs_condition=False,
                apply_ma_approach_condition=True,
                ma_target="sma50",
            )
        )


class ParseGridValuesTest(unittest.TestCase):
    def test_parse_numeric_grid_values_with_comma_and_range(self):
        values = _parse_numeric_grid_values("30:50:10,70", int)
        self.assertEqual(values, [30, 40, 50, 70])

    def test_parse_numeric_grid_values_rejects_invalid_step_direction(self):
        with self.assertRaises(ValueError):
            _parse_numeric_grid_values("30:50:-5", int)

    def test_parse_depth_range_grid_values(self):
        values = _parse_depth_range_grid_values("0.12-0.35,0.15~0.40,0.18:0.45")
        self.assertEqual(values, [(0.12, 0.35), (0.15, 0.4), (0.18, 0.45)])

class CalculateMinimumDataLengthTest(unittest.TestCase):
    def test_returns_base_requirement_and_reasons(self):
        required_length, reasons = _calculate_minimum_data_length(
            apply_rsi_condition=False,
            macd_condition="none",
            macd_lookback=5,
            require_sma20_trend=False,
            sma_trend_lookback=3,
            apply_volume_condition=False,
            apply_topix_rs_condition=False,
            topix_rs_lookback=20,
            apply_new_high_signal=False,
            new_high_lookback=60,
            apply_selling_climax_signal=False,
            selling_volume_lookback=20,
            signal_lookback_days=10,
            apply_canslim_condition=False,
            cup_window=35,
            saucer_window=40,
            handle_window=15,
            apply_weekly_volume_quartile=False,
            apply_cup_handle_condition=False,
            cup_handle_max_window=325,
            cup_handle_rs_lookback=60,
            apply_ma_approach_condition=False,
            ma_period=50,
            ma_approach_lookback=5,
        )

        self.assertEqual(required_length, 50)
        self.assertTrue(any("最低50本" in msg for msg in reasons))

    def test_respects_all_enabled_conditions(self):
        required_length, reasons = _calculate_minimum_data_length(
            apply_rsi_condition=True,
            macd_condition="golden",
            macd_lookback=8,
            require_sma20_trend=True,
            sma_trend_lookback=4,
            apply_volume_condition=True,
            apply_topix_rs_condition=False,
            topix_rs_lookback=20,
            apply_new_high_signal=False,
            new_high_lookback=60,
            apply_selling_climax_signal=False,
            selling_volume_lookback=20,
            signal_lookback_days=10,
            apply_canslim_condition=False,
            cup_window=35,
            saucer_window=40,
            handle_window=15,
            apply_weekly_volume_quartile=False,
            apply_cup_handle_condition=False,
            cup_handle_max_window=325,
            cup_handle_rs_lookback=60,
            apply_ma_approach_condition=False,
            ma_period=50,
            ma_approach_lookback=5,
        )

        # MACD requires 26, SMA20 requires 24, base is 50 -> expect 50
        self.assertEqual(required_length, 50)
        self.assertTrue(any("RSI(14)" in msg for msg in reasons))
        self.assertTrue(any("MACDクロス" in msg for msg in reasons))
        self.assertTrue(any("SMA20" in msg for msg in reasons))
        self.assertTrue(any("20日平均売買代金" in msg for msg in reasons))





    def test_updates_requirement_with_ma_approach_condition(self):
        required_length, reasons = _calculate_minimum_data_length(
            apply_rsi_condition=False,
            macd_condition="none",
            macd_lookback=5,
            require_sma20_trend=False,
            sma_trend_lookback=3,
            apply_volume_condition=False,
            apply_topix_rs_condition=False,
            topix_rs_lookback=20,
            apply_new_high_signal=False,
            new_high_lookback=60,
            apply_selling_climax_signal=False,
            selling_volume_lookback=20,
            signal_lookback_days=10,
            apply_canslim_condition=False,
            cup_window=35,
            saucer_window=40,
            handle_window=15,
            apply_weekly_volume_quartile=False,
            apply_cup_handle_condition=False,
            cup_handle_max_window=325,
            cup_handle_rs_lookback=60,
            apply_ma_approach_condition=True,
            ma_period=200,
            ma_approach_lookback=15,
        )

        self.assertEqual(required_length, 215)
        self.assertTrue(any("MA接近判定には" in msg and "215本以上必要" in msg for msg in reasons))


class SearchResultBulkGroupingTest(unittest.TestCase):
    def test_build_search_result_df_marks_checked_codes(self):
        df = _build_search_result_df(
            codes=["7203", "6758"],
            name_map={"7203": "トヨタ", "6758": "ソニー"},
            sector_map={"7203": "輸送用機器", "6758": "電気機器"},
            checked_codes=["6758"],
            classified_groups_map={"7203": ["大型株", "自動車"]},
        )

        self.assertEqual(df.loc[0, "選択"], False)
        self.assertEqual(df.loc[1, "選択"], True)
        self.assertEqual(df.loc[0, "名称"], "トヨタ")
        self.assertEqual(df.loc[0, "分類済みグループ"], "大型株 / 自動車")
        self.assertEqual(df.loc[1, "分類済みグループ"], "")

    def test_apply_checked_codes_to_groups_creates_and_appends(self):
        updated, applied_count, created_group_count = _apply_checked_codes_to_groups(
            custom_groups={"既存": ["7203"]},
            checked_codes=["7203", "6758"],
            target_groups=["既存", "成長株"],
        )

        self.assertEqual(created_group_count, 1)
        self.assertEqual(applied_count, 3)
        self.assertEqual(updated["既存"], ["7203", "6758"])
        self.assertEqual(updated["成長株"], ["7203", "6758"])

    def test_merge_checked_codes_replaces_only_current_page(self):
        edited_df = pd.DataFrame(
            [
                {"選択": False, "コード": "1301"},
                {"選択": True, "コード": "6758"},
            ]
        )

        merged = _merge_checked_codes_with_display_selection(
            previous_checked_codes=["1301", "7203"],
            display_codes=["1301", "6758"],
            edited_search_result_df=edited_df,
        )

        self.assertEqual(merged, ["7203", "6758"])

    def test_merge_checked_codes_keeps_previous_when_editor_empty(self):
        merged = _merge_checked_codes_with_display_selection(
            previous_checked_codes=["1301", "7203"],
            display_codes=["1301", "6758"],
            edited_search_result_df=pd.DataFrame(),
        )

        self.assertEqual(merged, ["1301", "7203"])


class PairGridSearchHistoryTest(unittest.TestCase):
    def test_append_and_load_history(self):
        with TemporaryDirectory() as tmp_dir:
            history_path = Path(tmp_dir) / "history.csv"
            optimization_df = pd.DataFrame(
                [
                    {
                        "lookback": 60,
                        "entry_z": 2.0,
                        "exit_z": 0.5,
                        "stop_z": 3.5,
                        "max_holding_days": 20,
                        "trade_count": 8,
                        "win_rate": 0.5,
                        "total_pnl": 123.4,
                        "avg_pnl": 15.425,
                    }
                ]
            )

            saved_rows = _append_pair_grid_search_history(
                optimization_df,
                min_trades=5,
                start_date="2023-01-01",
                end_date="2023-12-31",
                param_grid={"lookback": [40, 60]},
                history_path=history_path,
            )

            self.assertEqual(saved_rows, 1)
            loaded_df = _load_pair_grid_search_history(history_path=history_path)
            self.assertEqual(len(loaded_df), 1)
            self.assertEqual(int(loaded_df.loc[0, "min_trades"]), 5)
            self.assertEqual(str(loaded_df.loc[0, "start_date"]), "2023-01-01")
            self.assertIn('"lookback": [40, 60]', str(loaded_df.loc[0, "param_grid"]))

    def test_append_skips_empty_results(self):
        with TemporaryDirectory() as tmp_dir:
            history_path = Path(tmp_dir) / "history.csv"
            saved_rows = _append_pair_grid_search_history(
                pd.DataFrame(),
                min_trades=5,
                start_date=None,
                end_date=None,
                param_grid={"lookback": [40]},
                history_path=history_path,
            )

            self.assertEqual(saved_rows, 0)
            self.assertFalse(history_path.exists())



class BreadthExclusionHelpersTest(unittest.TestCase):
    def test_filter_symbols_by_search_matches_code_and_name(self):
        symbols = ["1301", "7203", "9984"]
        name_map = {"1301": "極洋", "7203": "トヨタ自動車", "9984": "ソフトバンクグループ"}

        self.assertEqual(
            _filter_symbols_by_search(symbols, name_map, "720"),
            ["7203"],
        )
        self.assertEqual(
            _filter_symbols_by_search(symbols, name_map, "ソフト"),
            ["9984"],
        )

    def test_apply_breadth_exclusions_removes_target_codes(self):
        symbols = ["1301", "7203", "9984"]
        excluded = ["7203", "9999"]

        self.assertEqual(
            _apply_breadth_exclusions(symbols, excluded),
            ["1301", "9984"],
        )


class BuildSwingPointSeriesTest(unittest.TestCase):
    def test_returns_sorted_swing_points(self):
        df = pd.DataFrame(
            {
                "date": pd.date_range("2024-01-01", periods=7, freq="D"),
                "high": [10, 12, 11, 14, 13, 15, 14],
                "low": [8, 9, 7, 10, 9, 11, 10],
            }
        )

        result = _build_swing_point_series(df, order=1)

        self.assertFalse(result.empty)
        self.assertEqual(result["index"].tolist(), sorted(result["index"].tolist()))
        self.assertIn("high", result["kind"].tolist())
        self.assertIn("low", result["kind"].tolist())

    def test_returns_empty_when_no_extrema(self):
        df = pd.DataFrame(
            {
                "date": pd.date_range("2024-01-01", periods=3, freq="D"),
                "high": [10, 10, 10],
                "low": [9, 9, 9],
            }
        )

        result = _build_swing_point_series(df, order=2)

        self.assertTrue(result.empty)


class PairCacheSectorGroupFilterTest(unittest.TestCase):
    def test_build_sector_group_symbol_map_uses_group_master_sector33(self):
        custom_groups = {
            "半導体A": ["1111", "2222"],
            "半導体B": ["3333", "9999"],
            "別業種": ["4444"],
        }
        group_master = {
            "半導体A": {"sector_type": "33業種", "sector_value": "電気機器"},
            "半導体B": {"sector_type": "33業種", "sector_value": "電気機器"},
            "別業種": {"sector_type": "33業種", "sector_value": "サービス業"},
        }

        result = _build_sector_group_symbol_map(
            custom_groups,
            group_master,
            symbols=["1111", "2222", "3333", "4444"],
            sector_col="sector33",
        )

        self.assertEqual(result["電気機器"], ["1111", "2222", "3333"])
        self.assertEqual(result["サービス業"], ["4444"])


    def test_build_sector_group_symbol_map_supports_multiple_sector_rules(self):
        custom_groups = {
            "横断グループ": ["1111", "2222", "3333"],
        }
        group_master = {
            "横断グループ": {
                "sector_type": "33業種",
                "sector_value": "電気機器",
                "sector_rules": [
                    {"sector_type": "33業種", "sector_value": "電気機器"},
                    {"sector_type": "33業種", "sector_value": "機械"},
                ],
            },
        }

        result = _build_sector_group_symbol_map(
            custom_groups,
            group_master,
            symbols=["1111", "2222", "3333"],
            sector_col="sector33",
        )

        self.assertEqual(result["電気機器"], ["1111", "2222", "3333"])
        self.assertEqual(result["機械"], ["1111", "2222", "3333"])

    def test_generate_pairs_by_sector_candidates_respects_sector_group_symbol_map(self):
        listed_df = pd.DataFrame(
            {
                "code": ["1111", "2222", "3333", "4444"],
                "sector33": ["電気機器", "電気機器", "電気機器", "サービス業"],
                "market": ["PRIME", "PRIME", "PRIME", "PRIME"],
            }
        )

        pairs = generate_pairs_by_sector_candidates(
            listed_df=listed_df,
            symbols=["1111", "2222", "3333", "4444"],
            sector33="電気機器",
            max_pairs_per_sector=None,
            sector_group_symbol_map={"電気機器": ["1111", "3333"]},
        )

        self.assertEqual(pairs, [("1111", "3333")])



class ManualGroupSectorVisibilityTest(unittest.TestCase):
    def test_group_matches_sector_filter_with_multiple_rules(self):
        group_master = {
            "横断": {
                "sector_rules": [
                    {"sector_type": "33業種", "sector_value": "電気機器"},
                    {"sector_type": "33業種", "sector_value": "機械"},
                ]
            }
        }

        self.assertTrue(
            _group_matches_sector_filter("横断", group_master, "33業種", "機械")
        )
        self.assertFalse(
            _group_matches_sector_filter("横断", group_master, "17業種", "輸送用機器")
        )



if __name__ == "__main__":
    unittest.main()
