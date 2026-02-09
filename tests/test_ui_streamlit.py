import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from ui_streamlit import (
    _apply_checked_codes_to_groups,
    _apply_sector_group_assignment,
    _build_search_result_df,
    _calculate_minimum_data_length,
    _has_macd_cross,
    _latest_has_required_data,
    _parse_bulk_group_lines,
    _append_pair_grid_search_history,
    _load_pair_grid_search_history,
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
            }
        )

        self.assertTrue(
            _latest_has_required_data(
                latest,
                apply_rsi_condition=False,
                macd_condition="none",
                require_sma20_trend=False,
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
            }
        )

        self.assertFalse(
            _latest_has_required_data(
                latest,
                apply_rsi_condition=True,
                macd_condition="golden",
                require_sma20_trend=True,
            )
        )


class CalculateMinimumDataLengthTest(unittest.TestCase):
    def test_returns_base_requirement_and_reasons(self):
        required_length, reasons = _calculate_minimum_data_length(
            apply_rsi_condition=False,
            macd_condition="none",
            macd_lookback=5,
            require_sma20_trend=False,
            sma_trend_lookback=3,
            apply_volume_condition=False,
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
        )

        # MACD requires 26, SMA20 requires 24, base is 50 -> expect 50
        self.assertEqual(required_length, 50)
        self.assertTrue(any("RSI(14)" in msg for msg in reasons))
        self.assertTrue(any("MACDクロス" in msg for msg in reasons))
        self.assertTrue(any("SMA20" in msg for msg in reasons))
        self.assertTrue(any("20日平均出来高" in msg for msg in reasons))



class SectorGroupAssignmentTest(unittest.TestCase):
    def test_add_and_remove_sector_codes(self):
        listed_df = pd.DataFrame(
            [
                {"code": "7203", "sector17": "輸送用機器"},
                {"code": "7267", "sector17": "輸送用機器"},
                {"code": "6758", "sector17": "電気機器"},
            ]
        )
        groups = {"既存": ["6758"]}

        updated_add, changed_add = _apply_sector_group_assignment(
            groups,
            listed_df,
            symbols=["7203", "7267", "6758"],
            sector_column="sector17",
            sector_value="輸送用機器",
            target_group="自動車",
            action="add",
        )
        self.assertEqual(changed_add, 2)
        self.assertEqual(updated_add["自動車"], ["7203", "7267"])

        updated_remove, changed_remove = _apply_sector_group_assignment(
            {"自動車": ["7203", "7267", "6758"]},
            listed_df,
            symbols=["7203", "7267", "6758"],
            sector_column="sector17",
            sector_value="輸送用機器",
            target_group="自動車",
            action="remove",
        )
        self.assertEqual(changed_remove, 2)
        self.assertEqual(updated_remove["自動車"], ["6758"])


class SearchResultBulkGroupingTest(unittest.TestCase):
    def test_build_search_result_df_marks_checked_codes(self):
        df = _build_search_result_df(
            codes=["7203", "6758"],
            name_map={"7203": "トヨタ", "6758": "ソニー"},
            sector_map={"7203": "輸送用機器", "6758": "電気機器"},
            checked_codes=["6758"],
        )

        self.assertEqual(df.loc[0, "選択"], False)
        self.assertEqual(df.loc[1, "選択"], True)
        self.assertEqual(df.loc[0, "名称"], "トヨタ")

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



if __name__ == "__main__":
    unittest.main()
