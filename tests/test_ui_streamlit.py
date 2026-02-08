import unittest

import pandas as pd

from ui_streamlit import (
    _apply_bulk_group_assignments,
    _calculate_minimum_data_length,
    _has_macd_cross,
    _latest_has_required_data,
    _parse_bulk_group_lines,
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



class BulkManualGroupAssignmentTest(unittest.TestCase):
    def test_parse_bulk_group_lines(self):
        assignments, errors = _parse_bulk_group_lines(
            "7203,自動車|大型株\n6758,ハイテク\nabc,不正\n"
        )

        self.assertEqual(
            assignments,
            [
                ("7203", ["自動車", "大型株"]),
                ("6758", ["ハイテク"]),
            ],
        )
        self.assertEqual(len(errors), 1)

    def test_apply_bulk_group_assignments(self):
        groups = {"既存": ["7203"]}
        assignments = [
            ("7203", ["既存", "自動車"]),
            ("6758", ["ハイテク"]),
            ("9999", ["対象外"]),
        ]

        updated, applied_count, created_group_count, unknown_symbols = _apply_bulk_group_assignments(
            groups,
            assignments,
            symbols=["7203", "6758"],
        )

        self.assertEqual(updated["既存"], ["7203"])
        self.assertEqual(updated["自動車"], ["7203"])
        self.assertEqual(updated["ハイテク"], ["6758"])
        self.assertEqual(applied_count, 2)
        self.assertEqual(created_group_count, 2)
        self.assertEqual(unknown_symbols, ["9999"])
if __name__ == "__main__":
    unittest.main()
