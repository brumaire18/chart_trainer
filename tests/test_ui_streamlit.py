import unittest

import pandas as pd

from ui_streamlit import _calculate_minimum_data_length, _has_macd_cross


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


class CalculateMinimumDataLengthTest(unittest.TestCase):
    def test_keeps_original_threshold_with_sma_trend(self):
        required_length, reasons = _calculate_minimum_data_length(
            require_sma20_trend=True,
            sma_trend_lookback=7,
            macd_condition="golden",
            macd_lookback=5,
            apply_volume_condition=True,
            apply_rsi_condition=True,
        )

        self.assertEqual(required_length, max(50, 7 + 1))
        self.assertTrue(any("SMAトレンド判定" in reason for reason in reasons))

    def test_macd_only_uses_macd_threshold(self):
        required_length, reasons = _calculate_minimum_data_length(
            require_sma20_trend=False,
            sma_trend_lookback=3,
            macd_condition="golden",
            macd_lookback=5,
            apply_volume_condition=False,
            apply_rsi_condition=False,
        )

        self.assertEqual(required_length, max(5 + 2, 26))
        self.assertTrue(any("MACD判定" in reason for reason in reasons))

    def test_volume_condition_requires_twenty_bars(self):
        required_length, reasons = _calculate_minimum_data_length(
            require_sma20_trend=False,
            sma_trend_lookback=3,
            macd_condition="none",
            macd_lookback=5,
            apply_volume_condition=True,
            apply_rsi_condition=False,
        )

        self.assertEqual(required_length, 20)
        self.assertTrue(any("出来高条件" in reason for reason in reasons))


if __name__ == "__main__":
    unittest.main()
