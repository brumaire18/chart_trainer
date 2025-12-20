import unittest

import pandas as pd

from ui_streamlit import _has_macd_cross, _latest_has_required_data


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


if __name__ == "__main__":
    unittest.main()
