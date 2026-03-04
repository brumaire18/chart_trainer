import unittest

import pandas as pd

from app.backtest import build_daily_turnover_universe


class DailyTurnoverUniverseTests(unittest.TestCase):
    def test_returns_date_to_symbols_top_k(self):
        panel = pd.DataFrame(
            {
                "date": ["2024-01-04", "2024-01-04", "2024-01-04", "2024-01-05", "2024-01-05"],
                "symbol": ["1111", "2222", "3333", "1111", "4444"],
                "close": [100.0, 300.0, 200.0, 150.0, 100.0],
                "volume": [10.0, 5.0, 8.0, 7.0, 20.0],
            }
        )

        result = build_daily_turnover_universe(panel, top_k=2)

        self.assertEqual(result[pd.Timestamp("2024-01-04")], ["3333", "2222"])
        self.assertEqual(result[pd.Timestamp("2024-01-05")], ["4444", "1111"])

    def test_filters_invalid_rows_and_handles_daily_existing_symbols_only(self):
        panel = pd.DataFrame(
            {
                "date": ["2024-01-04", "2024-01-04", "2024-01-05", "2024-01-05", "2024-01-05"],
                "symbol": ["1111", "2222", "1111", "2222", "3333"],
                "close": [100.0, None, 0.5, 120.0, 90.0],
                "volume": [1000.0, 1000.0, 1000.0, 0.0, 2000.0],
            }
        )

        result = build_daily_turnover_universe(panel, top_k=5, min_close=1.0)

        self.assertEqual(result[pd.Timestamp("2024-01-04")], ["1111"])
        self.assertEqual(result[pd.Timestamp("2024-01-05")], ["3333"])

    def test_returns_boolean_mask(self):
        panel = pd.DataFrame(
            {
                "date": ["2024-01-04", "2024-01-04", "2024-01-05"],
                "symbol": ["1111", "2222", "1111"],
                "close": [100.0, 200.0, 100.0],
                "volume": [10.0, 1.0, 8.0],
            }
        )

        mask = build_daily_turnover_universe(panel, top_k=1, return_bool_mask=True)

        self.assertEqual(mask.tolist(), [True, False, True])


if __name__ == "__main__":
    unittest.main()
