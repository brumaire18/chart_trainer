import unittest
from unittest.mock import patch

import pandas as pd

from app.backtest import grid_search_cup_shape


class GridSearchCupShapePreloadTests(unittest.TestCase):
    @patch("app.backtest.run_canslim_backtest")
    @patch("app.backtest.load_price_csv")
    def test_grid_search_does_not_reload_price_csv_in_combinations(
        self,
        mock_load_price_csv,
        mock_run_canslim_backtest,
    ):
        price_df = pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
                "open": [100.0, 101.0],
                "high": [101.0, 102.0],
                "low": [99.0, 100.0],
                "close": [100.5, 101.5],
                "volume": [1000, 1100],
            }
        )

        def load_side_effect(symbol):
            if symbol == "2222":
                raise ValueError("load failed")
            return price_df

        mock_load_price_csv.side_effect = load_side_effect

        results_df = pd.DataFrame(
            {
                "symbol": ["1111", "1111"],
                "dataset": ["train", "validation"],
            }
        )
        summary_df = pd.DataFrame(
            {
                "dataset": ["train", "validation"],
                "pattern": ["cup_with_handle", "cup_with_handle"],
                "label": ["up", "up"],
                "avg_return": [0.1, 0.08],
            }
        )
        mock_run_canslim_backtest.return_value = (results_df, summary_df)

        eval_df, best_summary = grid_search_cup_shape(
            symbols=["1111", "2222"],
            min_signals=0,
            cup_windows=[40, 50],
            handle_windows=[7],
            depth_ranges=[(0.12, 0.35)],
            recovery_ratios=[0.82],
            handle_max_depths=[0.1],
        )

        self.assertEqual(mock_load_price_csv.call_count, 2)
        self.assertEqual(mock_run_canslim_backtest.call_count, 2)
        self.assertEqual(len(eval_df), 2)
        self.assertFalse(best_summary.empty)

        for call in mock_run_canslim_backtest.call_args_list:
            kwargs = call.kwargs
            self.assertEqual(kwargs["symbols"], ["1111"])
            self.assertIn("symbol_price_data", kwargs)
            self.assertEqual(list(kwargs["symbol_price_data"].keys()), ["1111"])


if __name__ == "__main__":
    unittest.main()
