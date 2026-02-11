import unittest
from unittest.mock import patch

import pandas as pd

from app.backtest import grid_search_cup_shape, grid_search_selling_climax


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

    @patch("app.backtest.run_canslim_backtest")
    @patch("app.backtest.load_price_csv")
    def test_two_stage_search_reduces_evaluation_count(
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
        mock_load_price_csv.return_value = price_df

        def run_side_effect(**kwargs):
            cup_window = kwargs["cup_window"]
            recovery_ratio = kwargs["cup_recovery_ratio"]
            handle_max_depth = kwargs["cup_handle_max_depth"]
            score_base = (
                cup_window / 1000.0 + recovery_ratio / 10.0 - handle_max_depth / 5.0
            )
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
                    "avg_return": [score_base + 0.01, score_base],
                }
            )
            return results_df, summary_df

        mock_run_canslim_backtest.side_effect = run_side_effect

        grid_kwargs = {
            "symbols": ["1111"],
            "min_signals": 0,
            "cup_windows": [40, 50, 60, 70],
            "handle_windows": [7],
            "depth_ranges": [(0.12, 0.35)],
            "recovery_ratios": [0.82, 0.85, 0.88],
            "handle_max_depths": [0.1, 0.12, 0.15],
        }

        grid_search_cup_shape(**grid_kwargs)
        full_count = mock_run_canslim_backtest.call_count

        mock_run_canslim_backtest.reset_mock()

        grid_search_cup_shape(
            **grid_kwargs,
            top_k=1,
            coarse_stride=2,
        )
        two_stage_count = mock_run_canslim_backtest.call_count

        self.assertLess(two_stage_count, full_count)

    @patch("app.backtest.run_canslim_backtest")
    @patch("app.backtest.load_price_csv")
    def test_two_stage_off_keeps_legacy_result_order(
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
        mock_load_price_csv.return_value = price_df

        def run_side_effect(**kwargs):
            cup_window = kwargs["cup_window"]
            handle_window = kwargs["handle_window"]
            recovery_ratio = kwargs["cup_recovery_ratio"]
            handle_max_depth = kwargs["cup_handle_max_depth"]
            score_base = (
                cup_window / 1000.0
                + handle_window / 1000.0
                + recovery_ratio / 10.0
                - handle_max_depth / 5.0
            )
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
                    "avg_return": [score_base + 0.02, score_base],
                }
            )
            return results_df, summary_df

        mock_run_canslim_backtest.side_effect = run_side_effect

        baseline_eval_df, _ = grid_search_cup_shape(
            symbols=["1111"],
            min_signals=0,
            cup_windows=[40, 50],
            handle_windows=[7, 10],
            depth_ranges=[(0.12, 0.35)],
            recovery_ratios=[0.82, 0.85],
            handle_max_depths=[0.1],
        )

        off_eval_df, _ = grid_search_cup_shape(
            symbols=["1111"],
            min_signals=0,
            cup_windows=[40, 50],
            handle_windows=[7, 10],
            depth_ranges=[(0.12, 0.35)],
            recovery_ratios=[0.82, 0.85],
            handle_max_depths=[0.1],
            top_k=2,
            coarse_stride=1,
        )

        self.assertListEqual(
            baseline_eval_df["score"].tolist(),
            off_eval_df["score"].tolist(),
        )


if __name__ == "__main__":
    unittest.main()


class GridSearchSellingClimaxPreloadTests(unittest.TestCase):
    @patch("app.backtest.load_price_csv")
    def test_grid_search_selling_climax_does_not_reload_price_csv_in_combinations(
        self,
        mock_load_price_csv,
    ):
        price_df = pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-02", "2024-01-01"]),
                "open": [100.0, 101.0],
                "high": [101.0, 102.0],
                "low": [99.0, 100.0],
                "close": [100.5, 101.5],
                "volume": [1000, 1100],
            }
        )

        def load_side_effect(symbol):
            if symbol == "3333":
                raise ValueError("load failed")
            return price_df

        mock_load_price_csv.side_effect = load_side_effect

        eval_df, best_summary = grid_search_selling_climax(
            symbols=["1111", "2222", "3333"],
            min_signals=0,
            volume_lookbacks=[20, 30],
            volume_multipliers=[1.1, 1.3],
            drop_pcts=[0.03],
            close_positions=[0.5],
            confirm_ks=[2],
            atr_lookbacks=[14],
            drop_atr_mults=[1.5],
            drop_condition_modes=["drop_pct_only"],
            trend_ma_lens=[20],
            trend_modes=["none"],
            stop_atr_mults=[1.0],
            time_stop_bars_list=[3],
            trailing_atr_mults=[1.0],
            min_avg_dollar_volumes=[None],
            min_avg_volumes=[None],
            vol_percentile_thresholds=[None],
            vol_lookback2s=[20],
            max_gap_pcts=[None],
        )

        self.assertEqual(mock_load_price_csv.call_count, 3)
        self.assertEqual(len(eval_df), 4)
        self.assertFalse(best_summary.empty)


class GridSearchSellingClimaxSuccessCacheTests(unittest.TestCase):
    @patch("app.backtest._label_selling_climax_success")
    @patch("app.backtest.load_price_csv")
    def test_grid_search_selling_climax_reuses_success_cache_by_symbol_and_key(
        self,
        mock_load_price_csv,
        mock_label_success,
    ):
        rows = 60
        dates = pd.date_range("2024-01-01", periods=rows, freq="D")
        close = [100.0] * rows
        open_ = [100.0] * rows
        high = [101.0] * rows
        low = [99.0] * rows
        volume = [1000.0] * rows

        for idx in [30, 40]:
            close[idx] = 95.0
            open_[idx] = 101.0
            high[idx] = 102.0
            low[idx] = 94.0
            volume[idx] = 5000.0

        price_df = pd.DataFrame(
            {
                "date": dates,
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
            }
        )
        mock_load_price_csv.return_value = price_df

        mock_label_success.side_effect = lambda df, **kwargs: pd.Series(True, index=df.index)

        eval_df, best_summary = grid_search_selling_climax(
            symbols=["1111"],
            min_signals=0,
            volume_lookbacks=[20],
            volume_multipliers=[1.1, 1.2],
            drop_pcts=[0.03],
            close_positions=[0.5],
            confirm_ks=[2],
            atr_lookbacks=[14],
            drop_atr_mults=[1.5],
            drop_condition_modes=["drop_pct_only"],
            trend_ma_lens=[20],
            trend_modes=["none"],
            stop_atr_mults=[1.0],
            time_stop_bars_list=[3],
            trailing_atr_mults=[1.0],
            min_avg_dollar_volumes=[None],
            min_avg_volumes=[None],
            vol_percentile_thresholds=[None],
            vol_lookback2s=[20],
            max_gap_pcts=[None],
        )

        self.assertFalse(eval_df.empty)
        self.assertFalse(best_summary.empty)
        self.assertEqual(mock_label_success.call_count, 1)



class GridSearchSellingClimaxExtendedParamsTests(unittest.TestCase):
    @patch("app.backtest.load_price_csv")
    def test_grid_search_selling_climax_accepts_extended_parameters(self, mock_load_price_csv):
        rows = 80
        base_dates = pd.date_range("2024-01-01", periods=rows, freq="D")
        close = [100.0] * rows
        open_ = [100.0] * rows
        high = [101.0] * rows
        low = [99.0] * rows
        volume = [1000.0] * rows
        close[30] = 94.0
        open_[30] = 100.0
        high[30] = 100.5
        low[30] = 93.0
        volume[30] = 10000.0
        close[31] = 101.0
        high[31] = 102.0

        price_df = pd.DataFrame(
            {
                "date": base_dates,
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
            }
        )
        mock_load_price_csv.return_value = price_df

        eval_df, best_summary = grid_search_selling_climax(
            symbols=["1111", "2222"],
            min_signals=0,
            volume_lookbacks=[20],
            volume_multipliers=[1.1],
            drop_pcts=[0.03],
            close_positions=[0.5],
            confirm_ks=[2],
            atr_lookbacks=[14],
            drop_atr_mults=[1.5],
            drop_condition_modes=["both"],
            trend_ma_lens=[20],
            trend_modes=["exclude_downtrend"],
            stop_atr_mults=[1.0],
            time_stop_bars_list=[3],
            trailing_atr_mults=[1.0],
            min_avg_dollar_volumes=[50000.0],
            min_avg_volumes=[500.0],
            vol_percentile_thresholds=[80.0],
            vol_lookback2s=[20],
            max_gap_pcts=[0.03],
        )

        self.assertFalse(eval_df.empty)
        self.assertFalse(best_summary.empty)
        self.assertIn("drop_atr_mult", eval_df.columns)
        self.assertIn("trend_mode", eval_df.columns)
        self.assertIn("stop_atr_mult", eval_df.columns)
