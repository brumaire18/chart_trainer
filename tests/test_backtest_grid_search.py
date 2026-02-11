import unittest
from unittest.mock import patch

import pandas as pd

from app.backtest import (
    _build_selling_climax_feature_cache,
    _detect_selling_climax_candidates,
    _detect_selling_climax_candidates_with_cache,
    grid_search_cup_shape,
    grid_search_selling_climax,
)


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


class SellingClimaxFeatureCacheCompatibilityTests(unittest.TestCase):
    def _legacy_detect_selling_climax_candidates(
        self,
        df: pd.DataFrame,
        volume_lookback: int,
        volume_multiplier: float,
        drop_pct: float,
        close_position: float,
        atr_lookback=None,
        drop_atr_mult=None,
        drop_condition_mode: str = "drop_pct_only",
        trend_ma_len=None,
        trend_mode: str = "none",
        min_avg_dollar_volume=None,
        min_avg_volume=None,
        liquidity_lookback: int = 20,
        vol_percentile_threshold=None,
        vol_lookback2=None,
        max_gap_pct=None,
        index_filter_mask=None,
    ):
        volume_avg = df["volume"].rolling(volume_lookback, min_periods=volume_lookback).mean()
        volume_ratio = df["volume"] / volume_avg
        prev_close = df["close"].shift(1)
        drop_ratio = (df["close"] - prev_close) / prev_close
        candle_range = df["high"] - df["low"]
        close_pos = (df["close"] - df["low"]) / candle_range.replace(0, float("nan"))

        drop_pct_condition = drop_ratio <= -abs(drop_pct)

        drop_atr_condition = pd.Series(False, index=df.index)
        if atr_lookback is not None and drop_atr_mult is not None and atr_lookback > 0:
            true_range = pd.concat(
                [
                    df["high"] - df["low"],
                    (df["high"] - prev_close).abs(),
                    (df["low"] - prev_close).abs(),
                ],
                axis=1,
            ).max(axis=1)
            atr = true_range.rolling(atr_lookback, min_periods=atr_lookback).mean()
            drop_atr_condition = (-drop_ratio * prev_close) >= (atr * float(drop_atr_mult))

        if drop_condition_mode == "drop_atr_only":
            drop_condition = drop_atr_condition
        elif drop_condition_mode == "both":
            drop_condition = drop_pct_condition & drop_atr_condition
        else:
            drop_condition = drop_pct_condition

        trend_condition = pd.Series(True, index=df.index)
        if trend_ma_len is not None and trend_ma_len > 1 and trend_mode != "none":
            trend_ma = df["close"].rolling(trend_ma_len, min_periods=trend_ma_len).mean()
            ma_slope_up = trend_ma > trend_ma.shift(1)
            close_above_ma = df["close"] >= trend_ma
            if trend_mode == "reversion_only_in_uptrend":
                trend_condition = close_above_ma & ma_slope_up
            elif trend_mode == "exclude_downtrend":
                trend_condition = ma_slope_up
            elif trend_mode == "ma_slope_positive":
                trend_condition = ma_slope_up

        liquidity_condition = pd.Series(True, index=df.index)
        if min_avg_volume is not None:
            avg_volume = df["volume"].rolling(liquidity_lookback, min_periods=liquidity_lookback).mean()
            liquidity_condition &= avg_volume >= float(min_avg_volume)
        if min_avg_dollar_volume is not None:
            avg_dollar_volume = (df["close"] * df["volume"]).rolling(
                liquidity_lookback, min_periods=liquidity_lookback
            ).mean()
            liquidity_condition &= avg_dollar_volume >= float(min_avg_dollar_volume)

        volume_shape_condition = pd.Series(True, index=df.index)
        if vol_percentile_threshold is not None and vol_lookback2 is not None and vol_lookback2 > 1:
            volume_quantile = (
                df["volume"].shift(1).rolling(vol_lookback2, min_periods=vol_lookback2).quantile(
                    float(vol_percentile_threshold) / 100.0
                )
            )
            volume_shape_condition = df["volume"] >= volume_quantile

        gap_condition = pd.Series(True, index=df.index)
        if max_gap_pct is not None:
            gap_ratio = (df["open"] - prev_close) / prev_close
            gap_condition = gap_ratio <= float(max_gap_pct)

        candidates = (
            (df["close"] <= df["open"])
            & (volume_ratio >= volume_multiplier)
            & drop_condition
            & (candle_range > 0)
            & (close_pos <= close_position)
            & trend_condition
            & liquidity_condition
            & volume_shape_condition
            & gap_condition
        )
        if index_filter_mask is not None:
            candidates &= index_filter_mask.reindex(df.index).fillna(False)
        return candidates.fillna(False)

    def test_feature_cache_fast_path_matches_legacy_logic(self):
        rows = 120
        dates = pd.date_range("2024-01-01", periods=rows, freq="D")
        close = [100 + i * 0.05 for i in range(rows)]
        open_ = [c + 0.3 for c in close]
        high = [o + 0.8 for o in open_]
        low = [c - 1.0 for c in close]
        volume = [1000 + (i % 15) * 30 for i in range(rows)]

        for idx in [40, 65, 90]:
            open_[idx] = close[idx - 1] + 1.8
            close[idx] = close[idx - 1] - 5.0
            high[idx] = open_[idx] + 0.4
            low[idx] = close[idx] - 0.8
            volume[idx] = 9000

        df = pd.DataFrame(
            {
                "date": dates,
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
            }
        )

        kwargs = {
            "volume_lookback": 20,
            "volume_multiplier": 1.5,
            "drop_pct": 0.03,
            "close_position": 0.5,
            "atr_lookback": 14,
            "drop_atr_mult": 1.2,
            "drop_condition_mode": "both",
            "trend_ma_len": 30,
            "trend_mode": "exclude_downtrend",
            "min_avg_dollar_volume": 50000.0,
            "min_avg_volume": 800.0,
            "liquidity_lookback": 20,
            "vol_percentile_threshold": 80.0,
            "vol_lookback2": 25,
            "max_gap_pct": 0.05,
        }

        legacy = self._legacy_detect_selling_climax_candidates(df, **kwargs)

        feature_cache = _build_selling_climax_feature_cache(
            df,
            needed_lookbacks={kwargs["volume_lookback"], kwargs["liquidity_lookback"]},
            needed_atr_lookbacks={kwargs["atr_lookback"]},
            needed_vol_lookbacks={kwargs["vol_lookback2"]},
            needed_trend_ma_lens={kwargs["trend_ma_len"]},
        )
        fast = _detect_selling_climax_candidates_with_cache(
            df,
            feature_cache=feature_cache,
            **kwargs,
        )

        self.assertTrue(legacy.equals(fast))

        default_path = _detect_selling_climax_candidates(df, **kwargs)
        self.assertTrue(legacy.equals(default_path))
