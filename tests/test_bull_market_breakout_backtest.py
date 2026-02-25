import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from app.backtest import run_bull_market_new_high_momentum_backtest


class BullMarketBreakoutBacktestTests(unittest.TestCase):
    def _build_price_df(self, seed: int, base_price: float, volume: float) -> pd.DataFrame:
        rng = np.random.RandomState(seed)
        dates = pd.bdate_range("2022-01-03", periods=420)
        trend = np.linspace(0, 120, len(dates))
        noise = rng.normal(0, 0.2, len(dates)).cumsum()
        close = base_price + trend + noise
        open_ = close * (1 - 0.001)
        high = np.maximum(open_, close) * 1.01
        low = np.minimum(open_, close) * 0.99
        vol = np.full(len(dates), volume)
        # 中盤以降に出来高を増やしてブレイク時の条件を満たしやすくする
        vol[260:] = volume * 1.5
        return pd.DataFrame(
            {
                "date": dates,
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "volume": vol,
            }
        )

    def _build_topix_df(self) -> pd.DataFrame:
        dates = pd.bdate_range("2022-01-03", periods=420)
        close = np.linspace(1800, 2800, len(dates))
        return pd.DataFrame({"date": dates, "close": close})

    @patch("app.backtest.load_price_csv")
    def test_backtest_returns_summary_and_event_study(self, mock_load_price_csv):
        symbol_a = self._build_price_df(seed=7, base_price=100.0, volume=2_000_000)
        symbol_b = self._build_price_df(seed=11, base_price=80.0, volume=100_000)

        def side_effect(symbol):
            if symbol == "1111":
                return symbol_a
            if symbol == "2222":
                return symbol_b
            raise ValueError("unexpected symbol")

        mock_load_price_csv.side_effect = side_effect

        result = run_bull_market_new_high_momentum_backtest(
            symbols=["1111", "2222"],
            topix_df=self._build_topix_df(),
            high_lookback=63,
            hold_days=20,
            event_cooldown_days=20,
            top_liquidity_count=1,
            rebalance_weekday=None,
        )

        self.assertIn("summary", result)
        self.assertFalse(result["summary"].empty)
        self.assertFalse(result["trades"].empty)
        self.assertFalse(result["daily_returns"].empty)
        self.assertFalse(result["event_study"].empty)

        summary = result["summary"].iloc[0]
        self.assertEqual(int(summary["selected_symbols"]), 1)
        self.assertGreaterEqual(float(summary["trades"]), 1)
        self.assertIn("information_ratio", result["summary"].columns)

    @patch("app.backtest.load_price_csv")
    def test_returns_empty_when_no_symbol_data(self, mock_load_price_csv):
        mock_load_price_csv.return_value = pd.DataFrame()

        result = run_bull_market_new_high_momentum_backtest(
            symbols=["1111"],
            topix_df=self._build_topix_df(),
        )

        self.assertTrue(result["summary"].empty)
        self.assertTrue(result["trades"].empty)


if __name__ == "__main__":
    unittest.main()
