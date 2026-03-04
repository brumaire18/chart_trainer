import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from app.backtest import run_ma5_deviation_mean_reversion_backtest


class MA5DeviationMeanReversionBacktestTests(unittest.TestCase):
    def _build_price_df(self, seed: int, base_price: float, amp: float) -> pd.DataFrame:
        rng = np.random.RandomState(seed)
        dates = pd.bdate_range("2023-01-02", periods=60)
        t = np.arange(len(dates))
        close = base_price + amp * np.sin(t / 3.0) + rng.normal(0, 2.0, len(dates))
        open_ = close * (1 + rng.normal(0, 0.002, len(dates)))
        volume = np.full(len(dates), 60_000.0)
        return pd.DataFrame(
            {
                "date": dates,
                "open": open_,
                "high": np.maximum(open_, close) * 1.01,
                "low": np.minimum(open_, close) * 0.99,
                "close": close,
                "volume": volume,
            }
        )

    @patch("app.backtest.load_price_csv")
    def test_backtest_returns_trades_and_summary(self, mock_load_price_csv):
        universe = {
            "1111": self._build_price_df(seed=1, base_price=2000.0, amp=90.0),
            "2222": self._build_price_df(seed=2, base_price=2500.0, amp=120.0),
            "3333": self._build_price_df(seed=3, base_price=1800.0, amp=70.0),
            "4444": self._build_price_df(seed=4, base_price=3000.0, amp=150.0),
        }

        def side_effect(symbol):
            return universe[symbol]

        mock_load_price_csv.side_effect = side_effect

        trades_df, summary_df = run_ma5_deviation_mean_reversion_backtest(
            symbols=list(universe.keys()),
            top_n=1,
            entry_timing="next_open",
            exit_timing="next_open",
            weight_mode="equal",
            min_avg_dollar_volume=10_000_000.0,
        )

        self.assertFalse(trades_df.empty)
        self.assertFalse(summary_df.empty)
        self.assertIn("symbol", trades_df.columns)
        self.assertIn("direction", trades_df.columns)
        self.assertIn("entry_date", trades_df.columns)
        self.assertIn("exit_date", trades_df.columns)
        self.assertIn("pnl", trades_df.columns)
        self.assertIn("exit_reason", trades_df.columns)

        self.assertIn("cagr", summary_df.columns)
        self.assertIn("sharpe", summary_df.columns)
        self.assertIn("max_drawdown", summary_df.columns)
        self.assertIn("win_rate", summary_df.columns)
        self.assertIn("trade_count", summary_df.columns)

    @patch("app.backtest.load_price_csv")
    def test_supports_same_close_and_volatility_adjusted_weights(self, mock_load_price_csv):
        universe = {
            "1111": self._build_price_df(seed=11, base_price=2100.0, amp=80.0),
            "2222": self._build_price_df(seed=12, base_price=2600.0, amp=100.0),
            "3333": self._build_price_df(seed=13, base_price=1900.0, amp=60.0),
            "4444": self._build_price_df(seed=14, base_price=3200.0, amp=140.0),
        }
        mock_load_price_csv.side_effect = lambda symbol: universe[symbol]

        trades_df, summary_df = run_ma5_deviation_mean_reversion_backtest(
            symbols=list(universe.keys()),
            top_n=1,
            entry_timing="same_close",
            exit_timing="next_open",
            weight_mode="volatility_adjusted",
            min_avg_dollar_volume=10_000_000.0,
        )

        self.assertFalse(trades_df.empty)
        self.assertFalse(summary_df.empty)
        self.assertEqual(summary_df.iloc[0]["weight_mode"], "volatility_adjusted")


if __name__ == "__main__":
    unittest.main()
