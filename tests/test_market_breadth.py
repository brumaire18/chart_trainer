import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from app.market_breadth import aggregate_market_breadth


class AggregateMarketBreadthTests(unittest.TestCase):
    def test_excludes_market_holidays_by_default(self):
        with TemporaryDirectory() as tmp:
            price_dir = Path(tmp)
            pd.DataFrame(
                {
                    "date": ["2024-01-04", "2024-01-05", "2024-01-09"],
                    "open": [100, 101, 103],
                    "high": [101, 103, 104],
                    "low": [99, 100, 102],
                    "close": [100, 102, 101],
                    "volume": [1000, 1100, 1200],
                }
            ).to_csv(price_dir / "1111.csv", index=False)

            df = aggregate_market_breadth(symbols=["1111"], price_dir=price_dir)

            self.assertListEqual(
                df["date"].dt.strftime("%Y-%m-%d").tolist(),
                ["2024-01-04", "2024-01-05", "2024-01-09"],
            )

    def test_can_fill_missing_business_days_when_enabled(self):
        with TemporaryDirectory() as tmp:
            price_dir = Path(tmp)
            pd.DataFrame(
                {
                    "date": ["2024-01-04", "2024-01-05", "2024-01-09"],
                    "open": [100, 101, 103],
                    "high": [101, 103, 104],
                    "low": [99, 100, 102],
                    "close": [100, 102, 101],
                    "volume": [1000, 1100, 1200],
                }
            ).to_csv(price_dir / "1111.csv", index=False)

            df = aggregate_market_breadth(
                symbols=["1111"],
                price_dir=price_dir,
                fill_missing_business_days=True,
            )

            self.assertIn("2024-01-08", df["date"].dt.strftime("%Y-%m-%d").tolist())


if __name__ == "__main__":
    unittest.main()
