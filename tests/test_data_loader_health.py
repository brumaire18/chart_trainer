import unittest

import pandas as pd

from app.data_loader import inspect_price_data_health


class InspectPriceDataHealthTest(unittest.TestCase):
    def test_detects_recent_missing_and_stale_days(self):
        df = pd.DataFrame(
            {
                "date": [
                    "2026-04-06",
                    "2026-04-07",
                    "2026-04-09",
                    "2026-04-10",
                ],
                "open": [1, 1, 1, 1],
                "high": [1, 1, 1, 1],
                "low": [1, 1, 1, 1],
                "close": [1, 1, 1, 1],
                "volume": [100, 100, 100, 100],
            }
        )

        health = inspect_price_data_health(df, market="US", recent_business_days=5)

        self.assertIn("2026-04-08", health["missing_recent_dates"])
        self.assertGreaterEqual(health["stale_days"], 1)
        self.assertGreater(health["recent_missing_rate"], 0.0)

    def test_detects_duplicate_and_missing_value_rows(self):
        df = pd.DataFrame(
            {
                "date": ["2026-04-07", "2026-04-07", "2026-04-08"],
                "open": [1, 1, None],
                "high": [1, 1, 1],
                "low": [1, 1, 1],
                "close": [1, 1, 1],
                "volume": [100, 100, 100],
            }
        )

        health = inspect_price_data_health(df, market="JP", recent_business_days=5)

        self.assertEqual(health["duplicate_dates_count"], 1)
        self.assertEqual(health["missing_value_rows"], 1)


if __name__ == "__main__":
    unittest.main()
