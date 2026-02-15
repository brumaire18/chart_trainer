import unittest
from unittest.mock import patch

import pandas as pd

from app.jquants_fetcher import filter_listed_master_for_analysis, get_growth_universe


class ListedMasterFilterTests(unittest.TestCase):
    def test_filters_pro_delisted_and_preferred(self):
        df = pd.DataFrame(
            {
                "code": ["1111", "2222", "3333", "4444"],
                "name": ["通常銘柄", "優先株式A", "通常銘柄B", "通常銘柄C"],
                "market": ["PRIME", "PRIME", "PRO Market", "STANDARD"],
                "listing": ["1", "1", "1", "0"],
            }
        )

        filtered = filter_listed_master_for_analysis(df)

        self.assertEqual(filtered["code"].tolist(), ["1111"])


class UniverseFilterTests(unittest.TestCase):
    @patch("app.jquants_fetcher.load_listed_master")
    def test_growth_universe_excludes_preferred_and_pro(self, mock_load_listed_master):
        mock_load_listed_master.return_value = pd.DataFrame(
            {
                "code": ["1", "2", "3", "4"],
                "name": ["通常", "優先株", "通常", "通常"],
                "market": ["GROWTH", "GROWTH", "PRO Market", "GROWTH"],
                "listing": ["1", "1", "1", "0"],
            }
        )

        result = get_growth_universe()

        self.assertEqual(result, ["0001"])


if __name__ == "__main__":
    unittest.main()
