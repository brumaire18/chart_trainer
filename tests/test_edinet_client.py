import unittest
from unittest.mock import MagicMock, patch

import pandas as pd

from app.edinet_client import EdinetClient, date_range_descending, normalize_edinet_documents


class EdinetClientTests(unittest.TestCase):
    @patch("app.edinet_client.requests.get")
    def test_fetch_documents_returns_dataframe(self, mock_get):
        response = MagicMock()
        response.ok = True
        response.json.return_value = {
            "results": [
                {
                    "docID": "S100TEST",
                    "secCode": "72030",
                    "filerName": "トヨタ自動車",
                    "submitDateTime": "2024-01-05T10:30:00",
                }
            ]
        }
        mock_get.return_value = response

        client = EdinetClient(api_key="dummy")
        df = client.fetch_documents("2024-01-05")

        self.assertEqual(len(df), 1)
        self.assertEqual(df.loc[0, "docID"], "S100TEST")

    def test_normalize_edinet_documents(self):
        raw = pd.DataFrame(
            [
                {
                    "docID": "S100TEST",
                    "secCode": "72030",
                    "edinetCode": "E00001",
                    "filerName": "トヨタ自動車",
                    "docTypeCode": "120",
                    "docDescription": "有価証券報告書",
                    "formCode": "030000",
                    "submitDateTime": "2024-01-05T10:30:00",
                }
            ]
        )

        normalized = normalize_edinet_documents(raw)

        self.assertEqual(normalized.loc[0, "code"], "7203")
        self.assertEqual(normalized.loc[0, "doc_id"], "S100TEST")
        self.assertEqual(normalized.loc[0, "filer_name"], "トヨタ自動車")

    def test_date_range_descending(self):
        days = date_range_descending(3, end_date=pd.Timestamp("2024-01-05").date())
        self.assertEqual(days, ["2024-01-05", "2024-01-04", "2024-01-03"])


if __name__ == "__main__":
    unittest.main()
