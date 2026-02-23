import os
from datetime import date, timedelta
from typing import Dict, List, Optional

import pandas as pd
import requests


EDINET_BASE_URL_DEFAULT = "https://disclosure.edinet-fsa.go.jp/api/v2"
DOCUMENTS_ENDPOINT = "/documents.json"


class EdinetClient:
    """EDINET API から提出書類一覧を取得するクライアント。"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        self.api_key = (api_key or os.getenv("EDINET_API_KEY") or "").strip() or None
        self.base_url = (base_url or os.getenv("EDINET_BASE_URL", EDINET_BASE_URL_DEFAULT)).rstrip("/")

    def _request(self, path: str, params: Dict[str, str]) -> Dict:
        url = f"{self.base_url}{path}"
        request_params = dict(params)
        if self.api_key:
            request_params["Subscription-Key"] = self.api_key

        response = requests.get(url, params=request_params, timeout=20)
        if not response.ok:
            raise ValueError(f"EDINET API request failed: {response.status_code} {response.text}")

        try:
            return response.json()
        except ValueError as exc:
            raise ValueError("Failed to parse EDINET API response as JSON") from exc

    def fetch_documents(self, target_date: str) -> pd.DataFrame:
        payload = self._request(
            DOCUMENTS_ENDPOINT,
            params={"date": target_date, "type": "2"},
        )
        results = payload.get("results")
        if not isinstance(results, list):
            return pd.DataFrame()
        return pd.DataFrame(results)


def normalize_edinet_documents(df_raw: pd.DataFrame) -> pd.DataFrame:
    if df_raw is None or df_raw.empty:
        return pd.DataFrame(
            columns=[
                "date",
                "code",
                "doc_id",
                "edinet_code",
                "filer_name",
                "doc_type_code",
                "doc_type_name",
                "form_code",
                "description",
                "submit_datetime",
            ]
        )

    df = df_raw.copy()

    sec_code = df.get("secCode")
    if sec_code is None:
        sec_code = pd.Series([None] * len(df), index=df.index)

    normalized_code = (
        sec_code.astype(str)
        .str.extract(r"(\d{4})", expand=False)
        .fillna("")
    )

    normalized = pd.DataFrame(
        {
            "date": pd.to_datetime(df.get("submitDateTime"), errors="coerce").dt.date.astype(str),
            "code": normalized_code,
            "doc_id": df.get("docID"),
            "edinet_code": df.get("edinetCode"),
            "filer_name": df.get("filerName"),
            "doc_type_code": df.get("docTypeCode"),
            "doc_type_name": df.get("docDescription"),
            "form_code": df.get("formCode"),
            "description": df.get("docDescription"),
            "submit_datetime": pd.to_datetime(df.get("submitDateTime"), errors="coerce"),
        }
    )

    normalized = normalized[normalized["code"] != ""]
    normalized = normalized.dropna(subset=["doc_id", "submit_datetime"])
    normalized["submit_datetime"] = normalized["submit_datetime"].dt.strftime("%Y-%m-%dT%H:%M:%S")
    normalized = normalized.sort_values(["submit_datetime", "code"]).reset_index(drop=True)
    return normalized


def date_range_descending(days: int, end_date: Optional[date] = None) -> List[str]:
    if days <= 0:
        return []
    end = end_date or date.today()
    return [(end - timedelta(days=offset)).isoformat() for offset in range(days)]
