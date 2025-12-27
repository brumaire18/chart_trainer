import os
import time
from typing import Dict, Iterable, Optional


def _normalize_token(value: Optional[str]) -> Optional[str]:
    """Return a stripped token string or None if empty."""

    if value is None:
        return None
    cleaned = value.strip()
    return cleaned if cleaned else None

import pandas as pd
import requests


DAILY_QUOTES_ENDPOINT = "/v2/equities/bars/daily"
TOPIX_ENDPOINT = "/v2/indices/bars/daily/topix"


def _extract_paginated_rows(payload: Dict, candidate_keys: Iterable[str]) -> Optional[list]:
    for key in candidate_keys:
        if key in payload:
            return payload.get(key)
    return None


def _extract_pagination_key(payload: Dict) -> Optional[str]:
    for key in (
        "pagination_key",
        "paginationKey",
        "paginationkey",
        "next_pagination_key",
        "nextPaginationKey",
    ):
        value = payload.get(key)
        if value:
            return str(value)
    return None


class JQuantsClient:
    """Simple client for fetching J-Quants daily quotes."""

    # Backward-compatible instance helper retained so callers expecting
    # the old method name do not crash with an AttributeError.
    def _normalize_token(self, value: Optional[str]) -> Optional[str]:
        return _normalize_token(value)

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        self.api_key = api_key or os.getenv("JQUANTS_API_KEY")
        self.base_url = (base_url or os.getenv("JQUANTS_BASE_URL", "https://api.jquants.com")).rstrip("/")

        self._debug(
            "initialized client "
            f"base_url={self.base_url} "
            f"has_api_key={bool(self.api_key)}"
        )

    @staticmethod
    def _debug(message: str) -> None:
        """Lightweight console logger for debugging."""
        print(f"[JQuantsClient] {message}")

    @staticmethod
    def _extract_error_detail(response: requests.Response) -> str:
        """Return a human-readable error detail from the J-Quants API response."""

        try:
            payload = response.json()
            if isinstance(payload, dict):
                if payload.get("errors") and isinstance(payload["errors"], list):
                    first_error = payload["errors"][0]
                    if isinstance(first_error, dict):
                        message = first_error.get("message") or first_error.get("detail")
                        if message:
                            return str(message)
                    if first_error:
                        return str(first_error)
                if payload.get("message"):
                    return str(payload["message"])
                if payload.get("detail"):
                    return str(payload["detail"])
                if payload.get("error"):
                    if isinstance(payload["error"], dict):
                        return str(
                            payload["error"].get("message")
                            or payload["error"].get("code")
                            or payload["error"]
                        )
                    return str(payload["error"])
        except ValueError:
            pass

        return response.text or ""

    def _request(self, method: str, path: str, **kwargs) -> Dict:
        url = f"{self.base_url}{path}"
        max_retries = 3
        base_retry_delay = 5.0

        headers = kwargs.pop("headers", {}) or {}
        default_headers = {"Accept": "application/json"}
        if method.upper() == "POST" and "json" in kwargs and "Content-Type" not in headers:
            default_headers["Content-Type"] = "application/json"
        merged_headers = {**default_headers, **headers}
        kwargs["headers"] = merged_headers

        for attempt in range(max_retries):
            response = requests.request(method, url, timeout=10, **kwargs)
            if response.status_code == 429 and attempt < max_retries - 1:
                retry_after = response.headers.get("Retry-After")
                try:
                    delay = float(retry_after) if retry_after is not None else base_retry_delay * (attempt + 1)
                except ValueError:
                    delay = base_retry_delay * (attempt + 1)
                time.sleep(delay)
                continue

            if not response.ok:
                error_message = self._extract_error_detail(response)
                if response.status_code == 429:
                    error_message = (
                        f"Rate limit exceeded after {max_retries} attempts. "
                        "Please wait before retrying."
                    )
                raise ValueError(
                    f"J-Quants API request failed: {response.status_code} {error_message.strip()}"
                )

            try:
                return response.json()
            except ValueError as exc:  # pragma: no cover - defensive
                raise ValueError("Failed to parse J-Quants API response as JSON") from exc

    def authenticate(self) -> str:
        """Return the configured API key."""

        api_key = _normalize_token(self.api_key)
        if not api_key:
            raise ValueError("JQUANTS_API_KEY is not set.")
        return api_key

    def fetch_daily_quotes(self, code: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Retrieve OHLCV daily quotes for the specified code and date range."""
        api_key = self.authenticate()
        headers = {"x-api-key": api_key}
        params = {"symbol": code, "from": start_date, "to": end_date}

        all_rows = []
        pagination_key: Optional[str] = None
        while True:
            request_params = dict(params)
            if pagination_key:
                request_params["pagination_key"] = pagination_key

            data = self._request(
                "GET", DAILY_QUOTES_ENDPOINT, headers=headers, params=request_params
            )
            quotes = _extract_paginated_rows(
                data,
                (
                    "data",
                    "dailyQuotes",
                    "daily_quotes",
                    "prices",
                    "quotes",
                ),
            )
            if quotes is None:
                raise ValueError("dailyQuotes data was not found in the API response.")

            all_rows.extend(quotes)
            pagination_key = _extract_pagination_key(data)
            if not pagination_key:
                break

        df = pd.DataFrame(all_rows)
        if df.empty:
            raise ValueError("No price data returned for the requested symbol and date range.")

        return df

    def fetch_topix(self, from_date: str, to_date: str) -> pd.DataFrame:
        """TOPIX 指数を指定期間で取得する。"""

        api_key = self.authenticate()
        headers = {"x-api-key": api_key}
        base_params = {"from": from_date, "to": to_date}

        all_rows = []
        pagination_key: Optional[str] = None
        while True:
            params = dict(base_params)
            if pagination_key:
                params["pagination_key"] = pagination_key

            data = self._request("GET", TOPIX_ENDPOINT, headers=headers, params=params)
            rows = _extract_paginated_rows(
                data,
                (
                    "data",
                    "topix",
                    "indices",
                    "indexQuotes",
                ),
            )
            if rows is None:
                raise ValueError("topix data was not found in the API response.")

            all_rows.extend(rows)
            pagination_key = _extract_pagination_key(data)
            if not pagination_key:
                break

        df = pd.DataFrame(all_rows)
        if df.empty:
            raise ValueError("No TOPIX data returned for the requested date range.")

        return df
