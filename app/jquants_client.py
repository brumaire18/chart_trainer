import os
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, Iterable, Optional, Union


def _normalize_token(value: Optional[str]) -> Optional[str]:
    """Return a stripped token string or None if empty."""

    if value is None:
        return None
    cleaned = value.strip()
    return cleaned if cleaned else None

import pandas as pd
import requests


DAILY_QUOTES_ENDPOINT = "/v2/prices/daily_quotes"
TOPIX_ENDPOINT = "/v2/indices/topix"


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
        refresh_token: Optional[str] = None,
        refresh_token_expires_at: Optional[str] = None,
        base_url: Optional[str] = None,
        mailaddress: Optional[str] = None,
        password: Optional[str] = None,
    ):
        self.refresh_token = refresh_token or os.getenv("JQUANTS_REFRESH_TOKEN")
        self.refresh_token_expires_at = refresh_token_expires_at
        self.mailaddress = mailaddress or os.getenv("MAILADDRESS")
        self.password = password or os.getenv("PASSWORD")
        self.base_url = (base_url or os.getenv("JQUANTS_BASE_URL", "https://api.jquants.com")).rstrip("/")
        self._id_token: Optional[str] = None
        self._access_token: Optional[str] = None  # Backward-compat alias for id token

        self._debug(
            "initialized client "
            f"base_url={self.base_url} "
            f"has_mailaddress={bool(self.mailaddress)} "
            f"has_password={bool(self.password)} "
            f"has_refresh_token={bool(self.refresh_token)}"
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

    def _parse_refresh_expiry(self, value: Optional[Union[str, int, float]]) -> Optional[datetime]:
        """Best-effort parsing for refresh token expiration timestamps."""

        if value is None:
            return None

        try:
            # Numeric epoch seconds (can arrive as int/float or numeric string)
            if isinstance(value, (int, float)) or (isinstance(value, str) and value.strip().replace(".", "", 1).isdigit()):
                return datetime.fromtimestamp(float(value), tz=timezone.utc)

            # ISO-8601 strings; "Z" suffix is normalized to UTC
            normalized = value.strip().replace("Z", "+00:00")
            return datetime.fromisoformat(normalized)
        except Exception:
            return None

    def _refresh_token_is_valid(self) -> bool:
        """Check whether the cached refresh token is present and unexpired."""

        if not self.refresh_token:
            return False

        expiry = self._parse_refresh_expiry(self.refresh_token_expires_at)
        if expiry is None:
            return True

        return datetime.now(timezone.utc) < expiry

    def _extract_refresh_metadata(self, payload: Dict) -> Dict[str, Optional[str]]:
        """Normalize refresh token and expiration fields from API responses."""

        refresh_token = _normalize_token(
            payload.get("refresh_token")
            or payload.get("refreshToken")
            or payload.get("refreshtoken")
        )

        expiry_at_keys = (
            "refresh_token_expires_at",
            "refresh_token_expiration",
            "refreshTokenExpiresAt",
            "refreshTokenExpiration",
            "refreshTokenExpires",
            "refreshTokenExpiresAt",
        )
        expiry_in_keys = (
            "refresh_token_expires_in",
            "refreshTokenExpiresIn",
            "refresh_token_expires",
        )

        refresh_expires_at: Optional[str] = None

        for key in expiry_at_keys:
            if payload.get(key) is not None:
                refresh_expires_at = payload.get(key)
                break

        if refresh_expires_at is None:
            for key in expiry_in_keys:
                expires_in_value = payload.get(key)
                if expires_in_value is None:
                    continue
                try:
                    seconds = float(expires_in_value)
                    refresh_expires_at = (
                        datetime.now(timezone.utc) + timedelta(seconds=seconds)
                    ).isoformat()
                    break
                except (TypeError, ValueError):
                    continue

        return {"refresh_token": refresh_token, "refresh_token_expires_at": refresh_expires_at}

    def create_refresh_token(self) -> str:
        """Generate and store a refresh token along with its expiration."""

        if not self.mailaddress:
            raise ValueError("MAILADDRESS is not set.")
        if not self.password:
            raise ValueError("PASSWORD is not set.")

        self._debug("creating refresh token using configured mailaddress/password")
        auth_payload = {"mailaddress": self.mailaddress, "password": self.password}
        auth_data = self._request("POST", "/v2/token/auth_user", json=auth_payload)

        refresh_metadata = self._extract_refresh_metadata(auth_data)
        refresh_token = refresh_metadata.get("refresh_token")
        if not refresh_token:
            raise ValueError("refresh_token was not returned from J-Quants auth_user endpoint.")

        self.refresh_token = refresh_token
        self.refresh_token_expires_at = refresh_metadata.get("refresh_token_expires_at")

        preview = f"{refresh_token[:4]}...{refresh_token[-4:]}" if len(refresh_token) > 8 else "<short>"
        self._debug(
            f"obtained new refresh token length={len(refresh_token)} preview={preview} "
            f"expires_at={self.refresh_token_expires_at}"
        )

        return refresh_token

    def authenticate(self) -> str:
        """Obtain and cache an id token using the refresh token."""

        if self._id_token:
            self._debug("using cached id token")
            return self._id_token

        refresh_token = _normalize_token(self.refresh_token)
        if not refresh_token or not self._refresh_token_is_valid():
            self._debug(
                "refresh token missing or expired; requesting new one via auth_user"
            )
            refresh_token = self.create_refresh_token()

        preview = f"{refresh_token[:4]}...{refresh_token[-4:]}" if len(refresh_token) > 8 else "<short>"
        self._debug(
            f"requesting id token via auth_refresh length={len(refresh_token)} "
            f"preview={preview} expires_at={self.refresh_token_expires_at}"
        )

        refresh_payload = {"refresh_token": refresh_token}
        refresh_data = self._request("POST", "/v2/token/auth_refresh", json=refresh_payload)

        refresh_metadata = self._extract_refresh_metadata(refresh_data)

        new_refresh_token = refresh_metadata.get("refresh_token")
        if new_refresh_token and new_refresh_token != refresh_token:
            preview_new = (
                f"{new_refresh_token[:4]}...{new_refresh_token[-4:]}"
                if len(new_refresh_token) > 8
                else "<short>"
            )
            self._debug(
                f"refresh token rotated length={len(new_refresh_token)} "
                f"preview={preview_new}"
            )
            self.refresh_token = new_refresh_token

        if refresh_metadata.get("refresh_token_expires_at"):
            self.refresh_token_expires_at = refresh_metadata["refresh_token_expires_at"]

        self._id_token = _normalize_token(
            refresh_data.get("id_token")
            or refresh_data.get("idToken")
            or refresh_data.get("token")
        )
        if not self._id_token:
            raise ValueError("id_token was not returned from J-Quants auth_refresh endpoint.")

        self._access_token = self._id_token
        self._debug(
            f"successfully obtained id token length={len(self._id_token)} preview="
            f"{self._id_token[:4]}...{self._id_token[-4:]}"
        )

        return self._id_token

    def fetch_daily_quotes(self, code: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Retrieve OHLCV daily quotes for the specified code and date range."""
        id_token = self.authenticate()
        headers = {"Authorization": f"Bearer {id_token}"}
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

        id_token = self.authenticate()
        headers = {"Authorization": f"Bearer {id_token}"}
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
