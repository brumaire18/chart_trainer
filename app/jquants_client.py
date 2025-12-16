import os
import time
from datetime import datetime, timezone
from typing import Dict, Optional, Union


def _normalize_token(value: Optional[str]) -> Optional[str]:
    """Return a stripped token string or None if empty."""

    if value is None:
        return None
    cleaned = value.strip()
    return cleaned if cleaned else None

import pandas as pd
import requests


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

    def _request(self, method: str, path: str, **kwargs) -> Dict:
        url = f"{self.base_url}{path}"
        max_retries = 3
        base_retry_delay = 5.0

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
                error_message = response.text
                if response.status_code == 429:
                    error_message = (
                        f"Rate limit exceeded after {max_retries} attempts. "
                        "Please wait before retrying."
                    )
                raise ValueError(f"J-Quants API request failed: {response.status_code} {error_message}")

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

    def create_refresh_token(self) -> str:
        """Generate and store a refresh token along with its expiration."""

        if not self.mailaddress:
            raise ValueError("MAILADDRESS is not set.")
        if not self.password:
            raise ValueError("PASSWORD is not set.")

        self._debug("creating refresh token using configured mailaddress/password")
        auth_payload = {"mailaddress": self.mailaddress, "password": self.password}
        auth_data = self._request("POST", "/v1/token/auth_user", json=auth_payload)

        refresh_token = _normalize_token(auth_data.get("refreshToken"))
        if not refresh_token:
            raise ValueError("refreshToken was not returned from J-Quants auth_user endpoint.")

        self.refresh_token = refresh_token

        expiry_keys = (
            "refreshTokenExpiresAt",
            "refreshTokenExpiration",
            "refreshTokenExpires",
            "refreshTokenExpiresIn",
        )
        self.refresh_token_expires_at = next(
            (auth_data.get(key) for key in expiry_keys if auth_data.get(key) is not None),
            None,
        )

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

        refresh_params = {"refreshtoken": refresh_token}
        refresh_data = self._request("POST", "/v1/token/auth_refresh", params=refresh_params)

        new_refresh_token = _normalize_token(
            refresh_data.get("refreshToken") or refresh_data.get("refreshtoken")
        )
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

        expiry_keys = (
            "refreshTokenExpiresAt",
            "refreshTokenExpiration",
            "refreshTokenExpires",
            "refreshTokenExpiresIn",
        )
        self.refresh_token_expires_at = next(
            (refresh_data.get(key) for key in expiry_keys if refresh_data.get(key) is not None),
            self.refresh_token_expires_at,
        )

        self._id_token = _normalize_token(refresh_data.get("idToken"))
        if not self._id_token:
            raise ValueError("idToken was not returned from J-Quants auth_refresh endpoint.")

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
        params = {"code": code, "from": start_date, "to": end_date}

        data = self._request("GET", "/v1/prices/daily_quotes", headers=headers, params=params)
        quotes = data.get("daily_quotes") or data.get("dailyQuotes")
        if quotes is None:
            raise ValueError("daily_quotes data was not found in the API response.")

        df = pd.DataFrame(quotes)
        if df.empty:
            raise ValueError("No price data returned for the requested symbol and date range.")

        return df
