import os
from typing import Dict, Optional


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
        self.refresh_token = _normalize_token(
            refresh_token or os.getenv("JQUANTS_REFRESH_TOKEN")
        )
        self.refresh_token_expires_at = refresh_token_expires_at
        self.mailaddress = mailaddress or os.getenv("MAILADDRESS")
        self.password = password or os.getenv("PASSWORD")
        self.base_url = (base_url or os.getenv("JQUANTS_BASE_URL", "https://api.jquants.com")).rstrip("/")
        self._id_token: Optional[str] = None
        self._access_token: Optional[str] = None

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
        json_payload = kwargs.get("json")
        json_keys = list(json_payload.keys()) if isinstance(json_payload, dict) else None
        params = kwargs.get("params")
        self._debug(
            "sending request "
            f"method={method} path={path} params={params} json_keys={json_keys}"
        )
        response = requests.request(method, url, timeout=10, **kwargs)
        if not response.ok:
            self._debug(
                f"request failed status={response.status_code} body={response.text}"
            )
            raise ValueError(f"J-Quants API request failed: {response.status_code} {response.text}")
        self._debug(f"request succeeded status={response.status_code}")
        try:
            return response.json()
        except ValueError as exc:  # pragma: no cover - defensive
            raise ValueError("Failed to parse J-Quants API response as JSON") from exc

    def create_refresh_token(self) -> str:
        """Generate and store a refresh token along with its expiration."""

        if not self.mailaddress:
            raise ValueError("MAILADDRESS is not set.")
        if not self.password:
            raise ValueError("PASSWORD is not set.")

        self._debug(
            "creating refresh token using configured mailaddress/password"
        )
        auth_payload = {"mailaddress": self.mailaddress, "password": self.password}
        auth_data = self._request("POST", "/v1/token/auth_user", json=auth_payload)

        refresh_token = _normalize_token(auth_data.get("refreshToken"))
        if not refresh_token:
            raise ValueError("refreshToken was not returned from J-Quants auth_user endpoint.")

        self.refresh_token = refresh_token
        self._debug("received refresh token from auth_user")

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

        return refresh_token

    def authenticate(self) -> str:
        """Obtain and cache an access token using the refresh token."""
        if not self.mailaddress:
            raise ValueError("MAILADDRESS is not set.")

        if self._access_token:
            self._debug("using cached access token")
            return self._access_token

        refresh_token = self.refresh_token
        if not refresh_token:
            self._debug("no refresh token configured; generating via auth_user")
            refresh_token = self.create_refresh_token()
        else:
            preview = f"{refresh_token[:4]}...{refresh_token[-4:]}" if len(refresh_token) > 8 else "<short>"
            self._debug(
                "using provided refresh token for auth_refresh "
                f"length={len(refresh_token)} preview={preview}"
            )

        # The refresh endpoint expects a lower-case "refreshtoken" query parameter as
        # documented at https://jpx.gitbook.io/j-quants-ja/api-reference/refreshtoken.
        refresh_params = {"refreshtoken": refresh_token}
        refresh_data = self._request(
            "POST", "/v1/token/auth_refresh", params=refresh_params
        )
        new_refresh_token = _normalize_token(
            refresh_data.get("refreshToken") or refresh_data.get("refreshtoken")
        )
        if new_refresh_token:
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
        self._id_token = refresh_data.get("idToken")
        self._access_token = refresh_data.get("accessToken")
        if not self._access_token:
            raise ValueError("accessToken was not returned from J-Quants auth_refresh endpoint.")

        self._debug("successfully obtained access token")

        return self._access_token

    def fetch_daily_quotes(self, code: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Retrieve OHLCV daily quotes for the specified code and date range."""
        access_token = self.authenticate()
        headers = {"Authorization": f"Bearer {access_token}"}
        params = {"code": code, "from": start_date, "to": end_date}

        data = self._request("GET", "/v1/prices/daily_quotes", headers=headers, params=params)
        quotes = data.get("daily_quotes") or data.get("dailyQuotes")
        if quotes is None:
            raise ValueError("daily_quotes data was not found in the API response.")

        df = pd.DataFrame(quotes)
        if df.empty:
            raise ValueError("No price data returned for the requested symbol and date range.")

        return df
