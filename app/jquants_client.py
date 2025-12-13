import os
from typing import Dict, Optional

import pandas as pd
import requests


class JQuantsClient:
    """Simple client for fetching J-Quants daily quotes."""

    def __init__(
        self,
        refresh_token: Optional[str] = None,
        base_url: Optional[str] = None,
        mailaddress: Optional[str] = None,
        password: Optional[str] = None,
    ):
        self.refresh_token = refresh_token or os.getenv("JQUANTS_REFRESH_TOKEN")
        self.mailaddress = mailaddress or os.getenv("MAILADDRESS")
        self.password = password or os.getenv("PASSWORD")
        self.base_url = (base_url or os.getenv("JQUANTS_BASE_URL", "https://api.jquants.com")).rstrip("/")
        self._id_token: Optional[str] = None
        self._access_token: Optional[str] = None

    def _request(self, method: str, path: str, **kwargs) -> Dict:
        url = f"{self.base_url}{path}"
        response = requests.request(method, url, timeout=10, **kwargs)
        if not response.ok:
            raise ValueError(f"J-Quants API request failed: {response.status_code} {response.text}")
        try:
            return response.json()
        except ValueError as exc:  # pragma: no cover - defensive
            raise ValueError("Failed to parse J-Quants API response as JSON") from exc

    def authenticate(self) -> str:
        """Obtain and cache an access token using the refresh token."""
        if not self.mailaddress:
            raise ValueError("MAILADDRESS is not set.")

        if self._access_token:
            return self._access_token

        refresh_token = self.refresh_token
        if not refresh_token:
            if not self.password:
                raise ValueError("PASSWORD is not set.")

            auth_payload = {"mailaddress": self.mailaddress, "password": self.password}
            auth_data = self._request("POST", "/v1/token/auth_user", json=auth_payload)
            refresh_token = auth_data.get("refreshToken")
            if not refresh_token:
                raise ValueError("refreshToken was not returned from J-Quants auth_user endpoint.")

            self.refresh_token = refresh_token

        refresh_payload = {"refreshToken": refresh_token}
        refresh_data = self._request("POST", "/v1/token/auth_refresh", json=refresh_payload)
        self._id_token = refresh_data.get("idToken")
        self._access_token = refresh_data.get("accessToken")
        if not self._access_token:
            raise ValueError("accessToken was not returned from J-Quants auth_refresh endpoint.")

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
