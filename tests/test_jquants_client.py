import unittest
from datetime import datetime, timezone
from unittest.mock import patch

from app.jquants_client import DAILY_QUOTES_ENDPOINT, TOPIX_ENDPOINT, JQuantsClient


class _MockResponse:
    def __init__(self, status_code=200, json_data=None, headers=None, text=""):
        self.status_code = status_code
        self._json_data = json_data or {}
        self.headers = headers or {}
        self.text = text

    @property
    def ok(self):
        return 200 <= self.status_code < 300

    def json(self):
        return self._json_data


class JQuantsClientAuthTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = JQuantsClient(mailaddress="user@example.com", password="pw")

    @patch("app.jquants_client.requests.request")
    def test_create_refresh_token_uses_v2_and_parses_fields(self, mock_request):
        mock_request.return_value = _MockResponse(
            json_data={
                "refresh_token": "abcd1234",
                "refresh_token_expires_at": "2025-01-01T00:00:00Z",
            }
        )

        token = self.client.create_refresh_token()

        self.assertEqual(token, "abcd1234")
        self.assertEqual(self.client.refresh_token, "abcd1234")
        self.assertEqual(self.client.refresh_token_expires_at, "2025-01-01T00:00:00Z")

        args, kwargs = mock_request.call_args
        self.assertEqual(args[0], "POST")
        self.assertEqual(args[1], f"{self.client.base_url}/v2/token/auth_user")
        self.assertEqual(kwargs["json"], {"mailaddress": "user@example.com", "password": "pw"})
        self.assertEqual(kwargs["headers"]["Accept"], "application/json")
        self.assertEqual(kwargs["headers"]["Content-Type"], "application/json")

    @patch("app.jquants_client.requests.request")
    def test_authenticate_parses_refresh_rotation_and_id_token(self, mock_request):
        # Ensure refresh token is considered valid
        self.client.refresh_token = "currenttoken"
        self.client.refresh_token_expires_at = "2999-01-01T00:00:00Z"

        mock_request.return_value = _MockResponse(
            json_data={
                "id_token": "new-id-token",
                "refresh_token": "rotatedtoken",
                "refresh_token_expires_at": "2999-12-31T00:00:00Z",
            }
        )

        id_token = self.client.authenticate()

        self.assertEqual(id_token, "new-id-token")
        self.assertEqual(self.client.refresh_token, "rotatedtoken")
        self.assertEqual(self.client.refresh_token_expires_at, "2999-12-31T00:00:00Z")

        args, kwargs = mock_request.call_args
        self.assertEqual(args[0], "POST")
        self.assertEqual(args[1], f"{self.client.base_url}/v2/token/auth_refresh")
        self.assertEqual(kwargs["json"], {"refresh_token": "currenttoken"})
        self.assertEqual(kwargs["headers"]["Accept"], "application/json")
        self.assertEqual(kwargs["headers"]["Content-Type"], "application/json")

    @patch("app.jquants_client.requests.request")
    def test_request_raises_error_with_message(self, mock_request):
        mock_request.return_value = _MockResponse(
            status_code=400, json_data={"message": "invalid credentials"}
        )

        with self.assertRaises(ValueError) as ctx:
            self.client._request("POST", "/v2/token/auth_user", json={})

        self.assertIn("400", str(ctx.exception))
        self.assertIn("invalid credentials", str(ctx.exception))

    def test_fetch_daily_quotes_paginates(self):
        client = JQuantsClient(refresh_token="refresh")

        with patch.object(client, "authenticate", return_value="id-token"):
            with patch.object(
                client,
                "_request",
                side_effect=[
                    {"dailyQuotes": [{"date": "2024-01-01"}], "paginationKey": "next"},
                    {"dailyQuotes": [{"date": "2024-01-02"}]},
                ],
            ) as mock_request:
                df = client.fetch_daily_quotes("7203", "2024-01-01", "2024-01-02")

        self.assertEqual(len(df), 2)
        first_call = mock_request.call_args_list[0]
        second_call = mock_request.call_args_list[1]
        self.assertEqual(first_call.args[0], "GET")
        self.assertEqual(first_call.args[1], DAILY_QUOTES_ENDPOINT)
        self.assertEqual(
            first_call.kwargs["params"],
            {"symbol": "7203", "from": "2024-01-01", "to": "2024-01-02"},
        )
        self.assertEqual(second_call.args[1], DAILY_QUOTES_ENDPOINT)
        self.assertEqual(
            second_call.kwargs["params"],
            {
                "symbol": "7203",
                "from": "2024-01-01",
                "to": "2024-01-02",
                "pagination_key": "next",
            },
        )

    def test_fetch_topix_without_pagination(self):
        client = JQuantsClient(refresh_token="refresh")

        with patch.object(client, "authenticate", return_value="id-token"):
            with patch.object(client, "_request", return_value={"indices": [{"date": "2024-01-01"}]}) as mock_request:
                df = client.fetch_topix("2024-01-01", "2024-01-02")

        self.assertEqual(len(df), 1)
        args, kwargs = mock_request.call_args
        self.assertEqual(args[0], "GET")
        self.assertEqual(args[1], TOPIX_ENDPOINT)
        self.assertEqual(kwargs["params"], {"from": "2024-01-01", "to": "2024-01-02"})


if __name__ == "__main__":
    unittest.main()
