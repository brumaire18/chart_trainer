"""J-Quants から株価・銘柄マスタを取得するユーティリティ。

ライトプランの「直近約5年分取得可能」という制約を考慮し、
日足 CSV とメタ情報を整形・保存するための関数群を提供する。
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
import random
from dataclasses import dataclass
from datetime import date, datetime, time as dt_time, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import pandas as pd
import requests

from .config import (
    JQUANTS_BASE_URL,
    JQUANTS_API_KEY,
    META_DIR,
    PRICE_CSV_DIR,
)
from .jquants_client import DAILY_QUOTES_ENDPOINT, JQuantsClient, _normalize_token

logger = logging.getLogger(__name__)


class JQuantsError(RuntimeError):
    """J-Quants API 呼び出しに関する例外。"""


class JQuantsRateLimitError(JQuantsError):
    """J-Quants API のレートリミットに関する例外。"""


LIGHT_PLAN_WINDOW_DAYS = 365 * 5  # 約5年
LIGHT_PLAN_LAG_WEEKS = 0  # 直近データも取得可能
RATE_LIMIT_STATUS_CODES = {429, 503}
RATE_LIMIT_INITIAL_WAIT = 3600  # 秒（1時間）
RATE_LIMIT_MAX_WAIT = 21600  # 秒（6時間）
RATE_LIMIT_BACKOFF = 2.0
RATE_LIMIT_MAX_RETRIES = 5
RATE_LIMIT_MAX_RETRIES_PER_SYMBOL = 3
AUTH_ERROR_STATUS_CODES = {401}
AUTH_ERROR_WAIT = 600  # 秒（10分）
AUTH_ERROR_MAX_RETRIES = 3

TOPIX_CODE = "TOPIX"
TOPIX_CSV_PATH = PRICE_CSV_DIR / "topix.csv"
TOPIX_META_PATH = META_DIR / "topix.json"
LISTED_MASTER_ENDPOINT = "/v2/equities/master"

LISTED_MASTER_MARKET_CODE_MAP = {
    "0111": "PRIME",
    "0112": "STANDARD",
    "0113": "GROWTH",
    "0101": "PRIME",
    "0102": "STANDARD",
    "0103": "GROWTH",
}


@dataclass
class FetchResult:
    code: str
    market: Optional[str]
    df: pd.DataFrame
    meta_path: Path
    csv_path: Path


def _get_client() -> JQuantsClient:
    # 可能であれば実行時の環境変数を優先的に利用する
    runtime_api_key_raw = os.getenv("JQUANTS_API_KEY")
    runtime_api_key = _normalize_token(runtime_api_key_raw)
    configured_api_key = _normalize_token(JQUANTS_API_KEY)
    runtime_base_url = os.getenv("JQUANTS_BASE_URL")

    if runtime_api_key_raw and runtime_api_key_raw != runtime_api_key:
        logger.info(
            "JQUANTS_API_KEY の前後にある空白を除去しました (len=%s -> len=%s)",
            len(runtime_api_key_raw),
            len(runtime_api_key or ""),
        )

    snapshot = {
        "JQUANTS_API_KEY": bool(runtime_api_key or configured_api_key),
    }

    client = JQuantsClient(
        base_url=runtime_base_url or JQUANTS_BASE_URL,
        api_key=runtime_api_key or configured_api_key,
    )

    if not client.api_key:
        logger.error("必要な認証情報が見つかりませんでした: %s", snapshot)
    return client


def get_credential_status() -> Dict[str, bool]:
    """利用可能な認証情報の有無を返す。"""

    runtime_api_key = _normalize_token(os.getenv("JQUANTS_API_KEY") or JQUANTS_API_KEY)
    return {
        "JQUANTS_API_KEY": bool(runtime_api_key),
    }


def _token_preview(token: Optional[str]) -> str:
    """トークンの先頭・末尾のみを含むプレビュー文字列を生成する。"""

    cleaned = _normalize_token(token)
    if not cleaned:
        return "<missing>"
    if len(cleaned) <= 8:
        return f"<len={len(cleaned)}>"
    return f"<len={len(cleaned)} preview={cleaned[:4]}...{cleaned[-4:]}>"


def _credential_debug_info(client: JQuantsClient) -> Dict[str, str]:
    """現在使用している認証情報の概要を返す（機微情報はマスク）。"""

    runtime_api_key = _normalize_token(os.getenv("JQUANTS_API_KEY"))
    configured_api_key = _normalize_token(JQUANTS_API_KEY)
    source = "env" if runtime_api_key else "config" if configured_api_key else "missing"
    effective_api_key = runtime_api_key or configured_api_key or client.api_key

    return {
        "api_key_source": source,
        "api_key_preview": _token_preview(effective_api_key),
        "client_api_key_preview": _token_preview(client.api_key),
    }


def _get_api_key(client: JQuantsClient) -> str:
    try:
        return client.authenticate()
    except Exception as exc:  # pragma: no cover - thin wrapper
        raise JQuantsError("APIキーの取得に失敗しました。") from exc


def _extract_error_message(response: requests.Response) -> Optional[str]:
    """レスポンスボディからエラー文言を抽出する。"""

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
            for key in ("message", "detail", "error"):
                if key in payload and payload[key]:
                    return str(payload[key])
    except ValueError:
        pass

    if response.text:
        return response.text.strip()

    return None


def _describe_request(path: str, params: Optional[Dict[str, Any]]) -> str:
    if not params:
        return path

    safe_params = {k: v for k, v in params.items() if k.lower() not in {"authorization", "token"}}
    return f"{path} params={safe_params}"


def _request_with_token(client: JQuantsClient, path: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    api_key = _get_api_key(client)
    url = f"{client.base_url}{path}"
    headers = {"x-api-key": api_key}

    request_context = _describe_request(path, params)
    retry = 0
    auth_retry = 0
    wait_seconds = RATE_LIMIT_INITIAL_WAIT
    while True:
        try:
            response = requests.get(url, headers=headers, params=params, timeout=30)
        except Exception as exc:  # pragma: no cover - thin wrapper
            logger.exception("J-Quants API リクエスト送信に失敗しました: %s", request_context)
            raise JQuantsError(f"J-Quants API リクエスト送信に失敗しました: {path}") from exc

        if response.status_code in RATE_LIMIT_STATUS_CODES:
            retry += 1
            if retry > RATE_LIMIT_MAX_RETRIES:
                raise JQuantsRateLimitError("レートリミットを複数回超過したため中断しました。")

            retry_after_header = response.headers.get("Retry-After")
            try:
                retry_after = int(retry_after_header) if retry_after_header else None
            except ValueError:
                retry_after = None

            sleep_for = retry_after or wait_seconds
            # ジッターを加えて衝突を避ける
            sleep_for = min(sleep_for + random.randint(5, 30), RATE_LIMIT_MAX_WAIT)
            logger.warning(
                "レートリミットに達しました(status=%s)。%s 秒待機して再試行します (%s/%s) [%s]",
                response.status_code,
                sleep_for,
                retry,
                RATE_LIMIT_MAX_RETRIES,
                request_context,
            )
            time.sleep(sleep_for)
            wait_seconds = min(int(wait_seconds * RATE_LIMIT_BACKOFF), RATE_LIMIT_MAX_WAIT)
            continue

        if response.status_code in AUTH_ERROR_STATUS_CODES:
            error_message = _extract_error_message(response)
            if error_message and "api" in error_message.lower() and "key" in error_message.lower():
                credential_debug = _credential_debug_info(client)
                logger.error(
                    "提供されたAPIキーが無効です。JQUANTS_API_KEY を再発行してください"
                    " (status=%s, message=%s, credentials=%s, request=%s)",
                    response.status_code,
                    error_message,
                    credential_debug,
                    request_context,
                )
                raise JQuantsError(
                    "APIキーが無効です。JQUANTS_API_KEY を正しい値で再設定してください。"
                )

            auth_retry += 1
            if auth_retry > AUTH_ERROR_MAX_RETRIES:
                logger.error(
                    "認証エラーが繰り返されたためリトライを中断します (status=%s, retries=%s/%s, request=%s)",
                    response.status_code,
                    auth_retry,
                    AUTH_ERROR_MAX_RETRIES,
                    request_context,
                )
                raise JQuantsError(f"J-Quants API リクエストに失敗しました: {path}")

            logger.warning(
                "認証エラーを検知したため %s 秒待機して再試行します (status=%s, %s/%s) [%s]",
                AUTH_ERROR_WAIT,
                response.status_code,
                auth_retry,
                AUTH_ERROR_MAX_RETRIES,
                request_context,
            )
            time.sleep(AUTH_ERROR_WAIT)

            api_key = _get_api_key(client)
            headers = {"x-api-key": api_key}
            continue

        try:
            response.raise_for_status()
            payload = response.json()
            if isinstance(payload, dict) and payload.get("errors"):
                error_message = _extract_error_message(response)
                raise JQuantsError(
                    f"J-Quants API エラーが返されました: {error_message or 'unknown error'}"
                )
            return payload
        except Exception as exc:  # pragma: no cover - thin wrapper
            logger.exception("J-Quants API リクエストに失敗しました: %s", request_context)
            raise JQuantsError(f"J-Quants API リクエストに失敗しました: {path}") from exc


def _extract_pagination_key(payload: Dict[str, Any]) -> Optional[str]:
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


def _extract_daily_quotes_payload(payload: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
    for key in ("data", "dailyQuotes", "daily_quotes", "prices", "quotes"):
        if key in payload:
            return payload.get(key)
    return None


def _extract_listed_master_payload(payload: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
    for key in (
        "data",
        "info",
        "listedInfo",
        "listed_info",
        "listed",
        "listed_master",
    ):
        if key in payload:
            return payload.get(key)
    return None


def _find_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    return next((col for col in candidates if col in df.columns), None)


def _combine_columns(df: pd.DataFrame, candidates: List[str]) -> Optional[pd.Series]:
    series: Optional[pd.Series] = None
    for col in candidates:
        if col not in df.columns:
            continue
        if series is None:
            series = df[col]
        else:
            series = series.fillna(df[col])
    return series


def _normalize_market_value(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None

    text = str(value).strip()
    if not text:
        return None

    mapped = LISTED_MASTER_MARKET_CODE_MAP.get(text)
    if mapped:
        return mapped

    normalized = text.upper()
    if "PRIME" in normalized or "プライム" in text:
        return "PRIME"
    if "STANDARD" in normalized or "スタンダード" in text:
        return "STANDARD"
    if "GROWTH" in normalized or "グロース" in text or "MOTHERS" in normalized or "マザーズ" in text:
        return "GROWTH"

    return text


def _fetch_listed_master_paginated(
    client: JQuantsClient, params: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    all_rows: List[Dict[str, Any]] = []
    pagination_key: Optional[str] = None
    while True:
        request_params = dict(params or {})
        if pagination_key:
            request_params["pagination_key"] = pagination_key
        data = _request_with_token(client, LISTED_MASTER_ENDPOINT, params=request_params)
        rows = _extract_listed_master_payload(data)
        if rows is None:
            raise JQuantsError("listed master がレスポンスに存在しません。")
        all_rows.extend(rows)
        pagination_key = _extract_pagination_key(data)
        if not pagination_key:
            break
    return all_rows


def _normalize_listed_master(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()

    code_series = _combine_columns(
        df,
        [
            "code",
            "Code",
            "LocalCode",
            "Localcode",
            "SecurityCode",
            "Securitycode",
            "Symbol",
            "symbol",
        ],
    )
    if code_series is not None:
        df["code"] = code_series

    name_series = _combine_columns(
        df,
        [
            "name",
            "Name",
            "CompanyName",
            "CompanyNameJp",
            "CompanyNameJapanese",
            "CompanyNameEnglish",
            "companyName",
        ],
    )
    if name_series is not None:
        df["name"] = name_series

    if "Sector17Code" in df.columns and "sector17" not in df.columns:
        df["sector17"] = df["Sector17Code"]
    if "Sector33Code" in df.columns and "sector33" not in df.columns:
        df["sector33"] = df["Sector33Code"]

    market_code_series = _combine_columns(
        df,
        [
            "market_code",
            "MarketCode",
            "marketCode",
            "MarketSegmentCode",
            "marketSegmentCode",
        ],
    )
    market_name_series = _combine_columns(
        df,
        [
            "market_name",
            "MarketCodeName",
            "marketCodeName",
            "MarketName",
            "marketName",
            "Market",
            "market",
            "MarketSegment",
            "marketSegment",
        ],
    )

    if market_code_series is not None:
        df["market_code"] = market_code_series
    if market_name_series is not None:
        df["market_name"] = market_name_series

    market_series = _combine_columns(
        df,
        [
            "market",
            "market_name",
            "market_code",
            "Market",
            "MarketName",
            "MarketCodeName",
            "MarketCode",
        ],
    )
    if market_series is not None:
        df["market"] = market_series.apply(_normalize_market_value)

    required_cols = ["code", "name", "market"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise JQuantsError(f"銘柄マスタに必要なカラムが不足しています: {missing}")

    df["code"] = df["code"].astype(str).str.zfill(4)
    df = df[required_cols + [c for c in df.columns if c not in required_cols]].copy()
    df = df.sort_values("code").reset_index(drop=True)
    return df


def _fetch_daily_quotes_paginated(
    client: JQuantsClient, params: Dict[str, Any]
) -> List[Dict[str, Any]]:
    all_rows: List[Dict[str, Any]] = []
    pagination_key: Optional[str] = None
    while True:
        request_params = dict(params)
        if pagination_key:
            request_params["pagination_key"] = pagination_key
        data = _request_with_token(client, DAILY_QUOTES_ENDPOINT, params=request_params)
        rows = _extract_daily_quotes_payload(data)
        if rows is None:
            raise JQuantsError("dailyQuotes がレスポンスに存在しません。")
        all_rows.extend(rows)
        pagination_key = _extract_pagination_key(data)
        if not pagination_key:
            break
    return all_rows


def _normalize_daily_quotes(df_raw: pd.DataFrame, code: str) -> pd.DataFrame:
    df = df_raw.copy()

    date_col = _find_column(df, ["Date", "date", "quoteDate", "datetime", "DateTime"])
    if date_col:
        df["date"] = pd.to_datetime(df[date_col], errors="coerce")
    else:
        raise JQuantsError("Date 列が存在しません。")

    market_col = _find_column(
        df,
        [
            "Market",
            "market",
            "MarketCode",
            "marketCode",
            "MarketCodeName",
            "marketCodeName",
            "MarketName",
            "marketName",
        ],
    )
    market = df[market_col].iloc[0] if market_col and not df.empty else None

    adjusted_open = _find_column(
        df,
        ["AdjO", "AdjustmentOpen", "adjustmentOpen", "AdjustedOpen", "adjustedOpen"],
    )
    adjusted_high = _find_column(
        df,
        ["AdjH", "AdjustmentHigh", "adjustmentHigh", "AdjustedHigh", "adjustedHigh"],
    )
    adjusted_low = _find_column(
        df,
        ["AdjL", "AdjustmentLow", "adjustmentLow", "AdjustedLow", "adjustedLow"],
    )
    adjusted_close = _find_column(
        df,
        ["AdjC", "AdjustmentClose", "adjustmentClose", "AdjustedClose", "adjustedClose"],
    )
    adjusted_volume = _find_column(
        df,
        ["AdjVo", "AdjustmentVolume", "adjustmentVolume", "AdjustedVolume", "adjustedVolume"],
    )
    adjustment_factor_col = _find_column(
        df,
        ["AdjustmentFactor", "adjustmentFactor", "adjustment_factor"],
    )

    has_adjusted_cols = all([adjusted_open, adjusted_high, adjusted_low, adjusted_close])
    if has_adjusted_cols:
        open_col, high_col, low_col, close_col = (
            adjusted_open,
            adjusted_high,
            adjusted_low,
            adjusted_close,
        )
        vol_col = adjusted_volume or _find_column(df, ["AdjVo", "Volume", "volume", "Vo", "VO"])
    else:
        open_col = _find_column(df, ["O", "Open", "open"])
        high_col = _find_column(df, ["H", "High", "high"])
        low_col = _find_column(df, ["L", "Low", "low"])
        close_col = _find_column(df, ["C", "Close", "close"])
        vol_col = _find_column(df, ["Vo", "VO", "Volume", "volume"])

    if not open_col or not high_col or not low_col or not close_col:
        raise JQuantsError("OHLC 列が存在しません。")

    if not vol_col:
        raise JQuantsError("Volume 列が存在しません。")

    open_series = df[open_col]
    high_series = df[high_col]
    low_series = df[low_col]
    close_series = df[close_col]
    volume_series = df[vol_col]

    if not has_adjusted_cols and adjustment_factor_col:
        adjustment_factor = pd.to_numeric(
            df[adjustment_factor_col], errors="coerce"
        ).fillna(1.0)
        open_series = pd.to_numeric(open_series, errors="coerce") * adjustment_factor
        high_series = pd.to_numeric(high_series, errors="coerce") * adjustment_factor
        low_series = pd.to_numeric(low_series, errors="coerce") * adjustment_factor
        close_series = pd.to_numeric(close_series, errors="coerce") * adjustment_factor
        volume_series = pd.to_numeric(volume_series, errors="coerce") / adjustment_factor

    normalized = pd.DataFrame(
        {
            "date": df["date"],
            "datetime": pd.to_datetime(df["date"]),
            "code": str(code).zfill(4),
            "market": market,
            "open": open_series,
            "high": high_series,
            "low": low_series,
            "close": close_series,
            "volume": volume_series,
        }
    )

    normalized = normalized.dropna(subset=["date", "open", "high", "low", "close", "volume"])
    normalized["datetime"] = pd.to_datetime(normalized["datetime"]).dt.strftime("%Y-%m-%dT%H:%M:%S")
    normalized["date"] = normalized["date"].dt.date.astype(str)
    normalized = normalized.sort_values("date").reset_index(drop=True)
    return normalized


def _normalize_topix(df_raw: pd.DataFrame, code: str = TOPIX_CODE) -> pd.DataFrame:
    df = df_raw.copy()

    date_col = _find_column(df, ["Date", "date", "quoteDate", "datetime", "DateTime"])
    if date_col:
        df["date"] = pd.to_datetime(df[date_col], errors="coerce")
    else:
        raise JQuantsError("Date 列が存在しません。")

    open_col = _find_column(
        df,
        [
            "AdjO",
            "AdjustmentOpen",
            "adjustmentOpen",
            "AdjustedOpen",
            "adjustedOpen",
            "O",
            "Open",
            "open",
        ],
    )
    high_col = _find_column(
        df,
        [
            "AdjH",
            "AdjustmentHigh",
            "adjustmentHigh",
            "AdjustedHigh",
            "adjustedHigh",
            "H",
            "High",
            "high",
        ],
    )
    low_col = _find_column(
        df,
        [
            "AdjL",
            "AdjustmentLow",
            "adjustmentLow",
            "AdjustedLow",
            "adjustedLow",
            "L",
            "Low",
            "low",
        ],
    )
    close_col = _find_column(
        df,
        [
            "AdjC",
            "AdjustmentClose",
            "adjustmentClose",
            "AdjustedClose",
            "adjustedClose",
            "C",
            "Close",
            "close",
        ],
    )

    missing_cols = [name for name, col in {
        "open": open_col,
        "high": high_col,
        "low": low_col,
        "close": close_col,
    }.items() if col is None]
    if missing_cols:
        raise JQuantsError(f"TOPIX データに必要なカラムが不足しています: {missing_cols}")

    normalized = pd.DataFrame(
        {
            "date": df["date"],
            "datetime": pd.to_datetime(df["date"]),
            "code": code,
            "open": df[open_col],
            "high": df[high_col],
            "low": df[low_col],
            "close": df[close_col],
        }
    )

    normalized = normalized.dropna(subset=["date", "open", "high", "low", "close"])
    normalized["datetime"] = pd.to_datetime(normalized["datetime"]).dt.strftime("%Y-%m-%dT%H:%M:%S")
    normalized["date"] = normalized["date"].dt.date.astype(str)
    normalized = normalized.sort_values("date").reset_index(drop=True)
    return normalized


def _light_plan_window(client: Optional[JQuantsClient] = None) -> tuple[str, str]:
    _ = client
    today = date.today()
    from_date = today - timedelta(days=LIGHT_PLAN_WINDOW_DAYS)
    return from_date.isoformat(), today.isoformat()


def _extract_subscription_start_date(message: str) -> Optional[str]:
    """APIエラー文からサブスク開始日(YYYY-MM-DD)を抽出する。"""

    if not message:
        return None

    match = re.search(r"subscription covers the following dates:\s*([0-9]{4}-[0-9]{2}-[0-9]{2})", message)
    if not match:
        return None
    return match.group(1)


def _most_recent_weekday(target_weekday: int) -> date:
    """指定した曜日の直近日付 (同曜日を含む過去) を返す。"""

    today = date.today()
    delta = (today.weekday() - target_weekday) % 7
    return today - timedelta(days=delta)


def _previous_business_day(base_date: date) -> date:
    """前営業日 (祝日考慮なし) を返す。"""

    return pd.bdate_range(end=base_date - timedelta(days=1), periods=1)[0].date()


def _fetch_daily_quotes_raw(client: JQuantsClient, target_date: date) -> pd.DataFrame:
    params = {"date": target_date.isoformat()}
    raw_quotes = _fetch_daily_quotes_paginated(client, params)
    return pd.DataFrame(raw_quotes)


def _get_latest_trading_day(
    client: JQuantsClient,
    *,
    preferred_date: Optional[date] = None,
    now: Optional[datetime] = None,
    max_lookback_days: int = 10,
    return_raw: bool = False,
) -> tuple[date, Optional[pd.DataFrame]]:
    """取得可能な最新営業日を、APIレスポンスの空判定を用いて決定する。"""

    reference_dt = now or datetime.now()
    candidate_date = preferred_date or reference_dt.date()

    if candidate_date > reference_dt.date():
        candidate_date = reference_dt.date()

    if preferred_date is None:
        if reference_dt.weekday() >= 5 or reference_dt.time() < dt_time(hour=9):
            candidate_date = candidate_date - timedelta(days=1)

    candidate_date = pd.bdate_range(end=candidate_date, periods=1)[0].date()

    lookback = 0
    df_raw: Optional[pd.DataFrame] = None
    while lookback <= max_lookback_days:
        df_raw = _fetch_daily_quotes_raw(client, candidate_date)
        if not df_raw.empty:
            return candidate_date, df_raw if return_raw else None

        logger.info("%s の株価データが空のため前営業日にフォールバックします", candidate_date)
        candidate_date = _previous_business_day(candidate_date)
        lookback += 1

    raise JQuantsError("取得可能な最新営業日を特定できませんでした。")


def fetch_daily_quotes_for_date(target_date: date | str) -> pd.DataFrame:
    """指定日付の全銘柄株価を取得し、正規化した DataFrame を返す。"""

    client = _get_client()
    if isinstance(target_date, str):
        target_dt = pd.to_datetime(target_date).date()
    else:
        target_dt = target_date

    resolved_date, df_raw = _get_latest_trading_day(
        client, preferred_date=target_dt, return_raw=True
    )

    if resolved_date != target_dt:
        logger.info("%s の代わりに直近営業日 %s の全銘柄株価を取得します", target_dt, resolved_date)
    else:
        logger.info("%s の全銘柄株価を取得します", resolved_date)

    code_col = None
    for candidate in ("Code", "LocalCode", "code", "Localcode", "symbol", "Symbol"):
        if candidate in df_raw.columns:
            code_col = candidate
            break
    if code_col is None:
        raise JQuantsError("銘柄コード列を特定できませんでした。")

    normalized_frames: List[pd.DataFrame] = []
    for code_value, group in df_raw.groupby(code_col):
        normalized_frames.append(_normalize_daily_quotes(group, str(code_value).zfill(4)))

    normalized = pd.concat(normalized_frames, ignore_index=True)
    logger.info("%s の株価を %s 銘柄分取得しました", resolved_date, normalized["code"].nunique())
    return normalized


def fetch_listed_master() -> pd.DataFrame:
    """上場銘柄一覧を取得し、CSV として保存する。"""

    client = _get_client()
    listed_rows = _fetch_listed_master_paginated(client)
    df = pd.DataFrame(listed_rows)
    if df.empty:
        raise JQuantsError("銘柄マスタが空でした。")
    df = _normalize_listed_master(df)

    META_DIR.mkdir(parents=True, exist_ok=True)
    out_path = META_DIR / "listed_master.csv"
    df.to_csv(out_path, index=False)
    logger.info("listed_master.csv を保存しました: %s", out_path)
    return df


def load_listed_master() -> pd.DataFrame:
    path = META_DIR / "listed_master.csv"
    if not path.exists():
        return fetch_listed_master()
    return pd.read_csv(path, dtype={"code": str})


def get_all_listed_codes() -> List[str]:
    """listed_master.csv に記載の全銘柄コードを返す。"""

    df = load_listed_master()
    return sorted(df["code"].astype(str).str.zfill(4).tolist())


def _get_market_universe(markets: List[str]) -> List[str]:
    df = load_listed_master()
    market_source_col = _find_column(
        df,
        [
            "market",
            "market_name",
            "market_code",
            "Market",
            "MarketName",
            "MarketCodeName",
            "MarketCode",
        ],
    )
    if market_source_col is None:
        logger.warning("市場区分カラムが見つからないため全銘柄を対象にします。")
        return sorted(df["code"].astype(str).str.zfill(4).tolist())

    market_normalized = df[market_source_col].apply(_normalize_market_value)
    if market_normalized.isna().all():
        logger.warning("市場区分の正規化に失敗したため全銘柄を対象にします。")
        return sorted(df["code"].astype(str).str.zfill(4).tolist())

    universe = df[market_normalized.isin(markets)]
    return sorted(universe["code"].astype(str).str.zfill(4).tolist())


def get_default_universe() -> List[str]:
    return _get_market_universe(["PRIME", "STANDARD"])


def get_growth_universe() -> List[str]:
    return _get_market_universe(["GROWTH"])


def build_universe(
    include_custom: bool = False,
    custom_path: Optional[Path] = None,
    use_listed_master: bool = False,
    market_filter: str = "prime_standard",
) -> List[str]:
    """ユニバースを組み立てる。

    Args:
        include_custom: ``True`` の場合は ``custom_symbols.txt`` で定義した
            追加銘柄も含める。
        custom_path: カスタム銘柄リストのパス。未指定時は ``data/meta``
            配下の ``custom_symbols.txt`` を参照する。
        use_listed_master: True の場合は listed_master.csv に記載の全銘柄
            を対象にする（市場区分での絞り込みなし）。
        market_filter: use_listed_master が False の場合に適用する市場区分。
            ``prime_standard`` または ``growth`` を指定する。
    """

    if use_listed_master:
        codes: List[str] = get_all_listed_codes()
    else:
        if market_filter == "growth":
            codes = get_growth_universe()
        elif market_filter == "prime_standard":
            codes = get_default_universe()
        else:
            raise ValueError(f"不明な market_filter です: {market_filter}")
    if include_custom:
        codes += load_custom_symbols(path=custom_path)
    # zfill(4) 済みのため重複のみ除去
    return sorted(dict.fromkeys(codes).keys())


def _load_meta(code: str) -> Dict[str, Any]:
    meta_path = META_DIR / f"{code}.json"
    if meta_path.exists():
        return json.loads(meta_path.read_text())
    return {}


def _save_meta(code: str, meta: Dict[str, Any]) -> Path:
    meta_path = META_DIR / f"{code}.json"
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    return meta_path


def _load_topix_meta() -> Dict[str, Any]:
    if TOPIX_META_PATH.exists():
        return json.loads(TOPIX_META_PATH.read_text())
    return {}


def _save_topix_meta(meta: Dict[str, Any]) -> Path:
    TOPIX_META_PATH.write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    return TOPIX_META_PATH


def _load_existing_csv(code: str) -> Optional[pd.DataFrame]:
    csv_path = PRICE_CSV_DIR / f"{code}.csv"
    if not csv_path.exists():
        return None
    df = pd.read_csv(csv_path)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"]).dt.date.astype(str)
    if "datetime" not in df.columns and "date" in df.columns:
        df["datetime"] = pd.to_datetime(df["date"]).astype(str)
    return df


def _load_existing_topix() -> Optional[pd.DataFrame]:
    if not TOPIX_CSV_PATH.exists():
        return None
    df = pd.read_csv(TOPIX_CSV_PATH)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"]).dt.date.astype(str)
    if "datetime" not in df.columns and "date" in df.columns:
        df["datetime"] = pd.to_datetime(df["date"]).astype(str)
    return df


def _merge_and_save(
    code: str,
    normalized: pd.DataFrame,
    fetch_to: str,
    *,
    existing_df: Optional[pd.DataFrame] = None,
    meta: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    existing_df = existing_df if existing_df is not None else _load_existing_csv(code)

    if existing_df is not None and not existing_df.empty:
        merged = pd.concat([existing_df, normalized], ignore_index=True)
        merged = merged.drop_duplicates(subset=["date"], keep="last")
        merged = merged.sort_values("date").reset_index(drop=True)
    else:
        merged = normalized.sort_values("date").reset_index(drop=True)

    if "datetime" not in merged.columns:
        merged["datetime"] = pd.to_datetime(merged["date"], format="ISO8601").dt.strftime(
            "%Y-%m-%dT%H:%M:%S"
        )
    else:
        merged["datetime"] = pd.to_datetime(merged["datetime"], format="ISO8601").dt.strftime(
            "%Y-%m-%dT%H:%M:%S"
        )

    PRICE_CSV_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = PRICE_CSV_DIR / f"{code}.csv"
    merged.to_csv(csv_path, index=False)

    meta = meta or {}
    history_start = merged["date"].min()
    history_end = merged["date"].max()
    market_series = normalized["market"] if "market" in normalized.columns else None
    market = meta.get("market") or (market_series.iloc[0] if market_series is not None else None)
    meta_out = {
        "code": code,
        "market": market,
        "history_start": history_start,
        "history_end": history_end,
        "last_fetch_to": fetch_to,
        "last_fetch_at": datetime.now().astimezone().isoformat(),
        "plan": "LIGHT",
    }
    _save_meta(code, meta_out)
    return merged


def _merge_and_save_daily_snapshot(code: str, snapshot_df: pd.DataFrame, snapshot_date: str) -> pd.DataFrame:
    """指定日スナップショットを既存CSVへマージして保存する。"""

    if snapshot_df.empty:
        raise JQuantsError("指定日のスナップショットが空でした。")

    normalized = snapshot_df.copy()
    normalized["code"] = str(code).zfill(4)

    if "date" in normalized.columns:
        normalized["date"] = pd.to_datetime(normalized["date"]).dt.date.astype(str)
    else:
        raise JQuantsError("スナップショットに date 列が存在しません。")

    if "datetime" in normalized.columns:
        normalized["datetime"] = pd.to_datetime(normalized["datetime"]).dt.strftime("%Y-%m-%dT%H:%M:%S")
    else:
        normalized["datetime"] = pd.to_datetime(normalized["date"]).dt.strftime("%Y-%m-%dT%H:%M:%S")

    meta = _load_meta(code)
    return _merge_and_save(code, normalized, snapshot_date, meta=meta)


def _merge_topix_and_save(
    normalized: pd.DataFrame,
    fetch_to: str,
    *,
    existing_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    existing_df = existing_df if existing_df is not None else _load_existing_topix()

    if existing_df is not None and not existing_df.empty:
        merged = pd.concat([existing_df, normalized], ignore_index=True)
        merged = merged.drop_duplicates(subset=["date"], keep="last")
        merged = merged.sort_values("date").reset_index(drop=True)
    else:
        merged = normalized.sort_values("date").reset_index(drop=True)

    if "datetime" not in merged.columns:
        merged["datetime"] = pd.to_datetime(merged["date"], format="ISO8601").dt.strftime(
            "%Y-%m-%dT%H:%M:%S"
        )
    else:
        merged["datetime"] = pd.to_datetime(merged["datetime"], format="ISO8601").dt.strftime(
            "%Y-%m-%dT%H:%M:%S"
        )

    PRICE_CSV_DIR.mkdir(parents=True, exist_ok=True)
    merged.to_csv(TOPIX_CSV_PATH, index=False)

    meta_out = {
        "code": TOPIX_CODE,
        "history_start": merged["date"].min(),
        "history_end": merged["date"].max(),
        "last_fetch_to": fetch_to,
        "last_fetch_at": datetime.now().astimezone().isoformat(),
        "plan": "LIGHT",
    }
    _save_topix_meta(meta_out)
    return merged


def update_symbol(code: str, full_refresh: bool = False) -> pd.DataFrame:
    code = str(code).zfill(4)
    client = _get_client()

    meta = _load_meta(code)
    existing_df = None if full_refresh else _load_existing_csv(code)

    from_light, to_light = _light_plan_window(client)

    if not full_refresh and meta.get("history_end") and existing_df is not None:
        from_date = (pd.to_datetime(meta["history_end"]) + pd.Timedelta(days=1)).date()
        to_date = datetime.fromisoformat(to_light).date()
        if from_date > to_date:
            logger.info("%s は取得可能期間外のためスキップします", code)
            return existing_df
        fetch_from, fetch_to = from_date.isoformat(), to_light
    else:
        fetch_from, fetch_to = from_light, to_light

    resolved_to, _ = _get_latest_trading_day(
        client, preferred_date=datetime.fromisoformat(fetch_to).date()
    )
    if resolved_to.isoformat() != fetch_to:
        logger.info("最新営業日を %s から %s に補正しました", fetch_to, resolved_to)
    fetch_to = resolved_to.isoformat()

    fetch_from_dt = datetime.fromisoformat(fetch_from).date()
    fetch_to_dt = datetime.fromisoformat(fetch_to).date()
    if fetch_from_dt > fetch_to_dt:
        logger.info(
            "%s の取得対象期間が逆転しているためスキップします (from=%s, to=%s)",
            code,
            fetch_from,
            fetch_to,
        )
        return existing_df if existing_df is not None else pd.DataFrame()

    params = {"code": code, "from": fetch_from, "to": fetch_to}
    logger.info("%s の株価を取得します (from=%s, to=%s)", code, fetch_from, fetch_to)
    raw_quotes = _fetch_daily_quotes_paginated(client, params)
    df_raw = pd.DataFrame(raw_quotes)
    if df_raw.empty:
        logger.info("%s の価格データが空のため前営業日にフォールバックします", code)
        resolved_to, df_raw = _get_latest_trading_day(
            client, preferred_date=datetime.fromisoformat(fetch_to).date(), return_raw=True
        )
        fetch_to = resolved_to.isoformat()
        logger.info("%s の株価を取得し直します (from=%s, to=%s)", code, fetch_from, fetch_to)
        if df_raw.empty:
            raise JQuantsError(f"{code} の価格データが空でした。")

    normalized = _normalize_daily_quotes(df_raw, code)
    if normalized.empty:
        raise JQuantsError(f"{code} の正規化後データが空でした。")

    actual_fetch_to = str(normalized["date"].max())
    merged = _merge_and_save(
        code, normalized, actual_fetch_to, existing_df=existing_df, meta=meta
    )

    if "datetime" not in merged.columns:
        merged["datetime"] = pd.to_datetime(merged["date"], format="ISO8601").dt.strftime(
            "%Y-%m-%dT%H:%M:%S"
        )
    else:
        merged["datetime"] = pd.to_datetime(merged["datetime"], format="ISO8601").dt.strftime(
            "%Y-%m-%dT%H:%M:%S"
        )

    return merged


def update_topix(full_refresh: bool = False) -> pd.DataFrame:
    """TOPIX 指数をライトプランの取得可能期間で保存する。"""

    client = _get_client()

    meta = _load_topix_meta()
    existing_df = None if full_refresh else _load_existing_topix()

    from_light, to_light = _light_plan_window(client)

    if not full_refresh and meta.get("history_end") and existing_df is not None:
        from_date = (pd.to_datetime(meta["history_end"]) + pd.Timedelta(days=1)).date()
        to_date = datetime.fromisoformat(to_light).date()
        if from_date > to_date:
            logger.info("TOPIX は取得可能期間外のためスキップします")
            return existing_df
        fetch_from, fetch_to = from_date.isoformat(), to_light
    else:
        fetch_from, fetch_to = from_light, to_light

    resolved_to, _ = _get_latest_trading_day(
        client, preferred_date=datetime.fromisoformat(fetch_to).date()
    )
    if resolved_to.isoformat() != fetch_to:
        logger.info("TOPIX の最新営業日を %s から %s に補正しました", fetch_to, resolved_to)
    fetch_to = resolved_to.isoformat()

    fetch_from_dt = datetime.fromisoformat(fetch_from).date()
    fetch_to_dt = datetime.fromisoformat(fetch_to).date()
    if fetch_from_dt > fetch_to_dt:
        logger.info("TOPIX の取得対象期間が逆転しているためスキップします")
        return existing_df if existing_df is not None else pd.DataFrame()

    logger.info("TOPIX を取得します (from=%s, to=%s)", fetch_from, fetch_to)
    try:
        df_raw = client.fetch_topix(fetch_from, fetch_to)
    except ValueError as exc:
        subscription_start = _extract_subscription_start_date(str(exc))
        if subscription_start and subscription_start > fetch_from:
            logger.info(
                "TOPIX の取得開始日をサブスク範囲に合わせて補正します (from=%s -> %s)",
                fetch_from,
                subscription_start,
            )
            df_raw = client.fetch_topix(subscription_start, fetch_to)
        else:
            raise
    if df_raw.empty:
        raise JQuantsError("TOPIX の価格データが空でした。")

    normalized = _normalize_topix(df_raw)
    if normalized.empty:
        raise JQuantsError("TOPIX の正規化後データが空でした。")

    actual_fetch_to = str(normalized["date"].max())
    merged = _merge_topix_and_save(
        normalized, actual_fetch_to, existing_df=existing_df
    )
    return merged


def append_quotes_for_date(target_date: str, codes: Optional[List[str]] = None) -> None:
    try:
        datetime.fromisoformat(target_date)
    except ValueError as exc:
        raise JQuantsError("日付は YYYY-MM-DD 形式で指定してください。") from exc

    resolved, _, _ = _append_daily_quotes_for_date(
        pd.to_datetime(target_date).date(),
        target_codes=set(codes) if codes else None,
        allow_fallback=False,
    )
    logger.info("%s の株価データ追加処理が完了しました", resolved)


def _append_daily_quotes_for_date(
    target_date: date,
    *,
    target_codes: Optional[Set[str]] = None,
    allow_fallback: bool = False,
) -> tuple[str, Set[str], Set[str]]:
    client = _get_client()
    if allow_fallback:
        resolved_date, df_raw = _get_latest_trading_day(
            client, preferred_date=target_date, return_raw=True
        )
    else:
        resolved_date = target_date
        df_raw = _fetch_daily_quotes_raw(client, target_date)

    if df_raw is None or df_raw.empty:
        logger.info("%s の株価データが空でした。", resolved_date)
        return resolved_date.isoformat(), set(), set()

    code_col = next(
        (col for col in ("code", "LocalCode", "Code", "symbol", "Symbol") if col in df_raw.columns),
        None,
    )
    if code_col is None:
        raise JQuantsError("銘柄コードのカラムが見つかりませんでした。")

    df_raw["code_norm"] = df_raw[code_col].astype(str).str.zfill(4)
    target_set = {str(code).zfill(4) for code in target_codes} if target_codes else None

    grouped = df_raw.groupby("code_norm")
    available_codes = set(grouped.groups)
    updated_codes: Set[str] = set()
    for code in sorted(available_codes):
        if target_set is not None and code not in target_set:
            continue

        filtered = grouped.get_group(code)
        normalized = _normalize_daily_quotes(filtered, code)
        if normalized.empty:
            continue

        meta = _load_meta(code)
        _merge_and_save(code, normalized, resolved_date.isoformat(), meta=meta)
        updated_codes.add(code)
        logger.info("%s の株価を %s に追記しました", resolved_date, code)

    if target_set is not None:
        missing = sorted(target_set - available_codes)
        if missing:
            logger.info("%s にデータが見つからなかった銘柄: %s", resolved_date, ", ".join(missing))

    return resolved_date.isoformat(), updated_codes, available_codes


def update_universe_with_anchor_day(
    codes: Optional[List[str]] = None,
    anchor_date: Optional[str] = None,
    anchor_weekday: int = 1,
    include_custom: bool = False,
    custom_path: Optional[Path] = None,
    use_listed_master: bool = False,
    market_filter: str = "prime_standard",
    min_rows_refresh: Optional[int] = None,
) -> None:
    """指定曜日のスナップショットを反映したうえで日次データを最新化する。"""

    target_codes = codes or build_universe(
        include_custom=include_custom,
        custom_path=custom_path,
        use_listed_master=use_listed_master,
        market_filter=market_filter,
    )
    target_codes = [str(code).zfill(4) for code in target_codes]

    anchor_dt = (
        pd.to_datetime(anchor_date).date()
        if anchor_date is not None
        else _most_recent_weekday(anchor_weekday)
    )
    anchor_str = anchor_dt.isoformat()
    try:
        anchor_quotes = fetch_daily_quotes_for_date(anchor_dt)
        grouped_anchor = {code: df for code, df in anchor_quotes.groupby("code")}
    except Exception:
        logger.exception("指定日 (%s) の株価取得に失敗しました。", anchor_str)
        grouped_anchor = {}

    for code in target_codes:
        anchor_df = grouped_anchor.get(code)
        if anchor_df is not None:
            try:
                _merge_and_save_daily_snapshot(code, anchor_df, anchor_str)
            except Exception:
                logger.exception("%s の指定日データ保存に失敗しました", code)
        else:
            logger.warning("指定日の株価が見つかりませんでした: %s (%s)", code, anchor_str)

        try:
            force_refresh = False
            if min_rows_refresh:
                existing_df = _load_existing_csv(code)
                if existing_df is not None and len(existing_df) < min_rows_refresh:
                    logger.info(
                        "%s の行数が少ないため再取得します (rows=%s, threshold=%s)",
                        code,
                        len(existing_df),
                        min_rows_refresh,
                    )
                    force_refresh = True
            update_symbol(code, full_refresh=force_refresh)
        except Exception:
            logger.exception("%s の日次更新に失敗しました", code)
        time.sleep(0.3)


def update_universe(
    codes: Optional[List[str]] = None,
    full_refresh: bool = False,
    use_listed_master: bool = False,
    append_date: Optional[str] = None,
    market_filter: str = "prime_standard",
    min_rows_refresh: Optional[int] = None,
) -> None:
    target_codes = codes or build_universe(
        use_listed_master=use_listed_master,
        market_filter=market_filter,
    )
    target_codes = [str(code).zfill(4) for code in target_codes]

    bulk_updated: Set[str] = set()
    latest_snapshot_date: Optional[str] = None
    previous_meta_end: Dict[str, Optional[str]] = {}
    if not full_refresh and append_date is None:
        previous_meta_end = {code: _load_meta(code).get("history_end") for code in target_codes}
        try:
            latest_snapshot_date, bulk_updated, _ = _append_daily_quotes_for_date(
                date.today(), target_codes=set(target_codes), allow_fallback=True
            )
        except Exception:
            logger.exception("最新営業日の一括取得に失敗しました")
            latest_snapshot_date = None

    latest_snapshot_dt = (
        datetime.fromisoformat(latest_snapshot_date).date() if latest_snapshot_date else None
    )
    latest_prev_bday = _previous_business_day(latest_snapshot_dt) if latest_snapshot_dt else None

    for code in target_codes:
        needs_symbol_update = full_refresh or append_date is not None or latest_snapshot_dt is None
        force_refresh = False
        if min_rows_refresh and not full_refresh:
            existing_df = _load_existing_csv(code)
            if existing_df is not None and len(existing_df) < min_rows_refresh:
                logger.info(
                    "%s の行数が少ないため再取得します (rows=%s, threshold=%s)",
                    code,
                    len(existing_df),
                    min_rows_refresh,
                )
                needs_symbol_update = True
                force_refresh = True
        if not needs_symbol_update:
            prev_end_str = previous_meta_end.get(code)
            prev_end_dt = pd.to_datetime(prev_end_str).date() if prev_end_str else None
            if prev_end_dt is None:
                needs_symbol_update = True
            elif latest_prev_bday and prev_end_dt < latest_prev_bday:
                needs_symbol_update = True
            elif code not in bulk_updated:
                needs_symbol_update = True

        if not needs_symbol_update:
            continue

        rate_limit_retry = 0
        while True:
            try:
                update_symbol(code, full_refresh=full_refresh or force_refresh)
                break
            except JQuantsRateLimitError as exc:  # pragma: no cover - 継続実行
                rate_limit_retry += 1
                if rate_limit_retry > RATE_LIMIT_MAX_RETRIES_PER_SYMBOL:
                    logger.exception(
                        "%s の更新中にレートリミットを繰り返し超過したためスキップします",
                        code,
                    )
                    break

                wait_for = RATE_LIMIT_MAX_WAIT
                logger.warning(
                    "レートリミットを検知したため %s 秒待機して再試行します (%s/%s)",
                    wait_for,
                    rate_limit_retry,
                    RATE_LIMIT_MAX_RETRIES_PER_SYMBOL,
                )
                time.sleep(wait_for)
                continue
            except Exception as exc:  # pragma: no cover - 継続実行
                logger.exception("%s の更新中にエラーが発生しました", code)
                break
        time.sleep(0.3)

    if append_date:
        append_quotes_for_date(append_date)


def load_custom_symbols(path: Optional[Path] = None) -> List[str]:
    path = path or META_DIR / "custom_symbols.txt"
    if not path.exists():
        return []
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="J-Quants 株価の一括更新")
    parser.add_argument(
        "--codes",
        nargs="*",
        help="更新対象の銘柄コード（未指定時はプライム+スタンダード）",
    )
    parser.add_argument(
        "--full-refresh",
        action="store_true",
        help="Free プランで取得可能な期間を全件取り直して上書き",
    )
    parser.add_argument(
        "--include-custom",
        action="store_true",
        help="custom_symbols.txt に記載した銘柄も対象に含める",
    )
    parser.add_argument(
        "--custom-path",
        type=Path,
        help="カスタム銘柄リストのパス（デフォルト: data/meta/custom_symbols.txt）",
    )
    parser.add_argument(
        "--use-listed-master",
        action="store_true",
        help="listed_master.csv に記載の全銘柄を一括更新する",
    )
    parser.add_argument(
        "--market",
        choices=["prime_standard", "growth"],
        default="prime_standard",
        help="更新対象の市場区分 (prime_standard/growth)",
    )
    parser.add_argument(
        "--append-date",
        help="全銘柄の当日株価を追記する日付 (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--include-topix",
        action="store_true",
        help="銘柄の更新に加えて TOPIX 指数も保存する",
    )
    parser.add_argument(
        "--min-rows-refresh",
        type=int,
        help="既存データが指定行数より少ない銘柄は再取得する",
    )

    args = parser.parse_args()

    if args.codes:
        codes = [str(c).zfill(4) for c in args.codes]
    else:
        codes = build_universe(
            include_custom=args.include_custom,
            custom_path=args.custom_path,
            use_listed_master=args.use_listed_master,
            market_filter=args.market,
        )

    update_universe(
        codes=codes,
        full_refresh=args.full_refresh,
        use_listed_master=args.use_listed_master,
        append_date=args.append_date,
        market_filter=args.market,
        min_rows_refresh=args.min_rows_refresh,
    )

    if args.include_topix:
        try:
            update_topix(full_refresh=args.full_refresh)
        except Exception:
            logger.exception("TOPIX の更新に失敗しました")
