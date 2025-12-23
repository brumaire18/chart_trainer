"""J-Quants から株価・銘柄マスタを取得するユーティリティ。

ライトプランの「直近約5年分取得可能」という制約を考慮し、
日足 CSV とメタ情報を整形・保存するための関数群を提供する。
"""

from __future__ import annotations

import json
import logging
import os
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
    JQUANTS_MAILADDRESS,
    JQUANTS_PASSWORD,
    JQUANTS_REFRESH_TOKEN,
    META_DIR,
    PRICE_CSV_DIR,
)
from .jquants_client import JQuantsClient, _normalize_token

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


@dataclass
class FetchResult:
    code: str
    market: Optional[str]
    df: pd.DataFrame
    meta_path: Path
    csv_path: Path


def _get_client() -> JQuantsClient:
    # 可能であれば実行時の環境変数を優先的に利用する
    runtime_refresh_raw = os.getenv("JQUANTS_REFRESH_TOKEN")
    runtime_refresh = _normalize_token(runtime_refresh_raw)
    configured_refresh = _normalize_token(JQUANTS_REFRESH_TOKEN)
    runtime_base_url = os.getenv("JQUANTS_BASE_URL")
    runtime_mail = (os.getenv("MAILADDRESS") or JQUANTS_MAILADDRESS or "").strip() or None
    runtime_password = (os.getenv("PASSWORD") or JQUANTS_PASSWORD or "").strip() or None

    if runtime_refresh_raw and runtime_refresh_raw != runtime_refresh:
        logger.info(
            "JQUANTS_REFRESH_TOKEN の前後にある空白を除去しました (len=%s -> len=%s)",
            len(runtime_refresh_raw),
            len(runtime_refresh or ""),
        )

    snapshot = {
        "MAILADDRESS": bool(runtime_mail),
        "PASSWORD": bool(runtime_password),
        "JQUANTS_REFRESH_TOKEN": bool(runtime_refresh or configured_refresh),
    }

    client = JQuantsClient(
        base_url=runtime_base_url or JQUANTS_BASE_URL,
        refresh_token=runtime_refresh or configured_refresh,
        mailaddress=runtime_mail,
        password=runtime_password,
    )

    if not client.refresh_token and not client.mailaddress:
        logger.error("必要な認証情報が見つかりませんでした: %s", snapshot)
    return client


def get_credential_status() -> Dict[str, bool]:
    """利用可能な認証情報の有無を返す。"""

    runtime_refresh = _normalize_token(os.getenv("JQUANTS_REFRESH_TOKEN") or JQUANTS_REFRESH_TOKEN)
    runtime_mail = (os.getenv("MAILADDRESS") or JQUANTS_MAILADDRESS or "").strip()
    runtime_password = (os.getenv("PASSWORD") or JQUANTS_PASSWORD or "").strip()
    return {
        "MAILADDRESS": bool(runtime_mail),
        "PASSWORD": bool(runtime_password),
        "JQUANTS_REFRESH_TOKEN": bool(runtime_refresh),
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

    runtime_refresh = _normalize_token(os.getenv("JQUANTS_REFRESH_TOKEN"))
    configured_refresh = _normalize_token(JQUANTS_REFRESH_TOKEN)
    source = "env" if runtime_refresh else "config" if configured_refresh else "missing"
    effective_refresh = runtime_refresh or configured_refresh or client.refresh_token

    return {
        "refresh_token_source": source,
        "refresh_token_preview": _token_preview(effective_refresh),
        "client_refresh_token_preview": _token_preview(client.refresh_token),
        "MAILADDRESS": "set" if client.mailaddress else "missing",
        "PASSWORD": "set" if client.password else "missing",
    }


def _get_id_token(client: JQuantsClient) -> str:
    try:
        return client.authenticate()
    except Exception as exc:  # pragma: no cover - thin wrapper
        raise JQuantsError("idToken の取得に失敗しました。") from exc


def _extract_error_message(response: requests.Response) -> Optional[str]:
    """レスポンスボディからエラー文言を抽出する。"""

    try:
        payload = response.json()
        if isinstance(payload, dict):
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
    token = _get_id_token(client)
    url = f"{client.base_url}{path}"
    headers = {"Authorization": f"Bearer {token}"}

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
            if error_message and "token" in error_message.lower() and "invalid" in error_message.lower():
                credential_debug = _credential_debug_info(client)
                logger.error(
                    "提供されたトークンが無効または期限切れです。JQUANTS_REFRESH_TOKEN を再取得してください"
                    " (status=%s, message=%s, credentials=%s, request=%s)",
                    response.status_code,
                    error_message,
                    credential_debug,
                    request_context,
                )
                raise JQuantsError(
                    "認証トークンが無効または期限切れです。JQUANTS_REFRESH_TOKEN を正しい値で再設定してください。"
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

            token = _get_id_token(client)
            headers = {"Authorization": f"Bearer {token}"}
            continue

        try:
            response.raise_for_status()
            return response.json()
        except Exception as exc:  # pragma: no cover - thin wrapper
            logger.exception("J-Quants API リクエストに失敗しました: %s", request_context)
            raise JQuantsError(f"J-Quants API リクエストに失敗しました: {path}") from exc


def _normalize_daily_quotes(df_raw: pd.DataFrame, code: str) -> pd.DataFrame:
    df = df_raw.copy()

    if "Date" in df.columns:
        df["date"] = pd.to_datetime(df["Date"], errors="coerce")
    elif "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    else:
        raise JQuantsError("Date 列が存在しません。")

    market_col = None
    for candidate in ("Market", "market", "MarketCode"):
        if candidate in df.columns:
            market_col = candidate
            break
    market = df[market_col].iloc[0] if market_col and not df.empty else None

    if all(
        col in df.columns
        for col in ["AdjustmentOpen", "AdjustmentHigh", "AdjustmentLow", "AdjustmentClose"]
    ):
        open_col, high_col, low_col, close_col = (
            "AdjustmentOpen",
            "AdjustmentHigh",
            "AdjustmentLow",
            "AdjustmentClose",
        )
        vol_col = "AdjustmentVolume" if "AdjustmentVolume" in df.columns else "Volume"
    else:
        open_col, high_col, low_col, close_col = ("Open", "High", "Low", "Close")
        vol_col = "Volume"

    if vol_col not in df.columns:
        raise JQuantsError("Volume 列が存在しません。")

    normalized = pd.DataFrame(
        {
            "date": df["date"],
            "datetime": pd.to_datetime(df["date"]),
            "code": str(code).zfill(4),
            "market": market,
            "open": df[open_col],
            "high": df[high_col],
            "low": df[low_col],
            "close": df[close_col],
            "volume": df[vol_col],
        }
    )

    normalized = normalized.dropna(subset=["date", "open", "high", "low", "close", "volume"])
    normalized["datetime"] = pd.to_datetime(normalized["datetime"]).dt.strftime("%Y-%m-%dT%H:%M:%S")
    normalized["date"] = normalized["date"].dt.date.astype(str)
    normalized = normalized.sort_values("date").reset_index(drop=True)
    return normalized


def _normalize_topix(df_raw: pd.DataFrame, code: str = TOPIX_CODE) -> pd.DataFrame:
    df = df_raw.copy()

    if "Date" in df.columns:
        df["date"] = pd.to_datetime(df["Date"], errors="coerce")
    elif "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    else:
        raise JQuantsError("Date 列が存在しません。")

    open_col = next((col for col in ("AdjustmentOpen", "Open", "open") if col in df.columns), None)
    high_col = next((col for col in ("AdjustmentHigh", "High", "high") if col in df.columns), None)
    low_col = next((col for col in ("AdjustmentLow", "Low", "low") if col in df.columns), None)
    close_col = next((col for col in ("AdjustmentClose", "Close", "close") if col in df.columns), None)

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
    client = client or _get_client()
    latest_trading_day, _ = _get_latest_trading_day(client)
    from_date = latest_trading_day - timedelta(days=LIGHT_PLAN_WINDOW_DAYS)
    return from_date.isoformat(), latest_trading_day.isoformat()


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
    data = _request_with_token(client, "/v1/prices/daily_quotes", params=params)
    raw_quotes = data.get("daily_quotes") or data.get("dailyQuotes")
    if raw_quotes is None:
        raise JQuantsError("daily_quotes がレスポンスに存在しません。")

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
    for candidate in ("Code", "LocalCode", "code", "Localcode"):
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
    data = _request_with_token(client, "/v1/listed/info")
    listed = data.get("info") or data.get("listed") or data.get("listedInfo")
    if listed is None:
        raise JQuantsError("listed info がレスポンスに存在しません。")

    df = pd.DataFrame(listed)
    if df.empty:
        raise JQuantsError("銘柄マスタが空でした。")

    col_map = {
        "Code": "code",
        "LocalCode": "code",
        "Name": "name",
        "Market": "market",
        "Sector17Code": "sector17",
        "Sector33Code": "sector33",
    }
    for src, dest in col_map.items():
        if src in df.columns:
            df[dest] = df[src]

    # 別名・大小文字の揺れに対応
    name_candidates = [
        "name",
        "Name",
        "CompanyName",
        "CompanyNameJp",
        "CompanyNameJapanese",
    ]
    market_candidates = [
        "market",
        "Market",
        "MarketCodeName",
        "MarketName",
        "MarketCode",
    ]

    if "name" not in df.columns:
        for cand in name_candidates:
            if cand in df.columns:
                df["name"] = df[cand]
                break

    if "market" not in df.columns:
        for cand in market_candidates:
            if cand in df.columns:
                df["market"] = df[cand]
                break

    required_cols = ["code", "name", "market"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise JQuantsError(f"銘柄マスタに必要なカラムが不足しています: {missing}")

    df["code"] = df["code"].astype(str).str.zfill(4)
    df = df[required_cols + [c for c in df.columns if c not in required_cols]].copy()
    df = df.sort_values("code").reset_index(drop=True)

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


def get_default_universe() -> List[str]:
    df = load_listed_master()
    universe = df[df["market"].isin(["PRIME", "STANDARD"])]
    return sorted(universe["code"].astype(str).str.zfill(4).tolist())


def build_universe(
    include_custom: bool = False,
    custom_path: Optional[Path] = None,
    use_listed_master: bool = False,
) -> List[str]:
    """ユニバースを組み立てる。

    Args:
        include_custom: ``True`` の場合は ``custom_symbols.txt`` で定義した
            追加銘柄も含める。
        custom_path: カスタム銘柄リストのパス。未指定時は ``data/meta``
            配下の ``custom_symbols.txt`` を参照する。
        use_listed_master: True の場合は listed_master.csv に記載の全銘柄
            を対象にする（市場区分での絞り込みなし）。
    """

    if use_listed_master:
        codes: List[str] = get_all_listed_codes()
    else:
        codes = get_default_universe()
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

    params = {"code": code, "from": fetch_from, "to": fetch_to}
    logger.info("%s の株価を取得します (from=%s, to=%s)", code, fetch_from, fetch_to)
    data = _request_with_token(client, "/v1/prices/daily_quotes", params=params)
    raw_quotes = data.get("daily_quotes") or data.get("dailyQuotes")
    if raw_quotes is None:
        raise JQuantsError("daily_quotes がレスポンスに存在しません。")

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

    logger.info("TOPIX を取得します (from=%s, to=%s)", fetch_from, fetch_to)
    df_raw = client.fetch_topix(fetch_from, fetch_to)
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

    code_col = next((col for col in ("code", "LocalCode", "Code") if col in df_raw.columns), None)
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
) -> None:
    """指定曜日のスナップショットを反映したうえで日次データを最新化する。"""

    target_codes = codes or build_universe(
        include_custom=include_custom,
        custom_path=custom_path,
        use_listed_master=use_listed_master,
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
            update_symbol(code, full_refresh=False)
        except Exception:
            logger.exception("%s の日次更新に失敗しました", code)
        time.sleep(0.3)


def update_universe(
    codes: Optional[List[str]] = None,
    full_refresh: bool = False,
    use_listed_master: bool = False,
    append_date: Optional[str] = None,
) -> None:
    target_codes = codes or build_universe(use_listed_master=use_listed_master)
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
                update_symbol(code, full_refresh=full_refresh)
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
        "--append-date",
        help="全銘柄の当日株価を追記する日付 (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--include-topix",
        action="store_true",
        help="銘柄の更新に加えて TOPIX 指数も保存する",
    )

    args = parser.parse_args()

    if args.codes:
        codes = [str(c).zfill(4) for c in args.codes]
    else:
        codes = build_universe(
            include_custom=args.include_custom,
            custom_path=args.custom_path,
            use_listed_master=args.use_listed_master,
        )

    update_universe(
        codes=codes,
        full_refresh=args.full_refresh,
        use_listed_master=args.use_listed_master,
        append_date=args.append_date,
    )

    if args.include_topix:
        try:
            update_topix(full_refresh=args.full_refresh)
        except Exception:
            logger.exception("TOPIX の更新に失敗しました")
