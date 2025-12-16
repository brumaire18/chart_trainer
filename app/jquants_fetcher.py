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
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

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


def _light_plan_window() -> tuple[str, str]:
    today = date.today()
    to_date = today - timedelta(weeks=LIGHT_PLAN_LAG_WEEKS)
    from_date = to_date - timedelta(days=LIGHT_PLAN_WINDOW_DAYS)
    return from_date.isoformat(), to_date.isoformat()


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


def update_symbol(code: str, full_refresh: bool = False) -> pd.DataFrame:
    code = str(code).zfill(4)
    client = _get_client()

    meta = _load_meta(code)
    existing_df = None if full_refresh else _load_existing_csv(code)

    from_light, to_light = _light_plan_window()

    if not full_refresh and meta.get("history_end") and existing_df is not None:
        from_date = (pd.to_datetime(meta["history_end"]) + pd.Timedelta(days=1)).date()
        to_date = datetime.fromisoformat(to_light).date()
        if from_date > to_date:
            logger.info("%s は取得可能期間外のためスキップします", code)
            return existing_df
        fetch_from, fetch_to = from_date.isoformat(), to_light
    else:
        fetch_from, fetch_to = from_light, to_light

    params = {"code": code, "from": fetch_from, "to": fetch_to}
    logger.info("%s の株価を取得します (from=%s, to=%s)", code, fetch_from, fetch_to)
    data = _request_with_token(client, "/v1/prices/daily_quotes", params=params)
    raw_quotes = data.get("daily_quotes") or data.get("dailyQuotes")
    if raw_quotes is None:
        raise JQuantsError("daily_quotes がレスポンスに存在しません。")

    df_raw = pd.DataFrame(raw_quotes)
    if df_raw.empty:
        raise JQuantsError(f"{code} の価格データが空でした。")

    normalized = _normalize_daily_quotes(df_raw, code)
    if normalized.empty:
        raise JQuantsError(f"{code} の正規化後データが空でした。")

    if existing_df is not None and not existing_df.empty:
        merged = pd.concat([existing_df, normalized], ignore_index=True)
        merged = merged.drop_duplicates(subset=["date"], keep="last")
        merged = merged.sort_values("date").reset_index(drop=True)
    else:
        merged = normalized

    if "datetime" not in merged.columns:
        merged["datetime"] = pd.to_datetime(merged["date"]).dt.strftime("%Y-%m-%dT%H:%M:%S")
    else:
        merged["datetime"] = pd.to_datetime(merged["datetime"]).dt.strftime("%Y-%m-%dT%H:%M:%S")

    PRICE_CSV_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = PRICE_CSV_DIR / f"{code}.csv"
    merged.to_csv(csv_path, index=False)

    history_start = merged["date"].min()
    history_end = merged["date"].max()
    meta_out = {
        "code": code,
        "market": meta.get("market") or normalized.get("market").iloc[0],
        "history_start": history_start,
        "history_end": history_end,
        "last_fetch_to": fetch_to,
        "last_fetch_at": datetime.now().astimezone().isoformat(),
        "plan": "LIGHT",
    }
    meta_path = _save_meta(code, meta_out)

    logger.info("%s を更新しました: %s - %s", code, history_start, history_end)
    return merged


def update_universe(
    codes: Optional[List[str]] = None,
    full_refresh: bool = False,
    use_listed_master: bool = False,
) -> None:
    target_codes = codes or build_universe(use_listed_master=use_listed_master)
    for code in target_codes:
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
    )
