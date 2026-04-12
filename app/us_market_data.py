from datetime import date
from io import StringIO
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import requests

from .data_loader import _normalize_generic_ohlcv
from .leadlag_data import LEADLAG_US_PRICE_DIR, normalize_leadlag_symbol


STOOQ_DAILY_CSV_URL = "https://stooq.com/q/d/l/"
SUPPORTED_US_PROVIDERS = ["stooq_csv"]
US_PROVIDER_LABELS: Dict[str, str] = {
    "stooq_csv": "stooq_csv (stooq.com daily csv)",
}


def _to_iso_date(value: Optional[object], field_name: str) -> Optional[str]:
    if value is None:
        return None
    ts = pd.to_datetime(value, errors="coerce")
    if pd.isna(ts):
        raise ValueError("{} は YYYY-MM-DD 形式で指定してください。".format(field_name))
    return pd.Timestamp(ts).date().isoformat()


def _fetch_from_stooq_csv(symbol: str) -> pd.DataFrame:
    stooq_symbol = "{}.us".format(symbol.lower())
    response = requests.get(
        STOOQ_DAILY_CSV_URL,
        params={"s": stooq_symbol, "i": "d"},
        timeout=30,
    )
    response.raise_for_status()

    content = response.text.strip()
    if not content:
        raise ValueError("取得結果が空でした。ティッカーを確認してください。")

    if "No data" in content:
        raise ValueError("指定ティッカーのデータが見つかりませんでした。")

    try:
        df_raw = pd.read_csv(StringIO(content))
    except pd.errors.EmptyDataError:
        raise ValueError("取得結果が空でした。ティッカーを確認してください。")
    if df_raw.empty:
        raise ValueError("データが0件でした。ティッカーを確認してください。")

    if "No data" in "".join(df_raw.columns.astype(str)):
        raise ValueError("指定ティッカーのデータが見つかりませんでした。")

    return df_raw


def fetch_us_daily_ohlcv(
    symbol: str,
    start_date: Optional[object] = None,
    end_date: Optional[object] = None,
    provider: str = "stooq_csv",
) -> pd.DataFrame:
    normalized_symbol = normalize_leadlag_symbol(symbol)
    normalized_provider = str(provider).strip().lower()
    if normalized_provider not in SUPPORTED_US_PROVIDERS:
        raise ValueError(
            "provider は次のいずれかを指定してください: {}".format(
                ", ".join(SUPPORTED_US_PROVIDERS)
            )
        )

    start_iso = _to_iso_date(start_date, "start_date")
    end_iso = _to_iso_date(end_date, "end_date")
    if start_iso is not None and end_iso is not None and start_iso > end_iso:
        raise ValueError("start_date は end_date 以下にしてください。")

    if normalized_provider == "stooq_csv":
        source_df = _fetch_from_stooq_csv(normalized_symbol)
    else:
        raise ValueError("未対応のproviderです: {}".format(provider))

    normalized_df = _normalize_generic_ohlcv(source_df, symbol=normalized_symbol, market="US")
    normalized_df = normalized_df[["date", "open", "high", "low", "close", "volume"]].copy()
    normalized_df["date"] = pd.to_datetime(normalized_df["date"]).dt.normalize()

    if start_iso is not None:
        normalized_df = normalized_df[
            normalized_df["date"] >= pd.Timestamp(start_iso)
        ]
    if end_iso is not None:
        normalized_df = normalized_df[
            normalized_df["date"] <= pd.Timestamp(end_iso)
        ]

    normalized_df = normalized_df.sort_values("date").reset_index(drop=True)
    if normalized_df.empty:
        raise ValueError("指定期間で有効なOHLCVデータが取得できませんでした。")

    return normalized_df


def save_us_daily_csv(
    symbol: str,
    df: pd.DataFrame,
    target_dir: Optional[Path] = None,
) -> Path:
    normalized_symbol = normalize_leadlag_symbol(symbol)
    if df is None or df.empty:
        raise ValueError("保存対象データが空です。")

    normalized_df = _normalize_generic_ohlcv(df, symbol=normalized_symbol, market="US")
    output_df = normalized_df[["date", "open", "high", "low", "close", "volume"]].copy()
    output_df["date"] = pd.to_datetime(output_df["date"]).dt.strftime("%Y-%m-%d")

    base_dir = Path(target_dir) if target_dir is not None else LEADLAG_US_PRICE_DIR
    base_dir.mkdir(parents=True, exist_ok=True)
    output_path = base_dir / "{}.csv".format(normalized_symbol)
    output_df.to_csv(output_path, index=False)
    return output_path
