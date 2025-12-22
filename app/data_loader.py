from datetime import date
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np

from .config import (
    JQUANTS_BASE_URL,
    JQUANTS_MAILADDRESS,
    JQUANTS_PASSWORD,
    JQUANTS_REFRESH_TOKEN,
    PRICE_CSV_DIR,
)
from .jquants_client import JQuantsClient


def get_available_symbols() -> List[str]:
    """
    price_csvフォルダに存在するCSVファイル名から銘柄コード一覧を返す。
    例: 7203.csv -> "7203"
    """
    symbols: List[str] = []
    for path in PRICE_CSV_DIR.glob("*.csv"):
        symbols.append(path.stem)
    return sorted(symbols)


def enforce_light_plan_window(
    start_date: str,
    end_date: str,
    max_years: int = 5,
    expand_to_max_window: bool = True,
) -> Tuple[str, str, bool]:
    """
    J-Quants ライトプランが取得できる期間 (既定: 過去5年) に合わせて
    取得期間を補正する。

    - expand_to_max_window=True の場合は、指定日付が制限内でも
      可能な限り過去 (max_weeks 分) まで自動的に広げる。
    - False の場合は、指定期間が制限を超えないように切り上げるのみ。

    Returns:
        (adjusted_start, adjusted_end, adjusted_flag)

    Raises:
        ValueError: 終了日が開始日よりも前になる場合。
    """

    start_ts = pd.to_datetime(start_date).normalize()
    end_ts = pd.to_datetime(end_date).normalize()

    today = pd.Timestamp(date.today())

    earliest_allowed = today - pd.DateOffset(years=max_years)
    adjusted = False

    if expand_to_max_window and start_ts > earliest_allowed:
        start_ts = earliest_allowed
        adjusted = True
    elif start_ts < earliest_allowed:
        start_ts = earliest_allowed
        adjusted = True

    if end_ts > today:
        end_ts = today
        adjusted = True

    if end_ts < start_ts:
        raise ValueError(
            "終了日は開始日以降にしてください。(ライトプランは過去5年まで取得可能です)"
        )

    return start_ts.date().isoformat(), end_ts.date().isoformat(), adjusted


def _normalize_from_jquants(
    df_raw: pd.DataFrame, symbol: Optional[str] = None, market: Optional[str] = None
) -> pd.DataFrame:
    """
    J-Quantsのdaily_quotes形式
    (Date, Code, Open, High, Low, Close, ..., AdjustmentOpen, ...) から、
    date, open, high, low, close, volume の6列に正規化する。
    """
    # 日付列
    if "Date" in df_raw.columns:
        df_raw["date"] = pd.to_datetime(df_raw["Date"])
    elif "date" in df_raw.columns:
        df_raw["date"] = pd.to_datetime(df_raw["date"])
    else:
        raise ValueError("J-Quants形式のCSVに Date/date 列が見つかりません。")

    # 調整後OHLCVがあればそちらを優先
    if all(
        col in df_raw.columns
        for col in ["AdjustmentOpen", "AdjustmentHigh", "AdjustmentLow", "AdjustmentClose"]
    ):
        open_col = "AdjustmentOpen"
        high_col = "AdjustmentHigh"
        low_col = "AdjustmentLow"
        close_col = "AdjustmentClose"
        vol_col = "AdjustmentVolume" if "AdjustmentVolume" in df_raw.columns else "Volume"
    else:
        # 調整後がなければ素の Open/High/Low/Close/Volume を使う
        open_col = "Open" if "Open" in df_raw.columns else "open"
        high_col = "High" if "High" in df_raw.columns else "high"
        low_col = "Low" if "Low" in df_raw.columns else "low"
        close_col = "Close" if "Close" in df_raw.columns else "close"
        vol_col = "Volume" if "Volume" in df_raw.columns else "volume"

    df = pd.DataFrame(
        {
            "date": df_raw["date"],
            "datetime": pd.to_datetime(df_raw["date"]),
            "code": str(symbol).zfill(4) if symbol is not None else df_raw.get("code"),
            "market": market or df_raw.get("market"),
            "open": df_raw[open_col],
            "high": df_raw[high_col],
            "low": df_raw[low_col],
            "close": df_raw[close_col],
            "volume": df_raw[vol_col],
        }
    )

    df = df.dropna(subset=["date", "open", "high", "low", "close", "volume"])
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df["datetime"] = pd.to_datetime(df["datetime"]).dt.strftime("%Y-%m-%dT%H:%M:%S")
    df = df.sort_values("date").reset_index(drop=True)
    return df


def load_price_csv(symbol: str) -> pd.DataFrame:
    """
    シンボルに対応するCSVを読み込んで、
    date, open, high, low, close, volume の6列を持つDataFrameを返す。

    - 既にその形式になっているCSV
    - J-Quants daily_quotes 形式のCSV
    の両方をサポートする。
    """
    csv_path = PRICE_CSV_DIR / f"{symbol}.csv"
    df_raw = pd.read_csv(csv_path)

    # パターン1: すでに整形済み
    normalized_cols = ["date", "open", "high", "low", "close", "volume"]
    if all(col in df_raw.columns for col in normalized_cols):
        df_raw["date"] = pd.to_datetime(df_raw["date"])
        ordered_cols = []
        for col in [
            "date",
            "datetime",
            "code",
            "market",
            "open",
            "high",
            "low",
            "close",
            "volume",
        ]:
            if col in df_raw.columns and col not in ordered_cols:
                ordered_cols.append(col)
        df = df_raw[ordered_cols].copy()
        df = df.sort_values("date").reset_index(drop=True)
        if "code" not in df.columns:
            df["code"] = symbol
        if "market" not in df.columns:
            df["market"] = None
        df["date"] = pd.to_datetime(df["date"]).dt.normalize()
        if "datetime" not in df.columns:
            df["datetime"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%dT%H:%M:%S")
        return df

    # パターン2: J-Quants daily_quotes 形式
    if "Date" in df_raw.columns:
        return _normalize_from_jquants(df_raw, symbol=symbol)

    # どちらでもない場合はエラー
    raise ValueError(
        "サポートしていないCSV形式です。"
        "date/open/... または J-Quants daily_quotes のフォーマットにしてください。"
    )


def fetch_and_save_price_csv(symbol: str, start_date: str, end_date: str) -> Path:
    """
    J-Quantsから株価データを取得し、PRICE_CSV_DIRに保存する。
    start_date/end_dateはYYYY-MM-DD形式の文字列。
    """
    if not symbol:
        raise ValueError("銘柄コードが指定されていません。")

    adjusted_start, adjusted_end, _ = enforce_light_plan_window(start_date, end_date)

    client = JQuantsClient(
        refresh_token=JQUANTS_REFRESH_TOKEN,
        base_url=JQUANTS_BASE_URL,
        mailaddress=JQUANTS_MAILADDRESS,
        password=JQUANTS_PASSWORD,
    )
    df_raw = client.fetch_daily_quotes(symbol, adjusted_start, adjusted_end)
    df_normalized = _normalize_from_jquants(df_raw, symbol=symbol)

    PRICE_CSV_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = PRICE_CSV_DIR / f"{symbol}.csv"
    df_normalized.to_csv(csv_path, index=False)
    return csv_path


def load_topix_csv() -> pd.DataFrame:
    """TOPIX の日次CSVを読み込む。"""

    csv_path = PRICE_CSV_DIR / "topix.csv"
    if not csv_path.exists():
        raise FileNotFoundError("TOPIX のCSVファイル (data/price_csv/topix.csv) が見つかりません。")

    df_raw = pd.read_csv(csv_path)
    if "date" not in df_raw.columns or "close" not in df_raw.columns:
        raise ValueError("topix.csv に必要な date / close 列が存在しません。")

    df_raw["date"] = pd.to_datetime(df_raw["date"]).dt.normalize()
    if "datetime" in df_raw.columns:
        df_raw["datetime"] = pd.to_datetime(df_raw["datetime"])
    else:
        df_raw["datetime"] = df_raw["date"]

    for price_col in ["open", "high", "low", "close"]:
        if price_col in df_raw.columns:
            df_raw[price_col] = pd.to_numeric(df_raw[price_col], errors="coerce")

    df = df_raw[
        [col for col in ["date", "datetime", "open", "high", "low", "close"] if col in df_raw.columns]
    ].copy()
    df = df.dropna(subset=["date", "close"]).sort_values("date").reset_index(drop=True)
    df["datetime"] = df["datetime"].dt.strftime("%Y-%m-%dT%H:%M:%S")
    return df


def attach_topix_relative_strength(
    df_stock: pd.DataFrame, topix_df: Optional[pd.DataFrame] = None
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """
    銘柄データとTOPIXを日付でinner joinして RS (対TOPIX) を計算する。

    Returns:
        merged_df: TOPIX列とRS列を付与したDataFrame（重複日付は最後にソート）
        info: 追加情報 (coverage_ratio, missing_rows など)
    """

    if topix_df is None:
        topix_df = load_topix_csv()

    df_stock = df_stock.copy()
    df_stock["date"] = pd.to_datetime(df_stock["date"]).dt.normalize()

    df_topix = topix_df.copy()
    df_topix["date"] = pd.to_datetime(df_topix["date"]).dt.normalize()
    df_topix = df_topix.dropna(subset=["date", "close"])

    merged = pd.merge(
        df_stock,
        df_topix[["date", "close"]].rename(columns={"close": "topix_close"}),
        on="date",
        how="inner",
        validate="many_to_one",
    )

    info: Dict[str, object] = {
        "source_rows": float(len(df_stock)),
        "merged_rows": float(len(merged)),
        "missing_rows": float(len(df_stock) - len(merged)),
    }
    info["coverage_ratio"] = float(len(merged) / len(df_stock)) if len(df_stock) else 0.0

    if merged.empty:
        info["status"] = "empty_merge"
        return df_stock, info

    merged["topix_rs"] = merged["close"] / merged["topix_close"]
    merged["topix_rs_log"] = np.log(merged["close"]) - np.log(merged["topix_close"])
    merged = merged.sort_values("date").reset_index(drop=True)
    info["status"] = "ok"
    info["start_date"] = merged["date"].min()
    info["end_date"] = merged["date"].max()
    return merged, info
