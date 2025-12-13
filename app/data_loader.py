import pandas as pd
from typing import List
from .config import PRICE_CSV_DIR


def get_available_symbols() -> List[str]:
    """
    price_csvフォルダに存在するCSVファイル名から銘柄コード一覧を返す。
    例: 7203.csv -> "7203"
    """
    symbols: List[str] = []
    for path in PRICE_CSV_DIR.glob("*.csv"):
        symbols.append(path.stem)
    return sorted(symbols)


def _normalize_from_jquants(df_raw: pd.DataFrame) -> pd.DataFrame:
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
            "open": df_raw[open_col],
            "high": df_raw[high_col],
            "low": df_raw[low_col],
            "close": df_raw[close_col],
            "volume": df_raw[vol_col],
        }
    )

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

    # パターン1: すでに整形済み (date, open, high, low, close, volume)
    if all(col in df_raw.columns for col in ["date", "open", "high", "low", "close", "volume"]):
        df_raw["date"] = pd.to_datetime(df_raw["date"])
        df = df_raw[["date", "open", "high", "low", "close", "volume"]].copy()
        df = df.sort_values("date").reset_index(drop=True)
        return df

    # パターン2: J-Quants daily_quotes 形式
    if "Date" in df_raw.columns:
        return _normalize_from_jquants(df_raw)

    # どちらでもない場合はエラー
    raise ValueError(
        "サポートしていないCSV形式です。"
        "date/open/... または J-Quants daily_quotes のフォーマットにしてください。"
    )
