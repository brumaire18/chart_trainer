from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

from .config import PRICE_CSV_DIR
from .data_loader import get_available_symbols, load_price_csv


def _load_price_for_breadth(csv_path: Path, code: str, price_dir: Path) -> pd.DataFrame:
    if price_dir == PRICE_CSV_DIR:
        return load_price_csv(code)

    df_raw = pd.read_csv(csv_path)
    if df_raw.empty:
        return pd.DataFrame(columns=["date", "close", "volume"])

    if all(col in df_raw.columns for col in ["date", "close", "volume"]):
        return df_raw[["date", "close", "volume"]].copy()

    if "Date" in df_raw.columns:
        close_col = (
            "AdjustmentClose"
            if "AdjustmentClose" in df_raw.columns
            else "adjustmentClose"
            if "adjustmentClose" in df_raw.columns
            else "Close"
            if "Close" in df_raw.columns
            else None
        )
        volume_col = (
            "AdjustmentVolume"
            if "AdjustmentVolume" in df_raw.columns
            else "adjustmentVolume"
            if "adjustmentVolume" in df_raw.columns
            else "Volume"
            if "Volume" in df_raw.columns
            else None
        )
        if close_col is not None and volume_col is not None:
            return pd.DataFrame(
                {
                    "date": df_raw["Date"],
                    "close": df_raw[close_col],
                    "volume": df_raw[volume_col],
                }
            )

    raise ValueError(f"サポートしていないCSV形式です: {csv_path}")


def _summarize_symbol_breadth(df: pd.DataFrame) -> pd.DataFrame:
    """1銘柄分の終値推移から日次の上昇/下落集計を生成する。"""

    df_sorted = df.sort_values("date").copy()
    df_sorted["close_prev"] = df_sorted["close"].shift(1)
    df_sorted["change"] = df_sorted["close"] - df_sorted["close_prev"]

    daily = pd.DataFrame({
        "date": df_sorted["date"],
        "advancing_issues": (df_sorted["change"] > 0).astype(int),
        "declining_issues": (df_sorted["change"] < 0).astype(int),
        "unchanged_issues": (df_sorted["change"] == 0).astype(int),
        "advancing_volume": df_sorted["volume"].where(df_sorted["change"] > 0, 0),
        "declining_volume": df_sorted["volume"].where(df_sorted["change"] < 0, 0),
    })

    daily["advancing_volume"] = daily["advancing_volume"].fillna(0)
    daily["declining_volume"] = daily["declining_volume"].fillna(0)

    return daily


def aggregate_market_breadth(
    symbols: Optional[Iterable[str]] = None,
    price_dir: Path = PRICE_CSV_DIR,
    fill_missing_business_days: bool = False,
) -> pd.DataFrame:
    """
    data/price_csv 以下の銘柄を走査し、日次の上昇/下落銘柄数・出来高を集計する。

    既定では休場日を除外し、実データが存在する取引日のみを返す。
    必要に応じて fill_missing_business_days=True で営業日(B)補完を有効化できる。
    """

    target_symbols = list(symbols) if symbols is not None else get_available_symbols()

    if not target_symbols:
        return pd.DataFrame(
            columns=[
                "date",
                "advancing_issues",
                "declining_issues",
                "unchanged_issues",
                "advancing_volume",
                "declining_volume",
            ]
        )

    breadth_parts = []
    for code in target_symbols:
        csv_path = price_dir / f"{code}.csv"
        if not csv_path.exists():
            continue
        try:
            df_price = _load_price_for_breadth(csv_path, code, price_dir)
        except Exception:
            continue

        if df_price.empty:
            continue
        breadth_parts.append(_summarize_symbol_breadth(df_price))

    if not breadth_parts:
        return pd.DataFrame(
            columns=[
                "date",
                "advancing_issues",
                "declining_issues",
                "unchanged_issues",
                "advancing_volume",
                "declining_volume",
            ]
        )

    df_all = pd.concat(breadth_parts, ignore_index=True)
    df_all["date"] = pd.to_datetime(df_all["date"])

    grouped = (
        df_all.groupby("date", as_index=False)
        .agg(
            advancing_issues=("advancing_issues", "sum"),
            declining_issues=("declining_issues", "sum"),
            unchanged_issues=("unchanged_issues", "sum"),
            advancing_volume=("advancing_volume", "sum"),
            declining_volume=("declining_volume", "sum"),
        )
        .sort_values("date")
        .reset_index(drop=True)
    )

    if fill_missing_business_days and not grouped.empty:
        full_range = pd.date_range(grouped["date"].min(), grouped["date"].max(), freq="B")
        grouped = (
            grouped.set_index("date")
            .reindex(full_range, fill_value=0)
            .rename_axis("date")
            .reset_index()
        )

    grouped["net_issues"] = grouped["advancing_issues"] - grouped["declining_issues"]
    grouped["total_issues"] = (
        grouped["advancing_issues"]
        + grouped["declining_issues"]
        + grouped["unchanged_issues"]
    )
    grouped["total_volume"] = grouped["advancing_volume"] + grouped["declining_volume"]

    return grouped


def compute_breadth_indicators(df_breadth: pd.DataFrame) -> pd.DataFrame:
    """騰落ライン・マクレラン指標・TRINを計算して返す。"""

    if df_breadth.empty:
        return df_breadth.copy()

    df = df_breadth.copy()
    df["date"] = pd.to_datetime(df["date"])

    net_series = df["net_issues"]
    df["advance_decline_line"] = net_series.cumsum()

    ema_fast = net_series.ewm(span=19, adjust=False).mean()
    ema_slow = net_series.ewm(span=39, adjust=False).mean()
    df["mcclellan_oscillator"] = ema_fast - ema_slow
    df["mcclellan_summation"] = df["mcclellan_oscillator"].cumsum()

    adv_issues = df["advancing_issues"].replace(0, pd.NA)
    dec_issues = df["declining_issues"].replace(0, pd.NA)
    adv_volume = df["advancing_volume"].replace(0, pd.NA)
    dec_volume = df["declining_volume"].replace(0, pd.NA)

    trin_numerator = adv_issues / dec_issues
    trin_denominator = adv_volume / dec_volume
    df["trin"] = trin_numerator / trin_denominator

    return df
