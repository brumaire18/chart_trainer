from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd

from .config import META_DIR
from .data_loader import get_available_symbols, load_price_csv


MINERVINI_CACHE_DIR = META_DIR / "minervini_cache"
MINERVINI_CACHE_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class MinerviniScreenConfig:
    """ミネルヴィニ・トレンドテンプレート用の設定。"""

    rs_threshold: float = 70.0
    low_from_low_pct: float = -0.3
    high_from_high_pct: float = 0.25
    slope_lookback_days: int = 20


def _cache_path(symbol: str) -> Path:
    return MINERVINI_CACHE_DIR / f"{str(symbol).zfill(4)}.csv"


def _load_cached_indicators(symbol: str) -> Optional[pd.DataFrame]:
    cache_file = _cache_path(symbol)
    if not cache_file.exists():
        return None
    try:
        cached = pd.read_csv(cache_file)
    except Exception:
        return None
    if cached.empty:
        return None
    cached["date"] = pd.to_datetime(cached["date"]).dt.normalize()
    return cached


def _save_cached_indicators(symbol: str, df: pd.DataFrame) -> None:
    cache_file = _cache_path(symbol)
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache_file, index=False)


def _is_cache_valid(cache_df: pd.DataFrame, price_df: pd.DataFrame) -> bool:
    if cache_df is None or cache_df.empty or price_df.empty:
        return False
    try:
        cache_last_date = pd.to_datetime(cache_df["date"]).iloc[-1]
        price_last_date = pd.to_datetime(price_df["date"]).iloc[-1]
    except Exception:
        return False
    if cache_last_date != price_last_date:
        return False
    if len(cache_df) != len(price_df):
        return False
    return True


def compute_minervini_indicators(
    price_df: pd.DataFrame, slope_lookback_days: int = 20
) -> pd.DataFrame:
    """移動平均と52週高値/安値などを計算する。"""

    if price_df.empty:
        return price_df.copy()

    df = price_df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna(subset=["date", "close"]).sort_values("date").reset_index(drop=True)

    df["sma50"] = df["close"].rolling(window=50, min_periods=50).mean()
    df["sma150"] = df["close"].rolling(window=150, min_periods=150).mean()
    df["sma200"] = df["close"].rolling(window=200, min_periods=200).mean()

    df["sma200_slope"] = df["sma200"] - df["sma200"].shift(slope_lookback_days)

    df["low_52w"] = df["close"].rolling(window=252, min_periods=252).min()
    df["high_52w"] = df["close"].rolling(window=252, min_periods=252).max()
    df["return_52w"] = df["close"] / df["close"].shift(252) - 1

    return df[[
        "date",
        "close",
        "sma50",
        "sma150",
        "sma200",
        "sma200_slope",
        "low_52w",
        "high_52w",
        "return_52w",
    ]]


def get_minervini_indicators(
    symbol: str,
    price_df: pd.DataFrame,
    slope_lookback_days: int = 20,
) -> pd.DataFrame:
    """キャッシュを利用して指標を取得する。"""

    cached = _load_cached_indicators(symbol)
    if cached is not None and _is_cache_valid(cached, price_df):
        return cached

    computed = compute_minervini_indicators(price_df, slope_lookback_days=slope_lookback_days)
    if not computed.empty:
        _save_cached_indicators(symbol, computed)
    return computed


def _meets_low_condition(price: float, low_52w: float, low_from_low_pct: float) -> bool:
    if pd.isna(price) or pd.isna(low_52w):
        return False
    threshold = low_52w * (1 + low_from_low_pct)
    if low_from_low_pct >= 0:
        return price >= threshold
    return price <= threshold


def _meets_high_condition(price: float, high_52w: float, high_from_high_pct: float) -> bool:
    if pd.isna(price) or pd.isna(high_52w):
        return False
    return price >= high_52w * (1 - high_from_high_pct)


def screen_minervini_trend_template(
    symbols: Optional[Iterable[str]] = None,
    config: Optional[MinerviniScreenConfig] = None,
) -> pd.DataFrame:
    """ミネルヴィニ・トレンドテンプレート条件でスクリーニングする。"""

    config = config or MinerviniScreenConfig()
    target_symbols: List[str] = list(symbols) if symbols is not None else get_available_symbols()
    snapshots = []

    for symbol in target_symbols:
        try:
            price_df = load_price_csv(symbol)
        except Exception:
            continue
        if price_df.empty:
            continue

        indicator_df = get_minervini_indicators(
            symbol, price_df, slope_lookback_days=config.slope_lookback_days
        )
        if indicator_df.empty:
            continue

        latest = indicator_df.iloc[-1]
        price = latest["close"]
        sma50 = latest["sma50"]
        sma150 = latest["sma150"]
        sma200 = latest["sma200"]
        low_52w = latest["low_52w"]
        high_52w = latest["high_52w"]

        snapshots.append(
            {
                "symbol": str(symbol).zfill(4),
                "date": latest["date"],
                "close": price,
                "sma50": sma50,
                "sma150": sma150,
                "sma200": sma200,
                "sma200_slope": latest["sma200_slope"],
                "low_52w": low_52w,
                "high_52w": high_52w,
                "return_52w": latest["return_52w"],
            }
        )

    df = pd.DataFrame(snapshots)
    if df.empty:
        return df

    df["rs_rating"] = df["return_52w"].rank(pct=True) * 100

    df["price_above_150_200"] = (df["close"] > df["sma150"]) & (df["close"] > df["sma200"])
    df["ma150_above_200"] = df["sma150"] > df["sma200"]
    df["sma200_rising"] = df["sma200_slope"] > 0
    df["ma50_above_150_200"] = (df["sma50"] > df["sma150"]) & (df["sma50"] > df["sma200"])
    df["price_above_50"] = df["close"] > df["sma50"]

    df["low_condition"] = df.apply(
        lambda row: _meets_low_condition(
            row["close"], row["low_52w"], config.low_from_low_pct
        ),
        axis=1,
    )
    df["high_condition"] = df.apply(
        lambda row: _meets_high_condition(
            row["close"], row["high_52w"], config.high_from_high_pct
        ),
        axis=1,
    )
    df["rs_condition"] = df["rs_rating"] >= config.rs_threshold

    condition_cols = [
        "price_above_150_200",
        "ma150_above_200",
        "sma200_rising",
        "ma50_above_150_200",
        "price_above_50",
        "low_condition",
        "high_condition",
        "rs_condition",
    ]
    df["passes_trend_template"] = df[condition_cols].all(axis=1)

    return df.sort_values(["passes_trend_template", "rs_rating"], ascending=[False, False]).reset_index(
        drop=True
    )
