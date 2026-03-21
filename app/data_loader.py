from datetime import date
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np

from .config import (
    JQUANTS_BASE_URL,
    JQUANTS_API_KEY,
    PRICE_CSV_DIR,
)
from .jquants_client import JQuantsClient
from .leadlag_data import (
    LeadLagUniverseItem,
    find_leadlag_item,
    get_leadlag_price_dir,
    load_leadlag_universe,
    normalize_leadlag_market,
)


MANUAL_STOCK_SPLITS: Dict[str, List[Tuple[str, float]]] = {
    "6039": [("2025-12-15", 5.0)],
    "1333": [("2025-12-29", 3.0)],
    "1414": [("2025-12-29", 4.0)],
    "1980": [("2025-12-29", 3.0)],
    "2114": [("2025-12-29", 2.0)],
    "2146": [("2025-12-29", 15.0)],
    "2501": [("2025-12-29", 5.0)],
    "3661": [("2025-12-29", 2.0)],
    "3916": [("2025-12-29", 2.0)],
    "3986": [("2025-12-29", 3.0)],
    "4062": [("2025-12-29", 2.0)],
    "4107": [("2025-12-29", 10.0)],
    "4183": [("2025-12-29", 2.0)],
    "4396": [("2025-12-29", 2.0)],
    "4635": [("2025-12-29", 5.0)],
    "4811": [("2025-12-29", 3.0)],
    "4812": [("2025-12-29", 3.0)],
    "4828": [("2025-12-29", 5.0)],
    "4935": [("2025-12-29", 5.0)],
    "4976": [("2025-12-29", 3.0)],
    "5108": [("2025-12-29", 2.0)],
    "5262": [("2025-12-29", 2.0)],
    "6061": [("2025-12-29", 2.0)],
    "6328": [("2025-12-29", 2.0)],
    "6369": [("2025-12-29", 2.0)],
    "6592": [("2025-12-29", 2.0)],
    "6648": [("2025-12-29", 5.0)],
    "6772": [("2025-12-29", 5.0)],
    "7089": [("2025-12-29", 2.0)],
    "7409": [("2025-12-29", 3.0)],
    "7552": [("2025-12-29", 2.0)],
    "7609": [("2025-12-29", 2.0)],
    "7628": [("2025-12-29", 2.0)],
    "8001": [("2025-12-29", 5.0)],
    "8020": [("2025-12-29", 2.0)],
    "8179": [("2025-12-29", 2.0)],
    "8830": [("2025-12-29", 2.0)],
    "9722": [("2025-12-29", 5.0)],
    "9757": [("2025-12-29", 2.0)],
}


def _needs_manual_split_adjustment(
    df: pd.DataFrame, split_date: pd.Timestamp, factor: float
) -> bool:
    if df.empty:
        return False

    before_split = df[df["date"] < split_date]
    after_split = df[df["date"] >= split_date]
    if before_split.empty or after_split.empty:
        return False

    pre_close = pd.to_numeric(before_split.iloc[-1]["close"], errors="coerce")
    post_close = pd.to_numeric(after_split.iloc[0]["close"], errors="coerce")
    if pd.isna(pre_close) or pd.isna(post_close) or post_close == 0:
        return False

    observed_ratio = pre_close / post_close
    return observed_ratio >= factor * 0.8


def apply_manual_stock_split_adjustments(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    splits = MANUAL_STOCK_SPLITS.get(str(symbol).zfill(4))
    if not splits:
        return df

    adjusted = df.copy()
    adjusted["date"] = pd.to_datetime(adjusted["date"]).dt.normalize()

    for split_date_str, factor in sorted(splits, key=lambda item: item[0]):
        split_date = pd.to_datetime(split_date_str).normalize()
        if not _needs_manual_split_adjustment(adjusted, split_date, factor):
            continue

        before_mask = adjusted["date"] < split_date
        price_cols = [col for col in ["open", "high", "low", "close"] if col in adjusted.columns]
        for col in price_cols:
            adjusted.loc[before_mask, col] = (
                pd.to_numeric(adjusted.loc[before_mask, col], errors="coerce") / factor
            )

        if "volume" in adjusted.columns:
            adjusted.loc[before_mask, "volume"] = (
                pd.to_numeric(adjusted.loc[before_mask, "volume"], errors="coerce")
                * factor
            )

    return adjusted


def get_available_symbols() -> List[str]:
    """
    price_csvフォルダに存在するCSVファイル名から銘柄コード一覧を返す。
    例: 7203.csv -> "7203"
    """
    symbols: List[str] = []
    for path in PRICE_CSV_DIR.glob("*.csv"):
        symbols.append(path.stem)
    return sorted(symbols)


def _find_first_existing_column(
    df_raw: pd.DataFrame, candidates: List[str]
) -> Optional[str]:
    for candidate in candidates:
        if candidate in df_raw.columns:
            return candidate
    return None


def _looks_like_jquants_daily_quotes(df_raw: pd.DataFrame) -> bool:
    if "Date" not in df_raw.columns:
        return False
    if "Code" in df_raw.columns:
        return True
    adjustment_candidates = [
        "AdjustmentFactor",
        "AdjustmentOpen",
        "AdjustmentClose",
        "AdjO",
        "AdjC",
    ]
    return any(col in df_raw.columns for col in adjustment_candidates)


def _normalize_generic_ohlcv(
    df_raw: pd.DataFrame, symbol: str, market: str
) -> pd.DataFrame:
    date_col = _find_first_existing_column(
        df_raw,
        [
            "date",
            "Date",
            "datetime",
            "DateTime",
            "timestamp",
            "Timestamp",
            "time",
            "Time",
        ],
    )
    open_col = _find_first_existing_column(df_raw, ["open", "Open", "O"])
    high_col = _find_first_existing_column(df_raw, ["high", "High", "H"])
    low_col = _find_first_existing_column(df_raw, ["low", "Low", "L"])
    close_col = _find_first_existing_column(df_raw, ["close", "Close", "C"])
    volume_col = _find_first_existing_column(df_raw, ["volume", "Volume", "Vo"])

    if not all([date_col, open_col, high_col, low_col, close_col, volume_col]):
        raise ValueError(
            "Unsupported leadlag csv format. Expected date/open/high/low/close/volume columns."
        )

    df = pd.DataFrame(
        {
            "date": pd.to_datetime(df_raw[date_col], errors="coerce"),
            "open": pd.to_numeric(df_raw[open_col], errors="coerce"),
            "high": pd.to_numeric(df_raw[high_col], errors="coerce"),
            "low": pd.to_numeric(df_raw[low_col], errors="coerce"),
            "close": pd.to_numeric(df_raw[close_col], errors="coerce"),
            "volume": pd.to_numeric(df_raw[volume_col], errors="coerce"),
        }
    )
    df = df.dropna(subset=["date", "open", "high", "low", "close", "volume"])
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df["datetime"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%dT%H:%M:%S")
    df["code"] = symbol
    df["market"] = market
    return df[
        ["date", "datetime", "code", "market", "open", "high", "low", "close", "volume"]
    ].sort_values("date").reset_index(drop=True)


def _normalize_leadlag_price_df(
    df_raw: pd.DataFrame, symbol: str, market: str
) -> pd.DataFrame:
    normalized_cols = ["date", "open", "high", "low", "close", "volume"]
    if all(col in df_raw.columns for col in normalized_cols):
        df = df_raw.copy()
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=["date", "open", "high", "low", "close", "volume"])
        df["date"] = pd.to_datetime(df["date"]).dt.normalize()
        if "datetime" in df.columns:
            dt_values = pd.to_datetime(df["datetime"], errors="coerce")
            dt_values = dt_values.fillna(pd.to_datetime(df["date"]))
        else:
            dt_values = pd.to_datetime(df["date"])
        df["datetime"] = dt_values.dt.strftime("%Y-%m-%dT%H:%M:%S")
        df["code"] = symbol
        df["market"] = market
        return df[
            ["date", "datetime", "code", "market", "open", "high", "low", "close", "volume"]
        ].sort_values("date").reset_index(drop=True)

    if _looks_like_jquants_daily_quotes(df_raw):
        return _normalize_from_jquants(df_raw, symbol=symbol, market=market)

    return _normalize_generic_ohlcv(df_raw, symbol=symbol, market=market)


def _resolve_leadlag_item(symbol: str) -> LeadLagUniverseItem:
    item = find_leadlag_item(symbol, universe_items=load_leadlag_universe())
    if item is None:
        raise ValueError(
            "Symbol {} is not in leadlag universe definition.".format(symbol)
        )
    return item


def get_available_leadlag_symbols(market: Optional[str] = None) -> List[str]:
    normalized_market = normalize_leadlag_market(market)
    symbols: List[str] = []
    for item in load_leadlag_universe():
        if normalized_market and item.market != normalized_market:
            continue
        csv_path = get_leadlag_price_dir(item.market) / "{}.csv".format(item.symbol)
        if csv_path.exists():
            symbols.append(item.symbol)
    return sorted(symbols)


def load_leadlag_price(symbol: str) -> pd.DataFrame:
    item = _resolve_leadlag_item(symbol)
    csv_path = get_leadlag_price_dir(item.market) / "{}.csv".format(item.symbol)
    if not csv_path.exists():
        raise FileNotFoundError(
            "leadlag csv was not found: {}. expected under data/price_csv/leadlag_us or leadlag_jp".format(
                csv_path
            )
        )
    df_raw = pd.read_csv(csv_path)
    df = _normalize_leadlag_price_df(df_raw, symbol=item.symbol, market=item.market)
    df["symbol"] = item.symbol
    df["name"] = item.name
    df["sector"] = item.sector
    df["style_bucket"] = item.style_bucket
    df["path_group"] = item.path_group
    return df[
        [
            "date",
            "datetime",
            "symbol",
            "code",
            "market",
            "name",
            "sector",
            "style_bucket",
            "path_group",
            "open",
            "high",
            "low",
            "close",
            "volume",
        ]
    ].sort_values("date").reset_index(drop=True)


def load_leadlag_panel(market: Optional[str] = None) -> pd.DataFrame:
    target_symbols = get_available_leadlag_symbols(market=market)
    if not target_symbols:
        return pd.DataFrame(
            columns=[
                "date",
                "datetime",
                "symbol",
                "code",
                "market",
                "name",
                "sector",
                "style_bucket",
                "path_group",
                "open",
                "high",
                "low",
                "close",
                "volume",
            ]
        )

    frames: List[pd.DataFrame] = []
    for symbol in target_symbols:
        frames.append(load_leadlag_price(symbol))
    panel_df = pd.concat(frames, axis=0, ignore_index=True)
    panel_df["date"] = pd.to_datetime(panel_df["date"]).dt.normalize()
    panel_df = panel_df.sort_values(["date", "market", "symbol"]).reset_index(drop=True)
    return panel_df


def compute_us_close_to_close_returns(
    us_price_df: pd.DataFrame,
    return_col: str = "us_close_to_close_return",
) -> pd.DataFrame:
    if "date" not in us_price_df.columns or "close" not in us_price_df.columns:
        raise ValueError("us_price_df must include date and close columns")

    df = us_price_df.copy()
    if "market" in df.columns:
        df = df[df["market"].astype(str).str.upper() == "US"]

    symbol_col = "symbol" if "symbol" in df.columns else "code"
    if symbol_col not in df.columns:
        raise ValueError("us_price_df must include symbol or code column")

    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna(subset=["date", "close", symbol_col])
    df = df.sort_values([symbol_col, "date"]).reset_index(drop=True)

    df["prev_close_us"] = df.groupby(symbol_col)["close"].shift(1)
    df[return_col] = df["close"] / df["prev_close_us"] - 1.0
    df["signal_date_us"] = df["date"]
    df["signal_weekday_us"] = df["signal_date_us"].dt.day_name()
    df = df.dropna(subset=[return_col, "signal_date_us"]).reset_index(drop=True)
    return df


def compute_jp_open_to_close_returns(
    jp_price_df: pd.DataFrame,
    return_col: str = "jp_open_to_close_return",
) -> pd.DataFrame:
    required_cols = {"date", "open", "close"}
    missing_cols = [col for col in required_cols if col not in jp_price_df.columns]
    if missing_cols:
        raise ValueError(
            "jp_price_df must include columns: {}".format(",".join(sorted(required_cols)))
        )

    df = jp_price_df.copy()
    if "market" in df.columns:
        df = df[df["market"].astype(str).str.upper() == "JP"]

    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    df["open"] = pd.to_numeric(df["open"], errors="coerce")
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna(subset=["date", "open", "close"]).reset_index(drop=True)
    df = df[df["open"] != 0].copy()

    df[return_col] = df["close"] / df["open"] - 1.0
    df["trade_date_jp"] = df["date"]
    df["trade_weekday_jp"] = df["trade_date_jp"].dt.day_name()
    return df


def _build_us_to_jp_calendar_mapping(
    us_signal_dates: pd.Series, jp_trade_dates: pd.Series
) -> pd.DataFrame:
    normalized_us_dates = pd.DatetimeIndex(
        pd.to_datetime(us_signal_dates, errors="coerce").dropna().dt.normalize().unique()
    ).sort_values()
    normalized_jp_dates = pd.DatetimeIndex(
        pd.to_datetime(jp_trade_dates, errors="coerce").dropna().dt.normalize().unique()
    ).sort_values()

    mapping_columns = [
        "signal_date_us",
        "trade_date_jp",
        "signal_weekday_us",
        "trade_weekday_jp",
        "calendar_lag_days",
        "is_signal_friday",
    ]
    if len(normalized_us_dates) == 0 or len(normalized_jp_dates) == 0:
        return pd.DataFrame(columns=mapping_columns)

    records: List[Dict[str, object]] = []
    for signal_date in normalized_us_dates:
        next_idx = normalized_jp_dates.searchsorted(signal_date, side="right")
        if next_idx >= len(normalized_jp_dates):
            continue
        trade_date = normalized_jp_dates[next_idx]
        records.append(
            {
                "signal_date_us": signal_date,
                "trade_date_jp": trade_date,
                "signal_weekday_us": signal_date.day_name(),
                "trade_weekday_jp": trade_date.day_name(),
                "calendar_lag_days": int((trade_date - signal_date).days),
                "is_signal_friday": bool(signal_date.dayofweek == 4),
            }
        )
    return pd.DataFrame(records, columns=mapping_columns)


def align_us_signal_and_jp_target(
    us_signal_df: pd.DataFrame,
    jp_target_df: pd.DataFrame,
    join_key: str = "path_group",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if us_signal_df.empty or jp_target_df.empty:
        mapping_df = _build_us_to_jp_calendar_mapping(
            us_signal_df.get("signal_date_us", us_signal_df.get("date", pd.Series(dtype="datetime64[ns]"))),
            jp_target_df.get("trade_date_jp", jp_target_df.get("date", pd.Series(dtype="datetime64[ns]"))),
        )
        return pd.DataFrame(), mapping_df

    us = us_signal_df.copy()
    jp = jp_target_df.copy()

    if "market" in us.columns:
        us = us[us["market"].astype(str).str.upper() == "US"]
    if "market" in jp.columns:
        jp = jp[jp["market"].astype(str).str.upper() == "JP"]

    if us.empty or jp.empty:
        return pd.DataFrame(), pd.DataFrame(
            columns=[
                "signal_date_us",
                "trade_date_jp",
                "signal_weekday_us",
                "trade_weekday_jp",
                "calendar_lag_days",
                "is_signal_friday",
            ]
        )

    us_date_col = "signal_date_us" if "signal_date_us" in us.columns else "date"
    jp_date_col = "trade_date_jp" if "trade_date_jp" in jp.columns else "date"

    us["signal_date_us"] = pd.to_datetime(us[us_date_col], errors="coerce").dt.normalize()
    jp["trade_date_jp"] = pd.to_datetime(jp[jp_date_col], errors="coerce").dt.normalize()
    us = us.dropna(subset=["signal_date_us"]).reset_index(drop=True)
    jp = jp.dropna(subset=["trade_date_jp"]).reset_index(drop=True)

    mapping_df = _build_us_to_jp_calendar_mapping(
        us_signal_dates=us["signal_date_us"],
        jp_trade_dates=jp["trade_date_jp"],
    )
    if mapping_df.empty:
        return pd.DataFrame(), mapping_df

    mapped_us = pd.merge(
        us,
        mapping_df,
        on="signal_date_us",
        how="inner",
        validate="many_to_one",
    )

    join_cols = ["trade_date_jp"]
    if (
        join_key
        and join_key in mapped_us.columns
        and join_key in jp.columns
    ):
        join_cols.append(join_key)

    aligned_df = pd.merge(
        mapped_us,
        jp,
        on=join_cols,
        how="inner",
        suffixes=("_us", "_jp"),
    )
    aligned_df = aligned_df.sort_values(
        ["trade_date_jp", "signal_date_us"]
    ).reset_index(drop=True)
    return aligned_df, mapping_df


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
    date_col = None
    for candidate in ("Date", "date", "quoteDate", "datetime", "DateTime"):
        if candidate in df_raw.columns:
            date_col = candidate
            break
    if date_col:
        df_raw["date"] = pd.to_datetime(df_raw[date_col])
    else:
        raise ValueError("J-Quants形式のCSVに Date/date 列が見つかりません。")

    # 調整後OHLCVがあればそちらを優先
    adjustment_cols = (
        "AdjO",
        "AdjH",
        "AdjL",
        "AdjC",
        "AdjustmentOpen",
        "AdjustmentHigh",
        "AdjustmentLow",
        "AdjustmentClose",
        "adjustmentOpen",
        "adjustmentHigh",
        "adjustmentLow",
        "adjustmentClose",
    )
    adjustment_factor_cols = (
        "AdjustmentFactor",
        "adjustmentFactor",
        "adjustment_factor",
    )
    adjustment_factor_col = next(
        (col for col in adjustment_factor_cols if col in df_raw.columns), None
    )

    if any(col in df_raw.columns for col in adjustment_cols):
        open_col = (
            "AdjO"
            if "AdjO" in df_raw.columns
            else "AdjustmentOpen"
            if "AdjustmentOpen" in df_raw.columns
            else "adjustmentOpen"
        )
        high_col = (
            "AdjH"
            if "AdjH" in df_raw.columns
            else "AdjustmentHigh"
            if "AdjustmentHigh" in df_raw.columns
            else "adjustmentHigh"
        )
        low_col = (
            "AdjL"
            if "AdjL" in df_raw.columns
            else "AdjustmentLow"
            if "AdjustmentLow" in df_raw.columns
            else "adjustmentLow"
        )
        close_col = (
            "AdjC"
            if "AdjC" in df_raw.columns
            else "AdjustmentClose"
            if "AdjustmentClose" in df_raw.columns
            else "adjustmentClose"
        )
        vol_col = (
            "AdjVo"
            if "AdjVo" in df_raw.columns
            else "AdjustmentVolume"
            if "AdjustmentVolume" in df_raw.columns
            else "adjustmentVolume"
            if "adjustmentVolume" in df_raw.columns
            else "Vo"
            if "Vo" in df_raw.columns
            else "Volume"
            if "Volume" in df_raw.columns
            else "volume"
        )
    else:
        # 調整後がなければ素の Open/High/Low/Close/Volume を使う
        open_col = "O" if "O" in df_raw.columns else "Open" if "Open" in df_raw.columns else "open"
        high_col = "H" if "H" in df_raw.columns else "High" if "High" in df_raw.columns else "high"
        low_col = "L" if "L" in df_raw.columns else "Low" if "Low" in df_raw.columns else "low"
        close_col = "C" if "C" in df_raw.columns else "Close" if "Close" in df_raw.columns else "close"
        vol_col = (
            "Vo"
            if "Vo" in df_raw.columns
            else "Volume"
            if "Volume" in df_raw.columns
            else "volume"
        )

    open_series = df_raw[open_col]
    high_series = df_raw[high_col]
    low_series = df_raw[low_col]
    close_series = df_raw[close_col]
    volume_series = df_raw[vol_col]

    if not any(col in df_raw.columns for col in adjustment_cols) and adjustment_factor_col:
        adjustment_factor = pd.to_numeric(
            df_raw[adjustment_factor_col], errors="coerce"
        ).fillna(1.0)
        open_series = pd.to_numeric(open_series, errors="coerce") * adjustment_factor
        high_series = pd.to_numeric(high_series, errors="coerce") * adjustment_factor
        low_series = pd.to_numeric(low_series, errors="coerce") * adjustment_factor
        close_series = pd.to_numeric(close_series, errors="coerce") * adjustment_factor
        volume_series = pd.to_numeric(volume_series, errors="coerce") / adjustment_factor

    df = pd.DataFrame(
        {
            "date": df_raw["date"],
            "datetime": pd.to_datetime(df_raw["date"]),
            "code": str(symbol).zfill(4) if symbol is not None else df_raw.get("code"),
            "market": market or df_raw.get("market"),
            "open": open_series,
            "high": high_series,
            "low": low_series,
            "close": close_series,
            "volume": volume_series,
        }
    )

    df = df.dropna(subset=["date", "open", "high", "low", "close", "volume"])
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df["datetime"] = pd.to_datetime(df["datetime"]).dt.strftime("%Y-%m-%dT%H:%M:%S")
    df = df.sort_values("date").reset_index(drop=True)
    return df


def load_price_csv(symbol: str, tail_rows: Optional[int] = None) -> pd.DataFrame:
    """
    シンボルに対応するCSVを読み込んで、
    date, open, high, low, close, volume の6列を持つDataFrameを返す。

    - 既にその形式になっているCSV
    - J-Quants daily_quotes 形式のCSV
    の両方をサポートする。

    tail_rows を指定した場合は、CSVが日付昇順で保存されている前提で
    末尾の行だけを読み込む。
    """
    csv_path = PRICE_CSV_DIR / f"{symbol}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"CSVファイルが見つかりません: {csv_path}。"
            "data/price_csv に銘柄コードのCSVを配置するか、"
            "J-Quants取得機能でデータを保存してください。"
        )
    if tail_rows is None:
        df_raw = pd.read_csv(csv_path)
    elif tail_rows <= 0:
        df_raw = pd.read_csv(csv_path, nrows=0)
    else:
        with csv_path.open("r", encoding="utf-8") as csv_file:
            total_lines = sum(1 for _ in csv_file)
        total_rows = max(total_lines - 1, 0)
        if total_rows <= tail_rows:
            df_raw = pd.read_csv(csv_path)
        else:
            skip_count = total_rows - tail_rows
            df_raw = pd.read_csv(csv_path, skiprows=range(1, skip_count + 1))

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
        return apply_manual_stock_split_adjustments(df, symbol)

    # パターン2: J-Quants daily_quotes 形式
    if "Date" in df_raw.columns:
        df = _normalize_from_jquants(df_raw, symbol=symbol)
        return apply_manual_stock_split_adjustments(df, symbol)

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
        api_key=JQUANTS_API_KEY,
        base_url=JQUANTS_BASE_URL,
    )
    df_raw = client.fetch_daily_quotes(symbol, adjusted_start, adjusted_end)
    df_normalized = _normalize_from_jquants(df_raw, symbol=symbol)
    df_normalized = apply_manual_stock_split_adjustments(df_normalized, symbol)

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
