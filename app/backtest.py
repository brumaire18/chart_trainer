from dataclasses import dataclass
from itertools import product
import random
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from .data_loader import get_available_symbols, load_price_csv, load_topix_csv
from .minervini_screen import MinerviniScreenConfig, compute_minervini_indicators


@dataclass
class PatternSignal:
    symbol: str
    date: pd.Timestamp
    pattern: str
    dataset: str
    breakout_price: float
    breakout_volume_ratio: float
    peak_return: float
    label: str
    indicators: Dict[str, float]
    pattern_details: Dict[str, float]


def split_symbols_randomly(
    symbols: Iterable[str], seed: int = 42, train_ratio: float = 0.5
) -> Tuple[List[str], List[str]]:
    """
    銘柄リストをランダムに並び替え、train/validation に分割する。
    """

    symbol_list = list(symbols)
    rng = random.Random(seed)
    rng.shuffle(symbol_list)

    split_index = int(len(symbol_list) * train_ratio)
    train_symbols = symbol_list[:split_index]
    validation_symbols = symbol_list[split_index:]
    return train_symbols, validation_symbols


def compute_rolling_avg_dollar_volume(
    close: pd.Series,
    volume: pd.Series,
    lookback: int,
) -> pd.Series:
    """close*volume のローリング平均売買代金を計算する共通ヘルパー。"""

    if lookback <= 0:
        raise ValueError(f"lookback must be positive: {lookback}")
    return (close * volume).rolling(lookback, min_periods=lookback).mean()


def build_daily_turnover_universe(
    panel_df: pd.DataFrame,
    top_k: int = 1000,
    min_close: float = 1.0,
    max_close: Optional[float] = None,
    return_bool_mask: bool = False,
) -> Any:
    """日次売買代金ランキングからユニバースを作る。

    除外条件:
    - date/symbol/close/volume の欠損
    - volume <= 0
    - close < min_close
    - max_close 指定時は close > max_close
    """

    required_cols = {"date", "symbol", "close", "volume"}
    if not required_cols.issubset(panel_df.columns):
        missing = sorted(required_cols - set(panel_df.columns))
        raise ValueError(f"panel_df is missing required columns: {missing}")
    if top_k <= 0:
        raise ValueError(f"top_k must be positive: {top_k}")

    df = panel_df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    df["symbol"] = df["symbol"].astype(str)
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce")

    valid_mask = (
        df["date"].notna()
        & df["symbol"].notna()
        & df["close"].notna()
        & df["volume"].notna()
        & (df["volume"] > 0)
        & (df["close"] >= float(min_close))
    )
    if max_close is not None:
        valid_mask &= df["close"] <= float(max_close)

    valid_df = df.loc[valid_mask, ["date", "symbol", "close", "volume"]].copy()
    valid_df["turnover"] = valid_df["close"] * valid_df["volume"]

    ranked = valid_df.sort_values(["date", "turnover", "symbol"], ascending=[True, False, True])
    selected = ranked.groupby("date", sort=True).head(int(top_k)).copy()

    if return_bool_mask:
        selected_key = set(zip(selected["date"].tolist(), selected["symbol"].tolist()))
        return pd.Series(
            [
                (dt, sym) in selected_key
                for dt, sym in zip(df["date"].tolist(), df["symbol"].tolist())
            ],
            index=panel_df.index,
            dtype=bool,
        )

    date_to_symbols: Dict[pd.Timestamp, List[str]] = {}
    for dt, group in selected.groupby("date", sort=True):
        date_to_symbols[dt] = group["symbol"].tolist()
    return date_to_symbols


def _compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """主要なテクニカル指標を計算する。"""

    df_ind = df.copy()
    df_ind["sma20"] = df_ind["close"].rolling(20).mean()
    df_ind["sma50"] = df_ind["close"].rolling(50).mean()
    df_ind["sma200"] = df_ind["close"].rolling(200).mean()

    delta = df_ind["close"].diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    df_ind["rsi14"] = 100 - (100 / (1 + rs))

    ema12 = df_ind["close"].ewm(span=12, adjust=False).mean()
    ema26 = df_ind["close"].ewm(span=26, adjust=False).mean()
    df_ind["macd"] = ema12 - ema26
    df_ind["macd_signal"] = df_ind["macd"].ewm(span=9, adjust=False).mean()
    df_ind["macd_hist"] = df_ind["macd"] - df_ind["macd_signal"]

    df_ind["volume_sma20"] = df_ind["volume"].rolling(20).mean()
    return df_ind


def _compute_new_high_breakout_indicators(
    df: pd.DataFrame,
    high_lookback: int,
    volume_sma_window: int,
    sma_short_window: int,
    sma_long_window: int,
    sma_slope_lookback: int,
    atr_window: int,
) -> pd.DataFrame:
    """新高値ブレイクアウト向けの指標を計算する。"""

    df_ind = df.copy()
    df_ind["highest_high"] = df_ind["high"].shift(1).rolling(high_lookback).max()
    df_ind["volume_sma"] = df_ind["volume"].rolling(volume_sma_window).mean()
    df_ind["sma_short"] = df_ind["close"].rolling(sma_short_window).mean()
    df_ind["sma_long"] = df_ind["close"].rolling(sma_long_window).mean()
    df_ind["sma_short_slope_up"] = df_ind["sma_short"] > df_ind["sma_short"].shift(
        sma_slope_lookback
    )

    prev_close = df_ind["close"].shift(1)
    true_range = pd.concat(
        [
            df_ind["high"] - df_ind["low"],
            (df_ind["high"] - prev_close).abs(),
            (df_ind["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    df_ind["atr14"] = true_range.rolling(atr_window).mean()
    df_ind["low10"] = df_ind["low"].shift(1).rolling(10).min()
    return df_ind


def _analyze_cup_base(
    prices: pd.Series,
    depth_min: float,
    depth_max: float,
    recovery_ratio: float,
) -> Optional[Dict[str, float]]:
    """カップ/ソーサー形状を簡易判定して特徴量を返す。"""

    if len(prices) < 10:
        return None

    left_size = max(int(len(prices) * 0.2), 1)
    right_start = len(prices) - left_size

    left_section = prices.iloc[:left_size]
    middle_section = prices.iloc[left_size:right_start]
    right_section = prices.iloc[right_start:]

    if middle_section.empty or right_section.empty:
        return None

    left_peak = float(left_section.max())
    right_peak = float(right_section.max())
    bottom = float(middle_section.min())

    if left_peak <= 0:
        return None

    depth = (left_peak - bottom) / left_peak
    if not (depth_min <= depth <= depth_max):
        return None

    if right_peak < left_peak * recovery_ratio:
        return None

    return {
        "left_peak": left_peak,
        "right_peak": right_peak,
        "bottom": bottom,
        "depth": depth,
    }


def _analyze_handle(
    handle_prices: pd.Series,
    right_peak: float,
    handle_max_depth: float,
) -> bool:
    """ハンドル部分の浅さを確認する。"""

    if handle_prices.empty or right_peak <= 0:
        return False

    handle_low = float(handle_prices.min())
    handle_high = float(handle_prices.max())
    handle_depth = (right_peak - handle_low) / right_peak

    if handle_depth > handle_max_depth:
        return False

    if handle_high > right_peak * 1.02:
        return False

    return True


def _detect_pattern(
    df: pd.DataFrame,
    idx: int,
    cup_window: int,
    handle_window: int,
    depth_range: Tuple[float, float],
    recovery_ratio: float,
    handle_max_depth: float,
) -> Optional[Dict[str, float]]:
    """指定した窗口でカップ/ソーサーとハンドルの形状を確認する。"""

    start = idx - (cup_window + handle_window)
    if start < 0:
        return None

    cup_prices = df["close"].iloc[start : idx - handle_window]
    handle_prices = df["close"].iloc[idx - handle_window : idx]

    base_info = _analyze_cup_base(
        cup_prices,
        depth_min=depth_range[0],
        depth_max=depth_range[1],
        recovery_ratio=recovery_ratio,
    )
    if base_info is None:
        return None

    if not _analyze_handle(handle_prices, base_info["right_peak"], handle_max_depth):
        return None

    handle_low = float(handle_prices.min())
    handle_high = float(handle_prices.max())
    handle_depth = (base_info["right_peak"] - handle_low) / base_info["right_peak"]
    base_info["handle_low"] = handle_low
    base_info["handle_high"] = handle_high
    base_info["handle_depth"] = handle_depth

    return base_info


def _detect_selling_climax_candidates(
    df: pd.DataFrame,
    volume_lookback: int,
    volume_multiplier: float,
    drop_pct: float,
    close_position: float,
    atr_lookback: Optional[int] = None,
    drop_atr_mult: Optional[float] = None,
    drop_condition_mode: str = "drop_pct_only",
    trend_ma_len: Optional[int] = None,
    trend_mode: str = "none",
    min_avg_dollar_volume: Optional[float] = None,
    min_avg_volume: Optional[float] = None,
    liquidity_lookback: int = 20,
    vol_percentile_threshold: Optional[float] = None,
    vol_lookback2: Optional[int] = None,
    max_gap_pct: Optional[float] = None,
    index_filter_mask: Optional[pd.Series] = None,
) -> pd.Series:
    feature_cache = _build_selling_climax_feature_cache(
        df,
        needed_lookbacks={volume_lookback, liquidity_lookback},
        needed_atr_lookbacks={atr_lookback} if atr_lookback is not None else set(),
        needed_vol_lookbacks={vol_lookback2} if vol_lookback2 is not None else set(),
        needed_trend_ma_lens={trend_ma_len} if trend_ma_len is not None else set(),
    )
    return _detect_selling_climax_candidates_with_cache(
        df,
        feature_cache=feature_cache,
        volume_lookback=volume_lookback,
        volume_multiplier=volume_multiplier,
        drop_pct=drop_pct,
        close_position=close_position,
        atr_lookback=atr_lookback,
        drop_atr_mult=drop_atr_mult,
        drop_condition_mode=drop_condition_mode,
        trend_ma_len=trend_ma_len,
        trend_mode=trend_mode,
        min_avg_dollar_volume=min_avg_dollar_volume,
        min_avg_volume=min_avg_volume,
        liquidity_lookback=liquidity_lookback,
        vol_percentile_threshold=vol_percentile_threshold,
        vol_lookback2=vol_lookback2,
        max_gap_pct=max_gap_pct,
        index_filter_mask=index_filter_mask,
    )


def _build_selling_climax_feature_cache(
    df: pd.DataFrame,
    needed_lookbacks: Set[int],
    needed_atr_lookbacks: Set[int],
    needed_vol_lookbacks: Set[int],
    needed_trend_ma_lens: Set[int],
) -> Dict[str, Any]:
    if df.empty:
        return {}

    sorted_lookbacks = sorted({lb for lb in needed_lookbacks if lb is not None and lb > 0})
    sorted_atr_lookbacks = sorted(
        {lb for lb in needed_atr_lookbacks if lb is not None and lb > 0}
    )
    sorted_vol_lookbacks = sorted(
        {lb for lb in needed_vol_lookbacks if lb is not None and lb > 1}
    )
    sorted_trend_lens = sorted(
        {lb for lb in needed_trend_ma_lens if lb is not None and lb > 1}
    )

    prev_close = df["close"].shift(1)
    candle_range = df["high"] - df["low"]
    close_pos = (df["close"] - df["low"]) / candle_range.replace(0, np.nan)

    true_range = pd.concat(
        [
            candle_range,
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    volume_avg = {
        lookback: df["volume"].rolling(lookback, min_periods=lookback).mean()
        for lookback in sorted_lookbacks
    }
    trend_ma = {
        lookback: df["close"].rolling(lookback, min_periods=lookback).mean()
        for lookback in sorted_trend_lens
    }
    trend_ma_slope_up = {
        lookback: ma_series > ma_series.shift(1)
        for lookback, ma_series in trend_ma.items()
    }

    avg_volume = {
        lookback: df["volume"].rolling(lookback, min_periods=lookback).mean()
        for lookback in sorted_lookbacks
    }
    avg_dollar_volume = {
        lookback: compute_rolling_avg_dollar_volume(df["close"], df["volume"], lookback)
        for lookback in sorted_lookbacks
    }

    volume_quantile = {
        lookback: df["volume"].shift(1).rolling(lookback, min_periods=lookback)
        for lookback in sorted_vol_lookbacks
    }
    atr = {
        lookback: true_range.rolling(lookback, min_periods=lookback).mean()
        for lookback in sorted_atr_lookbacks
    }

    return {
        "prev_close": prev_close,
        "candle_range": candle_range,
        "close_pos": close_pos,
        "volume_avg": volume_avg,
        "trend_ma": trend_ma,
        "trend_ma_slope_up": trend_ma_slope_up,
        "avg_volume": avg_volume,
        "avg_dollar_volume": avg_dollar_volume,
        "volume_quantile": volume_quantile,
        "atr": atr,
    }


def _detect_selling_climax_candidates_with_cache(
    df: pd.DataFrame,
    feature_cache: Dict[str, Any],
    volume_lookback: int,
    volume_multiplier: float,
    drop_pct: float,
    close_position: float,
    atr_lookback: Optional[int] = None,
    drop_atr_mult: Optional[float] = None,
    drop_condition_mode: str = "drop_pct_only",
    trend_ma_len: Optional[int] = None,
    trend_mode: str = "none",
    min_avg_dollar_volume: Optional[float] = None,
    min_avg_volume: Optional[float] = None,
    liquidity_lookback: int = 20,
    vol_percentile_threshold: Optional[float] = None,
    vol_lookback2: Optional[int] = None,
    max_gap_pct: Optional[float] = None,
    index_filter_mask: Optional[pd.Series] = None,
) -> pd.Series:
    if df.empty:
        return pd.Series(dtype=bool)

    prev_close = feature_cache["prev_close"]
    candle_range = feature_cache["candle_range"]
    close_pos = feature_cache["close_pos"]

    volume_avg = feature_cache["volume_avg"][volume_lookback]
    volume_ratio = df["volume"] / volume_avg
    drop_ratio = (df["close"] - prev_close) / prev_close

    drop_pct_condition = drop_ratio <= -abs(drop_pct)

    drop_atr_condition = pd.Series(False, index=df.index)
    if atr_lookback is not None and drop_atr_mult is not None and atr_lookback > 0:
        atr = feature_cache["atr"][atr_lookback]
        drop_atr_condition = (-drop_ratio * prev_close) >= (atr * float(drop_atr_mult))

    if drop_condition_mode == "drop_atr_only":
        drop_condition = drop_atr_condition
    elif drop_condition_mode == "both":
        drop_condition = drop_pct_condition & drop_atr_condition
    else:
        drop_condition = drop_pct_condition

    trend_condition = pd.Series(True, index=df.index)
    if trend_ma_len is not None and trend_ma_len > 1 and trend_mode != "none":
        trend_ma = feature_cache["trend_ma"][trend_ma_len]
        ma_slope_up = feature_cache["trend_ma_slope_up"][trend_ma_len]
        close_above_ma = df["close"] >= trend_ma
        if trend_mode == "reversion_only_in_uptrend":
            trend_condition = close_above_ma & ma_slope_up
        elif trend_mode == "exclude_downtrend":
            trend_condition = ma_slope_up
        elif trend_mode == "ma_slope_positive":
            trend_condition = ma_slope_up

    liquidity_condition = pd.Series(True, index=df.index)
    if min_avg_volume is not None:
        avg_volume = feature_cache["avg_volume"][liquidity_lookback]
        liquidity_condition &= avg_volume >= float(min_avg_volume)
    if min_avg_dollar_volume is not None:
        avg_dollar_volume = feature_cache["avg_dollar_volume"][liquidity_lookback]
        liquidity_condition &= avg_dollar_volume >= float(min_avg_dollar_volume)

    volume_shape_condition = pd.Series(True, index=df.index)
    if vol_percentile_threshold is not None and vol_lookback2 is not None and vol_lookback2 > 1:
        volume_quantile = feature_cache["volume_quantile"][vol_lookback2].quantile(
            float(vol_percentile_threshold) / 100.0
        )
        volume_shape_condition = df["volume"] >= volume_quantile

    gap_condition = pd.Series(True, index=df.index)
    if max_gap_pct is not None:
        gap_ratio = (df["open"] - prev_close) / prev_close
        gap_condition = gap_ratio <= float(max_gap_pct)

    candidates = (
        (df["close"] <= df["open"])
        & (volume_ratio >= volume_multiplier)
        & drop_condition
        & (candle_range > 0)
        & (close_pos <= close_position)
        & trend_condition
        & liquidity_condition
        & volume_shape_condition
        & gap_condition
    )
    if index_filter_mask is not None:
        candidates &= index_filter_mask.reindex(df.index).fillna(False)
    return candidates.fillna(False)


def _label_selling_climax_success(
    df: pd.DataFrame,
    confirm_k: int,
    time_stop_bars: Optional[int] = None,
    stop_atr_mult: Optional[float] = None,
    trailing_atr_mult: Optional[float] = None,
    atr_lookback: int = 14,
) -> pd.Series:
    if df.empty or confirm_k <= 0:
        return pd.Series(dtype=bool)
    prev_close = df["close"].shift(1)
    true_range = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = true_range.rolling(atr_lookback, min_periods=atr_lookback).mean()

    max_hold = int(time_stop_bars) if time_stop_bars is not None and time_stop_bars > 0 else confirm_k
    max_hold = max(max_hold, confirm_k)

    success = pd.Series(False, index=df.index)
    for i in range(len(df)):
        if i + 1 >= len(df):
            continue
        entry_price = float(df.iloc[i]["close"])
        trigger_high = float(df.iloc[i]["high"])
        atr_value = atr.iloc[i]
        stop_price = -np.inf
        if stop_atr_mult is not None and pd.notna(atr_value):
            stop_price = entry_price - float(stop_atr_mult) * float(atr_value)

        highest_close = entry_price
        is_success = False
        is_failed = False

        for offset in range(1, max_hold + 1):
            j = i + offset
            if j >= len(df):
                break
            row = df.iloc[j]

            highest_close = max(highest_close, float(row["close"]))
            if trailing_atr_mult is not None and pd.notna(atr_value):
                trail_stop = highest_close - float(trailing_atr_mult) * float(atr_value)
                stop_price = max(stop_price, trail_stop)

            if stop_price != -np.inf and float(row["low"]) <= stop_price:
                is_failed = True
                break

            if offset <= confirm_k and float(row["close"]) > trigger_high:
                is_success = True
                break

            if time_stop_bars is not None and offset >= int(time_stop_bars):
                break

        success.iloc[i] = is_success and (not is_failed)

    return success.fillna(False)


def scan_canslim_patterns(
    df: pd.DataFrame,
    lookahead: int = 20,
    return_threshold: float = 0.03,
    volume_multiplier: float = 1.5,
    cup_window: int = 50,
    saucer_window: int = 80,
    handle_window: int = 10,
    cup_depth_range: Tuple[float, float] = (0.15, 0.4),
    cup_recovery_ratio: float = 0.85,
    cup_handle_max_depth: float = 0.15,
    saucer_depth_range: Tuple[float, float] = (0.05, 0.18),
    saucer_recovery_ratio: float = 0.9,
    saucer_handle_max_depth: float = 0.1,
) -> List[PatternSignal]:
    """
    CAN-SLIM のカップウィズハンドル/ソーサーウィズハンドルを探索し、
    出来高を伴った上昇ブレイクを抽出する。

    peak_return はブレイクアウト後に lookahead 本の範囲で付けた
    最高値ベースの上昇率を返す。
    """

    if df.empty:
        return []

    df_sorted = df.sort_values("date").reset_index(drop=True)
    df_ind = _compute_indicators(df_sorted)

    signals: List[PatternSignal] = []
    min_window = max(cup_window, saucer_window) + handle_window

    for idx in range(min_window, len(df_ind) - lookahead):
        breakout_row = df_ind.iloc[idx]
        volume_sma = breakout_row.get("volume_sma20")
        if pd.isna(volume_sma) or volume_sma <= 0:
            continue

        breakout_volume_ratio = float(breakout_row["volume"] / volume_sma)
        if breakout_volume_ratio < volume_multiplier:
            continue

        cup_info = _detect_pattern(
            df_ind,
            idx,
            cup_window=cup_window,
            handle_window=handle_window,
            depth_range=cup_depth_range,
            recovery_ratio=cup_recovery_ratio,
            handle_max_depth=cup_handle_max_depth,
        )
        saucer_info = _detect_pattern(
            df_ind,
            idx,
            cup_window=saucer_window,
            handle_window=handle_window,
            depth_range=saucer_depth_range,
            recovery_ratio=saucer_recovery_ratio,
            handle_max_depth=saucer_handle_max_depth,
        )

        pattern_info = None
        pattern_name = ""
        if cup_info is not None:
            pattern_info = cup_info
            pattern_name = "cup_with_handle"
        elif saucer_info is not None:
            pattern_info = saucer_info
            pattern_name = "saucer_with_handle"

        if pattern_info is None:
            continue

        if breakout_row["close"] <= pattern_info["right_peak"]:
            continue

        breakout_price = float(breakout_row["close"])
        lookahead_prices = df_ind["close"].iloc[idx + 1 : idx + lookahead + 1]
        if lookahead_prices.empty:
            continue
        peak_price = float(lookahead_prices.max())
        peak_return = (peak_price - breakout_price) / breakout_price
        label = "up" if peak_return >= return_threshold else "down"

        indicators = {
            "sma20": float(breakout_row.get("sma20", np.nan)),
            "sma50": float(breakout_row.get("sma50", np.nan)),
            "sma200": float(breakout_row.get("sma200", np.nan)),
            "rsi14": float(breakout_row.get("rsi14", np.nan)),
            "macd": float(breakout_row.get("macd", np.nan)),
            "macd_hist": float(breakout_row.get("macd_hist", np.nan)),
        }

        pattern_details = {
            "left_peak": float(pattern_info["left_peak"]),
            "right_peak": float(pattern_info["right_peak"]),
            "bottom": float(pattern_info["bottom"]),
            "depth": float(pattern_info["depth"]),
            "handle_low": float(pattern_info["handle_low"]),
            "handle_high": float(pattern_info["handle_high"]),
            "handle_depth": float(pattern_info["handle_depth"]),
        }

        signals.append(
            PatternSignal(
                symbol=str(breakout_row.get("code", "")),
                date=breakout_row["date"],
                pattern=pattern_name,
                dataset="",
                breakout_price=breakout_price,
                breakout_volume_ratio=breakout_volume_ratio,
                peak_return=peak_return,
                label=label,
                indicators=indicators,
                pattern_details=pattern_details,
            )
        )

    return signals


def scan_cup_with_handle_screen(
    df: pd.DataFrame,
    lookback_days: int = 30,
    cup_windows: Optional[List[int]] = None,
    handle_windows: Optional[List[int]] = None,
    depth_range: Tuple[float, float] = (0.12, 0.33),
    recovery_ratio: float = 0.85,
    handle_max_depth: float = 0.15,
    min_price_gain: float = 0.3,
    rs_lookback: int = 20,
    rs_min_change: float = 0.0,
    breakout_volume_multiplier: float = 1.5,
    handle_dry_volume_ratio: float = 0.8,
) -> List[Dict[str, float]]:
    """
    取っ手付きカップのスクリーニング条件を満たすブレイクアウトを探索する。

    Returns:
        条件を満たしたブレイクアウトの情報を辞書で返す。
    """

    if df.empty:
        return []

    cup_windows = cup_windows or [50]
    handle_windows = handle_windows or [10]

    df_sorted = df.sort_values("date").reset_index(drop=True)
    df_ind = _compute_indicators(df_sorted)

    results: List[Dict[str, float]] = []
    max_window = max(cup_windows) + max(handle_windows)
    start_idx = max(max_window, len(df_ind) - max(lookback_days, 1))

    for idx in range(start_idx, len(df_ind)):
        breakout_row = df_ind.iloc[idx]
        volume_sma = breakout_row.get("volume_sma20")
        if pd.isna(volume_sma) or volume_sma <= 0:
            continue

        breakout_volume_ratio = float(breakout_row["volume"] / volume_sma)
        if breakout_volume_ratio < breakout_volume_multiplier:
            continue

        for cup_window in cup_windows:
            for handle_window in handle_windows:
                min_window = cup_window + handle_window
                if idx - min_window < 0:
                    continue

                pattern_info = _detect_pattern(
                    df_ind,
                    idx,
                    cup_window=cup_window,
                    handle_window=handle_window,
                    depth_range=depth_range,
                    recovery_ratio=recovery_ratio,
                    handle_max_depth=handle_max_depth,
                )
                if pattern_info is None:
                    continue

                if breakout_row["close"] <= pattern_info["right_peak"]:
                    continue

                price_gain = (breakout_row["close"] - pattern_info["bottom"]) / pattern_info["bottom"]
                if price_gain < min_price_gain:
                    continue

                handle_slice = df_ind.iloc[idx - handle_window : idx]
                if handle_slice.empty:
                    continue

                handle_volume_avg = float(handle_slice["volume"].mean())
                handle_volume_sma = float(handle_slice["volume_sma20"].mean())
                if handle_volume_sma <= 0 or handle_volume_avg > handle_volume_sma * handle_dry_volume_ratio:
                    continue

                handle_low = float(handle_slice["close"].min())
                handle_sma_max = float(handle_slice["sma50"].max())
                if pd.isna(handle_sma_max) or handle_low < handle_sma_max:
                    continue

                if "topix_rs" not in df_ind.columns or idx - rs_lookback < 0:
                    continue
                rs_now = df_ind.iloc[idx]["topix_rs"]
                rs_prev = df_ind.iloc[idx - rs_lookback]["topix_rs"]
                if pd.isna(rs_now) or pd.isna(rs_prev) or rs_prev == 0:
                    continue
                rs_change = (rs_now / rs_prev - 1) * 100
                if rs_change < rs_min_change:
                    continue

                results.append(
                    {
                        "date": breakout_row["date"],
                        "cup_window": float(cup_window),
                        "handle_window": float(handle_window),
                        "breakout_volume_ratio": breakout_volume_ratio,
                        "handle_volume_ratio": handle_volume_avg / handle_volume_sma,
                        "price_gain": float(price_gain),
                        "rs_change": float(rs_change),
                        "handle_low": handle_low,
                        "sma50": handle_sma_max,
                    }
                )

    return results


def run_canslim_backtest(
    symbols: Optional[Iterable[str]] = None,
    symbol_price_data: Optional[Dict[str, pd.DataFrame]] = None,
    seed: int = 42,
    train_ratio: float = 0.5,
    lookahead: int = 20,
    return_threshold: float = 0.03,
    volume_multiplier: float = 1.5,
    cup_window: int = 50,
    saucer_window: int = 80,
    handle_window: int = 10,
    cup_depth_range: Tuple[float, float] = (0.15, 0.4),
    cup_recovery_ratio: float = 0.85,
    cup_handle_max_depth: float = 0.15,
    saucer_depth_range: Tuple[float, float] = (0.05, 0.18),
    saucer_recovery_ratio: float = 0.9,
    saucer_handle_max_depth: float = 0.1,
    progress_callback: Optional[Callable[[int, int, Optional[str]], None]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    銘柄をランダムに半分ずつに分割し、CAN-SLIM に基づいた
    カップ/ソーサーウィズハンドルのバックテストを実施する。
    """

    raw_symbols = list(symbols) if symbols is not None else get_available_symbols()
    target_symbols: List[str] = []
    symbol_price_data: Dict[str, pd.DataFrame] = {}
    for symbol in raw_symbols:
        try:
            df_price = load_price_csv(symbol)
        except Exception:
            continue
        if df_price.empty:
            continue
        symbol_price_data[symbol] = df_price.sort_values("date").reset_index(drop=True)
        target_symbols.append(symbol)

    if not target_symbols:
        return pd.DataFrame(), pd.DataFrame()

    train_symbols, validation_symbols = split_symbols_randomly(
        target_symbols, seed=seed, train_ratio=train_ratio
    )
    train_set = set(train_symbols)

    all_signals: List[PatternSignal] = []

    total_symbols = len(target_symbols)
    for idx, symbol in enumerate(target_symbols, start=1):
        try:
            if symbol_price_data is not None and symbol in symbol_price_data:
                df_price = symbol_price_data[symbol]
            else:
                df_price = load_price_csv(symbol)
            symbol_signals = scan_canslim_patterns(
                df_price,
                lookahead=lookahead,
                return_threshold=return_threshold,
                volume_multiplier=volume_multiplier,
                cup_window=cup_window,
                saucer_window=saucer_window,
                handle_window=handle_window,
                cup_depth_range=cup_depth_range,
                cup_recovery_ratio=cup_recovery_ratio,
                cup_handle_max_depth=cup_handle_max_depth,
                saucer_depth_range=saucer_depth_range,
                saucer_recovery_ratio=saucer_recovery_ratio,
                saucer_handle_max_depth=saucer_handle_max_depth,
            )
            dataset = "train" if symbol in train_set else "validation"
            for signal in symbol_signals:
                signal.dataset = dataset
                signal.symbol = symbol
            all_signals.extend(symbol_signals)
        except Exception:
            continue
        finally:
            if progress_callback:
                progress_callback(idx, total_symbols, "銘柄スキャン")

    if not all_signals:
        return pd.DataFrame(), pd.DataFrame()

    records = []
    for signal in all_signals:
        record = {
            "symbol": signal.symbol,
            "date": signal.date,
            "pattern": signal.pattern,
            "dataset": signal.dataset,
            "breakout_price": signal.breakout_price,
            "breakout_volume_ratio": signal.breakout_volume_ratio,
            "peak_return": signal.peak_return,
            "label": signal.label,
        }
        record.update(signal.indicators)
        record.update({f"pattern_{k}": v for k, v in signal.pattern_details.items()})
        records.append(record)

    results_df = pd.DataFrame(records)
    summary_df = (
        results_df.groupby(["dataset", "pattern", "label"], as_index=False)
        .agg(count=("symbol", "count"), avg_return=("peak_return", "mean"))
        .sort_values(["dataset", "pattern", "label"])
        .reset_index(drop=True)
    )

    return results_df, summary_df


def _build_minervini_panel(
    symbols: Optional[Iterable[str]],
    lookahead: int,
    slope_lookback_days: int,
    progress_callback: Optional[Callable[[int, int, Optional[str]], None]] = None,
) -> pd.DataFrame:
    target_symbols = list(symbols) if symbols is not None else get_available_symbols()
    if not target_symbols:
        return pd.DataFrame()

    indicator_frames: List[pd.DataFrame] = []
    total_symbols = len(target_symbols)

    for idx, symbol in enumerate(target_symbols, start=1):
        try:
            price_df = load_price_csv(symbol)
        except Exception:
            if progress_callback:
                progress_callback(idx, total_symbols, "銘柄スキャン")
            continue

        if price_df.empty:
            if progress_callback:
                progress_callback(idx, total_symbols, "銘柄スキャン")
            continue

        indicator_df = compute_minervini_indicators(
            price_df, slope_lookback_days=slope_lookback_days
        )
        if indicator_df.empty:
            if progress_callback:
                progress_callback(idx, total_symbols, "銘柄スキャン")
            continue

        indicator_df = indicator_df.copy()
        indicator_df["symbol"] = str(symbol).zfill(4)

        if lookahead > 0:
            future_max = (
                indicator_df["close"]
                .shift(-1)
                .rolling(lookahead)
                .max()
                .shift(-(lookahead - 1))
            )
        else:
            future_max = pd.Series(index=indicator_df.index, dtype=float)
        indicator_df["future_max_close"] = future_max

        indicator_frames.append(indicator_df)
        if progress_callback:
            progress_callback(idx, total_symbols, "銘柄スキャン")

    if not indicator_frames:
        return pd.DataFrame()

    combined = pd.concat(indicator_frames, ignore_index=True)
    combined["rs_rating"] = (
        combined.groupby("date")["return_52w"].rank(pct=True) * 100
    )
    return combined


def _apply_minervini_conditions(
    combined: pd.DataFrame, config: MinerviniScreenConfig
) -> pd.DataFrame:
    enriched = combined.copy()
    enriched["price_above_150_200"] = (enriched["close"] > enriched["sma150"]) & (
        enriched["close"] > enriched["sma200"]
    )
    enriched["ma150_above_200"] = enriched["sma150"] > enriched["sma200"]
    enriched["sma200_rising"] = enriched["sma200_slope"] > 0
    enriched["ma50_above_150_200"] = (enriched["sma50"] > enriched["sma150"]) & (
        enriched["sma50"] > enriched["sma200"]
    )
    enriched["price_above_50"] = enriched["close"] > enriched["sma50"]

    low_threshold = enriched["low_52w"] * (1 + config.low_from_low_pct)
    if config.low_from_low_pct >= 0:
        enriched["low_condition"] = enriched["close"] >= low_threshold
    else:
        enriched["low_condition"] = enriched["close"] <= low_threshold

    enriched["high_condition"] = enriched["close"] >= enriched["high_52w"] * (
        1 - config.high_from_high_pct
    )
    enriched["rs_condition"] = enriched["rs_rating"] >= config.rs_threshold

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
    enriched["passes_trend_template"] = enriched[condition_cols].all(axis=1)
    return enriched


def _evaluate_minervini_config(
    combined: pd.DataFrame,
    config: MinerviniScreenConfig,
    return_threshold: float,
) -> Optional[Dict[str, float]]:
    enriched = _apply_minervini_conditions(combined, config)
    signals = enriched[
        enriched["passes_trend_template"]
        & enriched["future_max_close"].notna()
        & enriched["close"].notna()
        & (enriched["close"] > 0)
    ].copy()
    if signals.empty:
        return None

    signals["peak_return"] = (signals["future_max_close"] - signals["close"]) / signals[
        "close"
    ]
    win_mask = signals["peak_return"] >= return_threshold
    signal_count = float(len(signals))
    win_rate = float(win_mask.mean()) if signal_count else 0.0
    return {
        "signal_count": signal_count,
        "win_rate": win_rate,
        "avg_return": float(signals["peak_return"].mean()),
        "median_return": float(signals["peak_return"].median()),
    }


def run_minervini_backtest(
    symbols: Optional[Iterable[str]] = None,
    lookahead: int = 20,
    return_threshold: float = 0.05,
    config: Optional[MinerviniScreenConfig] = None,
    progress_callback: Optional[Callable[[int, int, Optional[str]], None]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    ミネルヴィニ・トレンドテンプレートの条件に合致した日をシグナルとして扱い、
    その後のピークリターンで評価するバックテストを行う。
    """

    config = config or MinerviniScreenConfig()
    combined = _build_minervini_panel(
        symbols=symbols,
        lookahead=lookahead,
        slope_lookback_days=config.slope_lookback_days,
        progress_callback=progress_callback,
    )
    if combined.empty:
        return pd.DataFrame(), pd.DataFrame()

    combined = _apply_minervini_conditions(combined, config)
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

    signals = combined[
        combined["passes_trend_template"]
        & combined["future_max_close"].notna()
        & combined["close"].notna()
        & (combined["close"] > 0)
    ].copy()
    if signals.empty:
        return pd.DataFrame(), pd.DataFrame()

    signals["peak_return"] = (signals["future_max_close"] - signals["close"]) / signals[
        "close"
    ]
    signals["label"] = np.where(
        signals["peak_return"] >= return_threshold, "up", "down"
    )

    results_cols = [
        "symbol",
        "date",
        "close",
        "rs_rating",
        "return_52w",
        "peak_return",
        "label",
    ] + condition_cols
    results_df = signals[results_cols].sort_values(["date", "symbol"]).reset_index(
        drop=True
    )

    summary_df = (
        results_df.groupby("label", as_index=False)
        .agg(count=("symbol", "count"), avg_return=("peak_return", "mean"))
        .sort_values("label")
        .reset_index(drop=True)
    )
    return results_df, summary_df


def run_minervini_grid_search(
    symbols: Optional[Iterable[str]] = None,
    lookahead: int = 20,
    return_threshold: float = 0.05,
    rs_thresholds: Optional[Iterable[float]] = None,
    low_from_low_pcts: Optional[Iterable[float]] = None,
    high_from_high_pcts: Optional[Iterable[float]] = None,
    slope_lookback_days_list: Optional[Iterable[int]] = None,
    min_signals: int = 10,
    progress_callback: Optional[Callable[[int, int, Optional[str]], None]] = None,
) -> pd.DataFrame:
    rs_thresholds = list(rs_thresholds) if rs_thresholds is not None else [70.0]
    low_from_low_pcts = (
        list(low_from_low_pcts) if low_from_low_pcts is not None else [-0.3]
    )
    high_from_high_pcts = (
        list(high_from_high_pcts) if high_from_high_pcts is not None else [0.25]
    )
    slope_lookback_days_list = (
        list(slope_lookback_days_list)
        if slope_lookback_days_list is not None
        else [20]
    )

    results: List[Dict[str, float]] = []
    total_configs = (
        len(rs_thresholds)
        * len(low_from_low_pcts)
        * len(high_from_high_pcts)
        * len(slope_lookback_days_list)
    )
    progress_count = 0

    for slope_lookback_days in slope_lookback_days_list:
        combined = _build_minervini_panel(
            symbols=symbols,
            lookahead=lookahead,
            slope_lookback_days=slope_lookback_days,
            progress_callback=None,
        )
        if combined.empty:
            continue

        for rs_threshold in rs_thresholds:
            for low_from_low_pct in low_from_low_pcts:
                for high_from_high_pct in high_from_high_pcts:
                    progress_count += 1
                    if progress_callback:
                        progress_callback(progress_count, total_configs, "条件評価")
                    config = MinerviniScreenConfig(
                        rs_threshold=float(rs_threshold),
                        low_from_low_pct=float(low_from_low_pct),
                        high_from_high_pct=float(high_from_high_pct),
                        slope_lookback_days=int(slope_lookback_days),
                    )
                    stats = _evaluate_minervini_config(
                        combined, config, return_threshold
                    )
                    if stats is None:
                        continue
                    if stats["signal_count"] < min_signals:
                        continue
                    results.append(
                        {
                            "rs_threshold": float(rs_threshold),
                            "low_from_low_pct": float(low_from_low_pct),
                            "high_from_high_pct": float(high_from_high_pct),
                            "slope_lookback_days": int(slope_lookback_days),
                            **stats,
                        }
                    )

    if not results:
        return pd.DataFrame()

    result_df = pd.DataFrame(results)
    return result_df.sort_values(
        ["win_rate", "avg_return", "signal_count"], ascending=[False, False, False]
    ).reset_index(drop=True)


def run_new_high_breakout_backtest(
    symbols: Optional[Iterable[str]] = None,
    high_lookback: int = 20,
    volume_sma_window: int = 20,
    sma_short_window: int = 25,
    sma_long_window: int = 75,
    sma_slope_lookback: int = 5,
    atr_window: int = 14,
    stop_atr_multiplier: float = 2.0,
    breakeven_r: float = 1.0,
    partial_profit_r: float = 1.5,
    partial_profit_ratio: float = 0.5,
    trailing_type: str = "atr",
    atr_trail_multiplier: float = 2.0,
    time_stop_days: int = 10,
    time_stop_r: float = 0.8,
    risk_per_trade: float = 0.01,
    account_equity: float = 1_000_000.0,
    setup_condition: Optional[Callable[[pd.Series], bool]] = None,
    progress_callback: Optional[Callable[[int, int, Optional[str]], None]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    新高値ブレイクアウト戦略のバックテストを実施する。

    setup_condition は前日引けで評価されるユーザー定義の条件。
    """

    raw_symbols = list(symbols) if symbols is not None else get_available_symbols()
    target_symbols: List[str] = []
    symbol_price_data: Dict[str, pd.DataFrame] = {}
    for symbol in raw_symbols:
        try:
            df_price = load_price_csv(symbol)
        except Exception:
            continue
        if df_price.empty:
            continue
        symbol_price_data[symbol] = df_price.sort_values("date").reset_index(drop=True)
        target_symbols.append(symbol)

    if not target_symbols:
        return pd.DataFrame(), pd.DataFrame()

    if setup_condition is None:

        def setup_condition(row: pd.Series) -> bool:
            return bool(
                pd.notna(row["highest_high"])
                and pd.notna(row["volume_sma"])
                and pd.notna(row["sma_short"])
                and pd.notna(row["sma_long"])
                and row["close"] >= row["highest_high"]
                and row["volume"] >= row["volume_sma"]
                and row["sma_short"] >= row["sma_long"]
                and row["sma_short_slope_up"]
            )

    trades: List[Dict[str, object]] = []
    total_symbols = len(target_symbols)

    for idx, symbol in enumerate(target_symbols, start=1):
        try:
            df_price = load_price_csv(symbol)
        except Exception:
            if progress_callback:
                progress_callback(idx, total_symbols, "銘柄スキャン")
            continue

        if df_price.empty:
            if progress_callback:
                progress_callback(idx, total_symbols, "銘柄スキャン")
            continue

        df_sorted = df_price.sort_values("date").reset_index(drop=True)
        df_ind = _compute_new_high_breakout_indicators(
            df_sorted,
            high_lookback=high_lookback,
            volume_sma_window=volume_sma_window,
            sma_short_window=sma_short_window,
            sma_long_window=sma_long_window,
            sma_slope_lookback=sma_slope_lookback,
            atr_window=atr_window,
        )

        pending_entry: Optional[Dict[str, object]] = None
        in_position = False
        entry_price = 0.0
        entry_index = 0
        entry_date = pd.Timestamp.min
        stop_price = 0.0
        risk_per_share = 0.0
        qty = 0
        remaining_qty = 0
        partial_taken = False
        realized_proceeds = 0.0
        exit_reason = ""

        for i in range(1, len(df_ind)):
            row = df_ind.iloc[i]

            if pending_entry and pending_entry["index"] == i and not in_position:
                stop_price_candidate = float(pending_entry["stop_price"])
                if row["high"] >= stop_price_candidate:
                    if pd.isna(row["atr14"]):
                        pending_entry = None
                    else:
                        entry_price = stop_price_candidate
                        entry_index = i
                        entry_date = row["date"]
                        stop_price = entry_price - stop_atr_multiplier * float(row["atr14"])
                        risk_per_share = entry_price - stop_price
                        risk_jpy = account_equity * risk_per_trade
                        qty = int(np.floor(risk_jpy / risk_per_share)) if risk_per_share > 0 else 0
                        remaining_qty = qty
                        partial_taken = False
                        realized_proceeds = 0.0
                        exit_reason = ""
                        if qty > 0:
                            in_position = True
                        pending_entry = None
                else:
                    pending_entry = None

            if in_position:
                days_in_trade = i - entry_index + 1
                if row["low"] <= stop_price:
                    realized_proceeds += stop_price * remaining_qty
                    exit_reason = "stop"
                else:
                    if not partial_taken and partial_profit_ratio > 0:
                        target_price = entry_price + partial_profit_r * risk_per_share
                        partial_qty = int(np.floor(qty * partial_profit_ratio))
                        if partial_qty > 0 and row["high"] >= target_price:
                            realized_proceeds += target_price * partial_qty
                            remaining_qty -= partial_qty
                            partial_taken = True
                            if remaining_qty <= 0:
                                exit_reason = "partial_full"
                    if not exit_reason:
                        if (
                            days_in_trade >= time_stop_days
                            and row["close"] < entry_price + time_stop_r * risk_per_share
                        ):
                            realized_proceeds += row["close"] * remaining_qty
                            exit_reason = "time_stop"
                        elif trailing_type == "low10":
                            if pd.notna(row["low10"]) and row["low"] < row["low10"]:
                                realized_proceeds += row["close"] * remaining_qty
                                exit_reason = "low10_break"

                if exit_reason:
                    total_cost = entry_price * qty
                    pnl = realized_proceeds - total_cost
                    r_multiple = pnl / (risk_per_share * qty) if risk_per_share > 0 else 0.0
                    exit_price = realized_proceeds / qty if qty > 0 else 0.0
                    trades.append(
                        {
                            "symbol": symbol,
                            "entry_date": entry_date,
                            "entry_price": entry_price,
                            "exit_date": row["date"],
                            "exit_price": exit_price,
                            "qty": qty,
                            "pnl": pnl,
                            "r_multiple": r_multiple,
                            "holding_days": days_in_trade,
                            "exit_reason": exit_reason,
                        }
                    )
                    in_position = False
                    continue

                if row["high"] >= entry_price + breakeven_r * risk_per_share:
                    stop_price = max(stop_price, entry_price)

                if trailing_type == "atr" and pd.notna(row["atr14"]):
                    atr_stop = row["close"] - atr_trail_multiplier * float(row["atr14"])
                    stop_price = max(stop_price, atr_stop)

            if not in_position and pending_entry is None:
                if setup_condition(row):
                    if i + 1 < len(df_ind) and pd.notna(row["highest_high"]):
                        pending_entry = {
                            "index": i + 1,
                            "stop_price": float(row["highest_high"]),
                        }

        if progress_callback:
            progress_callback(idx, total_symbols, "銘柄スキャン")

    if not trades:
        return pd.DataFrame(), pd.DataFrame()

    trades_df = pd.DataFrame(trades)
    trades_df["win"] = trades_df["pnl"] > 0

    summary_records: List[Dict[str, object]] = []
    for symbol, group in trades_df.groupby("symbol"):
        win_rate = float(group["win"].mean())
        avg_pnl = float(group["pnl"].mean())
        expectancy = float(group["r_multiple"].mean())
        summary_records.append(
            {
                "symbol": symbol,
                "trades": int(group.shape[0]),
                "win_rate": win_rate,
                "avg_pnl": avg_pnl,
                "expectancy": expectancy,
            }
        )

    overall_win_rate = float(trades_df["win"].mean())
    overall_avg_pnl = float(trades_df["pnl"].mean())
    overall_expectancy = float(trades_df["r_multiple"].mean())
    summary_records.append(
        {
            "symbol": "ALL",
            "trades": int(trades_df.shape[0]),
            "win_rate": overall_win_rate,
            "avg_pnl": overall_avg_pnl,
            "expectancy": overall_expectancy,
        }
    )

    summary_df = pd.DataFrame(summary_records)
    return trades_df, summary_df


def _build_bull_regime_mask(
    topix_df: pd.DataFrame,
    ma_window: int = 200,
    slope_lookback: int = 20,
    momentum_lookback: int = 126,
    require_positive_momentum: bool = True,
) -> pd.DataFrame:
    """TOPIX から Bull レジーム（強い上昇相場）判定を作る。"""

    df = topix_df[["date", "close"]].copy()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df = df.sort_values("date").drop_duplicates(subset=["date"]).reset_index(drop=True)

    df["topix_ma"] = df["close"].rolling(ma_window).mean()
    slope = df["topix_ma"] - df["topix_ma"].shift(slope_lookback)
    bull = (df["close"] > df["topix_ma"]) & (slope > 0)
    if require_positive_momentum:
        bull = bull & (df["close"].pct_change(momentum_lookback) > 0)
    df["is_bull"] = bull.fillna(False)
    return df


def run_bull_market_new_high_momentum_backtest(
    symbols: Optional[Iterable[str]] = None,
    topix_df: Optional[pd.DataFrame] = None,
    high_lookback: int = 252,
    hold_days: int = 20,
    event_cooldown_days: int = 20,
    volume_sma_window: int = 20,
    volume_multiplier: float = 1.0,
    liquidity_lookback: int = 20,
    top_liquidity_count: int = 100,
    one_way_cost_bps: float = 15.0,
    rebalance_weekday: Optional[int] = 0,
) -> Dict[str, pd.DataFrame]:
    """強い上昇相場に限定した最高値更新モメンタム戦略を検証する。"""

    if topix_df is None:
        topix_df = load_topix_csv()
    regime_df = _build_bull_regime_mask(topix_df)
    topix_close_by_date = regime_df.set_index("date")["close"]
    if "open" in regime_df.columns:
        topix_open_by_date = regime_df.set_index("date")["open"].astype(float)
        topix_ret = topix_open_by_date.pct_change().shift(-1).rename("benchmark_return")
    else:
        topix_ret = topix_close_by_date.pct_change().rename("benchmark_return")
    bull_by_date = regime_df.set_index("date")["is_bull"]

    raw_symbols = list(symbols) if symbols is not None else get_available_symbols()
    if "topix" in {str(s).lower() for s in raw_symbols}:
        raw_symbols = [s for s in raw_symbols if str(s).lower() != "topix"]

    symbol_frames: Dict[str, pd.DataFrame] = {}
    liquidity_scores: List[Tuple[str, float]] = []
    for symbol in raw_symbols:
        try:
            df = load_price_csv(symbol)
        except Exception:
            continue
        if df.empty:
            continue
        df = df.sort_values("date").reset_index(drop=True)
        if any(col not in df.columns for col in ["date", "open", "close", "volume"]):
            continue
        df["date"] = pd.to_datetime(df["date"]).dt.normalize()
        df["close"] = pd.to_numeric(df["close"], errors="coerce")
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
        rolling_liquidity = compute_rolling_avg_dollar_volume(
            df["close"], df["volume"], liquidity_lookback
        )
        score = float(rolling_liquidity.dropna().iloc[-1]) if rolling_liquidity.notna().any() else np.nan
        if np.isnan(score):
            continue
        liquidity_scores.append((symbol, score))
        symbol_frames[symbol] = df

    if not liquidity_scores:
        empty_df = pd.DataFrame()
        return {
            "signals": empty_df,
            "trades": empty_df,
            "daily_returns": empty_df,
            "event_study": empty_df,
            "summary": empty_df,
        }

    liquidity_scores.sort(key=lambda x: x[1], reverse=True)
    selected_symbols = [s for s, _ in liquidity_scores[: max(top_liquidity_count, 1)]]

    trades: List[Dict[str, Any]] = []
    events: List[Dict[str, Any]] = []
    signal_records: List[Dict[str, Any]] = []

    for symbol in selected_symbols:
        df = symbol_frames[symbol].copy()
        df["highest_close_prev"] = df["close"].shift(1).rolling(high_lookback).max()
        df["volume_sma"] = df["volume"].rolling(volume_sma_window).mean()
        df["is_bull"] = df["date"].map(bull_by_date).fillna(False)

        last_signal_index: Optional[int] = None
        for i in range(len(df)):
            row = df.iloc[i]
            if pd.isna(row["highest_close_prev"]) or pd.isna(row["volume_sma"]):
                continue
            if last_signal_index is not None and i - last_signal_index < event_cooldown_days:
                continue
            if bool(row["is_bull"]) is False:
                continue
            if row["close"] <= row["highest_close_prev"]:
                continue
            if row["volume"] < row["volume_sma"] * volume_multiplier:
                continue

            entry_i = i + 1
            exit_i = i + hold_days + 1
            if exit_i >= len(df):
                continue
            if rebalance_weekday is not None:
                entry_weekday = pd.Timestamp(df.iloc[entry_i]["date"]).weekday()
                if entry_weekday != rebalance_weekday:
                    continue

            entry_open = float(df.iloc[entry_i]["open"])
            exit_open = float(df.iloc[exit_i]["open"])
            if entry_open <= 0:
                continue
            gross_ret = (exit_open - entry_open) / entry_open
            total_cost = 2.0 * one_way_cost_bps / 10000.0
            net_ret = gross_ret - total_cost

            signal_date = pd.Timestamp(row["date"])
            entry_date = pd.Timestamp(df.iloc[entry_i]["date"])
            exit_date = pd.Timestamp(df.iloc[exit_i]["date"])
            signal_records.append(
                {
                    "symbol": symbol,
                    "signal_date": signal_date,
                    "entry_date": entry_date,
                    "exit_date": exit_date,
                    "is_bull": True,
                    "breakout_close": float(row["close"]),
                    "highest_close_prev": float(row["highest_close_prev"]),
                    "volume_ratio": float(row["volume"] / row["volume_sma"]),
                }
            )
            trades.append(
                {
                    "symbol": symbol,
                    "signal_date": signal_date,
                    "entry_date": entry_date,
                    "exit_date": exit_date,
                    "entry_open": entry_open,
                    "exit_open": exit_open,
                    "gross_return": gross_ret,
                    "net_return": net_ret,
                    "holding_days": hold_days,
                }
            )

            for horizon in (20, 60):
                fwd_i = i + horizon
                if fwd_i >= len(df):
                    continue
                start_close = float(df.iloc[i]["close"])
                end_close = float(df.iloc[fwd_i]["close"])
                if start_close <= 0:
                    continue
                stock_ret = (end_close - start_close) / start_close

                topix_start = topix_close_by_date.get(signal_date)
                topix_end = topix_close_by_date.get(pd.Timestamp(df.iloc[fwd_i]["date"]))
                if pd.isna(topix_start) or pd.isna(topix_end) or topix_start == 0:
                    continue
                bench_ret = (float(topix_end) - float(topix_start)) / float(topix_start)
                events.append(
                    {
                        "symbol": symbol,
                        "signal_date": signal_date,
                        "horizon": horizon,
                        "stock_return": stock_ret,
                        "benchmark_return": bench_ret,
                        "excess_return": stock_ret - bench_ret,
                        "regime": "bull",
                    }
                )

            last_signal_index = i

    trades_df = pd.DataFrame(trades)
    signals_df = pd.DataFrame(signal_records)
    event_df = pd.DataFrame(events)

    if trades_df.empty:
        empty_df = pd.DataFrame()
        return {
            "signals": signals_df,
            "trades": trades_df,
            "daily_returns": empty_df,
            "event_study": event_df,
            "summary": empty_df,
        }

    active_by_date: Dict[pd.Timestamp, List[float]] = {}
    turnover_entry: Dict[pd.Timestamp, int] = {}
    turnover_exit: Dict[pd.Timestamp, int] = {}

    for trade in trades:
        symbol = str(trade["symbol"])
        df = symbol_frames[symbol].set_index(pd.to_datetime(symbol_frames[symbol]["date"]).dt.normalize())
        entry_date = pd.Timestamp(trade["entry_date"])
        exit_date = pd.Timestamp(trade["exit_date"])
        if entry_date not in df.index or exit_date not in df.index:
            continue
        entry_idx = df.index.get_loc(entry_date)
        exit_idx = df.index.get_loc(exit_date)
        if isinstance(entry_idx, slice) or isinstance(exit_idx, slice):
            continue
        if exit_idx <= entry_idx:
            continue

        open_series = df["open"].astype(float)
        trade_dates = open_series.index

        # 約定価格（open）と整合させるため、日次損益も open-to-open で計算する。
        # exit_date は open で手仕舞う前提なので、[entry_date, exit_date) までを保有区間にする。
        for di in range(entry_idx, exit_idx):
            dt_now = trade_dates[di]
            now_open = float(open_series.iloc[di])
            next_open = float(open_series.iloc[di + 1])
            if now_open <= 0:
                continue
            active_by_date.setdefault(pd.Timestamp(dt_now), []).append((next_open - now_open) / now_open)

        turnover_entry[entry_date] = turnover_entry.get(entry_date, 0) + 1
        turnover_exit[exit_date] = turnover_exit.get(exit_date, 0) + 1

    topix_dates = sorted(pd.to_datetime(topix_ret.index).normalize().unique())
    dates = topix_dates if len(topix_dates) > 0 else sorted(active_by_date.keys())
    daily_records: List[Dict[str, Any]] = []
    cumulative = 1.0
    for dt in dates:
        rets = active_by_date.get(dt, [])
        strategy_ret = float(np.mean(rets)) if rets else 0.0
        benchmark = float(topix_ret.get(dt, np.nan))
        if np.isnan(benchmark):
            benchmark = 0.0
        excess = strategy_ret - benchmark
        cumulative *= 1.0 + strategy_ret
        daily_records.append(
            {
                "date": dt,
                "strategy_return": strategy_ret,
                "benchmark_return": benchmark,
                "excess_return": excess,
                "equity_curve": cumulative,
                "entries": int(turnover_entry.get(dt, 0)),
                "exits": int(turnover_exit.get(dt, 0)),
            }
        )

    daily_df = pd.DataFrame(daily_records)
    if daily_df.empty:
        summary_df = pd.DataFrame(
            [
                {
                    "selected_symbols": len(selected_symbols),
                    "trades": int(trades_df.shape[0]),
                    "event_count": int(event_df.shape[0]),
                }
            ]
        )
        return {
            "signals": signals_df,
            "trades": trades_df,
            "daily_returns": daily_df,
            "event_study": event_df,
            "summary": summary_df,
        }

    daily_df = daily_df.sort_values("date").reset_index(drop=True)
    equity = daily_df["equity_curve"]
    running_max = equity.cummax()
    drawdown = equity / running_max - 1.0
    max_drawdown = float(drawdown.min()) if not drawdown.empty else 0.0

    ann_factor = 252.0
    mean_excess = float(daily_df["excess_return"].mean())
    std_excess = float(daily_df["excess_return"].std(ddof=1))
    information_ratio = (mean_excess / std_excess * np.sqrt(ann_factor)) if std_excess > 0 else np.nan

    strategy_total = float((1.0 + daily_df["strategy_return"]).prod() - 1.0)
    benchmark_total = float((1.0 + daily_df["benchmark_return"]).prod() - 1.0)
    years = max(float(len(daily_df)) / ann_factor, 1.0 / ann_factor)
    cagr = float((1.0 + strategy_total) ** (1.0 / years) - 1.0)
    calmar = cagr / abs(max_drawdown) if max_drawdown < 0 else np.nan

    up_days = daily_df["benchmark_return"] > 0
    down_days = daily_df["benchmark_return"] < 0
    up_capture = (
        float(daily_df.loc[up_days, "strategy_return"].sum() / daily_df.loc[up_days, "benchmark_return"].sum())
        if up_days.any() and daily_df.loc[up_days, "benchmark_return"].sum() != 0
        else np.nan
    )
    down_capture = (
        float(daily_df.loc[down_days, "strategy_return"].sum() / daily_df.loc[down_days, "benchmark_return"].sum())
        if down_days.any() and daily_df.loc[down_days, "benchmark_return"].sum() != 0
        else np.nan
    )

    avg_trade_events_on_active_days = float((daily_df["entries"] + daily_df["exits"]).replace(0, np.nan).mean())
    avg_daily_trade_events = float((daily_df["entries"] + daily_df["exits"]).sum() / max(len(daily_df), 1))

    event_summary = (
        event_df.groupby("horizon", as_index=False)
        .agg(
            event_count=("excess_return", "count"),
            avg_stock_return=("stock_return", "mean"),
            avg_benchmark_return=("benchmark_return", "mean"),
            avg_excess_return=("excess_return", "mean"),
            median_excess_return=("excess_return", "median"),
        )
        if not event_df.empty
        else pd.DataFrame()
    )

    summary_row = {
        "selected_symbols": int(len(selected_symbols)),
        "trades": int(trades_df.shape[0]),
        "event_count": int(event_df.shape[0]),
        "strategy_total_return": strategy_total,
        "benchmark_total_return": benchmark_total,
        "total_excess_return": strategy_total - benchmark_total,
        "information_ratio": information_ratio,
        "up_capture": up_capture,
        "down_capture": down_capture,
        "max_drawdown": max_drawdown,
        "calmar": calmar,
        "avg_daily_trade_events": avg_daily_trade_events,
        "avg_daily_turnover": avg_daily_trade_events,
        "avg_trade_events_on_active_days": avg_trade_events_on_active_days,
        "avg_daily_entries_exits": avg_trade_events_on_active_days,
        "avg_trade_net_return": float(trades_df["net_return"].mean()),
        "win_rate": float((trades_df["net_return"] > 0).mean()),
    }
    summary_df = pd.DataFrame([summary_row])
    if not event_summary.empty:
        event_summary = event_summary.sort_values("horizon").reset_index(drop=True)

    return {
        "signals": signals_df,
        "trades": trades_df,
        "daily_returns": daily_df,
        "event_study": event_summary,
        "summary": summary_df,
    }


def run_ma5_deviation_mean_reversion_backtest(
    symbols: Optional[Iterable[str]] = None,
    symbol_price_data: Optional[Dict[str, pd.DataFrame]] = None,
    top_n: int = 10,
    entry_timing: str = "next_open",
    exit_timing: str = "next_open",
    weight_mode: str = "equal",
    min_avg_dollar_volume: float = 50_000_000.0,
    liquidity_lookback: int = 20,
    volatility_lookback: int = 20,
    take_profit_pct: float = 0.05,
    stop_loss_pct: float = 0.03,
    mean_revert_exit_threshold: float = 0.005,
    max_holding_days: int = 5,
    commission_bps: float = 10.0,
    slippage_bps: float = 5.0,
    allow_reentry_after_exit_on_same_day: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """5日移動平均乖離の逆張りロングショート戦略を日次で検証する。"""

    valid_timing = {"same_close", "next_open"}
    if entry_timing not in valid_timing:
        raise ValueError(f"entry_timing must be one of {valid_timing}: {entry_timing}")
    if exit_timing not in valid_timing:
        raise ValueError(f"exit_timing must be one of {valid_timing}: {exit_timing}")
    valid_weight_mode = {"equal", "volatility_adjusted"}
    if weight_mode not in valid_weight_mode:
        raise ValueError(f"weight_mode must be one of {valid_weight_mode}: {weight_mode}")
    if max_holding_days <= 0:
        raise ValueError(f"max_holding_days must be > 0: {max_holding_days}")

    raw_symbols = list(symbols) if symbols is not None else get_available_symbols()
    if "topix" in {str(s).lower() for s in raw_symbols}:
        raw_symbols = [s for s in raw_symbols if str(s).lower() != "topix"]

    records: List[pd.DataFrame] = []
    for symbol in raw_symbols:
        try:
            if symbol_price_data is not None and symbol in symbol_price_data:
                df = symbol_price_data[symbol].copy()
            else:
                df = load_price_csv(symbol)
        except Exception:
            continue
        if df.empty:
            continue
        required_cols = {"date", "open", "close", "volume"}
        if not required_cols.issubset(df.columns):
            continue

        df = df.sort_values("date").reset_index(drop=True)
        df["date"] = pd.to_datetime(df["date"]).dt.normalize()
        for col in ["open", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=["date", "open", "close", "volume"])
        if df.empty:
            continue

        df["symbol"] = str(symbol)
        df["ma5"] = df["close"].rolling(5).mean()
        df["deviation"] = df["close"] / df["ma5"] - 1.0
        df["daily_ret"] = df["close"].pct_change()
        df["volatility"] = df["daily_ret"].rolling(volatility_lookback).std(ddof=0)
        df["avg_dollar_volume"] = compute_rolling_avg_dollar_volume(
            df["close"], df["volume"], liquidity_lookback
        )
        df["liquidity_pass"] = df["avg_dollar_volume"] >= min_avg_dollar_volume
        records.append(
            df[
                [
                    "date",
                    "symbol",
                    "open",
                    "close",
                    "deviation",
                    "volatility",
                    "liquidity_pass",
                ]
            ]
        )

    if not records:
        return pd.DataFrame(), pd.DataFrame()

    panel_df = pd.concat(records, ignore_index=True)
    panel_df = panel_df.sort_values(["date", "symbol"]).reset_index(drop=True)

    date_to_cross_section: Dict[pd.Timestamp, pd.DataFrame] = {
        date: group.copy()
        for date, group in panel_df.groupby("date", sort=True)
    }
    all_dates = sorted(date_to_cross_section.keys())
    date_to_index = {d: i for i, d in enumerate(all_dates)}
    symbol_date_to_row = {
        symbol: group.set_index("date").sort_index()
        for symbol, group in panel_df.groupby("symbol", sort=False)
    }
    trading_cost_pct = (commission_bps + slippage_bps) / 10_000.0

    trade_records: List[Dict[str, Any]] = []
    daily_pnl_by_date: Dict[pd.Timestamp, float] = {pd.Timestamp(d): 0.0 for d in all_dates}
    open_positions: Dict[Tuple[str, str], Dict[str, Any]] = {}

    def _pick_exit_reason(raw_ret: float, deviation: float, held_days: int) -> Optional[str]:
        # 優先順位: stop_loss > take_profit > mean_revert > time_stop
        if raw_ret <= -stop_loss_pct:
            return "stop_loss"
        if raw_ret >= take_profit_pct:
            return "take_profit"
        if abs(deviation) <= mean_revert_exit_threshold:
            return "mean_revert"
        if held_days >= max_holding_days:
            return "time_stop"
        return None

    for signal_date in all_dates:
        exited_symbols: set = set()

        # 先に既存ポジションの決済判定を行う（exit-first優先）。
        for key, pos in list(open_positions.items()):
            symbol = pos["symbol"]
            symbol_df = symbol_date_to_row.get(symbol)
            if symbol_df is None or signal_date not in symbol_df.index:
                continue
            row = symbol_df.loc[signal_date]
            current_close = float(row["close"])
            current_deviation = float(row["deviation"])
            if not np.isfinite(current_close) or current_close <= 0 or not np.isfinite(current_deviation):
                continue

            raw_ret = current_close / pos["entry_price"] - 1.0
            if pos["side"] == "short":
                raw_ret *= -1.0

            held_days = date_to_index[signal_date] - date_to_index[pos["entry_date"]] + 1
            exit_reason = _pick_exit_reason(raw_ret=raw_ret, deviation=current_deviation, held_days=held_days)
            if exit_reason is None:
                continue

            exit_date = signal_date
            exit_px = current_close
            if exit_timing == "next_open":
                next_idx = date_to_index[signal_date] + 1
                if next_idx >= len(all_dates):
                    continue
                next_date = all_dates[next_idx]
                if next_date not in symbol_df.index:
                    continue
                next_row = symbol_df.loc[next_date]
                next_open = float(next_row["open"])
                if not np.isfinite(next_open) or next_open <= 0:
                    continue
                exit_date = next_date
                exit_px = next_open
                raw_ret = exit_px / pos["entry_price"] - 1.0
                if pos["side"] == "short":
                    raw_ret *= -1.0

            net_ret = raw_ret - (2.0 * trading_cost_pct)
            weighted_return = net_ret * pos["weight"]
            daily_pnl_by_date[pd.Timestamp(exit_date)] = (
                daily_pnl_by_date.get(pd.Timestamp(exit_date), 0.0) + float(weighted_return)
            )
            exited_symbols.add(symbol)
            trade_records.append(
                {
                    "symbol": symbol,
                    "direction": pos["side"],
                    "signal_date": pd.Timestamp(pos["signal_date"]),
                    "entry_date": pd.Timestamp(pos["entry_date"]),
                    "exit_date": pd.Timestamp(exit_date),
                    "entry_price": float(pos["entry_price"]),
                    "exit_price": float(exit_px),
                    "weight": float(pos["weight"]),
                    "pnl": float(weighted_return),
                    "raw_return": float(raw_ret),
                    "net_return": float(net_ret),
                    "cost_pct": float(2.0 * trading_cost_pct),
                    "deviation": float(pos["signal_deviation"]),
                    "exit_reason": exit_reason,
                    "holding_days": int(held_days),
                }
            )
            del open_positions[key]

        cross = date_to_cross_section[signal_date]
        candidates = cross[
            cross["liquidity_pass"]
            & cross["deviation"].notna()
            & np.isfinite(cross["deviation"])
        ].copy()
        if candidates.empty:
            continue

        count_each_side = min(top_n, candidates.shape[0] // 2)
        if count_each_side <= 0:
            continue

        long_candidates = candidates.nsmallest(count_each_side, "deviation").copy()
        short_candidates = candidates.nlargest(count_each_side, "deviation").copy()
        long_candidates["side"] = "long"
        short_candidates["side"] = "short"
        picks = pd.concat([long_candidates, short_candidates], ignore_index=True)
        picks["abs_deviation"] = picks["deviation"].abs()
        picks = (
            picks.sort_values(["symbol", "abs_deviation"], ascending=[True, False])
            .drop_duplicates(subset=["symbol"], keep="first")
            .drop(columns=["abs_deviation"])
        )

        entry_date = signal_date
        if entry_timing == "next_open":
            next_idx = date_to_index[signal_date] + 1
            if next_idx >= len(all_dates):
                continue
            entry_date = all_dates[next_idx]

        entry_cross = date_to_cross_section.get(entry_date)
        if entry_cross is None:
            continue

        merged = picks.merge(
            entry_cross[["symbol", "open", "close", "volatility"]].rename(
                columns={"open": "entry_open", "close": "entry_close", "volatility": "entry_volatility"}
            ),
            on="symbol",
            how="inner",
        )
        if merged.empty:
            continue

        if entry_timing == "same_close":
            merged["entry_price"] = merged["entry_close"]
        else:
            merged["entry_price"] = merged["entry_open"]

        merged = merged[(merged["entry_price"] > 0)].copy()
        if merged.empty:
            continue

        if not allow_reentry_after_exit_on_same_day:
            merged = merged[~merged["symbol"].isin(exited_symbols)]
        merged = merged[
            ~merged.apply(lambda r: (r["symbol"], r["side"]) in open_positions, axis=1)
        ]
        if merged.empty:
            continue

        if weight_mode == "volatility_adjusted":
            merged["inv_vol"] = 1.0 / merged["entry_volatility"].replace(0, np.nan)
            for side in ["long", "short"]:
                side_mask = merged["side"] == side
                side_sum = merged.loc[side_mask, "inv_vol"].sum()
                if side_sum > 0:
                    merged.loc[side_mask, "weight"] = 0.5 * (merged.loc[side_mask, "inv_vol"] / side_sum)
                else:
                    side_count = int(side_mask.sum())
                    if side_count > 0:
                        merged.loc[side_mask, "weight"] = 0.5 / side_count
        else:
            for side in ["long", "short"]:
                side_mask = merged["side"] == side
                side_count = int(side_mask.sum())
                if side_count > 0:
                    merged.loc[side_mask, "weight"] = 0.5 / side_count

        for _, row in merged.iterrows():
            open_positions[(str(row["symbol"]), str(row["side"]))] = {
                "symbol": str(row["symbol"]),
                "side": str(row["side"]),
                "signal_date": pd.Timestamp(signal_date),
                "entry_date": pd.Timestamp(entry_date),
                "entry_price": float(row["entry_price"]),
                "weight": float(row["weight"]),
                "signal_deviation": float(row["deviation"]),
            }

    # バックテスト期間終了時に未決済ポジションを強制決済する。
    if open_positions:
        for key, pos in list(open_positions.items()):
            symbol = pos["symbol"]
            symbol_df = symbol_date_to_row.get(symbol)
            if symbol_df is None or symbol_df.empty:
                del open_positions[key]
                continue

            last_idx = symbol_df.index.max()
            if pd.isna(last_idx):
                del open_positions[key]
                continue

            last_row = symbol_df.loc[last_idx]
            exit_close = float(last_row["close"])
            if not np.isfinite(exit_close) or exit_close <= 0:
                del open_positions[key]
                continue

            exit_date = pd.Timestamp(last_idx)
            raw_ret = exit_close / pos["entry_price"] - 1.0
            if pos["side"] == "short":
                raw_ret *= -1.0

            if exit_date in date_to_index and pos["entry_date"] in date_to_index:
                holding_days = date_to_index[exit_date] - date_to_index[pos["entry_date"]] + 1
            else:
                holding_days = max(1, int((exit_date - pos["entry_date"]).days) + 1)

            net_ret = raw_ret - (2.0 * trading_cost_pct)
            weighted_return = net_ret * pos["weight"]
            daily_pnl_by_date[exit_date] = daily_pnl_by_date.get(exit_date, 0.0) + float(weighted_return)
            trade_records.append(
                {
                    "symbol": symbol,
                    "direction": pos["side"],
                    "signal_date": pd.Timestamp(pos["signal_date"]),
                    "entry_date": pd.Timestamp(pos["entry_date"]),
                    "exit_date": exit_date,
                    "entry_price": float(pos["entry_price"]),
                    "exit_price": float(exit_close),
                    "weight": float(pos["weight"]),
                    "pnl": float(weighted_return),
                    "raw_return": float(raw_ret),
                    "net_return": float(net_ret),
                    "cost_pct": float(2.0 * trading_cost_pct),
                    "deviation": float(pos["signal_deviation"]),
                    "exit_reason": "forced_liquidation",
                    "holding_days": int(holding_days),
                }
            )
            del open_positions[key]

    trades_df = pd.DataFrame(trade_records)
    if not daily_pnl_by_date:
        return trades_df, pd.DataFrame()

    daily_df = (
        pd.DataFrame(
            [{"date": date, "strategy_return": pnl} for date, pnl in daily_pnl_by_date.items()]
        )
        .groupby("date", as_index=False)
        .agg(strategy_return=("strategy_return", "sum"))
        .sort_values("date")
        .reset_index(drop=True)
    )
    daily_df["equity_curve"] = (1.0 + daily_df["strategy_return"]).cumprod()

    if daily_df.empty:
        return trades_df, pd.DataFrame()

    ann_factor = 252.0
    total_return = float(daily_df["equity_curve"].iloc[-1] - 1.0)
    years = max(float(len(daily_df)) / ann_factor, 1.0 / ann_factor)
    cagr = float((1.0 + total_return) ** (1.0 / years) - 1.0)
    std_daily = float(daily_df["strategy_return"].std(ddof=1))
    mean_daily = float(daily_df["strategy_return"].mean())
    sharpe = (mean_daily / std_daily) * np.sqrt(ann_factor) if std_daily > 0 else np.nan
    dd = daily_df["equity_curve"] / daily_df["equity_curve"].cummax() - 1.0
    mdd = float(dd.min()) if not dd.empty else 0.0

    win_rate = float((trades_df["net_return"] > 0).mean()) if not trades_df.empty else np.nan
    summary_df = pd.DataFrame(
        [
            {
                "strategy": "ma5_deviation_mean_reversion",
                "top_n": int(top_n),
                "entry_timing": entry_timing,
                "exit_timing": exit_timing,
                "weight_mode": weight_mode,
                "take_profit_pct": take_profit_pct,
                "stop_loss_pct": stop_loss_pct,
                "mean_revert_exit_threshold": mean_revert_exit_threshold,
                "max_holding_days": int(max_holding_days),
                "commission_bps": float(commission_bps),
                "slippage_bps": float(slippage_bps),
                "allow_reentry_after_exit_on_same_day": bool(allow_reentry_after_exit_on_same_day),
                "signal_priority": "exit_first_then_entry",
                "trade_count": int(trades_df.shape[0]),
                "win_rate": win_rate,
                "cagr": cagr,
                "sharpe": sharpe,
                "max_drawdown": mdd,
                "total_return": total_return,
            }
        ]
    )
    return trades_df, summary_df


def grid_search_selling_climax(
    symbols: Optional[Iterable[str]] = None,
    seed: int = 42,
    train_ratio: float = 0.5,
    volume_lookbacks: Optional[List[int]] = None,
    volume_multipliers: Optional[List[float]] = None,
    drop_pcts: Optional[List[float]] = None,
    close_positions: Optional[List[float]] = None,
    confirm_ks: Optional[List[int]] = None,
    atr_lookbacks: Optional[List[int]] = None,
    drop_atr_mults: Optional[List[float]] = None,
    drop_condition_modes: Optional[List[str]] = None,
    trend_ma_lens: Optional[List[int]] = None,
    trend_modes: Optional[List[str]] = None,
    stop_atr_mults: Optional[List[float]] = None,
    time_stop_bars_list: Optional[List[int]] = None,
    trailing_atr_mults: Optional[List[float]] = None,
    min_avg_dollar_volumes: Optional[List[float]] = None,
    min_avg_volumes: Optional[List[float]] = None,
    liquidity_lookback: int = 20,
    vol_percentile_thresholds: Optional[List[float]] = None,
    vol_lookback2s: Optional[List[int]] = None,
    max_gap_pcts: Optional[List[float]] = None,
    min_signals: int = 20,
    gap_penalty: float = 0.5,
    progress_callback: Optional[Callable[[int, int, Optional[str]], None]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    セリングクライマックスの検出パラメータをグリッドサーチする。

    train/validation で成功率(precision)を比較し、差分にペナルティを掛けて過剰最適化を抑える。
    """

    volume_lookbacks = volume_lookbacks or [20]
    volume_multipliers = volume_multipliers or [2.0, 2.5, 3.0]
    drop_pcts = drop_pcts or [0.03, 0.04, 0.05]
    close_positions = close_positions or [0.3, 0.4, 0.5]
    confirm_ks = confirm_ks or [2, 3, 5]
    atr_lookbacks = atr_lookbacks or [14]
    drop_atr_mults = drop_atr_mults or [2.0]
    drop_condition_modes = drop_condition_modes or ["drop_pct_only"]
    trend_ma_lens = trend_ma_lens or [50]
    trend_modes = trend_modes or ["none"]
    stop_atr_mults = stop_atr_mults or [1.5]
    time_stop_bars_list = time_stop_bars_list or [5]
    trailing_atr_mults = trailing_atr_mults or [None]
    min_avg_dollar_volumes = min_avg_dollar_volumes or [None]
    min_avg_volumes = min_avg_volumes or [None]
    vol_percentile_thresholds = vol_percentile_thresholds or [None]
    vol_lookback2s = vol_lookback2s or [20]
    max_gap_pcts = max_gap_pcts or [None]

    raw_symbols = list(symbols) if symbols is not None else get_available_symbols()
    target_symbols: List[str] = []
    symbol_price_data: Dict[str, pd.DataFrame] = {}
    for symbol in raw_symbols:
        try:
            df_price = load_price_csv(symbol)
        except Exception:
            continue
        if df_price.empty:
            continue
        symbol_price_data[symbol] = df_price.sort_values("date").reset_index(drop=True)
        target_symbols.append(symbol)

    if not target_symbols:
        return pd.DataFrame(), pd.DataFrame()

    train_symbols, validation_symbols = split_symbols_randomly(
        target_symbols, seed=seed, train_ratio=train_ratio
    )
    train_set = set(train_symbols)
    symbol_success_cache: Dict[
        str,
        Dict[
            Tuple[int, Optional[int], Optional[float], Optional[float], int],
            pd.Series,
        ],
    ] = {symbol: {} for symbol in target_symbols}

    eval_records = []
    best_score = float("-inf")
    best_record: Optional[Dict[str, float]] = None

    combinations = list(
        product(
            volume_lookbacks,
            volume_multipliers,
            drop_pcts,
            close_positions,
            confirm_ks,
            atr_lookbacks,
            drop_atr_mults,
            drop_condition_modes,
            trend_ma_lens,
            trend_modes,
            stop_atr_mults,
            time_stop_bars_list,
            trailing_atr_mults,
            min_avg_dollar_volumes,
            min_avg_volumes,
            vol_percentile_thresholds,
            vol_lookback2s,
            max_gap_pcts,
        )
    )
    total_combos = len(combinations)

    needed_volume_lookbacks = {int(v) for v in volume_lookbacks if v is not None and v > 0}
    needed_liquidity_lookbacks = {
        int(liquidity_lookback)
    } if liquidity_lookback is not None and liquidity_lookback > 0 else set()
    needed_lookbacks = needed_volume_lookbacks | needed_liquidity_lookbacks
    needed_atr_lookbacks = {int(v) for v in atr_lookbacks if v is not None and v > 0}
    needed_vol_lookbacks = {int(v) for v in vol_lookback2s if v is not None and v > 1}
    needed_trend_ma_lens = {int(v) for v in trend_ma_lens if v is not None and v > 1}

    symbol_feature_cache: Dict[str, Dict[str, Any]] = {}
    for symbol in target_symbols:
        symbol_feature_cache[symbol] = _build_selling_climax_feature_cache(
            symbol_price_data[symbol],
            needed_lookbacks=needed_lookbacks,
            needed_atr_lookbacks=needed_atr_lookbacks,
            needed_vol_lookbacks=needed_vol_lookbacks,
            needed_trend_ma_lens=needed_trend_ma_lens,
        )

    for combo_index, (
        volume_lookback,
        volume_multiplier,
        drop_pct,
        close_position,
        confirm_k,
        atr_lookback,
        drop_atr_mult,
        drop_condition_mode,
        trend_ma_len,
        trend_mode,
        stop_atr_mult,
        time_stop_bars,
        trailing_atr_mult,
        min_avg_dollar_volume,
        min_avg_volume,
        vol_percentile_threshold,
        vol_lookback2,
        max_gap_pct,
    ) in enumerate(combinations, start=1):
        train_signals = 0
        train_success = 0
        val_signals = 0
        val_success = 0

        for symbol in target_symbols:
            df_sorted = symbol_price_data[symbol]
            candidates = _detect_selling_climax_candidates_with_cache(
                df_sorted,
                feature_cache=symbol_feature_cache[symbol],
                volume_lookback=volume_lookback,
                volume_multiplier=volume_multiplier,
                drop_pct=drop_pct,
                close_position=close_position,
                atr_lookback=atr_lookback,
                drop_atr_mult=drop_atr_mult,
                drop_condition_mode=drop_condition_mode,
                trend_ma_len=trend_ma_len,
                trend_mode=trend_mode,
                min_avg_dollar_volume=min_avg_dollar_volume,
                min_avg_volume=min_avg_volume,
                liquidity_lookback=liquidity_lookback,
                vol_percentile_threshold=vol_percentile_threshold,
                vol_lookback2=vol_lookback2,
                max_gap_pct=max_gap_pct,
            )
            if not candidates.any():
                continue
            success_key = (
                confirm_k,
                time_stop_bars,
                stop_atr_mult,
                trailing_atr_mult,
                atr_lookback,
            )
            success = symbol_success_cache[symbol].get(success_key)
            if success is None:
                success = _label_selling_climax_success(
                    df_sorted,
                    confirm_k=confirm_k,
                    time_stop_bars=time_stop_bars,
                    stop_atr_mult=stop_atr_mult,
                    trailing_atr_mult=trailing_atr_mult,
                    atr_lookback=atr_lookback,
                )
                symbol_success_cache[symbol][success_key] = success
            signal_count = int(candidates.sum())
            success_count = int((candidates & success).sum())
            if symbol in train_set:
                train_signals += signal_count
                train_success += success_count
            else:
                val_signals += signal_count
                val_success += success_count

        if progress_callback:
            progress_callback(combo_index, total_combos, "グリッドサーチ")

        if train_signals < min_signals or val_signals < min_signals:
            continue

        train_precision = train_success / train_signals if train_signals else 0.0
        val_precision = val_success / val_signals if val_signals else 0.0
        score = val_precision - abs(train_precision - val_precision) * gap_penalty

        record = {
            "volume_lookback": volume_lookback,
            "volume_multiplier": volume_multiplier,
            "drop_pct": drop_pct,
            "close_position": close_position,
            "confirm_k": confirm_k,
            "atr_lookback": atr_lookback,
            "drop_atr_mult": drop_atr_mult,
            "drop_condition_mode": drop_condition_mode,
            "trend_ma_len": trend_ma_len,
            "trend_mode": trend_mode,
            "stop_atr_mult": stop_atr_mult,
            "time_stop_bars": time_stop_bars,
            "trailing_atr_mult": trailing_atr_mult,
            "min_avg_dollar_volume": min_avg_dollar_volume,
            "min_avg_volume": min_avg_volume,
            "vol_percentile_threshold": vol_percentile_threshold,
            "vol_lookback2": vol_lookback2,
            "max_gap_pct": max_gap_pct,
            "train_precision": train_precision,
            "validation_precision": val_precision,
            "train_signals": train_signals,
            "validation_signals": val_signals,
            "score": score,
        }
        eval_records.append(record)

        if score > best_score:
            best_score = score
            best_record = record

    eval_df = pd.DataFrame(eval_records).sort_values("score", ascending=False)
    if best_record is None:
        return eval_df, pd.DataFrame()

    best_summary = pd.DataFrame([best_record])
    return eval_df, best_summary


def optimize_canslim_parameters(
    symbols: Optional[Iterable[str]] = None,
    seed: int = 42,
    train_ratio: float = 0.5,
    lookahead: int = 20,
    return_threshold: float = 0.03,
    max_evals: int = 30,
    sample_ratio: float = 0.6,
    window_candidates: Optional[Dict[str, List[int]]] = None,
    volume_multipliers: Optional[List[float]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    CAN-SLIM検出パラメーターを簡易最適化し、train/validation の結果を返す。

    - max_evals で探索回数を制限し、時間を抑制する。
    - sample_ratio で対象銘柄を間引き、処理時間を短縮する。
    """

    target_symbols = list(symbols) if symbols is not None else get_available_symbols()
    if not target_symbols:
        return pd.DataFrame(), pd.DataFrame()

    rng = random.Random(seed)
    rng.shuffle(target_symbols)
    sample_size = max(1, int(len(target_symbols) * sample_ratio))
    sampled_symbols = target_symbols[:sample_size]

    window_candidates = window_candidates or {
        "cup_window": [40, 50, 60],
        "saucer_window": [70, 80, 90],
        "handle_window": [7, 10, 12],
    }
    volume_multipliers = volume_multipliers or [1.3, 1.5, 1.8]

    eval_records = []
    best_score = float("-inf")
    best_params: Optional[Dict[str, float]] = None
    best_results = pd.DataFrame()
    best_summary = pd.DataFrame()

    for _ in range(max_evals):
        params = {
            "cup_window": rng.choice(window_candidates["cup_window"]),
            "saucer_window": rng.choice(window_candidates["saucer_window"]),
            "handle_window": rng.choice(window_candidates["handle_window"]),
            "volume_multiplier": rng.choice(volume_multipliers),
        }

        results_df, summary_df = run_canslim_backtest(
            symbols=sampled_symbols,
            seed=seed,
            train_ratio=train_ratio,
            lookahead=lookahead,
            return_threshold=return_threshold,
            volume_multiplier=params["volume_multiplier"],
            cup_window=params["cup_window"],
            saucer_window=params["saucer_window"],
            handle_window=params["handle_window"],
        )

        if results_df.empty:
            continue

        score_df = summary_df[
            (summary_df["dataset"] == "train") & (summary_df["label"] == "up")
        ]
        score = float(score_df["avg_return"].mean()) if not score_df.empty else 0.0

        eval_records.append(
            {
                **params,
                "train_up_avg_return": score,
                "signal_count": int(results_df[results_df["dataset"] == "train"].shape[0]),
            }
        )

        if score > best_score:
            best_score = score
            best_params = params
            best_results = results_df
            best_summary = summary_df

    eval_df = pd.DataFrame(eval_records).sort_values(
        "train_up_avg_return", ascending=False
    )
    if best_params is None:
        return eval_df, pd.DataFrame()

    best_summary = best_summary.copy()
    for key, value in best_params.items():
        best_summary[f"best_{key}"] = [value] * len(best_summary)

    return eval_df, best_summary


def grid_search_cup_shape(
    symbols: Optional[Iterable[str]] = None,
    seed: int = 42,
    train_ratio: float = 0.5,
    lookahead: int = 20,
    return_threshold: float = 0.03,
    volume_multiplier: float = 1.5,
    min_signals: int = 10,
    gap_penalty: float = 0.5,
    cup_windows: Optional[List[int]] = None,
    handle_windows: Optional[List[int]] = None,
    depth_ranges: Optional[List[Tuple[float, float]]] = None,
    recovery_ratios: Optional[List[float]] = None,
    handle_max_depths: Optional[List[float]] = None,
    top_k: Optional[int] = None,
    coarse_stride: int = 1,
    progress_callback: Optional[Callable[[int, int, Optional[str]], None]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    カップウィズハンドル形状のパラメーターをグリッドサーチで確認する。

    validation を主指標にし、train との差分にペナルティを掛けて過剰最適化を抑える。
    """

    cup_windows = cup_windows or [40, 50, 60, 70]
    handle_windows = handle_windows or [7, 10, 12]
    depth_ranges = depth_ranges or [(0.12, 0.35), (0.15, 0.4), (0.18, 0.45)]
    recovery_ratios = recovery_ratios or [0.82, 0.85, 0.88]
    handle_max_depths = handle_max_depths or [0.1, 0.12, 0.15]

    raw_symbols = list(symbols) if symbols is not None else get_available_symbols()
    target_symbols: List[str] = []
    symbol_price_data: Dict[str, pd.DataFrame] = {}
    for symbol in raw_symbols:
        try:
            symbol_price_data[symbol] = load_price_csv(symbol)
            target_symbols.append(symbol)
        except Exception:
            continue

    if not target_symbols:
        return pd.DataFrame(), pd.DataFrame()

    eval_records = []
    best_score = float("-inf")
    best_params: Optional[Dict[str, float]] = None
    best_results = pd.DataFrame()
    best_summary = pd.DataFrame()

    def _evaluate_combo(
        cup_window: int,
        handle_window: int,
        depth_range: Tuple[float, float],
        recovery_ratio: float,
        handle_max_depth: float,
    ) -> Optional[Dict[str, float]]:
        results_df, summary_df = run_canslim_backtest(
            symbols=target_symbols,
            symbol_price_data=symbol_price_data,
            seed=seed,
            train_ratio=train_ratio,
            lookahead=lookahead,
            return_threshold=return_threshold,
            volume_multiplier=volume_multiplier,
            cup_window=cup_window,
            handle_window=handle_window,
            cup_depth_range=depth_range,
            cup_recovery_ratio=recovery_ratio,
            cup_handle_max_depth=handle_max_depth,
        )

        if results_df.empty:
            return None

        train_signals = results_df[results_df["dataset"] == "train"].shape[0]
        val_signals = results_df[results_df["dataset"] == "validation"].shape[0]
        if train_signals < min_signals or val_signals < min_signals:
            return None

        train_up = summary_df[
            (summary_df["dataset"] == "train")
            & (summary_df["pattern"] == "cup_with_handle")
            & (summary_df["label"] == "up")
        ]
        val_up = summary_df[
            (summary_df["dataset"] == "validation")
            & (summary_df["pattern"] == "cup_with_handle")
            & (summary_df["label"] == "up")
        ]
        train_up_avg = float(train_up["avg_return"].mean()) if not train_up.empty else 0.0
        val_up_avg = float(val_up["avg_return"].mean()) if not val_up.empty else 0.0
        generalization_gap = abs(train_up_avg - val_up_avg)
        score = val_up_avg - gap_penalty * generalization_gap

        return {
            "cup_window": cup_window,
            "handle_window": handle_window,
            "depth_min": depth_range[0],
            "depth_max": depth_range[1],
            "recovery_ratio": recovery_ratio,
            "handle_max_depth": handle_max_depth,
            "train_up_avg_return": train_up_avg,
            "validation_up_avg_return": val_up_avg,
            "generalization_gap": generalization_gap,
            "score": score,
            "train_signals": int(train_signals),
            "validation_signals": int(val_signals),
            "_results_df": results_df,
            "_summary_df": summary_df,
        }

    use_two_stage = top_k is not None and top_k > 0 and coarse_stride > 1
    all_combos = list(
        product(cup_windows, handle_windows, depth_ranges, recovery_ratios, handle_max_depths)
    )

    if use_two_stage:
        coarse_cup_windows = cup_windows[::coarse_stride]
        if cup_windows[-1] not in coarse_cup_windows:
            coarse_cup_windows = coarse_cup_windows + [cup_windows[-1]]
        coarse_recovery_ratios = recovery_ratios[::coarse_stride]
        if recovery_ratios[-1] not in coarse_recovery_ratios:
            coarse_recovery_ratios = coarse_recovery_ratios + [recovery_ratios[-1]]

        coarse_combos = list(
            product(
                coarse_cup_windows,
                handle_windows,
                depth_ranges,
                coarse_recovery_ratios,
                handle_max_depths,
            )
        )
        top_candidates: List[Tuple[int, int, Tuple[float, float], float, float]] = []
        coarse_records: List[Dict[str, float]] = []

        for combo in coarse_combos:
            score_record = _evaluate_combo(*combo)
            if score_record is not None:
                coarse_records.append(score_record)

        if coarse_records:
            coarse_records_sorted = sorted(coarse_records, key=lambda x: x["score"], reverse=True)
            top_candidates = [
                (
                    int(record["cup_window"]),
                    int(record["handle_window"]),
                    (float(record["depth_min"]), float(record["depth_max"])),
                    float(record["recovery_ratio"]),
                    float(record["handle_max_depth"]),
                )
                for record in coarse_records_sorted[:top_k]
            ]

        cup_window_to_idx = {value: idx for idx, value in enumerate(cup_windows)}
        refine_combos = set()
        for candidate in top_candidates:
            cup_window, handle_window, depth_range, recovery_ratio, handle_max_depth = candidate
            cup_idx = cup_window_to_idx[cup_window]
            for near_cup_idx in [max(0, cup_idx - 1), cup_idx, min(len(cup_windows) - 1, cup_idx + 1)]:
                near_cup_window = cup_windows[near_cup_idx]
                for near_handle_depth in handle_max_depths:
                    if abs(near_handle_depth - handle_max_depth) <= 0.02:
                        refine_combos.add(
                            (
                                near_cup_window,
                                handle_window,
                                depth_range,
                                recovery_ratio,
                                near_handle_depth,
                            )
                        )

        total_combos = len(coarse_combos) + len(refine_combos)
        combo_index = 0

        for _ in coarse_combos:
            combo_index += 1
            if progress_callback:
                progress_callback(combo_index, total_combos, "グリッドサーチ(粗探索)")

        for combo in refine_combos:
            combo_index += 1
            score_record = _evaluate_combo(*combo)
            if score_record is not None:
                eval_records.append(score_record)
                if score_record["score"] > best_score:
                    best_score = score_record["score"]
                    best_params = {
                        "cup_window": score_record["cup_window"],
                        "handle_window": score_record["handle_window"],
                        "cup_depth_range": (
                            score_record["depth_min"],
                            score_record["depth_max"],
                        ),
                        "cup_recovery_ratio": score_record["recovery_ratio"],
                        "cup_handle_max_depth": score_record["handle_max_depth"],
                        "volume_multiplier": volume_multiplier,
                    }
                    best_results = score_record["_results_df"]
                    best_summary = score_record["_summary_df"]
            if progress_callback:
                progress_callback(combo_index, total_combos, "グリッドサーチ(詳細探索)")
    else:
        total_combos = len(all_combos)
        combo_index = 0
        for combo in all_combos:
            combo_index += 1
            score_record = _evaluate_combo(*combo)
            if score_record is not None:
                eval_records.append(score_record)
                if score_record["score"] > best_score:
                    best_score = score_record["score"]
                    best_params = {
                        "cup_window": score_record["cup_window"],
                        "handle_window": score_record["handle_window"],
                        "cup_depth_range": (
                            score_record["depth_min"],
                            score_record["depth_max"],
                        ),
                        "cup_recovery_ratio": score_record["recovery_ratio"],
                        "cup_handle_max_depth": score_record["handle_max_depth"],
                        "volume_multiplier": volume_multiplier,
                    }
                    best_results = score_record["_results_df"]
                    best_summary = score_record["_summary_df"]
            if progress_callback:
                progress_callback(combo_index, total_combos, "グリッドサーチ")

    if eval_records:
        eval_df = (
            pd.DataFrame(eval_records)
            .drop(columns=["_results_df", "_summary_df"], errors="ignore")
            .sort_values("score", ascending=False)
        )
    else:
        eval_df = pd.DataFrame()
    if best_params is None:
        return eval_df, pd.DataFrame()

    best_summary = best_summary.copy()
    for key, value in best_params.items():
        best_summary[f"best_{key}"] = [value] * len(best_summary)

    return eval_df, best_summary


def _run_cli() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="CAN-SLIM backtest runner")
    parser.add_argument("--optimize", action="store_true", help="run parameter optimization")
    parser.add_argument("--grid-search", action="store_true", help="run grid search for cup shape")
    parser.add_argument("--lookahead", type=int, default=20)
    parser.add_argument("--return-threshold", type=float, default=0.03)
    parser.add_argument("--volume-multiplier", type=float, default=1.5)
    parser.add_argument("--cup-window", type=int, default=50)
    parser.add_argument("--saucer-window", type=int, default=80)
    parser.add_argument("--handle-window", type=int, default=10)
    parser.add_argument("--max-evals", type=int, default=30)
    parser.add_argument("--sample-ratio", type=float, default=0.6)
    parser.add_argument("--min-signals", type=int, default=10)
    parser.add_argument("--gap-penalty", type=float, default=0.5)
    args = parser.parse_args()

    if args.grid_search:
        eval_df, best_summary = grid_search_cup_shape(
            lookahead=args.lookahead,
            return_threshold=args.return_threshold,
            volume_multiplier=args.volume_multiplier,
            min_signals=args.min_signals,
            gap_penalty=args.gap_penalty,
        )
        print("=== Grid search results ===")
        print(eval_df.head(20).to_string(index=False))
        print("=== Best summary ===")
        print(best_summary.head(20).to_string(index=False))
    elif args.optimize:
        eval_df, best_summary = optimize_canslim_parameters(
            lookahead=args.lookahead,
            return_threshold=args.return_threshold,
            max_evals=args.max_evals,
            sample_ratio=args.sample_ratio,
        )
        print("=== Optimization results ===")
        print(eval_df.head(20).to_string(index=False))
        print("=== Best summary ===")
        print(best_summary.head(20).to_string(index=False))
    else:
        results_df, summary_df = run_canslim_backtest(
            lookahead=args.lookahead,
            return_threshold=args.return_threshold,
            volume_multiplier=args.volume_multiplier,
            cup_window=args.cup_window,
            saucer_window=args.saucer_window,
            handle_window=args.handle_window,
        )
        print("=== Summary ===")
        print(summary_df.to_string(index=False))
        if not results_df.empty:
            print("=== Sample results ===")
            print(results_df.head(20).to_string(index=False))


if __name__ == "__main__":
    _run_cli()
