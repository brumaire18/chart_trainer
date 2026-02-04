from dataclasses import dataclass
import random
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from .data_loader import get_available_symbols, load_price_csv
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
) -> pd.Series:
    if df.empty:
        return pd.Series(dtype=bool)

    volume_avg = df["volume"].rolling(volume_lookback, min_periods=volume_lookback).mean()
    volume_ratio = df["volume"] / volume_avg
    prev_close = df["close"].shift(1)
    drop_ratio = (df["close"] - prev_close) / prev_close
    candle_range = df["high"] - df["low"]
    close_pos = (df["close"] - df["low"]) / candle_range.replace(0, np.nan)

    candidates = (
        (df["close"] <= df["open"])
        & (volume_ratio >= volume_multiplier)
        & (drop_ratio <= -abs(drop_pct))
        & (candle_range > 0)
        & (close_pos <= close_position)
    )
    return candidates.fillna(False)


def _label_selling_climax_success(df: pd.DataFrame, confirm_k: int) -> pd.Series:
    if df.empty or confirm_k <= 0:
        return pd.Series(dtype=bool)
    max_close_fwd = df["close"].shift(-1).rolling(confirm_k).max().shift(-(confirm_k - 1))
    min_low_fwd = df["low"].shift(-1).rolling(confirm_k).min().shift(-(confirm_k - 1))
    success = (max_close_fwd > df["high"]) & (min_low_fwd >= df["low"])
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

    target_symbols = list(symbols) if symbols is not None else get_available_symbols()
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

    target_symbols = list(symbols) if symbols is not None else get_available_symbols()
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


def grid_search_selling_climax(
    symbols: Optional[Iterable[str]] = None,
    seed: int = 42,
    train_ratio: float = 0.5,
    volume_lookbacks: Optional[List[int]] = None,
    volume_multipliers: Optional[List[float]] = None,
    drop_pcts: Optional[List[float]] = None,
    close_positions: Optional[List[float]] = None,
    confirm_ks: Optional[List[int]] = None,
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

    target_symbols = list(symbols) if symbols is not None else get_available_symbols()
    if not target_symbols:
        return pd.DataFrame(), pd.DataFrame()

    train_symbols, validation_symbols = split_symbols_randomly(
        target_symbols, seed=seed, train_ratio=train_ratio
    )
    train_set = set(train_symbols)

    eval_records = []
    best_score = float("-inf")
    best_record: Optional[Dict[str, float]] = None

    total_combos = (
        len(volume_lookbacks)
        * len(volume_multipliers)
        * len(drop_pcts)
        * len(close_positions)
        * len(confirm_ks)
    )
    combo_index = 0

    for volume_lookback in volume_lookbacks:
        for volume_multiplier in volume_multipliers:
            for drop_pct in drop_pcts:
                for close_position in close_positions:
                    for confirm_k in confirm_ks:
                        combo_index += 1
                        train_signals = 0
                        train_success = 0
                        val_signals = 0
                        val_success = 0

                        for symbol in target_symbols:
                            try:
                                df_price = load_price_csv(symbol)
                            except Exception:
                                continue
                            if df_price.empty:
                                continue
                            df_sorted = df_price.sort_values("date").reset_index(drop=True)
                            candidates = _detect_selling_climax_candidates(
                                df_sorted,
                                volume_lookback=volume_lookback,
                                volume_multiplier=volume_multiplier,
                                drop_pct=drop_pct,
                                close_position=close_position,
                            )
                            if not candidates.any():
                                continue
                            success = _label_selling_climax_success(
                                df_sorted, confirm_k=confirm_k
                            )
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

    eval_records = []
    best_score = float("-inf")
    best_params: Optional[Dict[str, float]] = None
    best_results = pd.DataFrame()
    best_summary = pd.DataFrame()

    total_combos = (
        len(cup_windows)
        * len(handle_windows)
        * len(depth_ranges)
        * len(recovery_ratios)
        * len(handle_max_depths)
    )
    combo_index = 0
    for cup_window in cup_windows:
        for handle_window in handle_windows:
            for depth_range in depth_ranges:
                for recovery_ratio in recovery_ratios:
                    for handle_max_depth in handle_max_depths:
                        combo_index += 1
                        results_df, summary_df = run_canslim_backtest(
                            symbols=symbols,
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
                            if progress_callback:
                                progress_callback(combo_index, total_combos, "グリッドサーチ")
                            continue

                        train_signals = results_df[results_df["dataset"] == "train"].shape[
                            0
                        ]
                        val_signals = results_df[
                            results_df["dataset"] == "validation"
                        ].shape[0]
                        if train_signals < min_signals or val_signals < min_signals:
                            if progress_callback:
                                progress_callback(combo_index, total_combos, "グリッドサーチ")
                            continue

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
                        train_up_avg = (
                            float(train_up["avg_return"].mean())
                            if not train_up.empty
                            else 0.0
                        )
                        val_up_avg = (
                            float(val_up["avg_return"].mean()) if not val_up.empty else 0.0
                        )
                        generalization_gap = abs(train_up_avg - val_up_avg)
                        score = val_up_avg - gap_penalty * generalization_gap

                        eval_records.append(
                            {
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
                            }
                        )
                        if progress_callback:
                            progress_callback(combo_index, total_combos, "グリッドサーチ")

                        if score > best_score:
                            best_score = score
                            best_params = {
                                "cup_window": cup_window,
                                "handle_window": handle_window,
                                "cup_depth_range": depth_range,
                                "cup_recovery_ratio": recovery_ratio,
                                "cup_handle_max_depth": handle_max_depth,
                                "volume_multiplier": volume_multiplier,
                            }
                            best_results = results_df
                            best_summary = summary_df

    eval_df = pd.DataFrame(eval_records).sort_values("score", ascending=False)
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
