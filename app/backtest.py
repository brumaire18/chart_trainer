from dataclasses import dataclass
import random
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from .data_loader import get_available_symbols, load_price_csv


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
        best_summary[f"best_{key}"] = value

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
        best_summary[f"best_{key}"] = value

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
