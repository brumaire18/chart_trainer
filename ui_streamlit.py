from collections import Counter
from datetime import date, timedelta
import json
import time
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from app.config import DEFAULT_LOOKBACK_BARS, META_DIR, PRICE_CSV_DIR
from app.data_loader import (
    attach_topix_relative_strength,
    enforce_light_plan_window,
    fetch_and_save_price_csv,
    get_available_symbols,
    load_price_csv,
    load_topix_csv,
)
from app.market_breadth import aggregate_market_breadth, compute_breadth_indicators
from app.pair_trading import (
    coint as cointegration_test,
    compute_pair_metrics,
    compute_min_pair_samples,
    compute_spread_series,
    evaluate_pair_candidates,
    generate_anchor_pair_candidates,
    generate_pairs_by_sector_candidates,
)
from app.backtest import grid_search_cup_shape, run_canslim_backtest, scan_canslim_patterns
from app.jquants_fetcher import (
    append_quotes_for_date,
    build_universe,
    get_credential_status,
    load_listed_master,
    update_topix,
    update_universe_with_anchor_day,
    update_universe,
)
from app.pair_trading import (
    PairTradeConfig,
    backtest_pairs,
    generate_pairs_by_sector,
    optimize_pair_trade_parameters,
)


LIGHT_PLAN_YEARS = 5
PAIR_CACHE_PATH = META_DIR / "pair_candidates_cache.json"


def _format_eta(seconds: Optional[float]) -> str:
    if seconds is None or seconds < 0:
        return "計算中"
    seconds_int = int(round(seconds))
    if seconds_int < 60:
        return f"{seconds_int}秒"
    minutes, sec = divmod(seconds_int, 60)
    if minutes < 60:
        return f"{minutes}分{sec}秒"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}時間{minutes}分{sec}秒"


def _build_progress_updater(
    label: str,
) -> Tuple[Callable[[int, int, Optional[str]], None], Callable[[], None]]:
    bar = st.progress(0.0)
    status = st.empty()
    state = {"stage": None, "start": None}

    def update(current: int, total: int, stage: Optional[str] = None) -> None:
        if total is None or total <= 0:
            status.text(f"{label}を実行中...")
            bar.progress(0.0)
            return
        if stage != state["stage"]:
            state["stage"] = stage
            state["start"] = time.monotonic()
        start = state["start"] or time.monotonic()
        elapsed = max(0.0, time.monotonic() - start)
        if current <= 0 or elapsed <= 0:
            remaining = None
        else:
            remaining = (total - current) * (elapsed / current)
        stage_label = f"{stage}: " if stage else ""
        status.text(
            f"{label} {stage_label}{current}/{total} | 残り推定: {_format_eta(remaining)}"
        )
        progress_value = min(max(current / total, 0.0), 1.0)
        bar.progress(progress_value)

    def done() -> None:
        bar.progress(1.0)
        status.text(f"{label} 完了")

    return update, done


def _attach_pair_sector_info(
    pairs_df: pd.DataFrame, listed_df: pd.DataFrame
) -> pd.DataFrame:
    if pairs_df.empty:
        return pairs_df
    if "code" not in listed_df.columns:
        return pairs_df

    df = pairs_df.copy()
    code_series = listed_df["code"].astype(str).str.zfill(4)
    sector17_map = {}
    sector33_map = {}
    if "sector17" in listed_df.columns:
        sector17_map = dict(zip(code_series, listed_df["sector17"]))
    if "sector33" in listed_df.columns:
        sector33_map = dict(zip(code_series, listed_df["sector33"]))

    df["sector17_a"] = df["symbol_a"].astype(str).str.zfill(4).map(sector17_map)
    df["sector17_b"] = df["symbol_b"].astype(str).str.zfill(4).map(sector17_map)
    df["sector33_a"] = df["symbol_a"].astype(str).str.zfill(4).map(sector33_map)
    df["sector33_b"] = df["symbol_b"].astype(str).str.zfill(4).map(sector33_map)

    df["pair_sector17"] = df.apply(
        lambda row: row["sector17_a"]
        if pd.notna(row["sector17_a"]) and row["sector17_a"] == row["sector17_b"]
        else None,
        axis=1,
    )
    df["pair_sector33"] = df.apply(
        lambda row: row["sector33_a"]
        if pd.notna(row["sector33_a"]) and row["sector33_a"] == row["sector33_b"]
        else None,
        axis=1,
    )
    return df


def _load_pair_cache() -> Tuple[Optional[pd.DataFrame], Optional[dict]]:
    if not PAIR_CACHE_PATH.exists():
        return None, None
    try:
        payload = json.loads(PAIR_CACHE_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None, None
    pairs = payload.get("pairs", [])
    metadata = payload.get("metadata") or {}
    metadata["updated_at"] = payload.get("updated_at")
    return pd.DataFrame(pairs), metadata


def _save_pair_cache(pairs_df: pd.DataFrame, metadata: Dict[str, object]) -> None:
    payload = {
        "updated_at": date.today().isoformat(),
        "metadata": metadata,
        "pairs": pairs_df.to_dict(orient="records"),
    }
    PAIR_CACHE_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _filter_cached_pairs(
    pairs_df: pd.DataFrame,
    sector17: Optional[str],
    sector33: Optional[str],
    anchor_symbol: Optional[str],
    available_symbols: Optional[List[str]] = None,
) -> pd.DataFrame:
    df = pairs_df.copy()
    if available_symbols:
        symbol_set = {str(symbol).zfill(4) for symbol in available_symbols}
        df = df[
            df["symbol_a"].astype(str).str.zfill(4).isin(symbol_set)
            & df["symbol_b"].astype(str).str.zfill(4).isin(symbol_set)
        ]
    if sector17:
        df = df[df["pair_sector17"].astype(str) == str(sector17)]
    if sector33:
        df = df[df["pair_sector33"].astype(str) == str(sector33)]
    if anchor_symbol:
        anchor_symbol = str(anchor_symbol).zfill(4)
        df = df[
            (df["symbol_a"].astype(str).str.zfill(4) == anchor_symbol)
            | (df["symbol_b"].astype(str).str.zfill(4) == anchor_symbol)
        ]
    return df


def _limit_pairs_per_sector(
    pairs_df: pd.DataFrame, sector_col: str, max_pairs_per_sector: int
) -> pd.DataFrame:
    if pairs_df.empty:
        return pairs_df
    if sector_col == "sector33":
        sector_key = "pair_sector33"
    else:
        sector_key = "pair_sector17"
    if sector_key not in pairs_df.columns:
        return pairs_df
    limited = []
    for _, group in pairs_df.dropna(subset=[sector_key]).groupby(sector_key):
        limited.append(group.head(max_pairs_per_sector))
    if not limited:
        return pairs_df.iloc[0:0]
    return pd.concat(limited, ignore_index=True)


def _limit_pairs_per_symbol(
    pairs_df: pd.DataFrame, max_pairs_per_symbol: int
) -> pd.DataFrame:
    if pairs_df.empty:
        return pairs_df
    if max_pairs_per_symbol < 1:
        return pairs_df
    if "symbol_a" not in pairs_df.columns or "symbol_b" not in pairs_df.columns:
        return pairs_df
    df = pairs_df.copy()
    if "p_value" in df.columns:
        df = df.sort_values("p_value", ascending=True)
    counts: Dict[str, int] = {}
    kept_rows = []
    for _, row in df.iterrows():
        symbol_a = str(row.get("symbol_a"))
        symbol_b = str(row.get("symbol_b"))
        if not symbol_a or symbol_a == "None":
            continue
        if not symbol_b or symbol_b == "None":
            continue
        if counts.get(symbol_a, 0) >= max_pairs_per_symbol:
            continue
        if counts.get(symbol_b, 0) >= max_pairs_per_symbol:
            continue
        kept_rows.append(row)
        counts[symbol_a] = counts.get(symbol_a, 0) + 1
        counts[symbol_b] = counts.get(symbol_b, 0) + 1
    if not kept_rows:
        return pairs_df.iloc[0:0]
    return pd.DataFrame(kept_rows).reset_index(drop=True)


def _load_latest_update_date() -> Optional[date]:
    meta_files = sorted(META_DIR.glob("*.json"))
    if not meta_files:
        return None

    latest: Optional[date] = None
    for meta_path in meta_files:
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        history_end = meta.get("history_end")
        if not history_end:
            continue
        try:
            candidate = date.fromisoformat(history_end)
        except ValueError:
            continue
        if latest is None or candidate > latest:
            latest = candidate
    return latest


def _get_price_cache_key(symbol: str) -> str:
    """
    CSVファイルの更新を検知するためのキャッシュキー。

    ファイルの更新時刻とサイズをキャッシュのキーに含め、
    最新のファイルに置き換えられた場合でも再計算されるようにする。
    """

    csv_path = PRICE_CSV_DIR / f"{symbol}.csv"
    try:
        stat = csv_path.stat()
        return f"{stat.st_mtime_ns}-{stat.st_size}"
    except FileNotFoundError:
        return "missing"


def _get_price_csv_dir_cache_key() -> str:
    """
    price_csv 配下の更新を検知するためのキャッシュキー。

    CSVファイルの更新時刻の最大値をキーに含める。
    """

    try:
        csv_paths = list(PRICE_CSV_DIR.glob("*.csv"))
    except FileNotFoundError:
        return "missing"
    if not csv_paths:
        return "empty"
    max_mtime = max(path.stat().st_mtime_ns for path in csv_paths)
    return f"{max_mtime}-{len(csv_paths)}"


def _get_listed_master_cache_key() -> str:
    """listed_master.csv の更新を検知するためのキャッシュキー。"""

    path = META_DIR / "listed_master.csv"
    try:
        return str(path.stat().st_mtime_ns)
    except FileNotFoundError:
        return "missing"


@st.cache_data(show_spinner=False)
def _load_available_symbols(cache_key: str) -> List[str]:
    """CSV 更新をキーに銘柄一覧をキャッシュする。"""

    _ = cache_key
    return get_available_symbols()


@st.cache_data(show_spinner=False)
def _load_listed_master_cached(cache_key: str) -> pd.DataFrame:
    """listed_master の更新をキーに銘柄マスタをキャッシュする。"""

    _ = cache_key
    return load_listed_master()


@st.cache_data(show_spinner=False)
def _load_price_with_indicators(
    symbol: str,
    cache_key: str,
    topix_cache_key: Optional[str] = None,
    lookback: Optional[int] = None,
    compute_indicators: bool = True,
) -> Tuple[pd.DataFrame, int, Optional[dict]]:
    """
    日足データを読み込み、出来高0を除外したうえでインジケーターを計算する。

    cache_key を引数に含めることで、CSV更新時にキャッシュが破棄される。
    lookback を指定した場合は、指標計算に必要な本数だけ末尾を使用する。
    """

    _ = (cache_key, topix_cache_key)  # キャッシュ用に参照だけ行う
    indicator_padding = 60
    tail_rows = lookback + indicator_padding if lookback is not None else None
    df_daily = load_price_csv(symbol, tail_rows=tail_rows)
    df_daily_trading = df_daily[df_daily["volume"].fillna(0) > 0].copy()
    removed_rows = len(df_daily) - len(df_daily_trading)
    if lookback is not None:
        df_daily_trading = df_daily_trading.tail(lookback + indicator_padding)
    if compute_indicators:
        df_ind = _compute_indicators(df_daily_trading)
    else:
        df_ind = df_daily_trading.copy()
    topix_info = None
    try:
        df_ind, topix_info = attach_topix_relative_strength(df_ind)
    except FileNotFoundError:
        topix_info = {"status": "missing_file"}
    except ValueError as exc:
        topix_info = {"status": "invalid_file", "message": str(exc)}
    return df_ind, removed_rows, topix_info


@st.cache_data(show_spinner=False)
def _load_price_for_grid(symbol: str, cache_key: str) -> pd.DataFrame:
    """
    グリッド表示向けに軽量な価格データを読み込む。

    cache_key を引数に含めることで、CSV更新時にキャッシュが破棄される。
    """

    _ = cache_key  # キャッシュ用に参照だけ行う
    df_daily = load_price_csv(symbol)
    return df_daily.loc[:, ["date", "close", "volume"]].copy()


@st.cache_data(show_spinner=False)
def _compute_weekly_volume_map(
    symbols: List[str], cache_key: str
) -> Tuple[dict, dict]:
    """
    週足出来高の最新値と週足データをまとめて計算する。

    cache_key を引数に含めることで、PRICE_CSV_DIR 更新時にキャッシュが破棄される。
    """

    _ = cache_key  # キャッシュ用に参照だけ行う
    weekly_volume_map = {}
    weekly_df_map = {}
    for symbol in symbols:
        price_cache_key = _get_price_cache_key(symbol)
        try:
            grid_df = _load_price_for_grid(symbol, price_cache_key)
        except Exception:
            continue
        weekly_df = _resample_ohlc(grid_df, "weekly")
        if weekly_df.empty:
            continue
        weekly_df_map[symbol] = weekly_df
        latest_weekly_vol = weekly_df.iloc[-1]["volume"]
        if pd.notna(latest_weekly_vol):
            weekly_volume_map[symbol] = float(latest_weekly_vol)
    return weekly_volume_map, weekly_df_map


def _compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """主要なテクニカル指標を事前計算する。"""

    df_ind = df.copy()
    df_ind["sma20"] = df_ind["close"].rolling(20).mean()
    df_ind["sma50"] = df_ind["close"].rolling(50).mean()
    df_ind["sma200"] = df_ind["close"].rolling(200).mean()

    # ボリンジャーバンド (20, 1〜3σ)
    bb_basis = df_ind["close"].rolling(20).mean()
    bb_std = df_ind["close"].rolling(20).std()
    df_ind["bb_upper_1"] = bb_basis + 1 * bb_std
    df_ind["bb_lower_1"] = bb_basis - 1 * bb_std
    df_ind["bb_upper_2"] = bb_basis + 2 * bb_std
    df_ind["bb_lower_2"] = bb_basis - 2 * bb_std
    df_ind["bb_upper_3"] = bb_basis + 3 * bb_std
    df_ind["bb_lower_3"] = bb_basis - 3 * bb_std

    # RSI (14)
    delta = df_ind["close"].diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    df_ind["rsi14"] = 100 - (100 / (1 + rs))

    # MACD (12, 26, 9)
    ema12 = df_ind["close"].ewm(span=12, adjust=False).mean()
    ema26 = df_ind["close"].ewm(span=26, adjust=False).mean()
    df_ind["macd"] = ema12 - ema26
    df_ind["macd_signal"] = df_ind["macd"].ewm(span=9, adjust=False).mean()
    df_ind["macd_hist"] = df_ind["macd"] - df_ind["macd_signal"]

    # ストキャスティクス (14, 3)
    low14 = df_ind["low"].rolling(14).min()
    high14 = df_ind["high"].rolling(14).max()
    df_ind["stoch_k"] = (df_ind["close"] - low14) / (high14 - low14) * 100
    df_ind["stoch_d"] = df_ind["stoch_k"].rolling(3).mean()

    # オンバランスボリューム (OBV)
    price_diff = df_ind["close"].diff().fillna(0)
    direction = price_diff.apply(lambda x: 1 if x > 0 else -1 if x < 0 else 0)
    df_ind["obv"] = (direction * df_ind["volume"].fillna(0)).cumsum()

    # 一目均衡表
    if {"high", "low", "close"}.issubset(df_ind.columns):
        high9 = df_ind["high"].rolling(9).max()
        low9 = df_ind["low"].rolling(9).min()
        df_ind["ichimoku_tenkan"] = (high9 + low9) / 2

        high26 = df_ind["high"].rolling(26).max()
        low26 = df_ind["low"].rolling(26).min()
        df_ind["ichimoku_kijun"] = (high26 + low26) / 2

        df_ind["ichimoku_senkou_a"] = (
            (df_ind["ichimoku_tenkan"] + df_ind["ichimoku_kijun"]) / 2
        ).shift(26)
        high52 = df_ind["high"].rolling(52).max()
        low52 = df_ind["low"].rolling(52).min()
        df_ind["ichimoku_senkou_b"] = ((high52 + low52) / 2).shift(26)
        df_ind["ichimoku_chikou"] = df_ind["close"].shift(-26)
    else:
        df_ind["ichimoku_tenkan"] = pd.NA
        df_ind["ichimoku_kijun"] = pd.NA
        df_ind["ichimoku_senkou_a"] = pd.NA
        df_ind["ichimoku_senkou_b"] = pd.NA
        df_ind["ichimoku_chikou"] = pd.NA

    return df_ind


def _detect_selling_climax(
    df: pd.DataFrame,
    volume_lookback: int = 20,
    volume_multiplier: float = 2.0,
    drop_pct: float = 0.04,
    close_position: float = 0.3,
) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(
            columns=["flag", "volume_ratio", "drop_pct", "close_position"], index=df.index
        )
    volume_avg = df["volume"].rolling(volume_lookback).mean()
    volume_ratio = df["volume"] / volume_avg
    prev_close = df["close"].shift(1)
    drop_ratio = (df["close"] - prev_close) / prev_close
    candle_range = df["high"] - df["low"]
    close_pos = (df["close"] - df["low"]) / candle_range.replace(0, np.nan)
    flag = (
        (df["close"] <= df["open"])
        & (volume_ratio >= volume_multiplier)
        & (drop_ratio <= -abs(drop_pct))
        & (candle_range > 0)
        & (close_pos <= close_position)
    )
    return pd.DataFrame(
        {
            "flag": flag.fillna(False),
            "volume_ratio": volume_ratio,
            "drop_pct": drop_ratio,
            "close_position": close_pos,
        },
        index=df.index,
    )


def _detect_new_highs(
    df: pd.DataFrame, lookback: int = 252, enforce_bb_lower: bool = True
) -> pd.Series:
    if df.empty:
        return pd.Series(dtype=bool)
    rolling_high = df["high"].rolling(lookback, min_periods=lookback).max()
    is_new_high = df["high"] >= rolling_high.shift(1)
    if enforce_bb_lower and "bb_lower_1" in df.columns:
        is_new_high &= df["close"] >= df["bb_lower_1"]
    return is_new_high.fillna(False)


def _compute_moving_average(series: pd.Series, period: int, ma_type: str) -> pd.Series:
    if period <= 0:
        return pd.Series([pd.NA] * len(series), index=series.index)
    if ma_type == "EMA":
        return series.ewm(span=period, adjust=False).mean()
    return series.rolling(period).mean()


def _find_local_extrema(df: pd.DataFrame, order: int = 3) -> Tuple[List[int], List[int]]:
    if df.empty:
        return [], []
    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()
    local_max = []
    local_min = []
    for idx in range(order, len(df) - order):
        window_high = highs[idx - order : idx + order + 1]
        window_low = lows[idx - order : idx + order + 1]
        if np.isnan(window_high).any() or np.isnan(window_low).any():
            continue
        if highs[idx] >= window_high.max():
            local_max.append(idx)
        if lows[idx] <= window_low.min():
            local_min.append(idx)
    return local_min, local_max


def _fit_trendline(
    df: pd.DataFrame, indices: List[int], price_col: str
) -> Optional[Tuple[float, float]]:
    if len(indices) < 2:
        return None
    x_vals = df.loc[df.index[indices], "date"].astype("int64").to_numpy()
    y_vals = df.loc[df.index[indices], price_col].to_numpy()
    if np.isnan(x_vals).any() or np.isnan(y_vals).any():
        return None
    slope, intercept = np.polyfit(x_vals, y_vals, 1)
    return float(slope), float(intercept)


def _build_trendline_trace(
    df: pd.DataFrame,
    slope: float,
    intercept: float,
    name: str,
    color: str,
    dash: str = "dash",
) -> go.Scatter:
    start_date = df["date"].iloc[0]
    end_date = df["date"].iloc[-1]
    x_vals = np.array([start_date, end_date], dtype="datetime64[ns]").astype("int64")
    y_vals = slope * x_vals + intercept
    return go.Scatter(
        x=[start_date, end_date],
        y=y_vals,
        mode="lines",
        name=name,
        line=dict(color=color, dash=dash, width=2),
    )


def _compute_support_resistance(
    df: pd.DataFrame, order: int = 3, max_points: int = 5
) -> Dict[str, Optional[Tuple[float, float]]]:
    local_min, local_max = _find_local_extrema(df, order=order)
    support_idx = local_min[-max_points:]
    resistance_idx = local_max[-max_points:]
    support = _fit_trendline(df, support_idx, "low")
    resistance = _fit_trendline(df, resistance_idx, "high")
    return {"support": support, "resistance": resistance}


def _detect_head_and_shoulders(
    df: pd.DataFrame,
    order: int = 3,
    shoulder_tolerance: float = 0.08,
    min_separation: int = 5,
) -> Optional[dict]:
    local_min, local_max = _find_local_extrema(df, order=order)
    if len(local_max) < 3:
        return None
    peaks = local_max
    best = None
    for i in range(len(peaks) - 2):
        for j in range(i + 1, len(peaks) - 1):
            for k in range(j + 1, len(peaks)):
                left, head, right = peaks[i], peaks[j], peaks[k]
                if head - left < min_separation or right - head < min_separation:
                    continue
                left_price = df["high"].iloc[left]
                head_price = df["high"].iloc[head]
                right_price = df["high"].iloc[right]
                if not (head_price > left_price and head_price > right_price):
                    continue
                shoulder_diff = abs(left_price - right_price) / max(left_price, right_price)
                if shoulder_diff > shoulder_tolerance:
                    continue
                trough_candidates_left = [m for m in local_min if left < m < head]
                trough_candidates_right = [m for m in local_min if head < m < right]
                if not trough_candidates_left or not trough_candidates_right:
                    continue
                trough_left = trough_candidates_left[-1]
                trough_right = trough_candidates_right[0]
                best = {
                    "left": left,
                    "head": head,
                    "right": right,
                    "trough_left": trough_left,
                    "trough_right": trough_right,
                }
    return best


def _build_trading_rangebreaks(dates: pd.Series) -> List[dict]:
    """休場日を除外するための rangebreaks を作成する。"""

    if dates.empty:
        return []
    date_index = pd.DatetimeIndex(pd.to_datetime(dates).dt.normalize().unique()).sort_values()
    if date_index.empty:
        return []
    full_range = pd.date_range(start=date_index[0], end=date_index[-1], freq="D")
    missing_dates = full_range.difference(date_index)
    if missing_dates.empty:
        return []
    return [dict(values=list(missing_dates.to_pydatetime()))]


def _lookup_cached_pair_metrics(
    pairs_df: Optional[pd.DataFrame], symbol_a: str, symbol_b: str
) -> Optional[dict]:
    if pairs_df is None or pairs_df.empty:
        return None
    df = pairs_df.copy()
    df["symbol_a"] = df["symbol_a"].astype(str).str.zfill(4)
    df["symbol_b"] = df["symbol_b"].astype(str).str.zfill(4)
    symbol_a = str(symbol_a).zfill(4)
    symbol_b = str(symbol_b).zfill(4)
    row = df[
        ((df["symbol_a"] == symbol_a) & (df["symbol_b"] == symbol_b))
        | ((df["symbol_a"] == symbol_b) & (df["symbol_b"] == symbol_a))
    ]
    if row.empty:
        return None
    return row.iloc[0].to_dict()


def _build_pair_metrics_table(metrics: dict) -> pd.DataFrame:
    table_df = pd.DataFrame(
        {
            "p値": [metrics.get("p_value")],
            "半減期": [metrics.get("half_life")],
            "β": [metrics.get("beta")],
            "直近類似度": [metrics.get("recent_similarity")],
            "直近リターン相関": [metrics.get("recent_return_corr")],
            "長期類似度": [metrics.get("long_similarity")],
            "長期リターン相関": [metrics.get("long_return_corr")],
            "スプレッド平均": [metrics.get("spread_mean")],
            "スプレッド標準偏差": [metrics.get("spread_std")],
            "最新スプレッド": [metrics.get("spread_latest")],
            "最新Zスコア": [metrics.get("zscore_latest")],
        }
    )
    return table_df.round(
        {
            "p値": 4,
            "半減期": 2,
            "β": 3,
            "直近類似度": 3,
            "直近リターン相関": 3,
            "長期類似度": 3,
            "長期リターン相関": 3,
            "スプレッド平均": 4,
            "スプレッド標準偏差": 4,
            "最新スプレッド": 4,
            "最新Zスコア": 2,
        }
    )


def _render_pair_spread_chart(
    df_pair: pd.DataFrame,
    entry_threshold: float,
    exit_threshold: float,
    pair_metrics: Optional[dict] = None,
) -> None:
    if df_pair.empty:
        st.warning("スプレッドを計算するためのデータが不足しています。")
        return
    df_pair = df_pair.copy()
    df_pair["date_str"] = df_pair["date"].dt.strftime("%Y-%m-%d")
    rangebreaks = _build_trading_rangebreaks(df_pair["date"])
    fig_pair = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.55, 0.45],
    )
    fig_pair.add_trace(
        go.Scatter(
            x=df_pair["date_str"],
            y=df_pair["spread"],
            mode="lines",
            name="スプレッド",
            line=dict(color="#1f77b4"),
        ),
        row=1,
        col=1,
    )
    spread_mean = pair_metrics.get("spread_mean") if pair_metrics else None
    if spread_mean is not None:
        fig_pair.add_hline(
            y=spread_mean,
            line=dict(color="#888888", dash="dash"),
            row=1,
            col=1,
        )
    fig_pair.add_trace(
        go.Scatter(
            x=df_pair["date_str"],
            y=df_pair["zscore"],
            mode="lines",
            name="Zスコア",
            line=dict(color="#ff7f0e"),
        ),
        row=2,
        col=1,
    )
    for level, color, label in (
        (entry_threshold, "#d62728", "エントリー上限"),
        (-entry_threshold, "#d62728", "エントリー下限"),
        (exit_threshold, "#2ca02c", "エグジット上限"),
        (-exit_threshold, "#2ca02c", "エグジット下限"),
    ):
        fig_pair.add_hline(
            y=level,
            line=dict(color=color, dash="dash"),
            annotation_text=label,
            row=2,
            col=1,
        )
    fig_pair.update_layout(
        height=700,
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(orientation="h"),
    )
    fig_pair.update_xaxes(type="category", rangebreaks=rangebreaks)
    fig_pair.update_yaxes(title_text="スプレッド", row=1, col=1)
    fig_pair.update_yaxes(title_text="Zスコア", row=2, col=1)
    st.plotly_chart(fig_pair, use_container_width=True)


def _resample_ohlc(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """日足を週足・月足にリサンプリングする。"""

    if timeframe == "daily":
        return df.copy()

    rule = {"weekly": "W", "monthly": "M"}.get(timeframe)
    if not rule:
        return df.copy()

    df_idx = df.set_index("date")
    agg_map = {}
    if "open" in df.columns:
        agg_map["open"] = "first"
    if "high" in df.columns:
        agg_map["high"] = "max"
    if "low" in df.columns:
        agg_map["low"] = "min"
    if "close" in df.columns:
        agg_map["close"] = "last"
    if "volume" in df.columns:
        agg_map["volume"] = "sum"
    if "topix_close" in df.columns:
        agg_map["topix_close"] = "last"
    if "topix_rs" in df.columns:
        agg_map["topix_rs"] = "last"
    if "topix_rs_log" in df.columns:
        agg_map["topix_rs_log"] = "last"
    if not agg_map:
        return df.copy()

    resampled = df_idx.resample(rule).agg(agg_map).dropna().reset_index()
    return resampled


def _estimate_daily_lookback(lookback: int, timeframe: str) -> int:
    multiplier = {"weekly": 5, "monthly": 22}.get(timeframe, 1)
    return lookback * multiplier


def _point_and_figure(
    df: pd.DataFrame, box_size: Optional[float] = None, reversal: int = 3
) -> pd.DataFrame:
    """
    クローズ価格からシンプルなポイント・アンド・フィギュア用の箱データを生成する。
    box_size が未指定のときは終値の中央値の1%を自動設定する。
    """

    closes = df["close"].tolist()
    if not closes:
        return pd.DataFrame()

    if box_size is None:
        median_price = pd.Series(closes).median()
        box_size = max(median_price * 0.01, 1e-6)

    def _to_box_level(price: float) -> int:
        return int(round(price / box_size))

    columns = []
    curr_col = {"type": "X", "levels": []}

    curr_level = _to_box_level(closes[0])
    curr_col["levels"].append(curr_level)

    for price in closes[1:]:
        level = _to_box_level(price)
        if curr_col["type"] == "X":
            if level > curr_level:
                curr_col["levels"].extend(range(curr_level + 1, level + 1))
                curr_level = level
            elif curr_level - level >= reversal:
                columns.append(curr_col)
                curr_col = {"type": "O", "levels": list(range(curr_level - 1, level - 1, -1))}
                curr_level = level
        else:
            if level < curr_level:
                curr_col["levels"].extend(range(curr_level - 1, level - 1, -1))
                curr_level = level
            elif level - curr_level >= reversal:
                columns.append(curr_col)
                curr_col = {"type": "X", "levels": list(range(curr_level + 1, level + 1))}
                curr_level = level

    if columns and columns[-1] is not curr_col:
        columns.append(curr_col)
    elif not columns:
        columns.append(curr_col)

    rows = []
    for idx, col in enumerate(columns):
        for lvl in col["levels"]:
            rows.append({"col": idx, "price": lvl * box_size, "type": col["type"]})

    return pd.DataFrame(rows)


def _has_macd_cross(
    df: pd.DataFrame,
    direction: str,
    lookback: int = 5,
    debug_log: Optional[List[str]] = None,
) -> bool:
    """
    直近数本の足で MACD がシグナルを上抜け（ゴールデン）または下抜け（デッド）したかを判定する。

    direction: "golden" または "dead"
    lookback: 何本前までのクロスを許容するか
    """

    if len(df) < 2 or direction not in {"golden", "dead"}:
        if debug_log is not None:
            debug_log.append(
                f"skip: insufficient length ({len(df)}) or invalid direction={direction}"
            )
        return False

    df_tail = df.tail(lookback + 1)
    if debug_log is not None:
        debug_log.append(
            f"check last {len(df_tail)} bars (lookback={lookback}, direction={direction})"
        )
    for idx in range(1, len(df_tail)):
        curr = df_tail.iloc[idx]
        prev = df_tail.iloc[idx - 1]

        if pd.isna(curr[["macd", "macd_signal"]]).any() or pd.isna(
            prev[["macd", "macd_signal"]]
        ).any():
            if debug_log is not None:
                debug_log.append(f"idx {idx}: skip due to NaN macd/macd_signal")
            continue

        if direction == "golden":
            crossed = prev["macd"] <= prev["macd_signal"] and curr["macd"] > curr["macd_signal"]
            hist_ok = curr["macd_hist"] >= 0
            if debug_log is not None:
                debug_log.append(
                    f"idx {idx}: golden prev({prev['macd']:.4f},{prev['macd_signal']:.4f}) -> curr({curr['macd']:.4f},{curr['macd_signal']:.4f}), hist={curr['macd_hist']:.4f}"
                )
        else:
            crossed = prev["macd"] >= prev["macd_signal"] and curr["macd"] < curr["macd_signal"]
            hist_ok = curr["macd_hist"] <= 0
            if debug_log is not None:
                debug_log.append(
                    f"idx {idx}: dead prev({prev['macd']:.4f},{prev['macd_signal']:.4f}) -> curr({curr['macd']:.4f},{curr['macd_signal']:.4f}), hist={curr['macd_hist']:.4f}"
                )

        if crossed and hist_ok:
            if debug_log is not None:
                debug_log.append(f"idx {idx}: crossed and hist_ok -> True")
            return True

    if debug_log is not None:
        debug_log.append("no cross detected in lookback")
    return False


def _columns_to_check_latest(
    apply_rsi_condition: bool,
    macd_condition: str,
    require_sma20_trend: bool,
    apply_topix_rs_condition: bool,
) -> List[str]:
    columns = ["date", "close", "volume"]

    if apply_rsi_condition:
        columns.append("rsi14")

    if macd_condition != "none":
        columns.extend(["macd", "macd_signal", "macd_hist"])

    if require_sma20_trend:
        columns.append("sma20")

    if apply_topix_rs_condition:
        columns.append("topix_rs")

    return list(dict.fromkeys(columns))


def _latest_has_required_data(
    latest: pd.Series,
    apply_rsi_condition: bool,
    macd_condition: str,
    require_sma20_trend: bool,
    apply_topix_rs_condition: bool,
) -> bool:
    columns_to_check = _columns_to_check_latest(
        apply_rsi_condition, macd_condition, require_sma20_trend, apply_topix_rs_condition
    )
    missing_columns = [col for col in columns_to_check if col not in latest.index]
    if missing_columns:
        return False
    return not latest[columns_to_check].isna().any()


def _calculate_minimum_data_length(
    apply_rsi_condition: bool,
    macd_condition: str,
    macd_lookback: int,
    require_sma20_trend: bool,
    sma_trend_lookback: int,
    apply_volume_condition: bool,
    apply_topix_rs_condition: bool,
    topix_rs_lookback: int,
    apply_new_high_signal: bool,
    new_high_lookback: int,
    apply_selling_climax_signal: bool,
    selling_volume_lookback: int,
    signal_lookback_days: int,
    apply_canslim_condition: bool,
    cup_window: int,
    saucer_window: int,
    handle_window: int,
    apply_weekly_volume_quartile: bool,
) -> Tuple[int, List[str]]:
    """スクリーニングに必要な最低データ本数と理由を返す。"""

    required_length = 50
    reasons: List[str] = ["インジケーター計算を安定させるため最低50本"]

    if apply_rsi_condition:
        required_length = max(required_length, 14)
        reasons.append("RSI(14)を評価するには14本以上必要")

    if macd_condition != "none":
        macd_length = max(26, macd_lookback + 1)
        required_length = max(required_length, macd_length)
        reasons.append(
            f"MACDクロス判定にはEMA26と直近{macd_lookback}本を含めた{macd_length}本以上が必要"
        )

    if require_sma20_trend:
        sma_length = 20 + sma_trend_lookback
        required_length = max(required_length, sma_length)
        reasons.append(f"SMA20上向き判定には20+{sma_trend_lookback}本以上必要")

    if apply_volume_condition:
        required_length = max(required_length, 20)
        reasons.append("20日平均出来高を計算するには20本以上必要")

    if apply_topix_rs_condition:
        required_length = max(required_length, topix_rs_lookback + 1)
        reasons.append(f"TOPIX RSを{topix_rs_lookback}本前と比較するには十分な期間が必要")

    if apply_new_high_signal:
        new_high_length = max(new_high_lookback + 1, 20, signal_lookback_days + 1)
        required_length = max(required_length, new_high_length)
        reasons.append(f"新高値シグナルの判定には{new_high_length}本以上必要")

    if apply_selling_climax_signal:
        selling_length = max(selling_volume_lookback, signal_lookback_days + 1)
        required_length = max(required_length, selling_length)
        reasons.append(f"セリングクライマックス判定には{selling_length}本以上必要")

    if apply_weekly_volume_quartile:
        required_length = max(required_length, 5)
        reasons.append("週出来高判定には最低1週間分(5本程度)のデータが必要")
    if apply_canslim_condition:
        base_window = max(cup_window, saucer_window) + handle_window
        required_length = max(required_length, base_window + 2)
        reasons.append(
            f"CAN-SLIM判定にはカップ/ソーサー({base_window}本以上)のデータが必要"
        )

    return required_length, reasons


def _build_mini_chart(
    df: pd.DataFrame,
    symbol: str,
    name: str,
    timeframe: str,
    lookback: int,
    show_title: bool = True,
) -> Optional[go.Figure]:
    df_resampled = _resample_ohlc(df, timeframe)
    if df_resampled.empty:
        return None
    df_resampled = df_resampled.tail(lookback).copy()
    if df_resampled.empty:
        return None
    df_resampled["date_str"] = df_resampled["date"].dt.strftime("%y/%m/%d")
    df_resampled["ma25"] = df_resampled["close"].rolling(25).mean()
    df_resampled["ma50"] = df_resampled["close"].rolling(50).mean()
    df_resampled["ma200"] = df_resampled["close"].rolling(200).mean()
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.7, 0.3],
    )
    fig.add_trace(
        go.Scatter(
            x=df_resampled["date"],
            y=df_resampled["close"],
            mode="lines",
            line=dict(color="#1f77b4"),
            name="終値",
        ),
        row=1,
        col=1,
    )
    if df_resampled["ma25"].notna().any():
        fig.add_trace(
            go.Scatter(
                x=df_resampled["date"],
                y=df_resampled["ma25"],
                mode="lines",
                line=dict(color="#ff7f0e", width=1),
                name="MA25",
            ),
            row=1,
            col=1,
        )
    if df_resampled["ma50"].notna().any():
        fig.add_trace(
            go.Scatter(
                x=df_resampled["date"],
                y=df_resampled["ma50"],
                mode="lines",
                line=dict(color="#2ca02c", width=1),
                name="MA50",
            ),
            row=1,
            col=1,
        )
    if df_resampled["ma200"].notna().any():
        fig.add_trace(
            go.Scatter(
                x=df_resampled["date"],
                y=df_resampled["ma200"],
                mode="lines",
                line=dict(color="#7f7f7f", width=1),
                name="MA200",
            ),
            row=1,
            col=1,
        )
    if "volume" in df_resampled.columns:
        fig.add_trace(
            go.Bar(
                x=df_resampled["date"],
                y=df_resampled["volume"],
                name="出来高",
                marker_color="rgba(100,149,237,0.6)",
                opacity=0.7,
            ),
            row=2,
            col=1,
        )
    fig.update_layout(
        margin=dict(l=10, r=10, t=30, b=10),
        height=240,
        title=f"{symbol} {name}",
        showlegend=False,
    )
    rangebreaks = _build_trading_rangebreaks(df_resampled["date"])
    fig.update_xaxes(tickformat="%y/%m/%d", nticks=6, title="", rangebreaks=rangebreaks)
    fig.update_yaxes(title="")
    return fig


def _build_mini_chart_from_resampled(
    df_resampled: pd.DataFrame,
    symbol: str,
    name: str,
    lookback: int,
    show_title: bool = True,
) -> Optional[go.Figure]:
    if df_resampled.empty:
        return None
    df_resampled = df_resampled.tail(lookback).copy()
    if df_resampled.empty:
        return None
    df_resampled["ma25"] = df_resampled["close"].rolling(25).mean()
    df_resampled["ma50"] = df_resampled["close"].rolling(50).mean()
    df_resampled["ma200"] = df_resampled["close"].rolling(200).mean()
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.7, 0.3],
    )
    fig.add_trace(
        go.Scatter(
            x=df_resampled["date"],
            y=df_resampled["close"],
            mode="lines",
            line=dict(color="#1f77b4"),
            name="終値",
        ),
        row=1,
        col=1,
    )
    if df_resampled["ma25"].notna().any():
        fig.add_trace(
            go.Scatter(
                x=df_resampled["date"],
                y=df_resampled["ma25"],
                mode="lines",
                line=dict(color="#ff7f0e", width=1),
                name="MA25",
            ),
            row=1,
            col=1,
        )
    if df_resampled["ma50"].notna().any():
        fig.add_trace(
            go.Scatter(
                x=df_resampled["date"],
                y=df_resampled["ma50"],
                mode="lines",
                line=dict(color="#2ca02c", width=1),
                name="MA50",
            ),
            row=1,
            col=1,
        )
    if df_resampled["ma200"].notna().any():
        fig.add_trace(
            go.Scatter(
                x=df_resampled["date"],
                y=df_resampled["ma200"],
                mode="lines",
                line=dict(color="#7f7f7f", width=1),
                name="MA200",
            ),
            row=1,
            col=1,
        )
    if "volume" in df_resampled.columns:
        fig.add_trace(
            go.Bar(
                x=df_resampled["date"],
                y=df_resampled["volume"],
                name="出来高",
                marker_color="rgba(100,149,237,0.6)",
                opacity=0.7,
            ),
            row=2,
            col=1,
        )
    title_text = f"{symbol} {name}" if show_title else ""
    fig.update_layout(
        margin=dict(l=10, r=10, t=30, b=10),
        height=240,
        title=title_text,
        showlegend=False,
    )
    rangebreaks = _build_trading_rangebreaks(df_resampled["date"])
    fig.update_xaxes(tickformat="%Y-%m-%d", title="", rangebreaks=rangebreaks)
    use_log_scale = (df_resampled["close"] > 0).all()
    if use_log_scale:
        fig.update_yaxes(title="", type="log", row=1, col=1)
    else:
        fig.update_yaxes(title="", row=1, col=1)
    fig.update_yaxes(title="", row=2, col=1)
    return fig


def main():
    st.set_page_config(page_title="Chart Trainer (Line ver.)", layout="wide")
    st.title("Chart Trainer - フェーズ1（ラインチャート版）")

    # --- サイドバー ---
    st.sidebar.header("設定")

    st.sidebar.subheader("J-Quants からデータ取得")
    price_csv_cache_key = _get_price_csv_dir_cache_key()
    available_symbols = _load_available_symbols(price_csv_cache_key)
    default_symbol = available_symbols[0] if available_symbols else ""
    query_symbol = st.query_params.get("symbol")
    if isinstance(query_symbol, list):
        query_symbol = query_symbol[0] if query_symbol else None
    if query_symbol:
        default_symbol = str(query_symbol)

    download_symbol = st.sidebar.text_input(
        "銘柄コード (例: 7203)", value=default_symbol
    )
    default_start = date.today() - timedelta(days=365 * LIGHT_PLAN_YEARS)
    default_end = date.today()
    st.sidebar.caption(
        "※ ライトプランでは過去約5年分まで取得できます。"
        "指定にかかわらず取得可能な最大範囲（過去5年）に自動調整します。"
    )
    topix_full_refresh = st.sidebar.checkbox(
        "TOPIXを全期間で再取得", value=False, key="topix_full_refresh"
    )
    if st.sidebar.button("TOPIXを一括ダウンロード", key="download_topix_button"):
        try:
            with st.spinner("TOPIXをダウンロードしています..."):
                update_topix(full_refresh=topix_full_refresh)
            st.sidebar.success("TOPIXのダウンロードに成功しました。")
            st.rerun()
        except Exception as exc:
            st.sidebar.error(f"TOPIXのダウンロードに失敗しました: {exc}")

    with st.sidebar.expander("市場区分を一括更新"):
        creds = get_credential_status()
        st.write("認証情報の検知状況:")
        st.write(f"- JQUANTS_API_KEY: {'✅' if creds['JQUANTS_API_KEY'] else '❌'}")
        st.caption("新仕様ではリフレッシュトークンの設定が推奨です。")

        latest_update_date = _load_latest_update_date()
        if latest_update_date:
            st.date_input(
                "最終更新日 (全体)",
                value=latest_update_date,
                disabled=True,
                key="latest_update_date_display",
            )
        else:
            st.info("最終更新日は未取得です。")

        include_custom = st.checkbox(
            "custom_symbols.txt も含める", value=False, key="include_custom_universe"
        )
        universe_source = st.radio(
            "更新対象",
            options=["prime_standard", "growth", "listed_all"],
            format_func=lambda v: {
                "prime_standard": "プライム+スタンダード",
                "growth": "グロース",
                "listed_all": "listed_masterにある全銘柄",
            }[v],
            index=0,
            key="universe_source",
        )
        full_refresh = st.checkbox(
            "フルリフレッシュ（取得可能な5年分を再取得）",
            value=False,
            key="full_refresh_universe",
        )
        min_rows_refresh_enabled = st.checkbox(
            "データが少ない銘柄を再取得",
            value=False,
            key="min_rows_refresh_enabled",
        )
        min_rows_refresh: Optional[int] = None
        if min_rows_refresh_enabled:
            min_rows_refresh = int(
                st.number_input(
                    "再取得判定の最小行数",
                    min_value=1,
                    value=200,
                    step=50,
                    key="min_rows_refresh",
                )
            )
        st.markdown("---")
        st.write("スナップショット → 日次更新フロー")
        st.caption(
            "まず指定日の全銘柄株価を取得し、その後で日次データを最新化します。"
            "曜日未指定時は直近の火曜日を使用します。"
        )
        use_anchor_flow = st.checkbox(
            "スナップショットを取得してから更新する", value=True, key="use_anchor_flow"
        )
        anchor_weekday = st.selectbox(
            "スナップショット曜日（0=月曜, 1=火曜...）",
            options=list(range(7)),
            format_func=lambda w: f"{['月','火','水','木','金','土','日'][w]}曜 ({w})",
            index=1,
            key="anchor_weekday",
        )
        use_custom_anchor_date = st.checkbox(
            "スナップショット日を手動指定", value=False, key="use_custom_anchor_date"
        )
        anchor_date_input: Optional[date] = None
        if use_custom_anchor_date:
            anchor_date_input = st.date_input(
                "取得日 (YYYY-MM-DD)", value=date.today(), key="anchor_date_input"
            )
        st.markdown("---")
        st.write("日次更新 (指定日)")
        append_date_input = st.date_input(
            "日次更新日 (YYYY-MM-DD)",
            value=date.today(),
            key="append_date_input",
        )
        if st.button("指定日で日次更新", key="append_date_button"):
            try:
                with st.spinner("指定日の日次更新を実行しています..."):
                    target_codes = build_universe(
                        include_custom=include_custom,
                        use_listed_master=universe_source == "listed_all",
                        market_filter="growth" if universe_source == "growth" else "prime_standard",
                    )
                    append_quotes_for_date(
                        append_date_input.isoformat(),
                        codes=target_codes,
                    )
                    try:
                        update_topix(full_refresh=False)
                    except Exception as exc:
                        st.warning(f"TOPIX の更新に失敗しました: {exc}")
                st.success("指定日の日次更新が完了しました。")
                st.rerun()
            except Exception as exc:
                st.error(f"指定日の日次更新に失敗しました: {exc}")
        if st.button("ユニバースを更新", key="update_universe_button"):
            try:
                with st.spinner("ユニバースを更新しています..."):
                    target_codes = build_universe(
                        include_custom=include_custom,
                        use_listed_master=universe_source == "listed_all",
                        market_filter="growth" if universe_source == "growth" else "prime_standard",
                    )
                    if use_anchor_flow:
                        update_universe_with_anchor_day(
                            codes=target_codes,
                            anchor_date=anchor_date_input.isoformat() if anchor_date_input else None,
                            anchor_weekday=anchor_weekday,
                            include_custom=include_custom,
                            use_listed_master=universe_source == "listed_all",
                            market_filter="growth" if universe_source == "growth" else "prime_standard",
                            min_rows_refresh=min_rows_refresh,
                        )
                    else:
                        update_universe(
                            codes=target_codes,
                            full_refresh=full_refresh,
                            use_listed_master=universe_source == "listed_all",
                            market_filter="growth" if universe_source == "growth" else "prime_standard",
                            min_rows_refresh=min_rows_refresh,
                        )
                    try:
                        update_topix(full_refresh=full_refresh)
                    except Exception as exc:
                        st.warning(f"TOPIX の更新に失敗しました: {exc}")
                st.success("一括更新が完了しました。")
                st.rerun()
            except Exception as exc:  # ユーザー向けに簡易表示
                st.error(f"一括更新に失敗しました: {exc}")

    start_date = st.sidebar.date_input("開始日", value=default_start)
    end_date = st.sidebar.date_input("終了日", value=default_end)

    if st.sidebar.button("J-Quantsからダウンロード"):
        try:
            if start_date > end_date:
                raise ValueError("終了日は開始日以降にしてください。")

            start_requested = start_date.isoformat()
            end_requested = end_date.isoformat()
            request_start, request_end, adjusted = enforce_light_plan_window(
                start_requested, end_requested, LIGHT_PLAN_YEARS
            )
            if adjusted:
                if request_start != start_requested:
                    st.sidebar.info(
                        f"ライトプランの制限に合わせて開始日を {request_start} に調整しました。"
                    )
                if request_end != end_requested:
                    st.sidebar.info(
                        f"取得可能な最新日付 {request_end} までに終了日を調整しました。"
                    )

            fetch_and_save_price_csv(
                download_symbol.strip(),
                request_start,
                request_end,
            )
            try:
                update_topix(full_refresh=False)
            except Exception as exc:
                st.sidebar.warning(f"TOPIX の更新に失敗しました: {exc}")
            st.sidebar.success("ダウンロードに成功しました。")
            st.rerun()
        except Exception as exc:  # broad catch for user feedback
            st.sidebar.error(f"ダウンロードに失敗しました: {exc}")

    symbols = _load_available_symbols(price_csv_cache_key)
    try:
        listed_master_cache_key = _get_listed_master_cache_key()
        listed_df = _load_listed_master_cached(listed_master_cache_key)
    except Exception as exc:  # master が無い場合でも動作継続
        st.sidebar.warning(f"銘柄マスタの読み込みに失敗しました: {exc}")
        listed_df = pd.DataFrame(columns=["code", "name", "market"])

    name_map = {
        str(row.code).zfill(4): str(row.name)
        for row in listed_df.itertuples(index=False)
        if getattr(row, "name", None) is not None
    }
    if not symbols:
        st.sidebar.warning("data/price_csv にCSVファイルがありません。")
        st.info("先に data/price_csv に株価CSVを置くか、J-Quantsからダウンロードしてください。")
        return

    query_symbol = st.query_params.get("symbol")
    if isinstance(query_symbol, list):
        query_symbol = query_symbol[0] if query_symbol else None
    if query_symbol and query_symbol in symbols:
        if st.session_state.get("selected_symbol") != query_symbol:
            st.session_state["selected_symbol"] = query_symbol

    selected_symbol = st.sidebar.selectbox(
        "銘柄",
        symbols,
        format_func=lambda c: f"{c} ({name_map.get(c, '名称未登録')})",
        key="selected_symbol",
    )

    lookback = st.sidebar.number_input(
        "過去本数 (N)",
        min_value=20,
        max_value=300,
        value=DEFAULT_LOOKBACK_BARS,
        step=5,
    )

    timeframe = st.sidebar.radio(
        "足種別",
        options=["daily", "weekly", "monthly"],
        format_func=lambda x: {"daily": "日足", "weekly": "週足", "monthly": "月足"}[x],
        index=0,
    )

    chart_type = st.sidebar.radio(
        "チャートタイプ",
        options=["line", "candlestick", "pnf"],
        format_func=lambda x: {
            "line": "ライン", "candlestick": "ローソク足", "pnf": "ポイント・アンド・フィギュア"
        }[x],
        index=0,
    )

    show_volume = st.sidebar.checkbox("出来高", value=True)

    st.sidebar.markdown("---")
    st.sidebar.write("テクニカル表示")
    ma_type = st.sidebar.selectbox(
        "移動平均の種類",
        options=["SMA", "EMA"],
        index=0,
        help="EMA(指数移動平均)は直近の価格に重みを置いて計算します。",
    )
    show_ma_short = st.sidebar.checkbox("短期移動平均", value=True)
    ma_short_period = st.sidebar.number_input(
        "短期期間",
        min_value=2,
        max_value=200,
        value=20,
        step=1,
    )
    show_ma_mid = st.sidebar.checkbox("中期移動平均", value=False)
    ma_mid_period = st.sidebar.number_input(
        "中期期間",
        min_value=5,
        max_value=300,
        value=50,
        step=1,
    )
    show_ma_long = st.sidebar.checkbox("長期移動平均", value=False)
    ma_long_period = st.sidebar.number_input(
        "長期期間",
        min_value=10,
        max_value=400,
        value=200,
        step=1,
    )
    show_bbands = st.sidebar.checkbox("ボリンジャーバンド", value=False)
    show_ichimoku = st.sidebar.checkbox("一目均衡表", value=False)
    st.sidebar.write("チャート分析ライン")
    show_trendlines = st.sidebar.checkbox("支持線/抵抗線", value=False)
    show_head_shoulders = st.sidebar.checkbox("ヘッド＆ショルダーズ", value=False)
    extrema_order = st.sidebar.slider(
        "ライン検出の感度(局所判定幅)",
        min_value=2,
        max_value=6,
        value=3,
        help="値が大きいほど極値の数が減り、ラインが安定します。",
    )

    st.sidebar.write("オシレーター")
    show_rsi = st.sidebar.checkbox("RSI (14)", value=True)
    show_macd = st.sidebar.checkbox("MACD (12,26,9)", value=True)
    show_stoch = st.sidebar.checkbox("ストキャスティクス (14,3)", value=False)
    show_obv = st.sidebar.checkbox("オンバランスボリューム", value=False)
    show_topix_rs = st.sidebar.checkbox(
        "TOPIX RSを表示",
        value=False,
        help="TOPIXとの相対力(RS=株価/ TOPIX)をオシレーター領域に描画します。",
    )
    st.sidebar.write("シグナル")
    show_new_high = st.sidebar.checkbox("新高値を表示", value=False)
    new_high_lookback = st.sidebar.number_input(
        "新高値の判定期間",
        min_value=20,
        max_value=400,
        value=252,
        step=10,
        help="直近N本で高値更新したタイミングを抽出します。",
    )
    show_selling_climax = st.sidebar.checkbox("セリングクライマックスを表示", value=False)
    selling_volume_lookback = st.sidebar.number_input(
        "出来高平均期間",
        min_value=5,
        max_value=60,
        value=20,
        step=1,
    )
    selling_volume_multiplier = st.sidebar.number_input(
        "出来高倍率 (平均比)",
        min_value=1.0,
        max_value=10.0,
        value=2.0,
        step=0.1,
    )
    selling_drop_pct = st.sidebar.number_input(
        "下落率しきい値 (%)",
        min_value=0.5,
        max_value=20.0,
        value=4.0,
        step=0.5,
    )
    selling_close_position = st.sidebar.slider(
        "終値が安値寄りの割合",
        min_value=0.0,
        max_value=1.0,
        value=0.3,
        step=0.05,
        help="0.0は安値引け、1.0は高値引け。",
    )

    tab_chart, tab_screen, tab_pair, tab_backtest, tab_breadth = st.tabs(
        ["チャート表示", "スクリーナー", "ペアトレード", "バックテスト", "マーケットブレッドス"]
    )

    with tab_chart:
        # --- チャート表示 ---
        topix_cache_key = _get_price_cache_key("topix")

        with st.expander("4x4グリッド表示", expanded=False):
            st.caption("全銘柄の週出来高上位1/4に該当する銘柄を週足で表示します。")
            grid_symbols = symbols
            if "grid_compute" not in st.session_state:
                st.session_state["grid_compute"] = False
            if st.button("計算を開始", key="grid_compute_start"):
                st.session_state["grid_compute"] = True

            if grid_symbols and st.session_state["grid_compute"]:
                cache_key = _get_price_csv_dir_cache_key()
                weekly_volume_map, weekly_df_map = _compute_weekly_volume_map(
                    grid_symbols, cache_key
                )
                weekly_volumes = list(weekly_volume_map.values())

                if weekly_volumes:
                    threshold = pd.Series(weekly_volumes).quantile(0.75)
                    filtered_symbols = [
                        symbol
                        for symbol in grid_symbols
                        if weekly_volume_map.get(symbol, 0) >= threshold
                    ]
                else:
                    filtered_symbols = []

                filtered_symbols = sorted(
                    filtered_symbols,
                    key=lambda s: weekly_volume_map.get(s, 0),
                    reverse=True,
                )
                total_symbols = len(filtered_symbols)
                total_pages = max(1, (total_symbols + 15) // 16)
                if "grid_page" not in st.session_state:
                    st.session_state["grid_page"] = 0
                st.session_state["grid_page"] = min(
                    st.session_state["grid_page"], total_pages - 1
                )

                nav_cols = st.columns([1, 2, 1])
                with nav_cols[0]:
                    if st.button("前の16銘柄", disabled=st.session_state["grid_page"] == 0):
                        st.session_state["grid_page"] -= 1
                        st.rerun()
                with nav_cols[1]:
                    st.markdown(
                        f"**{st.session_state['grid_page'] + 1}/{total_pages} ページ**"
                    )
                    st.caption(f"全{total_symbols}銘柄のうち上位を表示")
                with nav_cols[2]:
                    if st.button(
                        "次の16銘柄",
                        disabled=st.session_state["grid_page"] >= total_pages - 1,
                    ):
                        st.session_state["grid_page"] += 1
                        st.rerun()

                start_idx = st.session_state["grid_page"] * 16
                grid_symbols = filtered_symbols[start_idx:start_idx + 16]

                if not grid_symbols:
                    st.info("週出来高上位1/4に該当する銘柄がありません。")
                    grid_symbols = []

                if grid_symbols:
                    rows = [grid_symbols[i:i + 4] for i in range(0, len(grid_symbols), 4)]
                    for row_symbols in rows:
                        cols = st.columns(4)
                        for col_idx, symbol in enumerate(row_symbols):
                            with cols[col_idx]:
                                symbol_name = name_map.get(symbol, "")
                                weekly_df = weekly_df_map.get(symbol)
                                if weekly_df is None:
                                    st.caption(f"{symbol} のデータ取得に失敗しました。")
                                    continue
                                st.markdown(f"[{symbol} {symbol_name}](?symbol={symbol})")
                                fig = _build_mini_chart_from_resampled(
                                    weekly_df, symbol, symbol_name, lookback
                                )
                                if fig is None:
                                    st.caption(f"{symbol} のチャートを表示できません。")
                                    continue
                                st.plotly_chart(fig, use_container_width=True)
            elif grid_symbols:
                st.info("「計算を開始」を押すと週出来高の集計を開始します。")

        cache_key = _get_price_cache_key(selected_symbol)
        daily_lookback = _estimate_daily_lookback(lookback, timeframe)
        df_daily_trading, removed_rows, topix_info = _load_price_with_indicators(
            selected_symbol,
            cache_key,
            topix_cache_key,
            lookback=daily_lookback,
            compute_indicators=False,
        )
        if removed_rows > 0:
            st.sidebar.info(
                f"出来高が0の日を{removed_rows}日分除外して日足チャートを表示します。"
            )

        if topix_info:
            coverage = float(topix_info.get("coverage_ratio", 1.0))
            status = topix_info.get("status")
            if status == "missing_file":
                st.sidebar.info("TOPIX CSVが見つからないためRS計算はスキップします。")
            elif status == "empty_merge":
                st.warning("TOPIXと重複する日付がなく、RSを計算できませんでした。")
            elif status == "invalid_file":
                st.warning(f"topix.csv の読み込みに失敗しました: {topix_info.get('message')}")
            elif coverage < 0.5:
                st.warning(
                    f"TOPIXデータの欠損が多いため、重複日付のみ({coverage*100:.1f}%の範囲)でRSを描画します。"
                )
            elif coverage < 0.8:
                st.info(
                    f"TOPIXデータに欠損があるため、重複する期間のみ({coverage*100:.1f}%)でRSを描画します。"
                )

        df_resampled = _resample_ohlc(df_daily_trading, timeframe)

        if df_resampled.empty:
            st.warning("表示できるデータがありません。先にデータを取得してください。")
            return

        lookback_window = min(lookback, len(df_resampled))
        df_problem = df_resampled.tail(lookback_window).copy()
        df_problem = _compute_indicators(df_problem)
        df_problem["ma_short"] = _compute_moving_average(
            df_problem["close"], int(ma_short_period), ma_type
        )
        df_problem["ma_mid"] = _compute_moving_average(
            df_problem["close"], int(ma_mid_period), ma_type
        )
        df_problem["ma_long"] = _compute_moving_average(
            df_problem["close"], int(ma_long_period), ma_type
        )
        df_problem["date_str"] = df_problem["date"].dt.strftime("%y/%m/%d")
        trading_rangebreaks = _build_trading_rangebreaks(df_problem["date"])

        title_name = name_map.get(selected_symbol, "")
        title = f"チャート（{selected_symbol} {title_name}）" if title_name else f"チャート（{selected_symbol}）"
        st.subheader(title)

        volume_applicable = chart_type != "pnf"
        if chart_type != "pnf":
            rows = 2 if show_volume else 1
            price_fig = make_subplots(
                rows=rows,
                cols=1,
                shared_xaxes=True,
                vertical_spacing=0.05 if show_volume else 0.08,
                row_heights=[0.7, 0.3] if show_volume else [1.0],
            )
        else:
            price_fig = go.Figure()

        if chart_type == "candlestick":
            price_fig.add_trace(
                go.Candlestick(
                    x=df_problem["date"],
                    open=df_problem["open"],
                    high=df_problem["high"],
                    low=df_problem["low"],
                    close=df_problem["close"],
                    name="ローソク足",
                ),
                row=1,
                col=1,
            )
        elif chart_type == "pnf":
            pnf_df = _point_and_figure(df_problem)
            if pnf_df.empty:
                st.warning("ポイント・アンド・フィギュアを描画するデータが不足しています。")
            else:
                if show_volume:
                    st.info("ポイント・アンド・フィギュア選択時は出来高表示を省略します。")
                price_fig.add_trace(
                    go.Scatter(
                        x=pnf_df["col"],
                        y=pnf_df["price"],
                        mode="markers",
                        name="ポイント・アンド・フィギュア",
                        marker=dict(
                            size=14,
                            symbol="square",
                            color=pnf_df["type"].map({"X": "#d62728", "O": "#1f77b4"}),
                            line=dict(width=1, color="#333333"),
                        ),
                    )
                )
                price_fig.update_layout(xaxis_title="列", yaxis_title="価格")
        else:
            price_fig.add_trace(
                go.Scatter(
                    x=df_problem["date"],
                    y=df_problem["close"],
                    name="Close",
                ),
                row=1,
                col=1,
            )

        if chart_type != "pnf":
            if show_ma_short and df_problem["ma_short"].notna().any():
                price_fig.add_trace(
                    go.Scatter(
                        x=df_problem["date"],
                        y=df_problem["ma_short"],
                        name=f"{ma_type} 短期({int(ma_short_period)})",
                        line=dict(dash="dash"),
                    ),
                    row=1,
                    col=1,
                )
            if show_ma_mid and df_problem["ma_mid"].notna().any():
                price_fig.add_trace(
                    go.Scatter(
                        x=df_problem["date"],
                        y=df_problem["ma_mid"],
                        name=f"{ma_type} 中期({int(ma_mid_period)})",
                        line=dict(dash="dot"),
                    ),
                    row=1,
                    col=1,
                )
            if show_ma_long and df_problem["ma_long"].notna().any():
                price_fig.add_trace(
                    go.Scatter(
                        x=df_problem["date"],
                        y=df_problem["ma_long"],
                        name=f"{ma_type} 長期({int(ma_long_period)})",
                        line=dict(color="#7f7f7f", dash="dash"),
                    ),
                    row=1,
                    col=1,
                )
            if show_bbands and df_problem["bb_upper_1"].notna().any():
                price_fig.add_trace(
                    go.Scatter(
                        x=df_problem["date"],
                        y=df_problem["bb_upper_1"],
                        name="BB +1σ",
                        line=dict(color="rgba(150,150,150,0.4)", dash="dot"),
                    ),
                    row=1,
                    col=1,
                )
                price_fig.add_trace(
                    go.Scatter(
                        x=df_problem["date"],
                        y=df_problem["bb_lower_1"],
                        name="BB -1σ",
                        line=dict(color="rgba(150,150,150,0.4)", dash="dot"),
                    ),
                    row=1,
                    col=1,
                )
                price_fig.add_trace(
                    go.Scatter(
                        x=df_problem["date"],
                        y=df_problem["bb_upper_2"],
                        name="BB +2σ",
                        line=dict(color="rgba(180,180,180,0.6)", dash="dot"),
                    ),
                    row=1,
                    col=1,
                )
                price_fig.add_trace(
                    go.Scatter(
                        x=df_problem["date"],
                        y=df_problem["bb_lower_2"],
                        name="BB -2σ",
                        line=dict(color="rgba(180,180,180,0.6)", dash="dot"),
                        fill="tonexty",
                        fillcolor="rgba(180,180,180,0.1)",
                    ),
                    row=1,
                    col=1,
                )
                price_fig.add_trace(
                    go.Scatter(
                        x=df_problem["date"],
                        y=df_problem["bb_upper_3"],
                        name="BB +3σ",
                        line=dict(color="rgba(120,120,120,0.35)", dash="dot"),
                    ),
                    row=1,
                    col=1,
                )
                price_fig.add_trace(
                    go.Scatter(
                        x=df_problem["date"],
                        y=df_problem["bb_lower_3"],
                        name="BB -3σ",
                        line=dict(color="rgba(120,120,120,0.35)", dash="dot"),
                    ),
                    row=1,
                    col=1,
                )

            if show_ichimoku and df_problem["ichimoku_tenkan"].notna().any():
                price_fig.add_trace(
                    go.Scatter(
                        x=df_problem["date"],
                        y=df_problem["ichimoku_tenkan"],
                        name="転換線",
                        line=dict(color="#ff7f0e", width=1.5),
                    ),
                    row=1,
                    col=1,
                )
                price_fig.add_trace(
                    go.Scatter(
                        x=df_problem["date"],
                        y=df_problem["ichimoku_kijun"],
                        name="基準線",
                        line=dict(color="#1f77b4", width=1.5),
                    ),
                    row=1,
                    col=1,
                )
                price_fig.add_trace(
                    go.Scatter(
                        x=df_problem["date"],
                        y=df_problem["ichimoku_chikou"],
                        name="遅行線",
                        line=dict(color="#2ca02c", width=1.2, dash="dot"),
                    ),
                    row=1,
                    col=1,
                )
                price_fig.add_trace(
                    go.Scatter(
                        x=df_problem["date"],
                        y=df_problem["ichimoku_senkou_b"],
                        name="先行スパンB",
                        line=dict(color="rgba(140,86,75,0.6)", width=1),
                    ),
                    row=1,
                    col=1,
                )
                price_fig.add_trace(
                    go.Scatter(
                        x=df_problem["date"],
                        y=df_problem["ichimoku_senkou_a"],
                        name="先行スパンA",
                        line=dict(color="rgba(44,160,44,0.6)", width=1),
                        fill="tonexty",
                        fillcolor="rgba(44,160,44,0.15)",
                    ),
                    row=1,
                    col=1,
                    )

            if show_trendlines:
                line_info = _compute_support_resistance(
                    df_problem, order=extrema_order, max_points=5
                )
                support_line = line_info.get("support")
                resistance_line = line_info.get("resistance")
                if support_line:
                    price_fig.add_trace(
                        _build_trendline_trace(
                            df_problem,
                            support_line[0],
                            support_line[1],
                            "支持線(トレンドライン)",
                            "#2ca02c",
                            dash="dash",
                        ),
                        row=1,
                        col=1,
                    )
                if resistance_line:
                    price_fig.add_trace(
                        _build_trendline_trace(
                            df_problem,
                            resistance_line[0],
                            resistance_line[1],
                            "抵抗線(トレンドライン)",
                            "#d62728",
                            dash="dash",
                        ),
                        row=1,
                        col=1,
                    )

            signal_rows = []
            selling_info = None
            selling_flags = None
            if show_selling_climax:
                selling_info = _detect_selling_climax(
                    df_problem,
                    volume_lookback=int(selling_volume_lookback),
                    volume_multiplier=float(selling_volume_multiplier),
                    drop_pct=float(selling_drop_pct) / 100,
                    close_position=float(selling_close_position),
                )
                selling_flags = selling_info["flag"]
                if selling_flags.any():
                    price_fig.add_trace(
                        go.Scatter(
                            x=df_problem.loc[selling_flags, "date"],
                            y=df_problem.loc[selling_flags, "low"],
                            mode="markers",
                            name="セリングクライマックス",
                            marker=dict(
                                color="#d62728",
                                size=10,
                                symbol="triangle-down",
                                line=dict(color="#333333", width=1),
                            ),
                        ),
                        row=1,
                        col=1,
                    )
                    for idx, row in df_problem.loc[selling_flags].iterrows():
                        info = selling_info.loc[idx]
                        signal_rows.append(
                            {
                                "日付": row["date"],
                                "シグナル": "セリングクライマックス",
                                "終値": row["close"],
                                "下落率%": round(info["drop_pct"] * 100, 2)
                                if pd.notna(info["drop_pct"])
                                else None,
                                "出来高倍率": round(info["volume_ratio"], 2)
                                if pd.notna(info["volume_ratio"])
                                else None,
                            }
                        )

            new_high_flags = None
            if show_new_high:
                new_high_flags = _detect_new_highs(df_problem, int(new_high_lookback))
                if new_high_flags.any():
                    price_fig.add_trace(
                        go.Scatter(
                            x=df_problem.loc[new_high_flags, "date"],
                            y=df_problem.loc[new_high_flags, "high"],
                            mode="markers",
                            name="新高値",
                            marker=dict(
                                color="#2ca02c",
                                size=10,
                                symbol="triangle-up",
                                line=dict(color="#333333", width=1),
                            ),
                        ),
                        row=1,
                        col=1,
                    )
                    for idx, row in df_problem.loc[new_high_flags].iterrows():
                        signal_rows.append(
                            {
                                "日付": row["date"],
                                "シグナル": f"新高値({int(new_high_lookback)})",
                                "終値": row["close"],
                                "下落率%": None,
                                "出来高倍率": None,
                            }
                        )

            if show_head_shoulders:
                pattern = _detect_head_and_shoulders(
                    df_problem, order=extrema_order, shoulder_tolerance=0.08
                )
                if pattern:
                    marker_x = [
                        df_problem["date"].iloc[pattern["left"]],
                        df_problem["date"].iloc[pattern["head"]],
                        df_problem["date"].iloc[pattern["right"]],
                    ]
                    marker_y = [
                        df_problem["high"].iloc[pattern["left"]],
                        df_problem["high"].iloc[pattern["head"]],
                        df_problem["high"].iloc[pattern["right"]],
                    ]
                    price_fig.add_trace(
                        go.Scatter(
                            x=marker_x,
                            y=marker_y,
                            mode="markers+text",
                            name="ヘッド＆ショルダーズ",
                            text=["左肩", "ヘッド", "右肩"],
                            textposition="top center",
                            marker=dict(color="#9467bd", size=10, symbol="triangle-up"),
                        ),
                        row=1,
                        col=1,
                    )
                    neckline_x = [
                        df_problem["date"].iloc[pattern["trough_left"]],
                        df_problem["date"].iloc[pattern["trough_right"]],
                    ]
                    neckline_y = [
                        df_problem["low"].iloc[pattern["trough_left"]],
                        df_problem["low"].iloc[pattern["trough_right"]],
                    ]
                    price_fig.add_trace(
                        go.Scatter(
                            x=neckline_x,
                            y=neckline_y,
                            mode="lines",
                            name="ネックライン",
                            line=dict(color="#9467bd", dash="dot", width=2),
                        ),
                        row=1,
                        col=1,
                    )

            if show_volume:
                price_fig.add_trace(
                    go.Bar(
                        x=df_problem["date"],
                        y=df_problem["volume"],
                        name="出来高",
                        marker_color="rgba(100,149,237,0.6)",
                        opacity=0.7,
                    ),
                    row=2 if volume_applicable else 1,
                    col=1,
                )

            price_fig.update_yaxes(title_text="価格", row=1, col=1)
            if show_volume:
                price_fig.update_yaxes(title_text="出来高", row=2 if volume_applicable else 1, col=1)

            price_fig.update_layout(
                xaxis_title="Date",
                margin=dict(l=10, r=10, t=40, b=10),
            )
            price_fig.update_xaxes(
                tickformat="%y/%m/%d",
                nticks=6,
                rangebreaks=trading_rangebreaks,
            )

        if chart_type == "pnf":
            price_fig.update_layout(margin=dict(l=10, r=10, t=40, b=10))

        st.plotly_chart(price_fig, use_container_width=True)

        osc_fig = make_subplots(specs=[[{"secondary_y": True}]])
        if show_rsi:
            osc_fig.add_trace(
                go.Scatter(
                    x=df_problem["date"],
                    y=df_problem["rsi14"],
                    name="RSI 14",
                    line=dict(color="#2ca02c"),
                ),
                secondary_y=False,
            )
            for level in (30, 50, 70):
                osc_fig.add_hline(y=level, line=dict(color="#cccccc", width=1, dash="dash"))
        if show_stoch:
            osc_fig.add_trace(
                go.Scatter(
                    x=df_problem["date"],
                    y=df_problem["stoch_k"],
                    name="%K",
                    line=dict(color="#1f77b4"),
                ),
                secondary_y=False,
            )
            osc_fig.add_trace(
                go.Scatter(
                    x=df_problem["date"],
                    y=df_problem["stoch_d"],
                    name="%D",
                    line=dict(color="#ff7f0e", dash="dot"),
                ),
                secondary_y=False,
            )
        if show_macd:
            osc_fig.add_trace(
                go.Bar(
                    x=df_problem["date"],
                    y=df_problem["macd_hist"],
                    name="MACD Hist",
                    marker_color="rgba(100,100,100,0.4)",
                ),
                secondary_y=True,
            )
            osc_fig.add_trace(
                go.Scatter(
                    x=df_problem["date"],
                    y=df_problem["macd"],
                    name="MACD",
                    line=dict(color="#d62728"),
                ),
                secondary_y=True,
            )
            osc_fig.add_trace(
                go.Scatter(
                    x=df_problem["date"],
                    y=df_problem["macd_signal"],
                    name="Signal",
                    line=dict(color="#9467bd", dash="dash"),
                ),
                secondary_y=True,
            )
        if show_obv:
            osc_fig.add_trace(
                go.Scatter(
                    x=df_problem["date"],
                    y=df_problem["obv"],
                    name="OBV",
                    line=dict(color="#8c564b"),
                ),
                secondary_y=True,
            )
        if show_topix_rs:
            if "topix_rs" in df_problem.columns:
                osc_fig.add_trace(
                    go.Scatter(
                        x=df_problem["date"],
                        y=df_problem["topix_rs"],
                        name="TOPIX RS",
                        line=dict(color="#17becf", dash="solid"),
                    ),
                    secondary_y=True,
                )
            else:
                st.info("TOPIX RS を計算できないため表示をスキップしました。")

        osc_fig.update_layout(
            xaxis_title="Date",
            yaxis_title="RSI / Stoch",
            margin=dict(l=10, r=10, t=40, b=10),
            legend=dict(orientation="h"),
        )
        osc_fig.update_xaxes(
            tickformat="%y/%m/%d",
            nticks=6,
            rangebreaks=trading_rangebreaks,
        )
        osc_fig.update_yaxes(title_text="MACD / RS", secondary_y=True)
        st.plotly_chart(osc_fig, use_container_width=True)

        if chart_type != "pnf" and (show_selling_climax or show_new_high):
            if signal_rows:
                signal_df = pd.DataFrame(signal_rows)
                signal_df = signal_df.sort_values("日付", ascending=False)
                st.subheader("直近シグナル一覧")
                st.dataframe(
                    signal_df.head(20),
                    use_container_width=True,
                )
            else:
                st.info("指定された期間内にシグナルは検出されませんでした。")

    with tab_screen:
        st.subheader("テクニカル・スクリーナー")
        st.caption("日足データを使い、直近のMACDゴールデンクロスやRSI帯などで抽出します。")

        if "screening_results" not in st.session_state:
            st.session_state["screening_results"] = None
        if "macd_debug_logs" not in st.session_state:
            st.session_state["macd_debug_logs"] = []

        target_markets = st.multiselect(
            "市場 (空欄なら全て)",
            options=sorted(listed_df["market"].dropna().unique()),
        )
        apply_rsi_condition = st.checkbox("RSI 条件を適用", value=True)
        rsi_range = st.slider("RSI(14) 範囲", min_value=0, max_value=100, value=(40, 65))
        apply_topix_rs_condition = st.checkbox(
            "TOPIX RS 条件を適用",
            value=False,
            help="TOPIXとの相対力を評価します。TOPIXデータがない場合は自動的にスキップされます。",
        )
        topix_rs_lookback = st.slider(
            "RSの比較期間（日）",
            min_value=5,
            max_value=120,
            value=20,
            help="最新のRSが過去の値よりどの程度変化したかを判定します。",
        )
        topix_rs_threshold = st.slider(
            "RS変化率の下限(%)",
            min_value=-50,
            max_value=50,
            value=0,
            help="0%以上ならTOPIXより直近でアウトパフォームしている銘柄を抽出します。",
        )
        macd_condition = st.selectbox(
            "MACD クロス条件",
            options=["none", "golden", "dead"],
            format_func=lambda v: {
                "none": "条件なし",
                "golden": "直近でゴールデンクロス",
                "dead": "直近でデッドクロス",
            }[v],
            index=1,
        )
        macd_lookback = st.slider(
            "MACDクロスを探す過去本数",
            min_value=1,
            max_value=30,
            value=5,
            help="クロスを検出する期間を広げることで、直近数日以内に発生したサインを拾いやすくします。",
        )
        macd_debug = st.checkbox(
            "MACDクロス判定のデバッグログを表示",
            value=False,
            help="判定ループの値を全銘柄分出力します（重くなる場合があります）",
        )
        show_detailed_log = st.checkbox(
            "スクリーニング詳細ログを表示",
            value=False,
            help="各銘柄がどの条件で除外されたかを一覧で確認できます。MACDデバッグとは独立して動作します。",
        )
        require_sma20_trend = st.checkbox("終値 > SMA20 かつ SMA20が上向き", value=True)
        sma_trend_lookback = st.slider("SMA20上向きの判定幅（日）", 1, 10, value=3)
        apply_volume_condition = st.checkbox("出来高条件を適用", value=True)
        volume_multiplier = st.number_input(
            "出来高/20日平均の下限 (倍)",
            min_value=0.0,
            max_value=10.0,
            value=0.8,
            step=0.1,
            help="条件を外したい場合はチェックを外してください。0.0 を指定した場合は出来高が取得できる銘柄のみ合格します。",
        )
        apply_canslim_condition = st.checkbox("CAN-SLIMパターン条件を適用", value=False)
        canslim_recent_days = st.slider(
            "CAN-SLIMのブレイクアウト判定期間（日）",
            min_value=5,
            max_value=120,
            value=30,
            help="指定日数以内にカップ/ソーサーウィズハンドルのブレイクアウトがある銘柄を抽出します。",
        )
        canslim_cup_window = st.number_input(
            "カップ判定期間（日）",
            min_value=20,
            max_value=120,
            value=50,
            step=1,
        )
        canslim_saucer_window = st.number_input(
            "ソーサー判定期間（日）",
            min_value=40,
            max_value=180,
            value=80,
            step=1,
        )
        canslim_handle_window = st.number_input(
            "ハンドル判定期間（日）",
            min_value=5,
            max_value=30,
            value=10,
            step=1,
        )
        canslim_volume_multiplier = st.number_input(
            "CAN-SLIM出来高/20日平均の下限 (倍)",
            min_value=0.0,
            max_value=10.0,
            value=1.5,
            step=0.1,
            help="最適化で得られた値をここに入力してスクリーニングへ反映できます。",
        )
        apply_weekly_volume_quartile = st.checkbox(
            "週出来高上位1/4を抽出",
            value=False,
            help="最新の週出来高がユニバースの上位25%に入る銘柄を抽出します。",
        )
        st.markdown("**シグナル抽出**")
        screen_new_high = st.checkbox("新高値シグナルを抽出", value=False)
        screen_selling_climax = st.checkbox("セリングクライマックスを抽出", value=False)
        signal_lookback_days = st.slider(
            "シグナル判定の過去日数",
            min_value=1,
            max_value=60,
            value=10,
            help="直近N日以内にシグナルがあった銘柄を抽出します。",
        )
        signal_new_high_lookback = st.number_input(
            "新高値の判定期間",
            min_value=20,
            max_value=400,
            value=252,
            step=10,
        )
        signal_selling_volume_lookback = st.number_input(
            "セリング出来高平均期間",
            min_value=5,
            max_value=60,
            value=20,
            step=1,
        )
        signal_selling_volume_multiplier = st.number_input(
            "セリング出来高倍率 (平均比)",
            min_value=1.0,
            max_value=10.0,
            value=2.0,
            step=0.1,
        )
        signal_selling_drop_pct = st.number_input(
            "セリング下落率しきい値 (%)",
            min_value=0.5,
            max_value=20.0,
            value=4.0,
            step=0.5,
        )
        signal_selling_close_position = st.slider(
            "セリング終値が安値寄りの割合",
            min_value=0.0,
            max_value=1.0,
            value=0.3,
            step=0.05,
        )
        topix_cache_key = _get_price_cache_key("topix")

        run_screening = st.button("スクリーニングを実行", type="primary")

        screening_results = st.session_state.get("screening_results")
        if run_screening:
            with st.spinner("スクリーニングを実行しています..."):
                screening_results = []
                macd_debug_logs = [] if macd_debug else None
                failure_logs = []
                reason_counter = Counter()
                weekly_volume_map = {}
                weekly_volume_threshold = None

                def _market_filter(code_str: str) -> bool:
                    if not target_markets:
                        return True
                    code_market = listed_df.loc[listed_df["code"] == code_str, "market"]
                    return not code_market.empty and code_market.iloc[0] in target_markets

                if apply_weekly_volume_quartile:
                    weekly_volumes = []
                    for code in symbols:
                        code_str = str(code).strip().zfill(4)
                        if not _market_filter(code_str):
                            continue
                        cache_key = _get_price_cache_key(code_str)
                        try:
                            df_ind_full, _, _ = _load_price_with_indicators(
                                code_str, cache_key, topix_cache_key
                            )
                        except Exception:
                            continue
                        weekly_df = _resample_ohlc(df_ind_full, "weekly")
                        if weekly_df.empty:
                            continue
                        latest_weekly_vol = weekly_df.iloc[-1]["volume"]
                        if pd.notna(latest_weekly_vol):
                            weekly_volume_map[code_str] = float(latest_weekly_vol)
                            weekly_volumes.append(float(latest_weekly_vol))
                    if weekly_volumes:
                        weekly_volume_threshold = pd.Series(weekly_volumes).quantile(0.75)

                for code in symbols:
                    code_str = str(code).strip().zfill(4)
                    code_reasons = []
                    if not _market_filter(code_str):
                        continue

                    cache_key = _get_price_cache_key(code_str)
                    try:
                        df_ind_full, _, _ = _load_price_with_indicators(
                            code_str, cache_key, topix_cache_key
                        )
                    except Exception:
                        code_reasons.append("データ取得エラー")
                        failure_logs.append(
                            {"code": code_str, "name": name_map.get(code_str, "-"), "reasons": code_reasons}
                        )
                        reason_counter.update(code_reasons)
                        continue

                    required_length, requirement_messages = _calculate_minimum_data_length(
                        apply_rsi_condition=apply_rsi_condition,
                        macd_condition=macd_condition,
                        macd_lookback=macd_lookback,
                        require_sma20_trend=require_sma20_trend,
                        sma_trend_lookback=sma_trend_lookback,
                        apply_volume_condition=apply_volume_condition,
                        apply_topix_rs_condition=apply_topix_rs_condition,
                        topix_rs_lookback=topix_rs_lookback,
                        apply_new_high_signal=screen_new_high,
                        new_high_lookback=int(signal_new_high_lookback),
                        apply_selling_climax_signal=screen_selling_climax,
                        selling_volume_lookback=int(signal_selling_volume_lookback),
                        signal_lookback_days=signal_lookback_days,
                        apply_weekly_volume_quartile=apply_weekly_volume_quartile,
                        apply_canslim_condition=apply_canslim_condition,
                        cup_window=canslim_cup_window,
                        saucer_window=canslim_saucer_window,
                        handle_window=canslim_handle_window,
                    )

                    if len(df_ind_full) < required_length:
                        code_reasons.append("データ不足")
                        failure_logs.append(
                            {
                                "code": code_str,
                                "name": name_map.get(code_str, "-"),
                                "reasons": code_reasons,
                                "details": requirement_messages,
                            }
                        )
                        reason_counter.update(code_reasons)
                        continue

                    if apply_topix_rs_condition and (
                        "topix_rs" not in df_ind_full.columns
                        or df_ind_full["topix_rs"].dropna().empty
                    ):
                        code_reasons.append("TOPIX RSデータなし")
                        failure_logs.append(
                            {"code": code_str, "name": name_map.get(code_str, "-"), "reasons": code_reasons}
                        )
                        reason_counter.update(code_reasons)
                        continue

                    df_ind = df_ind_full.tail(200)
                    latest = df_ind.iloc[-1]
                    if not _latest_has_required_data(
                        latest,
                        apply_rsi_condition=apply_rsi_condition,
                        macd_condition=macd_condition,
                        require_sma20_trend=require_sma20_trend,
                        apply_topix_rs_condition=apply_topix_rs_condition,
                    ):
                        code_reasons.append("最新データに欠損あり")
                        failure_logs.append(
                            {"code": code_str, "name": name_map.get(code_str, "-"), "reasons": code_reasons}
                        )
                        reason_counter.update(code_reasons)
                        continue

                    rsi_ok = True
                    if apply_rsi_condition:
                        rsi_ok = rsi_range[0] <= latest["rsi14"] <= rsi_range[1]
                    macd_log = [] if macd_debug and macd_condition != "none" else None
                    if macd_condition == "golden":
                        macd_ok = _has_macd_cross(
                            df_ind,
                            direction="golden",
                            lookback=macd_lookback,
                            debug_log=macd_log,
                        )
                    elif macd_condition == "dead":
                        macd_ok = _has_macd_cross(
                            df_ind,
                            direction="dead",
                            lookback=macd_lookback,
                            debug_log=macd_log,
                        )
                    else:
                        macd_ok = True

                    if macd_log is not None and macd_debug_logs is not None:
                        macd_debug_logs.append(
                            {
                                "code": code_str,
                                "result": macd_ok,
                                "logs": macd_log,
                            }
                        )

                    sma_ok = True
                    if require_sma20_trend:
                        past_idx = -1 - sma_trend_lookback
                        if abs(past_idx) <= len(df_ind):
                            past_sma = df_ind.iloc[past_idx]["sma20"]
                            sma_ok = (
                                latest["close"] > latest["sma20"]
                                and latest["sma20"] > past_sma
                            )
                        else:
                            sma_ok = False

                    avg_vol20 = df_ind["volume"].tail(20).mean()
                    vol_ok = True
                    if apply_volume_condition:
                        vol_ok = (
                            pd.notna(avg_vol20)
                            and avg_vol20 > 0
                            and latest["volume"] >= avg_vol20 * volume_multiplier
                        )
                    rs_ok = True
                    rs_change = None
                    if apply_topix_rs_condition:
                        if "topix_rs" not in df_ind.columns or df_ind["topix_rs"].isna().all():
                            rs_ok = False
                        elif len(df_ind) > topix_rs_lookback:
                            baseline = df_ind.iloc[-1 - topix_rs_lookback]["topix_rs"]
                            if pd.notna(latest["topix_rs"]) and pd.notna(baseline) and baseline != 0:
                                rs_change = (latest["topix_rs"] / baseline - 1) * 100
                                rs_ok = rs_change >= topix_rs_threshold
                            else:
                                rs_ok = False
                        else:
                            rs_ok = False

                    canslim_ok = True
                    canslim_pattern = None
                    canslim_signal_date = None
                    if apply_canslim_condition:
                        max_window = max(canslim_cup_window, canslim_saucer_window) + canslim_handle_window
                        scan_lookahead = 1
                        scan_window = max_window + scan_lookahead + 5
                        df_canslim = df_ind_full.tail(scan_window)
                        signals = scan_canslim_patterns(
                            df_canslim,
                            lookahead=scan_lookahead,
                            return_threshold=0.0,
                            volume_multiplier=canslim_volume_multiplier,
                            cup_window=canslim_cup_window,
                            saucer_window=canslim_saucer_window,
                            handle_window=canslim_handle_window,
                        )
                        if signals:
                            latest_signal = signals[-1]
                            if latest_signal.date >= latest["date"] - pd.Timedelta(days=canslim_recent_days):
                                canslim_pattern = latest_signal.pattern
                                canslim_signal_date = latest_signal.date
                            else:
                                canslim_ok = False
                        else:
                            canslim_ok = False

                    new_high_ok = True
                    new_high_date = None
                    selling_climax_ok = True
                    selling_climax_date = None
                    if screen_new_high:
                        new_high_flags = _detect_new_highs(
                            df_ind_full, int(signal_new_high_lookback)
                        )
                        if len(df_ind_full) >= signal_lookback_days:
                            recent_flags = new_high_flags.tail(signal_lookback_days)
                        else:
                            recent_flags = new_high_flags
                        if recent_flags.any():
                            last_idx = recent_flags[recent_flags].index[-1]
                            new_high_date = df_ind_full.loc[last_idx, "date"]
                        else:
                            new_high_ok = False
                    if screen_selling_climax:
                        selling_info = _detect_selling_climax(
                            df_ind_full,
                            volume_lookback=int(signal_selling_volume_lookback),
                            volume_multiplier=float(signal_selling_volume_multiplier),
                            drop_pct=float(signal_selling_drop_pct) / 100,
                            close_position=float(signal_selling_close_position),
                        )
                        selling_flags = selling_info["flag"]
                        if len(df_ind_full) >= signal_lookback_days:
                            recent_flags = selling_flags.tail(signal_lookback_days)
                        else:
                            recent_flags = selling_flags
                        if recent_flags.any():
                            last_idx = recent_flags[recent_flags].index[-1]
                            selling_climax_date = df_ind_full.loc[last_idx, "date"]
                        else:
                            selling_climax_ok = False

                    if all(
                        [
                            rsi_ok,
                            macd_ok,
                            sma_ok,
                            vol_ok,
                            rs_ok,
                            canslim_ok,
                            new_high_ok,
                            selling_climax_ok,
                        ]
                    ):
                        weekly_vol_ok = True
                        weekly_volume = None
                    if apply_weekly_volume_quartile:
                        weekly_volume = weekly_volume_map.get(code_str)
                        if weekly_volume_threshold is None or weekly_volume is None:
                            weekly_vol_ok = False
                        else:
                            weekly_vol_ok = weekly_volume >= weekly_volume_threshold

                    if all(
                        [
                            rsi_ok,
                            macd_ok,
                            sma_ok,
                            vol_ok,
                            rs_ok,
                            weekly_vol_ok,
                            new_high_ok,
                            selling_climax_ok,
                        ]
                    ):
                        change_pct = (
                            (latest["close"] - df_ind.iloc[-2]["close"]) / df_ind.iloc[-2]["close"] * 100
                            if len(df_ind) >= 2 and df_ind.iloc[-2]["close"] != 0
                            else None
                        )
                        screening_results.append(
                            {
                                "code": code_str,
                                "name": name_map.get(code_str, "-"),
                                "close": latest["close"],
                                "RSI14": round(latest["rsi14"], 2),
                                "MACD": round(latest["macd"], 3),
                                "Signal": round(latest["macd_signal"], 3),
                                "出来高": int(latest["volume"]),
                                "20日平均出来高": int(avg_vol20) if pd.notna(avg_vol20) else None,
                                "日次騰落率%": round(change_pct, 2) if change_pct is not None else None,
                                "RS(対TOPIX)%": round(rs_change, 2) if rs_change is not None else None,
                                "週出来高": int(weekly_volume) if weekly_volume is not None else None,
                                "CAN-SLIMパターン": canslim_pattern,
                                "CAN-SLIMシグナル日": canslim_signal_date,
                                "新高値シグナル日": new_high_date,
                                "セリングクライマックス日": selling_climax_date,
                            }
                        )
                        continue

                    if apply_rsi_condition and not rsi_ok:
                        code_reasons.append("RSI条件不合格")
                    if macd_condition != "none" and not macd_ok:
                        code_reasons.append("MACD条件不合格")
                    if require_sma20_trend and not sma_ok:
                        code_reasons.append("SMAトレンド不合格")
                    if apply_volume_condition and not vol_ok:
                        code_reasons.append("出来高条件不合格")
                    if apply_topix_rs_condition and not rs_ok:
                        if "topix_rs" not in df_ind.columns:
                            code_reasons.append("TOPIX RSデータなし")
                        elif len(df_ind) <= topix_rs_lookback:
                            code_reasons.append("TOPIX RS期間不足")
                        else:
                            code_reasons.append("TOPIX RS条件不合格")
                    if apply_weekly_volume_quartile and not weekly_vol_ok:
                        if weekly_volume_threshold is None or weekly_volume is None:
                            code_reasons.append("週出来高データ不足")
                        else:
                            code_reasons.append("週出来高上位条件不合格")
                    if apply_canslim_condition and not canslim_ok:
                        code_reasons.append("CAN-SLIM条件不合格")
                    if screen_new_high and not new_high_ok:
                        code_reasons.append("新高値シグナルなし")
                    if screen_selling_climax and not selling_climax_ok:
                        code_reasons.append("セリングクライマックスなし")

                    if code_reasons:
                        failure_logs.append(
                            {"code": code_str, "name": name_map.get(code_str, "-"), "reasons": code_reasons}
                        )
                        reason_counter.update(code_reasons)

                st.session_state["screening_results"] = screening_results
                st.session_state["macd_debug_logs"] = macd_debug_logs or []
                st.session_state["screening_failure_logs"] = failure_logs
                st.session_state["screening_reason_counter"] = reason_counter

        if screening_results is None:
            st.info("条件を設定して『スクリーニングを実行』を押してください。")
        elif screening_results:
            df_result = pd.DataFrame(screening_results)
            df_result = df_result.sort_values("日次騰落率%", ascending=False, na_position="last")
            st.success(f"{len(df_result)} 銘柄が条件に合致しました。")
            st.dataframe(df_result, use_container_width=True)
            enable_preview = st.checkbox(
                "日足チャートプレビューを表示（件数が多い場合は負荷が高くなります）",
                value=False,
            )
            max_preview = st.number_input(
                "プレビュー最大件数",
                min_value=1,
                max_value=50,
                value=10,
                step=1,
                help="プレビューを表示する場合に処理負荷を抑えるための件数上限です。",
            )

            if macd_debug and st.session_state.get("macd_debug_logs"):
                with st.expander("MACDクロス判定のデバッグログ", expanded=False):
                    for entry in st.session_state["macd_debug_logs"]:
                        st.markdown(
                            f"**{entry['code']}** : {'合格' if entry['result'] else '不合格'}"
                        )
                        if entry["logs"]:
                            st.code("\n".join(entry["logs"]), language="text")
                        else:
                            st.caption("ログなし")

            if show_detailed_log and st.session_state.get("screening_reason_counter"):
                with st.expander("不合格理由のサマリと詳細", expanded=False):
                    reason_counter = st.session_state.get("screening_reason_counter", Counter())
                    total_failures = sum(reason_counter.values())
                    if total_failures:
                        summary_lines = [
                            f"{reason}: {count}件 ({count / total_failures * 100:.1f}%)"
                            for reason, count in reason_counter.most_common()
                        ]
                        st.markdown("#### 不合格理由のサマリ")
                        st.markdown("\n".join(f"- {line}" for line in summary_lines))

                    failure_logs = st.session_state.get("screening_failure_logs", [])
                    if failure_logs:
                        st.markdown("#### 不合格銘柄一覧")
                        fail_df = pd.DataFrame(
                            [
                                {
                                    "code": log["code"],
                                    "name": log["name"],
                                    "理由": " / ".join(log["reasons"]),
                                }
                                for log in failure_logs
                            ]
                        )
                        st.dataframe(fail_df, use_container_width=True)

            if enable_preview:
                st.markdown("### 日足チャートプレビュー")
                for _, row in df_result.head(max_preview).iterrows():
                    code_str = str(row["code"]).zfill(4)
                    try:
                        cache_key = _get_price_cache_key(code_str)
                        chart_df, _, _ = _load_price_with_indicators(
                            code_str, cache_key, topix_cache_key
                        )
                    except Exception:
                        continue
                    chart_df = chart_df.tail(90)
                    if chart_df.empty:
                        continue
                    preview_rangebreaks = _build_trading_rangebreaks(chart_df["date"])

                    left, right = st.columns([1, 3])
                    with left:
                        st.markdown(f"**{code_str} {row['name']}**")
                        st.caption("直近90日間の終値推移")
                    with right:
                        preview_fig = go.Figure()
                        preview_fig.add_trace(
                            go.Scatter(
                                x=chart_df["date"],
                                y=chart_df["close"],
                                mode="lines",
                                line=dict(color="#1f77b4"),
                                name="終値",
                            )
                        )
                        preview_fig.update_layout(
                            margin=dict(l=10, r=10, t=10, b=10),
                            height=220,
                            xaxis_title="日付",
                            yaxis_title="終値",
                        )
                        preview_fig.update_xaxes(
                            tickformat="%Y-%m-%d",
                            rangebreaks=preview_rangebreaks,
                        )
                        st.plotly_chart(preview_fig, use_container_width=True)
        else:
            reason_counter = st.session_state.get("screening_reason_counter", Counter())
            total_failures = sum(reason_counter.values())
            if total_failures:
                summary_text = ", ".join(
                    [
                        f"{reason}が全体の{count / total_failures * 100:.1f}%"
                        for reason, count in reason_counter.most_common()
                    ]
                )
                st.info(
                    "条件に合致する銘柄はありませんでした。主な不合格理由: "
                    f"{summary_text}。条件を緩めて再検索してください。"
                )
            else:
                st.info("条件に合致する銘柄はありませんでした。条件を緩めて再検索してください。")

    with tab_pair:
        st.subheader("ペアトレード")
        st.caption("業種フィルタで候補を絞り込み、統計指標とスプレッドの推移を確認します。")
        trading_days_per_year = 252
        pair_search_years = 2
        pair_search_history = trading_days_per_year * pair_search_years

        if "pair_results" not in st.session_state:
            st.session_state["pair_results"] = None
        if "pair_manual_result" not in st.session_state:
            st.session_state["pair_manual_result"] = None
        if "pair_manual_metrics" not in st.session_state:
            st.session_state["pair_manual_metrics"] = None
        if "pair_manual_spread_metrics" not in st.session_state:
            st.session_state["pair_manual_spread_metrics"] = None
        if "pair_manual_cache_hit" not in st.session_state:
            st.session_state["pair_manual_cache_hit"] = False

        cached_pairs_df, cached_meta = _load_pair_cache()

        with st.expander("任意の2銘柄を指定して分析", expanded=True):
            use_cached_metrics = st.checkbox(
                "ペアキャッシュの指標を優先して表示",
                value=True,
                help="キャッシュにあるペアは再計算せず、保存済みの指標を表示します。",
            )
            manual_cols = st.columns(2)
            with manual_cols[0]:
                manual_symbol_a = st.selectbox(
                    "銘柄A",
                    symbols,
                    format_func=lambda c: f"{c} ({name_map.get(c, '名称未登録')})",
                    key="pair_manual_symbol_a",
                )
            with manual_cols[1]:
                manual_symbol_b = st.selectbox(
                    "銘柄B",
                    symbols,
                    format_func=lambda c: f"{c} ({name_map.get(c, '名称未登録')})",
                    key="pair_manual_symbol_b",
                    index=1 if len(symbols) > 1 else 0,
                )

            manual_window_cols = st.columns(2)
            with manual_window_cols[0]:
                manual_recent_window = st.number_input(
                    "直近比較本数(任意ペア)",
                    min_value=20,
                    max_value=pair_search_history,
                    value=60,
                    step=5,
                    key="pair_manual_recent_window",
                )
            with manual_window_cols[1]:
                manual_long_window_input = st.number_input(
                    "長期比較本数(任意ペア, 0で無効)",
                    min_value=0,
                    max_value=pair_search_history,
                    value=240,
                    step=20,
                    key="pair_manual_long_window",
                )

            manual_signal_cols = st.columns(2)
            with manual_signal_cols[0]:
                entry_threshold_manual = st.number_input(
                    "エントリー閾値 (Zスコア)",
                    min_value=0.5,
                    max_value=5.0,
                    value=2.0,
                    step=0.1,
                    key="pair_manual_entry_threshold",
                )
            with manual_signal_cols[1]:
                exit_threshold_manual = st.number_input(
                    "エグジット閾値 (Zスコア)",
                    min_value=0.0,
                    max_value=5.0,
                    value=0.5,
                    step=0.1,
                    key="pair_manual_exit_threshold",
                )

            run_manual_pair = st.button("選択ペアを分析", type="primary", key="pair_manual_run")
            if run_manual_pair:
                if manual_symbol_a == manual_symbol_b:
                    st.warning("異なる銘柄を選択してください。")
                    st.session_state["pair_manual_result"] = pd.DataFrame()
                    st.session_state["pair_manual_metrics"] = None
                    st.session_state["pair_manual_spread_metrics"] = None
                    st.session_state["pair_manual_cache_hit"] = False
                else:
                    with st.spinner("ペア指標を計算しています..."):
                        df_pair, spread_metrics = compute_spread_series(
                            manual_symbol_a, manual_symbol_b
                        )
                        manual_long_window = (
                            int(manual_long_window_input)
                            if manual_long_window_input and manual_long_window_input >= 5
                            else None
                        )
                        cached_metrics = _lookup_cached_pair_metrics(
                            cached_pairs_df, manual_symbol_a, manual_symbol_b
                        )
                        st.session_state["pair_manual_cache_hit"] = (
                            cached_metrics is not None
                        )
                        if use_cached_metrics and cached_metrics is not None:
                            manual_metrics = cached_metrics
                        else:
                            manual_metrics = compute_pair_metrics(
                                manual_symbol_a,
                                manual_symbol_b,
                                recent_window=int(manual_recent_window),
                                long_window=manual_long_window,
                            )
                        st.session_state["pair_manual_result"] = df_pair
                        st.session_state["pair_manual_metrics"] = manual_metrics
                        st.session_state["pair_manual_spread_metrics"] = spread_metrics

            manual_result = st.session_state.get("pair_manual_result")
            if manual_result is not None:
                manual_metrics = st.session_state.get("pair_manual_metrics")
                if st.session_state.get("pair_manual_cache_hit"):
                    st.caption("ペアキャッシュの指標を表示しています。")
                if manual_metrics:
                    st.dataframe(
                        _build_pair_metrics_table(manual_metrics), use_container_width=True
                    )
                else:
                    st.info("指標が算出できませんでした。比較本数やデータ量を確認してください。")
                _render_pair_spread_chart(
                    manual_result,
                    entry_threshold_manual,
                    exit_threshold_manual,
                    st.session_state.get("pair_manual_spread_metrics"),
                )

        sector17_values = (
            sorted(listed_df["sector17"].dropna().astype(str).unique().tolist())
            if "sector17" in listed_df.columns
            else []
        )
        sector33_values = (
            sorted(listed_df["sector33"].dropna().astype(str).unique().tolist())
            if "sector33" in listed_df.columns
            else []
        )

        pair_filters = st.columns([1, 1, 1, 1])
        with pair_filters[0]:
            sector17_choice = st.selectbox(
                "sector17 フィルタ",
                options=["指定なし", *sector17_values],
            )
        with pair_filters[1]:
            sector33_choice = st.selectbox(
                "sector33 フィルタ",
                options=["指定なし", *sector33_values],
            )
        with pair_filters[2]:
            search_scope = st.selectbox(
                "探索範囲",
                options=["セクター別(全銘柄)", "選択銘柄起点(同一セクター)"],
                help="総当たりを避け、セクター内でペア候補を探索します。",
            )
        with pair_filters[3]:
            max_pairs = st.number_input(
                "セクターあたり候補上限",
                min_value=10,
                max_value=300,
                value=80,
                step=5,
            )
        pair_limit_filters = st.columns([1])
        with pair_limit_filters[0]:
            max_pairs_per_symbol = st.number_input(
                "同一銘柄のペア上限(0で無効)",
                min_value=0,
                max_value=10,
                value=3,
                step=1,
            )
        similarity_filters = st.columns([1, 1, 1, 1])
        with similarity_filters[0]:
            recent_window = st.number_input(
                "直近比較本数",
                min_value=20,
                max_value=pair_search_history,
                value=60,
                step=5,
            )
        with similarity_filters[1]:
            long_window_input = st.number_input(
                "長期比較本数(0で無効)",
                min_value=0,
                max_value=pair_search_history,
                value=240,
                step=20,
            )
        with similarity_filters[2]:
            min_similarity = st.slider(
                "直近形状の類似度下限",
                min_value=0.0,
                max_value=1.0,
                value=0.85,
                step=0.05,
                disabled=True,
            )
        with similarity_filters[3]:
            min_long_similarity = st.slider(
                "長期形状の類似度下限",
                min_value=0.0,
                max_value=1.0,
                value=0.8,
                step=0.05,
                disabled=True,
            )
        return_filters = st.columns([1])
        with return_filters[0]:
            min_return_corr = st.slider(
                "直近リターン相関の下限",
                min_value=0.0,
                max_value=1.0,
                value=0.8,
                step=0.05,
                disabled=True,
            )
        cointegration_available = cointegration_test is not None
        if not cointegration_available:
            st.warning(
                "statsmodels が未導入のため、コインテグレーション p値フィルタを無効化しています。"
            )
        stat_filters = st.columns([1, 1, 1])
        with stat_filters[0]:
            max_p_value = 0.1 if cointegration_available else None
            st.number_input(
                "コインテグレーションp値上限（固定: 0.10）",
                min_value=0.1,
                max_value=0.1,
                value=0.1,
                step=0.01,
                format="%.2f",
                disabled=True,
            )
        with stat_filters[1]:
            max_half_life = st.number_input(
                "半減期の上限(日)",
                min_value=1.0,
                max_value=250.0,
                value=30.0,
                step=1.0,
                disabled=True,
            )
        with stat_filters[2]:
            max_abs_zscore = st.number_input(
                "最新Zスコア絶対値の上限(0で無効)",
                min_value=0.5,
                max_value=5.0,
                value=2.5,
                step=0.1,
            )
        volume_filters = st.columns([1])
        with volume_filters[0]:
            min_avg_volume = st.number_input(
                "平均出来高の下限(0で無効)",
                min_value=0.0,
                value=50000.0,
                step=1000.0,
            )
        score_filters = st.columns([1])
        with score_filters[0]:
            preselect_top_n = st.number_input(
                "スコア上位の評価件数",
                min_value=10,
                max_value=500,
                value=150,
                step=10,
                disabled=True,
            )
        st.caption("ペア検索の条件は現存銘柄 + p値 + 平均出来高 + 最新Zスコアのみを使用します。")

        long_window = int(long_window_input) if long_window_input and long_window_input >= 5 else None
        if long_window is None:
            min_long_similarity = None
        min_avg_volume_filter = float(min_avg_volume) if min_avg_volume and min_avg_volume > 0 else None
        max_abs_zscore_filter = (
            float(max_abs_zscore) if max_abs_zscore and max_abs_zscore > 0 else None
        )
        max_pairs_per_symbol_limit = (
            int(max_pairs_per_symbol) if max_pairs_per_symbol and max_pairs_per_symbol > 0 else None
        )
        min_pair_samples = compute_min_pair_samples(int(recent_window), long_window)
        required_samples = max(min_pair_samples, pair_search_history)
        st.caption(
            "必要本数の目安: 直近/長期比較の設定から最低 "
            f"{min_pair_samples} 本が必要です。探索対象は直近{pair_search_years}年"
            f"({pair_search_history}本)に限定されます。"
        )

        with st.expander("指標の説明", expanded=False):
            st.markdown(
                "- p値: コインテグレーション検定の結果。小さいほど統計的にペアの関係が強い。"
            )
            st.markdown("- 半減期: スプレッドが平均に戻るまでの目安期間。短いほど回帰が速い。")
            st.markdown("- β: 2銘柄の回帰係数。ヘッジ比率の目安。")
            st.markdown("- 直近類似度: 直近の価格推移の形状相関。1に近いほど類似。")
            st.markdown("- 直近リターン相関: 直近の対数リターンの相関係数。1に近いほど同期。")
            st.markdown(
                "- 長期類似度/長期リターン相関: 長期窓での形状相関とリターン相関。"
                "長期比較本数を0にすると計算しない。"
            )
            st.markdown("- スプレッド平均: 対数価格差の平均。")
            st.markdown("- スプレッド標準偏差: スプレッドのばらつき。")
            st.markdown("- 最新スプレッド: 直近日のスプレッド値。")
            st.markdown("- 最新Zスコア: 直近スプレッドの標準化値。")
            st.markdown("- 平均出来高: 直近比較本数の平均出来高。一定以上を必須にする。")
            st.markdown(
                "- 平均出来高(小さい方): 2銘柄の平均出来高のうち小さい方。フィルタ条件に使用。"
            )
            st.markdown(
                "- 総合スコア: 直近類似度、最新Zスコア、半減期から簡易スコアを算出し上位のみ評価。"
            )
            st.markdown("- 指数ETFはペア探索対象から除外。")

        with st.expander("ペアキャッシュ", expanded=False):
            if cached_pairs_df is None or cached_pairs_df.empty:
                st.info("ペアキャッシュが未作成です。手動更新で作成してください。")
            else:
                updated_at = cached_meta.get("updated_at") if cached_meta else None
                st.markdown(
                    f"- 更新日: {updated_at or '不明'}\n"
                    f"- ペア数: {len(cached_pairs_df)}"
                )
                if cached_meta:
                    st.caption("キャッシュ作成時の条件")
                    st.json(cached_meta, expanded=False)
                show_cache_details = st.checkbox(
                    "ペアキャッシュの詳細を表示",
                    value=False,
                    help="キャッシュ済みのペア組み合わせと指標を一覧します。",
                )
                if show_cache_details:
                    max_cache_rows = st.number_input(
                        "表示件数(上限)",
                        min_value=10,
                        max_value=1000,
                        value=200,
                        step=10,
                    )
                    cache_display_df = cached_pairs_df.copy()
                    cache_display_df["pair_label"] = cache_display_df.apply(
                        lambda row: (
                            f"{row['symbol_a']} ({name_map.get(row['symbol_a'], '名称未登録')}) / "
                            f"{row['symbol_b']} ({name_map.get(row['symbol_b'], '名称未登録')})"
                        ),
                        axis=1,
                    )
                    if "p_value" in cache_display_df.columns:
                        cache_display_df = cache_display_df.sort_values("p_value", ascending=True)
                    cache_table_data = {
                        "ペア": cache_display_df["pair_label"],
                    }
                    if "p_value" in cache_display_df.columns:
                        cache_table_data["p値"] = cache_display_df["p_value"]
                    if "half_life" in cache_display_df.columns:
                        cache_table_data["半減期"] = cache_display_df["half_life"]
                    if "beta" in cache_display_df.columns:
                        cache_table_data["β"] = cache_display_df["beta"]
                    if "recent_similarity" in cache_display_df.columns:
                        cache_table_data["直近類似度"] = cache_display_df["recent_similarity"]
                    if "recent_return_corr" in cache_display_df.columns:
                        cache_table_data["直近リターン相関"] = cache_display_df["recent_return_corr"]
                    if "long_similarity" in cache_display_df.columns:
                        cache_table_data["長期類似度"] = cache_display_df["long_similarity"]
                    if "long_return_corr" in cache_display_df.columns:
                        cache_table_data["長期リターン相関"] = cache_display_df["long_return_corr"]
                    if "spread_mean" in cache_display_df.columns:
                        cache_table_data["スプレッド平均"] = cache_display_df["spread_mean"]
                    if "spread_std" in cache_display_df.columns:
                        cache_table_data["スプレッド標準偏差"] = cache_display_df["spread_std"]
                    if "spread_latest" in cache_display_df.columns:
                        cache_table_data["最新スプレッド"] = cache_display_df["spread_latest"]
                    if "zscore_latest" in cache_display_df.columns:
                        cache_table_data["最新Zスコア"] = cache_display_df["zscore_latest"]
                    cache_table_df = pd.DataFrame(cache_table_data).head(int(max_cache_rows))
                    cache_table_df = cache_table_df.round(
                        {
                            "p値": 4,
                            "半減期": 2,
                            "β": 3,
                            "直近類似度": 3,
                            "直近リターン相関": 3,
                            "長期類似度": 3,
                            "長期リターン相関": 3,
                            "スプレッド平均": 4,
                            "スプレッド標準偏差": 4,
                            "最新スプレッド": 4,
                            "最新Zスコア": 2,
                        }
                    )
                    st.dataframe(cache_table_df, use_container_width=True)

            st.caption(
                "ペアキャッシュ更新時は対象セクターを絞り込んで作成できます。"
            )
            cache_scope = st.selectbox(
                "キャッシュ作成対象",
                options=["全セクター", "sector17", "sector33"],
                help="改善イテレーションのために1セクターのみを選んでキャッシュを作成できます。",
            )
            cache_sector17 = None
            cache_sector33 = None
            if cache_scope == "sector17":
                if sector17_values:
                    cache_sector17 = st.selectbox(
                        "対象 sector17",
                        options=sector17_values,
                    )
                else:
                    st.warning("sector17 の候補がありません。")
            elif cache_scope == "sector33":
                if sector33_values:
                    cache_sector33 = st.selectbox(
                        "対象 sector33",
                        options=sector33_values,
                    )
                else:
                    st.warning("sector33 の候補がありません。")
            update_cache = st.button("ペアキャッシュを更新", type="secondary")
            if update_cache:
                sector17_filter = cache_sector17
                sector33_filter = cache_sector33
                pair_candidates = generate_pairs_by_sector_candidates(
                    listed_df=listed_df,
                    symbols=symbols,
                    sector17=sector17_filter,
                    sector33=sector33_filter,
                    max_pairs_per_sector=None,
                )
                if not pair_candidates:
                    st.warning(
                        "条件に合致するペア候補がありません。業種フィルタを調整してください。"
                    )
                else:
                    progress_update, progress_done = _build_progress_updater(
                        "ペアキャッシュ更新"
                    )
                    results_df = evaluate_pair_candidates(
                        pair_candidates,
                        recent_window=int(recent_window),
                        long_window=long_window,
                        min_similarity=None,
                        min_long_similarity=min_long_similarity,
                        min_return_corr=None,
                        max_p_value=float(max_p_value) if max_p_value is not None else None,
                        max_half_life=None,
                        max_abs_zscore=max_abs_zscore_filter,
                        min_avg_volume=min_avg_volume_filter,
                        preselect_top_n=None,
                        listed_df=listed_df,
                        history_window=pair_search_history,
                        progress_callback=progress_update,
                    )
                    progress_done()
                    results_df = _attach_pair_sector_info(results_df, listed_df)
                    if max_pairs_per_symbol_limit is not None:
                        results_df = _limit_pairs_per_symbol(
                            results_df, max_pairs_per_symbol_limit
                        )
                    metadata = {
                        "sector17": sector17_filter,
                        "sector33": sector33_filter,
                        "recent_window": int(recent_window),
                        "long_window": long_window,
                        "min_similarity": None,
                        "min_long_similarity": min_long_similarity,
                        "min_return_corr": None,
                        "max_p_value": float(max_p_value) if max_p_value is not None else None,
                        "max_half_life": None,
                        "max_abs_zscore": max_abs_zscore_filter,
                        "min_avg_volume": min_avg_volume_filter,
                        "preselect_top_n": None,
                        "max_pairs_per_sector": None,
                        "max_pairs_per_symbol": max_pairs_per_symbol_limit,
                        "pair_search_history": int(pair_search_history),
                        "cache_scope": cache_scope,
                    }
                    _save_pair_cache(results_df, metadata)
                    st.success("ペアキャッシュを更新しました。")
                    cached_pairs_df, cached_meta = _load_pair_cache()

        st.caption("ペア候補スクリーニングはペアキャッシュのみを使用します。")
        cache_ready = cached_pairs_df is not None and not cached_pairs_df.empty
        run_pairs = st.button(
            "ペア候補を表示",
            type="primary",
            disabled=not cache_ready,
        )
        if run_pairs:
            sector17_filter = None if sector17_choice == "指定なし" else sector17_choice
            sector33_filter = None if sector33_choice == "指定なし" else sector33_choice
            anchor_symbol = selected_symbol if search_scope == "選択銘柄起点(同一セクター)" else None
            filtered_df = _filter_cached_pairs(
                cached_pairs_df,
                sector17_filter,
                sector33_filter,
                anchor_symbol,
                available_symbols=symbols,
            )
            if max_pairs_per_symbol_limit is not None:
                filtered_df = _limit_pairs_per_symbol(
                    filtered_df, max_pairs_per_symbol_limit
                )
            if max_p_value is not None and "p_value" in filtered_df.columns:
                filtered_df = filtered_df[filtered_df["p_value"] <= float(max_p_value)]
            if max_abs_zscore_filter is not None and "zscore_latest" in filtered_df.columns:
                filtered_df = filtered_df[
                    filtered_df["zscore_latest"].abs() <= max_abs_zscore_filter
                ]
            st.session_state["pair_results"] = filtered_df.reset_index(drop=True)
        elif not cache_ready:
            st.warning("ペアキャッシュが未作成です。先に更新してください。")

        pair_results = st.session_state.get("pair_results")
        if pair_results is None:
            st.info("条件を指定して『ペア候補を表示』を押してください。")
        elif pair_results.empty:
            st.info("有効なペア候補が見つかりませんでした。")
        else:
            display_df = pair_results.copy()
            display_df["pair_label"] = display_df.apply(
                lambda row: (
                    f"{row['symbol_a']} ({name_map.get(row['symbol_a'], '名称未登録')}) / "
                    f"{row['symbol_b']} ({name_map.get(row['symbol_b'], '名称未登録')})"
                ),
                axis=1,
            )
            display_df = display_df.sort_values("p_value", ascending=True)
            table_df = pd.DataFrame(
                {
                    "ペア": display_df["pair_label"],
                    "p値": display_df["p_value"],
                    "半減期": display_df["half_life"],
                    "β": display_df["beta"],
                    "直近類似度": display_df["recent_similarity"],
                    "直近リターン相関": display_df["recent_return_corr"],
                    "長期類似度": display_df["long_similarity"],
                    "長期リターン相関": display_df["long_return_corr"],
                    "スプレッド平均": display_df["spread_mean"],
                    "スプレッド標準偏差": display_df["spread_std"],
                    "最新スプレッド": display_df["spread_latest"],
                    "最新Zスコア": display_df["zscore_latest"],
                    "平均出来高(小さい方)": display_df.get("avg_volume_min"),
                }
            )
            table_df = table_df.round(
                {
                    "p値": 4,
                    "半減期": 2,
                    "β": 3,
                    "直近類似度": 3,
                    "直近リターン相関": 3,
                    "長期類似度": 3,
                    "長期リターン相関": 3,
                    "スプレッド平均": 4,
                    "スプレッド標準偏差": 4,
                    "最新スプレッド": 4,
                    "最新Zスコア": 2,
                    "平均出来高(小さい方)": 0,
                }
            )
            st.dataframe(table_df, use_container_width=True)

            selected_label = st.selectbox(
                "可視化するペア",
                options=table_df["ペア"].tolist(),
            )
            selected_row = display_df.loc[display_df["pair_label"] == selected_label].iloc[0]

            signal_cols = st.columns(2)
            with signal_cols[0]:
                entry_threshold = st.number_input(
                    "エントリー閾値 (Zスコア)",
                    min_value=0.5,
                    max_value=5.0,
                    value=2.0,
                    step=0.1,
                )
            with signal_cols[1]:
                exit_threshold = st.number_input(
                    "エグジット閾値 (Zスコア)",
                    min_value=0.0,
                    max_value=5.0,
                    value=0.5,
                    step=0.1,
                )

            df_pair, pair_metrics = compute_spread_series(
                selected_row["symbol_a"], selected_row["symbol_b"]
            )
            _render_pair_spread_chart(df_pair, entry_threshold, exit_threshold, pair_metrics)

    with tab_backtest:
        st.subheader("CAN-SLIM バックテスト")
        st.caption("CAN-SLIM検出を全銘柄に適用し、ブレイク後のピークリターンで評価します。")

        if "backtest_results" not in st.session_state:
            st.session_state["backtest_results"] = None
        if "backtest_summary" not in st.session_state:
            st.session_state["backtest_summary"] = None
        if "grid_search_results" not in st.session_state:
            st.session_state["grid_search_results"] = None
        if "grid_search_summary" not in st.session_state:
            st.session_state["grid_search_summary"] = None

        backtest_col1, backtest_col2 = st.columns(2)
        with backtest_col1:
            bt_lookahead = st.number_input("ピーク探索期間（日）", 5, 120, value=20, step=1)
            bt_return_threshold = st.number_input(
                "上昇判定の下限（ピークリターン）",
                min_value=0.0,
                max_value=1.0,
                value=0.03,
                step=0.01,
            )
            bt_volume_multiplier = st.number_input(
                "出来高/20日平均の下限 (倍)",
                min_value=0.0,
                max_value=10.0,
                value=1.5,
                step=0.1,
            )
        with backtest_col2:
            bt_cup_window = st.number_input(
                "カップ判定期間（日）",
                min_value=20,
                max_value=120,
                value=50,
                step=1,
                key="bt_cup_window",
            )
            bt_saucer_window = st.number_input(
                "ソーサー判定期間（日）",
                min_value=40,
                max_value=180,
                value=80,
                step=1,
                key="bt_saucer_window",
            )
            bt_handle_window = st.number_input(
                "ハンドル判定期間（日）",
                min_value=5,
                max_value=30,
                value=10,
                step=1,
                key="bt_handle_window",
            )

        run_backtest = st.button("バックテストを実行", type="primary")
        if run_backtest:
            progress_update, progress_done = _build_progress_updater("CAN-SLIM バックテスト")
            results_df, summary_df = run_canslim_backtest(
                lookahead=int(bt_lookahead),
                return_threshold=float(bt_return_threshold),
                volume_multiplier=float(bt_volume_multiplier),
                cup_window=int(bt_cup_window),
                saucer_window=int(bt_saucer_window),
                handle_window=int(bt_handle_window),
                progress_callback=progress_update,
            )
            progress_done()
            st.session_state["backtest_results"] = results_df
            st.session_state["backtest_summary"] = summary_df

        backtest_results = st.session_state.get("backtest_results")
        backtest_summary = st.session_state.get("backtest_summary")

        if backtest_summary is not None and not backtest_summary.empty:
            st.markdown("#### 集計結果")
            st.dataframe(backtest_summary, use_container_width=True)

        if backtest_results is not None and not backtest_results.empty:
            st.markdown("#### シグナル一覧")
            st.dataframe(backtest_results.head(200), use_container_width=True)

            st.markdown("#### 判定基準の図解")
            select_options = [
                f"{row.symbol} | {row.date.date()} | {row.pattern}"
                for _, row in backtest_results.iterrows()
            ]
            selected_label = st.selectbox("図解するシグナルを選択", options=select_options)
            selected_row = backtest_results.iloc[select_options.index(selected_label)]

            chart_window = st.number_input(
                "図解表示の前後期間（日）", min_value=20, max_value=200, value=80, step=5
            )
            df_price = load_price_csv(selected_row.symbol)
            df_price = df_price.sort_values("date")
            start_date = pd.to_datetime(selected_row.date) - pd.Timedelta(days=chart_window)
            end_date = pd.to_datetime(selected_row.date) + pd.Timedelta(days=chart_window)
            df_view = df_price[(df_price["date"] >= start_date) & (df_price["date"] <= end_date)].copy()
            if df_view.empty:
                st.warning("図解に必要な期間のデータが不足しています。")
            else:
                breakdown_rangebreaks = _build_trading_rangebreaks(df_view["date"])
                fig = go.Figure()
                fig.add_trace(
                    go.Candlestick(
                        x=df_view["date"],
                        open=df_view["open"],
                        high=df_view["high"],
                        low=df_view["low"],
                        close=df_view["close"],
                        name="ローソク足",
                    )
                )
                fig.add_hline(
                    y=selected_row.pattern_left_peak,
                    line=dict(color="#1f77b4", dash="dash"),
                    annotation_text="左ピーク",
                )
                fig.add_hline(
                    y=selected_row.pattern_right_peak,
                    line=dict(color="#2ca02c", dash="dash"),
                    annotation_text="右ピーク",
                )
                fig.add_hline(
                    y=selected_row.pattern_bottom,
                    line=dict(color="#9467bd", dash="dot"),
                    annotation_text="ボトム",
                )
                fig.add_hrect(
                    y0=selected_row.pattern_handle_low,
                    y1=selected_row.pattern_handle_high,
                    fillcolor="rgba(255,127,14,0.2)",
                    line_width=0,
                    annotation_text="ハンドル範囲",
                )
                fig.add_vline(
                    x=pd.to_datetime(selected_row.date).strftime("%Y-%m-%d"),
                    line=dict(color="#d62728", dash="solid"),
                    annotation_text="ブレイク",
                )
                fig.update_layout(
                    title=f"パターン判定図解: {selected_row.symbol} ({selected_row.pattern})",
                    xaxis_title="Date",
                    yaxis_title="Price",
                    margin=dict(l=20, r=20, t=50, b=20),
                )
                fig.update_xaxes(
                    tickformat="%Y-%m-%d",
                    rangebreaks=breakdown_rangebreaks,
                )

        st.divider()
        st.subheader("カップ形状のグリッドサーチ")
        st.caption("validation成績を主指標にしつつ、trainとの差分にペナルティを掛けて過剰最適化を抑えます。")

        depth_range_options = {
            "0.12-0.35": (0.12, 0.35),
            "0.15-0.40": (0.15, 0.4),
            "0.18-0.45": (0.18, 0.45),
        }

        grid_col1, grid_col2 = st.columns(2)
        with grid_col1:
            gs_lookahead = st.number_input(
                "ピーク探索期間（日）",
                5,
                120,
                value=20,
                step=1,
                key="gs_lookahead",
            )
            gs_return_threshold = st.number_input(
                "上昇判定の下限（ピークリターン）",
                min_value=0.0,
                max_value=1.0,
                value=0.03,
                step=0.01,
                key="gs_return_threshold",
            )
            gs_volume_multiplier = st.number_input(
                "出来高/20日平均の下限 (倍)",
                min_value=0.0,
                max_value=10.0,
                value=1.5,
                step=0.1,
                key="gs_volume_multiplier",
            )
            gs_min_signals = st.number_input(
                "train/validation の最小シグナル数",
                min_value=1,
                max_value=200,
                value=10,
                step=1,
                help="サンプルが少なすぎる組み合わせを除外します。",
            )
            gs_gap_penalty = st.number_input(
                "generalization gap のペナルティ係数",
                min_value=0.0,
                max_value=2.0,
                value=0.5,
                step=0.1,
                help="train/validation の差が大きいほどスコアを下げます。",
            )
        with grid_col2:
            gs_cup_windows = st.multiselect(
                "カップ判定期間（日）候補",
                options=[30, 40, 50, 60, 70, 80],
                default=[40, 50, 60, 70],
            )
            gs_handle_windows = st.multiselect(
                "ハンドル判定期間（日）候補",
                options=[5, 7, 10, 12, 15],
                default=[7, 10, 12],
            )
            gs_depth_ranges = st.multiselect(
                "深さレンジ候補",
                options=list(depth_range_options.keys()),
                default=["0.12-0.35", "0.15-0.40", "0.18-0.45"],
            )
            gs_recovery_ratios = st.multiselect(
                "回復率候補",
                options=[0.8, 0.82, 0.85, 0.88, 0.9],
                default=[0.82, 0.85, 0.88],
            )
            gs_handle_max_depths = st.multiselect(
                "ハンドル最大深さ候補",
                options=[0.08, 0.1, 0.12, 0.15, 0.18],
                default=[0.1, 0.12, 0.15],
            )

        run_grid_search = st.button("グリッドサーチを実行", type="primary")
        if run_grid_search:
            if not gs_cup_windows or not gs_handle_windows or not gs_depth_ranges:
                st.warning("カップ/ハンドル期間と深さレンジの候補を1つ以上選択してください。")
            else:
                depth_ranges = [depth_range_options[key] for key in gs_depth_ranges]
                progress_update, progress_done = _build_progress_updater("カップ形状グリッドサーチ")
                eval_df, best_summary = grid_search_cup_shape(
                    lookahead=int(gs_lookahead),
                    return_threshold=float(gs_return_threshold),
                    volume_multiplier=float(gs_volume_multiplier),
                    min_signals=int(gs_min_signals),
                    gap_penalty=float(gs_gap_penalty),
                    cup_windows=[int(v) for v in gs_cup_windows],
                    handle_windows=[int(v) for v in gs_handle_windows],
                    depth_ranges=depth_ranges,
                    recovery_ratios=[float(v) for v in gs_recovery_ratios],
                    handle_max_depths=[float(v) for v in gs_handle_max_depths],
                    progress_callback=progress_update,
                )
                progress_done()
                st.session_state["grid_search_results"] = eval_df
                st.session_state["grid_search_summary"] = best_summary

        grid_search_results = st.session_state.get("grid_search_results")
        grid_search_summary = st.session_state.get("grid_search_summary")

        if grid_search_summary is not None and not grid_search_summary.empty:
            st.markdown("#### ベスト設定")
            st.dataframe(grid_search_summary, use_container_width=True)

        if grid_search_results is not None and not grid_search_results.empty:
            st.markdown("#### グリッドサーチ結果（上位）")
            st.dataframe(grid_search_results.head(200), use_container_width=True)
            st.plotly_chart(fig, use_container_width=True)
        elif backtest_results is not None:
            st.info("該当するシグナルがありませんでした。")

        st.markdown("---")
        st.subheader("ペアトレード バックテスト")
        st.caption("業種ごとの銘柄ペアに対して、Zスコア型の平均回帰戦略を評価します。")
        st.caption("運用期間は直近3年に固定しています。")

        if "pair_trade_summary" not in st.session_state:
            st.session_state["pair_trade_summary"] = None
        if "pair_trade_trades" not in st.session_state:
            st.session_state["pair_trade_trades"] = None
        if "pair_trade_optimization" not in st.session_state:
            st.session_state["pair_trade_optimization"] = None

        sector_options = []
        if "sector33" in listed_df.columns:
            sector_options.append("sector33")
        if "sector17" in listed_df.columns:
            sector_options.append("sector17")
        if not sector_options:
            st.warning("銘柄マスタに sector33/sector17 が無いためペアトレードを実行できません。")
        else:
            sector_col = st.selectbox(
                "業種区分の列",
                options=sector_options,
                index=0,
                help="listed_master.csv にある sector33 / sector17 を選択できます。",
            )
            pair_col1, pair_col2 = st.columns(2)
            with pair_col1:
                min_symbols = st.number_input(
                    "業種内の最小銘柄数",
                    min_value=2,
                    max_value=200,
                    value=5,
                    step=1,
                )
                max_pairs_per_sector = st.number_input(
                    "業種あたりの最大ペア数",
                    min_value=1,
                    max_value=200,
                    value=20,
                    step=1,
                )
                lookback = st.number_input("ローリング窓（日）", 20, 200, value=60, step=5)
                max_holding_days = st.number_input(
                    "最大保有日数", 5, 60, value=20, step=1
                )
            with pair_col2:
                entry_z = st.number_input("エントリーZ", 0.5, 5.0, value=2.0, step=0.1)
                exit_z = st.number_input("イグジットZ", 0.1, 2.0, value=0.5, step=0.1)
                stop_z = st.number_input("ストップZ", 1.0, 6.0, value=3.5, step=0.1)
                pair_trade_years = 3
                pair_trade_start = date.today() - timedelta(days=365 * pair_trade_years)
                enable_date_filter = st.checkbox(
                    "期間フィルタを使う",
                    value=True,
                    disabled=True,
                    help="ペアトレードは直近3年で固定しています。",
                )
                date_range = st.date_input(
                    "対象期間",
                    value=(pair_trade_start, date.today()),
                    disabled=True,
                )
            cached_pairs_df, _ = _load_pair_cache()
            if cached_pairs_df is not None and not cached_pairs_df.empty:
                cached_pairs_df = _attach_pair_sector_info(cached_pairs_df, listed_df)
            use_cached_pairs = st.checkbox(
                "キャッシュ済みペアを使用（p値<=0.10）",
                value=True,
                help="手動更新したペアキャッシュを使うことでバックテストを高速化します。",
            )
            if use_cached_pairs and (cached_pairs_df is None or cached_pairs_df.empty):
                st.warning("ペアキャッシュが未作成です。キャッシュ更新後に利用できます。")

            run_pair_backtest = st.button("ペアトレードを実行", type="primary")
            if run_pair_backtest:
                if use_cached_pairs and cached_pairs_df is not None and not cached_pairs_df.empty:
                    filtered_pairs = _filter_cached_pairs(
                        cached_pairs_df,
                        None,
                        None,
                        None,
                    )
                    if "p_value" in filtered_pairs.columns:
                        filtered_pairs = filtered_pairs[filtered_pairs["p_value"] <= 0.1]
                    sector_key = "pair_sector33" if sector_col == "sector33" else "pair_sector17"
                    if sector_key in filtered_pairs.columns:
                        sector_counts = (
                            listed_df[sector_col].dropna().astype(str).value_counts()
                        )
                        eligible_sectors = set(
                            sector_counts[sector_counts >= int(min_symbols)].index
                        )
                        filtered_pairs = filtered_pairs[
                            filtered_pairs[sector_key].astype(str).isin(eligible_sectors)
                        ]
                    filtered_pairs = _limit_pairs_per_sector(
                        filtered_pairs, sector_col, int(max_pairs_per_sector)
                    )
                    pairs = list(
                        zip(
                            filtered_pairs["symbol_a"].astype(str).tolist(),
                            filtered_pairs["symbol_b"].astype(str).tolist(),
                        )
                    )
                else:
                    pairs = generate_pairs_by_sector(
                        sector_col=sector_col,
                        min_symbols=int(min_symbols),
                        max_pairs_per_sector=int(max_pairs_per_sector),
                    )
                config = PairTradeConfig(
                    lookback=int(lookback),
                    entry_z=float(entry_z),
                    exit_z=float(exit_z),
                    stop_z=float(stop_z),
                    max_holding_days=int(max_holding_days),
                )
                start_date = None
                end_date = None
                if date_range and len(date_range) == 2:
                    start_date = date_range[0].isoformat()
                    end_date = date_range[1].isoformat()
                progress_update, progress_done = _build_progress_updater("ペアトレードバックテスト")
                trades_df, summary_df = backtest_pairs(
                    pairs,
                    config=config,
                    start_date=start_date,
                    end_date=end_date,
                    progress_callback=progress_update,
                )
                progress_done()
                st.session_state["pair_trade_trades"] = trades_df
                st.session_state["pair_trade_summary"] = summary_df

            pair_summary = st.session_state.get("pair_trade_summary")
            pair_trades = st.session_state.get("pair_trade_trades")
            if pair_summary is not None and not pair_summary.empty:
                st.markdown("#### ペア別サマリ")
                st.dataframe(pair_summary.head(50), use_container_width=True)
            elif pair_summary is not None:
                st.info("条件に合致するペアがありませんでした。")

            if pair_trades is not None and not pair_trades.empty:
                st.markdown("#### トレード明細（上位200件）")
                st.dataframe(pair_trades.head(200), use_container_width=True)

            with st.expander("パラメータ探索（グリッドサーチ）", expanded=False):
                st.caption("カンマ区切りで候補値を入力すると自動探索できます。")

                def _parse_grid(text: str, cast_type: type) -> List[float]:
                    if not text.strip():
                        return []
                    values = []
                    for item in text.split(","):
                        item = item.strip()
                        if not item:
                            continue
                        values.append(cast_type(item))
                    return values

                grid_col1, grid_col2 = st.columns(2)
                with grid_col1:
                    grid_lookback = st.text_input("lookback候補", value="40,60,80")
                    grid_entry = st.text_input("entry_z候補", value="1.5,2.0,2.5")
                    grid_exit = st.text_input("exit_z候補", value="0.3,0.5")
                with grid_col2:
                    grid_stop = st.text_input("stop_z候補", value="3.0,3.5")
                    grid_holding = st.text_input("max_holding_days候補", value="10,20")
                    min_trades = st.number_input(
                        "最小トレード数",
                        min_value=1,
                        max_value=50,
                        value=5,
                        step=1,
                    )
                run_optimize = st.button("パラメータ探索を実行", type="secondary")
                if run_optimize:
                    lookback_values = _parse_grid(grid_lookback, int)
                    entry_values = _parse_grid(grid_entry, float)
                    exit_values = _parse_grid(grid_exit, float)
                    stop_values = _parse_grid(grid_stop, float)
                    holding_values = _parse_grid(grid_holding, int)
                    if not all(
                        [lookback_values, entry_values, exit_values, stop_values, holding_values]
                    ):
                        st.warning("すべての候補値を入力してください。")
                    else:
                        if use_cached_pairs and cached_pairs_df is not None and not cached_pairs_df.empty:
                            filtered_pairs = _filter_cached_pairs(
                                cached_pairs_df,
                                None,
                                None,
                                None,
                            )
                            if "p_value" in filtered_pairs.columns:
                                filtered_pairs = filtered_pairs[filtered_pairs["p_value"] <= 0.1]
                            sector_key = (
                                "pair_sector33"
                                if sector_col == "sector33"
                                else "pair_sector17"
                            )
                            if sector_key in filtered_pairs.columns:
                                sector_counts = (
                                    listed_df[sector_col]
                                    .dropna()
                                    .astype(str)
                                    .value_counts()
                                )
                                eligible_sectors = set(
                                    sector_counts[sector_counts >= int(min_symbols)].index
                                )
                                filtered_pairs = filtered_pairs[
                                    filtered_pairs[sector_key].astype(str).isin(eligible_sectors)
                                ]
                            filtered_pairs = _limit_pairs_per_sector(
                                filtered_pairs, sector_col, int(max_pairs_per_sector)
                            )
                            pairs = list(
                                zip(
                                    filtered_pairs["symbol_a"].astype(str).tolist(),
                                    filtered_pairs["symbol_b"].astype(str).tolist(),
                                )
                            )
                        else:
                            pairs = generate_pairs_by_sector(
                                sector_col=sector_col,
                                min_symbols=int(min_symbols),
                                max_pairs_per_sector=int(max_pairs_per_sector),
                            )
                        param_grid = {
                            "lookback": lookback_values,
                            "entry_z": entry_values,
                            "exit_z": exit_values,
                            "stop_z": stop_values,
                            "max_holding_days": holding_values,
                        }
                        start_date = None
                        end_date = None
                        if date_range and len(date_range) == 2:
                            start_date = date_range[0].isoformat()
                            end_date = date_range[1].isoformat()
                        progress_update, progress_done = _build_progress_updater("ペアトレード最適化")
                        optimization_df = optimize_pair_trade_parameters(
                            pairs,
                            param_grid=param_grid,
                            start_date=start_date,
                            end_date=end_date,
                            min_trades=int(min_trades),
                            progress_callback=progress_update,
                        )
                        progress_done()
                        st.session_state["pair_trade_optimization"] = optimization_df

                optimization_df = st.session_state.get("pair_trade_optimization")
                if optimization_df is not None and not optimization_df.empty:
                    st.markdown("#### 探索結果（上位20件）")
                    st.dataframe(optimization_df.head(20), use_container_width=True)

    with tab_breadth:
        st.subheader("マーケットブレッドス")
        st.caption("騰落銘柄数やTRIN、マクレラン指標に加えてTOPIX推移を可視化します。")

        breadth_period = st.radio(
            "対象期間",
            options=["1y", "all"],
            format_func=lambda v: "過去1年" if v == "1y" else "全期間",
            horizontal=True,
        )

        st.markdown("---")
        recompute = st.button("再計算", key="recompute_breadth")

        @st.cache_data(show_spinner=False)
        def _load_breadth_data(cache_key: str) -> pd.DataFrame:
            _ = cache_key
            return compute_breadth_indicators(aggregate_market_breadth())

        if "breadth_cache_version" not in st.session_state:
            st.session_state["breadth_cache_version"] = 0

        price_files = sorted(str(p) for p in PRICE_CSV_DIR.glob("*.csv"))
        if recompute:
            st.session_state["breadth_cache_version"] += 1

        cache_key = "|".join(
            [str(st.session_state["breadth_cache_version"]), *price_files]
        )

        df_breadth = _load_breadth_data(cache_key)

        if df_breadth.empty:
            st.warning("集計に必要なデータが不足しています。price_csvに銘柄CSVがあるか確認してください。")
            return

        df_breadth = df_breadth.dropna(subset=["advancing_issues", "declining_issues"])
        if df_breadth.empty:
            st.warning("有効な日次データが見つかりませんでした。")
            return

        if breadth_period == "1y":
            cutoff = pd.to_datetime(date.today() - timedelta(days=365))
            df_breadth = df_breadth[df_breadth["date"] >= cutoff]

        if df_breadth.empty:
            st.warning("指定期間にデータがありません。期間を広げてください。")
            return

        df_topix = None
        try:
            df_topix = load_topix_csv()
        except FileNotFoundError:
            st.info("TOPIXデータがないため、マーケットブレッドスではTOPIXの表示をスキップします。")
        except ValueError as exc:
            st.warning(f"TOPIXデータの読み込みに失敗しました: {exc}")

        if df_topix is not None:
            df_topix["date"] = pd.to_datetime(df_topix["date"])
            if breadth_period == "1y":
                cutoff = pd.to_datetime(date.today() - timedelta(days=365))
                df_topix = df_topix[df_topix["date"] >= cutoff]
            if df_topix.empty:
                st.info("指定期間にTOPIXデータがありません。")
            else:
                df_topix["date_str"] = df_topix["date"].dt.strftime("%Y-%m-%d")
                st.markdown("### TOPIX 終値")
                fig_topix = go.Figure()
                fig_topix.add_trace(
                    go.Scatter(
                        x=df_topix["date_str"],
                        y=df_topix["close"],
                        mode="lines",
                        name="TOPIX",
                    )
                )
                fig_topix.update_layout(margin=dict(l=10, r=10, t=30, b=10), height=300)
                fig_topix.update_xaxes(type="category")
                st.plotly_chart(fig_topix, use_container_width=True)

        df_breadth["date_str"] = pd.to_datetime(df_breadth["date"]).dt.strftime("%Y-%m-%d")

        fig_breadth = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05)

        fig_breadth.add_trace(
            go.Bar(
                x=df_breadth["date_str"],
                y=df_breadth["advancing_issues"],
                name="上昇銘柄",
                marker_color="#2ca02c",
            ),
            row=1,
            col=1,
        )
        fig_breadth.add_trace(
            go.Bar(
                x=df_breadth["date_str"],
                y=df_breadth["declining_issues"] * -1,
                name="下落銘柄",
                marker_color="#d62728",
            ),
            row=1,
            col=1,
        )
        fig_breadth.update_yaxes(title_text="銘柄数 (下落は負値)", row=1, col=1)

        fig_breadth.add_trace(
            go.Scatter(
                x=df_breadth["date_str"],
                y=df_breadth["advance_decline_line"],
                name="騰落ライン",
                line=dict(color="#1f77b4"),
            ),
            row=2,
            col=1,
        )
        fig_breadth.add_trace(
            go.Scatter(
                x=df_breadth["date_str"],
                y=df_breadth["mcclellan_oscillator"],
                name="マクレラン・オシレーター",
                line=dict(color="#ff7f0e"),
            ),
            row=3,
            col=1,
        )
        fig_breadth.add_trace(
            go.Scatter(
                x=df_breadth["date_str"],
                y=df_breadth["mcclellan_summation"],
                name="マクレラン・サマ",
                line=dict(color="#9467bd", dash="dash"),
            ),
            row=3,
            col=1,
        )

        fig_breadth.update_layout(
            height=900,
            margin=dict(l=10, r=10, t=30, b=10),
            legend=dict(orientation="h"),
        )
        fig_breadth.update_xaxes(type="category")

        st.plotly_chart(fig_breadth, use_container_width=True)

        st.markdown("### TRIN")
        fig_trin = go.Figure()
        fig_trin.add_trace(
            go.Scatter(
                x=df_breadth["date_str"],
                y=df_breadth["trin"],
                mode="lines",
                name="TRIN",
            )
        )
        fig_trin.add_hline(y=1.0, line=dict(color="#cccccc", dash="dash"))
        fig_trin.update_layout(margin=dict(l=10, r=10, t=30, b=10), height=300)
        fig_trin.update_xaxes(type="category")
        st.plotly_chart(fig_trin, use_container_width=True)

if __name__ == "__main__":
    main()
