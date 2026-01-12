from collections import Counter
from datetime import date, timedelta
import json
from typing import List, Optional, Tuple

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
from app.backtest import run_canslim_backtest, scan_canslim_patterns
from app.jquants_fetcher import (
    append_quotes_for_date,
    build_universe,
    get_credential_status,
    load_listed_master,
    update_topix,
    update_universe_with_anchor_day,
    update_universe,
)


LIGHT_PLAN_YEARS = 5


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
    symbol: str, cache_key: str, topix_cache_key: Optional[str] = None
) -> Tuple[pd.DataFrame, int, Optional[dict]]:
    """
    日足データを読み込み、出来高0を除外したうえでインジケーターを計算する。

    cache_key を引数に含めることで、CSV更新時にキャッシュが破棄される。
    """

    _ = (cache_key, topix_cache_key)  # キャッシュ用に参照だけ行う
    df_daily = load_price_csv(symbol)
    df_daily_trading = df_daily[df_daily["volume"].fillna(0) > 0].copy()
    removed_rows = len(df_daily) - len(df_daily_trading)
    df_ind = _compute_indicators(df_daily_trading)
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

    # ボリンジャーバンド (20, 2σ)
    bb_basis = df_ind["close"].rolling(20).mean()
    bb_std = df_ind["close"].rolling(20).std()
    df_ind["bb_upper"] = bb_basis + 2 * bb_std
    df_ind["bb_lower"] = bb_basis - 2 * bb_std

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

    return df_ind


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
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df_resampled["date"],
            y=df_resampled["close"],
            mode="lines",
            line=dict(color="#1f77b4"),
            name="終値",
        )
    )
    fig.update_layout(
        margin=dict(l=10, r=10, t=30, b=10),
        height=220,
        title=f"{symbol} {name}",
        showlegend=False,
    )
    fig.update_xaxes(tickformat="%y/%m/%d", nticks=6, title="")
    fig.update_yaxes(title="")
    return fig


def _build_mini_chart_from_resampled(
    df_resampled: pd.DataFrame, symbol: str, name: str, lookback: int
) -> Optional[go.Figure]:
    if df_resampled.empty:
        return None
    df_resampled = df_resampled.tail(lookback).copy()
    if df_resampled.empty:
        return None
    df_resampled["date_str"] = df_resampled["date"].dt.strftime("%Y-%m-%d")
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df_resampled["date_str"],
            y=df_resampled["close"],
            mode="lines",
            line=dict(color="#1f77b4"),
            name="終値",
        )
    )
    title_text = f"{symbol} {name}" if show_title else ""
    fig.update_layout(
        margin=dict(l=10, r=10, t=30, b=10),
        height=220,
        title=title_text,
        showlegend=False,
    )
    fig.update_xaxes(type="category", title="")
    fig.update_yaxes(title="")
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
                        )
                    else:
                        update_universe(
                            codes=target_codes,
                            full_refresh=full_refresh,
                            use_listed_master=universe_source == "listed_all",
                            market_filter="growth" if universe_source == "growth" else "prime_standard",
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
    show_sma20 = st.sidebar.checkbox("SMA 20", value=True)
    show_sma50 = st.sidebar.checkbox("SMA 50", value=False)
    show_bbands = st.sidebar.checkbox("ボリンジャーバンド", value=False)

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

    tab_chart, tab_screen, tab_backtest, tab_breadth = st.tabs(
        ["チャート表示", "スクリーナー", "バックテスト", "マーケットブレッドス"]
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
                cache_key = _get_price_dir_cache_key()
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
        df_daily_trading, removed_rows, topix_info = _load_price_with_indicators(
            selected_symbol, cache_key, topix_cache_key
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
        df_problem["date_str"] = df_problem["date"].dt.strftime("%y/%m/%d")

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
            if show_sma20:
                price_fig.add_trace(
                    go.Scatter(
                        x=df_problem["date"],
                        y=df_problem["sma20"],
                        name="SMA 20",
                        line=dict(dash="dash"),
                    ),
                    row=1,
                    col=1,
                )
            if show_sma50:
                price_fig.add_trace(
                    go.Scatter(
                        x=df_problem["date"],
                        y=df_problem["sma50"],
                        name="SMA 50",
                        line=dict(dash="dot"),
                    ),
                    row=1,
                    col=1,
                )
            if show_bbands:
                price_fig.add_trace(
                    go.Scatter(
                        x=df_problem["date"],
                        y=df_problem["bb_upper"],
                        name="BB Upper",
                        line=dict(color="rgba(180,180,180,0.6)", dash="dot"),
                    ),
                    row=1,
                    col=1,
                )
                price_fig.add_trace(
                    go.Scatter(
                        x=df_problem["date"],
                        y=df_problem["bb_lower"],
                        name="BB Lower",
                        line=dict(color="rgba(180,180,180,0.6)", dash="dot"),
                        fill="tonexty",
                        fillcolor="rgba(180,180,180,0.1)",
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
            price_fig.update_xaxes(tickformat="%y/%m/%d", nticks=6)

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
                        line=dict(color="#111111", dash="solid"),
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
        osc_fig.update_xaxes(tickformat="%y/%m/%d", nticks=6)
        osc_fig.update_yaxes(title_text="MACD / RS", secondary_y=True)
        st.plotly_chart(osc_fig, use_container_width=True)

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

                    if all([rsi_ok, macd_ok, sma_ok, vol_ok, rs_ok, canslim_ok]):
                    weekly_vol_ok = True
                    weekly_volume = None
                    if apply_weekly_volume_quartile:
                        weekly_volume = weekly_volume_map.get(code_str)
                        if weekly_volume_threshold is None or weekly_volume is None:
                            weekly_vol_ok = False
                        else:
                            weekly_vol_ok = weekly_volume >= weekly_volume_threshold

                    if all([rsi_ok, macd_ok, sma_ok, vol_ok, rs_ok, weekly_vol_ok]):
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

                    chart_df["date_str"] = chart_df["date"].dt.strftime("%Y-%m-%d")

                    left, right = st.columns([1, 3])
                    with left:
                        st.markdown(f"**{code_str} {row['name']}**")
                        st.caption("直近90日間の終値推移")
                    with right:
                        preview_fig = go.Figure()
                        preview_fig.add_trace(
                            go.Scatter(
                                x=chart_df["date_str"],
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
                        preview_fig.update_xaxes(type="category")
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

    with tab_backtest:
        st.subheader("CAN-SLIM バックテスト")
        st.caption("CAN-SLIM検出を全銘柄に適用し、ブレイク後のピークリターンで評価します。")

        if "backtest_results" not in st.session_state:
            st.session_state["backtest_results"] = None
        if "backtest_summary" not in st.session_state:
            st.session_state["backtest_summary"] = None

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
            with st.spinner("バックテストを実行しています..."):
                results_df, summary_df = run_canslim_backtest(
                    lookahead=int(bt_lookahead),
                    return_threshold=float(bt_return_threshold),
                    volume_multiplier=float(bt_volume_multiplier),
                    cup_window=int(bt_cup_window),
                    saucer_window=int(bt_saucer_window),
                    handle_window=int(bt_handle_window),
                )
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
                df_view["date_str"] = df_view["date"].dt.strftime("%Y-%m-%d")
                fig = go.Figure()
                fig.add_trace(
                    go.Candlestick(
                        x=df_view["date_str"],
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
                st.plotly_chart(fig, use_container_width=True)
        elif backtest_results is not None:
            st.info("該当するシグナルがありませんでした。")

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

        price_files = sorted(str(p) for p in PRICE_CSV_DIR.glob("*.csv"))
        cache_key = "|".join(price_files)
        if recompute:
            st.cache_data.clear()

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
