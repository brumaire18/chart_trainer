from datetime import date, timedelta
from typing import Optional

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from app.config import DEFAULT_LOOKBACK_BARS
from app.data_loader import (
    enforce_light_plan_window,
    fetch_and_save_price_csv,
    get_available_symbols,
    load_price_csv,
)
from app.jquants_fetcher import (
    build_universe,
    get_credential_status,
    load_listed_master,
    update_universe,
)


LIGHT_PLAN_YEARS = 5


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
    resampled = (
        df_idx.resample(rule)
        .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
        .dropna()
        .reset_index()
    )
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


def _has_macd_golden_cross(df: pd.DataFrame) -> bool:
    """
    直近足で MACD がシグナルを上抜けていれば True。

    ヒストグラムが正の方向へ転じていることも確認する。
    """

    if len(df) < 2:
        return False

    latest = df.iloc[-1]
    prev = df.iloc[-2]
    return bool(
        latest["macd"] > latest["macd_signal"]
        and prev["macd"] <= prev["macd_signal"]
        and latest["macd_hist"] > 0
    )


def main():
    st.set_page_config(page_title="Chart Trainer (Line ver.)", layout="wide")
    st.title("Chart Trainer - フェーズ1（ラインチャート版）")

    # --- サイドバー ---
    st.sidebar.header("設定")

    st.sidebar.subheader("J-Quants からデータ取得")
    available_symbols = get_available_symbols()
    default_symbol = available_symbols[0] if available_symbols else ""

    download_symbol = st.sidebar.text_input(
        "銘柄コード (例: 7203)", value=default_symbol
    )
    default_start = date.today() - timedelta(days=365 * LIGHT_PLAN_YEARS)
    default_end = date.today()
    st.sidebar.caption(
        "※ ライトプランでは過去約5年分まで取得できます。"
        "指定にかかわらず取得可能な最大範囲（過去5年）に自動調整します。"
    )

    with st.sidebar.expander("プライム + スタンダードを一括更新"):
        creds = get_credential_status()
        st.write("認証情報の検知状況:")
        st.write(
            f"- MAILADDRESS: {'✅' if creds['MAILADDRESS'] else '❌'}"
            f"  / PASSWORD: {'✅' if creds['PASSWORD'] else '❌'}"
        )
        st.write(f"- JQUANTS_REFRESH_TOKEN: {'✅' if creds['JQUANTS_REFRESH_TOKEN'] else '❌'}")

        include_custom = st.checkbox(
            "custom_symbols.txt も含める", value=False, key="include_custom_universe"
        )
        universe_source = st.radio(
            "更新対象",
            options=["prime_standard", "listed_all"],
            format_func=lambda v: "プライム+スタンダード" if v == "prime_standard" else "listed_masterにある全銘柄",
            index=0,
            key="universe_source",
        )
        full_refresh = st.checkbox(
            "フルリフレッシュ（取得可能な5年分を再取得）",
            value=False,
            key="full_refresh_universe",
        )
        if st.button("ユニバースを更新", key="update_universe_button"):
            try:
                with st.spinner("ユニバースを更新しています..."):
                    target_codes = build_universe(
                        include_custom=include_custom,
                        use_listed_master=universe_source == "listed_all",
                    )
                    update_universe(
                        codes=target_codes,
                        full_refresh=full_refresh,
                        use_listed_master=universe_source == "listed_all",
                    )
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
            st.sidebar.success("ダウンロードに成功しました。")
            st.rerun()
        except Exception as exc:  # broad catch for user feedback
            st.sidebar.error(f"ダウンロードに失敗しました: {exc}")

    symbols = get_available_symbols()
    try:
        listed_df = load_listed_master()
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

    selected_symbol = st.sidebar.selectbox(
        "銘柄", symbols, format_func=lambda c: f"{c} ({name_map.get(c, '名称未登録')})"
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

    tab_chart, tab_screen = st.tabs(["チャート表示", "スクリーナー"])

    with tab_chart:
        # --- チャート表示 ---
        df_daily = load_price_csv(selected_symbol)

        df_daily_trading = df_daily[df_daily["volume"].fillna(0) > 0].copy()
        removed_rows = len(df_daily) - len(df_daily_trading)
        if removed_rows > 0:
            st.sidebar.info(
                f"出来高が0の日を{removed_rows}日分除外して日足チャートを表示します。"
            )

        df_resampled = _resample_ohlc(df_daily_trading, timeframe)

        if df_resampled.empty:
            st.warning("表示できるデータがありません。先にデータを取得してください。")
            return

    lookback_window = min(lookback, len(df_resampled))
    df_problem = df_resampled.tail(lookback_window).copy()
    df_problem = _compute_indicators(df_problem)
    df_problem["date_str"] = df_problem["date"].dt.strftime("%Y-%m-%d")

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
                x=df_problem["date_str"],
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

        lookback_window = min(lookback, len(df_resampled))
        df_problem = df_resampled.tail(lookback_window).copy()
        df_problem = _compute_indicators(df_problem)
        df_problem["date_str"] = df_problem["date"].dt.strftime("%Y-%m-%d")

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
                    x=df_problem["date_str"],
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
                    x=df_problem["date_str"],
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
                        x=df_problem["date_str"],
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
                        x=df_problem["date_str"],
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
                        x=df_problem["date_str"],
                        y=df_problem["bb_upper"],
                        name="BB Upper",
                        line=dict(color="rgba(180,180,180,0.6)", dash="dot"),
                    ),
                    row=1,
                    col=1,
                )
                price_fig.add_trace(
                    go.Scatter(
                        x=df_problem["date_str"],
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
                        x=df_problem["date_str"],
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
            price_fig.update_xaxes(type="category")

        if chart_type == "pnf":
            price_fig.update_layout(margin=dict(l=10, r=10, t=40, b=10))

        st.plotly_chart(price_fig, use_container_width=True)

        osc_fig = make_subplots(specs=[[{"secondary_y": True}]])
        if show_rsi:
            osc_fig.add_trace(
                go.Scatter(
                    x=df_problem["date_str"],
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
                    x=df_problem["date_str"],
                    y=df_problem["stoch_k"],
                    name="%K",
                    line=dict(color="#1f77b4"),
                ),
                secondary_y=False,
            )
            osc_fig.add_trace(
                go.Scatter(
                    x=df_problem["date_str"],
                    y=df_problem["stoch_d"],
                    name="%D",
                    line=dict(color="#ff7f0e", dash="dot"),
                ),
                secondary_y=False,
            )
        if show_macd:
            osc_fig.add_trace(
                go.Bar(
                    x=df_problem["date_str"],
                    y=df_problem["macd_hist"],
                    name="MACD Hist",
                    marker_color="rgba(100,100,100,0.4)",
                ),
                secondary_y=True,
            )
            osc_fig.add_trace(
                go.Scatter(
                    x=df_problem["date_str"],
                    y=df_problem["macd"],
                    name="MACD",
                    line=dict(color="#d62728"),
                ),
                secondary_y=True,
            )
            osc_fig.add_trace(
                go.Scatter(
                    x=df_problem["date_str"],
                    y=df_problem["macd_signal"],
                    name="Signal",
                    line=dict(color="#9467bd", dash="dash"),
                ),
                secondary_y=True,
            )
        if show_obv:
            osc_fig.add_trace(
                go.Scatter(
                    x=df_problem["date_str"],
                    y=df_problem["obv"],
                    name="OBV",
                    line=dict(color="#8c564b"),
                ),
                secondary_y=True,
            )

        osc_fig.update_layout(
            xaxis_title="Date",
            yaxis_title="RSI / Stoch",
            margin=dict(l=10, r=10, t=40, b=10),
            legend=dict(orientation="h"),
        )
        osc_fig.update_yaxes(title_text="MACD", secondary_y=True)
        osc_fig.update_xaxes(type="category")

        st.plotly_chart(osc_fig, use_container_width=True)

    with tab_screen:
        st.subheader("テクニカル・スクリーナー")
        st.caption("日足データを使い、直近のMACDゴールデンクロスやRSI帯などで抽出します。")

        target_markets = st.multiselect(
            "市場 (空欄なら全て)",
            options=sorted(listed_df["market"].dropna().unique()),
        )
        rsi_range = st.slider("RSI(14) 範囲", min_value=0, max_value=100, value=(40, 65))
        require_macd_gc = st.checkbox("直近でMACDゴールデンクロス", value=True)
        require_sma20_trend = st.checkbox("終値 > SMA20 かつ SMA20が上向き", value=True)
        sma_trend_lookback = st.slider("SMA20上向きの判定幅（日）", 1, 10, value=3)
        volume_multiplier = st.number_input(
            "出来高/20日平均の下限 (倍)", min_value=0.0, max_value=10.0, value=0.8, step=0.1
        )
    if show_obv:
        osc_fig.add_trace(
            go.Scatter(
                x=df_problem["date_str"],
                y=df_problem["obv"],
                name="OBV",
                line=dict(color="#8c564b"),
            ),
            secondary_y=True,
        )

        screening_results = []
        for code in symbols:
            if target_markets:
                code_market = listed_df.loc[listed_df["code"] == int(code), "market"]
                if not code_market.empty and code_market.iloc[0] not in target_markets:
                    continue

            try:
                df_daily = load_price_csv(code)
            except Exception:
                continue

            df_daily = df_daily[df_daily["volume"].fillna(0) > 0]
            if len(df_daily) < max(50, sma_trend_lookback + 1):
                continue

            df_ind = _compute_indicators(df_daily.tail(200))
            latest = df_ind.iloc[-1]
            if latest.isna().any():
                continue

            rsi_ok = rsi_range[0] <= latest["rsi14"] <= rsi_range[1]
            macd_ok = _has_macd_golden_cross(df_ind) if require_macd_gc else True

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
            vol_ok = (
                avg_vol20 is not None
                and avg_vol20 > 0
                and latest["volume"] >= avg_vol20 * volume_multiplier
            )

            if all([rsi_ok, macd_ok, sma_ok, vol_ok]):
                change_pct = (
                    (latest["close"] - df_ind.iloc[-2]["close"]) / df_ind.iloc[-2]["close"] * 100
                    if len(df_ind) >= 2 and df_ind.iloc[-2]["close"] != 0
                    else None
                )
                screening_results.append(
                    {
                        "code": code,
                        "name": name_map.get(code, "-"),
                        "close": latest["close"],
                        "RSI14": round(latest["rsi14"], 2),
                        "MACD": round(latest["macd"], 3),
                        "Signal": round(latest["macd_signal"], 3),
                        "出来高": int(latest["volume"]),
                        "20日平均出来高": int(avg_vol20) if pd.notna(avg_vol20) else None,
                        "日次騰落率%": round(change_pct, 2) if change_pct is not None else None,
                    }
                )

        if screening_results:
            df_result = pd.DataFrame(screening_results)
            df_result = df_result.sort_values("日次騰落率%", ascending=False, na_position="last")
            st.success(f"{len(df_result)} 銘柄が条件に合致しました。")
            st.dataframe(df_result, use_container_width=True)
        else:
            st.info("条件に合致する銘柄が見つかりませんでした。条件を緩めて再検索してください。")

if __name__ == "__main__":
    main()
