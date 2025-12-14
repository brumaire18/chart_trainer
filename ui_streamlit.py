from datetime import date, timedelta

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from app.config import DEFAULT_LOOKBACK_BARS
from app.data_loader import (
    enforce_free_plan_window,
    fetch_and_save_price_csv,
    get_available_symbols,
    load_price_csv,
)


FREE_PLAN_WEEKS = 12


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

    return df_ind

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
    default_start = date.today() - timedelta(weeks=FREE_PLAN_WEEKS)
    default_end = date.today()
    st.sidebar.caption(
        "※ フリー版では過去12週間より前のデータは取得できません。古い日付を指定した場合は自動で切り上げます。"
    )
    start_date = st.sidebar.date_input("開始日", value=default_start)
    end_date = st.sidebar.date_input("終了日", value=default_end)

    if st.sidebar.button("J-Quantsからダウンロード"):
        try:
            if start_date > end_date:
                raise ValueError("終了日は開始日以降にしてください。")

            request_start, request_end, adjusted = enforce_free_plan_window(
                start_date.isoformat(), end_date.isoformat(), FREE_PLAN_WEEKS
            )
            if adjusted:
                st.sidebar.info(
                    f"フリー版の制限に合わせて開始日を {request_start} に調整しました。"
                )

            fetch_and_save_price_csv(
                download_symbol.strip(),
                request_start,
                request_end,
            )
            st.sidebar.success("ダウンロードに成功しました。")
            st.experimental_rerun()
        except Exception as exc:  # broad catch for user feedback
            st.sidebar.error(f"ダウンロードに失敗しました: {exc}")

    symbols = get_available_symbols()
    if not symbols:
        st.sidebar.warning("data/price_csv にCSVファイルがありません。")
        st.info("先に data/price_csv に株価CSVを置くか、J-Quantsからダウンロードしてください。")
        return

    selected_symbol = st.sidebar.selectbox("銘柄", symbols)

    lookback = st.sidebar.number_input(
        "過去本数 (N)",
        min_value=20,
        max_value=300,
        value=DEFAULT_LOOKBACK_BARS,
        step=5,
    )

    st.sidebar.markdown("---")
    st.sidebar.write("テクニカル表示")
    show_sma20 = st.sidebar.checkbox("SMA 20", value=True)
    show_sma50 = st.sidebar.checkbox("SMA 50", value=False)
    show_bbands = st.sidebar.checkbox("ボリンジャーバンド", value=False)

    st.sidebar.write("オシレーター")
    show_rsi = st.sidebar.checkbox("RSI (14)", value=True)
    show_macd = st.sidebar.checkbox("MACD (12,26,9)", value=True)
    show_stoch = st.sidebar.checkbox("ストキャスティクス (14,3)", value=False)

    # --- メイン処理 ---
    df = load_price_csv(selected_symbol)

    if len(df) < lookback:
        st.warning("過去本数に対してデータが不足しています。")
        return

    df_problem = df.iloc[-lookback:].copy()
    df_problem = _compute_indicators(df_problem)

    st.subheader(f"チャート（{selected_symbol}）")

    price_fig = go.Figure()
    price_fig.add_trace(
        go.Scatter(
            x=df_problem["date"],
            y=df_problem["close"],
            name="Close",
        )
    )
    if show_sma20:
        price_fig.add_trace(
            go.Scatter(
                x=df_problem["date"],
                y=df_problem["sma20"],
                name="SMA 20",
                line=dict(dash="dash"),
            )
        )
    if show_sma50:
        price_fig.add_trace(
            go.Scatter(
                x=df_problem["date"],
                y=df_problem["sma50"],
                name="SMA 50",
                line=dict(dash="dot"),
            )
        )
    if show_bbands:
        price_fig.add_trace(
            go.Scatter(
                x=df_problem["date"],
                y=df_problem["bb_upper"],
                name="BB Upper",
                line=dict(color="rgba(180,180,180,0.6)", dash="dot"),
            )
        )
        price_fig.add_trace(
            go.Scatter(
                x=df_problem["date"],
                y=df_problem["bb_lower"],
                name="BB Lower",
                line=dict(color="rgba(180,180,180,0.6)", dash="dot"),
                fill="tonexty",
                fillcolor="rgba(180,180,180,0.1)",
            )
        )

    price_fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Price",
        margin=dict(l=10, r=10, t=40, b=10),
    )

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

    osc_fig.update_layout(
        xaxis_title="Date",
        yaxis_title="RSI / Stoch",
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h"),
    )
    osc_fig.update_yaxes(title_text="MACD", secondary_y=True)

    st.plotly_chart(osc_fig, use_container_width=True)

if __name__ == "__main__":
    main()
