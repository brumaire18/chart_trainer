from datetime import date, timedelta

import plotly.graph_objects as go
import streamlit as st

from app.config import DEFAULT_LOOKBACK_BARS
from app.data_loader import (
    fetch_and_save_price_csv,
    get_available_symbols,
    load_price_csv,
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
    default_start = date.today() - timedelta(days=365)
    default_end = date.today()
    start_date = st.sidebar.date_input("開始日", value=default_start)
    end_date = st.sidebar.date_input("終了日", value=default_end)

    if st.sidebar.button("J-Quantsからダウンロード"):
        try:
            if start_date > end_date:
                raise ValueError("終了日は開始日以降にしてください。")

            fetch_and_save_price_csv(
                download_symbol.strip(),
                start_date.isoformat(),
                end_date.isoformat(),
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
    st.sidebar.write("テクニカル表示（フェーズ1はSMAのみ）")
    show_sma20 = st.sidebar.checkbox("SMA 20", value=True)
    show_sma50 = st.sidebar.checkbox("SMA 50", value=False)

    # --- メイン処理 ---
    df = load_price_csv(selected_symbol)

    if len(df) < lookback:
        st.warning("過去本数に対してデータが不足しています。")
        return

    df_problem = df.iloc[-lookback:].copy()
    df_problem["sma20"] = df_problem["close"].rolling(20).mean()
    df_problem["sma50"] = df_problem["close"].rolling(50).mean()

    st.subheader(f"チャート（{selected_symbol}）")

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df_problem["date"],
            y=df_problem["close"],
            name="Close",
        )
    )
    if show_sma20:
        fig.add_trace(
            go.Scatter(
                x=df_problem["date"],
                y=df_problem["sma20"],
                name="SMA 20",
                line=dict(dash="dash"),
            )
        )
    if show_sma50:
        fig.add_trace(
            go.Scatter(
                x=df_problem["date"],
                y=df_problem["sma50"],
                name="SMA 50",
                line=dict(dash="dot"),
            )
        )

    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Price",
        margin=dict(l=10, r=10, t=40, b=10),
    )

    st.plotly_chart(fig, use_container_width=True)

    st.info("ここから『買い / 売り / 様子見』や目標値・損切の入力UIを足していきます。")

if __name__ == "__main__":
    main()
