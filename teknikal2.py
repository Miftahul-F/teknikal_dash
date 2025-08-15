import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import ta

# Judul Aplikasi
st.title("📊 Multi-Timeframe Stock Analyzer (Lite Version)")

# Input & Pengaturan
ticker = st.text_input("Ticker (contoh: GOTO.JK atau AAPL)", "GOTO.JK")
period = st.selectbox("Period", ["1mo", "3mo", "6mo", "1y", "2y"], index=1)
interval = st.selectbox("Timeframe", ["1d", "1wk", "1mo"], index=0)

avg_buy = st.number_input("Avg Buy (Rp per lembar)", min_value=0.0, value=0.0)
lot = st.number_input("Jumlah lot", min_value=0, value=0)

if st.button("Ambil Data"):
    try:
        df = yf.download(ticker, period=period, interval=interval)

        if df.empty:
            st.error("Data tidak ditemukan. Coba ticker lain.")
        else:
            df.reset_index(inplace=True)
            df["RSI14"] = ta.momentum.RSIIndicator(df["Close"], window=14).rsi()
            df["SMA20"] = ta.trend.SMAIndicator(df["Close"], window=20).sma_indicator()
            df["SMA50"] = ta.trend.SMAIndicator(df["Close"], window=50).sma_indicator()

            st.subheader("📈 Data Harga")
            st.dataframe(df.tail(10))

            st.subheader("📊 Indikator Terbaru")
            last_row = df.iloc[-1]
            st.markdown(f"**Close:** {last_row['Close']:.2f}")
            st.markdown(f"**RSI(14):** {last_row['RSI14']:.2f}")
            st.markdown(f"**SMA20:** {last_row['SMA20']:.2f}")
            st.markdown(f"**SMA50:** {last_row['SMA50']:.2f}")

            if avg_buy > 0 and lot > 0:
                total_buy = avg_buy * lot * 100
                total_now = last_row['Close'] * lot * 100
                pl = total_now - total_buy
                st.subheader("💰 Profit / Loss")
                st.markdown(f"**Total Beli:** Rp {total_buy:,.0f}")
                st.markdown(f"**Total Sekarang:** Rp {total_now:,.0f}")
                st.markdown(f"**P/L:** Rp {pl:,.0f}")

    except Exception as e:
        st.error(f"Error mengambil data: {e}")