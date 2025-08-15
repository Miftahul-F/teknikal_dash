import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import ta

# =====================
# Fungsi Hitung Indikator
# =====================
def compute_indicators(df):
    if df.empty:
        return df
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.dropna(subset=["Close"], inplace=True)

    if len(df) < 20:
        return df

    df["RSI14"] = ta.momentum.RSIIndicator(df["Close"], window=14).rsi()
    macd = ta.trend.MACD(df["Close"])
    df["MACD"] = macd.macd()
    df["MACD_signal"] = macd.macd_signal()
    df["EMA20"] = ta.trend.EMAIndicator(df["Close"], window=20).ema_indicator()
    df["EMA50"] = ta.trend.EMAIndicator(df["Close"], window=50).ema_indicator()
    return df

# =====================
# Fungsi Ambil Data
# =====================
def get_data(ticker, period="6mo", interval="1d"):
    try:
        data = yf.download(ticker, period=period, interval=interval)
        if data.empty:
            st.error(f"Data untuk {ticker} kosong.")
            return pd.DataFrame()
        data.reset_index(inplace=True)
        return compute_indicators(data)
    except Exception as e:
        st.error(f"Gagal mengambil data: {e}")
        return pd.DataFrame()

# =====================
# UI Streamlit
# =====================
st.set_page_config(layout="wide")
st.title("📊 Multi-Timeframe Stock Analyzer (Aman untuk Streamlit Cloud)")

col1, col2, col3 = st.columns(3)
with col1:
    ticker = st.text_input("Ticker (misal: GOTO, AAPL)", "GOTO").upper()
with col2:
    period = st.selectbox("Period", ["1mo", "3mo", "6mo", "1y", "2y"], index=2)
with col3:
    interval = st.selectbox("Timeframe", ["1d", "1h", "4h", "1wk"], index=0)

avg_buy = st.number_input("Harga beli rata-rata (opsional)", value=0.0)
lot = st.number_input("Jumlah lot (opsional)", value=0)

# Ambil data
df = get_data(ticker, period, interval)

if not df.empty: