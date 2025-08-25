import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import ta
import pytz
from datetime import datetime

st.set_page_config(page_title="📊 Indonesia Stock Analyzer", layout="wide")

# --- Fungsi ambil data ---
def get_data(ticker):
    try:
        if not ticker.endswith(".JK"):
            ticker = ticker + ".JK"
        df = yf.download(ticker, period="6mo", interval="1d", progress=False)
        if df.empty or "Close" not in df:
            return None
        df.dropna(inplace=True)
        return df
    except Exception:
        return None

# --- Fungsi indikator ---
def add_indicators(df):
    try:
        if df is None or df.empty or df["Close"].isnull().all():
            return None
        df["RSI"] = ta.momentum.RSIIndicator(close=df["Close"], window=14).rsi()
        macd = ta.trend.MACD(close=df["Close"])
        df["MACD"] = macd.macd()
        df["Signal"] = macd.macd_signal()
        df["EMA20"] = ta.trend.EMAIndicator(close=df["Close"], window=20).ema_indicator()
        return df
    except Exception:
        return None

# --- Fungsi sinyal + estimasi beli ---
def generate_signal(df):
    if df is None or df.empty:
        return "⚠️ Error", None
    try:
        last = df.iloc[-1]
        if last["RSI"] < 30 and last["MACD"] > last["Signal"] and last["Close"] > last["EMA20"]:
            est_buy = round(last["Close"] * 1.01, 2)   # estimasi buy besok pagi = close hari ini +1%
            return "✅ BUY", est_buy
        elif last["RSI"] > 70 and last["MACD"] < last["Signal"] and last["Close"] < last["EMA20"]:
            return "❌ SELL", None
        else:
            return "⏳ HOLD", None
    except Exception:
        return "⚠️ Error", None

# --- Timezone Jakarta ---
wib = pytz.timezone("Asia/Jakarta")
now = datetime.now(wib)
st.caption(f"⏰ Sekarang: {now.strftime('%Y-%m-%d %H:%M:%S')} WIB")

# --- Input pencarian ticker ---
st.title("📊 Indonesia Stock Analyzer (All IDX Stocks)")
ticker_input = st.text_input("Masukkan kode saham (contoh: BBCA, BBRI, TLKM):")

if ticker_input:
    df = get_data(ticker_input)
    df = add_indicators(df)
    signal, est_buy = generate_signal(df)

    if df is not None:
        st.subheader(f"📈 Analisis {ticker_input.upper()}.JK")
        last = df.iloc[-1]
        st.write(f"""
        - Harga Close: **Rp {last['Close']:.2f}**
        - RSI(14): **{last['RSI']:.2f}**
        - EMA20: **Rp {last['EMA20']:.2f}**
        - Signal: {signal}
        """)

        if est_buy:
            st.success(f"💡 Estimasi harga beli besok pagi: **Rp {est_buy:.2f}**")
    else:
        st.error("⚠️ Data tidak ditemukan atau error indikator.")