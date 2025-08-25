# streamlit_nextday_plan.py
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import ta
from datetime import date

st.set_page_config(page_title="Next Day Stock Plan", layout="wide")
st.title("📊 Next Day Stock Plan (EOD Strategy)")

# -------------------------
# Input
# -------------------------
ticker = st.text_input("Ticker (contoh: BBCA.JK)", value="BBCA.JK").upper().strip()
period = st.selectbox("Periode data", ["3mo", "6mo", "1y"], index=1)

# -------------------------
# Download Data
# -------------------------
df = yf.download(ticker, period=period, interval="1d", auto_adjust=True, progress=False)

if df.empty:
    st.error("Data tidak ditemukan. Periksa ticker.")
    st.stop()

# -------------------------
# Indikator
# -------------------------
df['MA9'] = df['Close'].rolling(9).mean()
df['RSI'] = ta.momentum.RSIIndicator(close=df['Close'], window=14).rsi()
macd = ta.trend.MACD(close=df['Close'])
df['MACD'] = macd.macd()
df['MACD_signal'] = macd.macd_signal()
atr = ta.volatility.AverageTrueRange(df['High'], df['Low'], df['Close'], window=14)
df['ATR'] = atr.average_true_range()

last = df.iloc[-1]
close = last['Close']
ma9 = last['MA9']
rsi = last['RSI']
macd_val = last['MACD']
sig = last['MACD_signal']
atr_val = last['ATR']

# -------------------------
# Rekomendasi Entry Plan
# -------------------------
entry = close  # harga close hari ini → acuan entry besok
tp = close + 1.5 * atr_val
sl = close - 1.0 * atr_val

# Simple rekomendasi
if close > ma9 and macd_val > sig and rsi < 70:
    rekom = "BUY"
elif close < ma9 and macd_val < sig and rsi > 50:
    rekom = "SELL"
else:
    rekom = "HOLD"

# -------------------------
# Output
# -------------------------
col1, col2, col3 = st.columns(3)
col1.metric("Ticker", ticker)
col2.metric("Harga Close", f"{close:,.2f}")
col3.metric("Rekomendasi", rekom)

st.markdown(f"**Entry Besok**: {entry:.2f} • **TP**: {tp:.2f} • **SL**: {sl:.2f}")
st.markdown(f"MA9: {ma9:.2f} | RSI(14): {rsi:.2f} | MACD: {macd_val:.2f} vs Signal {sig:.2f}")
st.markdown(f"ATR(14): {atr_val:.2f}")

# Chart
st.line_chart(df[['Close','MA9']])