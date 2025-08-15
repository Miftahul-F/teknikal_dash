# Multi-Timeframe Stock Analyzer (Pro Version) - Anti Crash
# Fitur: 1h / 4h / Daily / Weekly dengan proteksi data kosong dan error indikator

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import ta
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# ---------------------
# CONFIG
# ---------------------
st.set_page_config(page_title="Multi-Timeframe Stock Analyzer", layout="wide")
st.title("📊 Multi-Timeframe Stock Analyzer (Anti-Crash)")

# ---------------------
# SIDEBAR
# ---------------------
with st.sidebar:
    st.markdown("## Input & Pengaturan")
    ticker_input = st.text_input("Ticker (contoh: GOTO atau AAPL)", value="GOTO").upper().strip()
    if not ticker_input.endswith(".JK") and len(ticker_input) <= 4:
        ticker = ticker_input + ".JK"
    else:
        ticker = ticker_input

    timeframe = st.selectbox("Timeframe", ["1h", "4h", "Daily", "Weekly"], index=2)
    period_map = {"1h": "60d", "4h": "60d", "Daily": "1y", "Weekly": "5y"}
    period = st.text_input("Period", value=period_map[timeframe])

    st.markdown("---")
    st.markdown("### Data Pembelian (opsional)")
    avg_buy = st.number_input("Avg Buy (Rp per lembar)", min_value=0.0, step=0.01)
    lots = st.number_input("Jumlah lot (1 lot = 100 lembar)", min_value=0, step=1)

# ---------------------
# FUNCTIONS
# ---------------------
def download_data(ticker, period, interval):
    try:
        df = yf.download(ticker, period=period, interval=interval, auto_adjust=True, progress=False)
        if df.empty:
            return None
        return df[["Open", "High", "Low", "Close", "Volume"]]
    except:
        return None

def resample_4h(df):
    return df.resample("4H").agg({"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}).dropna()

def compute_indicators(df):
    if "Close" not in df.columns:
        st.error("Data tidak memiliki kolom Close.")
        st.stop()

    df = df.copy()
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df["High"] = pd.to_numeric(df["High"], errors="coerce")
    df["Low"] = pd.to_numeric(df["Low"], errors="coerce")
    df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce")
    df = df.dropna()

    if len(df) < 15:
        st.error("Data terlalu sedikit untuk menghitung indikator.")
        st.stop