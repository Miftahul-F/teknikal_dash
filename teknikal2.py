import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import ta
import pytz
from datetime import datetime

st.set_page_config(page_title="LQ45 Multi-Stock Analyzer", layout="wide")

# --- Daftar LQ45 ---
LQ45 = [
    "ADRO.JK","AKRA.JK","AMRT.JK","ANTM.JK","ARTO.JK","ASII.JK",
    "BBCA.JK","BBNI.JK","BBRI.JK","BBTN.JK","BMRI.JK","BRPT.JK",
    "BUKA.JK","CPIN.JK","ELSA.JK","ERAA.JK","EXCL.JK","GGRM.JK",
    "HMSP.JK","HRUM.JK","ICBP.JK","INDF.JK","INDY.JK","INKP.JK",
    "INTP.JK","ITMG.JK","JPFA.JK","JSMR.JK","KLBF.JK","MDKA.JK",
    "MEDC.JK","MIKA.JK","MNCN.JK","PGAS.JK","PTBA.JK","SCMA.JK",
    "SMGR.JK","TBIG.JK","TINS.JK","TKIM.JK","TLKM.JK","TOWR.JK",
    "UNTR.JK","UNVR.JK","WIKA.JK"
]

# --- Fungsi ambil data ---
def get_data(ticker):
    try:
        df = yf.download(ticker, period="6mo", interval="1d", progress=False)
        if df.empty or "Close" not in df:
            return None
        df.dropna(inplace=True)
        return df
    except Exception:
        return None

# --- Fungsi hitung indikator ---
def add_indicators(df):
    try:
        if df is None or df.empty:
            return None
        if df["Close"].isnull().all():
            return None

        df["RSI"] = ta.momentum.RSIIndicator(close=df["Close"], window=14).rsi()
        macd = ta.trend.MACD(close=df["Close"])
        df["MACD"] = macd.macd()
        df["Signal"] = macd.macd_signal()
        df["EMA20"] = ta.trend.EMAIndicator(close=df["Close"], window=20).ema_indicator()
        return df
    except Exception:
        return None

# --- Fungsi generate sinyal ---
def generate_signal(df):
    if df is None or df.empty:
        return "⚠️ Error Indicators"
    try:
        last = df.iloc[-1]
        if last["RSI"] < 30 and last["MACD"] > last["Signal"] and last["Close"] > last["EMA20"]:
            return "✅ BUY"
        elif last["RSI"] > 70 and last["MACD"] < last["Signal"] and last["Close"] < last["EMA20"]:
            return "❌ SELL"
        else:
            return "⏳ HOLD"
    except Exception:
        return "⚠️ Error Indicators"

# --- Cek jam WIB ---
wib = pytz.timezone("Asia/Jakarta")
now = datetime.now(wib)
st.caption(f"⏰ Sekarang: {now.strftime('%Y-%m-%d %H:%M:%S')} WIB")

# --- Tombol refresh manual ---
if st.button("🔄 Refresh Sekarang"):
    st.cache_data.clear()

# --- Auto-refresh harian (setiap habis market close jam 17:00 WIB) ---
if now.hour >= 17:
    st.cache_data.clear()

# --- Jalankan analisis ---
st.title("📊 LQ45 Multi-Stock Analyzer (Auto Refresh Daily)")

results = []
for ticker in LQ45:
    df = get_data(ticker)
    df = add_indicators(df)
    signal = generate_signal(df)
    results.append({"Ticker": ticker, "Signal": signal})

df_results = pd.DataFrame(results)

# Tampilkan tabel
st.dataframe(df_results)

# Filter rekomendasi beli
st.subheader("📌 Rekomendasi Beli")
st.dataframe(df_results[df_results["Signal"]=="✅ BUY"])