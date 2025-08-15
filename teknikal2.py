import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import ta

# -----------------------------
# Fungsi: Ambil Data Saham
# -----------------------------
def load_data(ticker, period="60d", interval="1d"):
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False)
        if df.empty:
            st.error("Data tidak ditemukan. Periksa ticker atau period.")
            return None
        df.reset_index(inplace=True)
        return df
    except Exception as e:
        st.error(f"Gagal mengambil data: {e}")
        return None

# -----------------------------
# Fungsi: Hitung Indikator
# -----------------------------
def compute_indicators(df):
    df = df.copy()
    if 'Close' not in df.columns:
        st.error("Data tidak memiliki kolom 'Close'.")
        return df

    df["EMA20"] = ta.trend.ema_indicator(df["Close"], window=20)
    df["EMA50"] = ta.trend.ema_indicator(df["Close"], window=50)
    df["RSI14"] = ta.momentum.rsi(df["Close"], window=14)
    macd = ta.trend.macd_diff(df["Close"])
    df["MACD_diff"] = macd
    return df

# -----------------------------
# Fungsi: Analisis Sinyal
# -----------------------------
def get_signal(df):
    if df.empty or len(df) < 2:
        return "No Data"

    latest = df.iloc[-1]
    ema_signal = "Buy" if latest["EMA20"] > latest["EMA50"] else "Sell"
    rsi_signal = "Overbought" if latest["RSI14"] > 70 else "Oversold" if latest["RSI14"] < 30 else "Neutral"

    if ema_signal == "Buy" and rsi_signal != "Overbought":
        return "Buy"
    elif ema_signal == "Sell" and rsi_signal != "Oversold":
        return "Sell"
    else:
        return "Hold"

# -----------------------------
# UI Streamlit
# -----------------------------
st.set_page_config(page_title="Multi-Timeframe Stock Analyzer", layout="wide")
st.title("📊 Multi-Timeframe Stock Analyzer (Pro Version)")

col1, col2, col3 = st.columns(3)
with col1:
    ticker = st.text_input("Ticker (contoh: PWON.JK)", value="PWON.JK")
with col2:
    period = st.text_input("Period (misal: 60d / 1y)", value="60d")
with col3:
    avg_buy = st.number_input("Avg Buy (Rp per lembar)", min_value=0.0, value=0.0)

lot = st.number_input("Jumlah Lot", min_value=0, value=0)

# -----------------------------
# Ambil data multi-timeframe
# -----------------------------
timeframes = {
    "1h": ("7d", "1h"),
    "4h": ("60d", "4h"),
    "Daily": (period, "1d"),
    "Weekly": ("2y", "1wk")
}

results = {}
for tf_name, (tf_period, tf_interval) in timeframes.items():
    df_tf = load_data(ticker, tf_period, tf_interval)
    if df_tf is not None:
        df_tf = compute_indicators(df_tf)
        signal = get_signal(df_tf)
        results[tf_name] = (df_tf, signal)

# -----------------------------
# Tampilkan hasil
# -----------------------------
st.subheader("📈 Hasil Analisis Multi-Timeframe")
for tf_name, (df_tf, signal) in results.items():
    latest_close = df_tf["Close"].iloc[-1]
    st.markdown(f"**{tf_name}** — Harga Terakhir: Rp{latest_close:,.2f} — Sinyal: **{signal}**")

# -----------------------------
# Hitung keuntungan / kerugian
# -----------------------------
if avg_buy > 0 and lot > 0 and "Daily" in results:
    last_price = results["Daily"][0]["Close"].iloc[-1]
    profit_loss = (last_price - avg_buy) * lot * 100
    st.subheader("💰 Kalkulasi Posisi")
    st.write(f"Nilai Investasi: Rp {lot * 100 * avg_buy:,.0f}")
    st.write(f"Nilai Sekarang: Rp {lot * 100 * last_price:,.0f}")
    st.write(f"Profit / Loss: **Rp {profit_loss:,.0f}**")

# -----------------------------
# Tabel Dataframe
# -----------------------------
st.subheader("📄 Data Harga (Daily)")
if "Daily" in results:
    st.dataframe(results["Daily"][0].tail(20))