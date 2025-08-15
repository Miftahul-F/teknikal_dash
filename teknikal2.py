import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import ta

# ----------------------------
# Fungsi ambil data
# ----------------------------
def get_data(ticker, period="6mo", interval="1d"):
    try:
        data = yf.download(ticker, period=period, interval=interval)
        if data.empty:
            st.warning("Data kosong. Coba ticker, periode, atau interval lain.")
            return pd.DataFrame()
        data = data.dropna().reset_index()
        return data
    except Exception as e:
        st.error(f"Gagal mengambil data: {e}")
        return pd.DataFrame()

# ----------------------------
# Fungsi hitung indikator
# ----------------------------
def compute_indicators(df):
    try:
        # Pastikan kolom numerik
        numeric_cols = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)

        # Hitung indikator
        df["RSI14"] = ta.momentum.RSIIndicator(df["Close"], window=14).rsi()
        df["SMA20"] = ta.trend.SMAIndicator(df["Close"], window=20).sma_indicator()
        df["EMA20"] = ta.trend.EMAIndicator(df["Close"], window=20).ema_indicator()

        macd = ta.trend.MACD(df["Close"])
        df["MACD"] = macd.macd()
        df["MACD_signal"] = macd.macd_signal()

        return df
    except Exception as e:
        st.error(f"Error hitung indikator: {e}")
        return df

# ----------------------------
# Fungsi rekomendasi sederhana
# ----------------------------
def generate_signal(df):
    if df.empty:
        return "No Data"
    latest = df.iloc[-1]
    if latest["RSI14"] < 30 and latest["MACD"] > latest["MACD_signal"]:
        return "BUY"
    elif latest["RSI14"] > 70 and latest["MACD"] < latest["MACD_signal"]:
        return "SELL"
    else:
        return "HOLD"

# ----------------------------
# UI Streamlit
# ----------------------------
st.set_page_config(page_title="📊 Multi-Timeframe Stock Analyzer", layout="wide")
st.title("📊 Multi-Timeframe Stock Analyzer")

with st.sidebar:
    st.header("Input & Pengaturan")
    ticker = st.text_input("Ticker (contoh: GOTO.JK / AAPL)", value="BBRI.JK")
    period = st.selectbox("Period", ["1mo", "3mo", "6mo", "1y", "5y"], index=2)
    interval = st.selectbox("Timeframe", ["1h", "4h", "1d", "1wk"], index=2)
    avg_buy = st.number_input("Avg Buy (Rp per lembar)", value=0.0, step=1.0)
    lot = st.number_input("Jumlah Lot", value=0, step=1)

if ticker:
    df = get_data(ticker, period, interval)

    if not df.empty:
        df = compute_indicators(df)
        st.subheader(f"📈 Data {ticker} - {interval}")
        st.dataframe(df.tail())

        signal = generate_signal(df)
        st.markdown(f"### 🔍 Rekomendasi: **{signal}**")

        # Hitung profit/loss jika input pembelian diisi
        if avg_buy > 0 and lot > 0:
            latest_price = df["Close"].iloc[-1]
            invest_value = avg_buy * lot * 100
            current_value = latest_price * lot * 100
            pl = current_value - invest_value
            st.metric("Profit/Loss", f"Rp {pl:,.0f}")

        st.line_chart(df.set_index("Date")[["Close", "SMA20", "EMA20"]])