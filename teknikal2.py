# teknikal2.py
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import ta

st.set_page_config(page_title="Stock Analyzer", layout="wide")

st.title("📊 Multi-Timeframe Stock Analyzer Pro")

# ================================
# Fungsi Fetch Data
# ================================
@st.cache_data(show_spinner=False, ttl=30*60)
def fetch_ohlc(ticker_no_suffix: str):
    """Download OHLCV, auto append .JK, fallback period kalau kosong."""
    t = (ticker_no_suffix or "").strip().upper()
    if not t:
        return None, "Ticker kosong"
    if not t.endswith(".JK"):
        t += ".JK"

    for period in ["6mo", "1y", "2y"]:
        try:
            df = yf.download(
                t, period=period, interval="1d",
                auto_adjust=True, progress=False
            )
        except Exception as e:
            return None, f"Download error: {e}"

        if df is not None and not df.empty:
            # pastikan numeric
            for c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
            df = df.dropna(subset=["Close"])
            if not df.empty:
                return df, None

    return None, f"Tidak ada data untuk {t}"

# ================================
# Fungsi Tambah Indikator
# ================================
def add_indicators(df: pd.DataFrame):
    if df is None or df.empty:
        return None
    try:
        df["RSI14"] = ta.momentum.RSIIndicator(df["Close"], window=14).rsi()
        macd = ta.trend.MACD(df["Close"])
        df["MACD"] = macd.macd()
        df["MACD_SIGNAL"] = macd.macd_signal()
        df["MA20"] = df["Close"].rolling(20).mean()
        df["MA50"] = df["Close"].rolling(50).mean()
    except Exception as e:
        st.warning(f"⚠️ Gagal hitung indikator: {e}")
        return None
    return df

# ================================
# UI Input
# ================================
st.sidebar.header("⚙️ Pengaturan Analisis")
ticker = st.sidebar.text_input("Ticker (contoh: BBCA atau AAPL)", value="BBCA")
avg_buy = st.sidebar.number_input("Avg Buy (Rp per lembar)", value=0.0)
lot = st.sidebar.number_input("Jumlah lot (1 lot = 100 lembar)", value=0)

# ================================
# Main App
# ================================
df, err = fetch_ohlc(ticker)
if err:
    st.error(err)
elif df is None or df.empty:
    st.warning("⚠️ Data tidak ditemukan.")
else:
    df = add_indicators(df)

    if df is not None:
        st.subheader(f"📈 Grafik & Analisis — {ticker.upper()}.JK")
        st.line_chart(df[["Close", "MA20", "MA50"]])

        latest = df.iloc[-1]
        st.write("### 📊 Ringkasan Hari Ini")
        st.write(f"📅 Tanggal: {latest.name.date()}")
        st.write(f"💰 Close: {latest['Close']:.2f}")
        st.write(f"📈 RSI(14): {latest['RSI14']:.2f}")
        st.write(f"📉 MACD: {latest['MACD']:.2f} | Signal: {latest['MACD_SIGNAL']:.2f}")

        if avg_buy > 0 and lot > 0:
            total_buy = avg_buy * lot * 100
            total_now = latest["Close"] * lot * 100
            profit = total_now - total_buy
            st.write(f"💼 Estimasi nilai portofolio: Rp {total_now:,.0f}")
            st.write(f"📊 Profit/Loss: Rp {profit:,.0f}")

        # rekomendasi sederhana
        rec = "⏳ Tahan"
        if latest["RSI14"] < 30:
            rec = "✅ Potensi Beli (Oversold)"
        elif latest["RSI14"] > 70:
            rec = "⚠️ Potensi Jual (Overbought)"
        elif latest["MACD"] > latest["MACD_SIGNAL"]:
            rec = "📈 Momentum Naik"
        elif latest["MACD"] < latest["MACD_SIGNAL"]:
            rec = "📉 Momentum Turun"

        st.success(f"📌 Rekomendasi besok pagi: {rec}")

        with st.expander("📜 Data Tabel"):
            st.dataframe(df.tail(30))