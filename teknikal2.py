import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import ta
import plotly.graph_objects as go

st.set_page_config(page_title="📊 Multi-Timeframe Stock Analyzer", layout="wide")

# ---------------- Fungsi Download Data ---------------- #
def download_data(ticker, period, interval):
    try:
        df = yf.download(
            ticker, 
            period=period, 
            interval=interval, 
            auto_adjust=True, 
            progress=False
        )

        if df.empty:
            return None

        # Kalau kolom MultiIndex (multi ticker), ambil ticker pertama
        if isinstance(df.columns, pd.MultiIndex):
            first_ticker = df.columns.levels[0][0]
            df = df[first_ticker]

        # Ambil kolom penting
        keep_cols = ["Open", "High", "Low", "Close", "Volume"]
        df = df[[col for col in keep_cols if col in df.columns]]

        # Nama kolom jadi Title Case
        df.columns = [str(c).title() for c in df.columns]

        return df

    except Exception as e:
        st.error(f"Gagal mengambil data: {e}")
        return None

# ---------------- Fungsi Hitung Indikator ---------------- #
def compute_indicators(df):
    if df is None or "Close" not in df.columns:
        st.error("Data tidak valid untuk hitung indikator.")
        st.stop()

    # Pastikan semua kolom numeric
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna().copy()

    if len(df) < 15:
        st.error("Data terlalu sedikit untuk hitung indikator.")
        st.stop()

    close = df["Close"]

    df["MA9"] = close.rolling(9).mean()
    df["RSI14"] = ta.momentum.RSIIndicator(close, window=14).rsi()
    macd_obj = ta.trend.MACD(close)
    df["MACD"] = macd_obj.macd()
    df["MACD_signal"] = macd_obj.macd_signal()
    df["ATR14"] = ta.volatility.AverageTrueRange(
        df["High"], df["Low"], close, window=14
    ).average_true_range()
    df["VolMA20"] = df["Volume"].rolling(20).mean()

    return df

# ---------------- Fungsi Rekomendasi ---------------- #
def get_recommendation(df):
    latest = df.iloc[-1]
    if latest["Close"] > latest["MA9"] and latest["RSI14"] > 50 and latest["MACD"] > latest["MACD_signal"]:
        return "BUY"
    elif latest["Close"] < latest["MA9"] and latest["RSI14"] < 50 and latest["MACD"] < latest["MACD_signal"]:
        return "SELL"
    else:
        return "HOLD"

# ---------------- Fungsi Plot ---------------- #
def plot_chart(df, ticker):
    fig = go.Figure()

    # Candlestick
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
        name="Candlestick"
    ))

    # MA9
    fig.add_trace(go.Scatter(
        x=df.index, y=df["MA9"], mode="lines", name="MA9", line=dict(color="orange")
    ))

    fig.update_layout(
        title=f"Chart {ticker}",
        xaxis_rangeslider_visible=False,
        template="plotly_dark",
        height=600
    )

    st.plotly_chart(fig, use_container_width=True)

# ---------------- UI ---------------- #
st.title("📊 Multi-Timeframe Stock Analyzer (1h / 4h / Daily / Weekly)")

ticker_input = st.text_input("Ticker (contoh: GOTO atau AAPL)").upper()
timeframe = st.selectbox("Timeframe", ["1h", "4h", "1d", "1wk"])
period = st.text_input("Period (misal: 60d / 1y / 5y)", "60d")
avg_buy = st.number_input("Avg Buy (Rp per lembar)", value=0.0, step=0.01)
lot = st.number_input("Jumlah Lot (1 lot = 100 lembar)", value=0, step=1)

if ticker_input:
    if not ticker_input.endswith(".JK") and ticker_input.isalpha() and len(ticker_input) <= 4:
        ticker_input += ".JK"

    df = download_data(ticker_input, period, "60m" if timeframe in ["1h", "4h"] else timeframe)

    if df is not None:
        # Resample 4h
        if timeframe == "4h":
            df = df.resample("4H").agg({
                "Open": "first",
                "High": "max",
                "Low": "min",
                "Close": "last",
                "Volume": "sum"
            }).dropna()

        df = compute_indicators(df)
        rec = get_recommendation(df)

        st.subheader(f"Rekomendasi: **{rec}**")
        st.dataframe(df.tail(10))

        plot_chart(df, ticker_input)

        # Hitung P/L jika input pembelian
        if avg_buy > 0 and lot > 0:
            last_price = df["Close"].iloc[-1]
            total_buy = avg_buy * lot * 100
            total_now = last_price * lot * 100
            pl = total_now - total_buy
            st.info(f"Last Price: Rp {last_price:,.2f} | P/L: Rp {pl:,.2f}")