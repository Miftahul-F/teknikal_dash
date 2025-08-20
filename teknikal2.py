# streamlit_tv.py
import streamlit as st
import pandas as pd
import numpy as np
import ta
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from tvDatafeed import TvDatafeed, Interval

st.set_page_config(page_title="TradingView Analyzer", layout="wide")
st.title("📊 Stock Analyzer via TradingView")

# Sidebar input
with st.sidebar:
    st.markdown("## Input")
    ticker = st.text_input("Ticker (contoh: BBCA untuk IDX, AAPL untuk NASDAQ)", value="BBCA").upper().strip()
    exchange = st.text_input("Exchange (contoh: IDX, NASDAQ, NYSE)", value="IDX").upper().strip()
    timeframe = st.selectbox("Timeframe", ["1m","5m","15m","1h","1d","1W"], index=4)
    n_bars = st.number_input("Jumlah bar data", min_value=100, max_value=2000, value=500, step=100)

# Map timeframe
interval_map = {
    "1m": Interval.in_1_minute,
    "5m": Interval.in_5_minute,
    "15m": Interval.in_15_minute,
    "1h": Interval.in_1_hour,
    "1d": Interval.in_daily,
    "1W": Interval.in_weekly
}

# Init tvdatafeed
tv = TvDatafeed()  # anonymous mode

# Download OHLCV
df = tv.get_hist(symbol=ticker, exchange=exchange, interval=interval_map[timeframe], n_bars=n_bars)

if df is None or df.empty:
    st.error("Tidak ada data. Periksa ticker & exchange.")
    st.stop()

# Compute indicators
df['MA9'] = df['close'].rolling(9).mean()
df['RSI14'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
macd_obj = ta.trend.MACD(df['close'])
df['MACD'] = macd_obj.macd()
df['Signal'] = macd_obj.macd_signal()
df['VolMA20'] = df['volume'].rolling(20).mean()

last = df.iloc[-1]
rekom = "BUY" if last['close'] > last['MA9'] and last['MACD'] > last['Signal'] and last['RSI14'] < 70 else "SELL" if last['close'] < last['MA9'] and last['MACD'] < last['Signal'] else "HOLD"

# Show metrics
col1, col2, col3 = st.columns(3)
col1.metric("Ticker", f"{ticker}.{exchange}")
col2.metric("Harga Sekarang", f"{last['close']:.2f}")
col3.metric("Rekomendasi", rekom)

st.markdown("### 📈 Chart Interaktif")
fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                    row_heights=[0.6, 0.2, 0.2], vertical_spacing=0.02,
                    subplot_titles=("Harga + MA9", "RSI", "MACD"))

# Candlestick
fig.add_trace(go.Candlestick(x=df.index, open=df['open'], high=df['high'],
                             low=df['low'], close=df['close'], name="Price"), row=1, col=1)
fig.add_trace(go.Scatter(x=df.index, y=df['MA9'], mode='lines', name='MA9', line=dict(color='orange')), row=1, col=1)

# RSI
fig.add_trace(go.Scatter(x=df.index, y=df['RSI14'], mode='lines', name='RSI', line=dict(color='yellow')), row=2, col=1)
fig.add_hline(y=70, line_dash="dot", line_color="red", row=2, col=1)
fig.add_hline(y=30, line_dash="dot", line_color="green", row=2, col=1)

# MACD
fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], mode='lines', name='MACD', line=dict(color='cyan')), row=3, col=1)
fig.add_trace(go.Scatter(x=df.index, y=df['Signal'], mode='lines', name='Signal', line=dict(color='magenta')), row=3, col=1)
fig.add_hline(y=0, line_dash="dot", line_color="white", row=3, col=1)

fig.update_layout(template="plotly_dark", height=900, xaxis_rangeslider_visible=False, showlegend=True)
st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.info("Data diambil dari TradingView via `tvdatafeed`. Untuk realtime penuh butuh websocket TradingView (unofficial).")