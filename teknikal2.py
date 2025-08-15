# Multi-Timeframe Stock Analyzer (Pro Version)
# Fitur:
# - Analisa 1h / 4h / Daily / Weekly
# - Auto tambah .JK untuk saham BEI
# - MA9, RSI14, MACD, ATR14, Volume MA20
# - Support & Resistance otomatis (pivot + swing high/low)
# - Entry, TP, SL otomatis
# - Rekomendasi BUY / HOLD / SELL
# - Chart interaktif Plotly
# - Portfolio tracking

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
st.title("📊 Multi-Timeframe Stock Analyzer (1h / 4h / Daily / Weekly)")

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
    period = st.text_input("Period (misal: 60d / 1y / 5y)", value=period_map[timeframe])

    st.markdown("---")
    st.markdown("### Data Pembelian (opsional)")
    avg_buy = st.number_input("Avg Buy (Rp per lembar)", min_value=0.0, step=0.01)
    lots = st.number_input("Jumlah lot (1 lot = 100 lembar)", min_value=0, step=1)

# ---------------------
# HELPER FUNCTIONS
# ---------------------
def download_data(ticker, period, interval):
    try:
        df = yf.download(ticker, period=period, interval=interval, auto_adjust=True, progress=False)
        if df.empty:
            return None
        df = df[["Open", "High", "Low", "Close", "Volume"]]
        return df
    except:
        return None

def resample_4h(df):
    return df.resample("4H").agg({"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}).dropna()

def compute_indicators(df):
    df["MA9"] = df["Close"].rolling(9).mean()
    df["RSI14"] = ta.momentum.RSIIndicator(df["Close"], window=14).rsi()
    macd_obj = ta.trend.MACD(df["Close"])
    df["MACD"] = macd_obj.macd()
    df["MACD_signal"] = macd_obj.macd_signal()
    df["ATR14"] = ta.volatility.AverageTrueRange(df["High"], df["Low"], df["Close"], window=14).average_true_range()
    df["VolMA20"] = df["Volume"].rolling(20).mean()
    return df

def pivot_levels(df):
    h, l, c = df["High"].iloc[-1], df["Low"].iloc[-1], df["Close"].iloc[-1]
    pivot = (h + l + c) / 3
    r1 = (2 * pivot) - l
    s1 = (2 * pivot) - h
    r2 = pivot + (h - l)
    s2 = pivot - (h - l)
    return pivot, r1, r2, s1, s2

def entry_tp_sl(df):
    pivot, r1, r2, s1, s2 = pivot_levels(df)
    c = df["Close"].iloc[-1]
    swing_res = df["High"].tail(10).max()
    swing_sup = df["Low"].tail(10).min()
    atr = df["ATR14"].iloc[-1] if not np.isnan(df["ATR14"].iloc[-1]) else c * 0.02

    entry = max([lvl for lvl in [df["MA9"].iloc[-1], pivot, s1, swing_sup] if lvl <= c], default=c * 0.99)
    tp = min([lvl for lvl in [r1, r2, swing_res] if lvl >= c], default=c + 1.5 * atr)
    sl = max([lvl for lvl in [s1, s2, swing_sup] if lvl <= c], default=c - 1.0 * atr)
    return entry, tp, sl, (pivot, r1, r2, s1, s2, swing_res, swing_sup)

def rekomendasi(row):
    try:
        if row["Close"] > row["MA9"] and row["RSI14"] < 70 and row["MACD"] > row["MACD_signal"]:
            return "BUY"
        elif row["Close"] < row["MA9"] and row["RSI14"] > 50 and row["MACD"] < row["MACD_signal"]:
            return "SELL"
        else:
            return "HOLD"
    except:
        return "WAIT"

# ---------------------
# FETCH DATA
# ---------------------
interval_map = {"1h": "60m", "4h": "60m", "Daily": "1d", "Weekly": "1wk"}
df_raw = download_data(ticker, period, interval_map[timeframe])
if df_raw is None:
    st.error("Gagal mengambil data. Periksa ticker atau koneksi internet.")
    st.stop()

if timeframe == "4h":
    df = resample_4h(df_raw)
else:
    df = df_raw.copy()

df = compute_indicators(df)

# ---------------------
# LAST VALUES
# ---------------------
last = df.iloc[-1]
entry, tp, sl, levels = entry_tp_sl(df)
pivot, r1, r2, s1, s2, swing_res, swing_sup = levels
rekomen = rekomendasi(last)

shares = lots * 100
if avg_buy > 0 and shares > 0:
    modal = avg_buy * shares
    nilai_now = last["Close"] * shares
    pnl = nilai_now - modal
    pnl_pct = (pnl / modal) * 100
else:
    modal = nilai_now = pnl = pnl_pct = 0

# ---------------------
# SUMMARY
# ---------------------
col1, col2, col3, col4 = st.columns(4)
col1.metric("Ticker", ticker)
col2.metric("Harga Sekarang", f"{last['Close']:.2f}")
col3.metric("Rekomendasi", rekomen)
col4.metric("Timeframe", timeframe)

st.markdown(f"**Entry**: {entry:.2f} • **TP**: {tp:.2f} • **SL**: {sl:.2f}")
st.markdown(f"**Pivot | R1 | R2**: {pivot:.2f} | {r1:.2f} | {r2:.2f}")
st.markdown(f"**Swing High(10)**: {swing_res:.2f} • **Swing Low(10)**: {swing_sup:.2f}")
st.markdown(f"**ATR(14)**: {last['ATR14']:.2f} • **Volume**: {last['Volume']:,} • **VolMA20**: {int(last['VolMA20']) if not np.isnan(last['VolMA20']) else '-'}")

if shares > 0:
    st.subheader("💼 Portofolio")
    st.write(f"Qty: {shares} lembar • Modal: {modal:,.0f} • Nilai: {nilai_now:,.0f}")
    st.write(f"P/L: {pnl:,.0f} • P/L %: {pnl_pct:.2f}%")

# ---------------------
# CHART
# ---------------------
fig = make_subplots(rows=4, cols=1, shared_xaxes=True, row_heights=[0.5, 0.12, 0.18, 0.18],
                    subplot_titles=("Harga & MA9", "Volume", "RSI(14)", "MACD"))

fig.add_trace(go.Candlestick(x=df.index, open=df["Open"], high=df["High"],
                             low=df["Low"], close=df["Close"], name="Harga"), row=1, col=1)
fig.add_trace(go.Scatter(x=df.index, y=df["MA9"], mode="lines", name="MA9", line=dict(color="orange")), row=1, col=1)

for y, name, color in [(entry, "Entry", "#3498db"), (tp, "TP", "#2ecc71"), (sl, "SL", "#e74c3c")]:
    fig.add_hline(y=y, line_dash="dot", line_color=color, annotation_text=name, annotation_position="top right", row=1, col=1)

fig.add_trace(go.Bar(x=df.index, y=df["Volume"], name="Volume", marker_color="lightblue"), row=2, col=1)
fig.add_trace(go.Scatter(x=df.index, y=df["VolMA20"], mode="lines", name="VolMA20", line=dict(color="orange")), row=2, col=1)

fig.add_trace(go.Scatter(x=df.index, y=df["RSI14"], mode="lines", name="RSI", line=dict(color="yellow")), row=3, col=1)
fig.add_hline(y=70, line_dash="dot", line_color="red", row=3, col=1)
fig.add_hline(y=30, line_dash="dot", line_color="green", row=3, col=1)

fig.add_trace(go.Scatter(x=df.index, y=df["MACD"], mode="lines", name="MACD", line=dict(color="cyan")), row=4, col=1)
fig.add_trace(go.Scatter(x=df.index, y=df["MACD_signal"], mode="lines", name="Signal", line=dict(color="magenta")), row=4, col=1)
fig.add_hline(y=0, line_dash="dot", line_color="white", row=4, col=1)

fig.update_layout(template="plotly_dark", height=900, showlegend=True, xaxis_rangeslider_visible=False)
st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.markdown("**Catatan:**\n- Analisa teknikal sederhana, gunakan manajemen risiko.\n- Untuk intraday, data historis terbatas sesuai periode.")