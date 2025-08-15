# multi_timeframe_pro_portofolio.py
# Multi-Timeframe Stock Analyzer Pro (1 ticker, 4 tab timeframe + kalkulasi portofolio)

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import ta
from plotly.subplots import make_subplots
import plotly.graph_objects as go

st.set_page_config(page_title="Multi-Timeframe Stock Analyzer Pro", layout="wide")
st.title("📊 Multi-Timeframe Stock Analyzer Pro")

# Sidebar input
with st.sidebar:
    st.markdown("## Pengaturan Analisis")
    ticker = st.text_input("Ticker (contoh: BBCA.JK atau AAPL)", value="BBCA.JK").upper().strip()
    avg_buy = st.number_input("Avg Buy (Rp per lembar)", min_value=0.0, value=0.0, step=0.01)
    lots = st.number_input("Jumlah lot (1 lot = 100 lembar)", min_value=0, value=0, step=1)
    st.markdown("---")
    st.markdown("💡 Tips: BEI gunakan .JK di akhir kode saham")

# Fungsi ambil data
def get_data(ticker, period, interval):
    try:
        df = yf.download(ticker, period=period, interval=interval, auto_adjust=True, progress=False)
        if df.empty:
            return None
        return df
    except:
        return None

# Resample 4H dari 1H
def resample_4h(df):
    return df.resample('4H').agg({'Open':'first','High':'max','Low':'min','Close':'last','Volume':'sum'}).dropna()

# Hitung indikator
def add_indicators(df):
    df['MA9'] = df['Close'].rolling(9).mean()
    df['RSI14'] = ta.momentum.RSIIndicator(df['Close'], 14).rsi()
    macd = ta.trend.MACD(df['Close'])
    df['MACD'] = macd.macd()
    df['MACD_signal'] = macd.macd_signal()
    atr = ta.volatility.AverageTrueRange(df['High'], df['Low'], df['Close'], 14)
    df['ATR14'] = atr.average_true_range()
    df['VolMA20'] = df['Volume'].rolling(20).mean()
    return df

# Hitung entry / TP / SL
def calc_trade_levels(df):
    last = df.iloc[-1]
    h, l, c = last['High'], last['Low'], last['Close']
    pivot = (h + l + c) / 3
    r1 = (2*pivot) - l
    s1 = (2*pivot) - h
    atr = last['ATR14'] if not pd.isna(last['ATR14']) else c*0.02
    entry = max([lvl for lvl in [last['MA9'], s1] if lvl <= c] or [c*0.99])
    tp = r1
    sl = s1
    return entry, tp, sl, pivot, r1, s1, atr

# Hitung portofolio
def calc_portfolio(avg_buy, lots, last_price):
    shares = lots * 100
    modal = avg_buy * shares
    nilai_pasar = last_price * shares
    laba_rugi = nilai_pasar - modal
    persen = (laba_rugi / modal * 100) if modal > 0 else 0
    return shares, modal, nilai_pasar, laba_rugi, persen

# Gambar chart
def plot_chart(df, entry, tp, sl, pivot, r1, s1):
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True,
                        row_heights=[0.5, 0.12, 0.18, 0.18],
                        vertical_spacing=0.02,
                        subplot_titles=("Harga", "Volume", "RSI(14)", "MACD"))
    # Price
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'],
                                 low=df['Low'], close=df['Close'], name="Price"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['MA9'], mode='lines', name='MA9', line=dict(color='orange')), row=1, col=1)
    # Garis level
    for y, name, color in [(entry,"Entry","#3498db"), (tp,"TP","#2ecc71"), (sl,"SL","#e74c3c"),
                           (pivot,"Pivot","#7f8c8d"), (r1,"R1","#e67e22"), (s1,"S1","#27ae60")]:
        fig.add_hline(y=y, line_dash="dot", line_color=color, annotation_text=name, row=1, col=1)
    # Volume
    fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name='Volume', marker_color='lightblue'), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['VolMA20'], mode='lines', name='VolMA20', line=dict(color='orange')), row=2, col=1)
    # RSI
    fig.add_trace(go.Scatter(x=df.index, y=df['RSI14'], mode='lines', name='RSI', line=dict(color='yellow')), row=3, col=1)
    fig.add_hline(y=70, line_dash="dot", line_color="red", row=3, col=1)
    fig.add_hline(y=30, line_dash="dot", line_color="green", row=3, col=1)
    # MACD
    fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], mode='lines', name='MACD', line=dict(color='cyan')), row=4, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['MACD_signal'], mode='lines', name='Signal', line=dict(color='magenta')), row=4, col=1)
    fig.add_hline(y=0, line_dash="dot", line_color="white", row=4, col=1)
    fig.update_layout(template="plotly_dark", height=900, showlegend=True)
    return fig

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["🕐 1H", "⏳ 4H", "📅 Daily", "📆 Weekly"])

timeframes = {
    "1H": ("60d", "60m", tab1),
    "4H": ("60d", "60m", tab2),  # resample dari 1H
    "Daily": ("1y", "1d", tab3),
    "Weekly": ("5y", "1wk", tab4)
}

for tf_name, (period, interval, tab) in timeframes.items():
    with tab:
        st.subheader(f"{tf_name} Chart - {ticker}")
        df = get_data(ticker, period, interval)
        if df is None:
            st.error("Data tidak tersedia")
            continue
        if tf_name == "4H":
            df = resample_4h(df)
        df = add_indicators(df)
        entry, tp, sl, pivot, r1, s1, atr = calc_trade_levels(df)
        last_price = df['Close'].iloc[-1]

        # Info teknikal
        st.markdown(f"**Entry:** {entry:.2f} | **TP:** {tp:.2f} | **SL:** {sl:.2f} | **ATR:** {atr:.2f}")

        # Portofolio
        shares, modal, nilai_pasar, laba_rugi, persen = calc_portfolio(avg_buy, lots, last_price)
        if lots > 0 and avg_buy > 0:
            status = "📈 UNTUNG" if laba_rugi > 0 else "📉 RUGI"
            st.markdown(f"""
            **Portofolio:**
            - Lot: **{lots}** ({shares} lembar)
            - Modal: Rp {modal:,.0f}
            - Nilai pasar: Rp {nilai_pasar:,.0f}
            - Laba/Rugi: Rp {laba_rugi:,.0f} ({persen:.2f}%)
            - Status: **{status}**
            """)

        fig = plot_chart(df, entry, tp, sl, pivot, r1, s1)
        st.plotly_chart(fig, use_container_width=True)