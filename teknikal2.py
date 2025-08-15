import streamlit as st
import pandas as pd
import yfinance as yf
import ta
import numpy as np
import plotly.graph_objects as go

# -----------------------------
# Fungsi ambil data
# -----------------------------
def get_data(ticker, period="6mo", interval="1d"):
    try:
        df = yf.download(ticker, period=period, interval=interval)
        df.dropna(inplace=True)
        df.reset_index(inplace=True)
        return df
    except Exception as e:
        st.error(f"Gagal mengambil data: {e}")
        return pd.DataFrame()

# -----------------------------
# Tambah indikator teknikal
# -----------------------------
def add_indicators(df):
    if df.empty:
        return df
    df['MA9'] = df['Close'].rolling(window=9).mean()
    df['RSI14'] = ta.momentum.RSIIndicator(df['Close'], 14).rsi()
    macd = ta.trend.MACD(df['Close'])
    df['MACD'] = macd.macd()
    df['Signal'] = macd.macd_signal()
    df['ATR'] = ta.volatility.AverageTrueRange(df['High'], df['Low'], df['Close']).average_true_range()
    return df

# -----------------------------
# Cek arah tren
# -----------------------------
def check_trend(df):
    if df.empty:
        return None
    last_ma = df['MA9'].iloc[-1]
    last_price = df['Close'].iloc[-1]
    return "UP" if last_price > last_ma else "DOWN"

# -----------------------------
# Cek sinyal entry timeframe kecil
# -----------------------------
def check_entry(df):
    if df.empty:
        return False
    macd_val = df['MACD'].iloc[-1]
    macd_prev = df['MACD'].iloc[-2]
    signal_val = df['Signal'].iloc[-1]
    rsi_val = df['RSI14'].iloc[-1]
    return macd_val > signal_val and macd_prev <= df['Signal'].iloc[-2] and rsi_val > 50

# -----------------------------
# Cek status pembelian (dengan warna)
# -----------------------------
def check_buy_match(current_price, entry_price, tolerance=0.005):
    diff = abs(current_price - entry_price) / entry_price
    if diff <= tolerance:
        return ("✅ MATCH (Harga sudah di area entry)", "success")
    elif current_price < entry_price:
        return ("⏳ Belum nyentuh harga entry", "info")
    else:
        return ("⚠️ Harga sudah lewat dari entry", "error")

# -----------------------------
# Hitung confidence score
# -----------------------------
def get_confidence(trend_w, trend_d, entry_h4, entry_h1):
    score = 0
    if trend_w == "UP": score += 30
    if trend_d == "UP": score += 30
    if entry_h4: score += 20
    if entry_h1: score += 20
    return score

# -----------------------------
# Plot chart
# -----------------------------
def plot_chart(df, title):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df['Date'], open=df['Open'], high=df['High'],
        low=df['Low'], close=df['Close'], name="Candlestick"
    ))
    fig.add_trace(go.Scatter(x=df['Date'], y=df['MA9'], mode='lines', name='MA9', line=dict(color='orange')))
    fig.update_layout(title=title, xaxis_rangeslider_visible=False)
    return fig

# -----------------------------
# UI Streamlit
# -----------------------------
st.set_page_config(page_title="Multi-Timeframe Stock Analyzer", layout="wide")
st.title("📊 Multi-Timeframe Stock Analyzer Pro")

# Sidebar input
st.sidebar.header("Pengaturan Analisis")
ticker = st.sidebar.text_input("Ticker (contoh: BBCA.JK atau AAPL)", value="BBCA.JK")
avg_buy = st.sidebar.number_input("Avg Buy (Rp per lembar)", value=0.0, step=1.0)
lot = st.sidebar.number_input("Jumlah lot (1 lot = 100 lembar)", value=0, step=1)

# Ambil data
df_w = add_indicators(get_data(ticker, "2y", "1wk"))
df_d = add_indicators(get_data(ticker, "1y", "1d"))
df_h4 = add_indicators(get_data(ticker, "60d", "4h"))
df_h1 = add_indicators(get_data(ticker, "14d", "1h"))

# Analisis multi-timeframe
trend_weekly = check_trend(df_w)
trend_daily = check_trend(df_d)
entry_h4 = check_entry(df_h4)
entry_h1 = check_entry(df_h1)
confidence = get_confidence(trend_weekly, trend_daily, entry_h4, entry_h1)

# Tab layout
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📆 Weekly", "📅 Daily", "⏳ H4", "🕐 H1", "💡 Rekomendasi"])

with tab1:
    st.plotly_chart(plot_chart(df_w, "Weekly Chart"), use_container_width=True)

with tab2:
    st.plotly_chart(plot_chart(df_d, "Daily Chart"), use_container_width=True)

with tab3:
    st.plotly_chart(plot_chart(df_h4, "H4 Chart"), use_container_width=True)

with tab4:
    st.plotly_chart(plot_chart(df_h1, "H1 Chart"), use_container_width=True)

with tab5:
    st.subheader("📊 Hasil Analisis")
    st.write(f"**Weekly Trend**: {trend_weekly}")
    st.write(f"**Daily Trend**: {trend_daily}")
    st.write(f"**H4 Entry**: {'YES' if entry_h4 else 'NO'}")
    st.write(f"**H1 Entry**: {'YES' if entry_h1 else 'NO'}")
    st.progress(confidence / 100)
    st.write(f"🔥 Confidence Score: **{confidence}%**")

    if trend_weekly == "UP" and trend_daily == "UP" and (entry_h4 or entry_h1):
        last_close = df_d['Close'].iloc[-1]
        atr_val = df_d['ATR'].iloc[-1]
        entry_price = last_close
        stop_loss = entry_price - 1.5 * atr_val
        target = entry_price + 3 * atr_val

        st.success(f"✅ Rekomendasi BELI di sekitar Rp {entry_price:,.2f}")
        st.write(f"🎯 Target Price: Rp {target:,.2f}")
        st.write(f"🛑 Stop Loss: Rp {stop_loss:,.2f}")

        # Status pembelian berwarna
        msg, color = check_buy_match(last_close, entry_price)
        getattr(st, color)(f"📌 Status Pembelian: {msg}")
    else:
        st.warning("❌ Belum ada kondisi beli yang ideal (multi-timeframe tidak searah)")

    # Hitung P/L kalau ada pembelian
    if avg_buy > 0 and lot > 0:
        last_price = df_d['Close'].iloc[-1]
        pl = (last_price - avg_buy) * lot * 100
        st.info(f"💼 P/L saat ini: Rp {pl:,.0f}")