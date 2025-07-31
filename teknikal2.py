import streamlit as st
import yfinance as yf
import pandas as pd
import ta
import plotly.graph_objects as go

# Layout
st.set_page_config(page_title="Dashboard Analisis Saham", layout="wide")
st.title("📊 Dashboard Analisis Teknikal Saham")

# Input ticker
ticker = st.text_input("Masukkan kode saham (contoh: BRMS.JK):", "BRMS.JK")

# Ambil data
try:
    data = yf.download(ticker, period="3mo", interval="1d", auto_adjust=True)
    if data.empty:
        st.error("❌ Data tidak ditemukan. Periksa kode saham.")
        st.stop()
except Exception as e:
    st.error(f"Gagal mengambil data: {e}")
    st.stop()

data.dropna(inplace=True)
data.reset_index(inplace=True)

# Hitung indikator teknikal
close_series = data['Close'].squeeze()
data['MA9'] = close_series.rolling(window=9).mean()
data['RSI14'] = ta.momentum.RSIIndicator(close=close_series, window=14).rsi()
macd_obj = ta.trend.MACD(close=close_series)
data['MACD'] = macd_obj.macd()
data['MACD_signal'] = macd_obj.macd_signal()

# Pivot support & resistance
def calculate_pivot(df):
    high = df['High'].iloc[-1]
    low = df['Low'].iloc[-1]
    close = df['Close'].iloc[-1]
    pivot = (high + low + close) / 3
    r1 = (2 * pivot) - low
    s1 = (2 * pivot) - high
    r2 = pivot + (high - low)
    s2 = pivot - (high - low)
    return round(pivot, 2), round(r1, 2), round(r2, 2), round(s1, 2), round(s2, 2)

pivot, r1, r2, s1, s2 = calculate_pivot(data)

# Data terakhir
last_date = data['Date'].iloc[-1].strftime("%Y-%m-%d")
last_close = float(data['Close'].iloc[-1].item())
last_ma9 = float(data['MA9'].iloc[-1].item())
last_rsi = float(data['RSI14'].iloc[-1].item())
last_macd = float(data['MACD'].iloc[-1].item())
last_macd_signal = float(data['MACD_signal'].iloc[-1].item())

# Sinyal Otomatis
sinyal = ""
sinyal_waktu = last_date
if last_rsi < 30:
    sinyal = "✅ BUY SIGNAL: RSI oversold < 30"
elif last_macd > last_macd_signal and last_close > last_ma9 and last_rsi < 70:
    sinyal = "✅ BUY SIGNAL: MACD Golden Cross + harga di atas MA9"
elif last_rsi > 70:
    sinyal = "❌ SELL SIGNAL: RSI overbought > 70"
elif last_macd < last_macd_signal and last_close < last_ma9:
    sinyal = "⚠️ SELL SIGNAL: MACD Dead Cross + harga di bawah MA9"
else:
    sinyal = "⏸️ HOLD: Tidak ada sinyal kuat"

# Ringkasan metrik
st.subheader(f"📌 {ticker} - Harga Terakhir: Rp {round(last_close)} ({last_date})")
col1, col2, col3 = st.columns(3)
col1.metric("MA9", round(last_ma9, 2))
col2.metric("RSI (14)", round(last_rsi, 2))
col3.metric("MACD", round(last_macd, 2))

st.markdown(f"""
**📆 Tanggal Sinyal Terakhir:** `{sinyal_waktu}`  
### 📢 **Sinyal Otomatis:**  
**{sinyal}**

**Pivot**: `{pivot}`  
- Resistance 1: `{r1}`  
- Resistance 2: `{r2}`  
- Support 1: `{s1}`  
- Support 2: `{s2}`
""")

# Candlestick chart + MA9
fig = go.Figure()
fig.add_trace(go.Candlestick(x=data['Date'], open=data['Open'], high=data['High'],
                             low=data['Low'], close=data['Close'], name='Harga'))
fig.add_trace(go.Scatter(x=data['Date'], y=data['MA9'], mode='lines', name='MA9', line=dict(color='blue')))
fig.update_layout(title=f'Grafik Harga {ticker}', template='plotly_dark',
                  xaxis_title='Tanggal', yaxis_title='Harga', height=600)
st.plotly_chart(fig, use_container_width=True)

# RSI Chart
st.subheader("📉 RSI (14)")
fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=data['Date'], y=data['RSI14'], mode='lines', name='RSI (14)', line=dict(color='orange')))
fig2.add_hline(y=70, line_dash="dot", line_color="red")
fig2.add_hline(y=30, line_dash="dot", line_color="green")
fig2.update_layout(template='plotly_dark', height=300)
st.plotly_chart(fig2, use_container_width=True)

# MACD Chart
st.subheader("📉 MACD")
fig3 = go.Figure()
fig3.add_trace(go.Scatter(x=data['Date'], y=data['MACD'], mode='lines', name='MACD', line=dict(color='cyan')))
fig3.add_trace(go.Scatter(x=data['Date'], y=data['MACD_signal'], mode='lines', name='Signal Line', line=dict(color='yellow')))
fig3.update_layout(template='plotly_dark', height=300)
st.plotly_chart(fig3, use_container_width=True)
