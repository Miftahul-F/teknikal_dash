import streamlit as st
import yfinance as yf
import pandas as pd
import ta

st.set_page_config(page_title="Analisis Saham Sederhana", layout="centered")
st.title("📊 Apakah Saham Ini Layak Dibeli atau Dijual?")

ticker = st.text_input("Masukkan kode saham (misal: BRMS.JK)", "BRMS.JK")

# Ambil data harga
try:
    data = yf.download(ticker, period="3mo", interval="1d", auto_adjust=True)
    if data.empty:
        st.error("❌ Data tidak ditemukan.")
        st.stop()
except Exception as e:
    st.error(f"Gagal mengambil data: {e}")
    st.stop()

data.dropna(inplace=True)
data.reset_index(inplace=True)

# Pastikan data close adalah Series 1D
close = data['Close'].squeeze()
volume = data['Volume']
ma9 = close.rolling(window=9).mean()
rsi = ta.momentum.RSIIndicator(close=close).rsi()
macd_obj = ta.trend.MACD(close=close)
macd = macd_obj.macd()
macd_signal = macd_obj.macd_signal()

# Hitung support/resistance pivot
def calculate_pivot(df):
    high = df['High'].iloc[-1]
    low = df['Low'].iloc[-1]
    close = df['Close'].iloc[-1]
    pivot = (high + low + close) / 3
    r1 = (2 * pivot) - low
    s1 = (2 * pivot) - high
    return round(pivot, 2), round(r1, 2), round(s1, 2)

pivot, resistance, support = calculate_pivot(data)

# Ambil nilai indikator terbaru
last_close = float(close.iloc[-1])
last_ma9 = float(ma9.iloc[-1])
last_rsi = float(rsi.iloc[-1])
last_macd = float(macd.iloc[-1])
last_macd_signal = float(macd_signal.iloc[-1])
last_volume = int(volume.iloc[-1])
date = data['Date'].iloc[-1].strftime("%Y-%m-%d")

# Logika sinyal
sinyal = ""
harga_beli = "-"
target_profit = "-"
alasan = []

if last_rsi < 30:
    sinyal = "✅ BELI (RSI oversold)"
    harga_beli = f"< {round(last_close + 1, 2)}"
    target_profit = f"{round(last_close * 1.05, 2)} (+5%)"
    alasan.append("RSI < 30 → potensi rebound")
elif last_macd > last_macd_signal and last_close > last_ma9 and last_rsi < 70:
    sinyal = "✅ BELI"
    harga_beli = f"< {round(last_close + 0.5, 2)}"
    target_profit = f"{round(last_close * 1.05, 2)} (+5%)"
    alasan.append("MACD Golden Cross")
    alasan.append("Harga > MA9")
elif last_rsi > 70:
    sinyal = "❌ JUAL (RSI overbought)"
    alasan.append("RSI > 70 → rawan koreksi")
elif last_macd < last_macd_signal and last_close < last_ma9:
    sinyal = "❌ JUAL"
    alasan.append("MACD Dead Cross")
    alasan.append("Harga < MA9")
else:
    sinyal = "⏸️ HOLD / TUNGGU KONFIRMASI"
    alasan.append("Belum ada sinyal kuat berdasarkan indikator umum")

# OUTPUT HASIL
st.subheader(f"📅 Tanggal Data Terakhir: {date}")
st.subheader(f"📈 Sinyal: {sinyal}")

col1, col2 = st.columns(2)

col1.markdown(f"""
- **Harga Saat Ini**: `{round(last_close, 2)}`
- **Harga Beli Ideal**: `{harga_beli}`
- **Target Profit (TP)**: `{target_profit}`
""")

col2.markdown(f"""
### 🔧 Nilai Indikator:
- MA9: `{round(last_ma9, 2)}`
- RSI: `{round(last_rsi, 2)}`
- MACD: `{round(last_macd, 3)}`
- Signal Line: `{round(last_macd_signal, 3)}`
- Volume: `{last_volume:,}`
- Support: `{support}`
- Resistance: `{resistance}`
""")

st.markdown("### 📌 Alasan Teknis:")
for i in alasan:
    st.write(f"- {i}")