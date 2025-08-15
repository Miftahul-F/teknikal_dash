# teknikal_syariah_pro.py
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import talib

# ===================== CONFIG =====================
st.set_page_config(page_title="Syariah Stock Screener Pro", layout="wide")
st.title("🕌 Syariah Stock Screener Pro - Multi Timeframe + R/R")

# ===================== LOAD DAFTAR SAHAM SYARIAH =====================
@st.cache_data
def load_syariah_list():
    # Daftar ISSI terbaru (bisa update manual / load dari URL resmi IDX)
    syariah_list = pd.read_csv("https://raw.githubusercontent.com/tegarimansyah/data-saham/main/issi.csv")
    return set(syariah_list['Kode'].str.upper())

syariah_set = load_syariah_list()

# ===================== INPUT =====================
tickers_input = st.text_area(
    "Masukkan kode saham (pisahkan dengan koma) atau biarkan kosong untuk scan semua ISSI",
    ""
)

timeframe = st.selectbox(
    "Pilih Timeframe",
    ["1d", "1wk"]
)

period_map = {"1d": "6mo", "1wk": "2y"}
period = period_map[timeframe]

# ===================== FUNCTIONS =====================
def get_stock_data(ticker, period, interval):
    try:
        data = yf.download(f"{ticker}.JK", period=period, interval=interval, progress=False)
        if data.empty:
            return None
        data.dropna(inplace=True)
        return data
    except Exception:
        return None

def analyze_stock(ticker, df):
    try:
        df['RSI'] = talib.RSI(df['Close'], timeperiod=14)
        df['EMA20'] = talib.EMA(df['Close'], timeperiod=20)
        df['EMA50'] = talib.EMA(df['Close'], timeperiod=50)

        macd, macdsignal, _ = talib.MACD(df['Close'], fastperiod=12, slowperiod=26, signalperiod=9)
        df['MACD'] = macd
        df['MACDSignal'] = macdsignal

        df['VolBreak'] = df['Volume'] > df['Volume'].rolling(20).mean() * 1.5
        df['SwingBreak'] = df['Close'] > df['High'].rolling(20).max().shift(1)

        swing_high = df['High'].rolling(20).max().iloc[-1]
        swing_low = df['Low'].rolling(20).min().iloc[-1]
        close_price = df['Close'].iloc[-1]
        if swing_low and swing_high and close_price:
            reward = swing_high - close_price
            risk = close_price - swing_low
            rr_ratio = reward / risk if risk != 0 else np.nan
        else:
            rr_ratio = np.nan

        return {
            "Ticker": ticker,
            "Close": close_price,
            "RSI": df['RSI'].iloc[-1],
            "MACD": macd.iloc[-1],
            "MACDSignal": macdsignal.iloc[-1],
            "VolBreak": bool(df['VolBreak'].iloc[-1]),
            "SwingBreak": bool(df['SwingBreak'].iloc[-1]),
            "RR": rr_ratio
        }
    except Exception:
        return None

# ===================== PROCESS =====================
if tickers_input.strip():
    tickers = [t.strip().upper() for t in tickers_input.split(",")]
else:
    tickers = list(syariah_set)  # Scan semua saham syariah

results = []
for ticker in tickers:
    if ticker not in syariah_set:
        continue  # Skip kalau bukan saham syariah
    data = get_stock_data(ticker, period, timeframe)
    if data is not None:
        close_last = data['Close'].iloc[-1]
        if close_last < 1000:  # Filter harga < 1000
            analysis = analyze_stock(ticker, data)
            if analysis:
                results.append(analysis)

df_result = pd.DataFrame(results)

# ===================== DISPLAY =====================
if not df_result.empty:
    st.subheader("📌 Hasil Screening (Syariah & Harga < 1000)")
    for idx, r in df_result.iterrows():
        # Format R/R aman + emoji
        if pd.notna(r['RR']):
            rr_display = f"{r['RR']:.2f}"
            if r['RR'] >= 2:
                rr_display = f"🟢 {rr_display}"
            elif r['RR'] >= 1:
                rr_display = f"🟡 {rr_display}"
            else:
                rr_display = f"🔴 {rr_display}"
        else:
            rr_display = "-"

        vol_break = "YES" if r.get('VolBreak', False) else "NO"
        swing_break = "YES" if r.get('SwingBreak', False) else "NO"

        st.markdown(
            f"- **{r['Ticker']}**  •  Close: **{r['Close']:.2f}**  •  RSI: **{r['RSI']:.2f}**  •  VolBreak: **{vol_break}**  •  SwingBreak: **{swing_break}**  •  R/R: **{rr_display}**"
        )

    st.dataframe(df_result)
else:
    st.warning("Tidak ada saham syariah harga < 1000 yang memenuhi kriteria.")

# ===================== REQUIREMENTS =====================
st.markdown("""
**Requirements:**