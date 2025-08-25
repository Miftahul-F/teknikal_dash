# teknikal2.py
import streamlit as st
import yfinance as yf
import pandas as pd
import ta
import time

# =============================
# Fungsi indikator teknikal
# =============================
def add_indicators(df):
    try:
        df = df.copy()

        # pastikan kolom Close ada
        if "Close" not in df.columns:
            return None

        # pastikan tipe data numerik
        df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
        df = df.dropna(subset=["Close"])

        # kalau data terlalu pendek, skip
        if len(df) < 50:
            return None

        # tambahkan indikator
        df["RSI"] = ta.momentum.RSIIndicator(close=df["Close"], window=14).rsi()
        macd_ind = ta.trend.MACD(close=df["Close"])
        df["MACD"] = macd_ind.macd()
        df["Signal"] = macd_ind.macd_signal()
        df["EMA20"] = ta.trend.EMAIndicator(close=df["Close"], window=20).ema_indicator()
        df["EMA50"] = ta.trend.EMAIndicator(close=df["Close"], window=50).ema_indicator()

        return df.dropna()
    except Exception as e:
        print(f"⚠️ Error add_indicators: {e}")
        return None

# =============================
# Fungsi ambil data
# =============================
def get_data(ticker, period="6mo", interval="1d"):
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False)
        if df.empty:
            return None
        return df
    except Exception as e:
        print(f"⚠️ Error get_data {ticker}: {e}")
        return None

# =============================
# Fungsi analisis sinyal
# =============================
def analyze_signal(df):
    try:
        latest = df.iloc[-1]
        rsi = latest["RSI"]
        macd = latest["MACD"]
        signal = latest["Signal"]
        ema20 = latest["EMA20"]
        ema50 = latest["EMA50"]
        close = latest["Close"]

        if rsi < 30 and macd > signal and ema20 > ema50:
            return "✅ Strong Buy"
        elif rsi < 40 and macd > signal:
            return "👍 Buy"
        elif rsi > 70 and macd < signal:
            return "⚠️ Sell"
        else:
            return "⏳ Wait"
    except Exception as e:
        print(f"⚠️ Error analyze_signal: {e}")
        return "Error"

# =============================
# Streamlit App
# =============================
st.set_page_config(page_title="LQ45 Analyzer", layout="wide")
st.title("📊 LQ45 Multi-Stock Analyzer")

tickers = [
    "ADRO.JK","AKRA.JK","AMRT.JK","ANTM.JK","ARTO.JK","ASII.JK","BBCA.JK","BBNI.JK",
    "BBRI.JK","BBTN.JK","BMRI.JK","BRIS.JK","BUKA.JK","CPIN.JK","ELSA.JK","EXCL.JK",
    "GOTO.JK","HRUM.JK","ICBP.JK","INCO.JK","INDF.JK","INKP.JK","INTP.JK","ITMG.JK",
    "JPFA.JK","JRPT.JK","KLBF.JK","MDKA.JK","MEDC.JK","MIKA.JK","MNCN.JK","PGAS.JK",
    "PTBA.JK","PTPP.JK","SMGR.JK","SMRA.JK","TBIG.JK","TINS.JK","TKIM.JK","TLKM.JK",
    "TOWR.JK","UNTR.JK","UNVR.JK","WIKA.JK","WSKT.JK","WTKP.JK"
]

results = []
progress = st.progress(0)
status_text = st.empty()

for i, ticker in enumerate(tickers):
    status_text.text(f"⏳ Processing {ticker} ({i+1}/{len(tickers)}) ...")
    try:
        df = get_data(ticker)
        if df is None:
            results.append({"Ticker": ticker, "Signal": "⚠️ No Data"})
        else:
            df = add_indicators(df)
            if df is None or df.empty:
                results.append({"Ticker": ticker, "Signal": "⚠️ Error Indicators"})
            else:
                signal = analyze_signal(df)
                results.append({"Ticker": ticker, "Signal": signal})
    except Exception as e:
        results.append({"Ticker": ticker, "Signal": f"⚠️ Error {e}"})
    
    progress.progress((i+1)/len(tickers))
    time.sleep(0.1)

status_text.text("✅ Done")

df_results = pd.DataFrame(results)
st.dataframe(df_results)

# Highlight rekomendasi beli
st.subheader("📌 Rekomendasi Beli")
st.dataframe(df_results[df_results["Signal"].str.contains("Buy")])