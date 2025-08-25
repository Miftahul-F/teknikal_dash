# teknikal2.py
import streamlit as st
import pandas as pd
import yfinance as yf
import ta

st.set_page_config(page_title="LQ45 Stock Analyzer Pro", layout="wide")
st.title("📊 LQ45 Stock Analyzer Pro")

period = st.selectbox("Pilih periode data:", ["3mo", "6mo", "1y", "2y"], index=2)

lq45 = [
    "ADRO.JK","AKRA.JK","AMRT.JK","ANTM.JK","ARTO.JK","ASII.JK","BBCA.JK","BBNI.JK","BBRI.JK",
    "BBTN.JK","BMRI.JK","BRIS.JK","BUKA.JK","CPIN.JK","ELSA.JK","EMTK.JK","ESSA.JK","EXCL.JK",
    "GGRM.JK","GOTO.JK","HRUM.JK","ICBP.JK","INCO.JK","INDF.JK","INKP.JK","INTP.JK","ITMG.JK",
    "JPFA.JK","JRPT.JK","KLBF.JK","MDKA.JK","MEDC.JK","MIKA.JK","MTEL.JK","PGAS.JK","PTBA.JK",
    "PTPP.JK","SMGR.JK","TBIG.JK","TINS.JK","TKIM.JK","TLKM.JK","TOWR.JK","UNTR.JK","UNVR.JK"
]

def analyze_stock(ticker):
    try:
        df = yf.download(ticker, period=period, interval="1d", auto_adjust=True, progress=False)
        if df.empty:
            return None

        # pastikan dataframe flat
        df = df.reset_index()

        # ambil hanya kolom inti
        needed_cols = [c for c in ["Date","Open","High","Low","Close","Volume"] if c in df.columns]
        df = df[needed_cols].copy()

        # pastikan semua numeric
        for col in ["Open","High","Low","Close","Volume"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.dropna()
        if df.empty or "Close" not in df.columns:
            return None

        # pastikan 1D series float
        close = df["Close"].astype(float).squeeze()
        high = df["High"].astype(float).squeeze()
        low = df["Low"].astype(float).squeeze()

        df["MA9"] = close.rolling(9).mean()
        df["RSI"] = ta.momentum.RSIIndicator(close=close, window=14).rsi()
        macd_ind = ta.trend.MACD(close=close)
        df["MACD"] = macd_ind.macd()
        df["MACD_signal"] = macd_ind.macd_signal()
        atr_ind = ta.volatility.AverageTrueRange(high=high, low=low, close=close, window=14)
        df["ATR"] = atr_ind.average_true_range()

        last = df.iloc[-1]
        entry = last["Close"]
        tp = entry + 1.5 * last["ATR"]
        sl = entry - 1.0 * last["ATR"]

        if last["Close"] > last["MA9"] and last["MACD"] > last["MACD_signal"] and last["RSI"] < 70:
            rekom = "BUY"
        elif last["Close"] < last["MA9"] and last["MACD"] < last["MACD_signal"] and last["RSI"] > 50:
            rekom = "SELL"
        else:
            rekom = "HOLD"

        return {
            "Ticker": ticker,
            "Close": round(entry,2),
            "MA9": round(last["MA9"],2),
            "RSI": round(last["RSI"],2),
            "MACD": round(last["MACD"],2),
            "Signal": round(last["MACD_signal"],2),
            "ATR": round(last["ATR"],2),
            "Rekomendasi": rekom,
            "Entry": round(entry,2),
            "TP": round(tp,2),
            "SL": round(sl,2)
        }

    except Exception as e:
        st.warning(f"⚠️ Error {ticker}: {e}")
        return None

st.write("⏳ Mengambil data saham LQ45...")

results = [analyze_stock(t) for t in lq45]
results = [r for r in results if r]

if results:
    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values(by=["Rekomendasi","RSI"], ascending=[False,True]).reset_index(drop=True)
    st.dataframe(df_results, use_container_width=True)

    buy = (df_results["Rekomendasi"]=="BUY").sum()
    sell = (df_results["Rekomendasi"]=="SELL").sum()
    hold = (df_results["Rekomendasi"]=="HOLD").sum()
    st.success(f"✅ BUY: {buy} | ❌ SELL: {sell} | ⏸ HOLD: {hold}")
else:
    st.error("Tidak ada data berhasil diambil.")