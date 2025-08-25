# teknikal2.py
import streamlit as st
import pandas as pd
import yfinance as yf
import ta

# ==============================
# Konfigurasi awal
# ==============================
st.set_page_config(page_title="LQ45 Stock Analyzer Pro", layout="wide")

st.title("📊 LQ45 Stock Analyzer Pro")
st.markdown("Analisis otomatis saham LQ45 (data Yahoo Finance ~15 menit delay)")

period = st.selectbox("Pilih periode data:", ["3mo", "6mo", "1y", "2y"], index=2)

# ==============================
# Daftar saham LQ45
# ==============================
lq45 = [
    "ADRO.JK","AKRA.JK","AMRT.JK","ANTM.JK","ARTO.JK","ASII.JK","BBCA.JK","BBNI.JK","BBRI.JK",
    "BBTN.JK","BMRI.JK","BRIS.JK","BUKA.JK","CPIN.JK","ELSA.JK","EMTK.JK","ESSA.JK","EXCL.JK",
    "GGRM.JK","GOTO.JK","HRUM.JK","ICBP.JK","INCO.JK","INDF.JK","INKP.JK","INTP.JK","ITMG.JK",
    "JPFA.JK","JRPT.JK","KLBF.JK","MDKA.JK","MEDC.JK","MIKA.JK","MTEL.JK","PGAS.JK","PTBA.JK",
    "PTPP.JK","SMGR.JK","TBIG.JK","TINS.JK","TKIM.JK","TLKM.JK","TOWR.JK","UNTR.JK","UNVR.JK"
]

# ==============================
# Fungsi Analisis Saham
# ==============================
def analyze_stock(ticker):
    try:
        df = yf.download(ticker, period=period, interval="1d", auto_adjust=True, progress=False)

        if df.empty:
            return None

        # Reset index untuk hilangkan multi-index
        df = df.reset_index()

        # Pastikan hanya ambil kolom yang penting
        cols = ["Date","Open","High","Low","Close","Volume"]
        df = df[cols].copy()

        # Convert semua ke numeric
        for col in ["Open","High","Low","Close","Volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.dropna()
        if df.empty:
            return None

        # indikator teknikal (pakai float series agar aman)
        df["MA9"] = df["Close"].rolling(9).mean()

        rsi_ind = ta.momentum.RSIIndicator(close=df["Close"].astype(float), window=14)
        df["RSI"] = rsi_ind.rsi()

        macd = ta.trend.MACD(close=df["Close"].astype(float))
        df["MACD"] = macd.macd()
        df["MACD_signal"] = macd.macd_signal()

        atr = ta.volatility.AverageTrueRange(
            high=df["High"].astype(float),
            low=df["Low"].astype(float),
            close=df["Close"].astype(float),
            window=14
        )
        df["ATR"] = atr.average_true_range()

        last = df.iloc[-1]
        close = last["Close"]
        ma9 = last["MA9"]
        rsi = last["RSI"]
        macd_val = last["MACD"]
        sig = last["MACD_signal"]
        atr_val = last["ATR"]

        entry = close
        tp = close + 1.5 * atr_val
        sl = close - 1.0 * atr_val

        if close > ma9 and macd_val > sig and rsi < 70:
            rekom = "BUY"
        elif close < ma9 and macd_val < sig and rsi > 50:
            rekom = "SELL"
        else:
            rekom = "HOLD"

        return {
            "Ticker": ticker,
            "Close": round(close,2),
            "MA9": round(ma9,2),
            "RSI": round(rsi,2),
            "MACD": round(macd_val,2),
            "Signal": round(sig,2),
            "ATR": round(atr_val,2),
            "Rekomendasi": rekom,
            "Entry": round(entry,2),
            "TP": round(tp,2),
            "SL": round(sl,2)
        }

    except Exception as e:
        st.error(f"⚠️ Error {ticker}: {e}")
        return None

# ==============================
# Jalankan analisis semua saham
# ==============================
st.write("⏳ Mengambil data, mohon tunggu...")

results = []
for t in lq45:
    res = analyze_stock(t)
    if res:
        results.append(res)

if results:
    df_results = pd.DataFrame(results)
    # urutkan BUY paling atas
    df_results = df_results.sort_values(
        by=["Rekomendasi","RSI"],
        ascending=[False, True]
    ).reset_index(drop=True)

    st.dataframe(df_results, use_container_width=True)

    # ringkasan
    buy_count = (df_results["Rekomendasi"]=="BUY").sum()
    sell_count = (df_results["Rekomendasi"]=="SELL").sum()
    hold_count = (df_results["Rekomendasi"]=="HOLD").sum()

    st.success(f"✅ BUY: {buy_count} | ❌ SELL: {sell_count} | ⏸ HOLD: {hold_count}")

else:
    st.warning("Gagal mengambil data LQ45. Coba ulangi lagi.")