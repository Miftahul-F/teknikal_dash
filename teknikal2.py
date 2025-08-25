import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import pytz
from datetime import datetime

# ----------------------------
# Utils & Indicators (manual)
# ----------------------------
def ensure_jk(t):
    t = t.strip().upper()
    if not t:
        return ""
    # Jika sudah ada suffix lain (AAPL, MSFT), biarkan
    if "." in t:
        return t
    # Default ke .JK (IDX)
    return f"{t}.JK"

def rsi_series(close: pd.Series, window: int = 14) -> pd.Series:
    close = pd.to_numeric(close, errors="coerce")
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    # Wilder's smoothing
    avg_gain = gain.ewm(alpha=1/window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/window, min_periods=window, adjust=False).mean()
    rs = avg_gain / (avg_loss.replace(0, np.nan))
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)

def ema(close: pd.Series, window: int = 20) -> pd.Series:
    close = pd.to_numeric(close, errors="coerce")
    return close.ewm(span=window, adjust=False).mean()

def macd_series(close: pd.Series, fast=12, slow=26, signal=9):
    close = pd.to_numeric(close, errors="coerce")
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    hist = macd - macd_signal
    return macd, macd_signal, hist

def safe_download(ticker: str, period="6mo", interval="1d"):
    try:
        df = yf.download(ticker, period=period, interval=interval, auto_adjust=True, progress=False)
        if df is None or df.empty or "Close" not in df.columns:
            return None
        df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
        for c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df.dropna(subset=["Close"], inplace=True)
        return df
    except Exception:
        return None

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return None
    out = df.copy()
    out["EMA20"] = ema(out["Close"], 20)
    out["RSI14"] = rsi_series(out["Close"], 14)
    macd, sig, hist = macd_series(out["Close"], 12, 26, 9)
    out["MACD"] = macd
    out["MACD_sig"] = sig
    out["MACD_hist"] = hist
    return out.dropna(subset=["Close"])

def generate_signal_row(last):
    """
    Rule sederhana:
    - BUY: RSI < 60, Close > EMA20, MACD > Signal
    - SELL: RSI > 70 dan Close < EMA20 dan MACD < Signal
    - lainnya: HOLD
    Sekaligus kasih EstBuy = Close * 1.01 (estimasi gap kecil open besok).
    """
    try:
        c = float(last["Close"])
        rsi = float(last["RSI14"])
        ema20 = float(last["EMA20"])
        macd = float(last["MACD"])
        sig = float(last["MACD_sig"])
    except Exception:
        return "⚠️ Error", None, 0.0

    if (rsi < 60) and (c > ema20) and (macd > sig):
        est = round(c * 1.01, 2)
        conf = 0.9
        # tambah confidence sedikit jika jarak MACD ke signal besar dan RSI antara 45-55 (momentum baru jalan)
        conf += min(max((macd - sig) / max(abs(sig), 1e-6) * 0.1, 0), 0.05)
        if 45 <= rsi <= 55:
            conf += 0.03
        conf = min(conf, 0.98)
        return "✅ BUY", est, conf
    elif (rsi > 70) and (c < ema20) and (macd < sig):
        return "❌ SELL", None, 0.85
    else:
        # HOLD — hitung confidence sedang
        conf = 0.5
        if c > ema20: conf += 0.1
        if macd > sig: conf += 0.1
        if 40 <= rsi <= 60: conf += 0.05
        conf = min(conf, 0.8)
        return "⏳ HOLD", None, conf

# ----------------------------
# UI
# ----------------------------
st.set_page_config(page_title="📋 IDX Watchlist Analyzer", layout="wide")
st.title("📋 IDX Watchlist Analyzer — Multi Ticker (tanpa .JK)")

wib = pytz.timezone("Asia/Jakarta")
now = datetime.now(wib)
st.caption(f"⏰ {now.strftime('%Y-%m-%d %H:%M:%S')} WIB — data harian Yahoo Finance (delay ~15m)")

with st.sidebar:
    st.subheader("Input Watchlist")
    st.caption("Pisahkan dengan spasi, koma, atau baris baru. Contoh: `BBCA BBRI TLKM ADRO`")
    raw = st.text_area("Kode saham", value="BBCA BBRI TLKM ADRO")
    period = st.selectbox("Period data", ["3mo", "6mo", "1y", "2y"], index=1)
    st.caption("Estimasi harga beli besok = Close hari ini × 1.01 (bisa diganti cepat di kode).")

# Parse tickers
tokens = [t.strip().upper() for t in (raw.replace(",", " ").split()) if t.strip()]
tickers = [ensure_jk(t) for t in tokens]
tickers = [t for t in tickers if t]  # remove empty

if not tickers:
    st.info("Masukkan minimal satu kode saham.")
    st.stop()

rows = []
error_list = []

progress = st.progress(0)
for i, tk in enumerate(tickers, start=1):
    progress.progress(i / len(tickers))
    df = safe_download(tk, period=period, interval="1d")
    if df is None or df.empty:
        error_list.append(f"{tk}: data kosong")
        rows.append({"Ticker": tk, "Signal": "⚠️ No Data", "Close": np.nan, "RSI14": np.nan,
                     "EMA20": np.nan, "MACD": np.nan, "EstBuyTomorrow": None, "Confidence": 0.0})
        continue

    dfi = add_indicators(df)
    if dfi is None or dfi.empty:
        error_list.append(f"{tk}: indikator gagal")
        rows.append({"Ticker": tk, "Signal": "⚠️ Error", "Close": float(df["Close"].iloc[-1]),
                     "RSI14": np.nan, "EMA20": np.nan, "MACD": np.nan,
                     "EstBuyTomorrow": None, "Confidence": 0.0})
        continue

    last = dfi.iloc[-1]
    sig, est, conf = generate_signal_row(last)
    rows.append({
        "Ticker": tk,
        "Signal": sig,
        "Close": float(last["Close"]),
        "RSI14": float(last["RSI14"]),
        "EMA20": float(last["EMA20"]),
        "MACD": float(last["MACD"]),
        "EstBuyTomorrow": est,
        "Confidence": round(float(conf), 2)
    })

progress.progress(1.0)

# Tabel ringkas
df_res = pd.DataFrame(rows)
order = {"✅ BUY": 0, "⏳ HOLD": 1, "❌ SELL": 2, "⚠️ No Data": 3, "⚠️ Error": 4}
df_res["_ord"] = df_res["Signal"].map(order).fillna(9)
df_res = df_res.sort_values(by=["_ord", "Confidence", "Ticker"], ascending=[True, False, True]).drop(columns="_ord")

st.subheader("📜 Hasil Sinyal Watchlist")
st.dataframe(
    df_res.astype({
        "Close": "float64",
        "RSI14": "float64",
        "EMA20": "float64",
        "MACD": "float64",
        "Confidence": "float64"
    }),
    use_container_width=True
)

# Rekomendasi BUY (jika ada)
df_buy = df_res[df_res["Signal"] == "✅ BUY"].copy()
if not df_buy.empty:
    st.success("🎯 Rekomendasi BUY untuk besok pagi (estimasi):")
    st.dataframe(df_buy[["Ticker", "Close", "EstBuyTomorrow", "Confidence"]], use_container_width=True)
else:
    st.info("Belum ada sinyal BUY kuat hari ini.")

# Detail tiap ticker
st.markdown("---")
st.subheader("🔍 Detail per Ticker")
for _, row in df_res.iterrows():
    with st.expander(f"{row['Ticker']} • {row['Signal']} • Close {row['Close']:.2f} • Conf {row['Confidence']:.2f}"):
        tk = row["Ticker"]
        df = safe_download(tk, period=period, interval="1d")
        dfi = add_indicators(df) if df is not None else None
        if dfi is None or dfi.empty:
            st.warning("Data/indikator tidak tersedia.")
            continue

        last = dfi.iloc[-1]
        st.write(
            f"- RSI(14): **{last['RSI14']:.2f}**  •  EMA20: **{last['EMA20']:.2f}**  •  MACD/Signal: **{last['MACD']:.4f}/{last['MACD_sig']:.4f}**"
        )
        if not pd.isna(row["EstBuyTomorrow"]):
            st.write(f"- 💡 Estimasi beli besok: **{row['EstBuyTomorrow']:.2f}**")

        # mini chart
        chart_df = dfi[["Close", "EMA20"]].tail(120)
        st.line_chart(chart_df, height=220)