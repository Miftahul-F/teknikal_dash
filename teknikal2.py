# streamlit_multi_tf_pro_plus.py
# Multi-Timeframe Stock Analyzer Pro+
# - Auto .JK detection
# - Cached & retry download
# - Mode Hemat Bandwidth
# - Tema chart (dark/light)
# - Indikator manual (EMA/RSI/MACD)
# - Pivot/Swing & Entry/TP/SL
# - Rekomendasi per timeframe + Confidence Score (0–100)
# - Risk Manager & P/L

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import timedelta

# -----------------------------
# UI Config
# -----------------------------
st.set_page_config(page_title="Stock Analyzer Pro+", layout="wide")
st.title("📊 Multi-Timeframe Stock Analyzer Pro+")

# -----------------------------
# Helpers
# -----------------------------
def normalize_ticker(raw: str) -> str:
    """Auto-append .JK untuk kode BEI yang tidak punya suffix."""
    t = (raw or "").upper().strip()
    if not t:
        return t
    # jika sudah ada '.', jangan tambahkan .JK (AAPL, TLKM.JK, dll)
    if "." in t:
        return t
    # asumsi kode BEI umumnya 3–5 huruf
    if 2 <= len(t) <= 5 and t.isalnum():
        return t + ".JK"
    return t

# -----------------------------
# Indikator manual
# -----------------------------
def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = (delta.where(delta > 0, 0.0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(window=period).mean()
    rs = gain / loss.replace(0, np.nan)
    r = 100 - (100 / (1 + rs))
    return r.clip(0, 100)

def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    macd_line = ema(series, fast) - ema(series, slow)
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    out["EMA20"] = ema(out["Close"], 20)
    out["EMA50"] = ema(out["Close"], 50)
    out["RSI14"] = rsi(out["Close"], 14)
    out["MACD"], out["MACDsig"], out["MACDhist"] = macd(out["Close"])
    if "Volume" in out:
        out["VolMA20"] = out["Volume"].rolling(20).mean()
    return out.dropna()

def trend_label(last: pd.Series) -> str:
    return "🟢 Bullish" if last["EMA20"] > last["EMA50"] else "🔴 Bearish"

def rec_signal(last: pd.Series) -> str:
    buy = (last["RSI14"] < 30) or ((last["EMA20"] > last["EMA50"]) and (last["MACD"] > last["MACDsig"]))
    sell = (last["RSI14"] > 70) or ((last["EMA20"] < last["EMA50"]) and (last["MACD"] < last["MACDsig"]))
    if buy:
        return "✅ BUY"
    if sell:
        return "❌ SELL"
    return "⚠️ HOLD"

def confidence_score(d_last: pd.Series, w_last: pd.Series) -> int:
    score = 0
    if w_last["EMA20"] > w_last["EMA50"]:
        score += 25
    if d_last["EMA20"] > d_last["EMA50"]:
        score += 25
    if d_last["MACD"] > d_last["MACDsig"]:
        score += 15
    if 45 <= d_last["RSI14"] <= 65:
        score += 15
    if d_last["Close"] > d_last["EMA20"]:
        score += 10
    vol = d_last.get("Volume", np.nan)
    volma = d_last.get("VolMA20", np.nan)
    if pd.notna(vol) and pd.notna(volma) and vol > 1.2 * volma:
        score += 10
    return int(score)

def pivot_levels_lastbar(df: pd.DataFrame):
    h = float(df["High"].iloc[-1]); l = float(df["Low"].iloc[-1]); c = float(df["Close"].iloc[-1])
    pivot = (h + l + c) / 3.0
    r1 = 2 * pivot - l
    s1 = 2 * pivot - h
    r2 = pivot + (h - l)
    s2 = pivot - (h - l)
    return pivot, r1, r2, s1, s2

def swing_levels(df: pd.DataFrame, window: int = 10):
    return float(df["High"].tail(window).max()), float(df["Low"].tail(window).min())

def entry_tp_sl(df: pd.DataFrame):
    c = float(df["Close"].iloc[-1])
    pivot, r1, r2, s1, s2 = pivot_levels_lastbar(df)
    sh, slw = swing_levels(df, 10)
    # Entry <= harga sekarang
    candidates = [lvl for lvl in [df["EMA20"].iloc[-1], pivot, s1, slw] if pd.notna(lvl) and lvl <= c]
    entry = max(candidates) if candidates else c * 0.99
    # TP >= harga sekarang
    res = [lvl for lvl in [r1, r2, sh] if lvl >= c]
    tp = min(res) if res else c * 1.02
    # SL <= harga sekarang
    sup = [lvl for lvl in [s1, s2, slw] if lvl <= c]
    stop = max(sup) if sup else c * 0.98
    # jaga RR >= 1.0
    risk = c - stop
    reward = tp - c
    if risk > 0 and reward / risk < 1.0:
        tp = c + max(risk * 1.2, c * 0.01)
    return float(entry), float(tp), float(stop), (pivot, r1, r2, s1, s2, sh, slw)

# -----------------------------
# Cached download dengan retry
# -----------------------------
@st.cache_data(show_spinner=False, ttl=60*15)  # cache 15 menit
def cached_download(ticker: str, period: str, interval: str, retries: int = 2) -> pd.DataFrame:
    last_err = None
    for _ in range(max(1, retries)):
        try:
            df = yf.download(ticker, period=period, interval=interval, auto_adjust=True, progress=False)
            if df is not None and not df.empty:
                cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
                df = df[cols].copy()
                for c in cols:
                    df[c] = pd.to_numeric(df[c], errors="coerce")
                df.dropna(subset=["Close"], inplace=True)
                df.index = pd.to_datetime(df.index)
                df.sort_index(inplace=True)
                return df
        except Exception as e:
            last_err = e
    # gagal
    return pd.DataFrame()

def load_tf_data(ticker: str, hemat: bool):
    """Load semua timeframe sekaligus sesuai mode hemat."""
    if hemat:
        d1h = cached_download(ticker, "30d", "1h")
        d4h = cached_download(ticker, "60d", "4h")
        dd  = cached_download(ticker, "6mo", "1d")
        dw  = cached_download(ticker, "3y", "1wk")
    else:
        d1h = cached_download(ticker, "60d", "1h")
        d4h = cached_download(ticker, "60d", "4h")
        dd  = cached_download(ticker, "1y",  "1d")
        dw  = cached_download(ticker, "5y",  "1wk")
    return d1h, d4h, dd, dw

# -----------------------------
# Sidebar Controls
# -----------------------------
with st.sidebar:
    st.header("Pengaturan Analisis")
    raw_ticker = st.text_input("Ticker (contoh: GOTO / BBCA / AAPL / TLKM.JK)", value="BBCA")
    ticker = normalize_ticker(raw_ticker)
    avg_buy = st.number_input("Avg Buy (Rp per lembar)", min_value=0.0, value=0.0, step=1.0)
    lots = st.number_input("Jumlah lot (1 lot = 100 lembar)", min_value=0, value=0, step=1)
    hemat = st.checkbox("Mode Hemat Bandwidth", value=True)
    theme = st.selectbox("Tema Chart", ["Dark", "Light"], index=0)
    show_levels = st.checkbox("Tampilkan Level (Pivot/Swing/Entry-TP-SL)", value=True)

    st.markdown("---")
    st.subheader("🛡️ Risk Manager")
    acc_value = st.number_input("Nilai Akun (Rp)", min_value=0.0, value=0.0, step=100000.0, help="Estimasi total ekuitas Anda")
    risk_pct = st.number_input("Risiko per Transaksi (%)", min_value=0.1, max_value=5.0, value=1.0, step=0.1)
    if st.button("Clear Cache"):
        st.cache_data.clear()
        st.success("Cache dibersihkan.")

# -----------------------------
# Load & Indicators
# -----------------------------
if not ticker:
    st.warning("Masukkan ticker terlebih dahulu.")
    st.stop()

d1h, d4h, dd, dw = load_tf_data(ticker, hemat)

d1h_i = add_indicators(d1h)
d4h_i = add_indicators(d4h)
dd_i  = add_indicators(dd)
dw_i  = add_indicators(dw)

# Confidence & overall summary dari Daily + Weekly
if not dd_i.empty and not dw_i.empty:
    last_d = dd_i.iloc[-1]
    last_w = dw_i.iloc[-1]
    conf = confidence_score(last_d, last_w)
    overall_trend = "🟢 Uptrend" if (last_d["EMA20"] > last_d["EMA50"]) and (last_w["EMA20"] > last_w["EMA50"]) else "🔴 Mixed/Down"
    overall_rec = rec_signal(last_d)
else:
    conf = 0
    overall_trend = "❔ Data kurang"
    overall_rec = "⚠️ HOLD"

# Entry/TP/SL dari Daily
if not dd_i.empty:
    ent, tp, sl, lv = entry_tp_sl(dd_i)
    pivot, r1, r2, s1, s2, sh, slw = lv
else:
    ent = tp = sl = np.nan
    pivot = r1 = r2 = s1 = s2 = sh = slw = np.nan

# Risk Manager (ukuran posisi)
position_info = ""
if acc_value > 0 and not np.isnan(sl) and not dd_i.empty:
    last_close = float(dd_i["Close"].iloc[-1])
    risk_amt = acc_value * (risk_pct / 100.0)
    per_share_risk = max(last_close - sl, 0.0)
    if per_share_risk > 0:
        shares = int(risk_amt // per_share_risk)
        lots_suggest = max(shares // 100, 0)
        position_info = f"🎯 Saran ukuran posisi: **{lots_suggest} lot** (~{lots_suggest*100} lembar) dengan risiko ± Rp{risk_amt:,.0f}"
    else:
        position_info = "Tidak dapat menghitung ukuran posisi (SL >= harga)."

# -----------------------------
# Summary Header
# -----------------------------
c1, c2, c3, c4 = st.columns(4)
c1.metric("Ticker", ticker)
c2.metric("Overall Trend", overall_trend)
c3.metric("Confidence", f"{conf} / 100")
c4.metric("Rekomendasi", overall_rec)

if not np.isnan(ent):
    st.markdown(
        f"**Entry (BoW)**: `{ent:.2f}` • **TP**: `{tp:.2f}` • **SL**: `{sl:.2f}`  "
        f"• **Pivot | R1 | R2**: `{pivot:.2f} | {r1:.2f} | {r2:.2f}`  "
        f"• **SwingH | SwingL (10)**: `{sh:.2f} | {slw:.2f}`"
    )
else:
    st.info("Tidak cukup data Daily untuk menghitung Entry/TP/SL.")

if position_info:
    st.success(position_info)

# -----------------------------
# Chart helper
# -----------------------------
def chart_price(dfx: pd.DataFrame, title: str):
    template = "plotly_dark" if theme == "Dark" else "plotly_white"
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=dfx.index, open=dfx["Open"], high=dfx["High"], low=dfx["Low"], close=dfx["Close"], name="Price"
    ))
    fig.add_trace(go.Scatter(x=dfx.index, y=dfx["EMA20"], line=dict(color="#f39c12", width=1.5), name="EMA20"))
    fig.add_trace(go.Scatter(x=dfx.index, y=dfx["EMA50"], line=dict(color="#3498db", width=1.5), name="EMA50"))

    if show_levels and not dd_i.empty and title.endswith("Daily"):
        # tambahkan garis level pada chart Daily saja supaya tidak ramai
        for y, name, color in [
            (ent, "Entry", "#1abc9c"),
            (tp, "TP", "#2ecc71"),
            (sl, "SL", "#e74c3c"),
            (pivot, "Pivot", "#95a5a6"),
            (r1, "R1", "#e67e22"),
            (r2, "R2", "#d35400"),
            (sh, "SwingH", "#c0392b"),
            (slw, "SwingL", "#2980b9"),
        ]:
            if pd.notna(y):
                fig.add_hline(y=y, line_dash="dot", line_color=color, annotation_text=name, annotation_position="top right")

    fig.update_layout(
        title=title, xaxis_rangeslider_visible=False, height=420,
        template=template, legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0)
    )
    return fig

# -----------------------------
# Tabs
# -----------------------------
tabs = st.tabs(["🕐 1H", "⏳ 4H", "📅 Daily", "📆 Weekly"])
tf_map = {
    0: ("1H", d1h_i),
    1: ("4H", d4h_i),
    2: ("Daily", dd_i),
    3: ("Weekly", dw_i),
}

for i in range(4):
    label, dfx = tf_map[i]
    with tabs[i]:
        st.subheader(f"{label} - {ticker}")
        if dfx.empty:
            st.warning("⚠️ Data tidak tersedia untuk timeframe ini.")
            continue

        # Price chart
        fig = chart_price(dfx, f"{ticker} • {label}")
        st.plotly_chart(fig, use_container_width=True)

        # Metrics
        last = dfx.iloc[-1]
        vol_note = "-"
        if "VolMA20" in dfx.columns and pd.notna(last.get("VolMA20")) and pd.notna(last.get("Volume")):
            if last["Volume"] > 1.2 * last["VolMA20"]:
                vol_note = "Tinggi"
            elif last["Volume"] < 0.8 * last["VolMA20"]:
                vol_note = "Rendah"
            else:
                vol_note = "Normal"

        colA, colB, colC, colD = st.columns(4)
        colA.metric("Harga", f"{last['Close']:.2f}")
        colB.metric("RSI(14)", f"{last['RSI14']:.2f}")
        colC.metric("Trend", trend_label(last))
        colD.metric("Volume", f"{int(last.get('Volume', 0)):,} ({vol_note})")

        st.markdown(
            f"**MACD / Signal**: `{last['MACD']:.2f} / {last['MACDsig']:.2f}` • "
            f"**EMA20 / EMA50**: `{last['EMA20']:.2f} / {last['EMA50']:.2f}` • "
            f"**Rekomendasi {label}**: {rec_signal(last)}"
        )

# -----------------------------
# P/L di Sidebar
# -----------------------------
with st.sidebar:
    st.markdown("---")
    st.subheader("💰 Hasil Investasi")
    if avg_buy > 0 and lots > 0:
        d_now = cached_download(ticker, "5d", "1d")
        if not d_now.empty:
            last_px = float(d_now["Close"].iloc[-1])
            qty = lots * 100
            modal = avg_buy * qty
            nilai = last_px * qty
            pl = nilai - modal
            pl_pct = (pl / modal * 100) if modal else 0.0
            st.write(f"- Harga Sekarang: **{last_px:,.2f}**")
            st.write(f"- Qty: **{qty:,}** lembar")
            st.write(f"- Modal: **{modal:,.0f}**")
            st.write(f"- Nilai Sekarang: **{nilai:,.0f}**")
            st.write(f"- P/L: **{pl:,.0f} ({pl_pct:.2f}%)**")
        else:
            st.warning("Tidak bisa menghitung P/L (data harga terkini kosong).")
    else:
        st.info("Isi Avg Buy & Lot untuk melihat P/L.")

st.markdown("---")
st.caption("Disclaimer: Analisis ini bersifat edukasi dan berbasis indikator teknikal sederhana. Lakukan riset mandiri dan gunakan manajemen risiko yang disiplin.")