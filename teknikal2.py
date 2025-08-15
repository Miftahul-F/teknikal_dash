# streamlit_multi_tf_pro.py
# Pro: 1 ticker, 4 timeframe dalam tab (1H, 4H, Daily, Weekly)
# Fitur:
# - Input ticker (otomatis tambah .JK untuk kode BEI 3-4 huruf)
# - Indikator: MA9, RSI(14), MACD, ATR(14), VolMA20
# - Auto pivot R1/R2 S1/S2 + Swing High/Low
# - Entry (BOW), TP, SL otomatis + R/R
# - Rekomendasi BUY/HOLD/SELL per timeframe
# - Chart interaktif (Candlestick + MA9 + level, Volume, RSI, MACD)
# - Anti-crash untuk data kosong / format anomali yfinance

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import ta
from plotly.subplots import make_subplots
import plotly.graph_objects as go

st.set_page_config(page_title="Multi-Timeframe Stock Analyzer • Pro", layout="wide")
st.title("📊 Multi-Timeframe Stock Analyzer — Pro (1 Ticker, Multi-Tab)")

# -------------------------
# Utils & Helpers
# -------------------------
def normalize_ticker(raw: str) -> str:
    """Autofix kode BEI: jika 3-4 huruf alfabet & tanpa .JK → tambah .JK"""
    tk = (raw or "").upper().strip()
    if not tk:
        return tk
    if tk.endswith(".JK"):
        return tk
    if tk.isalpha() and 2 < len(tk) <= 4:
        return tk + ".JK"
    return tk

@st.cache_data(show_spinner=False, ttl=300)
def download_ohlcv(ticker: str, period: str, interval: str):
    """Unduh OHLCV dari yfinance. Return DataFrame index datetime, kolom Open,High,Low,Close,Volume."""
    try:
        df = yf.download(ticker, period=period, interval=interval, auto_adjust=True, progress=False)
    except Exception as e:
        st.error(f"Gagal download data: {e}")
        return None
    if df is None or df.empty:
        return None

    # Handle MultiIndex kolom (kasus multi-ticker/format tertentu)
    if isinstance(df.columns, pd.MultiIndex):
        try:
            # Coba ambil level terakhir sesuai ticker
            df = df.xs(ticker, axis=1, level=-1)
        except Exception:
            # Ambil level pertama saja
            first_level = df.columns.levels[0][0]
            df = df[first_level]

    keep = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    if not keep:
        return None
    df = df[keep].copy()

    # Pastikan numeric dan bersih
    for c in df.columns:
        s = df[c]
        # jika Series multi-dim/salah tipe, paksa jadi 1D
        if hasattr(s, "squeeze"):
            s = s.squeeze()
        df[c] = pd.to_numeric(s, errors="coerce").astype(float)
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    df = df.dropna(subset=["Close"])
    if df.empty:
        return None
    return df

def resample_to_4h(df_60m: pd.DataFrame) -> pd.DataFrame:
    """Resample dari 60m ke 4H (ohlc + volume)."""
    d = df_60m.copy()
    d.index = pd.to_datetime(d.index)
    d = d.sort_index()
    d4 = d.resample("4H").agg({"Open":"first","High":"max","Low":"min","Close":"last","Volume":"sum"})
    d4 = d4.dropna(subset=["Close"])
    return d4

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Hitung MA9, RSI(14), MACD, ATR(14), VolMA20. Aman untuk data minim."""
    d = df.copy()
    close = pd.to_numeric(d["Close"], errors="coerce")
    high  = pd.to_numeric(d["High"],  errors="coerce") if "High" in d else close
    low   = pd.to_numeric(d["Low"],   errors="coerce") if "Low" in d else close
    vol   = pd.to_numeric(d["Volume"],errors="coerce") if "Volume" in d else None

    d["MA9"] = close.rolling(9, min_periods=3).mean()

    try:
        d["RSI14"] = ta.momentum.RSIIndicator(close=close, window=14).rsi()
    except Exception:
        d["RSI14"] = np.nan

    try:
        macd = ta.trend.MACD(close=close)
        d["MACD"] = macd.macd()
        d["MACD_signal"] = macd.macd_signal()
    except Exception:
        d["MACD"] = np.nan
        d["MACD_signal"] = np.nan

    try:
        atr = ta.volatility.AverageTrueRange(high=high, low=low, close=close, window=14)
        d["ATR14"] = atr.average_true_range()
    except Exception:
        d["ATR14"] = np.nan

    if vol is not None:
        d["VolMA20"] = vol.rolling(20, min_periods=5).mean()
    else:
        d["VolMA20"] = np.nan

    return d

def compute_pivot_sr_lastbar(df: pd.DataFrame):
    """Pivot & S/R klasik bar terakhir."""
    h = float(df["High"].iloc[-1]); l = float(df["Low"].iloc[-1]); c = float(df["Close"].iloc[-1])
    pivot = (h + l + c) / 3.0
    r1 = (2 * pivot) - l
    s1 = (2 * pivot) - h
    r2 = pivot + (h - l)
    s2 = pivot - (h - l)
    return pivot, r1, r2, s1, s2

def compute_entry_tp_sl(df_i: pd.DataFrame, swing_window: int = 10):
    """Entry BOW (support terdekat <= close), TP di resistance terdekat >= close, SL di support terdekat."""
    c = float(df_i["Close"].iloc[-1])
    pivot, r1, r2, s1, s2 = compute_pivot_sr_lastbar(df_i)
    swing_res = float(df_i["High"].tail(swing_window).max())
    swing_sup = float(df_i["Low"].tail(swing_window).min())
    atr = float(df_i["ATR14"].iloc[-1]) if "ATR14" in df_i and not pd.isna(df_i["ATR14"].iloc[-1]) else max(c * 0.02, 1e-6)

    # Entry (<= c)
    candidates = [lvl for lvl in [df_i["MA9"].iloc[-1], pivot, s1, swing_sup] if (not pd.isna(lvl)) and (lvl <= c)]
    entry = max(candidates) if candidates else c * 0.99

    # TP (>= c)
    resistances = [lvl for lvl in [r1, r2, swing_res] if lvl >= c]
    tp = min(resistances) if resistances else c + 1.5 * atr

    # SL (<= c)
    supports = [lvl for lvl in [s1, s2, swing_sup] if lvl <= c]
    sl = max(supports) if supports else c - 1.0 * atr

    # Pastikan R/R ≥ 1 kalau memungkinkan
    risk = max(c - sl, 1e-6)
    reward = max(tp - c, 1e-6)
    if reward / risk < 1.0:
        tp = c + max(risk * 1.2, 1.2 * atr)
        reward = tp - c

    return float(entry), float(tp), float(sl), float(reward / risk), (pivot, r1, r2, s1, s2, swing_res, swing_sup)

def rekomendasi_row(row: pd.Series) -> str:
    """Sinyal sederhana: harga vs MA9, RSI zona, MACD cross."""
    try:
        c = float(row["Close"]); ma9 = float(row["MA9"])
        rsi = float(row["RSI14"]); macd = float(row["MACD"]); sig = float(row["MACD_signal"])
    except Exception:
        return "WAIT"
    if c > ma9 and rsi < 70 and macd > sig:
        return "BUY"
    if c < ma9 and rsi > 50 and macd < sig:
        return "SELL"
    return "HOLD"

def render_tab(tab_label: str, df_src: pd.DataFrame):
    """Render satu tab timeframe lengkap."""
    with st.tab(tab_label):
        if df_src is None or df_src.empty or df_src.shape[0] < 20:
            st.warning("Data timeframe ini belum cukup (butuh ≥ ~20 bar). Coba period lebih panjang.")
            return

        df_i = compute_indicators(df_src)
        last = df_i.iloc[-1]

        # Levels & trading plan
        entry, tp, sl, rr, levels = compute_entry_tp_sl(df_i, swing_window=10)
        pivot, r1, r2, s1, s2, swing_res, swing_sup = levels
        rekom = rekomendasi_row(last)

        # Headline metrics
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Close", f"{float(last['Close']):,.2f}")
        c2.metric("Rekomendasi", rekom)
        c3.metric("ATR(14)", f"{float(last['ATR14']) if 'ATR14' in last and not pd.isna(last['ATR14']) else 0.0:,.2f}")
        c4.metric("R/R (approx)", f"{rr:,.2f}")

        st.markdown(
            f"**Entry (BOW)**: `{entry:.2f}`  •  **TP**: `{tp:.2f}`  •  **SL**: `{sl:.2f}`  \n"
            f"**Pivot | R1 | R2 | S1 | S2**: `{pivot:.2f}` | `{r1:.2f}` | `{r2:.2f}` | `{s1:.2f}` | `{s2:.2f}`  \n"
            f"**Swing High/Low (10)**: `{swing_res:.2f}` / `{swing_sup:.2f}`"
        )

        # Indikator ringkas
        last_ma9 = float(last["MA9"]) if not pd.isna(last.get("MA9", np.nan)) else np.nan
        last_rsi = float(last["RSI14"]) if not pd.isna(last.get("RSI14", np.nan)) else np.nan
        last_macd = float(last["MACD"]) if not pd.isna(last.get("MACD", np.nan)) else np.nan
        last_sig = float(last["MACD_signal"]) if not pd.isna(last.get("MACD_signal", np.nan)) else np.nan
        last_vol = int(last.get("Volume", 0)) if "Volume" in df_i.columns else 0
        last_volma = float(last.get("VolMA20", np.nan)) if not pd.isna(last.get("VolMA20", np.nan)) else np.nan

        st.markdown("### 📌 Alasan singkat")
        bullets = []
        if not np.isnan(last_ma9):
            bullets.append(f"MA9: {last_ma9:.2f} → harga **{'di atas' if last['Close']>last_ma9 else 'di bawah'}** MA9")
        if not np.isnan(last_rsi):
            if last_rsi < 30: bullets.append(f"RSI(14): {last_rsi:.2f} → **Oversold**")
            elif last_rsi > 70: bullets.append(f"RSI(14): {last_rsi:.2f} → **Overbought**")
            else: bullets.append(f"RSI(14): {last_rsi:.2f} → **Netral**")
        if not np.isnan(last_macd) and not np.isnan(last_sig):
            bullets.append(f"MACD: {last_macd:.3f} vs Signal: {last_sig:.3f} → **{'Bullish' if last_macd>last_sig else 'Bearish'}**")
        if not np.isnan(last_volma):
            note = "tinggi" if last_vol > 1.2*last_volma else ("rendah" if last_vol < 0.8*last_volma else "normal")
            bullets.append(f"Volume: {last_vol:,} (MA20: {int(last_volma):,}) → **{note}**")
        for b in bullets:
            st.write(f"- {b}")

        # Chart lengkap (4 panel)
        st.markdown("### 📈 Chart")
        fig = make_subplots(
            rows=4, cols=1, shared_xaxes=True,
            row_heights=[0.50, 0.12, 0.18, 0.20],
            vertical_spacing=0.02,
            subplot_titles=("Harga & MA9 + Levels", "Volume", "RSI(14)", "MACD")
        )

        # Panel 1: Candles + MA9 + levels
        fig.add_trace(go.Candlestick(
            x=df_i.index, open=df_i["Open"], high=df_i["High"], low=df_i["Low"], close=df_i["Close"], name="Price"
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=df_i.index, y=df_i["MA9"], mode="lines", name="MA9", line=dict(color="orange", width=1.2)
        ), row=1, col=1)

        for y, txt, color in [
            (entry,"Entry","#3498db"), (tp,"TP","#2ecc71"), (sl,"SL","#e74c3c"),
            (pivot,"Pivot","#7f8c8d"), (r1,"R1","#e67e22"), (r2,"R2","#d35400"),
            (s1,"S1","#27ae60"), (s2,"S2","#16a085"),
            (swing_res,"SwingHigh","#c0392b"), (swing_sup,"SwingLow","#2980b9")
        ]:
            fig.add_hline(y=y, line_dash="dot", line_color=color, annotation_text=txt,
                          annotation_position="top right", row=1, col=1)

        # Panel 2: Volume
        if "Volume" in df_i.columns:
            fig.add_trace(go.Bar(x=df_i.index, y=df_i["Volume"], name="Volume", marker_color="lightblue"), row=2, col=1)
            if "VolMA20" in df_i.columns:
                fig.add_trace(go.Scatter(x=df_i.index, y=df_i["VolMA20"], mode="lines",
                                         name="VolMA20", line=dict(color="orange", width=1)), row=2, col=1)

        # Panel 3: RSI
        fig.add_trace(go.Scatter(x=df_i.index, y=df_i["RSI14"], mode="lines", name="RSI", line=dict(color="yellow", width=1.3)), row=3, col=1)
        fig.add_hline(y=70, line_dash="dot", line_color="red", row=3, col=1)
        fig.add_hline(y=30, line_dash="dot", line_color="green", row=3, col=1)

        # Panel 4: MACD
        fig.add_trace(go.Scatter(x=df_i.index, y=df_i["MACD"], mode="lines", name="MACD", line=dict(color="cyan", width=1.3)), row=4, col=1)
        fig.add_trace(go.Scatter(x=df_i.index, y=df_i["MACD_signal"], mode="lines", name="Signal", line=dict(color="magenta", width=1)), row=4, col=1)
        fig.add_hline(y=0, line_dash="dot", line_color="white", row=4, col=1)

        fig.update_layout(template="plotly_dark", xaxis_rangeslider_visible=False, height=900, showlegend=True)
        st.plotly_chart(fig, use_container_width=True)

# -------------------------
# Sidebar / Inputs
# -------------------------
with st.sidebar:
    st.markdown("## Input & Pengaturan")
    raw_ticker = st.text_input("Ticker (contoh: GOTO atau AAPL)", value="GOTO")
    ticker = normalize_ticker(raw_ticker)
    st.caption(f"Ticker digunakan: **{ticker or '-'}**")
    st.markdown("---")
    st.markdown("### Data Pembelian (opsional)")
    avg_buy = st.number_input("Avg Buy (Rp/lembar)", min_value=0.0, value=0.0, step=0.01)
    lots = st.number_input("Jumlah lot (1 lot = 100)", min_value=0, value=0, step=1)
    st.markdown("---")
    st.markdown("**Tips**\n- BEI: cukup ketik 3-4 huruf (otomatis `.JK`)\n- 4H diambil dari 60m yang di-resample")

# -------------------------
# Ambil Data untuk 4 Timeframe (sekali jalan)
# -------------------------
if not ticker:
    st.info("Masukkan ticker terlebih dahulu.")
    st.stop()

# Default period per timeframe (aman untuk yfinance)
period_map = {
    "1H": ("7d", "60m"),
    "4H": ("60d", "60m"),  # akan di-resample ke 4H
    "Daily": ("1y", "1d"),
    "Weekly": ("5y", "1wk")
}

d1h = download_ohlcv(ticker, *period_map["1H"])
d60m = download_ohlcv(ticker, *period_map["4H"])
d4h = resample_to_4h(d60m) if d60m is not None and not d60m.empty else None
dd  = download_ohlcv(ticker, *period_map["Daily"])
dw  = download_ohlcv(ticker, *period_map["Weekly"])

# -------------------------
# Header ringkas & P/L
# -------------------------
def latest_close(df):
    try:
        return float(df["Close"].iloc[-1])
    except Exception:
        return np.nan

last_prices = {
    "1H": latest_close(d1h),
    "4H": latest_close(d4h) if d4h is not None else np.nan,
    "Daily": latest_close(dd),
    "Weekly": latest_close(dw),
}

colA, colB, colC, colD = st.columns(4)
colA.metric("Last 1H", f"{last_prices['1H']:,.2f}" if not np.isnan(last_prices['1H']) else "-")
colB.metric("Last 4H", f"{last_prices['4H']:,.2f}" if not np.isnan(last_prices['4H']) else "-")
colC.metric("Last Daily", f"{last_prices['Daily']:,.2f}" if not np.isnan(last_prices['Daily']) else "-")
colD.metric("Last Weekly", f"{last_prices['Weekly']:,.2f}" if not np.isnan(last_prices['Weekly']) else "-")

if avg_buy > 0 and lots > 0 and not np.isnan(last_prices["Daily"]):
    shares = lots * 100
    modal = avg_buy * shares
    nilai_now = last_prices["Daily"] * shares
    pnl = nilai_now - modal
    pnl_pct = (pnl / modal * 100) if modal else 0.0
    st.success(f"💼 Posisi: {shares} lembar • Modal: Rp {modal:,.0f} • Nilai: Rp {nilai_now:,.0f} • P/L: Rp {pnl:,.0f} ({pnl_pct:.2f}%)")

st.markdown("---")

# -------------------------
# Tabs per timeframe
# -------------------------
tab1, tab2, tab3, tab4 = st.tabs(["🕐 1H", "🕓 4H", "📅 Daily", "📆 Weekly"])
render_tab("🕐 1H", d1h if d1h is not None else pd.DataFrame())
render_tab("🕓 4H", d4h if d4h is not None else pd.DataFrame())
render_tab("📅 Daily", dd if dd is not None else pd.DataFrame())
render_tab("📆 Weekly", dw if dw is not None else pd.DataFrame())

# -------------------------
# Footer
# -------------------------
st.markdown("---")
st.caption("Catatan: Rekomendasi berbasis indikator teknikal sederhana. Gunakan manajemen risiko dan konfirmasi multi-timeframe.")