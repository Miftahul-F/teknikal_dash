# multi_timeframe_pro.py
# Pro (stabil): 1 ticker, 4 tab timeframe (1H, 4H, Daily, Weekly) + kalkulasi portofolio
# Tanpa library 'ta' → indikator dihitung manual (RSI, MACD, ATR)

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objects as go

st.set_page_config(page_title="Multi-Timeframe Stock Analyzer Pro", layout="wide")
st.title("📊 Multi-Timeframe Stock Analyzer Pro (Stabil, Tanpa 'ta')")

# -------------------------
# Helpers umum
# -------------------------
def normalize_ticker(raw: str) -> str:
    """Jika input 3-4 huruf untuk BEI, otomatis tambahkan .JK"""
    tk = (raw or "").upper().strip()
    if not tk:
        return tk
    if tk.endswith(".JK"):
        return tk
    if tk.isalpha() and 2 < len(tk) <= 4:
        return tk + ".JK"
    return tk

def sanitize_ohlcv(df, ticker: str):
    """Pastikan kolom 1D: Open, High, Low, Close, Volume (float). Tangani MultiIndex dari yfinance."""
    if df is None or df.empty:
        return None
    d = df.copy()

    # Jika MultiIndex kolom
    if isinstance(d.columns, pd.MultiIndex):
        try:
            d = d.xs(ticker, axis=1, level=-1)
        except Exception:
            # fallback: ambil level pertama
            first = d.columns.levels[0][0]
            d = d[first]

    keep = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in d.columns]
    if not keep:
        return None
    d = d[keep].copy()

    # Pastikan numeric 1D
    for c in d.columns:
        s = d[c]
        if hasattr(s, "squeeze"):
            s = s.squeeze()
        d[c] = pd.to_numeric(s, errors="coerce").astype(float)

    d.index = pd.to_datetime(d.index)
    d = d.sort_index()
    d = d.dropna(subset=["Close"])
    if d.empty:
        return None
    return d

@st.cache_data(show_spinner=False, ttl=300)
def download_ohlcv(ticker: str, period: str, interval: str):
    try:
        raw = yf.download(ticker, period=period, interval=interval, auto_adjust=True, progress=False)
    except Exception as e:
        st.error(f"Gagal mengambil data: {e}")
        return None
    return sanitize_ohlcv(raw, ticker)

def resample_to_4h(df_60m: pd.DataFrame) -> pd.DataFrame:
    if df_60m is None or df_60m.empty:
        return None
    d = df_60m.copy()
    d.index = pd.to_datetime(d.index)
    d = d.sort_index()
    d4 = d.resample("4H").agg({
        "Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"
    }).dropna(subset=["Close"])
    return d4 if not d4.empty else None

# -------------------------
# Indikator manual (tanpa 'ta')
# -------------------------
def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    up = pd.Series(np.where(delta > 0, delta, 0.0), index=series.index)
    down = pd.Series(np.where(delta < 0, -delta, 0.0), index=series.index)
    roll_up = up.ewm(alpha=1/window, adjust=False).mean()
    roll_down = down.ewm(alpha=1/window, adjust=False).mean()
    rs = roll_up / roll_down.replace(0, np.nan)
    rsi_val = 100 - (100 / (1 + rs))
    return rsi_val

def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    macd_line = ema(series, fast) - ema(series, slow)
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    atr_val = tr.ewm(alpha=1/window, adjust=False).mean()
    return atr_val

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["MA9"] = d["Close"].rolling(9, min_periods=3).mean()
    d["RSI14"] = rsi(d["Close"], 14)
    d["MACD"], d["MACD_signal"], d["MACD_hist"] = macd(d["Close"])
    d["ATR14"] = atr(d["High"], d["Low"], d["Close"], 14)
    d["VolMA20"] = d["Volume"].rolling(20, min_periods=5).mean()
    return d

# -------------------------
# Level S/R & Trading Plan
# -------------------------
def pivot_sr_lastbar(df: pd.DataFrame):
    last = df.iloc[-1]
    h, l, c = float(last["High"]), float(last["Low"]), float(last["Close"])
    pivot = (h + l + c) / 3.0
    r1 = (2 * pivot) - l
    s1 = (2 * pivot) - h
    r2 = pivot + (h - l)
    s2 = pivot - (h - l)
    return pivot, r1, r2, s1, s2

def plan_entry_tp_sl(df_i: pd.DataFrame, swing_window: int = 10):
    c = float(df_i["Close"].iloc[-1])
    pivot, r1, r2, s1, s2 = pivot_sr_lastbar(df_i)
    swing_res = float(df_i["High"].tail(swing_window).max())
    swing_sup = float(df_i["Low"].tail(swing_window).min())
    atr_val = float(df_i["ATR14"].iloc[-1]) if not pd.isna(df_i["ATR14"].iloc[-1]) else max(c * 0.02, 1e-6)

    # Entry (<= c): ambil support terdekat/MA9/pivot
    cand = [df_i["MA9"].iloc[-1], pivot, s1, swing_sup]
    cand = [x for x in cand if (x is not None) and (not pd.isna(x)) and (x <= c)]
    entry = max(cand) if cand else c * 0.99

    # TP (>= c): resistance terdekat
    res = [r1, r2, swing_res]
    res = [x for x in res if x >= c]
    tp = min(res) if res else c + 1.5 * atr_val

    # SL (<= c): support terdekat
    sups = [s1, s2, swing_sup]
    sups = [x for x in sups if x <= c]
    sl = max(sups) if sups else c - 1.0 * atr_val

    risk = max(c - sl, 1e-6)
    reward = max(tp - c, 1e-6)
    if reward / risk < 1.0:
        tp = c + max(risk * 1.2, 1.2 * atr_val)
        reward = tp - c
    rr = reward / risk
    return float(entry), float(tp), float(sl), float(rr), (pivot, r1, r2, s1, s2, swing_res, swing_sup), atr_val

def simple_signal(row: pd.Series) -> str:
    try:
        c = float(row["Close"]); ma9 = float(row["MA9"])
        rsi14 = float(row["RSI14"]); macd_v = float(row["MACD"]); sig = float(row["MACD_signal"])
    except Exception:
        return "WAIT"
    if c > ma9 and rsi14 < 70 and macd_v > sig:
        return "BUY"
    if c < ma9 and rsi14 > 50 and macd_v < sig:
        return "SELL"
    return "HOLD"

# -------------------------
# Chart
# -------------------------
def plot_chart(df_i: pd.DataFrame, entry, tp, sl, levels):
    pivot, r1, r2, s1, s2, swing_res, swing_sup = levels
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True,
                        row_heights=[0.5, 0.12, 0.18, 0.18],
                        vertical_spacing=0.02,
                        subplot_titles=("Harga & MA9 + Levels", "Volume", "RSI(14)", "MACD"))

    # Price
    fig.add_trace(go.Candlestick(
        x=df_i.index, open=df_i["Open"], high=df_i["High"], low=df_i["Low"], close=df_i["Close"], name="Price"
    ), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_i.index, y=df_i["MA9"], mode="lines", name="MA9",
                             line=dict(color="orange", width=1.2)), row=1, col=1)

    for y, txt, color in [
        (entry,"Entry","#3498db"), (tp,"TP","#2ecc71"), (sl,"SL","#e74c3c"),
        (pivot,"Pivot","#7f8c8d"), (r1,"R1","#e67e22"), (r2,"R2","#d35400"),
        (s1,"S1","#27ae60"), (s2,"S2","#16a085"),
        (swing_res,"SwingHigh","#c0392b"), (swing_sup,"SwingLow","#2980b9")
    ]:
        fig.add_hline(y=y, line_dash="dot", line_color=color, annotation_text=txt,
                      annotation_position="top right", row=1, col=1)

    # Volume
    if "Volume" in df_i.columns:
        fig.add_trace(go.Bar(x=df_i.index, y=df_i["Volume"], name="Volume", marker_color="lightblue"), row=2, col=1)
        if "VolMA20" in df_i.columns:
            fig.add_trace(go.Scatter(x=df_i.index, y=df_i["VolMA20"], mode="lines", name="VolMA20",
                                     line=dict(color="orange", width=1)), row=2, col=1)

    # RSI
    fig.add_trace(go.Scatter(x=df_i.index, y=df_i["RSI14"], mode="lines", name="RSI",
                             line=dict(color="yellow", width=1.3)), row=3, col=1)
    fig.add_hline(y=70, line_dash="dot", line_color="red", row=3, col=1)
    fig.add_hline(y=30, line_dash="dot", line_color="green", row=3, col=1)

    # MACD
    fig.add_trace(go.Scatter(x=df_i.index, y=df_i["MACD"], mode="lines", name="MACD",
                             line=dict(color="cyan", width=1.3)), row=4, col=1)
    fig.add_trace(go.Scatter(x=df_i.index, y=df_i["MACD_signal"], mode="lines", name="Signal",
                             line=dict(color="magenta", width=1)), row=4, col=1)
    fig.add_hline(y=0, line_dash="dot", line_color="white", row=4, col=1)

    fig.update_layout(template="plotly_dark", xaxis_rangeslider_visible=False, height=900, showlegend=True)
    return fig

# -------------------------
# Sidebar
# -------------------------
with st.sidebar:
    st.markdown("## Pengaturan Analisis")
    raw_ticker = st.text_input("Ticker (contoh: GOTO atau AAPL)", value="BBCA")
    ticker = normalize_ticker(raw_ticker)
    st.caption(f"Ticker digunakan: **{ticker or '-'}**")
    avg_buy = st.number_input("Avg Buy (Rp per lembar)", min_value=0.0, value=0.0, step=0.01)
    lots = st.number_input("Jumlah lot (1 lot = 100 lembar)", min_value=0, value=0, step=1)
    st.markdown("---")
    st.caption("💡 BEI cukup ketik 3–4 huruf, otomatis ditambahkan `.JK`")

if not ticker:
    st.info("Masukkan ticker terlebih dahulu.")
    st.stop()

# -------------------------
# Ambil data untuk setiap timeframe
# -------------------------
periods = {
    "1H": ("60d", "60m"),
    "4H": ("60d", "60m"),   # nanti di-resample 4H
    "Daily": ("1y", "1d"),
    "Weekly": ("5y", "1wk"),
}

d1h = download_ohlcv(ticker, *periods["1H"])
d60m = download_ohlcv(ticker, *periods["4H"])
d4h = resample_to_4h(d60m) if d60m is not None else None
dd  = download_ohlcv(ticker, *periods["Daily"])
dw  = download_ohlcv(ticker, *periods["Weekly"])

# Header ringkas last prices
def last_close(df):
    try:
        return float(df["Close"].iloc[-1])
    except Exception:
        return np.nan

colA, colB, colC, colD = st.columns(4)
colA.metric("Last 1H", f"{last_close(d1h):,.2f}" if d1h is not None else "-")
colB.metric("Last 4H", f"{last_close(d4h):,.2f}" if d4h is not None else "-")
colC.metric("Last Daily", f"{last_close(dd):,.2f}" if dd is not None else "-")
colD.metric("Last Weekly", f"{last_close(dw):,.2f}" if dw is not None else "-")
st.markdown("---")

# -------------------------
# Tabs
# -------------------------
tab1, tab2, tab3, tab4 = st.tabs(["🕐 1H", "⏳ 4H", "📅 Daily", "📆 Weekly"])

def render_tf(df_src: pd.DataFrame, label: str):
    st.subheader(f"{label} Chart - {ticker}")
    if df_src is None or df_src.empty or df_src.shape[0] < 20:
        st.warning("Data timeframe ini belum cukup (butuh ≥ ~20 bar).")
        return
    df_i = add_indicators(df_src)
    entry, tp, sl, rr, levels, atr_val = plan_entry_tp_sl(df_i, swing_window=10)
    rekom = simple_signal(df_i.iloc[-1])

    # Info teknikal
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Close", f"{float(df_i['Close'].iloc[-1]):,.2f}")
    c2.metric("Rekomendasi", rekom)
    c3.metric("ATR(14)", f"{atr_val:,.2f}")
    c4.metric("R/R", f"{rr:,.2f}")
    st.markdown(
        f"**Entry:** `{entry:.2f}`  •  **TP:** `{tp:.2f}`  •  **SL:** `{sl:.2f}`  \n"
        f"**Pivot|R1|R2|S1|S2:** `{levels[0]:.2f}`|`{levels[1]:.2f}`|`{levels[2]:.2f}`|`{levels[3]:.2f}`|`{levels[4]:.2f}`  \n"
        f"**Swing High/Low(10):** `{levels[5]:.2f}` / `{levels[6]:.2f}`"
    )

    # Portofolio (opsional)
    if avg_buy > 0 and lots > 0:
        shares = int(lots * 100)
        modal = avg_buy * shares
        lastp = float(df_i["Close"].iloc[-1])
        nilai = lastp * shares
        pnl = nilai - modal
        pnl_pct = (pnl / modal * 100) if modal > 0 else 0.0
        status = "📈 UNTUNG" if pnl >= 0 else "📉 RUGI"
        st.info(f"💼 Lot: {lots} ({shares} lembar) • Modal: Rp {modal:,.0f} • Nilai: Rp {nilai:,.0f} • "
                f"P/L: Rp {pnl:,.0f} ({pnl_pct:.2f}%) • {status}")

    # Chart
    fig = plot_chart(df_i, entry, tp, sl, levels)
    st.plotly_chart(fig, use_container_width=True)

with tab1:
    render_tf(d1h, "1H")
with tab2:
    render_tf(d4h, "4H")
with tab3:
    render_tf(dd, "Daily")
with tab4:
    render_tf(dw, "Weekly")

# Footer
st.markdown("---")
st.caption("Catatan: Rekomendasi bersifat teknikal sederhana. Gunakan manajemen risiko & konfirmasi multi-timeframe.")