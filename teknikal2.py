# streamlit_multi_tf_pro.py
# Multi-Timeframe Stock Analyzer — PRO
# Features:
# - Auto-detect .JK for BEI tickers (cached)
# - Multi-timeframe analysis: 1h, 4h, Daily, Weekly
# - Indicators: MA9, RSI14, MACD, ATR14, VolMA20
# - Auto Entry / TP / SL
# - % to TP, % to SL, R/R Ratio
# - Confluence Score (how many timeframes BUY)
# - Volume breakout & Swing breakout checks
# - Colored result table + summary recommendation
# - Interactive plotly charts per timeframe

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import ta
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from typing import Tuple

st.set_page_config(page_title="Multi-Timeframe Stock Analyzer — PRO", layout="wide")
st.title("📈 Multi-Timeframe Stock Analyzer — PRO")

# -------------------------
# Utilities: cached download & ticker detect
# -------------------------
@st.cache_data(ttl=1800)  # cache 30 menit untuk unduhan data
def cached_download(ticker: str, period: str, interval: str):
    try:
        df = yf.download(ticker, period=period, interval=interval, auto_adjust=True, progress=False)
    except Exception:
        return None
    if df is None or df.empty:
        return None
    # if multiindex columns, try to simplify
    if isinstance(df.columns, pd.MultiIndex):
        try:
            # try to select the ticker column group
            df = df.xs(ticker, axis=1, level=-1)
        except Exception:
            df.columns = [c[0] for c in df.columns]
    # keep needed cols
    cols = [c for c in ['Open','High','Low','Close','Volume'] if c in df.columns]
    df = df[cols].copy()
    for c in df.columns:
        df[c] = pd.to_numeric(df[c].squeeze(), errors='coerce')
    df.dropna(subset=['Close'], inplace=True)
    return df

@st.cache_data(ttl=3600)  # cache deteksi ticker 1 jam
def detect_ticker_symbol(raw_ticker: str):
    rt = raw_ticker.upper().strip()
    if "." in rt:
        return rt
    # try plain
    try:
        df = cached_download(rt, period="5d", interval="1d")
        if df is not None and not df.empty:
            return rt
    except:
        pass
    # try .JK
    try:
        df = cached_download(f"{rt}.JK", period="5d", interval="1d")
        if df is not None and not df.empty:
            return f"{rt}.JK"
    except:
        pass
    return rt

# -------------------------
# Sidebar inputs
# -------------------------
with st.sidebar:
    st.markdown("## Input & Pengaturan")
    raw_ticker = st.text_input("Ticker (contoh: GOTO, AAPL, TLKM.JK)", value="GOTO")
    ticker = detect_ticker_symbol(raw_ticker)
    if ticker != raw_ticker.upper().strip():
        st.info(f"Ticker terdeteksi: **{ticker}**")
    else:
        st.write(f"Ticker digunakan: **{ticker}**")

    st.markdown("---")
    st.markdown("### Pengaturan tambahan")
    swing_window = st.number_input("Swing window (bars) untuk swing high/low", min_value=5, max_value=200, value=20)
    vol_break_mult = st.number_input("Multipler untuk menilai Volume breakout (Vol > mult * VolMA20)", min_value=1.0, max_value=5.0, value=1.1, step=0.1)
    rr_threshold = st.number_input("Minimal R/R yang diterima (untuk flag bagus)", min_value=0.1, max_value=10.0, value=1.0, step=0.1)
    st.markdown("---")
    st.write("Catatan: 4H dibentuk dari resample 60m → 4H. Cache data dipakai agar tidak hit Yahoo terlalu sering.")

# -------------------------
# Helper indicator & decision functions
# -------------------------
def resample_4h(df_60m: pd.DataFrame) -> pd.DataFrame:
    df = df_60m.copy()
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    df_4h = df.resample('4H').agg({'Open':'first','High':'max','Low':'min','Close':'last','Volume':'sum'})
    df_4h.dropna(subset=['Close'], inplace=True)
    return df_4h

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    close = out['Close']
    high = out['High'] if 'High' in out else close
    low = out['Low'] if 'Low' in out else close
    vol = out['Volume'] if 'Volume' in out else None

    out['MA9'] = close.rolling(9).mean()
    out['RSI14'] = ta.momentum.RSIIndicator(close=close, window=14).rsi()
    macd_obj = ta.trend.MACD(close=close)
    out['MACD'] = macd_obj.macd()
    out['MACD_signal'] = macd_obj.macd_signal()
    try:
        out['ATR14'] = ta.volatility.AverageTrueRange(high=high, low=low, close=close, window=14).average_true_range()
    except Exception:
        out['ATR14'] = np.nan
    if vol is not None:
        out['VolMA20'] = vol.rolling(20).mean()
    return out

def compute_sr_lastbar(df: pd.DataFrame) -> Tuple[float,float,float,float,float]:
    h = float(df['High'].iloc[-1]); l = float(df['Low'].iloc[-1]); c = float(df['Close'].iloc[-1])
    pivot = (h + l + c) / 3.0
    r1 = (2*pivot) - l; s1 = (2*pivot) - h
    r2 = pivot + (h - l); s2 = pivot - (h - l)
    return pivot, r1, r2, s1, s2

def compute_entry_tp_sl(df: pd.DataFrame, swing_window_local: int = 20) -> Tuple[float,float,float,float,float]:
    # returns entry, tp, sl, swing_res, swing_sup
    c = float(df['Close'].iloc[-1])
    pivot, r1, r2, s1, s2 = compute_sr_lastbar(df)
    swing_res = float(df['High'].tail(swing_window_local).max())
    swing_sup = float(df['Low'].tail(swing_window_local).min())
    atr = float(df['ATR14'].iloc[-1]) if 'ATR14' in df and not pd.isna(df['ATR14'].iloc[-1]) else c * 0.02

    # entry candidates <= price
    candidates = [lvl for lvl in [df['MA9'].iloc[-1], pivot, s1, swing_sup] if (not pd.isna(lvl)) and (lvl <= c)]
    entry = max(candidates) if candidates else c * 0.99

    # tp = nearest resistance >= price
    res_level = [lvl for lvl in [r1, r2, swing_res] if lvl >= c]
    tp = min(res_level) if res_level else c + 1.5 * atr

    # sl = nearest support <= price
    sup_level = [lvl for lvl in [s1, s2, swing_sup] if lvl <= c]
    sl = max(sup_level) if sup_level else c - 1.0 * atr

    # Ensure minimal RR by adjusting TP upward if necessary
    risk = c - sl
    reward = tp - c
    if risk > 0 and reward / risk < 1.0:
        tp = c + max(risk * 1.2, 1.2 * atr)
        reward = tp - c

    return float(entry), float(tp), float(sl), float(swing_res), float(swing_sup)

def recommend_from_row(row) -> str:
    # simple recommendation logic per timeframe (same as earlier)
    try:
        c = float(row['Close'])
        ma9 = float(row['MA9'])
        rsi = float(row['RSI14'])
        macd = float(row['MACD'])
        sig = float(row['MACD_signal'])
    except Exception:
        return "WAIT"
    if c > ma9 and rsi < 70 and macd > sig:
        return "BUY"
    if c < ma9 and rsi > 50 and macd < sig:
        return "SELL"
    return "HOLD"

# -------------------------
# Timeframes config
# -------------------------
timeframes = {
    "1h": {"interval": "60m", "period": "60d", "resample": False},
    "4h": {"interval": "60m", "period": "60d", "resample": True},
    "Daily": {"interval": "1d", "period": "1y", "resample": False},
    "Weekly": {"interval": "1wk", "period": "5y", "resample": False}
}

# -------------------------
# Fetch & compute per timeframe
# -------------------------
results = []
errors = []
for tf, cfg in timeframes.items():
    df_raw = cached_download(ticker, period=cfg["period"], interval=cfg["interval"])
    if df_raw is None:
        errors.append(tf)
        continue
    df = resample_4h(df_raw) if cfg["resample"] else df_raw.copy()
    if df.shape[0] < 10:
        errors.append(tf)
        continue
    df_i = compute_indicators(df)
    # ensure not NaN on last required fields
    if pd.isna(df_i['Close'].iloc[-1]) or pd.isna(df_i['MA9'].iloc[-1]):
        errors.append(tf)
        continue

    entry, tp, sl, swing_res, swing_sup = compute_entry_tp_sl(df_i, swing_window_local=swing_window)
    rekom = recommend_from_row(df_i.iloc[-1])
    last = df_i.iloc[-1]

    # % to TP / SL and R/R
    close_price = float(last['Close'])
    pct_to_tp = (tp - close_price) / close_price * 100.0 if close_price != 0 else np.nan
    pct_to_sl = (close_price - sl) / close_price * 100.0 if close_price != 0 else np.nan
    rr = (tp - close_price) / (close_price - sl) if (close_price - sl) > 0 else np.nan

    # volume breakout
    vol = float(last['Volume']) if 'Volume' in last and not pd.isna(last['Volume']) else np.nan
    volma = float(last['VolMA20']) if 'VolMA20' in last and not pd.isna(last['VolMA20']) else np.nan
    vol_break = False
    if not pd.isna(vol) and not pd.isna(volma) and volma > 0:
        vol_break = vol > (vol_break_mult * volma)

    # swing breakout (price > recent swing high)
    swing_break = False
    if not pd.isna(swing_res):
        swing_break = close_price > swing_res

    results.append({
        "Timeframe": tf,
        "df": df_i,
        "Close": close_price,
        "MA9": float(last['MA9']),
        "RSI14": float(last['RSI14']),
        "MACD": float(last['MACD']),
        "Signal": float(last['MACD_signal']),
        "ATR14": float(last['ATR14']) if 'ATR14' in last and not pd.isna(last['ATR14']) else np.nan,
        "Entry": entry,
        "TP": tp,
        "SL": sl,
        "PctTP": pct_to_tp,
        "PctSL": pct_to_sl,
        "RR": rr,
        "Rekom": rekom,
        "Vol": vol,
        "VolMA20": volma,
        "VolBreak": vol_break,
        "SwingRes": swing_res,
        "SwingSup": swing_sup,
        "SwingBreak": swing_break
    })

# Show errors if any timeframe missing
if errors:
    st.warning(f"Beberapa timeframe tidak memiliki data: {', '.join(errors)} (cek ticker / period).")

# -------------------------
# Summary & Confluence
# -------------------------
# build dataframe for presentation
df_rows = []
for r in results:
    df_rows.append({
        "Timeframe": r["Timeframe"],
        "Close": f"{r['Close']:.2f}",
        "MA9": f"{r['MA9']:.2f}",
        "RSI14": f"{r['RSI14']:.2f}",
        "MACD": f"{r['MACD']:.3f}",
        "Signal": f"{r['Signal']:.3f}",
        "ATR14": f"{(r['ATR14'] if not pd.isna(r['ATR14']) else np.nan):.2f}",
        "Entry": f"{r['Entry']:.2f}",
        "TP": f"{r['TP']:.2f}",
        "SL": f"{r['SL']:.2f}",
        "%→TP": f"{r['PctTP']:.2f}%",
        "%→SL": f"{r['PctSL']:.2f}%",
        "R/R": f"{(r['RR'] if not pd.isna(r['RR']) else np.nan):.2f}",
        "Vol": f"{int(r['Vol']) if not pd.isna(r['Vol']) else '-'}",
        "VolMA20": f"{int(r['VolMA20']) if not pd.isna(r['VolMA20']) else '-'}",
        "VolBreak": "YES" if r['VolBreak'] else "NO",
        "SwingBreak": "YES" if r['SwingBreak'] else "NO",
        "Rekomendasi": r['Rekom']
    })

df_summary = pd.DataFrame(df_rows).set_index("Timeframe")

# compute confluence (count BUY)
confluence_count = sum(1 for r in results if r["Rekom"] == "BUY")
total_tf = len(results)
confl_pct = confluence_count / total_tf * 100 if total_tf > 0 else 0

# final recommendation logic (simple, rule-based)
final_rekom = "HOLD"
if confl_pct >= 75 and any(r["Timeframe"]=="Weekly" and r["Rekom"]=="BUY" for r in results):
    final_rekom = "STRONG BUY"
elif confl_pct >= 50:
    final_rekom = "BUY"
elif confl_pct == 0 and any(r["Rekom"]=="SELL" for r in results):
    final_rekom = "SELL"
else:
    final_rekom = "HOLD"

# more nuance: if any BUY but R/R poor across most timeframes, downgrade to HOLD
avg_rr = np.nanmean([r['RR'] for r in results if not pd.isna(r['RR'])]) if results else np.nan
if final_rekom in ["STRONG BUY","BUY"] and (pd.isna(avg_rr) or avg_rr < rr_threshold):
    final_rekom = "HOLD (R/R rendah)"

# Summary card
st.subheader(f"Ringkasan Multi-Timeframe — {ticker}")
col1, col2, col3 = st.columns([2,2,4])
col1.metric("Confluence (BUY tf)", f"{confluence_count}/{total_tf}")
col2.metric("Avg R/R (timeframes)", f"{avg_rr:.2f}" if not pd.isna(avg_rr) else "-")
col3.metric("Rekomendasi Akhir", final_rekom)

st.markdown(f"- Confluence: **{confluence_count}** dari **{total_tf}** timeframe mendukung BUY ({confl_pct:.0f}%).")
st.markdown(f"- R/R rata-rata: **{avg_rr:.2f}** (threshold R/R minimal = {rr_threshold}).")

if final_rekom.startswith("STRONG BUY") or final_rekom=="BUY":
    st.success(f"Rekomendasi akhir: **{final_rekom}**")
elif final_rekom.startswith("HOLD"):
    st.info(f"Rekomendasi akhir: **{final_rekom}**")
else:
    st.warning(f"Rekomendasi akhir: **{final_rekom}**")

st.markdown("---")

# -------------------------
# Styled dataframe (colored rekom)
# -------------------------
def color_rekom(val):
    if val == "BUY":
        return 'background-color: #2ecc71; color: white; font-weight: bold'
    if val == "STRONG BUY":
        return 'background-color: #27ae60; color: white; font-weight: bold'
    if val == "SELL":
        return 'background-color: #e74c3c; color: white; font-weight: bold'
    if "R/R rendah" in str(val):
        return 'background-color: #f39c12; color: white; font-weight: bold'
    return 'background-color: #95a5a6; color: white; font-weight: bold'

# highlight RSI extremes
def highlight_rsi(val):
    try:
        v = float(str(val).replace('%',''))
    except:
        return ''
    if v >= 70:
        return 'color: #ff6b6b; font-weight: bold'  # overbought
    if v <= 30:
        return 'color: #2ecc71; font-weight: bold'  # oversold
    return ''

styled = df_summary.style.applymap(color_rekom, subset=["Rekomendasi"]) \
                         .applymap(highlight_rsi, subset=["RSI14"])

st.dataframe(styled, use_container_width=True)

# -------------------------
# Charts per timeframe
# -------------------------
for r in results:
    st.markdown(f"### {r['Timeframe']} — Harga terakhir: {r['Close']:.2f} • Rekom: {r['Rekom']}")
    df_i = r["df"]
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True,
                        row_heights=[0.5, 0.12, 0.18, 0.18],
                        vertical_spacing=0.02,
                        subplot_titles=("Harga & MA9 + Levels", "Volume", "RSI(14)", "MACD"))

    # Price candles
    fig.add_trace(go.Candlestick(x=df_i.index, open=df_i['Open'], high=df_i['High'],
                                 low=df_i['Low'], close=df_i['Close'], name="Price"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_i.index, y=df_i['MA9'], mode='lines', name='MA9', line=dict(color='orange')), row=1, col=1)

    # Add Entry / TP / SL lines
    fig.add_hline(y=r["Entry"], line_dash="dash", line_color="#3498db", annotation_text="Entry", annotation_position="top right", row=1, col=1)
    fig.add_hline(y=r["TP"], line_dash="dash", line_color="#2ecc71", annotation_text="TP", annotation_position="top right", row=1, col=1)
    fig.add_hline(y=r["SL"], line_dash="dash", line_color="#e74c3c", annotation_text="SL", annotation_position="top right", row=1, col=1)

    # show swing res/sup
    fig.add_hline(y=r["SwingRes"], line_dash="dot", line_color="#c0392b", annotation_text="SwingHigh", row=1, col=1)
    fig.add_hline(y=r["SwingSup"], line_dash="dot", line_color="#2980b9", annotation_text="SwingLow", row=1, col=1)

    # Volume
    if 'Volume' in df_i.columns:
        fig.add_trace(go.Bar(x=df_i.index, y=df_i['Volume'], name='Volume', marker_color='lightblue'), row=2, col=1)
        if 'VolMA20' in df_i.columns:
            fig.add_trace(go.Scatter(x=df_i.index, y=df_i['VolMA20'], mode='lines', name='VolMA20', line=dict(color='orange')), row=2, col=1)

    # RSI
    fig.add_trace(go.Scatter(x=df_i.index, y=df_i['RSI14'], mode='lines', name='RSI', line=dict(color='yellow')), row=3, col=1)
    fig.add_hline(y=70, line_dash="dot", line_color="red", row=3, col=1)
    fig.add_hline(y=30, line_dash="dot", line_color="green", row=3, col=1)

    # MACD
    fig.add_trace(go.Scatter(x=df_i.index, y=df_i['MACD'], mode='lines', name='MACD', line=dict(color='cyan')), row=4, col=1)
    fig.add_trace(go.Scatter(x=df_i.index, y=df_i['MACD_signal'], mode='lines', name='Signal', line=dict(color='magenta')), row=4, col=1)
    fig.add_hline(y=0, line_dash="dot", line_color="white", row=4, col=1)

    fig.update_layout(template="plotly_dark", xaxis_rangeslider_visible=False, height=900, showlegend=True)
    st.plotly_chart(fig, use_container_width=True)

    # small notes per timeframe
    st.markdown(f"- VolBreak: **{'YES' if r['VolBreak'] else 'NO'}**  •  SwingBreak: **{'YES' if r['SwingBreak'] else 'NO'}**  •  R/R: **{(r['RR'] if not pd.isna(r['RR']) else '-'): .2f}**")
    st.markdown("---")

# -------------------------
# Footer tips
# -------------------------
st.markdown("**Catatan penting:**")
st.markdown("- Rekomendasi bersifat teknikal dan rule-based; selalu gunakan manajemen risiko dan cek berita/fundamental.")
st.markdown("- Gunakan Weekly/Daily untuk arah tren besar, 4H/1H untuk timing entry.")
st.markdown("- Jika mau saya tambahkan export CSV / notifikasi telegram/email, beri tahu.")