# streamlit_app.py — Multi-Timeframe Stock Analyzer Pro (final)

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import ta
import plotly.graph_objects as go

# =========================
# Utils & Normalizer
# =========================
def normalize_ticker(raw: str) -> str:
    """Kalau user tidak menulis suffix, default-kan ke BEI (.JK)."""
    s = (raw or "").strip().upper()
    if not s:
        return ""
    return s if "." in s else f"{s}.JK"

def ensure_numeric_1d(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten MultiIndex kolom yfinance & pastikan kolom OHLC numeric 1D."""
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    cols_need = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    df = df[cols_need].copy()
    for c in cols_need:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["Close"]).sort_index()
    return df

# =========================
# Data Fetcher
# =========================
@st.cache_data(show_spinner=False, ttl=300)
def get_data(ticker: str, period: str, interval: str) -> pd.DataFrame:
    try:
        df = yf.download(ticker, period=period, interval=interval, auto_adjust=False, progress=False)
        return ensure_numeric_1d(df)
    except Exception as e:
        st.error(f"Gagal mengambil data {ticker} ({period}/{interval}): {e}")
        return pd.DataFrame()

# =========================
# Indicators (aman)
# =========================
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Tambahkan MA9, RSI(14), MACD, Signal, ATR. Aman untuk data pendek."""
    if df is None or df.empty:
        return pd.DataFrame(columns=["Open","High","Low","Close","Volume","MA9","RSI14","MACD","Signal","ATR"])
    out = df.copy()

    # Window aman — ta.* akan mengembalikan NaN di awal, pastikan 1D series
    close = out["Close"]
    high  = out["High"] if "High" in out else close
    low   = out["Low"]  if "Low"  in out else close

    # MA9
    out["MA9"] = close.rolling(9, min_periods=1).mean()

    # RSI / MACD / ATR: bungkus try agar tetap aman jika data terlalu pendek
    try:
        out["RSI14"] = ta.momentum.RSIIndicator(close=close, window=14).rsi()
    except Exception:
        out["RSI14"] = np.nan

    try:
        macd_obj = ta.trend.MACD(close=close)
        out["MACD"] = macd_obj.macd()
        out["Signal"] = macd_obj.macd_signal()
    except Exception:
        out["MACD"] = np.nan
        out["Signal"] = np.nan

    try:
        atr_obj = ta.volatility.AverageTrueRange(high=high, low=low, close=close, window=14)
        out["ATR"] = atr_obj.average_true_range()
    except Exception:
        out["ATR"] = np.nan

    return out

# =========================
# Signals & Scoring
# =========================
def get_trend(df: pd.DataFrame) -> str:
    """UP jika Close > MA9; else DOWN; jika data tak cukup -> '-'."""
    if df is None or df.empty or "MA9" not in df:
        return "-"
    last = df.iloc[-1]
    if pd.isna(last.get("MA9", np.nan)) or pd.isna(last.get("Close", np.nan)):
        return "-"
    return "UP" if last["Close"] > last["MA9"] else "DOWN"

def has_entry_signal(df: pd.DataFrame) -> bool:
    """Entry jika MACD cross-up & RSI > 50. Aman untuk data < 2 bar."""
    if df is None or df.empty or "MACD" not in df or "Signal" not in df or "RSI14" not in df:
        return False
    if len(df.dropna(subset=["MACD","Signal"])) < 2:
        return False
    macd_now   = df["MACD"].iloc[-1]
    macd_prev  = df["MACD"].iloc[-2]
    sig_now    = df["Signal"].iloc[-1]
    sig_prev   = df["Signal"].iloc[-2]
    rsi_now    = df["RSI14"].iloc[-1] if not pd.isna(df["RSI14"].iloc[-1]) else 0
    if any(pd.isna(x) for x in [macd_now, macd_prev, sig_now, sig_prev]):
        return False
    return (macd_now > sig_now) and (macd_prev <= sig_prev) and (rsi_now > 50)

def confidence_score(trend_w, trend_d, e4, e1) -> int:
    score = 0
    if trend_w == "UP": score += 30
    if trend_d == "UP": score += 30
    if e4: score += 20
    if e1: score += 20
    return int(score)

def buy_match_status(current_price: float, entry_price: float, atr: float) -> tuple[str,str]:
    """Kembalikan (pesan, warna). Toleransi max(0.5% * entry, 0.2*ATR)."""
    if any(pd.isna(x) for x in [current_price, entry_price]):
        return ("Data harga tidak memadai", "warning")
    tol = max(0.005 * entry_price, 0.2 * (atr if not pd.isna(atr) else 0))
    diff = current_price - entry_price
    if abs(diff) <= tol:
        return ("✅ MATCH (harga di area entry)", "success")
    elif current_price < entry_price:
        return ("⏳ Belum nyentuh harga entry", "info")
    else:
        return ("⚠️ Harga sudah melewati area entry", "error")

# =========================
# Plotting
# =========================
def plot_chart(df: pd.DataFrame, title: str, levels: dict | None = None) -> go.Figure:
    fig = go.Figure()
    if df is None or df.empty:
        fig.update_layout(title=f"{title} (data tidak tersedia)")
        return fig

    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"], name="Price"
    ))
    if "MA9" in df:
        fig.add_trace(go.Scatter(x=df.index, y=df["MA9"], mode="lines", name="MA9", line=dict(color="orange")))
    # Horizontal level lines via shapes (lebih kompatibel)
    if levels:
        x0 = df.index.min()
        x1 = df.index.max()
        shape_map = {
            "Entry": {"color":"#3498db"},
            "Target": {"color":"#2ecc71"},
            "Stop": {"color":"#e74c3c"},
        }
        for key, y in levels.items():
            if pd.isna(y): 
                continue
            color = shape_map.get(key, {}).get("color", "#888")
            fig.add_shape(type="line", x0=x0, x1=x1, y0=y, y1=y,
                          line=dict(color=color, width=1, dash="dot"))
            fig.add_annotation(x=x1, y=y, text=key, showarrow=False, font=dict(color=color), xanchor="right", yanchor="bottom")

    fig.update_layout(title=title, xaxis_rangeslider_visible=False, height=500, template="plotly_white")
    return fig

# =========================
# Streamlit UI
# =========================
st.set_page_config(page_title="Multi-Timeframe Stock Analyzer Pro", layout="wide")
st.title("📊 Multi-Timeframe Stock Analyzer Pro")

with st.sidebar:
    st.header("Pengaturan Analisis")
    raw_ticker = st.text_input("Ticker (contoh: BBCA atau AAPL)", value="BBCA")
    ticker = normalize_ticker(raw_ticker)
    avg_buy = st.number_input("Avg Buy (Rp per lembar)", min_value=0.0, value=0.0, step=1.0)
    lots    = st.number_input("Jumlah lot (1 lot = 100)", min_value=0, value=0, step=1)
    st.caption("💡 BEI otomatis ditambah **.JK** jika tidak diisi.")

if not ticker:
    st.warning("Masukkan ticker terlebih dahulu.")
    st.stop()

# Ambil data per timeframe
df_w  = add_indicators(get_data(ticker, period="2y",  interval="1wk"))
df_d  = add_indicators(get_data(ticker, period="1y",  interval="1d"))
df_h4 = add_indicators(get_data(ticker, period="60d", interval="4h"))
df_h1 = add_indicators(get_data(ticker, period="30d", interval="1h"))

# Analisis multi-timeframe
trend_w = get_trend(df_w)
trend_d = get_trend(df_d)
entry4  = has_entry_signal(df_h4)
entry1  = has_entry_signal(df_h1)
score   = confidence_score(trend_w, trend_d, entry4, entry1)

# Tabs
tab_w, tab_d, tab_h4, tab_h1, tab_rec = st.tabs(["📆 Weekly", "📅 Daily", "⏳ 4H", "🕐 1H", "💡 Rekomendasi"])

with tab_w:
    st.subheader(f"Weekly - {ticker}")
    st.plotly_chart(plot_chart(df_w, "Weekly Chart"), use_container_width=True)
    st.write(f"Trend: **{trend_w}**")

with tab_d:
    st.subheader(f"Daily - {ticker}")
    st.plotly_chart(plot_chart(df_d, "Daily Chart"), use_container_width=True)
    st.write(f"Trend: **{trend_d}**")

with tab_h4:
    st.subheader(f"4H - {ticker}")
    st.plotly_chart(plot_chart(df_h4, "4H Chart"), use_container_width=True)
    st.write(f"Entry signal (MACD cross-up & RSI>50): **{'YES' if entry4 else 'NO'}**")

with tab_h1:
    st.subheader(f"1H - {ticker}")
    st.plotly_chart(plot_chart(df_h1, "1H Chart"), use_container_width=True)
    st.write(f"Entry signal (MACD cross-up & RSI>50): **{'YES' if entry1 else 'NO'}**")

with tab_rec:
    st.subheader("Ringkas Rekomendasi")
    colA, colB, colC, colD, colE = st.columns(5)
    colA.metric("Weekly", trend_w)
    colB.metric("Daily", trend_d)
    colC.metric("H4 Entry", "YES" if entry4 else "NO")
    colD.metric("H1 Entry", "YES" if entry1 else "NO")
    colE.metric("Confidence", f"{score}%")

    st.progress(min(max(score/100, 0), 1.0))

    # Syarat beli: Weekly & Daily UP + (H4 atau H1 entry)
    can_buy = (trend_w == "UP") and (trend_d == "UP") and (entry4 or entry1)
    if not df_d.empty:
        last_close = float(df_d["Close"].iloc[-1])
        atr_d      = float(df_d["ATR"].iloc[-1]) if "ATR" in df_d and not pd.isna(df_d["ATR"].iloc[-1]) else np.nan
    else:
        last_close, atr_d = np.nan, np.nan

    if can_buy and not pd.isna(last_close):
        entry_price = last_close
        # Jika ATR NaN (data terlalu pendek), fallback 2% dari harga
        risk_unit  = atr_d if not pd.isna(atr_d) else 0.02 * entry_price
        stop_price = entry_price - 1.5 * risk_unit
        target     = entry_price + 3.0 * risk_unit

        st.success(f"✅ **Rekomendasi BELI** sekitar **Rp {entry_price:,.2f}**")
        st.write(f"🎯 **Target**: Rp {target:,.2f}   |   🛑 **Stop**: Rp {stop_price:,.2f}")

        # Status pembelian
        msg, color = buy_match_status(current_price=last_close, entry_price=entry_price, atr=atr_d)
        getattr(st, color)(f"📌 {msg}")

        # Chart daily dengan level
        levels = {"Entry": entry_price, "Target": target, "Stop": stop_price}
        st.plotly_chart(plot_chart(df_d, "Daily Chart + Levels", levels), use_container_width=True)
    else:
        st.warning("❌ Belum ada kondisi beli ideal (butuh Weekly & Daily **UP** dan sinyal entry di H4/H1).")

    # P/L sederhana jika user isi posisi
    if (avg_buy or 0) > 0 and (lots or 0) > 0 and not pd.isna(last_close):
        shares = int(lots) * 100
        pnl = (last_close - avg_buy) * shares
        pnl_pct = ((last_close - avg_buy) / avg_buy * 100) if avg_buy > 0 else 0
        st.info(f"💼 P/L: **Rp {pnl:,.0f}**  ( {pnl_pct:.2f}% ) — Qty: {shares} lembar")

# ====== End ======