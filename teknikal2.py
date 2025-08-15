import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import ta
import plotly.graph_objects as go

# ---------------------------
# Ambil data dari Yahoo Finance
# ---------------------------
def get_data(ticker, period, interval):
    try:
        df = yf.download(ticker, period=period, interval=interval, auto_adjust=False)
        if df.empty:
            return pd.DataFrame()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] for col in df.columns]
        for col in ['Open', 'High', 'Low', 'Close']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)
        return df.dropna()
    except Exception as e:
        st.error(f"Gagal mengambil data {ticker} ({period} {interval}): {e}")
        return pd.DataFrame()

# ---------------------------
# Tambahkan indikator teknikal
# ---------------------------
def add_indicators(df):
    if df.empty or len(df) < 20:
        return df

    df['MA9'] = df['Close'].rolling(window=9).mean()
    df['RSI14'] = ta.momentum.RSIIndicator(close=df['Close'], window=14).rsi()
    macd = ta.trend.MACD(close=df['Close'])
    df['MACD'] = macd.macd()
    df['Signal'] = macd.macd_signal()
    df['ATR'] = ta.volatility.AverageTrueRange(
        high=df['High'], low=df['Low'], close=df['Close']
    ).average_true_range()
    return df

# ---------------------------
# Plot candlestick + MA
# ---------------------------
def plot_chart(df, title):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df.index, open=df['Open'], high=df['High'],
        low=df['Low'], close=df['Close'], name='Candles'
    ))
    fig.add_trace(go.Scatter(
        x=df.index, y=df['MA9'], mode='lines', name='MA9', line=dict(color='orange')
    ))
    fig.update_layout(title=title, xaxis_rangeslider_visible=False)
    return fig

# ---------------------------
# Deteksi sinyal dasar
# ---------------------------
def base_signal(df):
    if df.empty:
        return "Data