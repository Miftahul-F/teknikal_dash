# streamlit_ministockbit.py

import streamlit as st
import requests, json, websocket, threading
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime

BASE_URL = "https://api.stockbit.com"

st.title("Real-Time Analyzer Stockbit (Mini)")

# Sidebar: login ke Stockbit
with st.sidebar:
    st.subheader("Login")
    USER = st.text_input("Email / Username")
    PASS = st.text_input("Password", type="password")
    if st.button("Login"):
        resp = requests.post(f"{BASE_URL}/v2/auth/login", json={"username": USER, "password": PASS})
        data = resp.json()
        token = data.get("data", {}).get("token")
        if token:
            st.session_state.token = token
            st.success("Login berhasil")
        else:
            st.error("Login gagal")

if "token" not in st.session_state:
    st.stop()

token = st.session_state.token
headers = {"Authorization": f"Bearer {token}"}

# Input ticker
ticker = st.text_input("Kode saham (IDX)","BBRI").upper()

# REST: ambil snapshot harga
if st.button("Fetch Now"):
    resp = requests.get(f"{BASE_URL}/v2/trade/quotes?symbols={ticker}", headers=headers)
    st.json(resp.json())

# Real-time WebSocket stream
st.subheader("Real-Time Ticker")

if st.button("Start Stream"):
    st.session_state.prices = []

    def on_message(ws, msg):
        j = json.loads(msg)
        if j.get("symbol")==ticker and "last" in j:
            st.session_state.prices.append({"time":datetime.now(), "price": float(j["last"])})
            st.session_state.prices = st.session_state.prices[-100:]

    def on_open(ws):
        ws.send(json.dumps({"type":"subscribe","symbols":[ticker]}))

    def run_ws():
        ws = websocket.WebSocketApp("wss://ws.stockbit.com",
            header=[f"Authorization:Bearer {token}"],
            on_message=on_message, on_open=on_open)
        ws.run_forever()

    threading.Thread(target=run_ws, daemon=True).start()
    st.success("Streaming...")

if "prices" in st.session_state and st.session_state.prices:
    df = pd.DataFrame(st.session_state.prices)
    fig = go.Figure(go.Scatter(x=df.time, y=df.price))
    st.plotly_chart(fig, use_container_width=True)