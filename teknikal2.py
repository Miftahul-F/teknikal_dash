# streamlit_stockbit.py
import streamlit as st
import requests, json, websocket, threading
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime

BASE_URL = "https://api.stockbit.com"

st.set_page_config(page_title="Stockbit Analyzer", layout="wide")
st.title("📊 Stockbit Real-Time Analyzer")

# --------------------
# Login Section
# --------------------
with st.sidebar:
    st.subheader("🔑 Login Broker")
    username = st.text_input("Username / Email")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        try:
            resp = requests.post(f"{BASE_URL}/v2/auth/login",
                                 json={"username": username, "password": password})
            data = resp.json()
            if "data" in data and "token" in data["data"]:
                st.session_state["token"] = data["data"]["token"]
                st.success("Login berhasil ✅")
            else:
                st.error("Login gagal ❌")
        except Exception as e:
            st.error(f"Gagal login: {e}")

# stop if not login
if "token" not in st.session_state:
    st.warning("Silakan login dulu")
    st.stop()

token = st.session_state["token"]

# --------------------
# Input saham
# --------------------
ticker = st.text_input("Kode saham (contoh: BBRI)", value="BBRI").upper()

col1, col2 = st.columns(2)
with col1:
    if st.button("Ambil Data Sekarang"):
        headers = {"Authorization": f"Bearer {token}"}
        resp = requests.get(f"{BASE_URL}/v2/trade/quotes?symbols={ticker}", headers=headers)
        data = resp.json()
        st.json(data)

# --------------------
# Real-Time WebSocket
# --------------------
st.subheader("📡 Real-Time Feed")

if st.button("Start Stream"):
    st.session_state["prices"] = []

    def on_message(ws, message):
        msg = json.loads(message)
        if "symbol" in msg and msg["symbol"] == ticker:
            ts = datetime.now()
            px = float(msg["last"]) if "last" in msg else None
            if px:
                st.session_state["prices"].append({"time": ts, "price": px})
                # keep last 100
                st.session_state["prices"] = st.session_state["prices"][-100:]

    def on_open(ws):
        ws.send(json.dumps({"type": "subscribe", "symbols": [ticker]}))

    def run_ws():
        ws = websocket.WebSocketApp(
            "wss://ws.stockbit.com",
            header=[f"Authorization: Bearer {token}"],
            on_message=on_message,
            on_open=on_open
        )
        ws.run_forever()

    thread = threading.Thread(target=run_ws, daemon=True)
    thread.start()
    st.success("Streaming dimulai ✅")

# --------------------
# Chart harga real-time
# --------------------
if "prices" in st.session_state and len(st.session_state["prices"]) > 0:
    df = pd.DataFrame(st.session_state["prices"])
    fig = go.Figure(go.Scatter(x=df["time"], y=df["price"], mode="lines+markers"))
    fig.update_layout(title=f"Real-Time {ticker}", xaxis_title="Waktu", yaxis_title="Harga")
    st.plotly_chart(fig, use_container_width=True)

# --------------------
# Order Section
# --------------------
st.subheader("🛒 Kirim Order")
side = st.radio("Aksi", ["BUY", "SELL"], horizontal=True)
price = st.number_input("Harga", min_value=0.0, step=1.0)
qty = st.number_input("Jumlah lembar", min_value=100, step=100)

if st.button("Kirim Order"):
    headers = {"Authorization": f"Bearer {token}"}
    order = {"symbol": ticker, "side": side, "price": price, "qty": qty, "order_type": "LIMIT"}
    resp = requests.post(f"{BASE_URL}/v2/trade/orders", headers=headers, json=order)
    st.json(resp.json())