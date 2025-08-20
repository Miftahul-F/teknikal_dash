import streamlit as st
import requests
import json
import websocket
import threading
import pandas as pd
import plotly.graph_objects as go
import time

BASE_URL = "https://api.stockbit.com"

# -------------------------
# Login function
# -------------------------
def stockbit_login(username, password):
    url = f"{BASE_URL}/v2/auth/login"
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Content-Type": "application/json",
        "Accept": "application/json"
    }
    payload = {"username": username, "password": password}
    resp = requests.post(url, json=payload, headers=headers)

    try:
        data = resp.json()
    except Exception:
        raise Exception(f"Gagal decode JSON: {resp.text}")

    if not data.get("status", False):
        raise Exception(f"Login gagal: {data}")

    return data["data"]["token"]

# -------------------------
# WebSocket client
# -------------------------
class StockbitWS:
    def __init__(self, token, symbols):
        self.token = token
        self.symbols = symbols
        self.ws = None
        self.data = []

    def on_message(self, ws, message):
        msg = json.loads(message)
        if "data" in msg:
            self.data.append(msg["data"])

    def on_error(self, ws, error):
        print("Error:", error)

    def on_close(self, ws, close_status_code, close_msg):
        print("WS Closed")

    def on_open(self, ws):
        # subscribe ke ticker tertentu
        for sym in self.symbols:
            sub_msg = {
                "cmd": "subscribe",
                "channel": f"trading.{sym}"
            }
            ws.send(json.dumps(sub_msg))

    def run(self):
        ws_url = f"wss://ws.stockbit.com/"
        self.ws = websocket.WebSocketApp(
            ws_url,
            header=[f"Authorization: Bearer {self.token}"],
            on_message=self.on_message,
            on_error=self.on_error,
            on_close=self.on_close,
            on_open=self.on_open
        )
        thread = threading.Thread(target=self.ws.run_forever, daemon=True)
        thread.start()

# -------------------------
# Streamlit App
# -------------------------
st.set_page_config(page_title="Stockbit Realtime Analyzer", layout="wide")
st.title("📊 Stockbit Realtime Analyzer")

with st.sidebar:
    st.markdown("### 🔑 Login")
    user = st.text_input("Username / Email")
    pwd = st.text_input("Password", type="password")
    ticker = st.text_input("Ticker (contoh: BBCA)", value="BBCA").upper()
    run_btn = st.button("Start Realtime Feed")

if run_btn:
    if not user or not pwd:
        st.error("Masukkan username & password dulu!")
        st.stop()

    try:
        token = stockbit_login(user, pwd)
        st.success("✅ Login berhasil")
    except Exception as e:
        st.error(f"Login gagal: {e}")
        st.stop()

    # Start websocket
    ws_client = StockbitWS(token, [ticker])
    ws_client.run()

    st.info(f"Streaming realtime data untuk {ticker} ...")

    chart_placeholder = st.empty()
    table_placeholder = st.empty()

    price_data = []

    while True:
        if ws_client.data:
            new_tick = ws_client.data.pop()
            ts = pd.to_datetime(new_tick.get("timestamp", time.time()), unit="s")
            px = float(new_tick.get("last", 0))
            vol = new_tick.get("volume", 0)

            price_data.append({"time": ts, "price": px, "volume": vol})
            df = pd.DataFrame(price_data)

            # Plot
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df["time"], y=df["price"], mode="lines+markers", name="Price"))
            fig.update_layout(template="plotly_dark", height=500)
            chart_placeholder.plotly_chart(fig, use_container_width=True)

            table_placeholder.dataframe(df.tail(10))

        time.sleep(1)