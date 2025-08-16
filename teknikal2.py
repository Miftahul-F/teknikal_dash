import yfinance as yf
import pandas as pd
import ta

lq45_tickers = [
    "ADRO.JK", "AKRA.JK", "AMRT.JK", "ARTO.JK", "ASII.JK", "BBCA.JK", "BBNI.JK", 
    "BBRI.JK", "BBTN.JK", "BMRI.JK", "BRPT.JK", "BUKA.JK", "CPIN.JK", "EMTK.JK", 
    "ESSA.JK", "GGRM.JK", "GOTO.JK", "HRUM.JK", "ICBP.JK", "INCO.JK", "INDF.JK", 
    "INKP.JK", "INTP.JK", "ITMG.JK", "KLBF.JK", "MDKA.JK", "MEDC.JK", "MIKA.JK", 
    "MNCN.JK", "PGAS.JK", "PTBA.JK", "PTPP.JK", "SMGR.JK", "TBIG.JK", "TINS.JK", 
    "TKIM.JK", "TLKM.JK", "TOWR.JK", "UNTR.JK", "UNVR.JK", "WIKA.JK"
]

def compute_confidence(ticker):
    try:
        df_daily = yf.download(ticker, period="6mo", interval="1d", progress=False)
        df_weekly = yf.download(ticker, period="2y", interval="1wk", progress=False)
        
        if df_daily.empty or df_weekly.empty:
            print(f"[SKIP] {ticker} - data kosong")
            return None
        
        for col in ["Open", "High", "Low", "Close"]:
            df_daily[col] = pd.to_numeric(df_daily[col], errors="coerce").astype(float)
            df_weekly[col] = pd.to_numeric(df_weekly[col], errors="coerce").astype(float)
        
        confidence = 0
        
        ema20_w = df_weekly['Close'].ewm(span=20).mean().iloc[-1]
        ema50_w = df_weekly['Close'].ewm(span=50).mean().iloc[-1]
        if ema20_w > ema50_w:
            confidence += 20
        
        ema20_d = df_daily['Close'].ewm(span=20).mean().iloc[-1]
        ema50_d = df_daily['Close'].ewm(span=50).mean().iloc[-1]
        if ema20_d > ema50_d:
            confidence += 20
        
        rsi_d = ta.momentum.RSIIndicator(df_daily['Close'], window=14).rsi().iloc[-1]
        if 50 < rsi_d < 70:
            confidence += 20
        
        macd = ta.trend.MACD(df_daily['Close'])
        macd_val = macd.macd().iloc[-1]
        signal_val = macd.macd_signal().iloc[-1]
        if macd_val > signal_val:
            confidence += 20
        
        if df_daily['Close'].iloc[-1] > ema20_d:
            confidence += 20
        
        return {
            "Ticker": ticker,
            "Weekly Trend": "UP" if ema20_w > ema50_w else "DOWN",
            "Daily Trend": "UP" if ema20_d > ema50_d else "DOWN",
            "RSI": round(rsi_d, 2),
            "Confidence": confidence,
            "Last Price": df_daily['Close'].iloc[-1]
        }
    except Exception as e:
        print(f"[ERROR] {ticker} - {e}")
        return None

results = []
for t in lq45_tickers:
    data = compute_confidence(t)
    if data and data["Confidence"] >= 90:
        results.append(data)

if results:
    df_results = pd.DataFrame(results)
    print(df_results.sort_values("Confidence", ascending=False))
else:
    print("❌ Tidak ada saham LQ45 yang memenuhi Confidence ≥ 90% hari ini.")