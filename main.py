import os
import requests
import pandas as pd
import numpy as np
import asyncio
import json
import websockets
import csv
from datetime import datetime, timezone
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

print("=== GOLD BOT STARTING ===", flush=True)
print("Python process started", flush=True)

# === ENV ===
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
DERIV_TOKEN = os.getenv("DERIV_TOKEN")
TWELVE_API_KEY = os.getenv("TWELVE_API_KEY")

# === CONFIG ===
MAX_TRADES_PER_DAY = 5
TRADE_INTERVAL = 3600  # seconds
LOG_FILE = 'data/trade_log.csv'
MAX_STAKE_PERCENT = 0.2
MAX_STAKE = 1000
MIN_BALANCE = 10
STAKE_CAP = 4999
CONF_THRESHOLD = 0.6
ADX_THRESHOLD = 20
CONF_OVERRIDE = 0.75

DERIV_ENDPOINTS = [
    "wss://ws.deriv.com/websockets/v3?app_id=1089",
    "wss://ws.binaryws.com/websockets/v3?app_id=1089",
    "wss://ws.deriv.be/websockets/v3?app_id=1089"
]

# === UTILS ===
def send_telegram(msg):
    try:
        requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
            data={"chat_id": TELEGRAM_CHAT_ID, "text": msg},
            timeout=15
        )
    except Exception as e:
        print("Telegram error:", e, flush=True)

def fetch_data():
    url = (
        f"https://api.twelvedata.com/time_series?"
        f"symbol=XAU/USD&interval=1h&outputsize=60&apikey={TWELVE_API_KEY}"
    )
    try:
        data = requests.get(url, timeout=20).json()
        if 'values' not in data:
            print("Twelve Data response did not contain values:", data, flush=True)
            return None
        df = pd.DataFrame(data['values'])
        df['datetime'] = pd.to_datetime(df['datetime'])
        df['price'] = df['close'].astype(float)
        df = df.sort_values('datetime').reset_index(drop=True)
        return df[['datetime', 'price']]
    except Exception as e:
        print("Data fetch error:", e, flush=True)
        return None

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = -delta.where(delta < 0, 0).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def compute_ema(series, period=20):
    return series.ewm(span=period).mean()

def compute_adx(prices, window=14):
    high = prices.rolling(window).max()
    low = prices.rolling(window).min()
    tr = high - low
    atr = tr.rolling(window).mean()
    adx = atr.rolling(window).mean()
    return adx.bfill()

def prepare_features(df):
    df = df.copy()
    df['return'] = df['price'].pct_change()
    df['rsi'] = compute_rsi(df['price'])
    df['ema20'] = compute_ema(df['price'])
    df['delta'] = df['price'] - df['ema20']
    df['volatility'] = df['return'].rolling(10).std()
    df['adx'] = compute_adx(df['price'])
    df = df.dropna().copy()
    df['target'] = (df['price'].shift(-1) > df['price']).astype(int)
    return df

def train_model(df):
    features = ['rsi', 'delta', 'volatility', 'adx']
    X = df[features]
    y = df['target']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, _, y_train, _ = train_test_split(
        X_scaled, y, test_size=0.2, shuffle=False
    )
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model, scaler

def count_today_trades():
    if not os.path.exists(LOG_FILE):
        return 0
    df = pd.read_csv(LOG_FILE, header=None)
    df[0] = pd.to_datetime(df[0])
    return df[df[0].dt.date == datetime.now(timezone.utc).date()].shape[0]

def log_trade(direction, price, stake):
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    with open(LOG_FILE, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.now(timezone.utc).isoformat(),
            direction,
            price,
            stake
        ])

async def ws_connect():
    for endpoint in DERIV_ENDPOINTS:
        try:
            print(f"Connecting to Deriv: {endpoint}", flush=True)
            return await websockets.connect(endpoint, open_timeout=15)
        except Exception as e:
            print(f"Deriv connection failed: {e}", flush=True)
    raise ConnectionError("All Deriv endpoints failed")

async def get_balance():
    ws = await ws_connect()
    try:
        await ws.send(json.dumps({"authorize": DERIV_TOKEN}))
        await ws.recv()
        await ws.send(json.dumps({"balance": 1}))
        resp = await ws.recv()
        return float(json.loads(resp)['balance']['balance'])
    finally:
        await ws.close()

async def place_trade(direction, stake):
    ws = await ws_connect()
    try:
        await ws.send(json.dumps({"authorize": DERIV_TOKEN}))
        await ws.recv()

        payload = {
            "buy": 1,
            "price": stake,
            "parameters": {
                "amount": stake,
                "basis": "stake",
                "contract_type": "CALL" if direction == "buy" else "PUT",
                "currency": "USD",
                "duration": 4,
                "duration_unit": "h",
                "symbol": "frxXAUUSD"
            }
        }

        await ws.send(json.dumps(payload))
        resp = await ws.recv()
        return "error" not in json.loads(resp)
    finally:
        await ws.close()

async def trade_cycle():
    if count_today_trades() >= MAX_TRADES_PER_DAY:
        print("Daily trade limit reached.", flush=True)
        return

    df = fetch_data()
    if df is None:
        return

    df = prepare_features(df)
    model, scaler = train_model(df)

    latest = df[['rsi', 'delta', 'volatility', 'adx']].iloc[-1:]
    X_latest = scaler.transform(latest)
    proba = model.predict_proba(X_latest)[0]
    pred_class = model.predict(X_latest)[0]
    confidence = proba[pred_class]

    current_price = df['price'].iloc[-1]
    rsi = df['rsi'].iloc[-1]
    adx = df['adx'].iloc[-1]

    direction = "buy" if pred_class == 1 else "sell"

    print(
        f"Signal: {direction.upper()} | price={current_price:.2f} "
        f"| RSI={rsi:.2f} | ADX={adx:.2f} | confidence={confidence:.2f}",
        flush=True
    )

    if adx < ADX_THRESHOLD and confidence < CONF_OVERRIDE:
        send_telegram(
            f"⚠️ ADX {adx:.2f} low and confidence {confidence:.2f} weak. "
            f"Skipping trade."
        )
        return
    elif adx < ADX_THRESHOLD:
        send_telegram(
            f"🔎 ADX {adx:.2f} low but confidence {confidence:.2f} high — "
            f"proceeding with trade."
        )

    try:
        balance = await get_balance()

        if balance < MIN_BALANCE:
            send_telegram(
                f"❌ Balance ${balance:.2f} is below minimum ${MIN_BALANCE:.2f}."
            )
            return

        stake = round(
            min(balance * MAX_STAKE_PERCENT, MAX_STAKE, STAKE_CAP),
            2
        )

        success = await place_trade(direction, stake)

        if success:
            log_trade(direction, current_price, stake)
            send_telegram(
                f"✅ Trade executed: {direction.upper()} | "
                f"Price: {current_price:.2f} | Stake: ${stake:.2f}"
            )
        else:
            send_telegram(
                f"❌ Trade failed. Direction: {direction} | Stake: ${stake:.2f}"
            )

    except Exception as e:
        print(f"Trade error: {type(e).__name__}: {e}", flush=True)
        send_telegram(f"❌ Trade error: {str(e)}")

async def main_loop():
    print("Entering main loop...", flush=True)
    send_telegram("🚀 ML-ADX Gold Bot Started")

    while True:
        try:
            await trade_cycle()
        except Exception as e:
            print(
                f"Cycle error: {type(e).__name__}: {e}",
                flush=True
            )

        print(
            f"Sleeping for {TRADE_INTERVAL} seconds...",
            flush=True
        )
        await asyncio.sleep(TRADE_INTERVAL)

if __name__ == '__main__':
    print("Starting main loop...", flush=True)
    try:
        asyncio.run(main_loop())
    except Exception as e:
        print(
            f"FATAL ERROR: {type(e).__name__}: {e}",
            flush=True
        )
        raise
