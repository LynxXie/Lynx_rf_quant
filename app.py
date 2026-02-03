import streamlit as st
import pandas as pd
import numpy as np
import ccxt
import threading
import time
import os
import glob
import logging
import gc
from datetime import datetime
from collections import deque
import math
import socket
import traceback
import joblib
import sys
import urllib.parse
import hashlib
import hmac
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

# --- Configuration & Logging ---
st.set_page_config(page_title="Lynx RF Quant", layout="wide", page_icon="📊")

st.markdown("""
    <style>
        .block-container {padding-top: 1rem; padding-bottom: 1rem;}
        h1 {font-size: 1.5rem !important;}
        h2 {font-size: 1.25rem !important;}
        h3 {font-size: 1.1rem !important;}
        .stButton>button {height: 2.5em; font-size: 0.9rem;}
        div[data-testid="stMetricValue"] {font-size: 1.4rem !important;}
        div[data-testid="stMetricLabel"] {font-size: 0.9rem !important;}
    </style>
""", unsafe_allow_html=True)

LOG_FILE = "runtime.log"
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format='%(asctime)s %(message)s',
    filemode='a',
    force=True
)

def write_log(msg):
    timestamp = datetime.now().strftime("%H:%M:%S")
    formatted_msg = f"[{timestamp}] {msg}"
    print(formatted_msg, flush=True)
    logging.info(msg)
    return formatted_msg

try:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(("8.8.8.8", 80))
    MY_LOCAL_IP = s.getsockname()[0]
    s.close()
except:
    MY_LOCAL_IP = "127.0.0.1"

def set_global_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

set_global_seed(42)

# --- Trading Bot Core ---
class ReversionBotThread(threading.Thread):
    def __init__(self, api_key, secret, proxy_url, symbol='BTC/USDT', preloaded_df=None, 
                 mode='Paper', initial_capital=10000.0,
                 horizon=15, cost_multiplier=1.2, rf_depth=10, rf_estimators=100):
        
        super().__init__(daemon=False)
        self.api_key = api_key
        self.secret = secret
        self.proxy_url = proxy_url
        self.symbol = symbol
        self.mode = mode 
        
        # Hyperparameters
        self.p_horizon = int(horizon)
        self.p_cost_mult = float(cost_multiplier)
        self.p_rf_depth = int(rf_depth)
        self.p_rf_estimators = int(rf_estimators)
        
        # Paper Account
        self.paper_balance = initial_capital
        self.paper_position = 0.0 
        self.paper_entry_price = 0.0
        self.paper_upnl = 0.0
        
        self.cached_local_df = preloaded_df
        
        self.running = False
        self.stop_event = threading.Event()
        self.status = "STOPPED"
        self.logs = [] 
        
        self.ai_lock = threading.Lock()
        
        # Default Params
        self.target_leverage = 10
        self.qty_mode = "Auto"
        self.target_fixed_qty = 0.002
        self.long_threshold = 0.60
        self.short_threshold = 0.40
        self.auc_threshold = 0.54
        
        # Market State
        self.current_mark_price = 0.0 
        self.current_ticker_price = 0.0 
        self.last_valid_price = 0.0 
        
        self.current_rsi = 50.0
        self.current_adx = 0.0 
        self.current_atr = 0.0 
        
        self.market_mode = "Init / 初始化" 
        self.ai_raw_prob = 0.5 
        self.ai_avg_prob = 0.5 
        self.validation_acc = 0.0
        self.validation_auc = 0.0
        self.decision = "WAIT"
        
        self.use_ai = False 
        self.auc_pass_streak = 0
        self.model = None 
        self.last_train_time = 0
        self.last_trade_time = 0 
        self.force_train_flag = False 
        self.manual_cmd = None 
        
        self.last_feat_warn_time = 0
        self.flip_prob = False
        self.last_label_mode = "normal"
        
        # Risk Params
        self.leverage = 10        
        self.stop_loss_pct = 0.04  
        self.take_profit_pct = 0.08 
        self.data_window = 2880 
        self.est_taker_fee = 0.0005 
        self.est_slippage = 0.0002 
        self.round_trip_cost = 2 * (self.est_taker_fee + self.est_slippage) 
        
        self.protective_orders = [] 
        self.display_balance = 0.0
        self.display_upnl = 0.0
        self.display_position = 0.0
        
        self.history = {
            'time': deque(maxlen=100),
            'price': deque(maxlen=100),
            'rsi': deque(maxlen=100),
            'prob': deque(maxlen=100),
            'action': deque(maxlen=100)
        }
        self.prob_deque = deque(maxlen=3) 
        
        self.feature_names = [
            'Return', 'RSI', 'MACD_Norm', 'Vol_Change', 'BB_Width', 'Momentum',
            'ret_1', 'ret_5', 'ret_15', 'ret_60',
            'vol_5', 'vol_15', 'vol_60',
            'zscore_20', 'close_ma20_ratio',
            'rsi_delta', 'macd_delta', 'atr_pct'
        ]
        
        self.session = requests.Session()
        retries = Retry(total=2, backoff_factor=0.5, status_forcelist=[500, 502, 503, 504])
        self.session.mount('https://', HTTPAdapter(max_retries=retries))
        self._update_proxy_env(proxy_url)

        self.exchange = ccxt.binance({
            'apiKey': self.api_key,
            'secret': self.secret,
            'enableRateLimit': True,
            'timeout': 5000,
            'options': {'defaultType': 'future', 'adjustForTimeDifference': True},
            'proxies': {'http': self.proxy_url, 'https': self.proxy_url} if self.proxy_url else None
        })
        
        if self.mode == 'Real':
            try: self.exchange.load_markets()
            except: pass
        
        gc.collect()
        write_log(f"Init | H:{self.p_horizon}m | Cost:x{self.p_cost_mult} | Depth:{self.p_rf_depth}")

    def _update_proxy_env(self, proxy_url):
        if proxy_url:
            self.session.proxies = {'http': proxy_url, 'https': proxy_url}
            os.environ["HTTP_PROXY"] = proxy_url
            os.environ["HTTPS_PROXY"] = proxy_url
        else:
            self.session.proxies = {}
            os.environ.pop("HTTP_PROXY", None)
            os.environ.pop("HTTPS_PROXY", None)
        os.environ["NO_PROXY"] = f"localhost,127.0.0.1,::1,{MY_LOCAL_IP}"
        os.environ["no_proxy"] = f"localhost,127.0.0.1,::1,{MY_LOCAL_IP}"

    def log(self, msg):
        log_line = write_log(msg)
        if not isinstance(self.logs, list): self.logs = []
        self.logs.insert(0, log_line)
        self.logs = self.logs[:50]

    @staticmethod
    def normalize_dataframe_static(df_in):
        df = df_in
        if 'Time' in df.columns:
            t0 = float(df['Time'].iloc[0])
            unit = 'us' if t0 > 1e14 else 'ms' 
            df['Date'] = pd.to_datetime(df['Time'], unit=unit)
        elif 'Open Time' in df.columns:
            t0 = float(df['Open Time'].iloc[0])
            unit = 'us' if t0 > 1e14 else 'ms'
            df['Date'] = pd.to_datetime(df['Open Time'], unit=unit)
        
        if 'Date' in df.columns: df = df.set_index('Date')
        if not isinstance(df.index, pd.DatetimeIndex): return df
            
        df = df.sort_index()
        df = df[~df.index.duplicated(keep='last')]
        
        for col in df.select_dtypes(include=['float64']).columns:
            df[col] = df[col].astype('float32')
            
        return df

    def normalize_dataframe(self, df_in):
        return self.normalize_dataframe_static(df_in)

    def calculate_indicators(self, df_in):
        df = df_in.copy()
        if len(df) < 80: return df 
        
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        df['RSI'] = 100 - (100 / (1 + gain/(loss+1e-8)))
        
        df['TR'] = np.maximum(df['High'] - df['Low'], np.maximum(abs(df['High'] - df['Close'].shift(1)), abs(df['Low'] - df['Close'].shift(1))))
        df['ATR'] = df['TR'].rolling(14).mean()
        
        df['UpMove'] = df['High'] - df['High'].shift(1)
        df['DownMove'] = df['Low'].shift(1) - df['Low']
        df['PDM'] = np.where((df['UpMove'] > df['DownMove']) & (df['UpMove'] > 0), df['UpMove'], 0)
        df['MDM'] = np.where((df['DownMove'] > df['UpMove']) & (df['DownMove'] > 0), df['DownMove'], 0)
        df['PDI'] = 100 * (df['PDM'].rolling(14).mean() / (df['ATR']+1e-8))
        df['MDI'] = 100 * (df['MDM'].rolling(14).mean() / (df['ATR']+1e-8))
        df['DX'] = 100 * abs(df['PDI'] - df['MDI']) / (df['PDI'] + df['MDI'] + 1e-8)
        df['ADX'] = df['DX'].rolling(14).mean()
        
        ema12 = df['Close'].ewm(span=12).mean()
        ema26 = df['Close'].ewm(span=26).mean()
        df['MACD_Norm'] = (ema12 - ema26) / df['Close'] * 100 
        
        df['BB_Mid'] = df['Close'].rolling(20).mean()
        df['BB_Std'] = df['Close'].rolling(20).std()
        df['BB_Up'] = df['BB_Mid'] + 2 * df['BB_Std']
        df['BB_Low'] = df['BB_Mid'] - 2 * df['BB_Std']
        df['BB_Width'] = (df['BB_Up'] - df['BB_Low']) / (df['BB_Mid'] + 1e-8)
        
        for w in [1, 5, 15, 60]: df[f'ret_{w}'] = df['Close'].pct_change(w)
        for w in [5, 15, 60]: df[f'vol_{w}'] = df['Close'].pct_change().rolling(w).std()

        df['zscore_20'] = (df['Close'] - df['BB_Mid']) / (df['BB_Std'] + 1e-8)
        df['close_ma20_ratio'] = df['Close'] / (df['BB_Mid'] + 1e-8) - 1
        df['rsi_delta'] = df['RSI'].diff()
        df['macd_delta'] = df['MACD_Norm'].diff()
        df['atr_pct'] = df['ATR'] / df['Close']

        df['Momentum'] = df['Close'] - df['Close'].shift(4)
        df['Return'] = df['Close'].pct_change()
        df['Vol_Change'] = df['Vol'].pct_change() 
        
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        return df

    def get_features(self, df):
        missing = [c for c in self.feature_names if c not in df.columns]
        if missing:
            now = time.time()
            if now - self.last_feat_warn_time > 30: 
                self.log(f"⚠️ Missing Features: {missing[:3]}...")
                self.last_feat_warn_time = now
            return None
        return df[self.feature_names]

    def build_dataset(self, df):
        df = self.calculate_indicators(df)
        
        horizon = self.p_horizon
        df['fwd_ret'] = df['Close'].shift(-horizon) / df['Close'] - 1
        
        atr_pct = df['atr_pct'].fillna(0.001)
        
        base_cost = self.round_trip_cost * self.p_cost_mult
        dynamic_thr = np.maximum(base_cost, 0.35 * atr_pct)
        
        df['Label'] = np.where(df['fwd_ret'] > dynamic_thr, 1,
                       np.where(df['fwd_ret'] < -dynamic_thr, 0, np.nan))
        
        feat_df = self.get_features(df)
        if feat_df is None: return None, None
        
        df_clean = pd.concat([feat_df, df['Label']], axis=1).dropna()
        
        if len(df_clean) < 1200:
            self.log(f"📉 Sparse Data ({len(df_clean)}) -> Observational Mode")
            self.last_label_mode = "sparse"
        else:
            self.last_label_mode = "normal"

        if len(df_clean) < 50: return None, None
        
        X = df_clean.drop(columns=['Label']).values
        y = df_clean['Label'].astype(int).values
        return X, y

    def train_mixed_model(self, df_recent):
        try:
            self.log(f"🧠 Training RF (H:{self.p_horizon}m Cost:x{self.p_cost_mult})...")
            
            df_recent = self.normalize_dataframe(df_recent)
            df_hist = self.cached_local_df
            
            if df_hist is not None:
                df_combined = pd.concat([df_hist, df_recent]).sort_index()
                df_combined = df_combined[~df_combined.index.duplicated(keep='last')]
                if len(df_combined) > 50000: df_combined = df_combined.tail(50000)
            else:
                df_combined = df_recent

            X, y = self.build_dataset(df_combined)
            if X is None: 
                self.log("⚠️ Insufficient samples")
                return

            split_idx = int(len(X) * 0.8)
            X_train, y_train = X[:split_idx], y[:split_idx]
            X_valid, y_valid = X[split_idx:], y[split_idx:]

            pos_rate = np.mean(y_train) if len(y_train) > 0 else 0
            self.log(f"📌 Pos Rate: {pos_rate*100:.1f}% | Train Size: {len(X_train)}")

            model = RandomForestClassifier(
                n_estimators=self.p_rf_estimators, 
                max_depth=self.p_rf_depth, 
                min_samples_split=30, 
                n_jobs=1, 
                class_weight='balanced', 
                random_state=42
            )
            model.fit(X_train, y_train)

            y_pred = model.predict(X_valid)
            try:
                y_prob = model.predict_proba(X_valid)[:, 1]
                auc = roc_auc_score(y_valid, y_prob)
            except: auc = 0.5
            acc = accuracy_score(y_valid, y_pred)

            auc_flip = roc_auc_score(y_valid, 1 - y_prob)
            is_flipped = False
            if auc_flip > auc:
                auc = auc_flip
                is_flipped = True
                self.log(f"🔄 Inverse Correlation Detected -> AUC {auc:.2f}")

            self.log(f"📊 Validation: AUC={auc:.2f} Acc={acc:.2f}")

            with self.ai_lock:
                self.model = model
                self.flip_prob = is_flipped
                self.validation_acc = acc
                self.validation_auc = auc
                
                if self.last_label_mode != "normal":
                    self.use_ai = False
                    self.log(f"🛡️ Sparse Mode -> Trading Disabled")
                elif auc > self.auc_threshold:
                    self.auc_pass_streak += 1
                    if self.auc_pass_streak >= 1:
                        self.use_ai = True
                        self.log(f"✅ AI Activated (AUC > {self.auc_threshold})")
                else:
                    self.use_ai = False
                    self.auc_pass_streak = 0
                    self.log(f"🛡️ Low AUC -> Safe Mode")
                
                self.last_train_time = time.time()
            
            gc.collect()

        except Exception as e:
            self.log(f"⚠️ Train Error: {str(e)[:50]}")
            print(traceback.format_exc())

    def predict_live(self, df_recent):
        with self.ai_lock:
            model = self.model
            do_flip = self.flip_prob
        if model is None: return 0.5
        
        try:
            df_recent = self.normalize_dataframe(df_recent)
            df = self.calculate_indicators(df_recent)
            feat_df = self.get_features(df)
            
            if feat_df is None: return 0.5
            feat_valid = feat_df.dropna()
            
            if len(feat_valid) == 0: return 0.5
            last_feat = feat_valid.iloc[-1].values.reshape(1, -1)
            
            prob = model.predict_proba(last_feat)[0][1]
            if do_flip: prob = 1.0 - prob
            return prob
        except:
            return 0.5

    def trigger_force_train(self):
        self.force_train_flag = True

    def load_local_data(self):
        if self.cached_local_df is not None: 
            write_log(f"📦 Inherited Cached Data: {len(self.cached_local_df)} rows")
            return
        pass

    def fetch_price_fast(self):
        price = 0.0
        try:
            url = "https://fapi.binance.com/fapi/v1/ticker/price?symbol=BTCUSDT"
            r = requests.get(url, timeout=2, proxies=self.session.proxies)
            if r.status_code == 200: price = float(r.json()['price'])
            elif r.status_code == 429: time.sleep(60) 
        except: pass
        
        if price == 0 and self.mode == 'Real':
            try: ticker = self.exchange.fetch_ticker(self.symbol); price = ticker['last']
            except: pass
        if price == 0 and self.mode == 'Real':
            try: ohlcv = self.exchange.fetch_ohlcv(self.symbol, limit=1); price = ohlcv[0][4] if ohlcv else 0
            except: pass
        
        if price > 0:
            self.last_valid_price = price
            return price
        elif self.last_valid_price > 0: return self.last_valid_price
        else: return 0.0

    def fetch_mark_price_fast(self):
        try:
            url = "https://fapi.binance.com/fapi/v1/premiumIndex?symbol=BTCUSDT"
            r = self.session.get(url, timeout=2)
            return float(r.json()['markPrice'])
        except: return self.last_valid_price

    def update_account_status(self):
        if self.mode == 'Real':
            try:
                acc = self._signed_fapi_get("v2", "account")
            except: 
                try: acc = self._signed_fapi_get("v1", "account")
                except: return
            
            self.wallet_balance = float(acc.get('totalWalletBalance', 0))
            self.unrealized_pnl = float(acc.get('totalUnrealizedProfit', 0))
            
            raw_positions = acc.get('positions', [])
            positions = [p for p in raw_positions if p['symbol'] == self.symbol.replace('/','')]
            if positions: 
                self.position_amt = float(positions[0].get('positionAmt', 0))
                self.entry_price = float(positions[0].get('entryPrice', 0))
            else:
                self.position_amt = 0.0
                self.entry_price = 0.0
                
            self.display_balance = self.wallet_balance
            self.display_upnl = self.unrealized_pnl
            self.display_position = self.position_amt
            
        else:
            price = self.last_valid_price if self.last_valid_price > 0 else 0
            if price > 0 and self.paper_position != 0:
                if self.paper_position > 0:
                    self.paper_upnl = (price - self.paper_entry_price) * self.paper_position
                else:
                    self.paper_upnl = (self.paper_entry_price - price) * abs(self.paper_position)
            else:
                self.paper_upnl = 0.0
                
            self.display_balance = self.paper_balance
            self.display_upnl = self.paper_upnl
            self.display_position = self.paper_position

    def _signed_fapi_get(self, version: str, endpoint: str, params=None):
        base = "https://fapi.binance.com"
        params = params or {}
        params["timestamp"] = int((time.time() - 1) * 1000) 
        params.setdefault("recvWindow", 5000)
        query = urllib.parse.urlencode(params, doseq=True)
        sig = hmac.new(self.secret.encode(), query.encode(), hashlib.sha256).hexdigest()
        url = f"{base}/fapi/{version}/{endpoint}?{query}&signature={sig}"
        headers = {"X-MBX-APIKEY": self.api_key}
        r = self.session.get(url, headers=headers, timeout=3)
        r.raise_for_status()
        return r.json()

    def switch_to_oneway_mode(self):
        try: self.exchange.fapiPrivatePostPositionSideDual({'dualSidePosition': 'false'})
        except: pass

    def check_and_set_leverage(self):
        if self.mode == 'Paper': return
        if self.leverage != self.target_leverage:
            try:
                self.exchange.fapiPrivatePostLeverage({'symbol': self.symbol.replace('/', ''), 'leverage': self.target_leverage})
                self.leverage = self.target_leverage
                self.log(f"🔧 Leverage Set: {self.leverage}x")
            except Exception as e: self.log(f"⚠️ Set Leverage Failed: {e}")

    def get_dynamic_qty(self):
        try:
            if self.qty_mode == "Fixed": return self.target_fixed_qty
            else:
                price = self.last_valid_price if self.last_valid_price > 0 else 96000
                balance = self.display_balance
                max_usd = balance * self.leverage * 0.2 
                raw_qty = max_usd / price
                qty = math.floor(raw_qty * 1000) / 1000.0
                
                min_qty = 0.001
                if self.mode == 'Real':
                    try:
                        market = self.exchange.market(self.symbol)
                        min_qty = market['limits']['amount']['min']
                    except: pass
                
                if qty < min_qty: 
                    self.log(f"❌ Insufficient Funds ({qty} < {min_qty})")
                    return 0.0
                return qty
        except Exception as e:
            self.log(f"⚠️ Qty Calc Error: {e}")
            return 0.001

    def place_protective_orders(self, side, entry_price, qty):
        if self.mode == 'Paper': return
        try:
            self.cancel_protective_orders()
            sl_price = float(self.exchange.price_to_precision(self.symbol, entry_price * (1 - self.stop_loss_pct / self.leverage) if side == 'buy' else entry_price * (1 + self.stop_loss_pct / self.leverage)))
            tp_price = float(self.exchange.price_to_precision(self.symbol, entry_price * (1 + self.take_profit_pct / self.leverage) if side == 'buy' else entry_price * (1 - self.take_profit_pct / self.leverage)))
            
            exit_side = 'sell' if side == 'buy' else 'buy'
            params_sl = {'stopPrice': sl_price, 'reduceOnly': True, 'workingType': 'MARK_PRICE'}
            self.exchange.create_order(self.symbol, 'STOP_MARKET', exit_side, qty, params=params_sl)
            
            params_tp = {'stopPrice': tp_price, 'reduceOnly': True, 'workingType': 'MARK_PRICE'}
            self.exchange.create_order(self.symbol, 'TAKE_PROFIT_MARKET', exit_side, qty, params=params_tp)
            self.log(f"🛡️ Orders: SL@{sl_price} TP@{tp_price}")
        except Exception as e: self.log(f"⚠️ Order Failed: {e}")

    def cancel_protective_orders(self):
        if self.mode == 'Paper': return
        try: self.exchange.cancel_all_orders(self.symbol)
        except: pass

    def router_create_order(self, side, qty):
        if self.mode == 'Paper':
            price = self.last_valid_price
            cost = price * qty
            fee = cost * 0.0005 
            
            if side == 'buy':
                new_size = self.paper_position + qty
                if new_size > 0:
                    old_cost = self.paper_position * self.paper_entry_price
                    new_cost_add = qty * price
                    self.paper_entry_price = (old_cost + new_cost_add) / new_size
                self.paper_position += qty
                self.paper_balance -= fee
                self.log(f"📝 Paper BUY: {qty} BTC @ {price}")
                
            elif side == 'sell':
                new_size = self.paper_position - qty
                if new_size < 0:
                    old_cost = abs(self.paper_position) * self.paper_entry_price
                    new_cost_add = qty * price
                    self.paper_entry_price = (old_cost + new_cost_add) / abs(new_size) if abs(new_size)>0 else price
                self.paper_position -= qty
                self.paper_balance -= fee
                self.log(f"📝 Paper SELL: {qty} BTC @ {price}")
                
            return {'average': price}
        else:
            return self.exchange.create_market_order(self.symbol, side, qty)

    def close_position(self, reason="Signal"):
        try:
            self.cancel_protective_orders()
            amt = self.display_position
            
            if amt != 0:
                side = 'sell' if amt > 0 else 'buy'
                
                if self.mode == 'Paper':
                    self.router_create_order(side, abs(amt))
                    self.paper_balance += self.paper_upnl
                    self.paper_upnl = 0.0
                    self.paper_position = 0.0
                    self.paper_entry_price = 0.0
                else:
                    self.exchange.create_market_order(self.symbol, side, abs(amt), params={'reduceOnly': True})
                
                self.log(f"⚡ Close ({reason})")
                self.last_trade_time = time.time() + 60 
                self.log("🧊 Cooldown 60s...")
        except Exception as e: self.log(f"❌ Close Failed: {e}")

    def execute_manual_order(self, side):
        try:
            self.cancel_protective_orders()
            qty = self.get_dynamic_qty()
            if qty <= 0: return

            amt = self.display_position
            if (side == 'buy' and amt < 0) or (side == 'sell' and amt > 0):
                 self.close_position("Manual Flip")
                 time.sleep(1)
            
            if self.display_position == 0:
                self.log(f"🔵 Manual {side}: {qty} BTC")
                order = self.router_create_order(side, qty)
                avg_price = float(order['average']) if order['average'] else self.last_valid_price
                self.place_protective_orders(side, avg_price, qty)
            else: self.log("⚠️ Position exists")
        except Exception as e: self.log(f"❌ Manual Failed: {e}")

    def check_risk_management(self, current_price):
        bal = self.display_balance
        pnl = self.display_upnl
        if bal <= 0: return
        
        if pnl / bal < -0.15:
            self.log("🛑 Hard Stop Loss (-15%) Triggered!")
            self.close_position("Hard Stop")
            self.stop()

    def execute_logic(self, decision, avg_prob):
        if self.last_valid_price <= 0:
            self.log("🛑 Price Lost")
            return

        if self.manual_cmd:
            if self.manual_cmd == 'LONG': self.execute_manual_order('buy')
            elif self.manual_cmd == 'SHORT': self.execute_manual_order('sell')
            elif self.manual_cmd == 'CLOSE': self.close_position("Manual Close")
            self.manual_cmd = None 
            return

        if time.time() < self.last_trade_time: return 

        self.check_and_set_leverage()
        qty = self.get_dynamic_qty()
        if qty <= 0: return 
        
        amt = self.display_position
        
        if amt > 0 and avg_prob < self.short_threshold: 
            self.log(f"🔄 Smart Stop (Avg {avg_prob:.2f} < {self.short_threshold})")
            self.close_position("Smart Stop Long")
            return
        if amt < 0 and avg_prob > self.long_threshold: 
            self.log(f"🔄 Smart Stop (Avg {avg_prob:.2f} > {self.long_threshold})")
            self.close_position("Smart Stop Short")
            return

        atr_pct = 0.0
        safe_price = self.last_valid_price if self.last_valid_price > 0 else 96000
        if safe_price > 0 and self.current_atr > 0: atr_pct = self.current_atr / safe_price
        
        cost_threshold = self.round_trip_cost * 3
        is_volatile_enough = atr_pct >= cost_threshold

        if amt == 0:
            try:
                if self.use_ai:
                    if not is_volatile_enough and "WAIT" not in decision:
                        if self.short_threshold < avg_prob < self.long_threshold: return

                    if "LONG" in decision:
                        self.log(f"🚀 AI OPEN LONG: {qty} BTC")
                        order = self.router_create_order('buy', qty)
                        avg_price = float(order['average']) if order.get('average') else self.last_valid_price
                        self.place_protective_orders('buy', avg_price, qty)
                        
                    elif "SHORT" in decision:
                        self.log(f"🚀 AI OPEN SHORT: {qty} BTC")
                        order = self.router_create_order('sell', qty)
                        avg_price = float(order['average']) if order.get('average') else self.last_valid_price
                        self.place_protective_orders('sell', avg_price, qty)
            except Exception as e: self.log(f"❌ Entry Failed: {e}")

    def run(self):
        self.running = True
        self.status = "INIT..."
        self.log(f"🔗 V30.41 Running ({self.mode})...")
        print("THREAD STARTING")
        self.log(f"📦 Preloaded Data: {len(self.cached_local_df) if self.cached_local_df is not None else 0} rows")
        
        try:
            if self.mode == 'Real':
                try:
                    acc = self.fetch_account_native()
                    self.log("✅ API Connected")
                    self.switch_to_oneway_mode()
                    self.check_and_set_leverage()
                except Exception as e:
                    self.log(f"❌ API Error: {str(e)[:50]}")
                    self.status = "NET ERROR"
                    return

            while self.running and not self.stop_event.is_set():
                try:
                    self.log(f"📥 Fetching Init Data...")
                    bars_init = self.exchange.fetch_ohlcv(self.symbol, timeframe='1m', limit=1500)
                    if bars_init:
                        df_init = pd.DataFrame(bars_init, columns=['Time','Open','High','Low','Close','Vol'])
                        self.train_mixed_model(df_init)
                        break 
                    time.sleep(2)
                except Exception as e:
                    self.log(f"Wait: {e}")
                    time.sleep(2)

            self.status = f"TRADING ({self.mode})"
            last_slow_check = 0
            
            while self.running and not self.stop_event.is_set():
                now = time.time()
                try:
                    self.current_mark_price = self.fetch_mark_price_fast()
                    self.current_ticker_price = self.fetch_price_fast()
                    if self.current_mark_price > 0: self.check_risk_management(self.current_mark_price)
                except: pass

                if now - last_slow_check > 1.5:
                    try:
                        last_slow_check = now
                        self.update_account_status()

                        should_train = False
                        if self.force_train_flag:
                            self.log("💪 Force Retrain...")
                            should_train = True
                            self.force_train_flag = False
                        elif time.time() - self.last_train_time > 1800:
                            should_train = True

                        limit = 1500 if should_train else 100
                        bars_fresh = self.exchange.fetch_ohlcv(self.symbol, timeframe='1m', limit=limit)
                        
                        if bars_fresh:
                            df_real = pd.DataFrame(bars_fresh, columns=['Time','Open','High','Low','Close','Vol'])
                            
                            if should_train: self.train_mixed_model(df_real)
                            
                            prob = self.predict_live(df_real)
                            self.prob_deque.append(prob)
                            if len(self.prob_deque) > 0:
                                self.ai_avg_prob = sum(self.prob_deque) / len(self.prob_deque)
                            self.ai_raw_prob = float(prob)
                            
                            df_ind = self.calculate_indicators(df_real.copy())
                            if len(df_ind) > 0:
                                self.current_rsi = float(df_ind['RSI'].iloc[-1])
                                self.current_adx = float(df_ind['ADX'].iloc[-1])
                                self.current_atr = float(df_ind['ATR'].iloc[-1])
                                ma20 = df_real['Close'].rolling(20).mean().iloc[-1]
                                
                                safe_price = self.current_ticker_price if self.current_ticker_price > 0 else self.last_valid_price
                                is_uptrend = safe_price > ma20
                                is_downtrend = safe_price < ma20 
                                
                                new_decision = "WAIT"
                                if self.current_adx > 25:
                                    self.market_mode = f"🌊 Trend ({self.current_adx:.0f})"
                                else:
                                    self.market_mode = f"🦀 Chop ({self.current_adx:.0f})"

                                if self.use_ai:
                                    if self.ai_avg_prob > self.long_threshold and is_uptrend: new_decision = "LONG"
                                    elif self.ai_avg_prob < self.short_threshold and is_downtrend: new_decision = "SHORT"
                                
                                self.decision = new_decision
                                self.history['time'].append(datetime.now().strftime("%H:%M:%S"))
                                self.history['price'].append(safe_price)
                                self.history['rsi'].append(self.current_rsi)
                                self.history['prob'].append(prob)
                                self.history['action'].append(np.nan)
                                
                                self.execute_logic(new_decision, self.ai_avg_prob)
                        
                    except Exception as e: self.log(f"⚠️ Loop Error: {str(e)[:30]}")

                time.sleep(0.2)
                    
        except Exception as e:
            self.status = "CRASHED"
            self.log(f"💥 Fatal: {e}")
            print(traceback.format_exc())

    def stop(self):
        self.stop_event.set()
        self.running = False

    def fetch_trade_history(self):
        try:
            trades = self.exchange.fetch_my_trades(self.symbol, limit=10)
            return trades
        except Exception as e: return str(e)

# --- UI Layout ---
if 'rev_bot' not in st.session_state: st.session_state.rev_bot = None
if 'shared_df' not in st.session_state: st.session_state.shared_df = None

with st.sidebar:
    st.title("🦁 Lynx RF Quant")
    st.caption("V30.41: Clean Release")
    
    st.markdown("### 🎛️ Control Panel / 控制面板")
    
    mode_sel = st.radio("Mode / 模式", ["Paper Trading / 模拟训练", "Live Trading / 实盘交易"])
    sim_capital = 10000.0
    if "Paper" in mode_sel:
        sim_capital = st.number_input("Paper Capital / 模拟本金 (USDT)", value=10000.0, step=1000.0)
    
    uploaded_files = st.file_uploader("📂 Load CSV / 加载数据", type=['csv'], accept_multiple_files=True)
    if uploaded_files:
        if st.button("📥 1. Process Data / 读取数据"):
            with st.spinner("Processing... / 处理中..."):
                try:
                    st.session_state.shared_df = None
                    gc.collect()
                    df_list = []
                    for ufile in uploaded_files:
                        ufile.seek(0)
                        df_peek = pd.read_csv(ufile, encoding='utf-8', on_bad_lines='skip', nrows=5)
                        has_header = 'Time' in df_peek.columns or 'Open Time' in df_peek.columns
                        ufile.seek(0)
                        if has_header:
                            df_t = pd.read_csv(ufile, encoding='utf-8', on_bad_lines='skip')
                            if 'Open Time' in df_t.columns: df_t.rename(columns={'Open Time':'Time', 'Volume':'Vol'}, inplace=True)
                        else:
                            df_t = pd.read_csv(ufile, header=None, names=['Time','Open','High','Low','Close','Vol','CT','QAV','NT','TBBAV','TBQAV','I'], encoding='utf-8', on_bad_lines='skip')
                        if 'Time' in df_t.columns:
                            df_t['Time'] = pd.to_numeric(df_t['Time'], errors='coerce')
                            df_t.dropna(subset=['Time'], inplace=True)
                            req_cols = ['Time','Open','High','Low','Close','Vol']
                            df_t = df_t[[c for c in req_cols if c in df_t.columns]]
                            for col in df_t.select_dtypes(include=['float64']).columns:
                                df_t[col] = df_t[col].astype('float32')
                            df_list.append(df_t)
                        del df_t
                        gc.collect()
                    if df_list:
                        raw_df = pd.concat(df_list).sort_values('Time')
                        final_df = ReversionBotThread.normalize_dataframe_static(raw_df).resample('1min').agg({'Open':'first','High':'max','Low':'min','Close':'last','Vol':'sum'}).dropna()
                        for col in final_df.select_dtypes(include=['float64']).columns:
                            final_df[col] = final_df[col].astype('float32')
                        st.session_state.shared_df = final_df
                        st.success(f"✅ Loaded {len(final_df)} rows")
                        del raw_df
                        del df_list
                        gc.collect()
                    else:
                        st.error("No valid data / 无效数据")
                except Exception as e: st.error(f"Error: {e}")

    if st.button("🗑️ Clear Cache / 清空缓存"):
        st.session_state.shared_df = None
        gc.collect()
        st.success("Cleared")
        st.rerun()

    if st.session_state.shared_df is not None:
        st.info(f"📚 Cached: {len(st.session_state.shared_df)} rows")

    st.write("---")
    st.markdown("### 🧬 Hyperparameters / 超参数")
    col_p1, col_p2 = st.columns(2)
    p_horizon = col_p1.selectbox("Horizon / 预测视野", [5, 15, 30, 60], index=1)
    p_cost_mult = col_p2.slider("Cost Mult / 成本系数", 0.5, 3.0, 1.2, 0.1)
    
    col_p3, col_p4 = st.columns(2)
    p_depth = col_p3.slider("Max Depth / 脑容量", 4, 20, 10)
    p_estimators = col_p4.selectbox("Trees / 决策树", [50, 100, 200], index=1)

    st.write("---")
    st.markdown("### 🚦 Thresholds / 交易阈值")
    long_th = st.slider("Long Th / 做多阈值", 0.5, 0.9, 0.60, 0.01)
    short_th = st.slider("Short Th / 做空阈值", 0.1, 0.5, 0.40, 0.01)
    auc_th = st.slider("Min AUC / 准入AUC", 0.50, 0.60, 0.54, 0.01)
    new_leverage = st.slider("Leverage / 杠杆", 1, 125, 10)
    
    mode_options = ["Dynamic / 动态仓位", "Fixed / 固定数量"]
    mode_sel_qty = st.radio("Pos Mode / 仓位模式", mode_options)
    fixed_qty_val = 0.002
    if "Fixed" in mode_sel_qty:
        fixed_qty_val = st.number_input("Fix Qty / 固定BTC", min_value=0.001, max_value=1.0, value=0.002, step=0.001, format="%.3f")

    if st.session_state.rev_bot:
        st.session_state.rev_bot.target_leverage = new_leverage
        st.session_state.rev_bot.qty_mode = "Fixed" if "Fixed" in mode_sel_qty else "Auto"
        st.session_state.rev_bot.target_fixed_qty = fixed_qty_val
        st.session_state.rev_bot.long_threshold = long_th
        st.session_state.rev_bot.short_threshold = short_th
        st.session_state.rev_bot.auc_threshold = auc_th

    st.write("---")
    api_key_in = st.text_input("API Key", type="password")
    secret_in = st.text_input("API Secret", type="password")
    proxy_in = st.text_input("Proxy / 代理 (http://...)", value="")
    
    st.write("---")
    c1, c2 = st.columns(2)
    if c1.button("🚀 2. Start / 启动"):
        if st.session_state.rev_bot is not None and st.session_state.rev_bot.is_alive():
            st.warning("Running / 已在运行")
        elif "Live" in mode_sel and not api_key_in: 
            st.error("Need API Key for Live")
        else:
            df_to_pass = st.session_state.shared_df
            real_mode = 'Real' if "Live" in mode_sel else 'Paper'
            st.session_state.rev_bot = ReversionBotThread(api_key_in, secret_in, proxy_in, 
                                                          symbol='BTC/USDT',
                                                          preloaded_df=df_to_pass,
                                                          mode=real_mode,
                                                          initial_capital=sim_capital,
                                                          horizon=p_horizon,
                                                          cost_multiplier=p_cost_mult,
                                                          rf_depth=p_depth,
                                                          rf_estimators=p_estimators)
            st.session_state.rev_bot.start()
            time.sleep(1.0)
            st.success(f"Started ({real_mode})")
            st.rerun()

    if c2.button("🔴 Stop / 停止"):
        if st.session_state.rev_bot:
            st.session_state.rev_bot.stop()
            st.session_state.rev_bot.join(timeout=2)
            st.session_state.rev_bot = None
            st.success("Stopped")
            st.rerun()

    if st.button("🔄 Refresh / 刷新"):
        st.rerun()

from streamlit_autorefresh import st_autorefresh
st_autorefresh(interval=5000, key="fizzbuzz")

tab1, tab2, tab3 = st.tabs(["📊 Dashboard / 账户概览", "🤖 RF Monitor / 策略监控", "📜 History / 历史记录"])

with tab1:
    if st.session_state.rev_bot:
        bot = st.session_state.rev_bot
        title_prefix = "🏦 Real Asset" if bot.mode == 'Real' else "🎮 Paper Asset"
        st.markdown(f"### {title_prefix}")
        
        c1, c2, c3 = st.columns(3)
        total_equity = bot.display_balance + bot.display_upnl
        c1.metric("Equity / 总权益", f"${total_equity:,.2f}")
        c2.metric("uPNL / 浮盈", f"${bot.display_upnl:+.2f}", delta_color="normal" if bot.display_upnl >=0 else "inverse")
        c3.metric("Position / 持仓", f"{bot.display_position} BTC")
    else:
        st.info("System Offline / 系统未启动")

with tab2:
    st.header("RF Strategy Monitor / 随机森林策略监控")
    
    if st.session_state.rev_bot:
        is_alive = st.session_state.rev_bot.is_alive()
        status = "🟢 ALIVE" if is_alive else "🔴 DEAD"
        st.markdown(f"**Thread Status:** {status}")
        
        if is_alive:
            bot = st.session_state.rev_bot
            if bot.use_ai:
                status_color = "🟢"
                status_text = f"AI Active ({bot.p_horizon}m)"
                flip_text = " (Flipped)" if bot.flip_prob else ""
                extra_text = f"Val AUC: {bot.validation_auc:.2f}{flip_text} | Acc: {bot.validation_acc:.2f}"
            else:
                status_color = "🛡️"
                status_text = "Safe Mode"
                extra_text = f"AUC: {bot.validation_auc:.2f} < {bot.auc_threshold}"

            st.info(f"🧭 Mode: **{bot.market_mode}** | {status_color} {status_text} | {extra_text}")
            
            c1, c2, c3 = st.columns(3)
            if bot.use_ai:
                c1.metric("Model Age", f"🟢 {int((time.time() - bot.last_train_time)/60)}m ago")
            else:
                c1.metric("Model Age", "🟡 Obs.")
            c2.metric("Confidence", f"{bot.ai_avg_prob*100:.1f}%")
            
            pos_text = "Flat"
            if bot.display_position > 0: pos_text = "Long"
            elif bot.display_position < 0: pos_text = "Short"
            c3.metric("Direction", f"{pos_text}")
            
            try:
                with open(LOG_FILE, "r") as f:
                    log_content = f.readlines()[-20:]
                st.text_area("Log (runtime.log)", "".join(log_content), height=200)
            except:
                st.text_area("Log (Mem)", "\n".join(bot.logs), height=200)

with tab3:
    st.header("Trade History (Real Mode Only)")
    if st.button("🔄 Fetch History"):
        if st.session_state.rev_bot and st.session_state.rev_bot.mode == 'Real':
            trades = st.session_state.rev_bot.fetch_trade_history()
            if isinstance(trades, list):
                if len(trades) > 0:
                    df_trades = pd.DataFrame(trades)
                    df_show = df_trades[['datetime', 'side', 'price', 'amount', 'cost']]
                    df_show.columns = ['Time', 'Side', 'Price', 'Amt', 'Cost']
                    st.dataframe(df_show, use_container_width=True)
                else:
                    st.info("No trades found")
            else:
                st.error(f"Error: {trades}")
        else:
            st.warning("Available in Live Trading mode only")