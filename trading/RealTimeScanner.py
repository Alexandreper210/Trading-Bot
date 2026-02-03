# trading/RealTimeScanner.py
"""
🔍 REAL-TIME MULTI-PAIR SCANNER
================================
Scanne toutes les paires en temps réel
Trade sur la première qui donne un signal fort
"""

import pandas as pd
import numpy as np
from binance.client import Client
from binance.exceptions import BinanceAPIException
import time
from datetime import datetime
import pickle
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    TRADING_PAIRS,
    BINANCE_API_KEY,
    BINANCE_API_SECRET,
    MODEL_DIR,
    get_pair_config,
    get_model_path
)

'''
class RealTimeScanner:
    """
    Scanner temps réel :
    - Récupère les données minute par minute (ou seconde)
    - Analyse toutes les paires
    - Trade sur le meilleur signal
    - 1 seule position à la fois
    """
    
    def __init__(self, api_key, api_secret, testnet=True, update_interval=60):
        """
        update_interval: fréquence de mise à jour en secondes
            60 = chaque minute
            30 = toutes les 30 secondes
            1 = chaque seconde (attention au rate limit Binance!)
        """
        self.client = Client(api_key, api_secret, testnet=testnet)
        self.pairs = TRADING_PAIRS
        self.update_interval = update_interval
        
        # État global : UNE SEULE POSITION
        self.current_position = None  # None ou {'pair': 'BTCUSDT', 'entry_price': ..., etc.}
        
        # Modèles chargés
        self.models = {}
        self.scalers = {}
        self.feature_cols = {}
        
        # Cache des données pour éviter trop d'appels API
        self.data_cache = {}
        self.last_update = {}
        
        self.load_all_models()
        
        print(f"\n{'='*70}")
        print(f"🔍 REAL-TIME MULTI-PAIR SCANNER")
        print(f"{'='*70}")
        print(f"Environment: {'TESTNET ✅' if testnet else '⚠️  PRODUCTION ⚠️'}")
        print(f"Update interval: {update_interval}s")
        print(f"Scanning {len(self.pairs)} pairs:")
        for pair in self.pairs:
            print(f"  - {pair}")
        print(f"{'='*70}\n")
    
    def load_all_models(self):
        """Charge tous les modèles"""
        print("📦 Loading models...")
        
        loaded = 0
        for pair in self.pairs:
            paths = get_model_path(pair)
            
            if not os.path.exists(paths['model']):
                print(f"⚠️  {pair}: Model not found")
                continue
            
            try:
                with open(paths['model'], 'rb') as f:
                    self.models[pair] = pickle.load(f)
                
                with open(paths['scaler'], 'rb') as f:
                    self.scalers[pair] = pickle.load(f)
                
                with open(paths['features'], 'r') as f:
                    self.feature_cols[pair] = json.load(f)
                
                loaded += 1
                print(f"✅ {pair}")
            
            except Exception as e:
                print(f"❌ {pair}: {e}")
        
        print(f"\n✅ {loaded}/{len(self.pairs)} models loaded\n")
        
        if loaded == 0:
            print("⚠️  WARNING: No models loaded! Train them first:")
            print("   python main.py train all")
    
    def get_account_balance(self):
        """Balance USDT disponible"""
        try:
            account = self.client.get_account()
            balances = {b['asset']: float(b['free']) 
                       for b in account['balances'] if float(b['free']) > 0}
            return balances.get('USDT', 0)
        except BinanceAPIException as e:
            print(f"❌ Error getting balance: {e}")
            return 0
    
    def get_current_price(self, symbol):
        """Prix actuel d'une paire"""
        try:
            ticker = self.client.get_symbol_ticker(symbol=symbol)
            return float(ticker['price'])
        except BinanceAPIException as e:
            print(f"❌ Error getting price for {symbol}: {e}")
            return None
    
    def get_recent_klines(self, symbol, interval='1m', limit=1000):
        """
        Récupère les dernières bougies
        interval: '1m', '5m', '15m', '1h', etc.
        """
        try:
            klines = self.client.get_klines(
                symbol=symbol,
                interval=interval,
                limit=limit
            )
            
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                'taker_buy_quote', 'ignore'
            ])
            
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = df[col].astype(float)
            
            return df[['open', 'high', 'low', 'close', 'volume']]
        
        except BinanceAPIException as e:
            print(f"❌ Error fetching data for {symbol}: {e}")
            return None
    
    def create_features(self, df):
        """Crée les features pour ML"""
        from features.MinuteFeatureEngineer import MinuteFeatureEngineer
        
        try:
            fe = MinuteFeatureEngineer(df)
            df_features = fe.build_all_features(downsample=None)
            return df_features
        except Exception as e:
            print(f"❌ Feature creation error: {e}")
            return None
    
    def predict_signal(self, df_features, symbol):
        """
        Fait une prédiction sur la dernière bougie
        Returns: (prediction, confidence, latest_data)
        """
        if symbol not in self.models:
            return 0, 0.0, None
        
        try:
            latest = df_features.iloc[-1]
            
            # Features
            X = latest[self.feature_cols[symbol]].values.reshape(1, -1)
            X_scaled = self.scalers[symbol].transform(X)
            
            # Prédiction
            prediction = self.models[symbol].predict(X_scaled)[0]
            
            # Conversion (XGBoost: 0,1,2 → -1,0,1)
            if prediction == 2:
                prediction = 1
            elif prediction == 0:
                prediction = -1
            else:
                prediction = 0
            
            # Confiance
            confidence = 0.0
            if hasattr(self.models[symbol], 'predict_proba'):
                proba = self.models[symbol].predict_proba(X_scaled)[0]
                confidence = float(max(proba))
                
                # Seuil de confiance
                threshold = get_pair_config(symbol)['confidence_threshold']
                
                if confidence < threshold:
                    prediction = 0  # Pas assez confiant
            
            return prediction, confidence, latest
        
        except Exception as e:
            print(f"❌ Prediction error for {symbol}: {e}")
            return 0, 0.0, None
    
    def scan_pair(self, symbol):
        """
        Scanne une paire et retourne le signal
        Returns: dict avec toutes les infos
        """
        if symbol not in self.models:
            return None
        
        try:
            # Prix actuel
            current_price = self.get_current_price(symbol)
            if current_price is None:
                return None
            
            # Données récentes (200 dernières bougies minute)
            df = self.get_recent_klines(symbol, interval='1m', limit=1000)
            if df is None or len(df) < 500:
                return None
            
            # Features
            df_features = self.create_features(df)
            if df_features is None or len(df_features) == 0:
                return None
            
            # Prédiction
            prediction, confidence, latest_data = self.predict_signal(df_features, symbol)
            
            return {
                'symbol': symbol,
                'price': current_price,
                'signal': prediction,  # -1, 0, 1
                'confidence': confidence,
                'timestamp': datetime.now(),
                'latest_data': latest_data
            }
        
        except Exception as e:
            print(f"❌ Error scanning {symbol}: {e}")
            return None
    
    def scan_all_pairs(self):
        """
        Scanne toutes les paires EN PARALLÈLE
        Returns: list de résultats triés par confiance
        """
        results = []
        
        # Scan en parallèle (plus rapide)
        with ThreadPoolExecutor(max_workers=len(self.pairs)) as executor:
            futures = {executor.submit(self.scan_pair, pair): pair 
                      for pair in self.pairs}
            
            for future in as_completed(futures):
                result = future.result()
                if result:
                    results.append(result)
        
        # Trie par confiance décroissante
        results.sort(key=lambda x: x['confidence'], reverse=True)
        
        return results
    
    def place_order(self, symbol, side, quantity):
        """Place un ordre MARKET"""
        try:
            order = self.client.order_market(
                symbol=symbol,
                side=side,
                quantity=quantity
            )
            print(f"✅ Order executed: {side} {quantity} {symbol}")
            return order
        
        except BinanceAPIException as e:
            print(f"❌ Order failed: {e}")
            return None
    
    def open_position(self, signal_data):
        """Ouvre UNE position sur la paire qui a donné le signal"""
        symbol = signal_data['symbol']
        price = signal_data['price']
        
        # Vérifie qu'on n'a pas déjà une position
        if self.current_position is not None:
            print(f"⚠️  Already in position on {self.current_position['pair']}")
            return False
        
        # Capital disponible
        usdt_balance = self.get_account_balance()
        
        if usdt_balance < 20:
            print(f"⚠️  Insufficient balance: {usdt_balance:.2f} USDT")
            return False
        
        # Config de la paire
        config = get_pair_config(symbol)
        position_size = config['position_size']
        
        # Montant à investir
        invest_amount = usdt_balance * position_size
        
        # Quantité à acheter
        quantity = invest_amount / price
        
        # Arrondi selon la paire
        if 'BTC' in symbol:
            quantity = round(quantity, 5)
        elif 'ETH' in symbol or 'BNB' in symbol:
            quantity = round(quantity, 4)
        else:
            quantity = round(quantity, 2)
        
        # Place l'ordre
        print(f"\n{'─'*70}")
        print(f"🟢 OPENING POSITION")
        print(f"{'─'*70}")
        print(f"Pair:       {symbol}")
        print(f"Signal:     {signal_data['signal']} (conf: {signal_data['confidence']:.2%})")
        print(f"Price:      {price:.6f} USDT")
        print(f"Quantity:   {quantity}")
        print(f"Value:      {quantity * price:.2f} USDT")
        
        order = self.place_order(symbol, 'BUY', quantity)
        
        if order:
            # Calcul SL/TP
            sl_price = price * (1 - config['stop_loss'])
            tp_price = price * (1 + config['take_profit'])
            
            self.current_position = {
                'pair': symbol,
                'entry_price': price,
                'quantity': quantity,
                'entry_time': datetime.now(),
                'stop_loss': sl_price,
                'take_profit': tp_price,
                'entry_signal': signal_data
            }
            
            print(f"Stop Loss:  {sl_price:.6f}")
            print(f"Take Profit: {tp_price:.6f}")
            print(f"{'─'*70}\n")
            
            return True
        
        return False
    
    def close_position(self, reason="Signal"):
        """Ferme la position actuelle"""
        if self.current_position is None:
            return False
        
        symbol = self.current_position['pair']
        quantity = self.current_position['quantity']
        entry_price = self.current_position['entry_price']
        
        # Prix actuel
        current_price = self.get_current_price(symbol)
        if current_price is None:
            print(f"❌ Cannot get price for {symbol}")
            return False
        
        print(f"\n{'─'*70}")
        print(f"🔴 CLOSING POSITION ({reason})")
        print(f"{'─'*70}")
        print(f"Pair:     {symbol}")
        print(f"Entry:    {entry_price:.6f} USDT")
        print(f"Exit:     {current_price:.6f} USDT")
        
        # Place l'ordre de vente
        order = self.place_order(symbol, 'SELL', quantity)
        
        if order:
            # PnL
            pnl = (current_price - entry_price) * quantity
            pnl_pct = (current_price / entry_price - 1) * 100
            
            # Durée
            duration = datetime.now() - self.current_position['entry_time']
            
            print(f"PnL:      {pnl:+.2f} USDT ({pnl_pct:+.2f}%)")
            print(f"Duration: {duration}")
            print(f"{'─'*70}\n")
            
            # Reset
            self.current_position = None
            
            return True
        
        return False
    
    def check_stop_loss_take_profit(self):
        """Vérifie SL/TP sur la position actuelle"""
        if self.current_position is None:
            return
        
        symbol = self.current_position['pair']
        current_price = self.get_current_price(symbol)
        
        if current_price is None:
            return
        
        sl = self.current_position['stop_loss']
        tp = self.current_position['take_profit']
        
        # Stop Loss
        if current_price <= sl:
            print(f"⛔ STOP LOSS TRIGGERED!")
            self.close_position(reason="Stop Loss")
        
        # Take Profit
        elif current_price >= tp:
            print(f"🎯 TAKE PROFIT TRIGGERED!")
            self.close_position(reason="Take Profit")
    
    def display_scan_results(self, results):
        """Affiche les résultats du scan"""
        print(f"\n{'─'*70}")
        print(f"📊 SCAN RESULTS ({len(results)} pairs)")
        print(f"{'─'*70}")
        
        # Affiche top 10
        for i, res in enumerate(results[:10], 1):
            signal_str = "🟢 BUY " if res['signal'] == 1 else "🔴 SELL" if res['signal'] == -1 else "⚪ HOLD"
            
            print(f"{i:2d}. {res['symbol']:12s} | {res['price']:>12.6f} | "
                  f"{signal_str} | Conf: {res['confidence']:>5.1%}")
        
        print(f"{'─'*70}\n")
    
    def run_once(self):
        """Une itération de scan"""
        print(f"\n{'='*70}")
        print(f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*70}")
        
        # Balance
        balance = self.get_account_balance()
        print(f"💼 Balance: {balance:.2f} USDT")
        
        # Position actuelle
        if self.current_position:
            pos = self.current_position
            current_price = self.get_current_price(pos['pair'])
            if current_price:
                pnl = (current_price - pos['entry_price']) * pos['quantity']
                pnl_pct = (current_price / pos['entry_price'] - 1) * 100
                
                print(f"📊 Position: {pos['pair']}")
                print(f"   Entry: {pos['entry_price']:.6f} | Current: {current_price:.6f}")
                print(f"   PnL: {pnl:+.2f} USDT ({pnl_pct:+.2f}%)")
        else:
            print(f"📊 Position: None")
        
        # 1. Vérifie SL/TP sur position existante
        self.check_stop_loss_take_profit()
        
        # 2. Scanne toutes les paires
        print(f"\n🔍 Scanning {len(self.pairs)} pairs...")
        results = self.scan_all_pairs()
        
        # 3. Affiche résultats
        self.display_scan_results(results)
        
        # 4. Logique de trading
        if len(results) > 0:
            best_signal = results[0]  # Le meilleur signal (plus haute confiance)
            
            # Si on a déjà une position
            if self.current_position:
                current_pair = self.current_position['pair']
                
                # Cherche le signal pour la paire actuelle
                current_signal = next((r for r in results if r['symbol'] == current_pair), None)
                
                if current_signal and current_signal['signal'] == -1:
                    # Signal de sortie sur la paire actuelle
                    print(f"🔴 EXIT signal detected on {current_pair}")
                    self.close_position(reason="Exit Signal")
            
            # Si pas de position et signal BUY fort
            elif best_signal['signal'] == 1:
                print(f"\n🟢 BEST BUY SIGNAL: {best_signal['symbol']} (conf: {best_signal['confidence']:.1%})")
                
                # Ouvre position si confiance élevée
                if best_signal['confidence'] >= 0.70:  # Seuil ajustable
                    self.open_position(best_signal)
                else:
                    print(f"   → Confidence too low, waiting...")
        
        print(f"{'='*70}")
    
    def run_loop(self):
        """Boucle infinie de scanning"""
        print(f"\n🚀 Starting real-time scanner")
        print(f"Update every {self.update_interval}s")
        print("Press Ctrl+C to stop\n")
        
        try:
            while True:
                self.run_once()
                
                print(f"\n😴 Next scan in {self.update_interval}s...\n")
                time.sleep(self.update_interval)
        
        except KeyboardInterrupt:
            print("\n\n🛑 Scanner stopped by user")
            
            # Ferme position si ouverte
            if self.current_position:
                print(f"\n⚠️  Closing open position...")
                self.close_position(reason="Bot Shutdown")
            
            print("\n✅ Goodbye! 👋")
        
        except Exception as e:
            print(f"\n❌ Fatal error: {e}")
            import traceback
            traceback.print_exc()
'''

class RealTimeScanner:
    """
    Scanner temps réel :
    - Récupère les données minute par minute (ou seconde)
    - Analyse toutes les paires
    - Trade sur le meilleur signal
    - 1 seule position à la fois
    """
    
    def __init__(self, api_key, api_secret, testnet=True, update_interval=60):
        """
        update_interval: fréquence de mise à jour en secondes
            60 = chaque minute
            30 = toutes les 30 secondes
            1 = chaque seconde (attention au rate limit Binance!)
        """
        self.client = Client(api_key, api_secret, testnet=testnet)
        self.pairs = TRADING_PAIRS
        self.update_interval = update_interval
        
        # État global : UNE SEULE POSITION
        self.current_position = None  # None ou {'pair': 'BTCUSDT', 'entry_price': ..., etc.}
        
        # Modèles chargés
        self.models = {}
        self.scalers = {}
        self.feature_cols = {}
        
        # Cache des données pour éviter trop d'appels API
        self.data_cache = {}
        self.last_update = {}
        
        self.load_all_models()
        
        print(f"\n{'='*70}")
        print(f"🔍 REAL-TIME MULTI-PAIR SCANNER")
        print(f"{'='*70}")
        print(f"Environment: {'TESTNET ✅' if testnet else '⚠️  PRODUCTION ⚠️'}")
        print(f"Update interval: {update_interval}s")
        print(f"Scanning {len(self.pairs)} pairs:")
        for pair in self.pairs:
            print(f"  - {pair}")
        print(f"{'='*70}\n")
    
    def load_all_models(self):
        """Charge tous les modèles"""
        print("📦 Loading models...")
        
        loaded = 0
        for pair in self.pairs:
            paths = get_model_path(pair)
            
            if not os.path.exists(paths['model']):
                print(f"⚠️  {pair}: Model not found")
                continue
            
            try:
                with open(paths['model'], 'rb') as f:
                    self.models[pair] = pickle.load(f)
                
                with open(paths['scaler'], 'rb') as f:
                    self.scalers[pair] = pickle.load(f)
                
                with open(paths['features'], 'r') as f:
                    self.feature_cols[pair] = json.load(f)
                
                loaded += 1
                print(f"✅ {pair}")
            
            except Exception as e:
                print(f"❌ {pair}: {e}")
        
        print(f"\n✅ {loaded}/{len(self.pairs)} models loaded\n")
        
        if loaded == 0:
            print("⚠️  WARNING: No models loaded! Train them first:")
            print("   python main.py train all")
    
    def get_account_balance(self):
        """Balance USDT disponible"""
        try:
            account = self.client.get_account()
            balances = {b['asset']: float(b['free']) 
                       for b in account['balances'] if float(b['free']) > 0}
            return balances.get('USDT', 0)
        except BinanceAPIException as e:
            print(f"❌ Error getting balance: {e}")
            return 0
    
    def get_current_price(self, symbol):
        """Prix actuel d'une paire"""
        try:
            ticker = self.client.get_symbol_ticker(symbol=symbol)
            return float(ticker['price'])
        except BinanceAPIException as e:
            print(f"❌ Error getting price for {symbol}: {e}")
            return None
    
    def get_recent_klines(self, symbol, interval='1m', limit=1000):
        """
        Récupère les dernières bougies
        interval: '1m', '5m', '15m', '1h', etc.
        """
        try:
            klines = self.client.get_klines(
                symbol=symbol,
                interval=interval,
                limit=limit
            )
            
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                'taker_buy_quote', 'ignore'
            ])
            
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = df[col].astype(float)
            
            return df[['open', 'high', 'low', 'close', 'volume']]
        
        except BinanceAPIException as e:
            print(f"❌ Error fetching data for {symbol}: {e}")
            return None
    
    def create_features(self, df):
        """Crée les features pour ML"""
        from features.MinuteFeatureEngineer import MinuteFeatureEngineer
        
        try:
            fe = MinuteFeatureEngineer(df)
            df_features = fe.build_all_features(downsample=None)
            return df_features
        except Exception as e:
            print(f"❌ Feature creation error: {e}")
            return None
    
    def predict_signal(self, df_features, symbol):
        """
        Fait une prédiction sur la dernière bougie
        Returns: (prediction, confidence, latest_data)
        """
        if symbol not in self.models:
            return 0, 0.0, None
        
        try:
            latest = df_features.iloc[-1]
            
            # Features
            X = latest[self.feature_cols[symbol]].values.reshape(1, -1)
            X_scaled = self.scalers[symbol].transform(X)
            
            # Prédiction
            prediction = self.models[symbol].predict(X_scaled)[0]
            
            # Conversion (XGBoost: 0,1,2 → -1,0,1)
            if prediction == 2:
                prediction = 1
            elif prediction == 0:
                prediction = -1
            else:
                prediction = 0
            
            # Confiance
            confidence = 0.0
            if hasattr(self.models[symbol], 'predict_proba'):
                proba = self.models[symbol].predict_proba(X_scaled)[0]
                confidence = float(max(proba))
                
                # Seuil de confiance
                threshold = get_pair_config(symbol)['confidence_threshold']
                
                if confidence < threshold:
                    prediction = 0  # Pas assez confiant
            
            return prediction, confidence, latest
        
        except Exception as e:
            print(f"❌ Prediction error for {symbol}: {e}")
            return 0, 0.0, None
    
    def scan_pair(self, symbol):
        """
        Scanne une paire et retourne le signal
        Returns: dict avec toutes les infos
        """
        if symbol not in self.models:
            return None
        
        try:
            # Prix actuel
            current_price = self.get_current_price(symbol)
            if current_price is None:
                return None
            
            # Données récentes (200 dernières bougies minute)
            df = self.get_recent_klines(symbol, interval='1m', limit=1000)
            if df is None or len(df) < 500:
                return None
            
            # Features
            df_features = self.create_features(df)
            if df_features is None or len(df_features) == 0:
                return None
            
            # Prédiction
            prediction, confidence, latest_data = self.predict_signal(df_features, symbol)
            
            return {
                'symbol': symbol,
                'price': current_price,
                'signal': prediction,  # -1, 0, 1
                'confidence': confidence,
                'timestamp': datetime.now(),
                'latest_data': latest_data
            }
        
        except Exception as e:
            print(f"❌ Error scanning {symbol}: {e}")
            return None
    
    def scan_all_pairs(self):
        """
        Scanne toutes les paires EN PARALLÈLE
        Returns: list de résultats triés par confiance
        """
        results = []
        
        # Scan en parallèle (plus rapide)
        with ThreadPoolExecutor(max_workers=len(self.pairs)) as executor:
            futures = {executor.submit(self.scan_pair, pair): pair 
                      for pair in self.pairs}
            
            for future in as_completed(futures):
                result = future.result()
                if result:
                    results.append(result)
        
        # Trie par confiance décroissante
        results.sort(key=lambda x: x['confidence'], reverse=True)
        
        return results
    
    def place_order(self, symbol, side, quantity):
        """Place un ordre MARKET sur Binance Futures"""
        try:
            order = self.client.futures_create_order(
                symbol=symbol,
                side=side,       # "BUY" ou "SELL"
                type="MARKET",
                quantity=quantity
            )
            print(f"✅ Futures order executed: {side} {quantity} {symbol}")
            return order
        except BinanceAPIException as e:
            print(f"❌ Futures order failed: {e}")
            return None

    def open_position(self, signal_data):
        symbol = signal_data['symbol']
        price = signal_data['price']

        if self.current_position is not None:
            print(f"⚠️ Already in position on {self.current_position['pair']}")
            return False

        usdt_balance = self.get_account_balance()
        if usdt_balance < 20:
            print(f"⚠️ Insufficient balance: {usdt_balance:.2f} USDT")
            return False

        config = get_pair_config(symbol)
        invest_amount = usdt_balance * config['position_size']
        quantity = round(invest_amount / price, 3)

        # Déterminer LONG ou SHORT
        if signal_data['signal'] == 1:
            side = "BUY"   # LONG
            print(f"\n🟢 Opening LONG on {symbol}")
        elif signal_data['signal'] == -1:
            side = "SELL"  # SHORT
            print(f"\n🔴 Opening SHORT on {symbol}")
        else:
            print("⚪ HOLD signal, no trade")
            return False

        order = self.place_order(symbol, side, quantity)
        if order:
            sl_price = price * (1 - config['stop_loss']) if side == "BUY" else price * (1 + config['stop_loss'])
            tp_price = price * (1 + config['take_profit']) if side == "BUY" else price * (1 - config['take_profit'])

            self.current_position = {
                'pair': symbol,
                'entry_price': price,
                'quantity': quantity,
                'entry_time': datetime.now(),
                'stop_loss': sl_price,
                'take_profit': tp_price,
                'side': side,
                'entry_signal': signal_data
            }
            print(f"Stop Loss:  {sl_price:.6f}")
            print(f"Take Profit: {tp_price:.6f}")
            return True
        return False

    
    def close_position(self, reason="Signal"):
        if self.current_position is None:
            return False

        symbol = self.current_position['pair']
        qty = self.current_position['quantity']
        side = "SELL" if self.current_position['side'] == "BUY" else "BUY"

        order = self.place_order(symbol, side, qty)
        if order:
            print(f"✅ Position closed ({reason})")
            self.current_position = None
            return True
        return False

    def check_stop_loss_take_profit(self):
        """Vérifie SL/TP sur la position actuelle"""
        if self.current_position is None:
            return
        
        symbol = self.current_position['pair']
        current_price = self.get_current_price(symbol)
        
        if current_price is None:
            return
        
        sl = self.current_position['stop_loss']
        tp = self.current_position['take_profit']
        
        # Stop Loss
        if current_price <= sl:
            print(f"⛔ STOP LOSS TRIGGERED!")
            self.close_position(reason="Stop Loss")
        
        # Take Profit
        elif current_price >= tp:
            print(f"🎯 TAKE PROFIT TRIGGERED!")
            self.close_position(reason="Take Profit")
    
    def display_scan_results(self, results):
        """Affiche les résultats du scan"""
        print(f"\n{'─'*70}")
        print(f"📊 SCAN RESULTS ({len(results)} pairs)")
        print(f"{'─'*70}")
        
        # Affiche top 10
        for i, res in enumerate(results[:10], 1):
            signal_str = "🟢 BUY " if res['signal'] == 1 else "🔴 SELL" if res['signal'] == -1 else "⚪ HOLD"
            
            print(f"{i:2d}. {res['symbol']:12s} | {res['price']:>12.6f} | "
                  f"{signal_str} | Conf: {res['confidence']:>5.1%}")
        
        print(f"{'─'*70}\n")
    
    def run_once(self):
        """Une itération de scan"""
        print(f"\n{'='*70}")
        print(f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*70}")
        
        # Balance
        balance = self.get_account_balance()
        print(f"💼 Balance: {balance:.2f} USDT")
        
        # Position actuelle
        if self.current_position:
            pos = self.current_position
            current_price = self.get_current_price(pos['pair'])
            if current_price:
                pnl = (current_price - pos['entry_price']) * pos['quantity']
                pnl_pct = (current_price / pos['entry_price'] - 1) * 100
                
                print(f"📊 Position: {pos['pair']}")
                print(f"   Entry: {pos['entry_price']:.6f} | Current: {current_price:.6f}")
                print(f"   PnL: {pnl:+.2f} USDT ({pnl_pct:+.2f}%)")
        else:
            print(f"📊 Position: None")
        
        # 1. Vérifie SL/TP sur position existante
        self.check_stop_loss_take_profit()
        
        # 2. Scanne toutes les paires
        print(f"\n🔍 Scanning {len(self.pairs)} pairs...")
        results = self.scan_all_pairs()
        
        # 3. Affiche résultats
        self.display_scan_results(results)
        
        # 4. Logique de trading
        if len(results) > 0:
            best_signal = results[0]  # Le meilleur signal (plus haute confiance)
            
            # Si on a déjà une position
            if self.current_position:
                current_pair = self.current_position['pair']
                
                # Cherche le signal pour la paire actuelle
                current_signal = next((r for r in results if r['symbol'] == current_pair), None)
                
                if current_signal and current_signal['signal'] == -1:
                    # Signal de sortie sur la paire actuelle
                    print(f"🔴 EXIT signal detected on {current_pair}")
                    self.close_position(reason="Exit Signal")
            
            # Si pas de position et signal BUY fort
            elif best_signal['signal'] == 1:
                print(f"\n🟢 BEST BUY SIGNAL: {best_signal['symbol']} (conf: {best_signal['confidence']:.1%})")
                
                # Ouvre position si confiance élevée
                if best_signal['confidence'] >= 0.70:  # Seuil ajustable
                    self.open_position(best_signal)
                else:
                    print(f"   → Confidence too low, waiting...")
        
        print(f"{'='*70}")
    
    def run_loop(self):
        """Boucle infinie de scanning"""
        print(f"\n🚀 Starting real-time scanner")
        print(f"Update every {self.update_interval}s")
        print("Press Ctrl+C to stop\n")
        
        try:
            while True:
                self.run_once()
                
                print(f"\n😴 Next scan in {self.update_interval}s...\n")
                time.sleep(self.update_interval)
        
        except KeyboardInterrupt:
            print("\n\n🛑 Scanner stopped by user")
            
            # Ferme position si ouverte
            if self.current_position:
                print(f"\n⚠️  Closing open position...")
                self.close_position(reason="Bot Shutdown")
            
            print("\n✅ Goodbye! 👋")
        
        except Exception as e:
            print(f"\n❌ Fatal error: {e}")
            import traceback
            traceback.print_exc()


