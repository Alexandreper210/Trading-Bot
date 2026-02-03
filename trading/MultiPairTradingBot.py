# ==========================================
# trading/multi_pair_bot.py
# ==========================================

import pandas as pd
import numpy as np
from binance.client import Client
from binance.exceptions import BinanceAPIException
import time
from datetime import datetime
import pickle
import json
import os
from concurrent.futures import ThreadPoolExecutor
from config import *

class MultiPairTradingBot:
    """
    Bot qui :
    - Charge 1 modèle par paire
    - Monitore toutes les paires en temps réel
    - Trade automatiquement sur les signaux
    """
    
    def __init__(self, api_key, api_secret, testnet=True):
        self.client = Client(api_key, api_secret, testnet=testnet)
        self.pairs = TRADING_PAIRS
        self.interval = INTERVAL
        
        # État des positions par paire
        self.positions = {pair: {
            'in_position': False,
            'entry_price': 0,
            'quantity': 0,
            'entry_time': None,
            'signal': 0  # 1=long, -1=short, 0=flat
        } for pair in self.pairs}
        
        # Modèles chargés (1 par paire)
        self.models = {}
        self.scalers = {}
        self.feature_cols = {}
        
        # Chargement des modèles
        self.load_all_models()
        
        print(f"\n{'='*70}")
        print(f"🤖 MULTI-PAIR TRADING BOT")
        print(f"{'='*70}")
        print(f"Environment: {'TESTNET' if testnet else '⚠️  PRODUCTION ⚠️'}")
        print(f"Trading pairs: {len(self.pairs)}")
        for pair in self.pairs:
            print(f"  - {pair}")
        print(f"{'='*70}\n")
    
    def load_all_models(self):
        """Charge les modèles pour toutes les paires"""
        print("📦 Loading models...")
        
        for pair in self.pairs:
            model_path = f"{MODEL_DIR}/{pair}_model.pkl"
            scaler_path = f"{MODEL_DIR}/{pair}_scaler.pkl"
            features_path = f"{MODEL_DIR}/{pair}_features.json"
            
            # Vérifie que les fichiers existent
            if not os.path.exists(model_path):
                print(f"❌ Model not found for {pair}: {model_path}")
                print(f"   → Run: python main.py train {pair}")
                continue
            
            try:
                # Charge le modèle
                with open(model_path, 'rb') as f:
                    self.models[pair] = pickle.load(f)
                
                # Charge le scaler
                with open(scaler_path, 'rb') as f:
                    self.scalers[pair] = pickle.load(f)
                
                # Charge les features
                with open(features_path, 'r') as f:
                    self.feature_cols[pair] = json.load(f)
                
                print(f"✅ {pair}: Model loaded ({len(self.feature_cols[pair])} features)")
            
            except Exception as e:
                print(f"❌ Error loading model for {pair}: {e}")
        
        if len(self.models) == 0:
            print("\n⚠️  WARNING: No models loaded!")
            print("Please train models first:")
            print("  python main.py train all")
        
        print(f"\n✅ {len(self.models)}/{len(self.pairs)} models loaded\n")
    
    def get_account_balance(self):
        """Récupère le solde du compte"""
        try:
            account = self.client.get_account()
            balances = {b['asset']: float(b['free']) 
                       for b in account['balances'] if float(b['free']) > 0}
            return balances
        except BinanceAPIException as e:
            print(f"❌ Error getting balance: {e}")
            return {}
    
    def get_total_balance_usdt(self):
        """Calcule le total du portefeuille en USDT"""
        balances = self.get_account_balance()
        total = balances.get('USDT', 0)
        
        # Ajoute la valeur des positions ouvertes
        for pair, pos in self.positions.items():
            if pos['in_position']:
                current_price = self.get_current_price(pair)
                if current_price:
                    total += pos['quantity'] * current_price
        
        return total
    
    def count_open_positions(self):
        """Nombre de positions ouvertes"""
        return sum(1 for pos in self.positions.values() if pos['in_position'])
    
    def can_open_new_position(self):
        """Vérifie si on peut ouvrir une nouvelle position"""
        # Limite du nombre de positions
        if self.count_open_positions() >= MAX_POSITIONS:
            return False, "Max positions reached"
        
        # Vérifie le capital disponible
        balances = self.get_account_balance()
        usdt_available = balances.get('USDT', 0)
        
        if usdt_available < 20:  # Minimum 20 USDT
            return False, "Insufficient USDT balance"
        
        return True, "OK"
    
    def get_current_price(self, symbol):
        """Prix actuel d'une paire"""
        try:
            ticker = self.client.get_symbol_ticker(symbol=symbol)
            return float(ticker['price'])
        except BinanceAPIException as e:
            print(f"❌ Error getting price for {symbol}: {e}")
            return None
    
    def get_historical_data(self, symbol, limit=200):
        """Récupère les données historiques"""
        try:
            klines = self.client.get_klines(
                symbol=symbol,
                interval=self.interval,
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
            print(f"❌ Error getting data for {symbol}: {e}")
            return None
    
    def create_features(self, df, symbol):
        """Crée les features pour une paire"""
        from features.MinuteFeatureEngineer import MinuteFeatureEngineer
        
        try:
            engineer = MinuteFeatureEngineer(df)
            df_features = engineer.build_all_features(downsample=None)
            return df_features
        except Exception as e:
            print(f"❌ Error creating features for {symbol}: {e}")
            return None
    
    def predict(self, df_features, symbol):
        """
        Fait une prédiction pour une paire
        Returns: (prediction, confidence)
            prediction: 1 (BUY), -1 (SELL), 0 (HOLD)
            confidence: 0.0 - 1.0
        """
        if symbol not in self.models:
            return 0, 0.0
        
        try:
            # Dernière ligne = données actuelles
            latest = df_features.iloc[-1]
            
            # Sélectionne les features
            X = latest[self.feature_cols[symbol]].values.reshape(1, -1)
            
            # Scale
            X_scaled = self.scalers[symbol].transform(X)
            
            # Prédiction
            prediction = self.models[symbol].predict(X_scaled)[0]
            
            # Convertit si nécessaire (XGBoost peut retourner 0,1,2)
            if prediction == 2:
                prediction = 1
            elif prediction == 0:
                prediction = -1
            else:
                prediction = 0
            
            # Calcule la confiance
            if hasattr(self.models[symbol], 'predict_proba'):
                proba = self.models[symbol].predict_proba(X_scaled)[0]
                confidence = max(proba)
                
                # Filtre selon seuil de confiance
                pair_config = get_pair_config(symbol)
                threshold = pair_config['confidence_threshold']
                
                if confidence < threshold:
                    return 0, confidence  # Pas assez confiant
                
                return prediction, confidence
            
            return prediction, 1.0
        
        except Exception as e:
            print(f"❌ Prediction error for {symbol}: {e}")
            return 0, 0.0
    
    def place_order(self, symbol, side, quantity):
        """Place un ordre MARKET"""
        try:
            order = self.client.order_market(
                symbol=symbol,
                side=side,
                quantity=quantity
            )
            return order
        
        except BinanceAPIException as e:
            print(f"❌ Order failed for {symbol}: {e}")
            return None
    
    def open_position(self, symbol, price, signal):
        """Ouvre une position sur une paire"""
        can_open, reason = self.can_open_new_position()
        
        if not can_open:
            print(f"⚠️  Cannot open {symbol}: {reason}")
            return False
        
        # Capital disponible
        balances = self.get_account_balance()
        total_usdt = balances.get('USDT', 0)
        
        # Capital alloué à cette paire
        pair_capital = get_capital_for_pair(symbol, total_usdt)
        
        # Taille de position selon config
        pair_config = get_pair_config(symbol)
        position_size = pair_config['position_size']
        
        # Montant à investir
        invest_amount = pair_capital * position_size
        
        if invest_amount < 10:  # Minimum Binance
            print(f"⚠️  Position too small for {symbol}: {invest_amount:.2f} USDT")
            return False
        
        # Calcul quantité
        quantity = invest_amount / price
        
        # Arrondi selon la paire (règles Binance)
        if 'BTC' in symbol:
            quantity = round(quantity, 5)
        elif 'ETH' in symbol or 'BNB' in symbol:
            quantity = round(quantity, 4)
        else:
            quantity = round(quantity, 2)
        
        # Place l'ordre
        print(f"\n🔵 Opening position on {symbol}...")
        order = self.place_order(symbol, 'BUY', quantity)
        
        if order:
            self.positions[symbol] = {
                'in_position': True,
                'entry_price': price,
                'quantity': quantity,
                'entry_time': datetime.now(),
                'signal': signal
            }
            
            print(f"✅ LONG opened on {symbol}")
            print(f"   Price: {price:.6f} USDT")
            print(f"   Quantity: {quantity}")
            print(f"   Value: {quantity * price:.2f} USDT")
            
            # Calcul SL/TP
            sl_price = price * (1 - pair_config['stop_loss'])
            tp_price = price * (1 + pair_config['take_profit'])
            print(f"   Stop Loss: {sl_price:.6f}")
            print(f"   Take Profit: {tp_price:.6f}")
            
            return True
        
        return False
    
    def close_position(self, symbol, price, reason="Signal"):
        """Ferme une position"""
        pos = self.positions[symbol]
        
        if not pos['in_position']:
            return False
        
        print(f"\n🔴 Closing position on {symbol} ({reason})...")
        
        # Place l'ordre de vente
        order = self.place_order(symbol, 'SELL', pos['quantity'])
        
        if order:
            # Calcul PnL
            pnl = (price - pos['entry_price']) * pos['quantity']
            pnl_pct = (price / pos['entry_price'] - 1) * 100
            
            # Durée
            duration = datetime.now() - pos['entry_time']
            
            print(f"✅ Position closed on {symbol}")
            print(f"   Entry: {pos['entry_price']:.6f} → Exit: {price:.6f}")
            print(f"   PnL: {pnl:+.2f} USDT ({pnl_pct:+.2f}%)")
            print(f"   Duration: {duration}")
            
            # Reset position
            self.positions[symbol] = {
                'in_position': False,
                'entry_price': 0,
                'quantity': 0,
                'entry_time': None,
                'signal': 0
            }
            
            return True
        
        return False
    
    def check_stop_loss_take_profit(self, symbol, current_price):
        """Vérifie stop loss et take profit"""
        pos = self.positions[symbol]
        
        if not pos['in_position']:
            return
        
        pair_config = get_pair_config(symbol)
        
        # Calcul PnL actuel
        pnl_pct = (current_price / pos['entry_price'] - 1)
        
        # Stop Loss
        if pnl_pct <= -pair_config['stop_loss']:
            print(f"⛔ STOP LOSS triggered on {symbol}")
            self.close_position(symbol, current_price, "Stop Loss")
            return
        
        # Take Profit
        if pnl_pct >= pair_config['take_profit']:
            print(f"🎯 TAKE PROFIT triggered on {symbol}")
            self.close_position(symbol, current_price, "Take Profit")
            return
    
    def process_pair(self, symbol):
        """
        Traite une paire :
        1. Récupère les données
        2. Crée les features
        3. Fait une prédiction
        4. Gère les positions (open/close)
        """
        try:
            # Vérifie que le modèle est chargé
            if symbol not in self.models:
                return
            
            # Prix actuel
            current_price = self.get_current_price(symbol)
            if current_price is None:
                return
            
            # Vérifie SL/TP sur positions existantes
            self.check_stop_loss_take_profit(symbol, current_price)
            
            # Données historiques
            df = self.get_historical_data(symbol, limit=200)
            if df is None or len(df) < 100:
                return
            
            # Features
            df_features = self.create_features(df, symbol)
            if df_features is None or len(df_features) == 0:
                return
            
            # Prédiction
            prediction, confidence = self.predict(df_features, symbol)
            
            # Affichage
            signal_emoji = "🟢" if prediction == 1 else "🔴" if prediction == -1 else "⚪"
            signal_str = "BUY" if prediction == 1 else "SELL" if prediction == -1 else "HOLD"
            
            print(f"{signal_emoji} {symbol:12s} | {current_price:>12.6f} | {signal_str:5s} (conf: {confidence:.2f})")
            
            # Logique de trading
            pos = self.positions[symbol]
            
            # Cas 1 : Signal BUY + pas en position → OUVRIR
            if prediction == 1 and not pos['in_position']:
                self.open_position(symbol, current_price, prediction)
            
            # Cas 2 : Signal SELL + en position → FERMER
            elif prediction == -1 and pos['in_position']:
                self.close_position(symbol, current_price, "Sell Signal")
            
            # Cas 3 : En position mais signal change → FERMER (optionnel)
            elif pos['in_position'] and prediction != pos['signal'] and prediction != 0:
                self.close_position(symbol, current_price, "Signal Changed")
        
        except Exception as e:
            print(f"❌ Error processing {symbol}: {e}")
    
    def run_once(self):
        """Exécute une itération sur toutes les paires"""
        print(f"\n{'='*70}")
        print(f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*70}")
        
        # Balance
        total_balance = self.get_total_balance_usdt()
        usdt_free = self.get_account_balance().get('USDT', 0)
        positions_count = self.count_open_positions()
        
        print(f"💼 Total Balance: {total_balance:.2f} USDT")
        print(f"💵 Free USDT: {usdt_free:.2f} USDT")
        print(f"📊 Positions: {positions_count}/{MAX_POSITIONS}")
        print(f"{'-'*70}")
        
        # Traite chaque paire (en parallèle pour plus de rapidité)
        with ThreadPoolExecutor(max_workers=len(self.pairs)) as executor:
            executor.map(self.process_pair, self.pairs)
        
        print(f"{'='*70}")
    
    def run_loop(self, interval_seconds=900):
        """
        Lance le bot en boucle continue
        interval_seconds: 900 = 15 minutes
        """
        print(f"\n🚀 Starting bot loop (interval: {interval_seconds}s = {interval_seconds//60}min)")
        print("Press Ctrl+C to stop\n")
        
        try:
            while True:
                self.run_once()
                
                print(f"\n😴 Sleeping {interval_seconds}s...")
                time.sleep(interval_seconds)
        
        except KeyboardInterrupt:
            print("\n\n🛑 Bot stopped by user")
            
            # Ferme toutes les positions ouvertes
            print("\n📤 Closing all open positions...")
            for symbol in self.pairs:
                if self.positions[symbol]['in_position']:
                    current_price = self.get_current_price(symbol)
                    if current_price:
                        self.close_position(symbol, current_price, "Bot Shutdown")
            
            print("\n✅ All positions closed. Goodbye! 👋")
        
        except Exception as e:
            print(f"\n❌ Fatal error: {e}")
            import traceback
            traceback.print_exc()


