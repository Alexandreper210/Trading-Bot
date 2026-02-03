import pandas as pd
import numpy as np
from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands, AverageTrueRange
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
import joblib
from sklearn.metrics import classification_report, confusion_matrix
from binance.client import Client
from binance.exceptions import BinanceAPIException
import time
from datetime import datetime
import pickle
import random


class MinuteFeatureEngineer:
    def __init__(self, df):
        self.df = df.copy()

    def resample_to_multiple_timeframes(self):
        """
        Crée des features sur plusieurs timeframes
        CRUCIAL pour réduire le bruit des données minute
        """
        # Sauvegarde index minute
        self.df_minute = self.df.copy()

        # Agrégations utiles
        timeframes = {
            '5min': self.df.resample('5min').agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }),
            '15min': self.df.resample('15min').agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }),
            '1h': self.df.resample('1h').agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }),
            '4h': self.df.resample('4h').agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            })
        }

        self.timeframes = timeframes
        return self

    def add_minute_specific_features(self):
        """Features spécifiques aux données minute"""
        df = self.df

        # 1. MICROSTRUCTURE
        # Spread bid-ask approximé
        df['high_low_spread'] = (df['high'] - df['low']) / df['close']

        # Volatilité intrabar
        df['intrabar_range'] = (df['high'] - df['low']) / df['open']

        # 2. TICK-BASED (approximations)
        # Proportion de la bougie
        df['body'] = abs(df['close'] - df['open'])
        df['upper_wick'] = df['high'] - df[['open', 'close']].max(axis=1)
        df['lower_wick'] = df[['open', 'close']].min(axis=1) - df['low']
        df['body_ratio'] = df['body'] / (df['high'] - df['low'] + 1e-10)

        # 3. VOLUME PROFILE
        # Volume par minute vs moyenne récente
        df['volume_ma_10'] = df['volume'].rolling(10).mean()
        df['volume_spike'] = df['volume'] / (df['volume_ma_10'] + 1e-10)

        # Volume + direction = pression achat/vente
        df['price_change'] = df['close'] - df['open']
        df['volume_pressure'] = df['volume'] * np.sign(df['price_change'])
        df['cumulative_volume_pressure'] = (
            df['volume_pressure'].rolling(60).sum()
        )

        return self

    def add_short_term_momentum(self):
        """Momentum adapté aux données minute"""
        df = self.df

        # Returns ultra court-terme (1, 5, 15, 30, 60 min)
        for period in [1, 5, 15, 30, 60]:
            df[f'return_{period}m'] = df['close'].pct_change(period)

        # Momentum cumulé
        df['momentum_5m'] = df['close'] / df['close'].shift(5) - 1
        df['momentum_30m'] = df['close'] / df['close'].shift(30) - 1
        df['momentum_60m'] = df['close'] / df['close'].shift(60) - 1

        # RSI court terme
        df['rsi_14'] = RSIIndicator(df['close'], window=14).rsi()
        df['rsi_30'] = RSIIndicator(df['close'], window=30).rsi()

        # MACD adapté
        macd = MACD(df['close'], window_fast=12, window_slow=26, window_sign=9)
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_hist'] = macd.macd_diff()

        return self

    def add_volatility_features_minute(self):
        """Volatilité pour données minute"""
        df = self.df

        # Volatilité réalisée (différentes fenêtres)
        for window in [15, 30, 60, 240]:  
            returns = df['close'].pct_change()
            df[f'volatility_{window}m'] = (
                returns.rolling(window).std() * np.sqrt(60 * 24 * 365)  # annualisée
            )

        # ATR adapté
        df['atr_14'] = AverageTrueRange(
            df['high'], df['low'], df['close'], window=14
        ).average_true_range()

        df['atr_30'] = AverageTrueRange(
            df['high'], df['low'], df['close'], window=30
        ).average_true_range()

        # Bollinger Bands
        for window in [20, 60]:
            bb = BollingerBands(df['close'], window=window)
            df[f'bb_high_{window}'] = bb.bollinger_hband()
            df[f'bb_low_{window}'] = bb.bollinger_lband()
            df[f'bb_position_{window}'] = (
                (df['close'] - bb.bollinger_lband()) /
                (bb.bollinger_hband() - bb.bollinger_lband())
            )
            df[f'bb_width_{window}'] = (
                (bb.bollinger_hband() - bb.bollinger_lband()) /
                bb.bollinger_mavg()
            )

        return self

    def add_multi_timeframe_features(self):
      """
      Ajoute features des timeframes supérieurs
      CRITIQUE pour réduire faux signaux
      """
      df = self.df

      # Pour chaque timeframe supérieur
      for tf_name, tf_df in self.timeframes.items():
          # Calcul features sur ce timeframe
          tf_df = tf_df.copy()  # Important!
          tf_df['return'] = tf_df['close'].pct_change()
          tf_df['sma_20'] = tf_df['close'].rolling(20).mean()
          tf_df['ema_12'] = tf_df['close'].ewm(span=12).mean()
          tf_df['rsi'] = RSIIndicator(tf_df['close'], window=14).rsi()

          # Merge avec données minute (forward fill)
          for col in ['return', 'sma_20', 'ema_12', 'rsi']:
              # Renomme pour éviter conflits
              new_col_name = f'{col}_{tf_name}'

              # Reindex et forward fill
              df[new_col_name] = tf_df[col].reindex(
                  df.index, method='ffill'
              )

      # Tendance multi-timeframe
      df['trend_alignment'] = 0

      # Si toutes les EMAs sont alignées = tendance forte
      ema_cols = [col for col in df.columns if 'ema_12' in col and col != 'ema_12']
      if len(ema_cols) > 0:
          # Bullish : prix > toutes les EMAs
          try:
              bullish = (df['close'].values[:, None] > df[ema_cols].values).all(axis=1)
              bearish = (df['close'].values[:, None] < df[ema_cols].values).all(axis=1)

              df.loc[bullish, 'trend_alignment'] = 1
              df.loc[bearish, 'trend_alignment'] = -1
          except:
              # Si erreur, on skip cette partie
              pass

      self.df = df
      return self

    def add_time_features_minute(self):
        """Features temporelles spécifiques trading"""
        df = self.df

        # Heure et jour
        df['hour'] = df.index.hour
        df['minute'] = df.index.minute
        df['day_of_week'] = df.index.dayofweek

        # Sessions de trading (UTC - ajuste selon ta timezone)
        df['session'] = 'other'

        # Session asiatique (00:00-08:00 UTC)
        df.loc[df['hour'].between(0, 8), 'session'] = 'asian'

        # Session européenne (08:00-16:00 UTC)
        df.loc[df['hour'].between(8, 16), 'session'] = 'european'

        # Session américaine (16:00-00:00 UTC)
        df.loc[df['hour'].between(16, 23), 'session'] = 'american'

        # One-hot encoding
        df = pd.get_dummies(df, columns=['session'], prefix='sess')

        # Weekend vs weekday (crypto trade 24/7 mais patterns différents)
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

        # Début/fin d'heure (souvent plus volatil)
        df['is_hour_start'] = (df['minute'] < 5).astype(int)
        df['is_hour_end'] = (df['minute'] >= 55).astype(int)

        return self

    def add_order_flow_approximation(self):
        """
        Approximation de l'order flow
        (sans accès au carnet d'ordres réel)
        """
        df = self.df

        # Volume-weighted price
        df['vwap_60'] = (
            (df['close'] * df['volume']).rolling(60).sum() /
            df['volume'].rolling(60).sum()
        )

        # Distance au VWAP
        df['distance_vwap'] = (df['close'] - df['vwap_60']) / df['vwap_60']

        # Accumulation/Distribution approximation
        mf_multiplier = (
            (df['close'] - df['low']) - (df['high'] - df['close'])
        ) / (df['high'] - df['low'] + 1e-10)

        mf_volume = mf_multiplier * df['volume']
        df['money_flow_20'] = mf_volume.rolling(20).sum()

        # Buy/Sell pressure (approximation)
        # Si close > open = achat, sinon vente
        df['buy_volume'] = df['volume'] * (df['close'] > df['open']).astype(int)
        df['sell_volume'] = df['volume'] * (df['close'] <= df['open']).astype(int)

        df['buy_sell_ratio_60'] = (
            df['buy_volume'].rolling(60).sum() /
            (df['sell_volume'].rolling(60).sum() + 1e-10)
        )

        return self

    def reduce_noise_with_smoothing(self):
        """
        Lisse les features bruitées
        Important pour données minute
        """
        df = self.df

        # Liste des features à lisser
        features_to_smooth = [
            'rsi_14', 'macd', 'volume_spike',
            'distance_vwap', 'bb_position_20'
        ]

        for feature in features_to_smooth:
            if feature in df.columns:
                # EMA pour lisser
                df[f'{feature}_smooth'] = df[feature].ewm(span=5).mean()

        return self

    def create_target_minute(self, forward_minutes=15, threshold=0.001):
        """
        Crée la target pour données minute

        forward_minutes : horizon de prédiction (15min = raisonnable)
        threshold : seuil minimal de mouvement (0.1% = raisonnable pour minute)
        """
        df = self.df

        # Return futur
        future_return = (
            df['close'].shift(-forward_minutes) / df['close'] - 1
        )

        # Classification avec seuil
        df['target'] = 0  # Flat
        df.loc[future_return > threshold, 'target'] = 1   # Up
        df.loc[future_return < -threshold, 'target'] = -1  # Down

        # Return futur pour analyse
        df['future_return'] = future_return

        # IMPORTANT : Distribution de la target
        print(f"\n📊 Target Distribution (forward={forward_minutes}min, threshold={threshold*100}%):")
        print(df['target'].value_counts(normalize=True).sort_index())

        return self

    def downsample_for_training(self, frequency='5T'):
        """
        Optionnel : Réduit la fréquence pour l'entraînement
        Recommandé si trop de données
        """
        df = self.df

        # Garde une minute sur N
        if frequency == '5T':
            mask = df.index.minute % 5 == 0
        elif frequency == '15T':
            mask = df.index.minute % 15 == 0
        elif frequency == '1H':
            mask = df.index.minute == 0
        else:
            mask = df.index == df.index  # Garde tout

        df_downsampled = df[mask].copy()

        print(f"\n⬇️ Downsampling: {len(df)} → {len(df_downsampled)} rows")

        return df_downsampled

    def build_all_features(self, downsample='5T'):
        """
        Pipeline complet pour données minute
        """
        print("🔧 Building features for minute data...")

        # 1. Resample multi-timeframe
        print("   → Multi-timeframe resampling...")
        self.resample_to_multiple_timeframes()

        # 2. Features minute
        print("   → Minute-specific features...")
        self.add_minute_specific_features()

        # 3. Momentum court terme
        print("   → Short-term momentum...")
        self.add_short_term_momentum()

        # 4. Volatilité
        print("   → Volatility features...")
        self.add_volatility_features_minute()

        # 5. Multi-timeframe
        print("   → Multi-timeframe features...")
        self.add_multi_timeframe_features()

        # 6. Time features
        print("   → Time features...")
        self.add_time_features_minute()

        # 7. Order flow
        print("   → Order flow approximation...")
        self.add_order_flow_approximation()

        # 8. Smoothing
        print("   → Noise reduction...")
        self.reduce_noise_with_smoothing()

        # 9. Target
        print("   → Creating target...")
        self.create_target_minute(forward_minutes=15, threshold=0.001)
        # Supprimer uniquement les dernières lignes NaN liées au shift
        self.df = self.df.iloc[:-15]

        # 10. Downsample (optionnel)
        if downsample:
            print(f"   → Downsampling to {downsample}...")
            self.df = self.downsample_for_training(downsample)

        # Nettoyage final
        initial_rows = len(self.df)
        self.df = self.df[self.df['target'].notna()]  # garder seulement target valide
        self.df.fillna(method='ffill', inplace=True)
        self.df.fillna(method='bfill', inplace=True)
        print(f"   → Dropped {initial_rows - len(self.df)} rows with NaN")
        # Mapping des nouveaux noms vers les anciens attendus par ton modèle

        # 👉 Ajout de la colonne 'trades'
        # Proxy simple basé sur le volume
        self.df['trades'] = self.df['volume'] / (self.df['volume_ma_10'] + 1e-10)

        rename_map = {
            'return_5min': 'return_5T',
            'sma_20_5min': 'sma_20_5T',
            'ema_12_5min': 'ema_12_5T',
            'rsi_5min': 'rsi_5T',
            'return_15min': 'return_15T',
            'sma_20_15min': 'sma_20_15T',
            'ema_12_15min': 'ema_12_15T',
            'rsi_15min': 'rsi_15T',
            'return_1h': 'return_1H',
            'sma_20_1h': 'sma_20_1H',
            'ema_12_1h': 'ema_12_1H',
            'rsi_1h': 'rsi_1H',
            'return_4h': 'return_4H',
            'sma_20_4h': 'sma_20_4H',
            'ema_12_4h': 'ema_12_4H',
            'rsi_4h': 'rsi_4H'
        }

        self.df.rename(columns=rename_map, inplace=True)

        print(f"\n✅ Final dataset: {len(self.df)} rows x {len(self.df.columns)} columns")

        return self.df

        
