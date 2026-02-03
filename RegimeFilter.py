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


class RegimeFilter:
    """Filtre les trades selon le régime de marché """

    def __init__(self, df):
        self.df = df.copy()  # Copie pour éviter SettingWithCopyWarning

        # Créer 'returns' si elle n'existe pas
        if 'returns' not in self.df.columns:
            self.df['returns'] = self.df['close'].pct_change()

    def detect_trend_regime(self, short_window=50, long_window=200):
        """Trend following : bull/bear/flat, calculé en mode online"""
        self.df['regime_trend'] = 'flat'

        # SMA calculées uniquement avec les données passées
        self.df['sma_short'] = self.df['close'].rolling(short_window, min_periods=1).mean()
        self.df['sma_long'] = self.df['close'].rolling(long_window, min_periods=1).mean()

        # Bull : SMA court > SMA long * 1.02
        self.df.loc[self.df['sma_short'] > self.df['sma_long'] * 1.02, 'regime_trend'] = 'bull'
        # Bear : SMA court < SMA long * 0.98
        self.df.loc[self.df['sma_short'] < self.df['sma_long'] * 0.98, 'regime_trend'] = 'bear'

        return self

    def detect_volatility_regime(self, window=20, threshold=0.5):
        """High vol / Low vol, calculé en mode online"""
        vol = self.df['returns'].rolling(window, min_periods=1).std()
        # rolling median avec uniquement les données passées
        vol_median = vol.rolling(window, min_periods=1).median()

        self.df['regime_vol'] = 'normal'
        self.df.loc[vol > vol_median * (1 + threshold), 'regime_vol'] = 'high'
        self.df.loc[vol < vol_median * (1 - threshold), 'regime_vol'] = 'low'

        return self

    def apply_regime_rules(self, df_with_signals):
        """Ajuste les signaux selon le régime (safe pour backtest)"""
        df = df_with_signals.copy()

        # S'assurer que les colonnes de régime existent
        if 'regime_vol' not in df.columns:
            df['regime_vol'] = 'normal'
        if 'regime_trend' not in df.columns:
            df['regime_trend'] = 'flat'

        # 1. Pas de trade en haute volatilité
        df.loc[df['regime_vol'] == 'high', 'signal'] = 0

        # 2. Longs uniquement en bull market
        mask_bear = df['regime_trend'] == 'bear'
        df.loc[mask_bear & (df['signal'] == 1), 'signal'] = 0

        # 3. Shorts uniquement en bear market
        mask_bull = df['regime_trend'] == 'bull'
        df.loc[mask_bull & (df['signal'] == -1), 'signal'] = 0

        return df