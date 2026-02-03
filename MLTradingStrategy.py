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


class MLTradingStrategy:
    def __init__(self, model, scaler, feature_cols,
                 confidence_threshold=0.6):
        self.model = model
        self.scaler = scaler
        self.feature_cols = feature_cols
        self.confidence_threshold = confidence_threshold

    def generate_signals(self, df):
        """
        Génère les signaux de trading
        Returns: df avec colonnes 'signal' et 'confidence'
        """
        df = df.copy()

        # Préparer les features
        X = df[self.feature_cols]
        X_scaled = self.scaler.transform(X)

        # Prédictions
        predictions = self.model.predict(X_scaled) - 1  # -1, 0, 1
        probabilities = self.model.predict_proba(X_scaled)

        # Confidence = probabilité de la classe prédite
        confidence = probabilities.max(axis=1)

        # Signal final : seulement si confidence > threshold
        df['raw_signal'] = predictions
        df['confidence'] = confidence
        df['signal'] = 0

        # Applique le signal uniquement si confiance suffisante
        mask_confident = confidence >= self.confidence_threshold
        df.loc[mask_confident, 'signal'] = df.loc[mask_confident, 'raw_signal']

        return df

    def add_position_sizing(self, df, base_size=1.0, use_volatility=True):
        """
        Ajuste la taille de position selon la volatilité
        """
        df = df.copy()

        if use_volatility and 'volatility_20' in df.columns:
            # Inverse de la volatilité (vol haute = position petite)
            vol_factor = 1 / (1 + df['volatility_20'])
            vol_factor = vol_factor / vol_factor.mean()  # Normalise
        else:
            vol_factor = 1.0

        # Position = signal * base_size * vol_factor * confidence
        df['position'] = (
            df['signal'] * base_size * vol_factor * df['confidence']
        )

        return df

    def add_risk_management(self, df, max_loss_pct=0.02,
                           use_atr_stop=True):
        """
        Ajoute stop-loss et take-profit
        """
        df = df.copy()

        if use_atr_stop and 'atr' in df.columns:
            # Stop-loss basé sur ATR
            df['stop_loss'] = df['close'] - 2 * df['atr']
            df['take_profit'] = df['close'] + 3 * df['atr']
        else:
            # Stop-loss fixe
            df['stop_loss'] = df['close'] * (1 - max_loss_pct)
            df['take_profit'] = df['close'] * (1 + max_loss_pct * 1.5)

        return df
