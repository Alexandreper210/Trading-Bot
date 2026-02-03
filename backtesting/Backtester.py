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

class Backtester:
    def __init__(self, initial_capital=10000,
                 commission=0.001,  # 0.1% fees
                 slippage=0.0005):   # 0.05% slippage
        self.initial_capital = float(initial_capital)
        self.commission = commission
        self.slippage = slippage

    def run(self, df):

      df = df.copy()

      # Initialisation avec types FLOAT explicites
      df['capital'] = float(self.initial_capital)
      df['holdings'] = 0.0
      df['cash'] = float(self.initial_capital)
      df['total'] = float(self.initial_capital)
      df['returns'] = 0.0
      df['trade'] = False
      df['position_held'] = 0.0  # Nombre d'unités détenues

      cash = float(self.initial_capital)
      position = 0.0  # Unités détenues
      entry_price = 0.0

      for i in range(1, len(df)):
          prev_signal = df.iloc[i-1]['signal']
          curr_signal = df.iloc[i]['signal']
          price = float(df.iloc[i]['close'])

          # Normaliser les signaux : 1 = BUY, 0 ou -1 = SELL
          normalized_signal = 1 if curr_signal == 1 else 0
          prev_normalized = 1 if prev_signal == 1 else 0

          # Détection d'un changement de signal
          if prev_normalized != normalized_signal:
              df.iloc[i, df.columns.get_loc('trade')] = True

              # CAS 1 : VENDRE - On a une position et signal != 1
              if position > 0 and normalized_signal == 0:
                  sell_price = price * (1 - self.slippage)               # slippage à la vente
                  proceeds = position * sell_price * (1 - self.commission)  # commission
                  cash += proceeds
                  position = 0.0
                  entry_price = 0.0

              # CAS 2 : ACHETER - Pas de position et signal == 1
              elif position == 0 and normalized_signal == 1:
                  position_size = min(abs(float(df.iloc[i]['position'])), 0.95)  # max 95% du cash
                  if position_size > 0 and cash > 0:
                      buy_price = price * (1 + self.slippage)                 # slippage à l'achat
                      cash_to_invest = cash * position_size
                      cost = cash_to_invest * (1 + self.commission)          # commission incluse
                      if cost <= cash:
                          position = cash_to_invest / buy_price              # nombre d'unités achetées
                          cash -= cost
                          entry_price = buy_price

          # Valeur du portfolio
          holdings_value = position * price
          portfolio_value = cash + holdings_value

          # Mise à jour des colonnes
          df.iloc[i, df.columns.get_loc('position_held')] = float(position)
          df.iloc[i, df.columns.get_loc('cash')] = float(cash)
          df.iloc[i, df.columns.get_loc('holdings')] = float(holdings_value)
          df.iloc[i, df.columns.get_loc('total')] = float(portfolio_value)
          df.iloc[i, df.columns.get_loc('capital')] = float(portfolio_value)

      # Returns
      df['returns'] = df['total'].pct_change()
      df['cumulative_returns'] = (1 + df['returns']).cumprod()

      # Buy & Hold benchmark
      df['bh_returns'] = df['close'].pct_change()
      df['bh_cumulative'] = (1 + df['bh_returns']).cumprod()

      self.results = df
      return self


    def calculate_metrics(self):
        df = self.results

        # Returns
        final_total = float(df['total'].iloc[-1])
        initial_total = float(self.initial_capital)
        total_return = (final_total / initial_total - 1) * 100

        first_close = float(df['close'].iloc[0])
        last_close = float(df['close'].iloc[-1])
        bh_return = (last_close / first_close - 1) * 100

        # Durée de la période
        delta = df.index[-1] - df.index[0]
        days = delta.total_seconds() / 86400  # plus précis que .days

        if days > 0:
            annual_return = ((final_total / initial_total) ** (365 / days) - 1) * 100
        else:
            annual_return = 0

        # Fréquence des données
        freq_seconds = (df.index[1] - df.index[0]).total_seconds()
        periods_per_year = 365 * 24 * 3600 / freq_seconds  # nb de points/an selon la fréquence

        # Volatilité annualisée
        returns_clean = df['returns'].replace([np.inf, -np.inf], np.nan).dropna()
        returns_std = float(returns_clean.std())
        annual_vol = returns_std * np.sqrt(periods_per_year) * 100

        # Sharpe ratios
        sharpe_annualized = (annual_return / annual_vol) if annual_vol != 0 else 0
        sharpe_period = (returns_clean.mean() / returns_clean.std()) if returns_clean.std() != 0 else 0

        # Max Drawdown
        cumulative = df['total']
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max * 100
        max_drawdown = float(drawdown.min())

        # Analyse des trades
        trades_df = df[df['trade'] == True].copy()
        if len(trades_df) > 2:
            trades_df['trade_pnl'] = trades_df['total'].diff()
            trades_df['trade_return'] = trades_df['total'].pct_change() * 100
            trade_returns = trades_df['trade_return'].dropna()

            wins = trade_returns[trade_returns > 0]
            losses = trade_returns[trade_returns < 0]

            win_rate = float(len(wins) / len(trade_returns) * 100) if len(trade_returns) > 0 else 0
            avg_win = float(wins.mean()) if len(wins) > 0 else 0
            avg_loss = float(losses.mean()) if len(losses) > 0 else 0

            total_wins = float(wins.sum()) if len(wins) > 0 else 0
            total_losses = abs(float(losses.sum())) if len(losses) > 0 else 0
            profit_factor = (total_wins / total_losses) if total_losses > 0 else 0
        else:
            win_rate = avg_win = avg_loss = profit_factor = 0

        # Calmar Ratio
        calmar = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0

        # Nombre de trades
        num_trades = int(df['trade'].sum())

        metrics = {
            'Total Return (%)': round(total_return, 2),
            'Buy & Hold Return (%)': round(bh_return, 2),
            'Annual Return (%)': round(annual_return, 2),
            'Annual Volatility (%)': round(annual_vol, 2),
            'Sharpe Ratio (Annualized)': round(sharpe_annualized, 2),
            'Sharpe Ratio (Period)': round(sharpe_period, 2),
            'Max Drawdown (%)': round(max_drawdown, 2),
            'Calmar Ratio': round(calmar, 2),
            'Number of Trades': num_trades,
            'Win Rate (%)': round(win_rate, 2),
            'Avg Win (%)': round(avg_win, 4),
            'Avg Loss (%)': round(avg_loss, 4),
            'Profit Factor': round(profit_factor, 2)
        }

        return metrics



    def plot_results(self):
        """Visualise les résultats"""
        df = self.results

        fig, axes = plt.subplots(4, 1, figsize=(14, 12))

        # 1. Equity curve
        axes[0].plot(df.index, df['total'], label='Strategy', linewidth=2)
        axes[0].plot(df.index,
                     df['bh_cumulative'] * self.initial_capital,
                     label='Buy & Hold', alpha=0.7)
        axes[0].axhline(y=self.initial_capital, color='gray', linestyle='--', alpha=0.5)
        axes[0].set_title('Equity Curve', fontsize=14, fontweight='bold')
        axes[0].set_ylabel('Portfolio Value ($)')
        axes[0].legend()
        axes[0].grid(alpha=0.3)

        # 2. Drawdown
        cumulative = df['total']
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max * 100

        axes[1].fill_between(df.index, drawdown, 0, color='red', alpha=0.3)
        axes[1].plot(df.index, drawdown, color='red', linewidth=1)
        axes[1].set_title('Drawdown', fontsize=14, fontweight='bold')
        axes[1].set_ylabel('Drawdown (%)')
        axes[1].grid(alpha=0.3)

        # 3. Signals et position
        axes[2].plot(df.index, df['close'], label='Price', alpha=0.7, linewidth=1)

        # Colorer les zones où on a une position
        in_position = df['position_held'] > 0
        axes[2].fill_between(df.index, df['close'].min(), df['close'].max(),
                            where=in_position, alpha=0.1, color='green', label='In Position')

        buy_signals = df[df['trade'] == True][df['signal'] == 1]
        sell_signals = df[df['trade'] == True][df['signal'] != 1]

        axes[2].scatter(buy_signals.index, buy_signals['close'],
                       color='green', marker='^', s=100, label='Buy', alpha=0.7, zorder=5)
        axes[2].scatter(sell_signals.index, sell_signals['close'],
                       color='red', marker='v', s=100, label='Sell', alpha=0.7, zorder=5)

        axes[2].set_title('Trading Signals & Position', fontsize=14, fontweight='bold')
        axes[2].set_ylabel('Price ($)')
        axes[2].legend()
        axes[2].grid(alpha=0.3)

        # 4. Returns distribution
        returns_clean = df['returns'].replace([np.inf, -np.inf], np.nan).dropna()
        axes[3].hist(returns_clean * 100, bins=50,
                    alpha=0.7, edgecolor='black')
        axes[3].axvline(returns_clean.mean() * 100,
                       color='red', linestyle='--', linewidth=2, label='Mean')
        axes[3].set_title('Returns Distribution', fontsize=14, fontweight='bold')
        axes[3].set_xlabel('Returns (%)')
        axes[3].set_ylabel('Frequency')
        axes[3].legend()
        axes[3].grid(alpha=0.3)

        plt.tight_layout()

        plt.show()

        print("\n📊 Backtest visualization saved to 'data/processed/backtest_results.png'")

    def print_metrics(self):
        """Affiche les métriques de manière formatée"""
        metrics = self.calculate_metrics()

        print("\n" + "="*60)
        print("📊 BACKTEST RESULTS")
        print("="*60)

        print("\n💰 RETURNS:")
        print(f"   Total Return:           {metrics['Total Return (%)']:>10.2f}%")
        print(f"   Buy & Hold Return:      {metrics['Buy & Hold Return (%)']:>10.2f}%")
        print(f"   Annual Return:          {metrics['Annual Return (%)']:>10.2f}%")

        print("\n📈 RISK METRICS:")
        print(f"   Annual Volatility:      {metrics['Annual Volatility (%)']:>10.2f}%")
        print(f"   Max Drawdown:           {metrics['Max Drawdown (%)']:>10.2f}%")
        # Affichage des deux Sharpe
        print(f"   Sharpe Ratio (Annual):  {metrics.get('Sharpe Ratio (Annualized)', 0):>10.2f}")
        print(f"   Sharpe Ratio (Period):  {metrics.get('Sharpe Ratio (Period)', 0):>10.2f}")
        print(f"   Calmar Ratio:           {metrics['Calmar Ratio']:>10.2f}")

        print("\n🎯 TRADING METRICS:")
        print(f"   Number of Trades:       {metrics['Number of Trades']:>10}")
        print(f"   Win Rate:               {metrics['Win Rate (%)']:>10.2f}%")
        print(f"   Avg Win:                {metrics['Avg Win (%)']:>10.4f}%")
        print(f"   Avg Loss:               {metrics['Avg Loss (%)']:>10.4f}%")
        print(f"   Profit Factor:          {metrics['Profit Factor']:>10.2f}")

        print("\n" + "="*60)

        return metrics

    def export_trades(self, filename='trades.csv'):
        """Export la liste des trades"""
        df = self.results
        trades = df[df['trade'] == True].copy()

        if len(trades) > 0:
            trades_export = trades[['close', 'signal', 'position_held', 'cash',
                                    'holdings', 'total', 'returns']].copy()

            trades_export.to_csv(f'data/processed/{filename}')
            print(f"\n💾 Trades exported to 'data/processed/{filename}'")

            return trades_export
        else:
            print("\n⚠️ No trades to export")
            return pd.DataFrame()
