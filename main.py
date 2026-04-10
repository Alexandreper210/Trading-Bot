"""
 TRADING BOT ML - MULTI-PAIR
================================
Bot de trading automatique utilisant Machine Learning

Usage:
    python main.py train BTCUSDT              # Entraîne 1 paires
    python main.py train all                  # Entraîne tous les paires
    python main.py backtest BTCUSDT           # Backtest 1 paire
    python main.py backtest all               # Backtest toutes les paires
    python main.py live                       # Lance le bot en live
    python main.py analyze BTCUSDT            # Analyse approfondie d'une paire
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
import pickle
import json
from datetime import datetime
from pathlib import Path
import glob

# Imports des modules custom
from features.MinuteFeatureEngineer import MinuteFeatureEngineer
from models.MLPipeline import MLPipeline
from strategies.MLTradingStrategy import MLTradingStrategy
from strategies.RegimeFilter import RegimeFilter
from backtesting.Backtester import Backtester
from trading.MultiPairTradingBot import MultiPairTradingBot
from config import *


# ==========================================
# FONCTIONS UTILITAIRES
# ==========================================

def create_directories():
    """Crée les dossiers nécessaires"""
    dirs = [
        'data/raw',
        'data/processed',
        'data/results',
        'models/saved_models',
        'logs'
    ]
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)


def load_data(symbol):
    """Charge les données d'une paire"""
    # Essaye différents formats de nom de fichier
    possible_paths = [
        f'data/raw/{symbol}_1m.csv',
        f'data/raw/{symbol}_1_min.csv',
        f'data/raw/{symbol}_2024_1_min.csv',
        f'data/raw/{symbol}.csv'
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            print(f" Loading data from: {path}")
            df = pd.read_csv(path)
            
            # Normalise les noms de colonnes
            df.columns = df.columns.str.lower()
            
            # Trouve la colonne de temps
            time_col = None
            for col in ['date', 'timestamp', 'datetime', 'time']:
                if col in df.columns:
                    time_col = col
                    break
            
            if time_col is None:
                print(f"  Warning: No time column found, using index")
            else:
                df[time_col] = pd.to_datetime(df[time_col])
                df.set_index(time_col, inplace=True)
            
            print(f" Data loaded: {len(df)} rows")
            print(f"   Period: {df.index[0]} → {df.index[-1]}")
            print(f"   Columns: {df.columns.tolist()}")
            
            return df
    
    print(f" Data file not found for {symbol}")
    print(f"   Searched in:")
    for path in possible_paths:
        print(f"   - {path}")
    return None


def save_model(pipeline, symbol, best_name):
    """Sauvegarde un modèle entraîné"""
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    model_path = f"{MODEL_DIR}/{symbol}_{best_name}_model.pkl"
    scaler_path = f"{MODEL_DIR}/{symbol}_scaler.pkl"
    features_path = f"{MODEL_DIR}/{symbol}_features.json"
    
    with open(model_path, 'wb') as f:
        pickle.dump(pipeline.best_model, f)
    
    with open(scaler_path, 'wb') as f:
        pickle.dump(pipeline.scaler, f)
    
    with open(features_path, 'w') as f:
        json.dump(pipeline.feature_cols, f, indent=2)
    
    print(f"\n Model saved:")
    print(f"   - {model_path}")
    print(f"   - {scaler_path}")
    print(f"   - {features_path}")


# ==========================================
# PHASE 1 : ENTRAÎNEMENT
# ==========================================

def train_model(symbol, confidence_threshold=0.65):
    """Entraîne un modèle pour une paire"""
    print(f"\n{'='*70}")
    print(f" TRAINING MODEL FOR {symbol}")
    print(f"{'='*70}\n")
    
    # 1. Charger données
    df = load_data(symbol)
    if df is None:
        return False
    
    # 2. Feature Engineering
    print("\n Building features...")
    fe = MinuteFeatureEngineer(df)
    df_features = fe.build_all_features(downsample='5T')
    
    # 3. Sélection des features
    exclude_cols = ['target', 'future_return', 'open', 'high', 'low', 'close', 'volume']
    numeric_cols = df_features.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [col for col in numeric_cols if col not in exclude_cols]
    
    print(f" Features created: {len(feature_cols)} features")
    
    # 4. Préparer le pipeline
    print("\n Preparing ML pipeline...")
    ml_pipeline = MLPipeline(df_features)
    ml_pipeline.prepare_data(feature_cols=feature_cols, test_size=TEST_SIZE)
    
    # 5. Entraîner les modèles
    print("\n Training models...")
    ml_pipeline.train_xgboost()
    ml_pipeline.train_lightgbm()
    ml_pipeline.train_random_forest()
    
    # 6. Évaluation
    print("\n Evaluating models...")
    results = ml_pipeline.evaluate_models()
    
    # 7. Feature importance
    ml_pipeline.plot_feature_importance(top_n=15)
    
    # 8. Sélection du meilleur
    best_name, best_model = ml_pipeline.select_best_model(metric='f1')
    
    # 9. Sauvegarde
    save_model(ml_pipeline, symbol, best_name)
    
    print(f"\nTraining completed for {symbol}!")
    
    return True


def train_all_pairs():
    """Entraîne tous les modèles"""
    print(f"\n{'='*70}")
    print(f"TRAINING ALL MODELS")
    print(f"{'='*70}\n")
    
    success = []
    failed = []
    
    for symbol in TRADING_PAIRS:
        print(f"\n{'─'*70}")
        try:
            if train_model(symbol):
                success.append(symbol)
            else:
                failed.append(symbol)
        except Exception as e:
            print(f" Error training {symbol}: {e}")
            failed.append(symbol)
    
    # Résumé
    print(f"\n{'='*70}")
    print(f" TRAINING SUMMARY")
    print(f"{'='*70}")
    print(f" Success: {len(success)}/{len(TRADING_PAIRS)} models")
    print(f"   {', '.join(success)}")
    
    if failed:
        print(f"\nFailed: {len(failed)} models")
        print(f"   {', '.join(failed)}")


# ==========================================
# PHASE 2 : BACKTEST
# ==========================================




def get_model_path(symbol):
    """Retourne les chemins du modèle, scaler et features pour une paire"""
    MODEL_DIR = "models/saved_models"
    
    # Cherche tous les modèles pour cette paire
    model_files = glob.glob(f"{MODEL_DIR}/{symbol}_*_model.pkl")
    if not model_files:
        raise FileNotFoundError(f"No model found for {symbol} in {MODEL_DIR}")
    
    # Prend le premier fichier trouvé (le meilleur modèle sauvegardé)
    model_path = model_files[0]
    
    scaler_path = f"{MODEL_DIR}/{symbol}_scaler.pkl"
    features_path = f"{MODEL_DIR}/{symbol}_features.json"
    
    return {"model": model_path, "scaler": scaler_path, "features": features_path}

def get_pair_config(symbol):
    """Retourne la configuration de trading pour une paire"""
    # Ici tu peux adapter selon tes paramètres par défaut ou config.py
    config = {
        'confidence_threshold': CONFIDENCE_THRESHOLD,  # depuis config.py
        'position_size': POSITION_SIZE,                # depuis config.py
        'stop_loss': STOP_LOSS                         # depuis config.py
    }
    return config

def get_capital_for_pair(symbol, total_capital):
    """Retourne le capital alloué pour une paire"""
    # Option 1 : capital fixe par paire
    return total_capital

    # Option 2 : si tu veux répartir le capital sur toutes les paires :
    # return total_capital / len(TRADING_PAIRS)


def backtest_model(symbol, show_plots=True):
    """Backtest pour une paire"""
    print(f"\n{'='*70}")
    print(f" BACKTESTING {symbol}")
    print(f"{'='*70}\n")
    
    # 1. Charger données
    df = load_data(symbol)
    if df is None:
        return None
    
    # 2. Features
    print("Building features...")
    fe = MinuteFeatureEngineer(df)
    df_features = fe.build_all_features(downsample='5T')
    
    # 3. Charger le modèle
    model_paths = get_model_path(symbol)
    
    if not os.path.exists(model_paths['model']):
        print(f"Model not found for {symbol}")
        print(f"   Run: python main.py train {symbol}")
        return None
    
    print(" Loading model...")
    with open(model_paths['model'], 'rb') as f:
        model = pickle.load(f)
    
    with open(model_paths['scaler'], 'rb') as f:
        scaler = pickle.load(f)
    
    with open(model_paths['features'], 'r') as f:
        feature_cols = json.load(f)
    
    # 4. Split train/test
    split_idx = int(len(df_features) * 0.8)
    test_df = df_features.iloc[split_idx:]
    
    # 5. Stratégie
    print(" Generating signals...")
    pair_config = get_pair_config(symbol)
    
    strategy = MLTradingStrategy(
        model=model,
        scaler=scaler,
        feature_cols=feature_cols,
        confidence_threshold=pair_config['confidence_threshold']
    )
    
    df_signals = strategy.generate_signals(test_df)
    df_signals = strategy.add_position_sizing(df_signals, base_size=pair_config['position_size'])
    df_signals = strategy.add_risk_management(df_signals, max_loss_pct=pair_config['stop_loss'])
    
    # 6. Filtres de régime
    print(" Applying regime filters...")
    regime = RegimeFilter(test_df)
    regime.detect_trend_regime()
    regime.detect_volatility_regime()
    df_signals = regime.apply_regime_rules(df_signals)
    
    # 7. Backtest
    print("  Running backtest...")
    bt = Backtester(
        initial_capital=get_capital_for_pair(symbol, INITIAL_CAPITAL),
        commission=COMMISSION,
        slippage=SLIPPAGE
    )
    bt.run(df_signals)
    
    # 8. Résultats
    metrics = bt.print_metrics()
    
    if show_plots:
        bt.plot_results()
    
    # 9. Sauvegarder
    results_dir = f'data/results/{symbol}'
    os.makedirs(results_dir, exist_ok=True)
    bt.export_trades(filename=f'{symbol}_trades.csv')
    
    return metrics


def backtest_all_pairs():
    """Backtest sur toutes les paires"""
    print(f"\n{'='*70}")
    print(f" BACKTESTING ALL PAIRS")
    print(f"{'='*70}\n")
    
    all_metrics = {}
    
    for symbol in TRADING_PAIRS:
        try:
            metrics = backtest_model(symbol, show_plots=False)
            if metrics:
                all_metrics[symbol] = metrics
        except Exception as e:
            print(f" Error backtesting {symbol}: {e}")
    
    # Résumé comparatif
    print(f"\n{'='*70}")
    print(f" COMPARATIVE SUMMARY")
    print(f"{'='*70}\n")
    
    if all_metrics:
        summary_df = pd.DataFrame(all_metrics).T
        summary_df = summary_df.sort_values(by='Sharpe Ratio (Period)', ascending=False)
        
        print(summary_df[['Total Return (%)', 'Sharpe Ratio (Period)', 'Sharpe Ratio (Annualized)', 'Max Drawdown (%)', 
                         'Win Rate (%)', 'Profit Factor']])
        
        # Sauvegarde
        summary_df.to_csv('data/results/summary_all_pairs.csv')
        print(f"\n Summary saved to: data/results/summary_all_pairs.csv")
        
        # Top 3
        print(f"\n TOP 3 PAIRS:")
        for i, (pair, row) in enumerate(summary_df.head(3).iterrows(), 1):
            print(f"   {i}. {pair:12s} | Sharpe: {row['Sharpe Ratio (Period)']:>6.2f} | Return: {row['Total Return (%)']:>8.2f}%")


# ==========================================
# PHASE 3 : ANALYSE APPROFONDIE
# ==========================================

def analyze_pair(symbol):
    """Analyse approfondie d'une paire"""
    print(f"\n{'='*70}")
    print(f" DEEP ANALYSIS OF {symbol}")
    print(f"{'='*70}\n")
    
    # 1. Charger données
    df = load_data(symbol)
    if df is None:
        return
    
    # 2. Features
    print("Building features...")
    fe = MinuteFeatureEngineer(df)
    df_features = fe.build_all_features(downsample='5T')
    
    # 3. Analyse de la target
    print("\n TARGET ANALYSIS:")
    print(df_features['target'].value_counts())
    print(f"\nTarget balance:")
    print(df_features['target'].value_counts(normalize=True) * 100)
    
    # 4. Corrélations
    exclude_cols = ['target', 'future_return', 'open', 'high', 'low', 'close', 'volume']
    numeric_cols = df_features.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [col for col in numeric_cols if col not in exclude_cols]
    
    correlations = df_features[feature_cols + ['target']].corr()['target'].sort_values(ascending=False)
    
    print(f"\nTOP 10 CORRELATIONS:")
    print(correlations.head(10))
    
    print(f"\nBOTTOM 10 CORRELATIONS:")
    print(correlations.tail(10))
    
    # 5. Visualisations
    fig, axes = plt.subplots(4, 1, figsize=(15, 12))
    
    # Prix
    axes[0].plot(df_features.index[-1000:], df_features['close'][-1000:])
    axes[0].set_title(f'{symbol} - Price (last 1000 samples)')
    axes[0].grid(alpha=0.3)
    
    # Target par heure
    target_by_hour = df_features.groupby('hour')['target'].mean()
    axes[1].bar(target_by_hour.index, target_by_hour.values)
    axes[1].set_title('Average Target by Hour')
    axes[1].set_xlabel('Hour of Day')
    axes[1].grid(alpha=0.3)
    
    # Volatilité
    if 'volatility_60m' in df_features.columns:
        axes[2].plot(df_features.index[-1000:], df_features['volatility_60m'][-1000:])
        axes[2].set_title('60-min Volatility')
        axes[2].grid(alpha=0.3)
    
    # Top features correlation
    top_features = correlations.abs().sort_values(ascending=False).head(15)
    colors = ['green' if x > 0 else 'red' for x in correlations[top_features.index]]
    axes[3].barh(range(len(top_features)), correlations[top_features.index], color=colors)
    axes[3].set_yticks(range(len(top_features)))
    axes[3].set_yticklabels(top_features.index, fontsize=8)
    axes[3].set_xlabel('Correlation')
    axes[3].set_title('Top 15 Features Correlation')
    axes[3].axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    axes[3].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'data/results/{symbol}_analysis.png', dpi=300)
    plt.show()
    
    print(f"\n💾 Analysis saved to: data/results/{symbol}_analysis.png")


# ==========================================
# PHASE 4 : TRADING LIVE
# ==========================================

def run_live_bot():
    """Lance le bot en live"""
    print(f"\n{'='*70}")
    print(f" STARTING LIVE TRADING BOT")
    print(f"{'='*70}\n")
    
    # Vérifier que les modèles existent
    missing = []
    available = []
    
    for pair in TRADING_PAIRS:
        try:
            model_path = get_model_path(pair)['model']
            if os.path.exists(model_path):
                available.append(pair)
            else:
                missing.append(pair)
        except FileNotFoundError:
            missing.append(pair)
    
    # Affichage des résultats
    print(f"\n✅ Models loaded for {len(available)} pairs.")
    if available:
        print(f"   {', '.join(available)}")
    
    if missing:
        print(f"\n⚠️ WARNING: Missing models for {len(missing)} pairs:")
        print(f"   {', '.join(missing)}")
        print(f"\nTrain them first:")
        print(f"   python main.py train all")
        
        response = input("\nContinue anyway? (y/n): ")
        if response.lower() != 'y':
            return


    # Créer et lancer le bot
    bot = MultiPairTradingBot(
        api_key=BINANCE_API_KEY,
        api_secret=BINANCE_API_SECRET,
        testnet=True  # IMPORTANT: True pour testnet
    )
    
    # Lance en boucle
    bot.run_loop(interval_seconds=900)  # 15 minutes


def run_realtime_scanner(interval=60):  # ← Ajoutez le paramètre interval
    """Lance le scanner temps réel"""
    from trading.RealTimeScanner import RealTimeScanner
    
    print(f"\n{'='*70}")
    print(f"🔍 STARTING REAL-TIME SCANNER")
    print(f"{'='*70}\n")
    
    scanner = RealTimeScanner(
        api_key=BINANCE_API_KEY,
        api_secret=BINANCE_API_SECRET,
        testnet=True,
        update_interval=interval  # ← Utilisez le paramètre
    )
    
    scanner.run_loop()
# ==========================================
# MENU PRINCIPAL
# ==========================================
def print_help():
    print("""
TRADING BOT ML - Multi-Pair
================================

Commands:
    train <pair|all>       Train model(s)
    backtest <pair|all>    Run backtest(s)
    analyze <pair>         Deep analysis of a pair
    
    scan [interval]        Real-time scanner (1 position)
    live                   Multi-pair bot (up to 3 positions)
    
    help                   Show this help

Examples:
    # Training
    python main.py train BTCUSDT
    python main.py train all
    
    # Backtesting
    python main.py backtest ETHUSDT
    python main.py backtest all
    
    # Live Trading
    python main.py scan              # Scan every 60 seconds (recommended)
    python main.py scan 30           # Scan every 30 seconds
    python main.py live              # Multi-position bot (15 min intervals)

Trading Strategies:
    scan - Monitors all pairs in real-time, trades 1 position on best signal
    live - Trades multiple positions (max 3), one per pair with signal
    """)


def main():
    """Point d'entrée principal"""
    
    # Créer dossiers
    create_directories()
    
    # Parser arguments
    if len(sys.argv) < 2:
        print_help()
        return
    
    command = sys.argv[1].lower()
    
    # Exécuter commande
    if command == 'train':
        if len(sys.argv) < 3:
            print(" Missing argument: specify 'all' or a pair name")
            return
        
        target = sys.argv[2].upper()
        
        if target == 'ALL':
            train_all_pairs()
        else:
            train_model(target)
    
    elif command == 'backtest':
        if len(sys.argv) < 3:
            print("Missing argument: specify 'all' or a pair name")
            return
        
        target = sys.argv[2].upper()
        
        if target == 'ALL':
            backtest_all_pairs()
        else:
            backtest_model(target)
    
    elif command == 'analyze':
        if len(sys.argv) < 3:
            print(" Missing argument: specify a pair name")
            return
        
        target = sys.argv[2].upper()
        analyze_pair(target)
    
    elif command == 'scan':
        interval = int(sys.argv[2]) if len(sys.argv) > 2 else 60
        run_realtime_scanner(interval=interval)
    
    elif command == 'live':
        run_live_bot()
    
    elif command == 'help':
        print_help()
    
    else:
        print(f" Unknown command: {command}")
        print_help()


if __name__ == "__main__":
    main()



