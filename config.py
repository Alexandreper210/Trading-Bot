BINANCE_API_KEY = "QWdhpbLRxoqQ52lW3o9NzhRjuCYfZehJBuuNZyFt1u5arjT2hatzY71aGzFkOG6k"
BINANCE_API_SECRET = "SxwncfNxc29calMoQH8TBvDZJvyetgYDuZIZHc7Y1Uqwse0MgXQs4xjDzAim4f6y"


# config.py - Configuration optimisée pour le Scanner

import glob
import os


# ==================== PAIRES À SCANNER ====================
TRADING_PAIRS = [
    "LISTAUSDT", "SCRUSDT", "DASHUSDT", "DEGOUSDT", "HFTUSDT",
    "ARKMUSDT", "LDOUSDT", "WIFUSDT", "SUSHIUSDT", "CTKUSDT",
    "ATOMUSDT", "QTUMUSDT", "FISUSDT", "ADAUSDT", "BTCUSDT"
]

# ==================== SCANNER TEMPS RÉEL ====================
# Fréquence de mise à jour
SCANNER_UPDATE_INTERVAL = 60  # Secondes (60 = 1 minute)
                              # Options: 1, 5, 10, 30, 60, 120

# Données utilisées
SCANNER_USE_1MIN_CANDLES = True   # Utilise des bougies 1 minute
SCANNER_LOOKBACK_CANDLES = 200    # Nombre de bougies à récupérer

# Seuils de trading
SCANNER_MIN_CONFIDENCE = 0.6     # Confiance minimale pour ENTRER (75%)
SCANNER_EXIT_CONFIDENCE = 0.60    # Seuil pour SORTIR si signal inverse

# Position sizing
SCANNER_POSITION_SIZE = 0.10      # 10% du capital disponible
SCANNER_MAX_POSITION_VALUE = 100000 # Max 1000 USDT par position

# Stop Loss / Take Profit
SCANNER_STOP_LOSS = 0.02          # 2% stop loss
SCANNER_TAKE_PROFIT = 0.10        # 10% take profit
SCANNER_USE_TRAILING_STOP = False # Trailing stop (à implémenter)
SCANNER_TRAILING_STOP_PERCENT = 0.015  # 1.5% trailing

'''# Filtres additionnels
SCANNER_MIN_VOLUME_24H = 100000   # Volume min 24h en USDT (filtre paires mortes)
SCANNER_MIN_PRICE = 0.01          # Prix minimum (évite shitcoins à 0.0001$)
SCANNER_MAX_PRICE = 100000        # Prix maximum
'''
# Mode de trading
SCANNER_ALLOW_SHORT = True       # Autorise les positions SHORT (False = LONG seulement)
SCANNER_ONE_POSITION_ONLY = True  # Une seule position à la fois

# Affichage
SCANNER_SHOW_TOP_N = 5            # Affiche les N meilleurs signaux
SCANNER_VERBOSE = True            # Affichage détaillé



# Configuration par défaut (utilisée si paire non listée ci-dessus)
DEFAULT_CONFIG = {
    "position_size": SCANNER_POSITION_SIZE,
    "stop_loss": SCANNER_STOP_LOSS,
    "take_profit": SCANNER_TAKE_PROFIT,
    "confidence_threshold": SCANNER_MIN_CONFIDENCE,
    "min_confidence": SCANNER_MIN_CONFIDENCE
}

# ==================== MODÈLES ML ====================
MODEL_DIR = "models/saved_models"
TEST_SIZE = 0.2
USE_PAIR_SPECIFIC_MODELS = True

# ==================== BACKTESTING ====================
INITIAL_CAPITAL = 100000
COMMISSION = 0.001
SLIPPAGE = 0.0005

# ==================== LEGACY (pour compatibilité) ====================
INTERVAL = "15m"
POSITION_SIZE = 0.10
STOP_LOSS = 0.02
TAKE_PROFIT = 0.10
CONFIDENCE_THRESHOLD = 0.83

# ==================== HELPER FUNCTIONS ====================

def get_pair_config(symbol):
    """Retourne la config par défaut pour toutes les paires"""
    return {
        "position_size": SCANNER_POSITION_SIZE,
        "stop_loss": SCANNER_STOP_LOSS,
        "take_profit": SCANNER_TAKE_PROFIT,
        "confidence_threshold": SCANNER_MIN_CONFIDENCE,
        "min_confidence": SCANNER_MIN_CONFIDENCE
    }

def get_model_path(symbol):
    """Retourne les chemins du modèle"""
    if USE_PAIR_SPECIFIC_MODELS:
        model_pattern = f"{MODEL_DIR}/{symbol}_*_model.pkl"
        matching_models = glob.glob(model_pattern)
        
        if matching_models:
            model_file = matching_models[0]
        else:
            model_file = f"{MODEL_DIR}/{symbol}_model.pkl"
        
        return {
            'model': model_file,
            'scaler': f"{MODEL_DIR}/{symbol}_scaler.pkl",
            'features': f"{MODEL_DIR}/{symbol}_features.json"
        }
    else:
        return {
            'model': f"{MODEL_DIR}/model.pkl",
            'scaler': f"{MODEL_DIR}/scaler.pkl",
            'features': f"{MODEL_DIR}/features.json"
        }

def get_capital_for_pair(symbol, total_capital):
    """Pour MultiPairBot uniquement"""
    return total_capital / len(TRADING_PAIRS)

def get_position_quantity(symbol, available_capital, current_price):
    """Calcule la quantité à acheter"""
    position_value = available_capital * SCANNER_POSITION_SIZE
    
    if position_value > SCANNER_MAX_POSITION_VALUE:
        position_value = SCANNER_MAX_POSITION_VALUE
    
    quantity = position_value / current_price
    
    if 'BTC' in symbol:
        quantity = round(quantity, 5)
    elif 'ETH' in symbol or 'BNB' in symbol:
        quantity = round(quantity, 4)
    else:
        quantity = round(quantity, 2)
    
    return quantity

def validate_config():
    """Valide la configuration"""
    errors = []
    warnings = []
    
    if not BINANCE_API_KEY or BINANCE_API_KEY == "votre_clé":
        warnings.append("⚠️  BINANCE_API_KEY not configured")
    
    if not BINANCE_API_SECRET or BINANCE_API_SECRET == "votre_secret":
        warnings.append("⚠️  BINANCE_API_SECRET not configured")
    
    if len(TRADING_PAIRS) == 0:
        errors.append("❌ No trading pairs defined")
    
    if SCANNER_UPDATE_INTERVAL < 5:
        warnings.append("⚠️  Update interval < 5s may hit rate limits")
    
    if SCANNER_POSITION_SIZE > 0.5:
        warnings.append("⚠️  Position size > 50% is very risky")
    
    if SCANNER_MIN_CONFIDENCE < 0.5:
        warnings.append("⚠️  Min confidence < 50% will trade too often")
    
    return errors, warnings

def print_scanner_config():
    """Affiche la config du scanner"""
    print("\n" + "="*70)
    print("🔍 SCANNER CONFIGURATION")
    print("="*70)
    
    print(f"\n📊 Pairs: {len(TRADING_PAIRS)}")
    print(f"⏱️  Update: {SCANNER_UPDATE_INTERVAL}s")
    print(f"💰 Position size: {SCANNER_POSITION_SIZE:.1%}")
    print(f"🎯 Min confidence: {SCANNER_MIN_CONFIDENCE:.0%}")
    print(f"🛑 Stop Loss: {SCANNER_STOP_LOSS:.1%}")
    print(f"🎯 Take Profit: {SCANNER_TAKE_PROFIT:.1%}")
    print(f"💵 Max position: ${SCANNER_MAX_POSITION_VALUE}")
    
    print("\n" + "="*70)

# ==================== VALIDATION ====================
if __name__ == "__main__":
    print("🔍 Validating scanner configuration...\n")
    
    errors, warnings = validate_config()
    
    if errors:
        print("❌ ERRORS:")
        for error in errors:
            print(f"   {error}")
        print()
    
    if warnings:
        print("⚠️  WARNINGS:")
        for warning in warnings:
            print(f"   {warning}")
        print()
    
    if not errors:
        print("✅ Configuration is valid!\n")
        print_scanner_config()