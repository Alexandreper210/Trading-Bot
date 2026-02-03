# test_binance.py
from binance.client import Client
from config import BINANCE_API_KEY, BINANCE_API_SECRET

# Connexion testnet
client = Client(BINANCE_API_KEY, BINANCE_API_SECRET, testnet=True)

print("🔗 Testing Binance Testnet connection...\n")

try:
    # Test 1 : Balance
    account = client.get_account()
    balances = {b['asset']: float(b['free']) 
                for b in account['balances'] if float(b['free']) > 0}
    
    print("✅ Connection successful!")
    print(f"\n💼 Your Testnet Balance:")
    for asset, amount in balances.items():
        print(f"   {asset}: {amount}")
    
    # Test 2 : Prix BTC
    ticker = client.get_symbol_ticker(symbol="BTCUSDT")
    print(f"\n📊 BTC Price: ${float(ticker['price']):,.2f}")
    
    # Test 3 : Historique
    klines = client.get_klines(symbol='BTCUSDT', interval='1m', limit=5)
    print(f"\n📈 Last 5 candles retrieved: OK")
    
    print("\n✅ All tests passed! Ready to trade.")

except Exception as e:
    print(f"❌ Error: {e}")
    print("\n💡 Check:")
    print("   1. API keys are correct")
    print("   2. You're using TESTNET keys (not production)")
    print("   3. Internet connection is working")