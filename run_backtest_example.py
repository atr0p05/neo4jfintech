#!/usr/bin/env python3
"""
Example: How to run backtests with the Neo4j Investment Platform
"""

import requests
import json
import sys
import os

# Add backend to path
sys.path.append('neo4j-investment-platform/backend')

def run_api_backtest():
    """Run backtest via API"""
    print("🔬 Running Backtest via API...")
    
    # Backtest configuration
    backtest_request = {
        "symbols": ["AAPL", "MSFT", "JPM", "TLT", "GLD", "VTI"],
        "strategy": {
            "name": "Balanced Portfolio",
            "optimization_model": "markowitz",
            "rebalance_frequency": "quarterly",
            "transaction_cost": 0.001
        },
        "startDate": "2020-01-01",
        "endDate": "2023-12-31",
        "initialCapital": 100000
    }
    
    try:
        response = requests.post(
            "http://localhost:8000/api/backtest",
            headers={"Content-Type": "application/json"},
            json=backtest_request,
            timeout=60
        )
        
        if response.status_code == 200:
            results = response.json()
            print("✅ Backtest Complete!")
            print(f"📊 Strategy: {results['strategyName']}")
            print(f"📈 Total Return: {results['metrics']['totalReturn']*100:.2f}%")
            print(f"📉 Max Drawdown: {results['metrics']['maxDrawdown']*100:.2f}%")
            print(f"⚡ Sharpe Ratio: {results['metrics']['sharpeRatio']:.3f}")
            print(f"🎯 Win Rate: {results['metrics']['winRate']*100:.1f}%")
            
            if 'comparison' in results:
                print(f"\n📊 vs Benchmark ({results['comparison']['benchmark']}):")
                print(f"   Alpha: {results['comparison']['alpha']*100:.2f}%")
                print(f"   Beta: {results['comparison']['beta']:.2f}")
                print(f"   Excess Return: {results['comparison']['excess_return']*100:.2f}%")
        else:
            print(f"❌ Error: {response.status_code} - {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to backend. Make sure it's running on localhost:8000")
    except Exception as e:
        print(f"❌ Error: {e}")

def compare_strategies():
    """Compare multiple strategies"""
    print("\n🔄 Comparing Multiple Strategies...")
    
    strategies = [
        {"name": "Equal Weight", "optimization_model": "equal_weight"},
        {"name": "Risk Parity", "optimization_model": "risk_parity"},
        {"name": "Markowitz", "optimization_model": "markowitz"},
        {"name": "Min Volatility", "optimization_model": "min_volatility"}
    ]
    
    compare_request = {
        "symbols": ["AAPL", "MSFT", "JPM", "TLT", "GLD"],
        "strategies": strategies,
        "start_date": "2020-01-01",
        "end_date": "2023-12-31",
        "initial_capital": 100000
    }
    
    try:
        response = requests.post(
            "http://localhost:8000/api/backtest/compare",
            headers={"Content-Type": "application/json"},
            json=compare_request
        )
        
        if response.status_code == 200:
            results = response.json()
            print("✅ Strategy Comparison Complete!")
            print("\n📊 Results Summary:")
            print("-" * 70)
            print(f"{'Strategy':<15} {'Return':<10} {'Risk':<10} {'Sharpe':<10}")
            print("-" * 70)
            
            for result in results:
                print(f"{result['strategy']:<15} {result['total_return']*100:>8.1f}% {result['volatility']*100:>8.1f}% {result['sharpe_ratio']:>8.2f}")
                
        else:
            print(f"❌ Error: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Error: {e}")

def run_direct_backtest():
    """Run backtest directly using the backtesting engine"""
    print("\n🔧 Running Direct Backtest...")
    
    try:
        from neo4j import GraphDatabase
        from services.backtesting.backtest_engine import OptimizationBacktester
        
        # Neo4j connection
        uri = "neo4j+s://1c34ddb0.databases.neo4j.io"
        driver = GraphDatabase.driver(uri, auth=("neo4j", "Sw-DAlLfV1_Rw4lFlXAxl0DoN4lCauWrp29IZnfq_zM"))
        
        # Initialize backtester
        backtester = OptimizationBacktester(driver)
        
        # Run backtest
        symbols = ['AAPL', 'MSFT', 'JPM', 'TLT', 'GLD']
        strategy_params = {
            'optimization_model': 'equal_weight',
            'rebalance_frequency': 'quarterly'
        }
        
        result = backtester.backtest_strategy(
            symbols,
            strategy_params,
            '2020-01-01',
            '2023-12-31',
            100000
        )
        
        print("✅ Direct Backtest Complete!")
        print(f"📈 Total Return: {result.total_return*100:.2f}%")
        print(f"⚡ Sharpe Ratio: {result.sharpe_ratio:.3f}")
        print(f"📉 Max Drawdown: {result.max_drawdown*100:.2f}%")
        
        driver.close()
        
    except ImportError:
        print("📝 Direct backtesting requires the full backend environment")
        print("💡 Use the API method instead: run_api_backtest()")
    except Exception as e:
        print(f"❌ Error: {e}")

def interactive_backtest():
    """Interactive backtest configuration"""
    print("\n🎮 Interactive Backtest Setup")
    print("=" * 50)
    
    # Get user inputs
    symbols_input = input("📊 Enter symbols (comma-separated) [AAPL,MSFT,JPM,TLT,GLD]: ").strip()
    symbols = [s.strip().upper() for s in symbols_input.split(',')] if symbols_input else ["AAPL", "MSFT", "JPM", "TLT", "GLD"]
    
    print("\n🎯 Available Optimization Models:")
    models = ["markowitz", "risk_parity", "equal_weight", "min_volatility"]
    for i, model in enumerate(models, 1):
        print(f"  {i}. {model.replace('_', ' ').title()}")
    
    model_choice = input("\nSelect model (1-4) [1]: ").strip() or "1"
    model = models[int(model_choice) - 1] if model_choice.isdigit() and 1 <= int(model_choice) <= 4 else "markowitz"
    
    start_date = input("📅 Start date (YYYY-MM-DD) [2020-01-01]: ").strip() or "2020-01-01"
    end_date = input("📅 End date (YYYY-MM-DD) [2023-12-31]: ").strip() or "2023-12-31"
    
    capital_input = input("💰 Initial capital [$100,000]: ").strip()
    capital = float(capital_input.replace('$', '').replace(',', '')) if capital_input else 100000
    
    # Configure backtest
    backtest_config = {
        "symbols": symbols,
        "strategy": {
            "name": f"Custom {model.replace('_', ' ').title()} Strategy",
            "optimization_model": model,
            "rebalance_frequency": "quarterly"
        },
        "startDate": start_date,
        "endDate": end_date,
        "initialCapital": capital
    }
    
    print(f"\n🔬 Running backtest...")
    print(f"📊 Symbols: {', '.join(symbols)}")
    print(f"🎯 Model: {model.replace('_', ' ').title()}")
    print(f"📅 Period: {start_date} to {end_date}")
    print(f"💰 Capital: ${capital:,.0f}")
    
    # Run the backtest
    try:
        response = requests.post(
            "http://localhost:8000/api/backtest",
            headers={"Content-Type": "application/json"},
            json=backtest_config,
            timeout=60
        )
        
        if response.status_code == 200:
            results = response.json()
            print("\n" + "="*50)
            print("📈 BACKTEST RESULTS")
            print("="*50)
            print(f"🎯 Strategy: {results['strategyName']}")
            print(f"📊 Total Return: {results['metrics']['totalReturn']*100:.2f}%")
            print(f"📈 Annual Return: {results['metrics']['annualizedReturn']*100:.2f}%")
            print(f"📉 Volatility: {results['metrics']['volatility']*100:.2f}%")
            print(f"⚡ Sharpe Ratio: {results['metrics']['sharpeRatio']:.3f}")
            print(f"🔻 Max Drawdown: {results['metrics']['maxDrawdown']*100:.2f}%")
            print(f"📊 Calmar Ratio: {results['metrics']['calmarRatio']:.2f}")
            print(f"🎯 Win Rate: {results['metrics']['winRate']*100:.1f}%")
            
            if 'comparison' in results:
                print(f"\n📊 BENCHMARK COMPARISON")
                print(f"🔄 vs {results['comparison']['benchmark']}")
                print(f"📈 Alpha: {results['comparison']['alpha']*100:.2f}%")
                print(f"📊 Beta: {results['comparison']['beta']:.2f}")
                
        else:
            print(f"❌ Backtest failed: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Error running backtest: {e}")

if __name__ == "__main__":
    print("🚀 Neo4j Investment Platform - Backtesting Examples")
    print("=" * 60)
    
    print("\n📋 Available Options:")
    print("1. 🔬 Run API Backtest")
    print("2. 🔄 Compare Strategies") 
    print("3. 🎮 Interactive Backtest")
    print("4. 🔧 Direct Backtest (requires full backend)")
    print("5. 🌐 Open API Documentation")
    
    choice = input("\nSelect option (1-5): ").strip()
    
    if choice == "1":
        run_api_backtest()
    elif choice == "2":
        compare_strategies()
    elif choice == "3":
        interactive_backtest()
    elif choice == "4":
        run_direct_backtest()
    elif choice == "5":
        print("🌐 Opening API docs...")
        print("📖 Swagger UI: http://localhost:8000/docs")
        print("📚 ReDoc: http://localhost:8000/redoc")
        import webbrowser
        webbrowser.open("http://localhost:8000/docs")
    else:
        print("📋 Running default API backtest...")
        run_api_backtest()
    
    print("\n✨ Done! Check the analytics dashboard: http://localhost:8000/analytics/dashboard")