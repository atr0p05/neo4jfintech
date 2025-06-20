#!/usr/bin/env python3
"""
Test the Advanced Portfolio Optimization Engine
"""
import sys
sys.path.append('neo4j-investment-platform/backend')

from neo4j import GraphDatabase
from core.optimization.advanced_optimizer import AdvancedPortfolioOptimizer
import pandas as pd

def main():
    # Neo4j connection
    uri = "neo4j+s://1c34ddb0.databases.neo4j.io"
    user = "neo4j"
    password = "Sw-DAlLfV1_Rw4lFlXAxl0DoN4lCauWrp29IZnfq_zM"
    
    driver = GraphDatabase.driver(uri, auth=(user, password))
    optimizer = AdvancedPortfolioOptimizer(driver)
    
    client_id = "CLI_100001"
    
    print("🚀 Testing Advanced Portfolio Optimization\n")
    print("="*60)
    
    # Test 1: Markowitz Optimization
    print("\n1️⃣ MARKOWITZ OPTIMIZATION")
    print("-"*40)
    try:
        results = optimizer.optimize_portfolio(client_id, 'markowitz')
        print(f"✅ Expected Return: {results['expected_return']*100:.2f}%")
        print(f"✅ Volatility: {results['volatility']*100:.2f}%")
        print(f"✅ Sharpe Ratio: {results['sharpe_ratio']:.3f}")
        print(f"✅ Diversification Ratio: {results.get('diversification_ratio', 0):.2f}")
        print("\nTop 5 Holdings:")
        for symbol, weight in sorted(results['weights'].items(), key=lambda x: x[1], reverse=True)[:5]:
            if weight > 0.01:
                print(f"  • {symbol}: {weight*100:.1f}%")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Test 2: Black-Litterman with Views
    print("\n\n2️⃣ BLACK-LITTERMAN OPTIMIZATION (with market views)")
    print("-"*40)
    try:
        views = {
            'absolute': {
                'AAPL': 0.15,  # Bullish on Apple
                'TLT': 0.03    # Bearish on long bonds
            }
        }
        results = optimizer.optimize_portfolio(client_id, 'black_litterman', views=views)
        print(f"✅ Expected Return: {results['expected_return']*100:.2f}%")
        print(f"✅ Volatility: {results['volatility']*100:.2f}%")
        print(f"✅ Sharpe Ratio: {results['sharpe_ratio']:.3f}")
        print("\nTop 5 Holdings:")
        for symbol, weight in sorted(results['weights'].items(), key=lambda x: x[1], reverse=True)[:5]:
            if weight > 0.01:
                print(f"  • {symbol}: {weight*100:.1f}%")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Test 3: Risk Parity
    print("\n\n3️⃣ RISK PARITY OPTIMIZATION")
    print("-"*40)
    try:
        results = optimizer.optimize_portfolio(client_id, 'risk_parity')
        print(f"✅ Expected Return: {results['expected_return']*100:.2f}%")
        print(f"✅ Volatility: {results['volatility']*100:.2f}%")
        print(f"✅ Sharpe Ratio: {results['sharpe_ratio']:.3f}")
        print("\nRisk Contribution (Top 5):")
        if 'risk_contribution' in results:
            for symbol, contrib in sorted(results['risk_contribution'].items(), key=lambda x: x[1], reverse=True)[:5]:
                print(f"  • {symbol}: {contrib*100:.1f}%")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Test 4: Factor-Based Optimization
    print("\n\n4️⃣ FACTOR-BASED OPTIMIZATION")
    print("-"*40)
    try:
        results = optimizer.optimize_portfolio(client_id, 'factor_based')
        print(f"✅ Expected Return: {results['expected_return']*100:.2f}%")
        print(f"✅ Volatility: {results['volatility']*100:.2f}%")
        print(f"✅ Sharpe Ratio: {results['sharpe_ratio']:.3f}")
        if 'factor_exposures' in results:
            print("\nAchieved Factor Exposures:")
            for factor, exposure in sorted(results['factor_exposures'].items(), key=lambda x: abs(x[1]), reverse=True)[:5]:
                print(f"  • {factor}: {exposure:.3f}")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Test 5: Model Comparison
    print("\n\n5️⃣ MODEL COMPARISON")
    print("-"*40)
    try:
        comparison = optimizer.compare_optimization_models(client_id)
        print(comparison.to_string(index=False))
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Test 6: Custom Constraints
    print("\n\n6️⃣ OPTIMIZATION WITH CUSTOM CONSTRAINTS")
    print("-"*40)
    try:
        custom_constraints = {
            'min_weight': 0.05,  # Minimum 5% per position
            'max_weight': 0.15,  # Maximum 15% per position
            'asset_class_limits': {
                'Equity': (0.60, 0.80),  # 60-80% stocks
                'Bond': (0.15, 0.30),    # 15-30% bonds
                'Alternative': (0.05, 0.15)  # 5-15% alternatives
            }
        }
        results = optimizer.optimize_portfolio(client_id, 'markowitz', constraints=custom_constraints)
        print(f"✅ Custom constraints applied successfully")
        print(f"✅ Number of holdings: {results['number_of_assets']}")
        print(f"✅ Sharpe Ratio: {results['sharpe_ratio']:.3f}")
        
        # Check allocations
        weights = results['weights']
        equity_weight = sum(w for s, w in weights.items() if s in ['AAPL', 'MSFT', 'JPM', 'BAC', 'JNJ', 'PFE', 'XOM', 'CVX', 'PG', 'KO'])
        print(f"\n✅ Equity allocation: {equity_weight*100:.1f}% (target: 60-80%)")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    print("\n" + "="*60)
    print("✨ Advanced optimization testing complete!")
    
    driver.close()

if __name__ == "__main__":
    main()