import numpy as np
import pandas as pd
from neo4j import GraphDatabase
from datetime import datetime, timedelta
import random

class PortfolioRiskAnalytics:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
    
    def close(self):
        self.driver.close()
    
    def add_risk_metrics_to_assets(self):
        """Add MSCI/Axioma-style risk metrics to assets"""
        
        # Realistic risk metrics by asset class
        risk_profiles = {
            'Equity': {
                'Technology': {'beta': (1.2, 1.5), 'volatility': (0.25, 0.35), 'sharpe': (0.8, 1.5)},
                'Financials': {'beta': (1.0, 1.2), 'volatility': (0.20, 0.28), 'sharpe': (0.6, 1.2)},
                'Healthcare': {'beta': (0.8, 1.0), 'volatility': (0.15, 0.22), 'sharpe': (0.7, 1.3)},
                'Energy': {'beta': (1.1, 1.4), 'volatility': (0.28, 0.40), 'sharpe': (0.5, 1.0)},
                'Consumer Staples': {'beta': (0.6, 0.8), 'volatility': (0.12, 0.18), 'sharpe': (0.9, 1.4)}
            },
            'Bond': {
                'Government': {'beta': (0.1, 0.3), 'volatility': (0.04, 0.08), 'sharpe': (0.3, 0.6)},
                'Corporate': {'beta': (0.3, 0.5), 'volatility': (0.06, 0.12), 'sharpe': (0.4, 0.8)},
                'Municipal': {'beta': (0.2, 0.4), 'volatility': (0.05, 0.10), 'sharpe': (0.3, 0.7)}
            },
            'Alternative': {
                'Commodities': {'beta': (0.4, 0.7), 'volatility': (0.20, 0.35), 'sharpe': (0.2, 0.6)},
                'Real Estate': {'beta': (0.6, 0.9), 'volatility': (0.15, 0.25), 'sharpe': (0.5, 0.9)}
            }
        }
        
        with self.driver.session() as session:
            # Get all assets
            assets = session.run("""
                MATCH (a:Asset)
                RETURN a.symbol as symbol, a.assetClass as assetClass, a.sector as sector
            """).data()
            
            for asset in assets:
                asset_class = asset['assetClass']
                sector = asset['sector']
                
                if asset_class in risk_profiles and sector in risk_profiles[asset_class]:
                    profile = risk_profiles[asset_class][sector]
                else:
                    # Default profile
                    profile = {'beta': (0.8, 1.2), 'volatility': (0.15, 0.25), 'sharpe': (0.5, 1.0)}
                
                # Generate risk metrics
                beta = round(random.uniform(*profile['beta']), 3)
                volatility = round(random.uniform(*profile['volatility']), 3)
                sharpe_ratio = round(random.uniform(*profile['sharpe']), 3)
                expected_return = round(volatility * sharpe_ratio + 0.02, 3)  # Risk-free rate of 2%
                
                # Add metrics to asset
                session.run("""
                    MATCH (a:Asset {symbol: $symbol})
                    SET a.beta = $beta,
                        a.volatility = $volatility,
                        a.sharpeRatio = $sharpeRatio,
                        a.expectedReturn = $expectedReturn,
                        a.lastUpdated = datetime()
                """, symbol=asset['symbol'], beta=beta, volatility=volatility,
                    sharpeRatio=sharpe_ratio, expectedReturn=expected_return)
            
            print(f"✓ Added risk metrics to {len(assets)} assets")
    
    def create_factor_exposures(self):
        """Create MSCI Barra-style factor exposures"""
        
        factors = [
            # Style factors
            {'name': 'Value', 'description': 'Exposure to value stocks'},
            {'name': 'Growth', 'description': 'Exposure to growth stocks'},
            {'name': 'Momentum', 'description': 'Recent price performance'},
            {'name': 'Quality', 'description': 'Profitability and earnings quality'},
            {'name': 'Size', 'description': 'Market capitalization factor'},
            {'name': 'Volatility', 'description': 'Low volatility anomaly'},
            
            # Macro factors
            {'name': 'Interest Rate Sensitivity', 'description': 'Duration and rate exposure'},
            {'name': 'Credit Spread', 'description': 'Credit risk premium'},
            {'name': 'Inflation', 'description': 'Inflation sensitivity'},
            {'name': 'Currency', 'description': 'Foreign exchange exposure'}
        ]
        
        with self.driver.session() as session:
            # Create factor nodes
            for factor in factors:
                session.run("""
                    MERGE (f:Factor {name: $name})
                    SET f.description = $description,
                        f.category = CASE 
                            WHEN $name IN ['Value', 'Growth', 'Momentum', 'Quality', 'Size', 'Volatility'] 
                            THEN 'Style'
                            ELSE 'Macro'
                        END
                """, name=factor['name'], description=factor['description'])
            
            # Create factor exposures for assets
            assets = session.run("MATCH (a:Asset) RETURN a.symbol as symbol, a.sector as sector").data()
            
            for asset in assets:
                # Generate realistic factor exposures
                exposures = self._generate_factor_exposures(asset['sector'])
                
                for factor_name, exposure in exposures.items():
                    if exposure != 0:  # Only create relationship if exposure exists
                        session.run("""
                            MATCH (a:Asset {symbol: $symbol})
                            MATCH (f:Factor {name: $factor})
                            MERGE (a)-[e:HAS_FACTOR_EXPOSURE]->(f)
                            SET e.exposure = $exposure,
                                e.tStat = $tstat
                        """, symbol=asset['symbol'], factor=factor_name, 
                            exposure=exposure, tstat=round(exposure / 0.3, 2))  # Simplified t-stat
            
            print("✓ Created factor exposures")
    
    def _generate_factor_exposures(self, sector):
        """Generate realistic factor exposures based on sector"""
        exposures = {}
        
        # Sector-based factor tilts
        sector_profiles = {
            'Technology': {'Growth': 0.8, 'Momentum': 0.6, 'Quality': 0.4, 'Value': -0.5},
            'Financials': {'Value': 0.7, 'Interest Rate Sensitivity': 0.8, 'Credit Spread': 0.6},
            'Healthcare': {'Quality': 0.7, 'Growth': 0.5, 'Volatility': -0.3},
            'Energy': {'Value': 0.6, 'Inflation': 0.7, 'Currency': 0.4},
            'Consumer Staples': {'Quality': 0.6, 'Volatility': -0.5, 'Value': 0.3},
            'Government': {'Interest Rate Sensitivity': 0.9, 'Credit Spread': -0.8, 'Inflation': -0.6},
            'Corporate': {'Credit Spread': 0.7, 'Interest Rate Sensitivity': 0.5},
            'Commodities': {'Inflation': 0.8, 'Currency': 0.6, 'Momentum': 0.4},
            'Real Estate': {'Interest Rate Sensitivity': -0.6, 'Inflation': 0.5}
        }
        
        base_profile = sector_profiles.get(sector, {})
        
        # Add some randomness
        all_factors = ['Value', 'Growth', 'Momentum', 'Quality', 'Size', 'Volatility',
                      'Interest Rate Sensitivity', 'Credit Spread', 'Inflation', 'Currency']
        
        for factor in all_factors:
            if factor in base_profile:
                # Add noise to base exposure
                exposures[factor] = round(base_profile[factor] + random.uniform(-0.2, 0.2), 3)
            else:
                # Small random exposure
                if random.random() > 0.7:
                    exposures[factor] = round(random.uniform(-0.3, 0.3), 3)
                else:
                    exposures[factor] = 0
        
        return exposures
    
    def calculate_portfolio_risk_metrics(self):
        """Calculate portfolio-level risk metrics"""
        
        with self.driver.session() as session:
            portfolios = session.run("""
                MATCH (p:Portfolio)-[h:HOLDS]->(a:Asset)
                RETURN p.portfolioId as portfolioId,
                       collect({
                           symbol: a.symbol,
                           weight: h.weight,
                           volatility: a.volatility,
                           beta: a.beta,
                           expectedReturn: a.expectedReturn
                       }) as holdings
            """).data()
            
            for portfolio in portfolios:
                holdings = portfolio['holdings']
                
                # Calculate portfolio metrics
                portfolio_return = sum(h['weight'] * h['expectedReturn'] for h in holdings)
                portfolio_beta = sum(h['weight'] * h['beta'] for h in holdings)
                
                # Simplified portfolio volatility (ignoring correlations for now)
                portfolio_vol = np.sqrt(sum((h['weight'] * h['volatility'])**2 for h in holdings))
                
                # Calculate risk metrics
                sharpe = (portfolio_return - 0.02) / portfolio_vol if portfolio_vol > 0 else 0
                var_95 = 1.65 * portfolio_vol  # 95% VaR assuming normal distribution
                cvar_95 = 2.06 * portfolio_vol  # 95% CVaR
                
                # Update portfolio with metrics
                session.run("""
                    MATCH (p:Portfolio {portfolioId: $portfolioId})
                    SET p.expectedReturn = $return,
                        p.volatility = $volatility,
                        p.beta = $beta,
                        p.sharpeRatio = $sharpe,
                        p.var95 = $var95,
                        p.cvar95 = $cvar95,
                        p.lastCalculated = datetime()
                """, portfolioId=portfolio['portfolioId'],
                    **{'return': round(portfolio_return, 4),
                       'volatility': round(portfolio_vol, 4),
                       'beta': round(portfolio_beta, 3),
                       'sharpe': round(sharpe, 3),
                       'var95': round(var_95, 4),
                       'cvar95': round(cvar_95, 4)})
            
            print(f"✓ Calculated risk metrics for {len(portfolios)} portfolios")
    
    def create_correlation_relationships(self):
        """Create correlation relationships between assets"""
        
        with self.driver.session() as session:
            # Get all assets
            assets = session.run("""
                MATCH (a:Asset)
                RETURN a.symbol as symbol, a.sector as sector, a.assetClass as assetClass
            """).data()
            
            # Create correlations (simplified - in reality, calculate from returns)
            for i, asset1 in enumerate(assets):
                for j, asset2 in enumerate(assets[i+1:], i+1):
                    # Calculate correlation based on asset characteristics
                    correlation = self._calculate_correlation(asset1, asset2)
                    
                    if abs(correlation) > 0.3:  # Only store significant correlations
                        session.run("""
                            MATCH (a1:Asset {symbol: $symbol1})
                            MATCH (a2:Asset {symbol: $symbol2})
                            MERGE (a1)-[c:CORRELATED_WITH]-(a2)
                            SET c.correlation = $correlation,
                                c.period = '1Y',
                                c.lastUpdated = datetime()
                        """, symbol1=asset1['symbol'], symbol2=asset2['symbol'],
                            correlation=correlation)
            
            print("✓ Created correlation relationships")
    
    def _calculate_correlation(self, asset1, asset2):
        """Calculate correlation between two assets"""
        if asset1['symbol'] == asset2['symbol']:
            return 1.0
        
        # Same asset class and sector = high correlation
        if asset1['assetClass'] == asset2['assetClass'] and asset1['sector'] == asset2['sector']:
            return round(0.7 + random.uniform(0, 0.2), 3)
        
        # Same asset class = moderate correlation
        elif asset1['assetClass'] == asset2['assetClass']:
            return round(0.4 + random.uniform(-0.1, 0.2), 3)
        
        # Bonds and stocks = negative correlation
        elif (asset1['assetClass'] == 'Bond' and asset2['assetClass'] == 'Equity') or \
             (asset1['assetClass'] == 'Equity' and asset2['assetClass'] == 'Bond'):
            return round(-0.3 + random.uniform(-0.1, 0.1), 3)
        
        # Default low correlation
        else:
            return round(random.uniform(-0.2, 0.2), 3)

# Usage
if __name__ == "__main__":
    # Your Neo4j Aura credentials
    URI = "neo4j+s://1c34ddb0.databases.neo4j.io"
    USER = "neo4j"
    PASSWORD = "Sw-DAlLfV1_Rw4lFlXAxl0DoN4lCauWrp29IZnfq_zM"
    
    analytics = PortfolioRiskAnalytics(URI, USER, PASSWORD)
    
    try:
        print("Setting up risk analytics...")
        
        # Add risk metrics to assets
        analytics.add_risk_metrics_to_assets()
        
        # Create factor exposures
        analytics.create_factor_exposures()
        
        # Calculate portfolio metrics
        analytics.calculate_portfolio_risk_metrics()
        
        # Create correlations
        analytics.create_correlation_relationships()
        
        print("\n✓ Risk analytics setup complete!")
        
    finally:
        analytics.close()