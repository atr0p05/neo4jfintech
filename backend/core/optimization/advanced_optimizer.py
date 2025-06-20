"""
Portfolio Optimization Engine
Implements Markowitz, Black-Litterman, Risk Parity, and custom constraints
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import cvxpy as cp
from pypfopt import EfficientFrontier, BlackLittermanModel, HRPOpt, plotting
from pypfopt import risk_models, expected_returns, objective_functions
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices
import riskfolio as rp
from neo4j import GraphDatabase
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


class AdvancedPortfolioOptimizer:
    """
    Multi-model portfolio optimizer integrated with Neo4j
    Supports: Markowitz, Black-Litterman, Risk Parity, Factor-based optimization
    """
    
    def __init__(self, neo4j_driver):
        self.driver = neo4j_driver
        self.models = {
            'markowitz': self._optimize_markowitz,
            'black_litterman': self._optimize_black_litterman,
            'risk_parity': self._optimize_risk_parity,
            'max_sharpe': self._optimize_max_sharpe,
            'min_volatility': self._optimize_min_volatility,
            'factor_based': self._optimize_factor_based,
            'cvar': self._optimize_cvar
        }
    
    def optimize_portfolio(self, 
                         client_id: str, 
                         optimization_model: str = 'markowitz',
                         constraints: Optional[Dict] = None,
                         views: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Main optimization function that routes to specific models
        
        Args:
            client_id: Client ID
            optimization_model: Model to use (markowitz, black_litterman, etc.)
            constraints: Custom constraints dict
            views: Market views for Black-Litterman
        
        Returns:
            Optimization results including weights, metrics, and recommendations
        """
        
        # Get client profile and current portfolio
        client_data = self._get_client_data(client_id)
        market_data = self._get_market_data(client_data['universe'])
        
        # Apply default constraints based on client profile
        if constraints is None:
            constraints = self._get_default_constraints(client_data)
        
        # Run optimization
        if optimization_model in self.models:
            results = self.models[optimization_model](
                client_data, market_data, constraints, views
            )
        else:
            raise ValueError(f"Unknown optimization model: {optimization_model}")
        
        # Post-process results
        results = self._post_process_results(results, client_data, market_data)
        
        # Save to Neo4j
        self._save_optimization_results(client_id, results)
        
        return results
    
    def _get_client_data(self, client_id: str) -> Dict[str, Any]:
        """Get client profile and investment universe from Neo4j"""
        with self.driver.session() as session:
            # Get client profile
            client = session.run("""
                MATCH (c:Client {clientId: $clientId})-[:HAS_RISK_PROFILE]->(rp:RiskProfile)
                OPTIONAL MATCH (c)-[:OWNS_PORTFOLIO]->(p:Portfolio)
                RETURN c, rp, collect(p) as portfolios
            """, clientId=client_id).single()
            
            # Get investment universe (current holdings + recommended assets)
            universe = session.run("""
                // Current holdings
                MATCH (c:Client {clientId: $clientId})-[:OWNS_PORTFOLIO]->(p:Portfolio)
                MATCH (p)-[:HOLDS]->(a:Asset)
                WITH collect(DISTINCT a) as currentAssets
                
                // Add high-quality assets matching risk profile
                MATCH (c:Client {clientId: $clientId})-[:HAS_RISK_PROFILE]->(rp:RiskProfile)
                MATCH (candidate:Asset)
                WHERE candidate.volatility <= rp.volatilityTolerance
                AND candidate.sharpeRatio > 0.5
                AND NOT candidate IN currentAssets
                WITH currentAssets + collect(candidate)[0..20] as universe
                
                UNWIND universe as asset
                RETURN DISTINCT asset.symbol as symbol,
                       asset.expectedReturn as expectedReturn,
                       asset.volatility as volatility,
                       asset.currentPrice as price,
                       asset.assetClass as assetClass,
                       asset.sector as sector
            """, clientId=client_id).data()
            
            return {
                'client': dict(client['c']),
                'risk_profile': dict(client['rp']),
                'portfolios': [dict(p) for p in client['portfolios']],
                'universe': universe
            }
    
    def _get_market_data(self, universe: List[Dict]) -> Dict[str, pd.DataFrame]:
        """Prepare market data for optimization"""
        # Create DataFrames for optimization
        symbols = [asset['symbol'] for asset in universe]
        
        # Expected returns
        mu = pd.Series({
            asset['symbol']: asset['expectedReturn'] 
            for asset in universe
        })
        
        # Covariance matrix (simplified - in production, calculate from historical data)
        # For now, use volatilities and assumed correlations
        n = len(symbols)
        cov_matrix = np.zeros((n, n))
        
        for i, asset_i in enumerate(universe):
            for j, asset_j in enumerate(universe):
                if i == j:
                    cov_matrix[i, j] = asset_i['volatility'] ** 2
                else:
                    # Correlation based on asset class and sector
                    corr = self._estimate_correlation(asset_i, asset_j)
                    cov_matrix[i, j] = corr * asset_i['volatility'] * asset_j['volatility']
        
        S = pd.DataFrame(cov_matrix, index=symbols, columns=symbols)
        
        # Prices
        prices = pd.Series({
            asset['symbol']: asset['price'] 
            for asset in universe
        })
        
        # Asset metadata
        metadata = pd.DataFrame(universe).set_index('symbol')
        
        return {
            'expected_returns': mu,
            'cov_matrix': S,
            'prices': prices,
            'metadata': metadata
        }
    
    def _estimate_correlation(self, asset1: Dict, asset2: Dict) -> float:
        """Estimate correlation between assets based on characteristics"""
        if asset1['symbol'] == asset2['symbol']:
            return 1.0
        
        # Same asset class and sector
        if asset1['assetClass'] == asset2['assetClass'] and asset1['sector'] == asset2['sector']:
            return 0.8
        
        # Same asset class
        if asset1['assetClass'] == asset2['assetClass']:
            return 0.5
        
        # Stocks vs Bonds (negative correlation)
        if (asset1['assetClass'] == 'Equity' and asset2['assetClass'] == 'Bond') or \
           (asset1['assetClass'] == 'Bond' and asset2['assetClass'] == 'Equity'):
            return -0.3
        
        # Default
        return 0.2
    
    def _get_default_constraints(self, client_data: Dict) -> Dict[str, Any]:
        """Get default constraints based on client risk profile"""
        risk_tolerance = client_data['client']['riskTolerance']
        
        constraints = {
            'min_weight': 0.02,  # Minimum 2% position
            'max_weight': 0.20,  # Maximum 20% position
            'sector_limits': {},
            'asset_class_limits': {}
        }
        
        # Risk-based constraints
        if risk_tolerance == 'Conservative':
            constraints['asset_class_limits'] = {
                'Equity': (0.2, 0.4),
                'Bond': (0.5, 0.7),
                'Alternative': (0, 0.1)
            }
            constraints['max_weight'] = 0.15
        elif risk_tolerance == 'Moderate':
            constraints['asset_class_limits'] = {
                'Equity': (0.4, 0.6),
                'Bond': (0.3, 0.5),
                'Alternative': (0, 0.2)
            }
        else:  # Aggressive
            constraints['asset_class_limits'] = {
                'Equity': (0.6, 0.8),
                'Bond': (0.1, 0.3),
                'Alternative': (0, 0.3)
            }
            constraints['max_weight'] = 0.25
        
        return constraints
    
    def _optimize_markowitz(self, client_data: Dict, market_data: Dict, 
                          constraints: Dict, views: Optional[Dict]) -> Dict:
        """Classic Markowitz Mean-Variance Optimization"""
        mu = market_data['expected_returns']
        S = market_data['cov_matrix']
        
        # Create efficient frontier
        ef = EfficientFrontier(mu, S)
        
        # Add constraints
        for asset in mu.index:
            ef.add_constraint(lambda w, asset=asset: w[mu.index.get_loc(asset)] >= constraints['min_weight'])
            ef.add_constraint(lambda w, asset=asset: w[mu.index.get_loc(asset)] <= constraints['max_weight'])
        
        # Add asset class constraints
        metadata = market_data['metadata']
        for asset_class, (min_weight, max_weight) in constraints.get('asset_class_limits', {}).items():
            class_assets = metadata[metadata['assetClass'] == asset_class].index.tolist()
            if class_assets:
                ef.add_constraint(
                    lambda w, assets=class_assets: sum(w[mu.index.get_loc(a)] for a in assets) >= min_weight
                )
                ef.add_constraint(
                    lambda w, assets=class_assets: sum(w[mu.index.get_loc(a)] for a in assets) <= max_weight
                )
        
        # Optimize for maximum Sharpe ratio
        weights = ef.max_sharpe(risk_free_rate=0.02)
        
        # Get performance
        performance = ef.portfolio_performance(verbose=False, risk_free_rate=0.02)
        
        return {
            'weights': dict(ef.clean_weights()),
            'expected_return': performance[0],
            'volatility': performance[1],
            'sharpe_ratio': performance[2],
            'model': 'markowitz'
        }
    
    def _optimize_black_litterman(self, client_data: Dict, market_data: Dict,
                                 constraints: Dict, views: Optional[Dict]) -> Dict:
        """Black-Litterman Optimization with market views"""
        S = market_data['cov_matrix']
        prices = market_data['prices']
        
        # Calculate market implied returns
        delta = 2.5  # Market risk aversion
        market_weights = prices / prices.sum()
        market_implied_returns = delta * S @ market_weights
        
        # Create Black-Litterman model
        bl = BlackLittermanModel(S, pi=market_implied_returns)
        
        # Add views if provided
        if views:
            # Example views format:
            # views = {
            #     'AAPL': 0.15,  # Expect 15% return
            #     'relative': [('AAPL', 'MSFT', 0.02)]  # AAPL outperforms MSFT by 2%
            # }
            if 'absolute' in views:
                for asset, expected_return in views['absolute'].items():
                    if asset in market_implied_returns.index:
                        bl.add_view(asset, expected_return, confidence=0.7)
            
            if 'relative' in views:
                for asset1, asset2, outperformance in views['relative']:
                    if asset1 in market_implied_returns.index and asset2 in market_implied_returns.index:
                        bl.add_view([asset1, asset2], [1, -1], outperformance, confidence=0.8)
        
        # Get Black-Litterman returns
        bl_returns = bl.bl_returns()
        
        # Optimize using BL returns
        ef = EfficientFrontier(bl_returns, S)
        
        # Apply constraints (same as Markowitz)
        for asset in bl_returns.index:
            ef.add_constraint(lambda w, asset=asset: w[bl_returns.index.get_loc(asset)] >= constraints['min_weight'])
            ef.add_constraint(lambda w, asset=asset: w[bl_returns.index.get_loc(asset)] <= constraints['max_weight'])
        
        weights = ef.max_sharpe(risk_free_rate=0.02)
        performance = ef.portfolio_performance(verbose=False, risk_free_rate=0.02)
        
        return {
            'weights': dict(ef.clean_weights()),
            'expected_return': performance[0],
            'volatility': performance[1],
            'sharpe_ratio': performance[2],
            'model': 'black_litterman',
            'bl_returns': bl_returns.to_dict()
        }
    
    def _optimize_risk_parity(self, client_data: Dict, market_data: Dict,
                            constraints: Dict, views: Optional[Dict]) -> Dict:
        """Hierarchical Risk Parity Optimization"""
        mu = market_data['expected_returns']
        S = market_data['cov_matrix']
        prices = market_data['prices']
        
        # Use HRP from PyPortfolioOpt
        hrp = HRPOpt(mu, S)
        weights = hrp.optimize()
        
        # Calculate performance metrics
        portfolio_return = np.sum(weights * mu)
        portfolio_vol = np.sqrt(weights @ S @ weights)
        sharpe_ratio = (portfolio_return - 0.02) / portfolio_vol
        
        return {
            'weights': dict(hrp.clean_weights()),
            'expected_return': portfolio_return,
            'volatility': portfolio_vol,
            'sharpe_ratio': sharpe_ratio,
            'model': 'risk_parity'
        }
    
    def _optimize_max_sharpe(self, client_data: Dict, market_data: Dict,
                           constraints: Dict, views: Optional[Dict]) -> Dict:
        """Maximize Sharpe Ratio with custom constraints"""
        mu = market_data['expected_returns']
        S = market_data['cov_matrix']
        
        # Use cvxpy for more control
        n = len(mu)
        w = cp.Variable(n)
        
        # Expected return and risk
        ret = mu.values @ w
        risk = cp.quad_form(w, S.values)
        
        # Constraints
        constraints_list = [
            cp.sum(w) == 1,
            w >= constraints['min_weight'],
            w <= constraints['max_weight']
        ]
        
        # Risk constraint based on client profile
        max_risk = client_data['risk_profile']['volatilityTolerance']
        constraints_list.append(risk <= max_risk ** 2)
        
        # Optimize
        # Max Sharpe is non-convex, so we use a convex approximation
        # Maximize return for a given risk level
        objective = cp.Maximize(ret - 0.5 * risk)  # Risk-adjusted return
        
        prob = cp.Problem(objective, constraints_list)
        prob.solve()
        
        if w.value is not None:
            weights_dict = {asset: float(w.value[i]) for i, asset in enumerate(mu.index)}
            portfolio_return = float(ret.value)
            portfolio_vol = float(np.sqrt(risk.value))
            sharpe_ratio = (portfolio_return - 0.02) / portfolio_vol
            
            return {
                'weights': weights_dict,
                'expected_return': portfolio_return,
                'volatility': portfolio_vol,
                'sharpe_ratio': sharpe_ratio,
                'model': 'max_sharpe'
            }
        else:
            raise ValueError("Optimization failed to converge")
    
    def _optimize_min_volatility(self, client_data: Dict, market_data: Dict,
                               constraints: Dict, views: Optional[Dict]) -> Dict:
        """Minimum Volatility Portfolio"""
        mu = market_data['expected_returns']
        S = market_data['cov_matrix']
        
        ef = EfficientFrontier(mu, S)
        
        # Add constraints
        for asset in mu.index:
            ef.add_constraint(lambda w, asset=asset: w[mu.index.get_loc(asset)] >= constraints['min_weight'])
            ef.add_constraint(lambda w, asset=asset: w[mu.index.get_loc(asset)] <= constraints['max_weight'])
        
        # Optimize for minimum volatility
        weights = ef.min_volatility()
        performance = ef.portfolio_performance(verbose=False, risk_free_rate=0.02)
        
        return {
            'weights': dict(ef.clean_weights()),
            'expected_return': performance[0],
            'volatility': performance[1],
            'sharpe_ratio': performance[2],
            'model': 'min_volatility'
        }
    
    def _optimize_factor_based(self, client_data: Dict, market_data: Dict,
                             constraints: Dict, views: Optional[Dict]) -> Dict:
        """Factor-based optimization using Neo4j factor exposures"""
        # Get factor exposures from Neo4j
        with self.driver.session() as session:
            factor_data = session.run("""
                MATCH (a:Asset)-[e:HAS_FACTOR_EXPOSURE]->(f:Factor)
                WHERE a.symbol IN $symbols
                RETURN a.symbol as symbol, f.name as factor, e.exposure as exposure
            """, symbols=list(market_data['expected_returns'].index)).data()
        
        # Create factor exposure matrix
        symbols = list(market_data['expected_returns'].index)
        factors = list(set(row['factor'] for row in factor_data))
        
        exposure_matrix = pd.DataFrame(0, index=symbols, columns=factors)
        for row in factor_data:
            exposure_matrix.loc[row['symbol'], row['factor']] = row['exposure']
        
        # Target factor exposures based on client profile
        target_exposures = self._get_target_factor_exposures(client_data)
        
        # Optimize to match target factor exposures while maximizing return
        n = len(symbols)
        w = cp.Variable(n)
        
        # Portfolio factor exposures
        portfolio_exposures = exposure_matrix.T.values @ w
        
        # Objective: Minimize tracking error to target exposures + maximize return
        factor_tracking = cp.sum_squares(portfolio_exposures - list(target_exposures.values()))
        ret = market_data['expected_returns'].values @ w
        
        objective = cp.Maximize(ret - 10 * factor_tracking)  # Balance return and factor matching
        
        constraints_list = [
            cp.sum(w) == 1,
            w >= constraints['min_weight'],
            w <= constraints['max_weight']
        ]
        
        prob = cp.Problem(objective, constraints_list)
        prob.solve()
        
        if w.value is not None:
            weights_dict = {asset: float(w.value[i]) for i, asset in enumerate(symbols)}
            
            # Calculate metrics
            S = market_data['cov_matrix']
            portfolio_return = float(market_data['expected_returns'].values @ w.value)
            portfolio_vol = float(np.sqrt(w.value @ S.values @ w.value))
            sharpe_ratio = (portfolio_return - 0.02) / portfolio_vol
            
            # Calculate achieved factor exposures
            achieved_exposures = {
                factor: float(exposure_matrix[factor].values @ w.value)
                for factor in factors
            }
            
            return {
                'weights': weights_dict,
                'expected_return': portfolio_return,
                'volatility': portfolio_vol,
                'sharpe_ratio': sharpe_ratio,
                'model': 'factor_based',
                'factor_exposures': achieved_exposures,
                'target_exposures': target_exposures
            }
        else:
            raise ValueError("Factor optimization failed to converge")
    
    def _optimize_cvar(self, client_data: Dict, market_data: Dict,
                      constraints: Dict, views: Optional[Dict]) -> Dict:
        """Conditional Value at Risk (CVaR) Optimization using Riskfolio-Lib"""
        # Prepare data for Riskfolio
        symbols = list(market_data['expected_returns'].index)
        mu = market_data['expected_returns']
        S = market_data['cov_matrix']
        
        # Create portfolio object
        port = rp.Portfolio(returns=pd.DataFrame(index=pd.date_range('2020-01-01', periods=252, freq='D')))
        
        # Set estimates
        port.mu = mu
        port.cov = S
        port.semidev = S  # Simplified
        
        # Constraints
        port.upperlng = constraints['max_weight']
        port.lowerlng = constraints['min_weight']
        
        # Optimize for minimum CVaR
        w = port.optimization(
            model='CVaR',
            rm='CVaR',
            obj='MinRisk',
            rf=0.02,
            hist=False
        )
        
        if w is not None:
            weights_dict = {asset: float(w.loc[asset, 0]) for asset in symbols if w.loc[asset, 0] > 0.001}
            
            # Calculate metrics
            portfolio_return = float(mu @ w.values.flatten())
            portfolio_vol = float(np.sqrt(w.T @ S @ w).iloc[0, 0])
            sharpe_ratio = (portfolio_return - 0.02) / portfolio_vol
            
            return {
                'weights': weights_dict,
                'expected_return': portfolio_return,
                'volatility': portfolio_vol,
                'sharpe_ratio': sharpe_ratio,
                'model': 'cvar',
                'risk_measure': 'CVaR@95%'
            }
        else:
            raise ValueError("CVaR optimization failed")
    
    def _get_target_factor_exposures(self, client_data: Dict) -> Dict[str, float]:
        """Get target factor exposures based on client profile"""
        risk_tolerance = client_data['client']['riskTolerance']
        
        if risk_tolerance == 'Conservative':
            return {
                'Value': 0.7,
                'Quality': 0.8,
                'Low Volatility': 0.9,
                'Growth': 0.2,
                'Momentum': 0.1
            }
        elif risk_tolerance == 'Moderate':
            return {
                'Value': 0.5,
                'Quality': 0.6,
                'Growth': 0.5,
                'Momentum': 0.4,
                'Low Volatility': 0.4
            }
        else:  # Aggressive
            return {
                'Growth': 0.8,
                'Momentum': 0.7,
                'Quality': 0.4,
                'Value': 0.3,
                'Low Volatility': 0.1
            }
    
    def _post_process_results(self, results: Dict, client_data: Dict, market_data: Dict) -> Dict:
        """Post-process optimization results"""
        weights = results['weights']
        prices = market_data['prices']
        portfolio_value = sum(client_data['portfolios'][0]['totalValue'] for p in client_data['portfolios']) if client_data['portfolios'] else 1000000
        
        # Calculate discrete allocation
        da = DiscreteAllocation(weights, prices, total_portfolio_value=portfolio_value)
        allocation, leftover = da.greedy_portfolio()
        
        # Add additional metrics
        results['discrete_allocation'] = allocation
        results['leftover_cash'] = leftover
        results['number_of_assets'] = len([w for w in weights.values() if w > 0.001])
        
        # Risk contribution
        S = market_data['cov_matrix']
        w = pd.Series(weights)
        portfolio_vol = results['volatility']
        marginal_contrib = S @ w
        contrib = w * marginal_contrib / portfolio_vol
        results['risk_contribution'] = contrib.to_dict()
        
        # Diversification ratio
        weighted_avg_vol = np.sum(w * np.sqrt(np.diag(S)))
        results['diversification_ratio'] = weighted_avg_vol / portfolio_vol
        
        return results
    
    def _save_optimization_results(self, client_id: str, results: Dict) -> None:
        """Save optimization results to Neo4j"""
        with self.driver.session() as session:
            # Create optimization node
            session.run("""
                CREATE (opt:Optimization {
                    optimizationId: randomUUID(),
                    clientId: $clientId,
                    model: $model,
                    timestamp: datetime(),
                    expectedReturn: $expectedReturn,
                    volatility: $volatility,
                    sharpeRatio: $sharpeRatio,
                    diversificationRatio: $diversificationRatio
                })
                WITH opt
                MATCH (c:Client {clientId: $clientId})
                CREATE (c)-[:HAS_OPTIMIZATION]->(opt)
            """, clientId=client_id, **results)
            
            # Save recommended weights
            for symbol, weight in results['weights'].items():
                if weight > 0.001:  # Only save meaningful weights
                    session.run("""
                        MATCH (opt:Optimization {clientId: $clientId})
                        WHERE opt.timestamp = datetime()
                        MATCH (a:Asset {symbol: $symbol})
                        CREATE (opt)-[:RECOMMENDS_WEIGHT {
                            weight: $weight,
                            shares: $shares,
                            value: $value
                        }]->(a)
                    """, clientId=client_id, symbol=symbol, weight=weight,
                        shares=results['discrete_allocation'].get(symbol, 0),
                        value=results['discrete_allocation'].get(symbol, 0) * 
                              results.get('prices', {}).get(symbol, 0))
    
    def compare_optimization_models(self, client_id: str, models: List[str] = None) -> pd.DataFrame:
        """Compare different optimization models for a client"""
        if models is None:
            models = ['markowitz', 'black_litterman', 'risk_parity', 'min_volatility']
        
        results = []
        for model in models:
            try:
                opt_result = self.optimize_portfolio(client_id, model)
                results.append({
                    'Model': model,
                    'Expected Return': f"{opt_result['expected_return']*100:.2f}%",
                    'Volatility': f"{opt_result['volatility']*100:.2f}%",
                    'Sharpe Ratio': f"{opt_result['sharpe_ratio']:.3f}",
                    'Diversification': f"{opt_result.get('diversification_ratio', 0):.2f}",
                    'Assets': opt_result['number_of_assets']
                })
            except Exception as e:
                print(f"Error with {model}: {e}")
                
        return pd.DataFrame(results)


# Example usage
if __name__ == "__main__":
    from neo4j import GraphDatabase
    
    # Neo4j connection
    uri = "neo4j+s://1c34ddb0.databases.neo4j.io"
    driver = GraphDatabase.driver(uri, auth=("neo4j", "Sw-DAlLfV1_Rw4lFlXAxl0DoN4lCauWrp29IZnfq_zM"))
    
    optimizer = AdvancedPortfolioOptimizer(driver)
    
    # Example: Optimize for a client
    client_id = "CLI_100001"
    
    # Run Markowitz optimization
    print("Running Markowitz Optimization...")
    results = optimizer.optimize_portfolio(client_id, 'markowitz')
    print(f"Expected Return: {results['expected_return']*100:.2f}%")
    print(f"Volatility: {results['volatility']*100:.2f}%")
    print(f"Sharpe Ratio: {results['sharpe_ratio']:.3f}")
    print("\nTop Holdings:")
    for symbol, weight in sorted(results['weights'].items(), key=lambda x: x[1], reverse=True)[:5]:
        if weight > 0.01:
            print(f"  {symbol}: {weight*100:.1f}%")
    
    # Compare models
    print("\n\nComparing Optimization Models:")
    comparison = optimizer.compare_optimization_models(client_id)
    print(comparison)
    
    driver.close()