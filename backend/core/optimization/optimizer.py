"""
Portfolio optimization using modern portfolio theory and advanced methods
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import cvxpy as cp
from pypfopt import EfficientFrontier, risk_models, expected_returns
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices
import warnings
warnings.filterwarnings('ignore')

class PortfolioOptimizer:
    def __init__(self, returns_data: pd.DataFrame, prices_data: pd.DataFrame):
        """
        Initialize optimizer with historical returns and current prices
        
        Args:
            returns_data: DataFrame with asset returns (assets as columns)
            prices_data: DataFrame with current asset prices
        """
        self.returns = returns_data
        self.prices = prices_data
        self.expected_returns = None
        self.cov_matrix = None
        
    def calculate_expected_returns(self, method: str = "mean") -> pd.Series:
        """Calculate expected returns using various methods"""
        if method == "mean":
            self.expected_returns = expected_returns.mean_historical_return(self.returns)
        elif method == "ema":
            self.expected_returns = expected_returns.ema_historical_return(self.returns)
        elif method == "capm":
            self.expected_returns = expected_returns.capm_return(self.returns)
        
        return self.expected_returns
    
    def calculate_risk_matrix(self, method: str = "sample") -> pd.DataFrame:
        """Calculate covariance matrix using various methods"""
        if method == "sample":
            self.cov_matrix = risk_models.sample_cov(self.returns)
        elif method == "semicovariance":
            self.cov_matrix = risk_models.semicovariance(self.returns)
        elif method == "exp_cov":
            self.cov_matrix = risk_models.exp_cov(self.returns)
        elif method == "ledoit_wolf":
            self.cov_matrix, _ = risk_models.ledoit_wolf(self.returns)
        
        return self.cov_matrix
    
    def optimize_portfolio(
        self,
        objective: str = "sharpe",
        constraints: Optional[Dict] = None,
        risk_tolerance: str = "moderate"
    ) -> Dict[str, float]:
        """
        Optimize portfolio weights based on objective and constraints
        
        Args:
            objective: Optimization objective (sharpe, min_volatility, max_return, etc.)
            constraints: Additional constraints (sector limits, etc.)
            risk_tolerance: Client risk tolerance level
        
        Returns:
            Dictionary of asset symbols to weights
        """
        if self.expected_returns is None:
            self.calculate_expected_returns()
        if self.cov_matrix is None:
            self.calculate_risk_matrix()
        
        # Initialize Efficient Frontier
        ef = EfficientFrontier(self.expected_returns, self.cov_matrix)
        
        # Apply basic constraints based on risk tolerance
        if risk_tolerance == "conservative":
            ef.add_constraint(lambda w: w <= 0.15)  # Max 15% per asset
        elif risk_tolerance == "moderate":
            ef.add_constraint(lambda w: w <= 0.25)  # Max 25% per asset
        else:  # aggressive
            ef.add_constraint(lambda w: w <= 0.40)  # Max 40% per asset
        
        # Apply additional constraints
        if constraints:
            if "sector_limits" in constraints:
                # TODO: Implement sector constraints
                pass
            if "min_assets" in constraints:
                # Ensure minimum number of assets
                ef.add_constraint(lambda w: cp.sum(w > 0.01) >= constraints["min_assets"])
        
        # Optimize based on objective
        if objective == "sharpe":
            weights = ef.max_sharpe()
        elif objective == "min_volatility":
            weights = ef.min_volatility()
        elif objective == "max_return":
            # Set a target volatility based on risk tolerance
            target_vol = {"conservative": 0.10, "moderate": 0.15, "aggressive": 0.25}
            ef.efficient_risk(target_vol[risk_tolerance])
            weights = ef.clean_weights()
        elif objective == "risk_parity":
            weights = self._risk_parity_optimization()
        elif objective == "black_litterman":
            # TODO: Implement Black-Litterman
            weights = ef.max_sharpe()
        else:
            weights = ef.max_sharpe()
        
        # Clean weights (remove very small allocations)
        cleaned_weights = ef.clean_weights(cutoff=0.01, rounding=3)
        
        # Calculate portfolio metrics
        self.portfolio_performance = ef.portfolio_performance(verbose=False)
        
        return cleaned_weights
    
    def _risk_parity_optimization(self) -> Dict[str, float]:
        """Implement risk parity optimization"""
        n_assets = len(self.expected_returns)
        
        # Define optimization variables
        w = cp.Variable(n_assets)
        
        # Risk parity objective (equal risk contribution)
        portfolio_risk = cp.quad_form(w, self.cov_matrix.values)
        
        # Constraints
        constraints = [
            cp.sum(w) == 1,  # Weights sum to 1
            w >= 0,          # Long only
            w <= 0.40        # Max position size
        ]
        
        # Solve
        prob = cp.Problem(cp.Minimize(portfolio_risk), constraints)
        prob.solve()
        
        # Convert to dictionary
        weights = {}
        for i, asset in enumerate(self.expected_returns.index):
            if w.value[i] > 0.01:
                weights[asset] = round(w.value[i], 3)
        
        return weights
    
    def get_discrete_allocation(
        self,
        weights: Dict[str, float],
        total_portfolio_value: float
    ) -> Tuple[Dict[str, int], float]:
        """
        Convert optimal weights to discrete share allocations
        
        Args:
            weights: Optimal portfolio weights
            total_portfolio_value: Total amount to invest
        
        Returns:
            Tuple of (shares_dict, leftover_cash)
        """
        latest_prices = get_latest_prices(self.prices)
        
        da = DiscreteAllocation(
            weights,
            latest_prices,
            total_portfolio_value
        )
        
        allocation, leftover = da.greedy_portfolio()
        
        return allocation, leftover
    
    def calculate_portfolio_metrics(
        self,
        weights: Dict[str, float]
    ) -> Dict[str, float]:
        """Calculate detailed portfolio metrics"""
        weights_array = np.array([weights.get(asset, 0) for asset in self.expected_returns.index])
        
        # Expected return
        portfolio_return = np.dot(weights_array, self.expected_returns)
        
        # Portfolio volatility
        portfolio_variance = np.dot(weights_array.T, np.dot(self.cov_matrix, weights_array))
        portfolio_volatility = np.sqrt(portfolio_variance)
        
        # Sharpe ratio (assuming 2% risk-free rate)
        sharpe_ratio = (portfolio_return - 0.02) / portfolio_volatility
        
        # Maximum drawdown estimation
        max_drawdown = self._estimate_max_drawdown(weights_array)
        
        # Value at Risk (95% confidence)
        var_95 = -1.65 * portfolio_volatility * np.sqrt(252)  # Annualized
        
        # Conditional Value at Risk
        cvar_95 = -2.06 * portfolio_volatility * np.sqrt(252)  # Annualized
        
        return {
            "expected_return": float(portfolio_return),
            "volatility": float(portfolio_volatility),
            "sharpe_ratio": float(sharpe_ratio),
            "max_drawdown": float(max_drawdown),
            "var_95": float(var_95),
            "cvar_95": float(cvar_95)
        }
    
    def _estimate_max_drawdown(self, weights: np.ndarray) -> float:
        """Estimate maximum drawdown based on historical data"""
        portfolio_returns = self.returns.dot(weights)
        cumulative_returns = (1 + portfolio_returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        return float(drawdown.min())
    
    def optimize_with_views(
        self,
        market_views: Dict[str, float],
        view_confidences: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Black-Litterman optimization with market views
        
        Args:
            market_views: Dictionary of asset to expected return views
            view_confidences: Confidence levels for each view (0-1)
        
        Returns:
            Optimized weights incorporating views
        """
        # TODO: Implement Black-Litterman model
        # For now, return standard optimization
        return self.optimize_portfolio(objective="sharpe")