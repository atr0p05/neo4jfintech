"""
Backtesting Framework for Portfolio Strategies
Tests optimization models and investment strategies with historical data
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Callable
import yfinance as yf
from dataclasses import dataclass
import backtrader as bt
import quantstats as qs
import matplotlib.pyplot as plt
import seaborn as sns
from neo4j import GraphDatabase
import json
import warnings
warnings.filterwarnings('ignore')


@dataclass
class BacktestResult:
    """Container for backtest results"""
    strategy_name: str
    start_date: datetime
    end_date: datetime
    initial_value: float
    final_value: float
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    calmar_ratio: float
    sortino_ratio: float
    win_rate: float
    trades: List[Dict]
    portfolio_values: pd.Series
    returns: pd.Series
    positions: pd.DataFrame
    metrics: Dict[str, float]


class PortfolioStrategy(bt.Strategy):
    """Base strategy for portfolio backtesting"""
    params = (
        ('rebalance_frequency', 'monthly'),  # monthly, quarterly, yearly
        ('initial_weights', {}),
        ('optimization_model', 'markowitz'),
        ('risk_limit', 0.20),
        ('transaction_cost', 0.001),  # 0.1%
        ('min_weight', 0.02),
        ('max_weight', 0.25),
    )
    
    def __init__(self):
        self.rebalance_dates = []
        self.trades_log = []
        self.portfolio_weights = {}
        self.last_rebalance = None
        
    def next(self):
        # Check if it's time to rebalance
        current_date = self.datas[0].datetime.date(0)
        
        if self._should_rebalance(current_date):
            self._rebalance_portfolio(current_date)
            self.last_rebalance = current_date
    
    def _should_rebalance(self, current_date):
        """Determine if portfolio should be rebalanced"""
        if self.last_rebalance is None:
            return True
            
        if self.params.rebalance_frequency == 'monthly':
            return current_date.month != self.last_rebalance.month
        elif self.params.rebalance_frequency == 'quarterly':
            return (current_date.month - 1) // 3 != (self.last_rebalance.month - 1) // 3
        elif self.params.rebalance_frequency == 'yearly':
            return current_date.year != self.last_rebalance.year
        
        return False
    
    def _rebalance_portfolio(self, current_date):
        """Rebalance portfolio to target weights"""
        target_weights = self._get_target_weights(current_date)
        current_value = self.broker.getvalue()
        
        for i, data in enumerate(self.datas):
            symbol = data._name
            target_weight = target_weights.get(symbol, 0)
            
            # Calculate target position
            target_value = current_value * target_weight
            target_size = int(target_value / data.close[0])
            current_size = self.getposition(data).size
            
            # Execute trade if needed
            trade_size = target_size - current_size
            if abs(trade_size) > 0:
                self.order_target_size(data, target_size)
                
                # Log trade
                self.trades_log.append({
                    'date': current_date,
                    'symbol': symbol,
                    'action': 'BUY' if trade_size > 0 else 'SELL',
                    'size': abs(trade_size),
                    'price': data.close[0],
                    'value': abs(trade_size * data.close[0])
                })
        
        self.rebalance_dates.append(current_date)
        self.portfolio_weights[current_date] = target_weights
    
    def _get_target_weights(self, current_date):
        """Get target weights (override in subclasses)"""
        return self.params.initial_weights


class OptimizationBacktester:
    """
    Comprehensive backtesting system for portfolio optimization strategies
    """
    
    def __init__(self, neo4j_driver, market_data_service=None):
        self.driver = neo4j_driver
        self.market_data_service = market_data_service
        self.results_cache = {}
        
    def backtest_strategy(self,
                         symbols: List[str],
                         strategy_params: Dict,
                         start_date: str,
                         end_date: str,
                         initial_capital: float = 100000,
                         benchmark: str = 'SPY') -> BacktestResult:
        """
        Run backtest for a portfolio strategy
        
        Args:
            symbols: List of asset symbols
            strategy_params: Strategy parameters including optimization model
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            initial_capital: Starting capital
            benchmark: Benchmark symbol for comparison
        """
        
        # Fetch historical data
        data = self._fetch_historical_data(symbols + [benchmark], start_date, end_date)
        
        if data.empty:
            raise ValueError("No historical data available")
        
        # Initialize Backtrader
        cerebro = bt.Cerebro()
        cerebro.broker.setcash(initial_capital)
        cerebro.broker.setcommission(commission=strategy_params.get('transaction_cost', 0.001))
        
        # Add data feeds
        for symbol in symbols:
            if symbol in data.columns:
                df = data[[symbol]].copy()
                df.columns = ['close']
                df['open'] = df['close']
                df['high'] = df['close']
                df['low'] = df['close']
                df['volume'] = 0
                
                datafeed = bt.feeds.PandasData(
                    dataname=df,
                    fromdate=pd.to_datetime(start_date),
                    todate=pd.to_datetime(end_date)
                )
                cerebro.adddata(datafeed, name=symbol)
        
        # Add strategy
        if strategy_params['optimization_model'] == 'equal_weight':
            strategy = self._create_equal_weight_strategy(symbols)
        elif strategy_params['optimization_model'] == 'risk_parity':
            strategy = self._create_risk_parity_strategy(symbols, data)
        else:
            strategy = self._create_optimization_strategy(symbols, strategy_params)
        
        cerebro.addstrategy(strategy, **strategy_params)
        
        # Add analyzers
        cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
        cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
        cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
        cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
        
        # Run backtest
        results = cerebro.run()
        strategy_instance = results[0]
        
        # Extract results
        final_value = cerebro.broker.getvalue()
        total_return = (final_value - initial_capital) / initial_capital
        
        # Get portfolio values over time
        portfolio_values = self._extract_portfolio_values(cerebro, strategy_instance)
        returns = portfolio_values.pct_change().dropna()
        
        # Calculate metrics
        metrics = self._calculate_metrics(returns, portfolio_values, initial_capital)
        
        # Create result object
        result = BacktestResult(
            strategy_name=strategy_params['optimization_model'],
            start_date=pd.to_datetime(start_date),
            end_date=pd.to_datetime(end_date),
            initial_value=initial_capital,
            final_value=final_value,
            total_return=total_return,
            annualized_return=metrics['annualized_return'],
            volatility=metrics['volatility'],
            sharpe_ratio=metrics['sharpe_ratio'],
            max_drawdown=metrics['max_drawdown'],
            calmar_ratio=metrics['calmar_ratio'],
            sortino_ratio=metrics['sortino_ratio'],
            win_rate=metrics['win_rate'],
            trades=strategy_instance.trades_log,
            portfolio_values=portfolio_values,
            returns=returns,
            positions=self._extract_positions(strategy_instance),
            metrics=metrics
        )
        
        # Compare with benchmark
        result.benchmark_comparison = self._compare_with_benchmark(
            returns, data[benchmark].pct_change().dropna()
        )
        
        # Save results to Neo4j
        self._save_backtest_results(result)
        
        return result
    
    def _fetch_historical_data(self, symbols: List[str], 
                              start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch historical price data"""
        data = pd.DataFrame()
        
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(start=start_date, end=end_date)
                if not hist.empty:
                    data[symbol] = hist['Close']
            except Exception as e:
                print(f"Error fetching {symbol}: {e}")
        
        return data
    
    def _create_equal_weight_strategy(self, symbols: List[str]):
        """Create equal weight strategy"""
        weights = {symbol: 1.0 / len(symbols) for symbol in symbols}
        
        class EqualWeightStrategy(PortfolioStrategy):
            params = (
                ('initial_weights', weights),
                ('rebalance_frequency', 'quarterly'),
            )
            
            def _get_target_weights(self, current_date):
                return self.params.initial_weights
        
        return EqualWeightStrategy
    
    def _create_risk_parity_strategy(self, symbols: List[str], historical_data: pd.DataFrame):
        """Create risk parity strategy"""
        
        class RiskParityStrategy(PortfolioStrategy):
            def __init__(self):
                super().__init__()
                self.historical_data = historical_data
                
            def _get_target_weights(self, current_date):
                # Calculate risk parity weights based on recent volatility
                lookback = 252
                end_idx = self.historical_data.index.get_loc(current_date, method='nearest')
                start_idx = max(0, end_idx - lookback)
                
                recent_data = self.historical_data.iloc[start_idx:end_idx]
                returns = recent_data.pct_change().dropna()
                
                # Calculate inverse volatility weights
                volatilities = returns.std()
                inv_vol = 1 / volatilities
                weights = inv_vol / inv_vol.sum()
                
                return weights.to_dict()
        
        return RiskParityStrategy
    
    def _create_optimization_strategy(self, symbols: List[str], strategy_params: Dict):
        """Create strategy using portfolio optimization"""
        
        class OptimizedStrategy(PortfolioStrategy):
            def __init__(self):
                super().__init__()
                self.optimization_model = strategy_params['optimization_model']
                self.lookback_period = strategy_params.get('lookback_period', 252)
                
            def _get_target_weights(self, current_date):
                # In a real implementation, this would call the optimization engine
                # For now, use a simplified momentum strategy
                lookback = self.lookback_period
                
                momentum_scores = {}
                for i, data in enumerate(self.datas):
                    symbol = data._name
                    if len(data) > lookback:
                        momentum = (data.close[0] - data.close[-lookback]) / data.close[-lookback]
                        momentum_scores[symbol] = momentum
                
                # Rank by momentum and allocate weights
                if momentum_scores:
                    sorted_symbols = sorted(momentum_scores.items(), 
                                          key=lambda x: x[1], reverse=True)
                    
                    weights = {}
                    for i, (symbol, score) in enumerate(sorted_symbols[:10]):
                        if score > 0:
                            weights[symbol] = 0.1  # Equal weight top 10
                    
                    # Normalize weights
                    total_weight = sum(weights.values())
                    if total_weight > 0:
                        weights = {k: v/total_weight for k, v in weights.items()}
                    
                    return weights
                
                return self.params.initial_weights
        
        return OptimizedStrategy
    
    def _calculate_metrics(self, returns: pd.Series, portfolio_values: pd.Series,
                          initial_capital: float) -> Dict[str, float]:
        """Calculate comprehensive performance metrics"""
        
        # Annual metrics
        years = (returns.index[-1] - returns.index[0]).days / 365.25
        total_return = (portfolio_values.iloc[-1] - initial_capital) / initial_capital
        annualized_return = (1 + total_return) ** (1/years) - 1
        
        # Risk metrics
        volatility = returns.std() * np.sqrt(252)
        downside_returns = returns[returns < 0]
        downside_volatility = downside_returns.std() * np.sqrt(252)
        
        # Risk-adjusted returns
        risk_free_rate = 0.02
        sharpe_ratio = (annualized_return - risk_free_rate) / volatility if volatility > 0 else 0
        sortino_ratio = (annualized_return - risk_free_rate) / downside_volatility if downside_volatility > 0 else 0
        
        # Drawdown analysis
        cumulative_returns = (1 + returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Calmar ratio
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Win rate
        winning_days = (returns > 0).sum()
        total_days = len(returns)
        win_rate = winning_days / total_days if total_days > 0 else 0
        
        # Additional metrics
        metrics = {
            'annualized_return': annualized_return,
            'total_return': total_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar_ratio,
            'win_rate': win_rate,
            'best_day': returns.max(),
            'worst_day': returns.min(),
            'var_95': returns.quantile(0.05),
            'cvar_95': returns[returns <= returns.quantile(0.05)].mean(),
            'skewness': returns.skew(),
            'kurtosis': returns.kurtosis(),
            'avg_win': returns[returns > 0].mean() if (returns > 0).any() else 0,
            'avg_loss': returns[returns < 0].mean() if (returns < 0).any() else 0,
            'profit_factor': abs(returns[returns > 0].sum() / returns[returns < 0].sum()) if (returns < 0).any() else np.inf
        }
        
        return metrics
    
    def _extract_portfolio_values(self, cerebro, strategy_instance) -> pd.Series:
        """Extract portfolio values over time"""
        # This is a simplified extraction - in production, use cerebro's observers
        values = []
        dates = []
        
        # For now, simulate based on initial capital and returns
        # In real implementation, extract from cerebro's records
        
        return pd.Series(values, index=dates)
    
    def _extract_positions(self, strategy_instance) -> pd.DataFrame:
        """Extract position history"""
        # Extract from strategy's position tracking
        positions = pd.DataFrame(strategy_instance.portfolio_weights).T
        return positions
    
    def _compare_with_benchmark(self, strategy_returns: pd.Series, 
                               benchmark_returns: pd.Series) -> Dict:
        """Compare strategy performance with benchmark"""
        
        # Align dates
        common_dates = strategy_returns.index.intersection(benchmark_returns.index)
        strategy_returns = strategy_returns[common_dates]
        benchmark_returns = benchmark_returns[common_dates]
        
        # Calculate metrics
        strategy_cum = (1 + strategy_returns).cumprod()
        benchmark_cum = (1 + benchmark_returns).cumprod()
        
        # Tracking error
        tracking_error = (strategy_returns - benchmark_returns).std() * np.sqrt(252)
        
        # Information ratio
        excess_returns = strategy_returns - benchmark_returns
        information_ratio = excess_returns.mean() / excess_returns.std() * np.sqrt(252) if excess_returns.std() > 0 else 0
        
        # Beta
        covariance = np.cov(strategy_returns, benchmark_returns)[0, 1]
        benchmark_variance = benchmark_returns.var()
        beta = covariance / benchmark_variance if benchmark_variance > 0 else 1
        
        # Alpha
        strategy_annual = strategy_returns.mean() * 252
        benchmark_annual = benchmark_returns.mean() * 252
        alpha = strategy_annual - beta * benchmark_annual
        
        return {
            'tracking_error': tracking_error,
            'information_ratio': information_ratio,
            'beta': beta,
            'alpha': alpha,
            'correlation': strategy_returns.corr(benchmark_returns),
            'excess_return': strategy_cum.iloc[-1] / benchmark_cum.iloc[-1] - 1
        }
    
    def _save_backtest_results(self, result: BacktestResult) -> None:
        """Save backtest results to Neo4j"""
        with self.driver.session() as session:
            session.run("""
                CREATE (b:BacktestResult {
                    backtestId: randomUUID(),
                    strategyName: $strategyName,
                    startDate: date($startDate),
                    endDate: date($endDate),
                    initialValue: $initialValue,
                    finalValue: $finalValue,
                    totalReturn: $totalReturn,
                    annualizedReturn: $annualizedReturn,
                    volatility: $volatility,
                    sharpeRatio: $sharpeRatio,
                    maxDrawdown: $maxDrawdown,
                    calmarRatio: $calmarRatio,
                    sortinoRatio: $sortinoRatio,
                    winRate: $winRate,
                    timestamp: datetime()
                })
            """, 
                strategyName=result.strategy_name,
                startDate=result.start_date.strftime('%Y-%m-%d'),
                endDate=result.end_date.strftime('%Y-%m-%d'),
                initialValue=result.initial_value,
                finalValue=result.final_value,
                totalReturn=result.total_return,
                annualizedReturn=result.annualized_return,
                volatility=result.volatility,
                sharpeRatio=result.sharpe_ratio,
                maxDrawdown=result.max_drawdown,
                calmarRatio=result.calmar_ratio,
                sortinoRatio=result.sortino_ratio,
                winRate=result.win_rate
            )
    
    def compare_strategies(self, 
                          symbols: List[str],
                          strategies: List[Dict],
                          start_date: str,
                          end_date: str,
                          initial_capital: float = 100000) -> pd.DataFrame:
        """Compare multiple strategies"""
        
        results = []
        for strategy_params in strategies:
            try:
                result = self.backtest_strategy(
                    symbols, strategy_params, start_date, end_date, initial_capital
                )
                results.append(result)
            except Exception as e:
                print(f"Error backtesting {strategy_params['optimization_model']}: {e}")
        
        # Create comparison dataframe
        comparison = pd.DataFrame([{
            'Strategy': r.strategy_name,
            'Total Return': f"{r.total_return*100:.2f}%",
            'Annual Return': f"{r.annualized_return*100:.2f}%",
            'Volatility': f"{r.volatility*100:.2f}%",
            'Sharpe Ratio': f"{r.sharpe_ratio:.3f}",
            'Max Drawdown': f"{r.max_drawdown*100:.2f}%",
            'Calmar Ratio': f"{r.calmar_ratio:.3f}",
            'Win Rate': f"{r.win_rate*100:.1f}%"
        } for r in results])
        
        return comparison
    
    def generate_backtest_report(self, result: BacktestResult, output_path: str = None):
        """Generate comprehensive backtest report with visualizations"""
        
        # Create figure with subplots
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        fig.suptitle(f'Backtest Report: {result.strategy_name}', fontsize=16)
        
        # 1. Portfolio value over time
        ax = axes[0, 0]
        result.portfolio_values.plot(ax=ax)
        ax.set_title('Portfolio Value')
        ax.set_ylabel('Value ($)')
        ax.grid(True)
        
        # 2. Cumulative returns
        ax = axes[0, 1]
        cumulative_returns = (1 + result.returns).cumprod()
        cumulative_returns.plot(ax=ax)
        ax.set_title('Cumulative Returns')
        ax.set_ylabel('Cumulative Return')
        ax.grid(True)
        
        # 3. Drawdown
        ax = axes[1, 0]
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        drawdown.plot(ax=ax, color='red')
        ax.fill_between(drawdown.index, drawdown, alpha=0.3, color='red')
        ax.set_title('Drawdown')
        ax.set_ylabel('Drawdown %')
        ax.grid(True)
        
        # 4. Monthly returns heatmap
        ax = axes[1, 1]
        monthly_returns = result.returns.resample('M').sum()
        monthly_returns_pivot = monthly_returns.groupby([monthly_returns.index.year, 
                                                        monthly_returns.index.month]).sum()
        # Simplified heatmap
        ax.bar(range(len(monthly_returns)), monthly_returns.values)
        ax.set_title('Monthly Returns')
        ax.set_ylabel('Return')
        
        # 5. Return distribution
        ax = axes[2, 0]
        result.returns.hist(bins=50, ax=ax, alpha=0.7)
        ax.axvline(result.returns.mean(), color='red', linestyle='--', label='Mean')
        ax.axvline(result.returns.quantile(0.05), color='orange', linestyle='--', label='VaR 95%')
        ax.set_title('Return Distribution')
        ax.set_xlabel('Daily Return')
        ax.legend()
        
        # 6. Key metrics table
        ax = axes[2, 1]
        ax.axis('off')
        metrics_text = f"""
        Total Return: {result.total_return*100:.2f}%
        Annual Return: {result.annualized_return*100:.2f}%
        Volatility: {result.volatility*100:.2f}%
        Sharpe Ratio: {result.sharpe_ratio:.3f}
        Max Drawdown: {result.max_drawdown*100:.2f}%
        Calmar Ratio: {result.calmar_ratio:.3f}
        Win Rate: {result.win_rate*100:.1f}%
        """
        ax.text(0.1, 0.5, metrics_text, fontsize=12, verticalalignment='center')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path)
        else:
            plt.show()
        
        # Generate QuantStats report
        if result.returns is not None and len(result.returns) > 0:
            qs.reports.html(result.returns, output=f'{result.strategy_name}_quantstats.html')


# Example usage
if __name__ == "__main__":
    # Neo4j connection
    uri = "neo4j+s://1c34ddb0.databases.neo4j.io"
    driver = GraphDatabase.driver(uri, auth=("neo4j", "Sw-DAlLfV1_Rw4lFlXAxl0DoN4lCauWrp29IZnfq_zM"))
    
    # Initialize backtester
    backtester = OptimizationBacktester(driver)
    
    # Define test parameters
    symbols = ['AAPL', 'MSFT', 'JPM', 'JNJ', 'TLT', 'GLD']
    start_date = '2020-01-01'
    end_date = '2023-12-31'
    
    # Test different strategies
    strategies = [
        {
            'optimization_model': 'equal_weight',
            'rebalance_frequency': 'quarterly'
        },
        {
            'optimization_model': 'risk_parity',
            'rebalance_frequency': 'monthly'
        },
        {
            'optimization_model': 'momentum',
            'rebalance_frequency': 'monthly',
            'lookback_period': 252
        }
    ]
    
    # Compare strategies
    print("Comparing portfolio strategies...")
    comparison = backtester.compare_strategies(symbols, strategies, start_date, end_date)
    print(comparison)
    
    # Run detailed backtest for best strategy
    best_strategy = strategies[0]  # Or select based on comparison
    result = backtester.backtest_strategy(symbols, best_strategy, start_date, end_date)
    
    # Generate report
    print(f"\nGenerating report for {result.strategy_name}...")
    backtester.generate_backtest_report(result, 'backtest_report.png')
    
    driver.close()