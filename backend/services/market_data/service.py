"""
Market Data Integration Service
Fetches real-time and historical data from Yahoo Finance, Alpha Vantage, and other sources
Includes caching, error handling, and Neo4j synchronization
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import asyncio
import aiohttp
from neo4j import GraphDatabase
import redis
import json
import logging
from functools import lru_cache
import pandas_datareader as pdr
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MarketDataService:
    """
    Comprehensive market data service with multiple providers
    """
    
    def __init__(self, neo4j_driver, redis_client=None, alpha_vantage_key=None):
        self.driver = neo4j_driver
        self.redis = redis_client or redis.Redis(decode_responses=True)
        self.alpha_vantage_key = alpha_vantage_key
        self.cache_ttl = 300  # 5 minutes for real-time data
        self.historical_cache_ttl = 86400  # 24 hours for historical data
        
    def fetch_real_time_quotes(self, symbols: List[str]) -> Dict[str, Dict]:
        """Fetch real-time quotes for multiple symbols"""
        quotes = {}
        
        # Batch fetch from Yahoo Finance
        tickers = yf.Tickers(' '.join(symbols))
        
        with ThreadPoolExecutor(max_workers=10) as executor:
            future_to_symbol = {
                executor.submit(self._fetch_single_quote, symbol, tickers.tickers[symbol]): symbol
                for symbol in symbols
            }
            
            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    quote = future.result()
                    if quote:
                        quotes[symbol] = quote
                except Exception as e:
                    logger.error(f"Error fetching quote for {symbol}: {e}")
        
        # Update Neo4j with latest prices
        self._update_neo4j_prices(quotes)
        
        return quotes
    
    def _fetch_single_quote(self, symbol: str, ticker) -> Optional[Dict]:
        """Fetch quote for a single symbol with caching"""
        # Check cache first
        cache_key = f"quote:{symbol}"
        cached = self.redis.get(cache_key)
        if cached:
            return json.loads(cached)
        
        try:
            info = ticker.info
            fast_info = ticker.fast_info
            
            quote = {
                'symbol': symbol,
                'price': fast_info.get('lastPrice', info.get('regularMarketPrice', 0)),
                'previousClose': info.get('previousClose', 0),
                'change': 0,
                'changePercent': 0,
                'volume': fast_info.get('lastVolume', info.get('volume', 0)),
                'marketCap': fast_info.get('marketCap', info.get('marketCap', 0)),
                'dayHigh': info.get('dayHigh', 0),
                'dayLow': info.get('dayLow', 0),
                'fiftyTwoWeekHigh': info.get('fiftyTwoWeekHigh', 0),
                'fiftyTwoWeekLow': info.get('fiftyTwoWeekLow', 0),
                'bid': info.get('bid', 0),
                'ask': info.get('ask', 0),
                'bidSize': info.get('bidSize', 0),
                'askSize': info.get('askSize', 0),
                'timestamp': datetime.now().isoformat()
            }
            
            # Calculate change
            if quote['previousClose'] > 0:
                quote['change'] = quote['price'] - quote['previousClose']
                quote['changePercent'] = (quote['change'] / quote['previousClose']) * 100
            
            # Cache the result
            self.redis.setex(cache_key, self.cache_ttl, json.dumps(quote))
            
            return quote
            
        except Exception as e:
            logger.error(f"Error fetching quote for {symbol}: {e}")
            return None
    
    def fetch_historical_data(self, symbol: str, period: str = "1y", 
                            interval: str = "1d") -> pd.DataFrame:
        """Fetch historical price data"""
        cache_key = f"history:{symbol}:{period}:{interval}"
        
        # Check cache
        cached = self.redis.get(cache_key)
        if cached:
            return pd.read_json(cached)
        
        try:
            ticker = yf.Ticker(symbol)
            history = ticker.history(period=period, interval=interval)
            
            if not history.empty:
                # Clean the data
                history = history[['Open', 'High', 'Low', 'Close', 'Volume']]
                history.index = history.index.tz_localize(None)
                
                # Cache the result
                self.redis.setex(cache_key, self.historical_cache_ttl, history.to_json())
                
                return history
            
        except Exception as e:
            logger.error(f"Error fetching historical data for {symbol}: {e}")
            
        return pd.DataFrame()
    
    def calculate_technical_indicators(self, symbol: str, 
                                     history: Optional[pd.DataFrame] = None) -> Dict:
        """Calculate technical indicators for a symbol"""
        if history is None:
            history = self.fetch_historical_data(symbol)
        
        if history.empty:
            return {}
        
        close = history['Close']
        high = history['High']
        low = history['Low']
        volume = history['Volume']
        
        indicators = {}
        
        # Moving averages
        indicators['sma_20'] = close.rolling(window=20).mean().iloc[-1]
        indicators['sma_50'] = close.rolling(window=50).mean().iloc[-1]
        indicators['sma_200'] = close.rolling(window=200).mean().iloc[-1]
        indicators['ema_12'] = close.ewm(span=12).mean().iloc[-1]
        indicators['ema_26'] = close.ewm(span=26).mean().iloc[-1]
        
        # MACD
        macd_line = indicators['ema_12'] - indicators['ema_26']
        signal_line = close.ewm(span=9).mean().iloc[-1]
        indicators['macd'] = macd_line
        indicators['macd_signal'] = signal_line
        indicators['macd_histogram'] = macd_line - signal_line
        
        # RSI
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        indicators['rsi'] = (100 - (100 / (1 + rs))).iloc[-1]
        
        # Bollinger Bands
        sma_20 = close.rolling(window=20).mean()
        std_20 = close.rolling(window=20).std()
        indicators['bollinger_upper'] = (sma_20 + 2 * std_20).iloc[-1]
        indicators['bollinger_middle'] = sma_20.iloc[-1]
        indicators['bollinger_lower'] = (sma_20 - 2 * std_20).iloc[-1]
        
        # Volume indicators
        indicators['volume_sma'] = volume.rolling(window=20).mean().iloc[-1]
        indicators['volume_ratio'] = volume.iloc[-1] / indicators['volume_sma']
        
        # Support and Resistance
        indicators['support'] = low.rolling(window=20).min().iloc[-1]
        indicators['resistance'] = high.rolling(window=20).max().iloc[-1]
        
        return indicators
    
    def fetch_fundamental_data(self, symbol: str) -> Dict:
        """Fetch fundamental data for a symbol"""
        cache_key = f"fundamentals:{symbol}"
        
        # Check cache
        cached = self.redis.get(cache_key)
        if cached:
            return json.loads(cached)
        
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            fundamentals = {
                'symbol': symbol,
                'sector': info.get('sector', 'Unknown'),
                'industry': info.get('industry', 'Unknown'),
                'marketCap': info.get('marketCap', 0),
                'enterpriseValue': info.get('enterpriseValue', 0),
                'trailingPE': info.get('trailingPE', 0),
                'forwardPE': info.get('forwardPE', 0),
                'pegRatio': info.get('pegRatio', 0),
                'priceToBook': info.get('priceToBook', 0),
                'priceToSales': info.get('priceToSalesTrailing12Months', 0),
                'profitMargins': info.get('profitMargins', 0),
                'operatingMargins': info.get('operatingMargins', 0),
                'returnOnAssets': info.get('returnOnAssets', 0),
                'returnOnEquity': info.get('returnOnEquity', 0),
                'revenue': info.get('totalRevenue', 0),
                'revenueGrowth': info.get('revenueGrowth', 0),
                'grossProfits': info.get('grossProfits', 0),
                'ebitda': info.get('ebitda', 0),
                'netIncome': info.get('netIncomeToCommon', 0),
                'eps': info.get('trailingEps', 0),
                'dividendYield': info.get('dividendYield', 0),
                'dividendRate': info.get('dividendRate', 0),
                'beta': info.get('beta', 1.0),
                'debtToEquity': info.get('debtToEquity', 0),
                'currentRatio': info.get('currentRatio', 0),
                'quickRatio': info.get('quickRatio', 0),
                'targetMeanPrice': info.get('targetMeanPrice', 0),
                'recommendationKey': info.get('recommendationKey', 'none')
            }
            
            # Cache for 1 hour
            self.redis.setex(cache_key, 3600, json.dumps(fundamentals))
            
            return fundamentals
            
        except Exception as e:
            logger.error(f"Error fetching fundamentals for {symbol}: {e}")
            return {}
    
    def calculate_risk_metrics(self, symbols: List[str], period: str = "1y") -> Dict:
        """Calculate risk metrics for multiple symbols"""
        # Fetch historical data for all symbols
        data = {}
        for symbol in symbols:
            hist = self.fetch_historical_data(symbol, period)
            if not hist.empty:
                data[symbol] = hist['Close']
        
        if not data:
            return {}
        
        # Create returns dataframe
        prices_df = pd.DataFrame(data)
        returns_df = prices_df.pct_change().dropna()
        
        # Calculate metrics
        metrics = {}
        
        # Individual asset metrics
        for symbol in symbols:
            if symbol in returns_df.columns:
                returns = returns_df[symbol]
                
                metrics[symbol] = {
                    'annualized_return': returns.mean() * 252,
                    'annualized_volatility': returns.std() * np.sqrt(252),
                    'sharpe_ratio': (returns.mean() * 252) / (returns.std() * np.sqrt(252)),
                    'max_drawdown': self._calculate_max_drawdown(prices_df[symbol]),
                    'var_95': returns.quantile(0.05),
                    'cvar_95': returns[returns <= returns.quantile(0.05)].mean(),
                    'skewness': returns.skew(),
                    'kurtosis': returns.kurtosis()
                }
        
        # Correlation matrix
        metrics['correlation_matrix'] = returns_df.corr().to_dict()
        
        # Covariance matrix (annualized)
        metrics['covariance_matrix'] = (returns_df.cov() * 252).to_dict()
        
        return metrics
    
    def _calculate_max_drawdown(self, prices: pd.Series) -> float:
        """Calculate maximum drawdown"""
        cumulative = (1 + prices.pct_change()).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()
    
    def fetch_market_indices(self) -> Dict[str, Dict]:
        """Fetch major market indices"""
        indices = {
            '^GSPC': 'S&P 500',
            '^DJI': 'Dow Jones',
            '^IXIC': 'NASDAQ',
            '^RUT': 'Russell 2000',
            '^VIX': 'VIX',
            '^TNX': '10-Year Treasury'
        }
        
        results = {}
        for symbol, name in indices.items():
            quote = self._fetch_single_quote(symbol, yf.Ticker(symbol))
            if quote:
                quote['name'] = name
                results[symbol] = quote
        
        return results
    
    def fetch_economic_calendar(self) -> List[Dict]:
        """Fetch upcoming economic events"""
        # This would integrate with an economic calendar API
        # For now, return sample data
        return [
            {
                'date': (datetime.now() + timedelta(days=1)).isoformat(),
                'event': 'FOMC Meeting Minutes',
                'importance': 'High',
                'forecast': None,
                'previous': None
            },
            {
                'date': (datetime.now() + timedelta(days=3)).isoformat(),
                'event': 'Non-Farm Payrolls',
                'importance': 'High',
                'forecast': '200K',
                'previous': '185K'
            }
        ]
    
    def fetch_news_sentiment(self, symbol: str) -> Dict:
        """Fetch news and sentiment for a symbol"""
        try:
            ticker = yf.Ticker(symbol)
            news = ticker.news
            
            if news:
                # Simple sentiment scoring (in production, use NLP)
                sentiment_score = 0
                for article in news[:10]:  # Last 10 articles
                    title = article.get('title', '').lower()
                    # Very simple sentiment
                    positive_words = ['gain', 'rise', 'up', 'positive', 'growth', 'beat']
                    negative_words = ['loss', 'fall', 'down', 'negative', 'decline', 'miss']
                    
                    pos_count = sum(1 for word in positive_words if word in title)
                    neg_count = sum(1 for word in negative_words if word in title)
                    
                    sentiment_score += (pos_count - neg_count)
                
                return {
                    'symbol': symbol,
                    'news_count': len(news),
                    'sentiment_score': sentiment_score,
                    'sentiment': 'positive' if sentiment_score > 0 else 'negative' if sentiment_score < 0 else 'neutral',
                    'latest_news': news[:5]  # Return latest 5 articles
                }
            
        except Exception as e:
            logger.error(f"Error fetching news for {symbol}: {e}")
        
        return {'symbol': symbol, 'news_count': 0, 'sentiment': 'neutral'}
    
    def _update_neo4j_prices(self, quotes: Dict[str, Dict]) -> None:
        """Update Neo4j with latest price data"""
        with self.driver.session() as session:
            for symbol, quote in quotes.items():
                session.run("""
                    MATCH (a:Asset {symbol: $symbol})
                    SET a.currentPrice = $price,
                        a.previousClose = $previousClose,
                        a.dayChange = $change,
                        a.dayChangePercent = $changePercent,
                        a.volume = $volume,
                        a.marketCap = $marketCap,
                        a.lastUpdated = datetime($timestamp)
                """, symbol=symbol, **quote)
    
    def sync_all_asset_data(self, symbols: Optional[List[str]] = None) -> Dict:
        """Comprehensive sync of all asset data to Neo4j"""
        if symbols is None:
            # Get all assets from Neo4j
            with self.driver.session() as session:
                result = session.run("MATCH (a:Asset) RETURN a.symbol as symbol")
                symbols = [record['symbol'] for record in result]
        
        results = {
            'quotes': 0,
            'fundamentals': 0,
            'technicals': 0,
            'errors': []
        }
        
        # Fetch and update quotes
        quotes = self.fetch_real_time_quotes(symbols)
        results['quotes'] = len(quotes)
        
        # Fetch and update fundamentals and technicals
        with ThreadPoolExecutor(max_workers=5) as executor:
            # Fundamentals
            fundamental_futures = {
                executor.submit(self.fetch_fundamental_data, symbol): symbol
                for symbol in symbols
            }
            
            for future in as_completed(fundamental_futures):
                symbol = fundamental_futures[future]
                try:
                    fundamentals = future.result()
                    if fundamentals:
                        self._update_neo4j_fundamentals(symbol, fundamentals)
                        results['fundamentals'] += 1
                except Exception as e:
                    results['errors'].append(f"{symbol}: {str(e)}")
            
            # Technical indicators
            technical_futures = {
                executor.submit(self.calculate_technical_indicators, symbol): symbol
                for symbol in symbols
            }
            
            for future in as_completed(technical_futures):
                symbol = technical_futures[future]
                try:
                    technicals = future.result()
                    if technicals:
                        self._update_neo4j_technicals(symbol, technicals)
                        results['technicals'] += 1
                except Exception as e:
                    results['errors'].append(f"{symbol}: {str(e)}")
        
        # Calculate and update risk metrics
        risk_metrics = self.calculate_risk_metrics(symbols)
        self._update_neo4j_risk_metrics(risk_metrics)
        
        return results
    
    def _update_neo4j_fundamentals(self, symbol: str, fundamentals: Dict) -> None:
        """Update Neo4j with fundamental data"""
        with self.driver.session() as session:
            session.run("""
                MATCH (a:Asset {symbol: $symbol})
                SET a += $fundamentals
            """, symbol=symbol, fundamentals=fundamentals)
    
    def _update_neo4j_technicals(self, symbol: str, technicals: Dict) -> None:
        """Update Neo4j with technical indicators"""
        with self.driver.session() as session:
            # Store as a separate node linked to asset
            session.run("""
                MATCH (a:Asset {symbol: $symbol})
                MERGE (t:TechnicalIndicators {symbol: $symbol})
                SET t += $technicals,
                    t.lastUpdated = datetime()
                MERGE (a)-[:HAS_TECHNICALS]->(t)
            """, symbol=symbol, technicals=technicals)
    
    def _update_neo4j_risk_metrics(self, metrics: Dict) -> None:
        """Update Neo4j with risk metrics"""
        with self.driver.session() as session:
            # Update individual asset risk metrics
            for symbol, asset_metrics in metrics.items():
                if isinstance(asset_metrics, dict) and 'annualized_return' in asset_metrics:
                    session.run("""
                        MATCH (a:Asset {symbol: $symbol})
                        SET a.expectedReturn = $return,
                            a.volatility = $volatility,
                            a.sharpeRatio = $sharpe,
                            a.maxDrawdown = $maxDD,
                            a.var95 = $var95
                    """, symbol=symbol,
                        return=asset_metrics['annualized_return'],
                        volatility=asset_metrics['annualized_volatility'],
                        sharpe=asset_metrics['sharpe_ratio'],
                        maxDD=asset_metrics['max_drawdown'],
                        var95=asset_metrics['var_95'])
            
            # Update correlation relationships
            if 'correlation_matrix' in metrics:
                corr_matrix = metrics['correlation_matrix']
                for symbol1, correlations in corr_matrix.items():
                    for symbol2, correlation in correlations.items():
                        if symbol1 != symbol2 and abs(correlation) > 0.3:
                            session.run("""
                                MATCH (a1:Asset {symbol: $symbol1})
                                MATCH (a2:Asset {symbol: $symbol2})
                                MERGE (a1)-[c:CORRELATED_WITH]-(a2)
                                SET c.correlation = $correlation,
                                    c.period = '1Y',
                                    c.lastUpdated = datetime()
                            """, symbol1=symbol1, symbol2=symbol2, correlation=correlation)
    
    async def stream_real_time_data(self, symbols: List[str], callback) -> None:
        """Stream real-time data using WebSocket (if available)"""
        # This is a placeholder for real WebSocket implementation
        # Yahoo Finance doesn't provide WebSocket, but you could use other providers
        while True:
            quotes = self.fetch_real_time_quotes(symbols)
            await callback(quotes)
            await asyncio.sleep(5)  # Update every 5 seconds


# Example usage
if __name__ == "__main__":
    # Neo4j connection
    uri = "neo4j+s://1c34ddb0.databases.neo4j.io"
    driver = GraphDatabase.driver(uri, auth=("neo4j", "Sw-DAlLfV1_Rw4lFlXAxl0DoN4lCauWrp29IZnfq_zM"))
    
    # Redis connection
    redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)
    
    # Initialize service
    market_service = MarketDataService(driver, redis_client)
    
    # Test fetching real-time quotes
    symbols = ['AAPL', 'MSFT', 'JPM', 'TLT']
    print("Fetching real-time quotes...")
    quotes = market_service.fetch_real_time_quotes(symbols)
    for symbol, quote in quotes.items():
        print(f"{symbol}: ${quote['price']:.2f} ({quote['changePercent']:.2f}%)")
    
    # Test technical indicators
    print("\nCalculating technical indicators for AAPL...")
    technicals = market_service.calculate_technical_indicators('AAPL')
    print(f"RSI: {technicals.get('rsi', 0):.2f}")
    print(f"SMA 50: ${technicals.get('sma_50', 0):.2f}")
    
    # Test risk metrics
    print("\nCalculating risk metrics...")
    risk_metrics = market_service.calculate_risk_metrics(['AAPL', 'MSFT', 'JPM'])
    for symbol, metrics in risk_metrics.items():
        if isinstance(metrics, dict) and 'sharpe_ratio' in metrics:
            print(f"{symbol} Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
    
    driver.close()