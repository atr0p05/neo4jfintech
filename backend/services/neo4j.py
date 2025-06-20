"""
Neo4j Database Service
"""
from neo4j import GraphDatabase
from typing import Dict, List, Any, Optional
import logging

logger = logging.getLogger(__name__)


class Neo4jService:
    """Neo4j database connection and query service"""
    
    def __init__(self, uri: str, user: str, password: str):
        self.uri = uri
        self.user = user
        self.password = password
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        
    def close(self):
        """Close the database connection"""
        if self.driver:
            self.driver.close()
            
    def verify_connection(self):
        """Verify the database connection is working"""
        with self.driver.session() as session:
            result = session.run("RETURN 1")
            return result.single()[0] == 1
            
    def execute_query(self, query: str, parameters: Dict = None) -> List[Dict]:
        """Execute a Cypher query and return results"""
        with self.driver.session() as session:
            result = session.run(query, parameters or {})
            return [record.data() for record in result]
            
    def execute_write(self, query: str, parameters: Dict = None) -> Any:
        """Execute a write transaction"""
        with self.driver.session() as session:
            return session.write_transaction(
                lambda tx: tx.run(query, parameters or {}).data()
            )
            
    # Portfolio specific methods
    def get_client_portfolio(self, client_id: str) -> Optional[Dict]:
        """Get client portfolio data"""
        query = """
        MATCH (c:Client {clientId: $clientId})-[:OWNS_PORTFOLIO]->(p:Portfolio)
        OPTIONAL MATCH (p)-[h:HOLDS]->(a:Asset)
        WITH c, p, collect({
            symbol: a.symbol,
            name: a.name,
            weight: h.weight,
            value: h.value,
            shares: h.shares,
            asset_class: a.assetClass,
            sector: a.sector,
            current_price: a.currentPrice
        }) as holdings
        RETURN {
            client_id: c.clientId,
            client_name: c.name,
            portfolio_id: p.portfolioId,
            total_value: p.totalValue,
            expected_return: p.expectedReturn,
            volatility: p.volatility,
            sharpe_ratio: p.sharpeRatio,
            holdings: holdings
        } as portfolio
        """
        
        results = self.execute_query(query, {"clientId": client_id})
        return results[0]["portfolio"] if results else None
        
    def get_portfolio_risk_metrics(self, client_id: str) -> Optional[Dict]:
        """Get portfolio risk metrics"""
        query = """
        MATCH (c:Client {clientId: $clientId})-[:OWNS_PORTFOLIO]->(p:Portfolio)
        MATCH (c)-[:HAS_RISK_PROFILE]->(rp:RiskProfile)
        RETURN {
            portfolio_volatility: p.volatility,
            portfolio_var95: p.var95,
            portfolio_beta: p.beta,
            max_drawdown: p.maxDrawdown,
            risk_tolerance: c.riskTolerance,
            volatility_limit: rp.volatilityTolerance
        } as metrics
        """
        
        results = self.execute_query(query, {"clientId": client_id})
        return results[0]["metrics"] if results else None
        
    def get_optimization_results(self, client_id: str, model: str = None) -> List[Dict]:
        """Get optimization results for a client"""
        query = """
        MATCH (c:Client {clientId: $clientId})-[:HAS_OPTIMIZATION]->(opt:Optimization)
        WHERE $model IS NULL OR opt.model = $model
        OPTIONAL MATCH (opt)-[rec:RECOMMENDS_WEIGHT]->(a:Asset)
        WITH opt, collect({
            symbol: a.symbol,
            weight: rec.weight
        }) as recommendations
        RETURN {
            model: opt.model,
            timestamp: opt.timestamp,
            expected_return: opt.expectedReturn,
            volatility: opt.volatility,
            sharpe_ratio: opt.sharpeRatio,
            recommendations: recommendations
        } as result
        ORDER BY opt.timestamp DESC
        """
        
        return self.execute_query(query, {"clientId": client_id, "model": model})
        
    def save_optimization_result(self, client_id: str, model: str, result: Dict) -> None:
        """Save optimization result to database"""
        query = """
        MATCH (c:Client {clientId: $clientId})
        CREATE (opt:Optimization {
            optimizationId: randomUUID(),
            model: $model,
            timestamp: datetime(),
            expectedReturn: $expectedReturn,
            volatility: $volatility,
            sharpeRatio: $sharpeRatio
        })
        CREATE (c)-[:HAS_OPTIMIZATION]->(opt)
        
        WITH opt
        UNWIND $weights as weight
        MATCH (a:Asset {symbol: weight.symbol})
        CREATE (opt)-[:RECOMMENDS_WEIGHT {
            weight: weight.weight,
            value: weight.value
        }]->(a)
        """
        
        weights = [
            {"symbol": symbol, "weight": weight, "value": weight * 100000}
            for symbol, weight in result["weights"].items()
            if weight > 0.01
        ]
        
        self.execute_write(query, {
            "clientId": client_id,
            "model": model,
            "expectedReturn": result["expected_return"],
            "volatility": result["volatility"],
            "sharpeRatio": result["sharpe_ratio"],
            "weights": weights
        })
        
    def get_market_data(self, symbols: List[str]) -> List[Dict]:
        """Get market data for assets"""
        query = """
        MATCH (a:Asset)
        WHERE a.symbol IN $symbols
        RETURN {
            symbol: a.symbol,
            name: a.name,
            current_price: a.currentPrice,
            expected_return: a.expectedReturn,
            volatility: a.volatility,
            sharpe_ratio: a.sharpeRatio,
            sector: a.sector,
            asset_class: a.assetClass
        } as asset
        """
        
        return self.execute_query(query, {"symbols": symbols})