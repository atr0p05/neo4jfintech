"""
Neo4j service for database operations
"""
from neo4j import GraphDatabase
from typing import Dict, List, Any, Optional
import pandas as pd
from datetime import datetime

class Neo4jService:
    def __init__(self, uri: str, user: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
    
    def close(self):
        self.driver.close()
    
    def verify_connection(self):
        """Verify Neo4j connection"""
        with self.driver.session() as session:
            result = session.run("RETURN 1 as test")
            return result.single()["test"] == 1
    
    def get_client_portfolio(self, client_id: str) -> Dict[str, Any]:
        """Get complete client portfolio data"""
        query = """
        MATCH (c:Client {clientId: $clientId})
        OPTIONAL MATCH (c)-[:HAS_RISK_PROFILE]->(rp:RiskProfile)
        OPTIONAL MATCH (c)-[:OWNS_PORTFOLIO]->(p:Portfolio)
        OPTIONAL MATCH (p)-[h:HOLDS]->(a:Asset)
        WITH c, rp, p, 
             collect({
                 symbol: a.symbol,
                 name: a.name,
                 assetClass: a.assetClass,
                 weight: h.weight,
                 value: h.value,
                 shares: h.shares,
                 metrics: {
                     expectedReturn: a.expectedReturn,
                     volatility: a.volatility,
                     sharpeRatio: a.sharpeRatio,
                     beta: a.beta
                 }
             }) as holdings
        RETURN {
            client: {
                id: c.clientId,
                name: c.name,
                riskTolerance: c.riskTolerance,
                netWorth: c.netWorth
            },
            riskProfile: {
                score: rp.riskScore,
                volatilityTolerance: rp.volatilityTolerance,
                var95: rp.var95
            },
            portfolio: {
                id: p.portfolioId,
                totalValue: p.totalValue,
                expectedReturn: p.expectedReturn,
                volatility: p.volatility,
                sharpeRatio: p.sharpeRatio,
                holdings: holdings
            }
        } as data
        """
        
        with self.driver.session() as session:
            result = session.run(query, clientId=client_id)
            record = result.single()
            return record["data"] if record else None
    
    def get_portfolio_metrics(self, portfolio_id: str) -> Dict[str, Any]:
        """Calculate detailed portfolio metrics"""
        query = """
        MATCH (p:Portfolio {portfolioId: $portfolioId})
        MATCH (p)-[h:HOLDS]->(a:Asset)
        OPTIONAL MATCH (a)-[fe:HAS_FACTOR_EXPOSURE]->(f:Factor)
        WITH p, a, h, f, fe
        RETURN {
            portfolio_id: p.portfolioId,
            total_value: p.totalValue,
            metrics: {
                expected_return: sum(h.weight * a.expectedReturn),
                volatility: sqrt(sum(h.weight * h.weight * a.volatility * a.volatility)),
                beta: sum(h.weight * a.beta),
                sharpe_ratio: (sum(h.weight * a.expectedReturn) - 0.02) / 
                             sqrt(sum(h.weight * h.weight * a.volatility * a.volatility))
            },
            allocation: {
                equity: sum(CASE WHEN a.assetClass = 'Equity' THEN h.weight ELSE 0 END),
                bond: sum(CASE WHEN a.assetClass = 'Bond' THEN h.weight ELSE 0 END),
                alternative: sum(CASE WHEN a.assetClass = 'Alternative' THEN h.weight ELSE 0 END)
            },
            factor_exposures: collect(DISTINCT {
                factor: f.name,
                exposure: sum(h.weight * fe.exposure)
            })
        } as metrics
        """
        
        with self.driver.session() as session:
            result = session.run(query, portfolioId=portfolio_id)
            return result.single()["metrics"]
    
    def get_risk_analytics(self, client_id: str) -> Dict[str, Any]:
        """Get comprehensive risk analytics for a client"""
        query = """
        MATCH (c:Client {clientId: $clientId})-[:OWNS_PORTFOLIO]->(p:Portfolio)
        MATCH (p)-[h:HOLDS]->(a:Asset)
        
        // Concentration risk
        WITH c, p, a.sector as sector, sum(h.weight) as sectorWeight
        WITH c, p, collect({sector: sector, weight: sectorWeight}) as sectorConcentrations
        
        // Factor exposures
        MATCH (p)-[h2:HOLDS]->(a2:Asset)-[fe:HAS_FACTOR_EXPOSURE]->(f:Factor)
        WITH c, p, sectorConcentrations, f.name as factor, sum(h2.weight * fe.exposure) as factorExposure
        WITH c, p, sectorConcentrations, collect({factor: factor, exposure: factorExposure}) as factorExposures
        
        // Correlation risk
        MATCH (p)-[h3:HOLDS]->(a3:Asset)
        OPTIONAL MATCH (a3)-[corr:CORRELATED_WITH]-(a4:Asset)<-[h4:HOLDS]-(p)
        WITH c, p, sectorConcentrations, factorExposures,
             avg(corr.correlation) as avgCorrelation,
             max(corr.correlation) as maxCorrelation
        
        RETURN {
            client_id: c.clientId,
            portfolio_id: p.portfolioId,
            concentration_risk: [s IN sectorConcentrations WHERE s.weight > 0.25],
            factor_exposures: factorExposures,
            correlation_metrics: {
                average: avgCorrelation,
                maximum: maxCorrelation
            },
            risk_score: CASE
                WHEN size([s IN sectorConcentrations WHERE s.weight > 0.3]) > 0 THEN 'High'
                WHEN size([s IN sectorConcentrations WHERE s.weight > 0.25]) > 0 THEN 'Medium'
                ELSE 'Low'
            END
        } as risk_analytics
        """
        
        with self.driver.session() as session:
            result = session.run(query, clientId=client_id)
            return result.single()["risk_analytics"]
    
    def get_investment_opportunities(self, client_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Find suitable investment opportunities"""
        query = """
        MATCH (c:Client {clientId: $clientId})
        MATCH (c)-[:OWNS_PORTFOLIO]->(p:Portfolio)
        MATCH (a:Asset)
        WHERE NOT EXISTS((p)-[:HOLDS]->(a))
        
        // Match risk profile
        WITH c, p, a
        WHERE (c.riskTolerance = 'Conservative' AND a.volatility < 0.15) OR
              (c.riskTolerance = 'Moderate' AND a.volatility < 0.25) OR
              (c.riskTolerance = 'Aggressive' AND a.volatility < 0.40)
        
        // Score opportunities
        WITH c, p, a,
             a.sharpeRatio * 0.4 + 
             (1 - a.volatility) * 0.3 + 
             a.expectedReturn * 0.3 as score
        
        RETURN {
            symbol: a.symbol,
            name: a.name,
            assetClass: a.assetClass,
            expectedReturn: a.expectedReturn,
            volatility: a.volatility,
            sharpeRatio: a.sharpeRatio,
            beta: a.beta,
            score: score,
            rationale: CASE
                WHEN a.sharpeRatio > 1.2 THEN 'Excellent risk-adjusted returns'
                WHEN a.expectedReturn > 0.15 THEN 'High growth potential'
                WHEN a.volatility < 0.1 THEN 'Low risk, stable returns'
                ELSE 'Balanced opportunity'
            END
        } as opportunity
        ORDER BY score DESC
        LIMIT $limit
        """
        
        with self.driver.session() as session:
            result = session.run(query, clientId=client_id, limit=limit)
            return [record["opportunity"] for record in result]
    
    def update_portfolio_holdings(self, portfolio_id: str, holdings: List[Dict[str, Any]]) -> bool:
        """Update portfolio holdings"""
        query = """
        MATCH (p:Portfolio {portfolioId: $portfolioId})
        // Remove existing holdings
        OPTIONAL MATCH (p)-[h:HOLDS]->()
        DELETE h
        
        // Add new holdings
        WITH p
        UNWIND $holdings as holding
        MATCH (a:Asset {symbol: holding.symbol})
        CREATE (p)-[h:HOLDS {
            weight: holding.weight,
            shares: holding.shares,
            value: holding.value
        }]->(a)
        
        RETURN count(h) as holdings_created
        """
        
        with self.driver.session() as session:
            result = session.run(query, portfolioId=portfolio_id, holdings=holdings)
            return result.single()["holdings_created"] == len(holdings)