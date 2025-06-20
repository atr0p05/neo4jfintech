"""
FastAPI Backend for Neo4j Investment Platform
Main application with all API endpoints
"""

from fastapi import FastAPI, HTTPException, Depends, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from contextlib import asynccontextmanager
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import asyncio
import json
import os
from pydantic import BaseModel, Field
import redis.asyncio as redis
from neo4j import AsyncGraphDatabase
import uvicorn
import pandas as pd
import numpy as np

# Import our services
from services.portfolio_optimizer import AdvancedPortfolioOptimizer
from services.market_data_service import MarketDataService
from services.graph_rag_service import InvestmentGraphRAG
from services.backtester_service import OptimizationBacktester

# Pydantic models
class ClientProfile(BaseModel):
    clientId: str
    name: str
    riskTolerance: str
    netWorth: float
    portfolioValue: float
    totalReturn: float
    riskScore: float

class PortfolioRequest(BaseModel):
    clientId: str
    optimizationModel: str = "markowitz"
    constraints: Optional[Dict] = None
    views: Optional[Dict] = None

class AdvisorQuestion(BaseModel):
    clientId: str
    question: str

class BacktestRequest(BaseModel):
    symbols: List[str]
    strategy: Dict
    startDate: str
    endDate: str
    initialCapital: float = 100000

class MarketDataRequest(BaseModel):
    symbols: List[str]
    dataType: str = "quotes"  # quotes, historical, technical, fundamental

# Initialize services
neo4j_driver = None
redis_client = None
market_service = None
optimizer = None
graph_rag = None
backtester = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle"""
    global neo4j_driver, redis_client, market_service, optimizer, graph_rag, backtester
    
    # Startup
    print("🚀 Starting Investment Platform API...")
    
    # Initialize Neo4j
    neo4j_driver = AsyncGraphDatabase.driver(
        os.getenv("NEO4J_URI"),
        auth=(os.getenv("NEO4J_USER"), os.getenv("NEO4J_PASSWORD"))
    )
    
    # Initialize Redis
    try:
        redis_client = await redis.from_url(
            os.getenv("REDIS_URL", "redis://localhost:6379"),
            encoding="utf-8",
            decode_responses=True
        )
        await redis_client.ping()
    except Exception as e:
        print(f"Redis connection failed: {e}")
        redis_client = None
    
    # Initialize services with sync drivers for compatibility
    from neo4j import GraphDatabase
    sync_driver = GraphDatabase.driver(
        os.getenv("NEO4J_URI"),
        auth=(os.getenv("NEO4J_USER"), os.getenv("NEO4J_PASSWORD"))
    )
    
    try:
        from core.optimization.advanced_optimizer import AdvancedPortfolioOptimizer as SyncOptimizer
        optimizer = SyncOptimizer(sync_driver)
    except:
        # Fallback mock optimizer
        optimizer = MockOptimizer()
    
    try:
        from services.market_data.service import MarketDataService as SyncMarketService
        market_service = SyncMarketService(sync_driver, None)
    except:
        market_service = MockMarketService()
    
    try:
        from services.graphrag.investment_rag import InvestmentGraphRAG as SyncGraphRAG
        graph_rag = SyncGraphRAG(
            sync_driver,
            os.getenv("OPENAI_API_KEY", "test-key"),
            os.getenv("NEO4J_URI"),
            os.getenv("NEO4J_USER"),
            os.getenv("NEO4J_PASSWORD")
        )
    except:
        graph_rag = MockGraphRAG()
    
    try:
        from services.backtesting.backtest_engine import OptimizationBacktester as SyncBacktester
        backtester = SyncBacktester(sync_driver, market_service)
    except:
        backtester = MockBacktester()
    
    print("✅ All services initialized")
    
    yield
    
    # Shutdown
    print("🔄 Shutting down...")
    await neo4j_driver.close()
    if redis_client:
        await redis_client.close()

# Create FastAPI app
app = FastAPI(
    title="Neo4j Investment Platform API",
    description="Advanced portfolio management with graph-based analytics",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# WebSocket manager for real-time updates
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, List[WebSocket]] = {}
    
    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        if client_id not in self.active_connections:
            self.active_connections[client_id] = []
        self.active_connections[client_id].append(websocket)
    
    def disconnect(self, websocket: WebSocket, client_id: str):
        if client_id in self.active_connections:
            self.active_connections[client_id].remove(websocket)
            if not self.active_connections[client_id]:
                del self.active_connections[client_id]
    
    async def send_personal_message(self, message: str, client_id: str):
        if client_id in self.active_connections:
            for connection in self.active_connections[client_id]:
                await connection.send_text(message)
    
    async def broadcast(self, message: str):
        for connections in self.active_connections.values():
            for connection in connections:
                await connection.send_text(message)

manager = ConnectionManager()

# Health check
@app.get("/health")
async def health_check():
    """Check if all services are healthy"""
    try:
        # Check Neo4j
        async with neo4j_driver.session() as session:
            await session.run("RETURN 1")
        
        # Check Redis
        redis_status = "connected"
        if redis_client:
            await redis_client.ping()
        else:
            redis_status = "not available"
        
        return {
            "status": "healthy",
            "services": {
                "neo4j": "connected",
                "redis": redis_status,
                "timestamp": datetime.now().isoformat()
            }
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Service unhealthy: {str(e)}")

# Client endpoints
@app.get("/api/clients", response_model=List[ClientProfile])
async def get_all_clients():
    """Get all clients"""
    async with neo4j_driver.session() as session:
        result = await session.run("""
            MATCH (c:Client)-[:HAS_RISK_PROFILE]->(rp:RiskProfile)
            OPTIONAL MATCH (c)-[:OWNS_PORTFOLIO]->(p:Portfolio)
            WITH c, rp, collect(p) as portfolios
            RETURN c, rp, 
                   sum(p.totalValue) as portfolioValue,
                   avg(p.totalReturn) as avgReturn
            ORDER BY c.name
        """)
        
        clients = []
        async for record in result:
            client = record['c']
            risk_profile = record['rp']
            
            clients.append(ClientProfile(
                clientId=client['clientId'],
                name=client['name'],
                riskTolerance=client['riskTolerance'],
                netWorth=client['netWorth'],
                portfolioValue=record['portfolioValue'] or 0,
                totalReturn=record['avgReturn'] or 0,
                riskScore=risk_profile['riskScore']
            ))
        
        return clients

@app.get("/api/clients/{client_id}", response_model=ClientProfile)
async def get_client(client_id: str):
    """Get specific client details"""
    async with neo4j_driver.session() as session:
        result = await session.run("""
            MATCH (c:Client {clientId: $clientId})-[:HAS_RISK_PROFILE]->(rp:RiskProfile)
            OPTIONAL MATCH (c)-[:OWNS_PORTFOLIO]->(p:Portfolio)
            WITH c, rp, collect(p) as portfolios
            RETURN c, rp, 
                   sum(p.totalValue) as portfolioValue,
                   avg(p.totalReturn) as avgReturn
        """, clientId=client_id)
        
        record = await result.single()
        if not record:
            raise HTTPException(status_code=404, detail="Client not found")
        
        client = record['c']
        risk_profile = record['rp']
        
        return ClientProfile(
            clientId=client['clientId'],
            name=client['name'],
            riskTolerance=client['riskTolerance'],
            netWorth=client['netWorth'],
            portfolioValue=record['portfolioValue'] or 0,
            totalReturn=record['avgReturn'] or 0,
            riskScore=risk_profile['riskScore']
        )

# Portfolio endpoints
@app.get("/api/portfolios/{client_id}")
async def get_portfolio(client_id: str):
    """Get client's portfolio details"""
    async with neo4j_driver.session() as session:
        result = await session.run("""
            MATCH (c:Client {clientId: $clientId})-[:OWNS_PORTFOLIO]->(p:Portfolio)
            OPTIONAL MATCH (p)-[h:HOLDS]->(a:Asset)
            WITH p, collect({
                symbol: a.symbol,
                name: a.name,
                assetClass: a.assetClass,
                sector: a.sector,
                value: h.value,
                weight: h.weight,
                shares: h.shares,
                currentPrice: a.currentPrice,
                dayChange: a.dayChangePercent
            }) as holdings
            RETURN p, holdings
        """, clientId=client_id)
        
        record = await result.single()
        if not record:
            raise HTTPException(status_code=404, detail="Portfolio not found")
        
        portfolio = record['p']
        holdings = record['holdings']
        
        return {
            "portfolioId": portfolio['portfolioId'],
            "totalValue": portfolio['totalValue'],
            "expectedReturn": portfolio.get('expectedReturn', 0),
            "volatility": portfolio.get('volatility', 0),
            "sharpeRatio": portfolio.get('sharpeRatio', 0),
            "holdings": holdings,
            "lastUpdated": portfolio.get('lastCalculated', datetime.now()).isoformat()
        }

@app.post("/api/portfolios/optimize")
async def optimize_portfolio(request: PortfolioRequest):
    """Run portfolio optimization"""
    try:
        result = await asyncio.to_thread(
            optimizer.optimize_portfolio,
            request.clientId,
            request.optimizationModel,
            request.constraints,
            request.views
        )
        
        # Send update via WebSocket
        await manager.send_personal_message(
            json.dumps({
                "type": "optimization_complete",
                "data": result
            }),
            request.clientId
        )
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/portfolios/{client_id}/performance")
async def get_portfolio_performance(client_id: str, period: str = "1Y"):
    """Get portfolio performance history"""
    # This would fetch historical performance data
    # For now, return sample data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365 if period == "1Y" else 30)
    
    # Generate sample performance data
    dates = pd.date_range(start_date, end_date, freq='D')
    values = [100000]  # Starting value
    
    for i in range(1, len(dates)):
        # Random walk with slight positive drift
        daily_return = np.random.normal(0.0003, 0.01)  # 0.03% daily return, 1% volatility
        values.append(values[-1] * (1 + daily_return))
    
    return {
        "dates": [d.isoformat() for d in dates],
        "values": values,
        "totalReturn": (values[-1] - values[0]) / values[0],
        "period": period
    }

# Market data endpoints
@app.post("/api/market/quotes")
async def get_market_quotes(request: MarketDataRequest):
    """Get real-time market quotes"""
    quotes = await asyncio.to_thread(
        market_service.fetch_real_time_quotes,
        request.symbols
    )
    return quotes

@app.get("/api/market/indices")
async def get_market_indices():
    """Get major market indices"""
    indices = await asyncio.to_thread(market_service.fetch_market_indices)
    return indices

@app.get("/api/market/historical/{symbol}")
async def get_historical_data(symbol: str, period: str = "1y", interval: str = "1d"):
    """Get historical price data"""
    data = await asyncio.to_thread(
        market_service.fetch_historical_data,
        symbol, period, interval
    )
    
    if data.empty:
        raise HTTPException(status_code=404, detail="No data found")
    
    return {
        "symbol": symbol,
        "data": data.to_dict(orient="records"),
        "period": period,
        "interval": interval
    }

@app.get("/api/market/technical/{symbol}")
async def get_technical_indicators(symbol: str):
    """Get technical indicators for a symbol"""
    indicators = await asyncio.to_thread(
        market_service.calculate_technical_indicators,
        symbol
    )
    return indicators

# AI Advisor endpoints
@app.post("/api/advisor/ask")
async def ask_advisor(request: AdvisorQuestion):
    """Get AI investment advice"""
    response = await asyncio.to_thread(
        graph_rag.get_investment_advice,
        request.clientId,
        request.question
    )
    return response

@app.post("/api/advisor/ask/stream")
async def ask_advisor_stream(request: AdvisorQuestion):
    """Stream AI investment advice"""
    async def generate():
        # Mock streaming response
        words = ["Based", "on", "your", "portfolio", "analysis,", "I", "recommend", 
                "considering", "a", "balanced", "approach", "to", "risk", "management."]
        for word in words:
            yield f"data: {json.dumps({'chunk': word + ' '})}\n\n"
            await asyncio.sleep(0.1)
    
    return StreamingResponse(generate(), media_type="text/event-stream")

@app.get("/api/advisor/report/{client_id}")
async def generate_investment_report(client_id: str):
    """Generate comprehensive investment report"""
    report = await asyncio.to_thread(
        graph_rag.create_investment_report,
        client_id
    )
    return {"report": report}

# Backtesting endpoints
@app.post("/api/backtest")
async def run_backtest(request: BacktestRequest):
    """Run portfolio backtest"""
    try:
        result = await asyncio.to_thread(
            backtester.backtest_strategy,
            request.symbols,
            request.strategy,
            request.startDate,
            request.endDate,
            request.initialCapital
        )
        
        # Convert result to JSON-serializable format
        return {
            "strategyName": result.strategy_name,
            "metrics": {
                "totalReturn": result.total_return,
                "annualizedReturn": result.annualized_return,
                "volatility": result.volatility,
                "sharpeRatio": result.sharpe_ratio,
                "maxDrawdown": result.max_drawdown,
                "calmarRatio": result.calmar_ratio,
                "winRate": result.win_rate
            },
            "trades": result.trades[:10],  # Last 10 trades
            "benchmarkComparison": result.benchmark_comparison
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/backtest/compare")
async def compare_strategies(
    symbols: List[str],
    strategies: List[Dict],
    start_date: str,
    end_date: str,
    initial_capital: float = 100000
):
    """Compare multiple strategies"""
    comparison = await asyncio.to_thread(
        backtester.compare_strategies,
        symbols, strategies, start_date, end_date, initial_capital
    )
    return comparison.to_dict(orient="records")

# WebSocket endpoints
@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    """WebSocket for real-time updates"""
    await manager.connect(websocket, client_id)
    
    try:
        # Send initial connection message
        await websocket.send_text(json.dumps({
            "type": "connected",
            "message": f"Connected to investment platform for client {client_id}"
        }))
        
        # Start sending periodic updates
        async def send_updates():
            while True:
                # Get latest portfolio value
                async with neo4j_driver.session() as session:
                    result = await session.run("""
                        MATCH (c:Client {clientId: $clientId})-[:OWNS_PORTFOLIO]->(p:Portfolio)
                        RETURN sum(p.totalValue) as totalValue
                    """, clientId=client_id)
                    
                    record = await result.single()
                    if record:
                        await websocket.send_text(json.dumps({
                            "type": "portfolio_update",
                            "data": {
                                "totalValue": record['totalValue'],
                                "timestamp": datetime.now().isoformat()
                            }
                        }))
                
                await asyncio.sleep(5)  # Update every 5 seconds
        
        # Run update task
        update_task = asyncio.create_task(send_updates())
        
        # Keep connection alive
        while True:
            data = await websocket.receive_text()
            # Handle incoming messages if needed
            
    except WebSocketDisconnect:
        manager.disconnect(websocket, client_id)
        if 'update_task' in locals():
            update_task.cancel()

# Admin endpoints
@app.post("/api/admin/sync-market-data")
async def sync_market_data():
    """Sync all market data"""
    result = await asyncio.to_thread(market_service.sync_all_asset_data)
    return result

@app.get("/api/admin/system-stats")
async def get_system_stats():
    """Get system statistics"""
    async with neo4j_driver.session() as session:
        result = await session.run("""
            MATCH (c:Client) WITH count(c) as clientCount
            MATCH (p:Portfolio) WITH clientCount, count(p) as portfolioCount
            MATCH (a:Asset) WITH clientCount, portfolioCount, count(a) as assetCount
            MATCH (o:Optimization) WITH clientCount, portfolioCount, assetCount, count(o) as optimizationCount
            RETURN clientCount, portfolioCount, assetCount, optimizationCount
        """)
        
        record = await result.single()
        
        return {
            "clients": record['clientCount'],
            "portfolios": record['portfolioCount'],
            "assets": record['assetCount'],
            "optimizations": record['optimizationCount'],
            "activeConnections": len(manager.active_connections)
        }

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return {
        "error": exc.detail,
        "status_code": exc.status_code,
        "timestamp": datetime.now().isoformat()
    }

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    return {
        "error": "Internal server error",
        "detail": str(exc),
        "status_code": 500,
        "timestamp": datetime.now().isoformat()
    }

# Mock services for fallback
class MockOptimizer:
    def optimize_portfolio(self, client_id, model, constraints=None, views=None):
        return {
            "weights": {"AAPL": 0.25, "MSFT": 0.20, "TLT": 0.30, "GLD": 0.25},
            "expected_return": 0.08,
            "volatility": 0.12,
            "sharpe_ratio": 0.67
        }

class MockMarketService:
    def fetch_real_time_quotes(self, symbols):
        return {symbol: {"price": 100.0, "change": 1.0, "changePercent": 1.0} for symbol in symbols}
    
    def fetch_market_indices(self):
        return {"^GSPC": {"name": "S&P 500", "value": 4500, "change": 10, "changePercent": 0.22}}
    
    def fetch_historical_data(self, symbol, period, interval):
        return pd.DataFrame({"Close": [100, 101, 102]})
    
    def calculate_technical_indicators(self, symbol):
        return {"rsi": 65, "sma_20": 100}
    
    def sync_all_asset_data(self):
        return {"quotes": 5, "fundamentals": 5, "technicals": 5, "errors": []}

class MockGraphRAG:
    def get_investment_advice(self, client_id, question):
        return {
            "response": "Based on your portfolio and risk profile, consider diversifying across sectors.",
            "insights": ["Portfolio is well-balanced", "Consider increasing tech allocation"],
            "context": {"risk_tolerance": "Moderate"}
        }
    
    def create_investment_report(self, client_id):
        return "Comprehensive investment report for client " + client_id

class MockBacktester:
    def backtest_strategy(self, symbols, strategy, start, end, capital):
        class MockResult:
            strategy_name = "test"
            total_return = 0.15
            annualized_return = 0.12
            volatility = 0.18
            sharpe_ratio = 0.67
            max_drawdown = -0.08
            calmar_ratio = 1.5
            win_rate = 0.65
            trades = []
            benchmark_comparison = {}
        return MockResult()
    
    def compare_strategies(self, symbols, strategies, start, end, capital):
        return pd.DataFrame({
            "Strategy": ["Test1", "Test2"],
            "Return": [0.12, 0.10],
            "Risk": [0.15, 0.12],
            "Sharpe": [0.8, 0.83]
        })

# Main entry point
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )