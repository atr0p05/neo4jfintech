"""
Portfolio API endpoints
"""
from fastapi import APIRouter, HTTPException, Depends, Request
from typing import Dict, Any, List
from pydantic import BaseModel, Field
from datetime import datetime

from services.neo4j import Neo4jService

router = APIRouter()

class PortfolioResponse(BaseModel):
    portfolio_id: str
    client_id: str
    total_value: float
    expected_return: float
    volatility: float
    sharpe_ratio: float
    holdings: List[Dict[str, Any]]
    last_updated: datetime

class HoldingUpdate(BaseModel):
    symbol: str
    weight: float = Field(ge=0, le=1)
    shares: int = Field(ge=0)
    value: float = Field(ge=0)

class RebalanceRequest(BaseModel):
    portfolio_id: str
    target_allocations: Dict[str, float]
    rebalance_method: str = "proportional"

def get_neo4j(request: Request) -> Neo4jService:
    return request.app.state.neo4j

@router.get("/client/{client_id}")
async def get_client_portfolio(
    client_id: str,
    neo4j: Neo4jService = Depends(get_neo4j)
) -> PortfolioResponse:
    """Get complete portfolio data for a client"""
    try:
        portfolio_data = neo4j.get_client_portfolio(client_id)
        
        if not portfolio_data or not portfolio_data.get("portfolio"):
            raise HTTPException(status_code=404, detail="Portfolio not found")
        
        portfolio = portfolio_data["portfolio"]
        
        return PortfolioResponse(
            portfolio_id=portfolio["id"],
            client_id=client_id,
            total_value=portfolio["totalValue"],
            expected_return=portfolio.get("expectedReturn", 0),
            volatility=portfolio.get("volatility", 0),
            sharpe_ratio=portfolio.get("sharpeRatio", 0),
            holdings=portfolio.get("holdings", []),
            last_updated=datetime.now()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{portfolio_id}/metrics")
async def get_portfolio_metrics(
    portfolio_id: str,
    neo4j: Neo4jService = Depends(get_neo4j)
) -> Dict[str, Any]:
    """Get detailed portfolio metrics"""
    try:
        metrics = neo4j.get_portfolio_metrics(portfolio_id)
        
        if not metrics:
            raise HTTPException(status_code=404, detail="Portfolio not found")
        
        return {
            "portfolio_id": portfolio_id,
            "metrics": metrics["metrics"],
            "allocation": metrics["allocation"],
            "factor_exposures": metrics["factor_exposures"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/{portfolio_id}/rebalance")
async def rebalance_portfolio(
    portfolio_id: str,
    request: RebalanceRequest,
    neo4j: Neo4jService = Depends(get_neo4j)
) -> Dict[str, Any]:
    """Rebalance portfolio to target allocations"""
    try:
        # Validate allocations sum to 1
        total_weight = sum(request.target_allocations.values())
        if abs(total_weight - 1.0) > 0.01:
            raise HTTPException(
                status_code=400,
                detail=f"Target allocations must sum to 1.0, got {total_weight}"
            )
        
        # Get current portfolio value
        portfolio = neo4j.get_portfolio_metrics(portfolio_id)
        if not portfolio:
            raise HTTPException(status_code=404, detail="Portfolio not found")
        
        total_value = portfolio["total_value"]
        
        # Calculate new holdings
        new_holdings = []
        for symbol, weight in request.target_allocations.items():
            # Get current price (in production, fetch from market data service)
            # For now, using placeholder
            price = 100  # TODO: Fetch real price
            value = total_value * weight
            shares = int(value / price)
            
            new_holdings.append({
                "symbol": symbol,
                "weight": weight,
                "shares": shares,
                "value": value
            })
        
        # Update portfolio in Neo4j
        success = neo4j.update_portfolio_holdings(portfolio_id, new_holdings)
        
        if success:
            return {
                "status": "success",
                "portfolio_id": portfolio_id,
                "new_holdings": new_holdings,
                "rebalanced_at": datetime.now()
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to update portfolio")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{portfolio_id}/performance")
async def get_portfolio_performance(
    portfolio_id: str,
    period: str = "1M",
    neo4j: Neo4jService = Depends(get_neo4j)
) -> Dict[str, Any]:
    """Get portfolio performance metrics over time"""
    # TODO: Implement historical performance tracking
    return {
        "portfolio_id": portfolio_id,
        "period": period,
        "returns": {
            "1D": 0.015,
            "1W": 0.032,
            "1M": 0.058,
            "3M": 0.145,
            "YTD": 0.223,
            "1Y": 0.287
        },
        "benchmark_comparison": {
            "portfolio_return": 0.287,
            "benchmark_return": 0.215,
            "alpha": 0.072
        }
    }

@router.get("/{portfolio_id}/transactions")
async def get_portfolio_transactions(
    portfolio_id: str,
    limit: int = 50,
    neo4j: Neo4jService = Depends(get_neo4j)
) -> List[Dict[str, Any]]:
    """Get recent portfolio transactions"""
    # TODO: Implement transaction history
    return [
        {
            "transaction_id": "TXN_001",
            "date": "2024-01-15",
            "type": "BUY",
            "symbol": "AAPL",
            "shares": 100,
            "price": 185.50,
            "total": 18550.00
        }
    ]