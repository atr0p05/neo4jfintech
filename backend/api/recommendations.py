"""
Recommendations API endpoints
"""
from fastapi import APIRouter, HTTPException, Depends, Request
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from datetime import datetime

from services.neo4j import Neo4jService
from core.optimization import PortfolioOptimizer
from services.graphrag import GraphRAGService

router = APIRouter()

class RecommendationRequest(BaseModel):
    client_id: str
    recommendation_type: str = Field(
        default="all",
        description="Type of recommendations: all, rebalance, opportunities, risk"
    )
    constraints: Optional[Dict[str, Any]] = None

class OptimizationRequest(BaseModel):
    client_id: str
    objective: str = Field(
        default="sharpe",
        description="Optimization objective: sharpe, min_volatility, max_return, risk_parity"
    )
    constraints: Optional[Dict[str, Any]] = None

def get_neo4j(request: Request) -> Neo4jService:
    return request.app.state.neo4j

@router.post("/generate")
async def generate_recommendations(
    request: RecommendationRequest,
    neo4j: Neo4jService = Depends(get_neo4j)
) -> Dict[str, Any]:
    """Generate personalized recommendations for a client"""
    try:
        # Get client profile
        client_data = neo4j.get_client_portfolio(request.client_id)
        if not client_data:
            raise HTTPException(status_code=404, detail="Client not found")
        
        recommendations = {
            "client_id": request.client_id,
            "generated_at": datetime.now(),
            "recommendations": {}
        }
        
        # Generate different types of recommendations
        if request.recommendation_type in ["all", "rebalance"]:
            recommendations["recommendations"]["rebalancing"] = await _get_rebalancing_recommendations(
                neo4j, client_data, request.constraints
            )
        
        if request.recommendation_type in ["all", "opportunities"]:
            recommendations["recommendations"]["opportunities"] = neo4j.get_investment_opportunities(
                request.client_id, limit=10
            )
        
        if request.recommendation_type in ["all", "risk"]:
            recommendations["recommendations"]["risk_management"] = await _get_risk_recommendations(
                neo4j, client_data
            )
        
        return recommendations
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/optimize")
async def optimize_portfolio(
    request: OptimizationRequest,
    neo4j: Neo4jService = Depends(get_neo4j)
) -> Dict[str, Any]:
    """Run portfolio optimization for a client"""
    try:
        # Get client data
        client_data = neo4j.get_client_portfolio(request.client_id)
        if not client_data:
            raise HTTPException(status_code=404, detail="Client not found")
        
        # TODO: Fetch historical returns data
        # For demo, using placeholder data
        returns_data = None  # Would fetch from market data service
        prices_data = None   # Would fetch current prices
        
        # Initialize optimizer
        # optimizer = PortfolioOptimizer(returns_data, prices_data)
        
        # Run optimization
        # optimal_weights = optimizer.optimize_portfolio(
        #     objective=request.objective,
        #     constraints=request.constraints,
        #     risk_tolerance=client_data["client"]["riskTolerance"].lower()
        # )
        
        # For demo, return mock optimized portfolio
        optimal_weights = {
            "AAPL": 0.15,
            "MSFT": 0.12,
            "JPM": 0.10,
            "TLT": 0.20,
            "IEF": 0.15,
            "LQD": 0.10,
            "GLD": 0.08,
            "VNQ": 0.10
        }
        
        return {
            "client_id": request.client_id,
            "optimization_method": request.objective,
            "optimal_weights": optimal_weights,
            "expected_metrics": {
                "return": 0.085,
                "volatility": 0.12,
                "sharpe_ratio": 1.25
            },
            "constraints_applied": request.constraints or {}
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/insights/{client_id}")
async def get_ai_insights(
    client_id: str,
    neo4j: Neo4jService = Depends(get_neo4j)
) -> Dict[str, Any]:
    """Get AI-generated insights using GraphRAG"""
    try:
        # TODO: Implement GraphRAG integration
        # graphrag = GraphRAGService()
        # insights = await graphrag.generate_insights(client_id)
        
        # For demo, return mock insights
        return {
            "client_id": client_id,
            "insights": [
                {
                    "type": "market_opportunity",
                    "title": "Technology Sector Rotation",
                    "description": "Based on factor analysis, consider rotating from growth to value tech stocks",
                    "confidence": 0.85,
                    "action_items": [
                        "Reduce MSFT position by 5%",
                        "Add IBM and CSCO for value exposure"
                    ]
                },
                {
                    "type": "risk_alert",
                    "title": "Interest Rate Sensitivity",
                    "description": "Portfolio has high duration risk with current bond allocation",
                    "confidence": 0.92,
                    "action_items": [
                        "Consider short-duration bonds",
                        "Add floating-rate securities"
                    ]
                }
            ],
            "generated_at": datetime.now()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

async def _get_rebalancing_recommendations(
    neo4j: Neo4jService,
    client_data: Dict[str, Any],
    constraints: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    """Generate rebalancing recommendations"""
    portfolio = client_data["portfolio"]
    risk_tolerance = client_data["client"]["riskTolerance"]
    
    # Define target allocations based on risk tolerance
    target_allocations = {
        "Conservative": {"equity": 0.3, "bond": 0.6, "alternative": 0.1},
        "Moderate": {"equity": 0.5, "bond": 0.35, "alternative": 0.15},
        "Aggressive": {"equity": 0.7, "bond": 0.2, "alternative": 0.1}
    }
    
    targets = target_allocations[risk_tolerance]
    
    # Calculate current allocations
    current = {"equity": 0, "bond": 0, "alternative": 0}
    for holding in portfolio["holdings"]:
        asset_class = holding["assetClass"].lower()
        if asset_class in current:
            current[asset_class] += holding["weight"]
    
    # Generate recommendations
    recommendations = []
    for asset_class, target in targets.items():
        diff = target - current[asset_class]
        if abs(diff) > 0.02:  # Only recommend if difference > 2%
            recommendations.append({
                "asset_class": asset_class,
                "current_allocation": round(current[asset_class] * 100, 1),
                "target_allocation": round(target * 100, 1),
                "action": "increase" if diff > 0 else "decrease",
                "amount": round(abs(diff) * portfolio["totalValue"], 2)
            })
    
    return {
        "recommendations": recommendations,
        "estimated_improvement": {
            "sharpe_ratio": 0.15,
            "risk_reduction": 0.08
        }
    }

async def _get_risk_recommendations(
    neo4j: Neo4jService,
    client_data: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """Generate risk management recommendations"""
    risk_analytics = neo4j.get_risk_analytics(client_data["client"]["id"])
    
    recommendations = []
    
    # Check concentration risk
    if risk_analytics["concentration_risk"]:
        for concentration in risk_analytics["concentration_risk"]:
            recommendations.append({
                "type": "concentration_risk",
                "severity": "high",
                "sector": concentration["sector"],
                "current_weight": concentration["weight"],
                "recommendation": f"Reduce {concentration['sector']} exposure below 25%",
                "impact": "Reduce portfolio concentration risk"
            })
    
    # Check factor exposures
    high_exposures = [
        f for f in risk_analytics["factor_exposures"]
        if abs(f["exposure"]) > 0.5
    ]
    
    for exposure in high_exposures:
        recommendations.append({
            "type": "factor_risk",
            "severity": "medium",
            "factor": exposure["factor"],
            "exposure": exposure["exposure"],
            "recommendation": f"Hedge {exposure['factor']} exposure",
            "impact": "Reduce systematic risk"
        })
    
    return recommendations