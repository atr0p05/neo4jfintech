"""
Advanced Portfolio Optimization API endpoints
"""
from fastapi import APIRouter, HTTPException, Depends, Request, BackgroundTasks
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from datetime import datetime
import pandas as pd

from core.optimization.advanced_optimizer import AdvancedPortfolioOptimizer
from services.neo4j import Neo4jService

router = APIRouter()

class OptimizationRequest(BaseModel):
    client_id: str
    optimization_model: str = Field(
        default="markowitz",
        description="Model: markowitz, black_litterman, risk_parity, max_sharpe, min_volatility, factor_based, cvar"
    )
    constraints: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Custom constraints: min_weight, max_weight, sector_limits, asset_class_limits"
    )
    views: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Market views for Black-Litterman: {'absolute': {'AAPL': 0.15}, 'relative': [('AAPL', 'MSFT', 0.02)]}"
    )

class ModelComparisonRequest(BaseModel):
    client_id: str
    models: Optional[List[str]] = Field(
        default=None,
        description="List of models to compare. Default: markowitz, black_litterman, risk_parity, min_volatility"
    )

class BacktestRequest(BaseModel):
    client_id: str
    optimization_model: str
    start_date: str = Field(description="Start date in YYYY-MM-DD format")
    end_date: str = Field(description="End date in YYYY-MM-DD format")
    rebalance_frequency: str = Field(
        default="quarterly",
        description="Rebalancing frequency: daily, weekly, monthly, quarterly, yearly"
    )

def get_neo4j(request: Request) -> Neo4jService:
    return request.app.state.neo4j

@router.post("/optimize")
async def optimize_portfolio(
    request: OptimizationRequest,
    neo4j: Neo4jService = Depends(get_neo4j)
) -> Dict[str, Any]:
    """Run advanced portfolio optimization"""
    try:
        # Initialize optimizer
        optimizer = AdvancedPortfolioOptimizer(neo4j.driver)
        
        # Run optimization
        results = optimizer.optimize_portfolio(
            client_id=request.client_id,
            optimization_model=request.optimization_model,
            constraints=request.constraints,
            views=request.views
        )
        
        # Format response
        return {
            "status": "success",
            "client_id": request.client_id,
            "optimization_model": request.optimization_model,
            "results": {
                "weights": results['weights'],
                "metrics": {
                    "expected_return": round(results['expected_return'] * 100, 2),
                    "volatility": round(results['volatility'] * 100, 2),
                    "sharpe_ratio": round(results['sharpe_ratio'], 3),
                    "diversification_ratio": round(results.get('diversification_ratio', 0), 2)
                },
                "allocation": {
                    "number_of_assets": results['number_of_assets'],
                    "discrete_allocation": results['discrete_allocation'],
                    "leftover_cash": round(results['leftover_cash'], 2)
                },
                "risk_contribution": results.get('risk_contribution', {}),
                "factor_exposures": results.get('factor_exposures', {})
            },
            "timestamp": datetime.now()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/compare-models")
async def compare_optimization_models(
    request: ModelComparisonRequest,
    neo4j: Neo4jService = Depends(get_neo4j)
) -> Dict[str, Any]:
    """Compare different optimization models for a client"""
    try:
        # Initialize optimizer
        optimizer = AdvancedPortfolioOptimizer(neo4j.driver)
        
        # Run comparison
        comparison_df = optimizer.compare_optimization_models(
            client_id=request.client_id,
            models=request.models
        )
        
        # Convert DataFrame to dict for JSON response
        comparison_results = comparison_df.to_dict('records')
        
        # Get detailed results for each model
        detailed_results = {}
        for model in request.models or ['markowitz', 'black_litterman', 'risk_parity', 'min_volatility']:
            try:
                result = optimizer.optimize_portfolio(request.client_id, model)
                detailed_results[model] = {
                    "top_holdings": sorted(
                        [(k, v) for k, v in result['weights'].items() if v > 0.01],
                        key=lambda x: x[1],
                        reverse=True
                    )[:5],
                    "risk_contribution": list(sorted(
                        result.get('risk_contribution', {}).items(),
                        key=lambda x: x[1],
                        reverse=True
                    )[:5])
                }
            except:
                detailed_results[model] = {"error": "Optimization failed"}
        
        return {
            "status": "success",
            "client_id": request.client_id,
            "comparison": comparison_results,
            "detailed_results": detailed_results,
            "recommendation": _get_model_recommendation(comparison_results),
            "timestamp": datetime.now()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/efficient-frontier/{client_id}")
async def get_efficient_frontier(
    client_id: str,
    num_points: int = 20,
    neo4j: Neo4jService = Depends(get_neo4j)
) -> Dict[str, Any]:
    """Calculate efficient frontier points for visualization"""
    try:
        optimizer = AdvancedPortfolioOptimizer(neo4j.driver)
        
        # Get client data
        client_data = optimizer._get_client_data(client_id)
        market_data = optimizer._get_market_data(client_data['universe'])
        
        # Calculate efficient frontier points
        frontier_points = []
        target_returns = np.linspace(
            market_data['expected_returns'].min(),
            market_data['expected_returns'].max(),
            num_points
        )
        
        for target_return in target_returns:
            try:
                # Create efficient frontier
                ef = EfficientFrontier(
                    market_data['expected_returns'],
                    market_data['cov_matrix']
                )
                
                # Optimize for target return
                ef.efficient_return(target_return)
                performance = ef.portfolio_performance(verbose=False)
                
                frontier_points.append({
                    "expected_return": performance[0],
                    "volatility": performance[1],
                    "sharpe_ratio": performance[2]
                })
            except:
                continue
        
        # Add current portfolio point
        current_portfolio = neo4j.get_portfolio_metrics(client_data['portfolios'][0]['portfolioId'])
        
        return {
            "client_id": client_id,
            "efficient_frontier": frontier_points,
            "current_portfolio": {
                "expected_return": current_portfolio['metrics']['expected_return'],
                "volatility": current_portfolio['metrics']['volatility'],
                "sharpe_ratio": current_portfolio['metrics']['sharpe_ratio']
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/backtest")
async def backtest_optimization(
    request: BacktestRequest,
    background_tasks: BackgroundTasks,
    neo4j: Neo4jService = Depends(get_neo4j)
) -> Dict[str, Any]:
    """Run backtesting for optimization strategy"""
    try:
        # For now, return a placeholder
        # In production, this would trigger a background job
        job_id = f"backtest_{request.client_id}_{datetime.now().timestamp()}"
        
        # Add background task
        background_tasks.add_task(
            _run_backtest,
            neo4j.driver,
            request.client_id,
            request.optimization_model,
            request.start_date,
            request.end_date,
            request.rebalance_frequency,
            job_id
        )
        
        return {
            "status": "accepted",
            "job_id": job_id,
            "message": "Backtest job started. Check /api/optimization/backtest-results/{job_id} for results",
            "estimated_time": "2-5 minutes"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/optimization-history/{client_id}")
async def get_optimization_history(
    client_id: str,
    limit: int = 10,
    neo4j: Neo4jService = Depends(get_neo4j)
) -> List[Dict[str, Any]]:
    """Get historical optimization results for a client"""
    try:
        with neo4j.driver.session() as session:
            results = session.run("""
                MATCH (c:Client {clientId: $clientId})-[:HAS_OPTIMIZATION]->(opt:Optimization)
                OPTIONAL MATCH (opt)-[rw:RECOMMENDS_WEIGHT]->(a:Asset)
                WITH opt, collect({
                    symbol: a.symbol,
                    weight: rw.weight,
                    shares: rw.shares
                }) as recommendations
                RETURN opt {
                    .*,
                    recommendations: recommendations
                }
                ORDER BY opt.timestamp DESC
                LIMIT $limit
            """, clientId=client_id, limit=limit)
            
            history = []
            for record in results:
                opt = dict(record['opt'])
                opt['timestamp'] = str(opt['timestamp'])
                history.append(opt)
            
            return history
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def _get_model_recommendation(comparison_results: List[Dict]) -> Dict[str, str]:
    """Get recommendation based on model comparison"""
    if not comparison_results:
        return {"model": "markowitz", "reason": "Default recommendation"}
    
    # Find model with highest Sharpe ratio
    best_model = max(comparison_results, key=lambda x: float(x['Sharpe Ratio']))
    
    return {
        "model": best_model['Model'],
        "reason": f"Highest risk-adjusted returns with Sharpe ratio of {best_model['Sharpe Ratio']}"
    }

async def _run_backtest(
    neo4j_driver,
    client_id: str,
    optimization_model: str,
    start_date: str,
    end_date: str,
    rebalance_frequency: str,
    job_id: str
):
    """Background task to run backtesting"""
    # TODO: Implement actual backtesting logic
    # This would:
    # 1. Fetch historical price data
    # 2. Run optimization at each rebalance date
    # 3. Calculate returns and track performance
    # 4. Save results to database
    pass