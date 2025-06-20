#!/usr/bin/env python3
"""
Show where backtest results are stored and how to access them
"""
import requests
import json
import os
from datetime import datetime

def show_storage_locations():
    """Show all storage locations for backtest results"""
    print("📂 BACKTEST RESULTS STORAGE LOCATIONS")
    print("=" * 60)
    
    print("\n1️⃣ NEO4J DATABASE (Primary Storage)")
    print("   📍 Location: Neo4j Aura Cloud Database")
    print("   🔗 URI: neo4j+s://1c34ddb0.databases.neo4j.io")
    print("   📊 Node Type: BacktestResult")
    print("   🏷️ Properties: strategyName, metrics, timestamp, etc.")
    
    print("\n2️⃣ LOCAL FILE SYSTEM")
    base_path = "/Users/test/neo4j-investment-platform"
    
    storage_paths = {
        "📁 Main Project": base_path,
        "🔬 Backtest Results": f"{base_path}/data/backtest_results/",
        "💾 Cache Data": f"{base_path}/data/cache/",
        "📊 Reports": f"{base_path}/data/reports/",
        "🔧 Backend Code": f"{base_path}/backend/",
        "🎯 Scripts": f"{base_path}/scripts/"
    }
    
    for name, path in storage_paths.items():
        exists = "✅ EXISTS" if os.path.exists(path) else "❌ NOT FOUND"
        print(f"   {name}: {path} {exists}")
    
    print("\n3️⃣ API RESPONSES (Runtime)")
    print("   📡 HTTP Responses: Real-time JSON data")
    print("   🔄 WebSocket: Live streaming results")
    print("   🖥️ Frontend State: Browser memory/localStorage")

def show_neo4j_backtest_query():
    """Show how to query backtest results from Neo4j"""
    print("\n📊 NEO4J BACKTEST QUERIES")
    print("=" * 40)
    
    queries = {
        "All Backtests": """
        MATCH (b:BacktestResult)
        RETURN b.backtestId, b.strategyName, b.totalReturn, b.timestamp
        ORDER BY b.timestamp DESC
        """,
        
        "Recent Backtests": """
        MATCH (b:BacktestResult)
        WHERE b.timestamp > datetime() - duration('P7D')
        RETURN b.strategyName, b.sharpeRatio, b.totalReturn
        ORDER BY b.sharpeRatio DESC
        """,
        
        "Best Performing": """
        MATCH (b:BacktestResult)
        WHERE b.totalReturn > 0.10
        RETURN b.strategyName, b.totalReturn, b.sharpeRatio
        ORDER BY b.totalReturn DESC
        LIMIT 10
        """,
        
        "Strategy Comparison": """
        MATCH (b:BacktestResult)
        WITH b.strategyName as strategy, 
             avg(b.totalReturn) as avgReturn,
             avg(b.sharpeRatio) as avgSharpe,
             count(b) as numTests
        RETURN strategy, avgReturn, avgSharpe, numTests
        ORDER BY avgSharpe DESC
        """
    }
    
    for name, query in queries.items():
        print(f"\n🔍 {name}:")
        print(f"```cypher{query.strip()}```")

def show_api_access_methods():
    """Show how to access backtest results via API"""
    print("\n🌐 API ACCESS METHODS")
    print("=" * 40)
    
    endpoints = {
        "Run New Backtest": {
            "method": "POST",
            "url": "http://localhost:8000/api/backtest",
            "description": "Creates new backtest and returns results"
        },
        "Compare Strategies": {
            "method": "POST", 
            "url": "http://localhost:8000/api/backtest/compare",
            "description": "Compare multiple strategies"
        },
        "System Stats": {
            "method": "GET",
            "url": "http://localhost:8000/api/admin/system-stats", 
            "description": "Get database statistics"
        },
        "Analytics Dashboard": {
            "method": "GET",
            "url": "http://localhost:8000/analytics/dashboard",
            "description": "Interactive dashboard with all results"
        }
    }
    
    for name, info in endpoints.items():
        print(f"\n📡 {name}:")
        print(f"   Method: {info['method']}")
        print(f"   URL: {info['url']}")
        print(f"   Purpose: {info['description']}")

def check_actual_storage():
    """Check what's actually stored right now"""
    print("\n🔍 CURRENT STORAGE CHECK")
    print("=" * 40)
    
    # Check if results directory exists
    results_dir = "/Users/test/neo4j-investment-platform/data/backtest_results"
    if os.path.exists(results_dir):
        files = os.listdir(results_dir)
        print(f"📁 Local files found: {len(files)}")
        for file in files[:5]:  # Show first 5
            print(f"   📄 {file}")
    else:
        print("📁 Local results directory: Not created yet")
    
    # Check API connectivity
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            print("🌐 API Status: ✅ Connected")
            
            # Check if we can get system stats
            stats_response = requests.get("http://localhost:8000/api/admin/system-stats", timeout=5)
            if stats_response.status_code == 200:
                stats = stats_response.json()
                print(f"📊 Database Status: {stats['database']['neo4j_status']}")
            
        else:
            print("🌐 API Status: ❌ Error")
    except:
        print("🌐 API Status: ❌ Not reachable")

def create_sample_result_file():
    """Create a sample result file to show local storage"""
    print("\n💾 CREATING SAMPLE STORAGE")
    print("=" * 40)
    
    # Create directories if they don't exist
    base_dir = "/Users/test/neo4j-investment-platform/data"
    results_dir = f"{base_dir}/backtest_results"
    
    os.makedirs(results_dir, exist_ok=True)
    
    # Create sample result
    sample_result = {
        "backtest_id": f"BT_{int(datetime.now().timestamp())}",
        "strategy_name": "Sample Markowitz Strategy",
        "symbols": ["AAPL", "MSFT", "JPM", "TLT", "GLD"],
        "period": {
            "start_date": "2020-01-01",
            "end_date": "2023-12-31"
        },
        "metrics": {
            "total_return": 0.157,
            "annualized_return": 0.123,
            "volatility": 0.168,
            "sharpe_ratio": 0.73,
            "max_drawdown": -0.089
        },
        "created_at": datetime.now().isoformat(),
        "storage_location": "Local file + Neo4j database"
    }
    
    filename = f"{results_dir}/sample_backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    with open(filename, 'w') as f:
        json.dump(sample_result, f, indent=2)
    
    print(f"✅ Sample result saved to: {filename}")
    print(f"📂 Directory: {results_dir}")
    
    return filename

def show_retrieval_examples():
    """Show how to retrieve stored results"""
    print("\n🔄 HOW TO RETRIEVE STORED RESULTS")
    print("=" * 50)
    
    print("1️⃣ Via API (Recommended):")
    print("""
    # Get all backtests for a client
    curl http://localhost:8000/api/portfolios/CLI_100001/backtests
    
    # Get specific backtest
    curl http://localhost:8000/api/backtest/results/BT_12345
    """)
    
    print("2️⃣ Via Neo4j Browser:")
    print("""
    # Open: https://browser.neo4j.io/
    # Connect to: neo4j+s://1c34ddb0.databases.neo4j.io
    # Run: MATCH (b:BacktestResult) RETURN b LIMIT 25
    """)
    
    print("3️⃣ Via Python:")
    print("""
    import requests
    
    # Get latest backtests
    response = requests.get('http://localhost:8000/api/admin/system-stats')
    data = response.json()
    
    # Or run new backtest (auto-stored)
    backtest_data = {
        "symbols": ["AAPL", "MSFT"],
        "strategy": {"optimization_model": "markowitz"},
        "startDate": "2020-01-01",
        "endDate": "2023-12-31"
    }
    result = requests.post('http://localhost:8000/api/backtest', json=backtest_data)
    """)
    
    print("4️⃣ Via Frontend Dashboard:")
    print("   🌐 http://localhost:8000/analytics/dashboard")
    print("   📊 Interactive charts and tables")
    print("   📁 Download options for results")

if __name__ == "__main__":
    print("🗄️ BACKTEST RESULTS STORAGE GUIDE")
    print("=" * 60)
    
    show_storage_locations()
    check_actual_storage()
    
    # Create sample file
    sample_file = create_sample_result_file()
    
    show_neo4j_backtest_query()
    show_api_access_methods()
    show_retrieval_examples()
    
    print(f"\n✨ SUMMARY:")
    print("📊 Results are stored in:")
    print("   1. Neo4j Database (primary)")
    print("   2. Local JSON files (backup)")
    print("   3. API responses (real-time)")
    print("   4. Analytics dashboard (visual)")
    print(f"\n📁 Sample file created: {sample_file}")
    print("🌐 Access dashboard: http://localhost:8000/analytics/dashboard")
    print("📖 API docs: http://localhost:8000/docs")