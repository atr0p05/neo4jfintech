#!/usr/bin/env python3
"""
Initialize Neo4j database with the investment platform schema and sample data
"""
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# Import the setup scripts we created earlier
sys.path.append(str(Path(__file__).parent.parent.parent))

from neo4j_portfolio_setup import Neo4jPortfolioSetup
from portfolio_risk_analytics import PortfolioRiskAnalytics

def main():
    """Initialize the database"""
    # Load environment variables
    load_dotenv()
    
    # Get Neo4j credentials
    uri = os.getenv("NEO4J_URI")
    user = os.getenv("NEO4J_USER")
    password = os.getenv("NEO4J_PASSWORD")
    
    if not all([uri, user, password]):
        print("❌ Error: Please set NEO4J_URI, NEO4J_USER, and NEO4J_PASSWORD in .env file")
        return
    
    print("🚀 Initializing Neo4j Investment Platform Database...")
    
    try:
        # Step 1: Set up base data
        print("\n📊 Setting up portfolio data...")
        setup = Neo4jPortfolioSetup(uri, user, password)
        setup.run_full_setup()
        setup.close()
        
        # Step 2: Add risk analytics
        print("\n📈 Adding risk analytics...")
        analytics = PortfolioRiskAnalytics(uri, user, password)
        analytics.add_risk_metrics_to_assets()
        analytics.create_factor_exposures()
        analytics.calculate_portfolio_risk_metrics()
        analytics.create_correlation_relationships()
        analytics.close()
        
        print("\n✅ Database initialization complete!")
        print("\n📋 You can now:")
        print("1. Start the backend API: cd backend && uvicorn main:app --reload")
        print("2. Start the frontend: cd frontend && npm run dev")
        print("3. Access the API docs at: http://localhost:8000/docs")
        
    except Exception as e:
        print(f"\n❌ Error during initialization: {str(e)}")
        print("Please check your Neo4j connection and try again.")

if __name__ == "__main__":
    main()