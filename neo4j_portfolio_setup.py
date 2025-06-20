from neo4j import GraphDatabase
import random
import numpy as np
from datetime import datetime, timedelta
import pandas as pd

class Neo4jPortfolioSetup:
    def __init__(self, uri, user, password, database="neo4j"):
        self.driver = GraphDatabase.driver(uri, auth=(user, password), database=database)
        
    def close(self):
        self.driver.close()
        
    def clear_database(self):
        """Clear existing data - USE WITH CAUTION"""
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
            print("Database cleared")
    
    def create_constraints(self):
        """Create constraints and indexes"""
        constraints = [
            "CREATE CONSTRAINT client_id_unique IF NOT EXISTS FOR (c:Client) REQUIRE c.clientId IS UNIQUE",
            "CREATE CONSTRAINT portfolio_id_unique IF NOT EXISTS FOR (p:Portfolio) REQUIRE p.portfolioId IS UNIQUE",
            "CREATE CONSTRAINT asset_symbol_unique IF NOT EXISTS FOR (a:Asset) REQUIRE a.symbol IS UNIQUE",
            "CREATE CONSTRAINT risk_profile_id_unique IF NOT EXISTS FOR (rp:RiskProfile) REQUIRE rp.profileId IS UNIQUE"
        ]
        
        indexes = [
            "CREATE INDEX client_risk_tolerance IF NOT EXISTS FOR (c:Client) ON (c.riskTolerance)",
            "CREATE INDEX asset_class IF NOT EXISTS FOR (a:Asset) ON (a.assetClass)",
            "CREATE INDEX asset_sector IF NOT EXISTS FOR (a:Asset) ON (a.sector)",
            "CREATE INDEX portfolio_type IF NOT EXISTS FOR (p:Portfolio) ON (p.portfolioType)"
        ]
        
        with self.driver.session() as session:
            for constraint in constraints:
                session.run(constraint)
            for index in indexes:
                session.run(index)
            print("Constraints and indexes created")
    
    def generate_synthetic_clients(self, num_clients=100):
        """Generate synthetic client data"""
        first_names = ["Sarah", "Michael", "Emma", "James", "Lisa", "David", "Jennifer", "Robert", "Maria", "John"]
        last_names = ["Chen", "Rodriguez", "Thompson", "Wilson", "Anderson", "Martinez", "Taylor", "Lee", "Garcia", "Brown"]
        
        risk_tolerances = ["Conservative", "Moderate", "Aggressive"]
        experience_levels = ["<1 year", "1-5 years", "5-10 years", "10+ years"]
        
        clients = []
        for i in range(num_clients):
            # Generate correlated wealth and risk tolerance
            age = random.randint(25, 75)
            wealth_factor = np.random.lognormal(13, 1.5)  # Log-normal wealth distribution
            
            # Older clients tend to be more conservative
            risk_weight = max(0, min(2, int(2.5 - (age - 25) / 25)))
            risk_tolerance = risk_tolerances[risk_weight] if random.random() > 0.3 else random.choice(risk_tolerances)
            
            # Experience correlates with age
            exp_index = min(3, int((age - 25) / 12))
            experience = experience_levels[exp_index]
            
            client = {
                'clientId': f'CLI_{100000 + i}',
                'name': f'{random.choice(first_names)} {random.choice(last_names)}',
                'riskTolerance': risk_tolerance,
                'netWorth': round(wealth_factor, 2),
                'investmentExperience': experience,
                'age': age
            }
            clients.append(client)
            
        return clients
    
    def generate_synthetic_assets(self):
        """Generate realistic asset universe"""
        # Real-world inspired assets
        equity_assets = [
            {"symbol": "AAPL", "name": "Apple Inc.", "sector": "Technology", "price": 185.50},
            {"symbol": "MSFT", "name": "Microsoft Corp.", "sector": "Technology", "price": 415.00},
            {"symbol": "JPM", "name": "JPMorgan Chase", "sector": "Financials", "price": 170.25},
            {"symbol": "BAC", "name": "Bank of America", "sector": "Financials", "price": 35.50},
            {"symbol": "JNJ", "name": "Johnson & Johnson", "sector": "Healthcare", "price": 155.00},
            {"symbol": "PFE", "name": "Pfizer Inc.", "sector": "Healthcare", "price": 28.50},
            {"symbol": "XOM", "name": "Exxon Mobil", "sector": "Energy", "price": 105.00},
            {"symbol": "CVX", "name": "Chevron Corp.", "sector": "Energy", "price": 155.00},
            {"symbol": "PG", "name": "Procter & Gamble", "sector": "Consumer Staples", "price": 155.00},
            {"symbol": "KO", "name": "Coca-Cola Co.", "sector": "Consumer Staples", "price": 60.00}
        ]
        
        bond_assets = [
            {"symbol": "TLT", "name": "20+ Year Treasury", "sector": "Government", "price": 95.50},
            {"symbol": "IEF", "name": "7-10 Year Treasury", "sector": "Government", "price": 101.00},
            {"symbol": "LQD", "name": "Investment Grade Corp", "sector": "Corporate", "price": 105.25},
            {"symbol": "HYG", "name": "High Yield Corp", "sector": "Corporate", "price": 75.00},
            {"symbol": "MUB", "name": "Municipal Bonds", "sector": "Municipal", "price": 108.00}
        ]
        
        alternative_assets = [
            {"symbol": "GLD", "name": "Gold ETF", "sector": "Commodities", "price": 185.00},
            {"symbol": "SLV", "name": "Silver ETF", "sector": "Commodities", "price": 23.50},
            {"symbol": "VNQ", "name": "Real Estate ETF", "sector": "Real Estate", "price": 85.50},
            {"symbol": "DBC", "name": "Commodity Index", "sector": "Commodities", "price": 20.00}
        ]
        
        assets = []
        for asset in equity_assets:
            asset['assetClass'] = 'Equity'
            assets.append(asset)
        for asset in bond_assets:
            asset['assetClass'] = 'Bond'
            assets.append(asset)
        for asset in alternative_assets:
            asset['assetClass'] = 'Alternative'
            assets.append(asset)
            
        return assets
    
    def generate_risk_profiles(self, clients):
        """Generate risk profiles based on client characteristics"""
        risk_profiles = []
        
        risk_mapping = {
            'Conservative': {'score': (2, 4), 'var95': (0.03, 0.06), 'vol': (0.08, 0.12)},
            'Moderate': {'score': (4.5, 6.5), 'var95': (0.06, 0.09), 'vol': (0.12, 0.18)},
            'Aggressive': {'score': (7, 9), 'var95': (0.09, 0.15), 'vol': (0.18, 0.30)}
        }
        
        for i, client in enumerate(clients):
            mapping = risk_mapping[client['riskTolerance']]
            
            profile = {
                'profileId': f'RP_{300000 + i}',
                'clientId': client['clientId'],
                'riskScore': round(random.uniform(*mapping['score']), 1),
                'assessmentDate': datetime.now() - timedelta(days=random.randint(1, 30)),
                'var95': round(random.uniform(*mapping['var95']), 3),
                'volatilityTolerance': round(random.uniform(*mapping['vol']), 3)
            }
            risk_profiles.append(profile)
            
        return risk_profiles
    
    def create_nodes_batch(self, node_type, data, batch_size=1000):
        """Create nodes in batches for better performance"""
        with self.driver.session() as session:
            for i in range(0, len(data), batch_size):
                batch = data[i:i + batch_size]
                
                if node_type == 'Client':
                    query = """
                    UNWIND $batch AS client
                    CREATE (c:Client {
                        clientId: client.clientId,
                        name: client.name,
                        riskTolerance: client.riskTolerance,
                        netWorth: client.netWorth,
                        investmentExperience: client.investmentExperience
                    })
                    """
                elif node_type == 'Asset':
                    query = """
                    UNWIND $batch AS asset
                    CREATE (a:Asset {
                        symbol: asset.symbol,
                        name: asset.name,
                        assetClass: asset.assetClass,
                        currentPrice: asset.price,
                        sector: asset.sector
                    })
                    """
                elif node_type == 'RiskProfile':
                    query = """
                    UNWIND $batch AS profile
                    CREATE (rp:RiskProfile {
                        profileId: profile.profileId,
                        riskScore: profile.riskScore,
                        assessmentDate: profile.assessmentDate,
                        var95: profile.var95,
                        volatilityTolerance: profile.volatilityTolerance
                    })
                    """
                
                session.run(query, batch=batch)
            
            print(f"Created {len(data)} {node_type} nodes")
    
    def create_relationships(self, risk_profiles):
        """Create relationships between nodes"""
        with self.driver.session() as session:
            # Connect clients to risk profiles
            for profile in risk_profiles:
                query = """
                MATCH (c:Client {clientId: $clientId})
                MATCH (rp:RiskProfile {profileId: $profileId})
                CREATE (c)-[:HAS_RISK_PROFILE]->(rp)
                """
                session.run(query, clientId=profile['clientId'], profileId=profile['profileId'])
            
            print(f"Created {len(risk_profiles)} client-risk profile relationships")
            
            # Create risk factor relationships for assets
            risk_factors = [
                {'name': 'Market Risk', 'assets': ['AAPL', 'MSFT', 'JPM'], 'exposure': 0.8},
                {'name': 'Interest Rate Risk', 'assets': ['TLT', 'IEF', 'LQD'], 'exposure': 0.9},
                {'name': 'Credit Risk', 'assets': ['HYG', 'LQD'], 'exposure': 0.7},
                {'name': 'Commodity Risk', 'assets': ['GLD', 'SLV', 'DBC'], 'exposure': 0.85}
            ]
            
            # First create risk factor nodes
            for rf in risk_factors:
                session.run("""
                CREATE (rf:RiskFactor {
                    name: $name,
                    category: 'Systematic'
                })
                """, name=rf['name'])
            
            # Then create relationships
            for rf in risk_factors:
                for asset in rf['assets']:
                    session.run("""
                    MATCH (a:Asset {symbol: $symbol})
                    MATCH (rf:RiskFactor {name: $riskFactor})
                    CREATE (a)-[:EXPOSED_TO {exposureLevel: $exposure}]->(rf)
                    """, symbol=asset, riskFactor=rf['name'], 
                    exposure=rf['exposure'] + random.uniform(-0.1, 0.1))
    
    def generate_sample_portfolios(self, clients, assets):
        """Generate portfolios with holdings"""
        portfolios = []
        portfolio_holdings = []
        
        # Asset allocation templates based on risk tolerance
        allocation_templates = {
            'Conservative': {
                'Equity': 0.3, 'Bond': 0.6, 'Alternative': 0.1
            },
            'Moderate': {
                'Equity': 0.5, 'Bond': 0.35, 'Alternative': 0.15
            },
            'Aggressive': {
                'Equity': 0.7, 'Bond': 0.2, 'Alternative': 0.1
            }
        }
        
        with self.driver.session() as session:
            for i, client in enumerate(clients[:20]):  # Create portfolios for first 20 clients
                portfolio_id = f'PRT_{200000 + i}'
                portfolio_value = client['netWorth'] * random.uniform(0.3, 0.8)
                
                # Create portfolio
                session.run("""
                CREATE (p:Portfolio {
                    portfolioId: $portfolioId,
                    totalValue: $totalValue,
                    portfolioType: $portfolioType,
                    benchmark: $benchmark
                })
                """, portfolioId=portfolio_id, 
                    totalValue=round(portfolio_value, 2),
                    portfolioType=random.choice(['Discretionary', 'Advisory']),
                    benchmark=random.choice(['S&P 500', 'Russell 3000', '60/40 Balanced']))
                
                # Connect to client
                session.run("""
                MATCH (c:Client {clientId: $clientId})
                MATCH (p:Portfolio {portfolioId: $portfolioId})
                CREATE (c)-[:OWNS_PORTFOLIO {since: date()}]->(p)
                """, clientId=client['clientId'], portfolioId=portfolio_id)
                
                # Create holdings based on allocation template
                template = allocation_templates[client['riskTolerance']]
                
                for asset_class, target_weight in template.items():
                    # Select assets from this class
                    class_assets = [a for a in assets if a['assetClass'] == asset_class]
                    num_holdings = random.randint(2, min(4, len(class_assets)))
                    selected_assets = random.sample(class_assets, num_holdings)
                    
                    # Distribute weight among selected assets
                    weights = np.random.dirichlet(np.ones(num_holdings)) * target_weight
                    
                    for asset, weight in zip(selected_assets, weights):
                        shares = int((portfolio_value * weight) / asset['price'])
                        if shares > 0:
                            session.run("""
                            MATCH (p:Portfolio {portfolioId: $portfolioId})
                            MATCH (a:Asset {symbol: $symbol})
                            CREATE (p)-[:HOLDS {
                                shares: $shares,
                                weight: $weight,
                                value: $value
                            }]->(a)
                            """, portfolioId=portfolio_id,
                                symbol=asset['symbol'],
                                shares=shares,
                                weight=round(weight, 3),
                                value=round(shares * asset['price'], 2))
        
        print("Created sample portfolios with holdings")
    
    def run_full_setup(self):
        """Run the complete setup process"""
        print("Starting Neo4j portfolio setup...")
        
        # 1. Create constraints and indexes
        self.create_constraints()
        
        # 2. Generate synthetic data
        print("Generating synthetic data...")
        clients = self.generate_synthetic_clients(100)
        assets = self.generate_synthetic_assets()
        risk_profiles = self.generate_risk_profiles(clients)
        
        # 3. Create nodes
        print("Creating nodes...")
        self.create_nodes_batch('Client', clients)
        self.create_nodes_batch('Asset', assets)
        self.create_nodes_batch('RiskProfile', risk_profiles)
        
        # 4. Create relationships
        print("Creating relationships...")
        self.create_relationships(risk_profiles)
        
        # 5. Generate sample portfolios
        print("Generating sample portfolios...")
        self.generate_sample_portfolios(clients, assets)
        
        # 6. Verify setup
        with self.driver.session() as session:
            result = session.run("""
            MATCH (n)
            RETURN labels(n)[0] as NodeType, count(n) as Count
            ORDER BY Count DESC
            """)
            print("\nNode counts:")
            for record in result:
                print(f"  {record['NodeType']}: {record['Count']}")
        
        print("\nSetup complete!")

# Usage
if __name__ == "__main__":
    # Neo4j Aura connection details
    uri = "neo4j+s://1c34ddb0.databases.neo4j.io"
    user = "neo4j"
    password = "Sw-DAlLfV1_Rw4lFlXAxl0DoN4lCauWrp29IZnfq_zM"
    database = "neo4j"
    
    # Test connection first
    print(f"Connecting to Neo4j at {uri}...")
    
    try:
        setup = Neo4jPortfolioSetup(uri, user, password, database)
        
        # Test connection with a simple query
        with setup.driver.session() as session:
            result = session.run("RETURN 1 as test")
            print("Connection successful!")
            
            # Check if database has existing data
            count_result = session.run("MATCH (n) RETURN count(n) as count")
            count = count_result.single()['count']
            print(f"Database currently has {count} nodes")
            
            if count > 0:
                print("Database has existing data. Clearing it...")
                setup.clear_database()
        
        # Run full setup
        setup.run_full_setup()
        
    except Exception as e:
        print(f"Error: {e}")
        print("Please check:")
        print("1. Your Neo4j instance is running")
        print("2. The connection URI is correct") 
        print("3. Your credentials are valid")
        print("4. You have network connectivity")
    finally:
        if 'setup' in locals():
            setup.close()