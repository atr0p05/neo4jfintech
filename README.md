# Neo4j Investment Platform

A sophisticated investment management platform built with Neo4j graph database, providing advanced portfolio analytics, AI-driven recommendations, and comprehensive risk management.

## 🚀 Features

### Core Capabilities
- **Graph-Based Portfolio Management**: Leverage Neo4j for complex relationship modeling
- **Advanced Portfolio Optimization**: Multiple optimization strategies (Markowitz, Risk Parity, Black-Litterman)
- **Real-Time Risk Analytics**: Factor exposures, concentration risk, stress testing
- **AI-Powered Recommendations**: GraphRAG integration for intelligent insights
- **Comprehensive Backtesting**: Historical performance analysis and strategy validation

### Technical Features
- **FastAPI Backend**: High-performance async API
- **Next.js Frontend**: Modern React-based UI with real-time updates
- **Redis Caching**: Optimized performance for frequently accessed data
- **WebSocket Support**: Real-time portfolio updates
- **Docker Deployment**: Easy containerized deployment

## 📋 Prerequisites

- Python 3.8+
- Node.js 18+
- Neo4j Database (Aura or self-hosted)
- Redis (for caching)
- Docker (optional)

## 🛠️ Installation

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/neo4j-investment-platform.git
cd neo4j-investment-platform
```

### 2. Set Up Environment Variables
```bash
cp .env.example .env
# Edit .env with your credentials
```

### 3. Run Setup Script
```bash
python setup.py
```

Or manually:

#### Backend Setup
```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

#### Frontend Setup
```bash
cd frontend
npm install
```

### 4. Initialize Database
```bash
python scripts/init_database.py
```

## 🚀 Running the Application

### Development Mode

#### Start Backend
```bash
cd backend
uvicorn main:app --reload --port 8000
```

#### Start Frontend
```bash
cd frontend
npm run dev
```

### Production Mode with Docker
```bash
docker-compose up -d
```

## 📡 API Documentation

Once running, access the API documentation at:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

### Key Endpoints

#### Portfolio Management
- `GET /api/portfolio/client/{client_id}` - Get client portfolio
- `POST /api/portfolio/{portfolio_id}/rebalance` - Rebalance portfolio
- `GET /api/portfolio/{portfolio_id}/metrics` - Get portfolio metrics

#### Recommendations
- `POST /api/recommendations/generate` - Generate recommendations
- `POST /api/recommendations/optimize` - Run portfolio optimization
- `GET /api/recommendations/insights/{client_id}` - Get AI insights

#### Risk Analytics
- `GET /api/risk/analysis/{client_id}` - Comprehensive risk analysis
- `POST /api/risk/stress-test` - Run stress tests
- `GET /api/risk/factors/{portfolio_id}` - Factor exposure analysis

## 🏗️ Architecture

```
neo4j-investment-platform/
├── backend/
│   ├── api/              # FastAPI endpoints
│   ├── core/             # Business logic
│   │   ├── optimization/ # Portfolio optimization
│   │   ├── risk/        # Risk analytics
│   │   └── data/        # Data processing
│   ├── services/         # External services
│   │   ├── neo4j/       # Neo4j integration
│   │   ├── market_data/ # Market data feeds
│   │   ├── graphrag/    # AI/LLM integration
│   │   └── backtesting/ # Backtesting engine
│   ├── models/          # Pydantic models
│   └── utils/           # Utilities
├── frontend/
│   ├── components/      # React components
│   ├── pages/          # Next.js pages
│   ├── services/       # API services
│   └── utils/          # Frontend utilities
└── data/               # Data storage
```

## 🔧 Configuration

### Neo4j Schema

The platform uses a sophisticated graph model:

```cypher
// Core Entities
(Client)-[:HAS_RISK_PROFILE]->(RiskProfile)
(Client)-[:OWNS_PORTFOLIO]->(Portfolio)
(Portfolio)-[:HOLDS {weight, shares, value}]->(Asset)
(Asset)-[:HAS_FACTOR_EXPOSURE {exposure}]->(Factor)
(Asset)-[:CORRELATED_WITH {correlation}]->(Asset)
```

### Risk Profiles

- **Conservative**: Low risk, stable returns (30% equity, 60% bonds)
- **Moderate**: Balanced approach (50% equity, 35% bonds)
- **Aggressive**: High growth focus (70% equity, 20% bonds)

## 📊 Features Deep Dive

### Portfolio Optimization

Multiple optimization strategies:
- **Mean-Variance (Markowitz)**: Classic efficient frontier
- **Risk Parity**: Equal risk contribution
- **Black-Litterman**: Incorporating market views
- **Minimum Volatility**: Risk minimization
- **Maximum Sharpe**: Risk-adjusted return optimization

### Risk Analytics

Comprehensive risk metrics:
- **Factor Exposures**: Style and macro factor analysis
- **Concentration Risk**: Sector and asset concentration
- **Stress Testing**: Scenario analysis
- **VaR/CVaR**: Value at Risk calculations
- **Correlation Analysis**: Asset correlation matrix

### AI Integration

GraphRAG-powered features:
- **Intelligent Insights**: Context-aware recommendations
- **Market Analysis**: Real-time market intelligence
- **Peer Comparison**: Similar portfolio analysis
- **Anomaly Detection**: Unusual pattern identification

## 🧪 Testing

Run the test suite:

```bash
# Backend tests
cd backend
pytest tests/ -v --cov=.

# Frontend tests
cd frontend
npm test
```

## 📈 Performance Optimization

- **Redis Caching**: Frequently accessed data cached
- **Batch Processing**: Efficient bulk operations
- **Async Operations**: Non-blocking API calls
- **Connection Pooling**: Optimized database connections
- **Query Optimization**: Efficient Cypher queries

## 🔐 Security

- **Authentication**: JWT-based authentication
- **Authorization**: Role-based access control
- **Data Encryption**: Encrypted sensitive data
- **API Rate Limiting**: Prevent abuse
- **Input Validation**: Comprehensive validation

## 🚢 Deployment

### Docker Deployment
```bash
docker-compose -f docker-compose.prod.yml up -d
```

### Kubernetes Deployment
```bash
kubectl apply -f k8s/
```

### Cloud Deployment
- **AWS**: ECS/EKS deployment guides
- **GCP**: Cloud Run/GKE deployment
- **Azure**: Container Instances/AKS

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📞 Support

- Documentation: [Link to full docs]
- Issues: [GitHub Issues]
- Email: support@example.com

## 🙏 Acknowledgments

- Neo4j for the powerful graph database
- FastAPI for the modern Python framework
- Next.js for the React framework
- All open-source contributors