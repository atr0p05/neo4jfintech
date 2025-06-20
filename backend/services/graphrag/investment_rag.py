"""
GraphRAG Implementation for Investment Advisory
Combines Neo4j graph traversal with LLMs for intelligent, context-aware recommendations
"""

import os
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import json
import numpy as np
from datetime import datetime

# LangChain imports
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA, LLMChain
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.schema import BaseRetriever, Document
from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain.memory import ConversationBufferMemory
from langchain.agents import Tool, AgentExecutor, create_react_agent

# Neo4j specific
from langchain_neo4j import Neo4jVector, Neo4jGraph
from langchain.chains import GraphCypherQAChain

# Additional imports
from neo4j import GraphDatabase
import tiktoken
import asyncio
from concurrent.futures import ThreadPoolExecutor
import logging

logger = logging.getLogger(__name__)


@dataclass
class GraphContext:
    """Container for graph-based context"""
    client_profile: Dict
    portfolio_data: Dict
    risk_metrics: Dict
    market_context: Dict
    relationships: List[Dict]
    recommendations: List[Dict]


class InvestmentGraphRAG:
    """
    Advanced GraphRAG system for investment advisory
    Combines graph traversal, vector search, and LLM reasoning
    """
    
    def __init__(self, neo4j_driver, openai_api_key: str, neo4j_uri: str, 
                 neo4j_user: str, neo4j_password: str):
        self.driver = neo4j_driver
        self.neo4j_uri = neo4j_uri
        self.neo4j_user = neo4j_user
        self.neo4j_password = neo4j_password
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            model="gpt-4-turbo-preview",
            temperature=0.7,
            openai_api_key=openai_api_key,
            streaming=True,
            callbacks=[StreamingStdOutCallbackHandler()]
        )
        
        # Initialize embeddings
        self.embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        
        # Initialize Neo4j graph for Cypher queries
        self.graph = Neo4jGraph(
            url=neo4j_uri,
            username=neo4j_user,
            password=neo4j_password
        )
        
        # Initialize vector store
        self.vector_store = self._initialize_vector_store()
        
        # Initialize conversation memory
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        # Create specialized chains
        self.qa_chain = self._create_qa_chain()
        self.cypher_chain = self._create_cypher_chain()
        self.agent = self._create_investment_agent()
    
    def _initialize_vector_store(self) -> Neo4jVector:
        """Initialize Neo4j vector store for semantic search"""
        return Neo4jVector.from_existing_graph(
            embedding=self.embeddings,
            url=self.neo4j_uri,
            username=self.neo4j_user,
            password=self.neo4j_password,
            index_name="investment_entities",
            node_label="Asset",
            text_node_properties=["name", "sector", "description"],
            embedding_node_property="embedding"
        )
    
    def _create_qa_chain(self) -> RetrievalQA:
        """Create Q&A chain with custom prompt"""
        prompt_template = """You are an expert investment advisor with deep knowledge of portfolio management, risk analysis, and market dynamics.

Context from Knowledge Graph:
{context}

Client Question: {question}

Provide a comprehensive, personalized investment recommendation that:
1. Directly addresses the client's question
2. Considers their risk profile and current portfolio
3. Cites specific data from the knowledge graph
4. Explains the reasoning behind recommendations
5. Highlights any risks or considerations

Investment Advice:"""

        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        return RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vector_store.as_retriever(search_kwargs={"k": 5}),
            chain_type_kwargs={"prompt": PROMPT}
        )
    
    def _create_cypher_chain(self) -> GraphCypherQAChain:
        """Create Cypher generation chain for complex graph queries"""
        cypher_prompt = """You are an expert at writing Neo4j Cypher queries for investment analysis.
        
The graph schema includes:
- Client nodes with properties: clientId, name, riskTolerance, netWorth
- Portfolio nodes with properties: portfolioId, totalValue, expectedReturn, volatility
- Asset nodes with properties: symbol, currentPrice, expectedReturn, volatility, sector
- Relationships: OWNS_PORTFOLIO, HOLDS, HAS_RISK_PROFILE, CORRELATED_WITH

Given the question below, write a Cypher query to extract relevant information.
Only return the Cypher query, nothing else.

Question: {question}
Cypher Query:"""

        return GraphCypherQAChain.from_llm(
            llm=self.llm,
            graph=self.graph,
            verbose=True,
            cypher_prompt=cypher_prompt,
            return_intermediate_steps=True
        )
    
    def _create_investment_agent(self) -> AgentExecutor:
        """Create an agent with multiple tools for investment analysis"""
        tools = [
            Tool(
                name="Portfolio_Analysis",
                func=self._analyze_portfolio,
                description="Analyze a client's current portfolio composition, risk metrics, and performance"
            ),
            Tool(
                name="Market_Research", 
                func=self._research_market,
                description="Research market conditions, asset performance, and investment opportunities"
            ),
            Tool(
                name="Risk_Assessment",
                func=self._assess_risk,
                description="Assess portfolio risk, concentration, and compliance with client risk tolerance"
            ),
            Tool(
                name="Optimization_Recommendations",
                func=self._get_optimization_recommendations,
                description="Get portfolio optimization recommendations based on various models"
            ),
            Tool(
                name="Graph_Query",
                func=self._execute_graph_query,
                description="Execute complex graph queries to find patterns and relationships"
            )
        ]
        
        agent_prompt = """You are an expert investment advisor assistant with access to comprehensive portfolio and market data.

You have access to the following tools:
{tools}

When answering questions:
1. Use multiple tools to gather comprehensive information
2. Analyze the data critically
3. Provide specific, actionable recommendations
4. Always consider the client's risk profile
5. Cite specific data points

Current conversation:
{chat_history}

Question: {input}
{agent_scratchpad}"""

        prompt = PromptTemplate(
            template=agent_prompt,
            input_variables=["tools", "chat_history", "input", "agent_scratchpad"]
        )
        
        agent = create_react_agent(
            llm=self.llm,
            tools=tools,
            prompt=prompt
        )
        
        return AgentExecutor(
            agent=agent,
            tools=tools,
            memory=self.memory,
            verbose=True,
            max_iterations=5
        )
    
    def get_investment_advice(self, client_id: str, question: str) -> Dict[str, Any]:
        """
        Main entry point for getting investment advice
        Combines graph context with LLM reasoning
        """
        # Get comprehensive graph context
        context = self._get_graph_context(client_id)
        
        # Enhance question with context
        enhanced_question = self._enhance_question_with_context(question, context)
        
        # Get response from agent
        response = self.agent.run(enhanced_question)
        
        # Post-process response with additional insights
        insights = self._generate_insights(context, response)
        
        return {
            'question': question,
            'response': response,
            'insights': insights,
            'context': self._summarize_context(context),
            'timestamp': datetime.now().isoformat()
        }
    
    def _get_graph_context(self, client_id: str) -> GraphContext:
        """Gather comprehensive context from the graph"""
        with self.driver.session() as session:
            # Get client profile
            client_result = session.run("""
                MATCH (c:Client {clientId: $clientId})-[:HAS_RISK_PROFILE]->(rp:RiskProfile)
                RETURN c, rp
            """, clientId=client_id).single()
            
            # Get portfolio data
            portfolio_result = session.run("""
                MATCH (c:Client {clientId: $clientId})-[:OWNS_PORTFOLIO]->(p:Portfolio)
                OPTIONAL MATCH (p)-[h:HOLDS]->(a:Asset)
                WITH p, collect({
                    asset: a,
                    holding: h
                }) as holdings
                RETURN p, holdings
            """, clientId=client_id).data()
            
            # Get risk metrics
            risk_result = session.run("""
                MATCH (c:Client {clientId: $clientId})-[:OWNS_PORTFOLIO]->(p:Portfolio)
                RETURN p.volatility as portfolioVol,
                       p.expectedReturn as portfolioReturn,
                       p.sharpeRatio as sharpeRatio,
                       p.var95 as valueAtRisk
            """, clientId=client_id).single()
            
            # Get market context
            market_result = session.run("""
                MATCH (idx:MarketIndex)
                RETURN idx.symbol as symbol,
                       idx.value as value,
                       idx.change as change
                ORDER BY idx.symbol
                LIMIT 5
            """).data()
            
            # Get relevant relationships
            relationships = session.run("""
                MATCH (c:Client {clientId: $clientId})-[:OWNS_PORTFOLIO]->(p:Portfolio)
                MATCH (p)-[:HOLDS]->(a:Asset)
                OPTIONAL MATCH (a)-[r:CORRELATED_WITH|EXPOSED_TO|HAS_FACTOR_EXPOSURE]-(related)
                RETURN type(r) as relationship,
                       a.symbol as asset,
                       related.name as relatedEntity,
                       r as properties
                LIMIT 20
            """, clientId=client_id).data()
            
            # Get recommendations
            recommendations = session.run("""
                MATCH (c:Client {clientId: $clientId})-[:HAS_OPTIMIZATION]->(opt:Optimization)
                WHERE opt.timestamp > datetime() - duration('P7D')
                MATCH (opt)-[rec:RECOMMENDS_WEIGHT]->(a:Asset)
                RETURN a.symbol as symbol,
                       a.name as name,
                       rec.weight as weight,
                       opt.model as model
                ORDER BY opt.timestamp DESC, rec.weight DESC
                LIMIT 10
            """, clientId=client_id).data()
            
            return GraphContext(
                client_profile=dict(client_result['c']) if client_result else {},
                portfolio_data=portfolio_result,
                risk_metrics=dict(risk_result) if risk_result else {},
                market_context=market_result,
                relationships=relationships,
                recommendations=recommendations
            )
    
    def _enhance_question_with_context(self, question: str, context: GraphContext) -> str:
        """Add relevant context to the user's question"""
        risk_tolerance = context.client_profile.get('riskTolerance', 'Unknown')
        portfolio_value = sum(p['p']['totalValue'] for p in context.portfolio_data if p['p'])
        
        enhanced = f"""
Client Profile:
- Risk Tolerance: {risk_tolerance}
- Portfolio Value: ${portfolio_value:,.0f}
- Current Volatility: {context.risk_metrics.get('portfolioVol', 0)*100:.1f}%
- Expected Return: {context.risk_metrics.get('portfolioReturn', 0)*100:.1f}%

User Question: {question}
"""
        return enhanced
    
    def _analyze_portfolio(self, query: str) -> str:
        """Tool: Analyze portfolio composition and metrics"""
        # Extract client ID from query or use default
        client_id = self._extract_client_id(query) or "CLI_100001"
        
        with self.driver.session() as session:
            analysis = session.run("""
                MATCH (c:Client {clientId: $clientId})-[:OWNS_PORTFOLIO]->(p:Portfolio)
                MATCH (p)-[h:HOLDS]->(a:Asset)
                WITH p, a.assetClass as class, 
                     sum(h.weight) as classWeight,
                     sum(h.weight * a.expectedReturn) as weightedReturn,
                     collect({symbol: a.symbol, weight: h.weight}) as assets
                RETURN class, 
                       classWeight,
                       weightedReturn,
                       size(assets) as numAssets,
                       assets[0..3] as topAssets
                ORDER BY classWeight DESC
            """, clientId=client_id).data()
            
            result = "Portfolio Analysis:\n"
            for row in analysis:
                result += f"\n{row['class']}: {row['classWeight']*100:.1f}% "
                result += f"({row['numAssets']} assets, "
                result += f"weighted return: {row['weightedReturn']*100:.1f}%)\n"
                result += f"  Top holdings: {', '.join(a['symbol'] for a in row['topAssets'])}"
            
            return result
    
    def _research_market(self, query: str) -> str:
        """Tool: Research market conditions and opportunities"""
        with self.driver.session() as session:
            # Find top performing assets
            top_assets = session.run("""
                MATCH (a:Asset)
                WHERE a.sharpeRatio > 1.0
                AND a.expectedReturn > 0.08
                RETURN a.symbol as symbol,
                       a.name as name,
                       a.sector as sector,
                       a.expectedReturn as return,
                       a.sharpeRatio as sharpe
                ORDER BY a.sharpeRatio DESC
                LIMIT 10
            """).data()
            
            # Get sector performance
            sectors = session.run("""
                MATCH (a:Asset)
                WITH a.sector as sector,
                     avg(a.expectedReturn) as avgReturn,
                     avg(a.volatility) as avgVol,
                     count(a) as numAssets
                WHERE numAssets > 3
                RETURN sector, avgReturn, avgVol, 
                       avgReturn/avgVol as riskAdjReturn
                ORDER BY riskAdjReturn DESC
            """).data()
            
            result = "Market Research Results:\n\n"
            result += "Top Performing Assets:\n"
            for asset in top_assets[:5]:
                result += f"- {asset['symbol']} ({asset['sector']}): "
                result += f"Return {asset['return']*100:.1f}%, Sharpe {asset['sharpe']:.2f}\n"
            
            result += "\nSector Analysis:\n"
            for sector in sectors[:5]:
                result += f"- {sector['sector']}: "
                result += f"Avg Return {sector['avgReturn']*100:.1f}%, "
                result += f"Risk-Adj Return {sector['riskAdjReturn']:.2f}\n"
            
            return result
    
    def _assess_risk(self, query: str) -> str:
        """Tool: Assess portfolio risk and concentrations"""
        client_id = self._extract_client_id(query) or "CLI_100001"
        
        with self.driver.session() as session:
            # Get risk metrics
            risk_data = session.run("""
                MATCH (c:Client {clientId: $clientId})-[:HAS_RISK_PROFILE]->(rp:RiskProfile)
                MATCH (c)-[:OWNS_PORTFOLIO]->(p:Portfolio)
                RETURN p.volatility as currentVol,
                       rp.volatilityTolerance as maxVol,
                       p.var95 as var95,
                       p.maxDrawdown as maxDD,
                       p.beta as beta
            """, clientId=client_id).single()
            
            # Check concentrations
            concentrations = session.run("""
                MATCH (c:Client {clientId: $clientId})-[:OWNS_PORTFOLIO]->(p:Portfolio)
                MATCH (p)-[h:HOLDS]->(a:Asset)
                WITH a.sector as sector, sum(h.weight) as sectorWeight
                WHERE sectorWeight > 0.25
                RETURN sector, sectorWeight
                ORDER BY sectorWeight DESC
            """, clientId=client_id).data()
            
            # Check correlations
            correlations = session.run("""
                MATCH (c:Client {clientId: $clientId})-[:OWNS_PORTFOLIO]->(p:Portfolio)
                MATCH (p)-[h1:HOLDS]->(a1:Asset)
                MATCH (p)-[h2:HOLDS]->(a2:Asset)
                MATCH (a1)-[cor:CORRELATED_WITH]-(a2)
                WHERE cor.correlation > 0.7 AND a1.symbol < a2.symbol
                RETURN a1.symbol as asset1, a2.symbol as asset2, 
                       cor.correlation as correlation,
                       h1.weight * h2.weight as weightProduct
                ORDER BY weightProduct DESC
                LIMIT 5
            """, clientId=client_id).data()
            
            result = "Risk Assessment:\n\n"
            
            if risk_data:
                current_vol = risk_data['currentVol']
                max_vol = risk_data['maxVol']
                result += f"Volatility: {current_vol*100:.1f}% "
                result += f"(Limit: {max_vol*100:.1f}%) - "
                result += "WITHIN LIMITS\n" if current_vol <= max_vol else "EXCEEDS LIMITS!\n"
                result += f"Value at Risk (95%): {risk_data['var95']*100:.1f}%\n"
                result += f"Max Drawdown: {risk_data['maxDD']*100:.1f}%\n"
                result += f"Portfolio Beta: {risk_data['beta']:.2f}\n"
            
            if concentrations:
                result += "\nConcentration Risks:\n"
                for conc in concentrations:
                    result += f"- {conc['sector']}: {conc['sectorWeight']*100:.1f}% (HIGH)\n"
            
            if correlations:
                result += "\nHigh Correlations:\n"
                for cor in correlations:
                    result += f"- {cor['asset1']} & {cor['asset2']}: "
                    result += f"{cor['correlation']:.2f} correlation\n"
            
            return result
    
    def _get_optimization_recommendations(self, query: str) -> str:
        """Tool: Get portfolio optimization recommendations"""
        client_id = self._extract_client_id(query) or "CLI_100001"
        
        with self.driver.session() as session:
            # Get latest optimization
            opt_results = session.run("""
                MATCH (c:Client {clientId: $clientId})-[:HAS_OPTIMIZATION]->(opt:Optimization)
                WITH opt ORDER BY opt.timestamp DESC LIMIT 1
                MATCH (opt)-[rec:RECOMMENDS_WEIGHT]->(a:Asset)
                WHERE rec.weight > 0.02
                RETURN opt.model as model,
                       opt.expectedReturn as expReturn,
                       opt.sharpeRatio as sharpe,
                       collect({
                           symbol: a.symbol,
                           name: a.name,
                           weight: rec.weight,
                           value: rec.value
                       }) as recommendations
            """, clientId=client_id).single()
            
            if not opt_results:
                return "No optimization results found. Run portfolio optimization first."
            
            result = f"Portfolio Optimization ({opt_results['model']}):\n"
            result += f"Expected Return: {opt_results['expReturn']*100:.1f}%\n"
            result += f"Sharpe Ratio: {opt_results['sharpe']:.2f}\n\n"
            result += "Recommended Allocation:\n"
            
            for rec in sorted(opt_results['recommendations'], 
                            key=lambda x: x['weight'], reverse=True):
                result += f"- {rec['symbol']}: {rec['weight']*100:.1f}% "
                result += f"(${rec['value']:,.0f})\n"
            
            return result
    
    def _execute_graph_query(self, query: str) -> str:
        """Tool: Execute custom graph queries"""
        try:
            # Use the Cypher chain to generate and execute query
            result = self.cypher_chain.run(query)
            return f"Query Result:\n{result}"
        except Exception as e:
            return f"Error executing query: {str(e)}"
    
    def _extract_client_id(self, query: str) -> Optional[str]:
        """Extract client ID from query if mentioned"""
        # Simple extraction - in production use NER
        import re
        match = re.search(r'CLI_\d+', query)
        return match.group(0) if match else None
    
    def _generate_insights(self, context: GraphContext, response: str) -> List[str]:
        """Generate additional insights based on context and response"""
        insights = []
        
        # Risk insight
        if context.risk_metrics.get('portfolioVol', 0) > context.client_profile.get('volatilityTolerance', 1):
            insights.append("⚠️ Portfolio volatility exceeds your risk tolerance. Consider rebalancing.")
        
        # Concentration insight
        if context.portfolio_data:
            holdings = []
            for p in context.portfolio_data:
                if p['holdings']:
                    holdings.extend(p['holdings'])
            
            if len(holdings) < 5:
                insights.append("📊 Low diversification detected. Consider adding more assets.")
        
        # Performance insight
        market_return = 0.08  # Assume 8% market return
        portfolio_return = context.risk_metrics.get('portfolioReturn', 0)
        if portfolio_return < market_return * 0.8:
            insights.append("📉 Portfolio underperforming market. Review asset selection.")
        
        # Recommendation insight
        if context.recommendations:
            latest_model = context.recommendations[0]['model']
            insights.append(f"💡 Latest optimization used {latest_model} model")
        
        return insights
    
    def _summarize_context(self, context: GraphContext) -> Dict[str, Any]:
        """Create a summary of the context for the response"""
        return {
            'risk_profile': context.client_profile.get('riskTolerance'),
            'portfolio_value': sum(p['p']['totalValue'] for p in context.portfolio_data if p['p']),
            'num_holdings': sum(len(p['holdings']) for p in context.portfolio_data),
            'current_return': context.risk_metrics.get('portfolioReturn', 0),
            'current_risk': context.risk_metrics.get('portfolioVol', 0),
            'recommendations_available': len(context.recommendations) > 0
        }
    
    async def stream_advice(self, client_id: str, question: str):
        """Stream investment advice responses"""
        context = self._get_graph_context(client_id)
        enhanced_question = self._enhance_question_with_context(question, context)
        
        # Stream response
        async for chunk in self.agent.astream(enhanced_question):
            yield chunk
    
    def create_investment_report(self, client_id: str) -> str:
        """Generate a comprehensive investment report"""
        context = self._get_graph_context(client_id)
        
        report_prompt = """Create a comprehensive investment report for the client based on the following information:

Client Profile:
{client_profile}

Portfolio Summary:
{portfolio_summary}

Risk Metrics:
{risk_metrics}

Recent Recommendations:
{recommendations}

Market Context:
{market_context}

Generate a professional report with:
1. Executive Summary
2. Portfolio Analysis
3. Risk Assessment
4. Recommendations
5. Market Outlook
6. Action Items

Make it detailed but accessible to a non-technical investor."""
        
        # Prepare data for prompt
        portfolio_summary = self._analyze_portfolio(f"analyze portfolio for {client_id}")
        
        # Generate report
        response = self.llm.predict(
            report_prompt.format(
                client_profile=json.dumps(context.client_profile, indent=2),
                portfolio_summary=portfolio_summary,
                risk_metrics=json.dumps(context.risk_metrics, indent=2),
                recommendations=json.dumps(context.recommendations[:5], indent=2),
                market_context=json.dumps(context.market_context, indent=2)
            )
        )
        
        return response


# Example usage
if __name__ == "__main__":
    # Initialize
    neo4j_uri = "neo4j+s://1c34ddb0.databases.neo4j.io"
    neo4j_user = "neo4j"
    neo4j_password = "Sw-DAlLfV1_Rw4lFlXAxl0DoN4lCauWrp29IZnfq_zM"
    openai_key = os.getenv("OPENAI_API_KEY")  # Set your OpenAI key
    
    driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
    
    # Create GraphRAG system
    graph_rag = InvestmentGraphRAG(
        driver, openai_key, neo4j_uri, neo4j_user, neo4j_password
    )
    
    # Example queries
    client_id = "CLI_100001"
    
    # Get investment advice
    print("Getting investment advice...")
    advice = graph_rag.get_investment_advice(
        client_id,
        "Should I increase my equity allocation given the current market conditions?"
    )
    
    print("\nResponse:", advice['response'])
    print("\nInsights:", advice['insights'])
    
    # Generate report
    print("\n\nGenerating investment report...")
    report = graph_rag.create_investment_report(client_id)
    print(report)
    
    driver.close()