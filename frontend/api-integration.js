/**
 * API Integration for Neo4j Investment Platform
 * Connects the HTML dashboard with the FastAPI backend
 */

class InvestmentPlatformAPI {
    constructor(baseURL = 'http://localhost:8000') {
        this.baseURL = baseURL;
        this.ws = null;
        this.clientId = 'CLI_100001'; // Default client
    }

    // Generic API call method
    async apiCall(endpoint, method = 'GET', data = null) {
        const config = {
            method,
            headers: {
                'Content-Type': 'application/json',
            }
        };

        if (data) {
            config.body = JSON.stringify(data);
        }

        try {
            const response = await fetch(`${this.baseURL}${endpoint}`, config);
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            return await response.json();
        } catch (error) {
            console.error('API call failed:', error);
            throw error;
        }
    }

    // Dashboard Data
    async getDashboardData() {
        try {
            const [portfolioData, systemStats, performance] = await Promise.all([
                this.getPortfolio(this.clientId),
                this.getSystemStats(),
                this.getPortfolioPerformance(this.clientId)
            ]);

            return {
                portfolio: portfolioData,
                stats: systemStats,
                performance: performance
            };
        } catch (error) {
            console.error('Failed to fetch dashboard data:', error);
            return null;
        }
    }

    // Portfolio Management
    async getPortfolio(clientId) {
        return this.apiCall(`/api/portfolios/${clientId}`);
    }

    async getPortfolioPerformance(clientId, period = '1Y') {
        return this.apiCall(`/api/portfolios/${clientId}/performance?period=${period}`);
    }

    async optimizePortfolio(clientId, model = 'markowitz', constraints = null, views = null) {
        const data = {
            clientId,
            optimizationModel: model,
            constraints,
            views
        };
        return this.apiCall('/api/portfolios/optimize', 'POST', data);
    }

    // Market Data
    async getMarketQuotes(symbols) {
        const data = { symbols, dataType: 'quotes' };
        return this.apiCall('/api/market/quotes', 'POST', data);
    }

    async getMarketIndices() {
        return this.apiCall('/api/market/indices');
    }

    async getHistoricalData(symbol, period = '1y', interval = '1d') {
        return this.apiCall(`/api/market/historical/${symbol}?period=${period}&interval=${interval}`);
    }

    async getTechnicalIndicators(symbol) {
        return this.apiCall(`/api/market/technical/${symbol}`);
    }

    // AI Advisor
    async askAIAdvisor(clientId, question) {
        const data = { clientId, question };
        return this.apiCall('/api/advisor/ask', 'POST', data);
    }

    async generateInvestmentReport(clientId) {
        return this.apiCall(`/api/advisor/report/${clientId}`);
    }

    // Backtesting
    async runBacktest(symbols, strategy, startDate, endDate, initialCapital = 100000) {
        const data = {
            symbols,
            strategy,
            startDate,
            endDate,
            initialCapital
        };
        return this.apiCall('/api/backtest', 'POST', data);
    }

    async compareStrategies(symbols, strategies, startDate, endDate, initialCapital = 100000) {
        const data = {
            symbols,
            strategies,
            start_date: startDate,
            end_date: endDate,
            initial_capital: initialCapital
        };
        return this.apiCall('/api/backtest/compare', 'POST', data);
    }

    // Admin
    async getSystemStats() {
        return this.apiCall('/api/admin/system-stats');
    }

    async syncMarketData() {
        return this.apiCall('/api/admin/sync-market-data', 'POST');
    }

    // WebSocket for real-time updates
    connectWebSocket(clientId) {
        if (this.ws) {
            this.ws.close();
        }

        const wsURL = `ws://localhost:8000/ws/${clientId}`;
        this.ws = new WebSocket(wsURL);

        this.ws.onopen = () => {
            console.log('WebSocket connected');
        };

        this.ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            this.handleWebSocketMessage(data);
        };

        this.ws.onclose = () => {
            console.log('WebSocket disconnected');
            // Attempt to reconnect after 5 seconds
            setTimeout(() => this.connectWebSocket(clientId), 5000);
        };

        this.ws.onerror = (error) => {
            console.error('WebSocket error:', error);
        };
    }

    handleWebSocketMessage(data) {
        switch (data.type) {
            case 'portfolio_update':
                this.updatePortfolioValue(data.data);
                break;
            case 'optimization_complete':
                this.updateOptimizationResults(data.data);
                break;
            case 'market_data_update':
                this.updateMarketData(data.data);
                break;
            default:
                console.log('Unknown WebSocket message:', data);
        }
    }

    updatePortfolioValue(data) {
        const valueElement = document.querySelector('[data-portfolio-value]');
        if (valueElement) {
            valueElement.textContent = `$${data.totalValue.toLocaleString()}`;
        }
    }

    updateOptimizationResults(data) {
        // Trigger Alpine.js update for optimization results
        if (window.Alpine) {
            const store = Alpine.store('optimization');
            if (store) {
                store.results = data;
            }
        }
    }

    updateMarketData(data) {
        // Update market data displays
        console.log('Market data updated:', data);
    }
}

// Dashboard integration functions
class DashboardIntegration {
    constructor() {
        this.api = new InvestmentPlatformAPI();
        this.charts = {};
        this.init();
    }

    async init() {
        await this.loadDashboardData();
        this.setupRealTimeUpdates();
        this.setupEventListeners();
    }

    async loadDashboardData() {
        try {
            const data = await this.api.getDashboardData();
            if (data) {
                this.updateDashboard(data);
            }
        } catch (error) {
            console.error('Failed to load dashboard data:', error);
            this.showError('Failed to load dashboard data');
        }
    }

    updateDashboard(data) {
        // Update KPI cards
        this.updateKPICards(data.portfolio);
        
        // Update charts
        this.updateCharts(data.performance);
        
        // Update system status
        this.updateSystemStatus(data.stats);
    }

    updateKPICards(portfolio) {
        const elements = {
            portfolioValue: document.querySelector('[data-portfolio-value]'),
            annualReturn: document.querySelector('[data-annual-return]'),
            riskScore: document.querySelector('[data-risk-score]'),
            sharpeRatio: document.querySelector('[data-sharpe-ratio]')
        };

        if (elements.portfolioValue && portfolio.totalValue) {
            elements.portfolioValue.textContent = `$${portfolio.totalValue.toLocaleString()}`;
        }
        if (elements.annualReturn && portfolio.expectedReturn) {
            elements.annualReturn.textContent = `${(portfolio.expectedReturn * 100).toFixed(1)}%`;
        }
        if (elements.sharpeRatio && portfolio.sharpeRatio) {
            elements.sharpeRatio.textContent = portfolio.sharpeRatio.toFixed(2);
        }
    }

    updateCharts(performanceData) {
        if (performanceData && this.charts.performance) {
            this.charts.performance.data.datasets[0].data = performanceData.values;
            this.charts.performance.data.labels = performanceData.dates.map(d => 
                new Date(d).toLocaleDateString('en-US', { month: 'short' })
            );
            this.charts.performance.update();
        }
    }

    updateSystemStatus(stats) {
        const statusElement = document.querySelector('[data-system-status]');
        if (statusElement && stats) {
            statusElement.innerHTML = `
                <div class="text-sm text-gray-600">
                    Clients: ${stats.clients} | Portfolios: ${stats.portfolios} | Assets: ${stats.assets}
                </div>
            `;
        }
    }

    setupRealTimeUpdates() {
        this.api.connectWebSocket(this.api.clientId);
    }

    setupEventListeners() {
        // Optimization button
        const optimizeBtn = document.querySelector('[data-optimize-btn]');
        if (optimizeBtn) {
            optimizeBtn.addEventListener('click', this.handleOptimization.bind(this));
        }

        // AI Chat
        const chatInputs = document.querySelectorAll('[data-chat-input]');
        chatInputs.forEach(input => {
            input.addEventListener('keypress', (e) => {
                if (e.key === 'Enter') {
                    this.handleChatMessage(e.target.value);
                    e.target.value = '';
                }
            });
        });

        // Market data refresh
        const refreshBtn = document.querySelector('[data-refresh-market]');
        if (refreshBtn) {
            refreshBtn.addEventListener('click', this.refreshMarketData.bind(this));
        }
    }

    async handleOptimization() {
        try {
            const model = document.querySelector('[name="optimization-model"]')?.value || 'markowitz';
            const riskTolerance = document.querySelector('[name="risk-tolerance"]')?.value || 5;
            const minPosition = document.querySelector('[name="min-position"]')?.value || 2;
            const maxPosition = document.querySelector('[name="max-position"]')?.value || 25;

            const constraints = {
                min_weight: parseFloat(minPosition) / 100,
                max_weight: parseFloat(maxPosition) / 100
            };

            const result = await this.api.optimizePortfolio(
                this.api.clientId,
                model,
                constraints
            );

            this.displayOptimizationResults(result);
        } catch (error) {
            console.error('Optimization failed:', error);
            this.showError('Portfolio optimization failed');
        }
    }

    displayOptimizationResults(result) {
        // Update Alpine.js data for optimization results
        if (window.Alpine) {
            const component = document.querySelector('[x-data]');
            if (component && component._x_dataStack) {
                component._x_dataStack[0].optimizationResults = result;
            }
        }
    }

    async handleChatMessage(message) {
        if (!message.trim()) return;

        try {
            const response = await this.api.askAIAdvisor(this.api.clientId, message);
            this.displayChatResponse(response);
        } catch (error) {
            console.error('AI chat failed:', error);
            this.showError('AI assistant is currently unavailable');
        }
    }

    displayChatResponse(response) {
        const chatContainer = document.querySelector('[data-chat-messages]');
        if (chatContainer) {
            const messageHTML = `
                <div class="flex items-start space-x-3 chat-message">
                    <div class="w-8 h-8 bg-purple-100 rounded-full flex items-center justify-center flex-shrink-0">
                        <i class="fas fa-robot text-purple-600 text-sm"></i>
                    </div>
                    <div class="bg-gray-100 rounded-lg p-4 max-w-md">
                        <p class="text-gray-800">${response.response}</p>
                        ${response.insights ? `
                            <ul class="mt-2 space-y-1 text-sm text-gray-600">
                                ${response.insights.map(insight => `<li>• ${insight}</li>`).join('')}
                            </ul>
                        ` : ''}
                    </div>
                </div>
            `;
            chatContainer.insertAdjacentHTML('beforeend', messageHTML);
            chatContainer.scrollTop = chatContainer.scrollHeight;
        }
    }

    async refreshMarketData() {
        try {
            await this.api.syncMarketData();
            this.showSuccess('Market data updated successfully');
        } catch (error) {
            console.error('Market data refresh failed:', error);
            this.showError('Failed to refresh market data');
        }
    }

    showError(message) {
        this.showNotification(message, 'error');
    }

    showSuccess(message) {
        this.showNotification(message, 'success');
    }

    showNotification(message, type = 'info') {
        const notification = document.createElement('div');
        notification.className = `fixed top-4 right-4 p-4 rounded-lg shadow-lg z-50 ${
            type === 'error' ? 'bg-red-500 text-white' :
            type === 'success' ? 'bg-green-500 text-white' :
            'bg-blue-500 text-white'
        }`;
        notification.textContent = message;
        
        document.body.appendChild(notification);
        
        setTimeout(() => {
            notification.remove();
        }, 5000);
    }
}

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.dashboardIntegration = new DashboardIntegration();
});

// Export for use in modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { InvestmentPlatformAPI, DashboardIntegration };
}