import axios from 'axios'
import toast from 'react-hot-toast'

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

const api = axios.create({
  baseURL: API_URL,
  headers: {
    'Content-Type': 'application/json',
  },
})

// Request interceptor
api.interceptors.request.use(
  (config) => {
    // Add auth token if available
    const token = localStorage.getItem('auth_token')
    if (token) {
      config.headers.Authorization = `Bearer ${token}`
    }
    return config
  },
  (error) => {
    return Promise.reject(error)
  }
)

// Response interceptor
api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      // Handle unauthorized
      toast.error('Session expired. Please login again.')
      // Redirect to login
    } else if (error.response?.status === 500) {
      toast.error('Server error. Please try again later.')
    }
    return Promise.reject(error)
  }
)

export const fetchAPI = async (endpoint: string, options?: any) => {
  try {
    const response = await api.get(`/api${endpoint}`, options)
    return response.data
  } catch (error) {
    console.error('API Error:', error)
    throw error
  }
}

export const postAPI = async (endpoint: string, data: any, options?: any) => {
  try {
    const response = await api.post(`/api${endpoint}`, data, options)
    return response.data
  } catch (error) {
    console.error('API Error:', error)
    throw error
  }
}

export const putAPI = async (endpoint: string, data: any, options?: any) => {
  try {
    const response = await api.put(`/api${endpoint}`, data, options)
    return response.data
  } catch (error) {
    console.error('API Error:', error)
    throw error
  }
}

export const deleteAPI = async (endpoint: string, options?: any) => {
  try {
    const response = await api.delete(`/api${endpoint}`, options)
    return response.data
  } catch (error) {
    console.error('API Error:', error)
    throw error
  }
}

// Portfolio API calls
export const portfolioAPI = {
  getPortfolio: (clientId: string) => fetchAPI(`/portfolio/${clientId}`),
  getHoldings: (clientId: string) => fetchAPI(`/portfolio/${clientId}/holdings`),
  updateHoldings: (clientId: string, holdings: any) => postAPI(`/portfolio/${clientId}/holdings`, holdings),
}

// Optimization API calls
export const optimizationAPI = {
  optimize: (clientId: string, model: string, params?: any) => 
    postAPI(`/optimization/optimize/${clientId}`, { model, ...params }),
  compareModels: (clientId: string) => fetchAPI(`/optimization/compare/${clientId}`),
  getConstraints: (clientId: string) => fetchAPI(`/optimization/constraints/${clientId}`),
}

// Risk API calls
export const riskAPI = {
  getMetrics: (clientId: string) => fetchAPI(`/risk/metrics/${clientId}`),
  getExposures: (clientId: string) => fetchAPI(`/risk/exposures/${clientId}`),
  getStressTest: (clientId: string, scenario: string) => 
    postAPI(`/risk/stress-test/${clientId}`, { scenario }),
}

// Market Data API calls
export const marketAPI = {
  getQuotes: (symbols: string[]) => postAPI('/market/quotes', { symbols }),
  getHistorical: (symbol: string, period: string) => 
    fetchAPI(`/market/historical/${symbol}?period=${period}`),
  getIndices: () => fetchAPI('/market/indices'),
  getTechnicals: (symbol: string) => fetchAPI(`/market/technicals/${symbol}`),
}

// Backtesting API calls
export const backtestAPI = {
  runBacktest: (params: any) => postAPI('/backtest/run', params),
  getResults: (backtestId: string) => fetchAPI(`/backtest/results/${backtestId}`),
  compareStrategies: (params: any) => postAPI('/backtest/compare', params),
}

export default api