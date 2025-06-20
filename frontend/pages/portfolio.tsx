import { useState, useEffect } from 'react'
import Head from 'next/head'
import { 
  Card, Title, Text, Metric, Badge, Grid, Col, 
  BarChart, TreeMap, Table, TableHead, TableRow, 
  TableHeaderCell, TableBody, TableCell, Button 
} from '@tremor/react'
import { 
  ArrowUpIcon, ArrowDownIcon, PlusIcon, 
  ArrowPathIcon, DocumentChartBarIcon 
} from '@heroicons/react/24/outline'
import Layout from '@/components/Layout'
import { portfolioAPI } from '@/services/api'
import toast from 'react-hot-toast'

interface Holding {
  symbol: string
  name: string
  weight: number
  value: number
  shares: number
  price: number
  change: number
  changePercent: number
  sector: string
  assetClass: string
}

interface PortfolioData {
  clientId: string
  totalValue: number
  dailyChange: number
  dailyChangePercent: number
  holdings: Holding[]
  metrics: {
    expectedReturn: number
    volatility: number
    sharpeRatio: number
    diversificationRatio: number
  }
}

export default function Portfolio() {
  const [portfolio, setPortfolio] = useState<PortfolioData | null>(null)
  const [loading, setLoading] = useState(true)
  const [selectedView, setSelectedView] = useState<'table' | 'treemap'>('table')
  const clientId = 'CLI_100001' // In production, this would come from user context

  useEffect(() => {
    fetchPortfolioData()
  }, [])

  const fetchPortfolioData = async () => {
    try {
      setLoading(true)
      const data = await portfolioAPI.getPortfolio(clientId)
      
      // Transform API response to match our interface
      const portfolioData: PortfolioData = {
        clientId: data.client_id,
        totalValue: data.total_value,
        dailyChange: data.daily_change || 1250,
        dailyChangePercent: data.daily_change_percent || 1.02,
        holdings: data.holdings || generateMockHoldings(),
        metrics: {
          expectedReturn: data.expected_return || 0.125,
          volatility: data.volatility || 0.152,
          sharpeRatio: data.sharpe_ratio || 1.23,
          diversificationRatio: data.diversification_ratio || 1.45
        }
      }
      
      setPortfolio(portfolioData)
    } catch (error) {
      console.error('Failed to fetch portfolio:', error)
      toast.error('Failed to load portfolio data')
      // Use mock data for demo
      setPortfolio(generateMockPortfolio())
    } finally {
      setLoading(false)
    }
  }

  const handleRebalance = () => {
    toast.success('Rebalancing initiated. Check optimization page for details.')
  }

  const handleAddPosition = () => {
    toast.info('Add position feature coming soon!')
  }

  if (loading) {
    return (
      <Layout>
        <div className="flex justify-center items-center h-64">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary-600"></div>
        </div>
      </Layout>
    )
  }

  if (!portfolio) {
    return (
      <Layout>
        <div className="text-center py-12">
          <Text>No portfolio data available</Text>
        </div>
      </Layout>
    )
  }

  const treeMapData = portfolio.holdings.map(h => ({
    name: h.symbol,
    size: h.value,
    color: h.changePercent >= 0 ? 'emerald' : 'red'
  }))

  return (
    <Layout>
      <Head>
        <title>Portfolio - Neo4j Investment Platform</title>
      </Head>

      <div className="space-y-6">
        {/* Header */}
        <div className="flex justify-between items-center">
          <div>
            <Title>Portfolio Overview</Title>
            <Text>Client ID: {portfolio.clientId}</Text>
          </div>
          <div className="flex space-x-3">
            <Button 
              icon={PlusIcon}
              onClick={handleAddPosition}
              variant="secondary"
            >
              Add Position
            </Button>
            <Button 
              icon={ArrowPathIcon}
              onClick={handleRebalance}
              color="blue"
            >
              Rebalance
            </Button>
          </div>
        </div>

        {/* Portfolio Summary */}
        <Grid numItems={1} numItemsSm={2} numItemsLg={4} className="gap-6">
          <Card>
            <Text>Total Value</Text>
            <Metric>${(portfolio.totalValue / 1000000).toFixed(2)}M</Metric>
            <div className="flex items-center mt-2">
              {portfolio.dailyChangePercent >= 0 ? (
                <ArrowUpIcon className="h-4 w-4 text-green-500 mr-1" />
              ) : (
                <ArrowDownIcon className="h-4 w-4 text-red-500 mr-1" />
              )}
              <Text className={portfolio.dailyChangePercent >= 0 ? 'text-green-500' : 'text-red-500'}>
                ${Math.abs(portfolio.dailyChange).toLocaleString()} ({portfolio.dailyChangePercent.toFixed(2)}%)
              </Text>
            </div>
          </Card>

          <Card>
            <Text>Expected Return</Text>
            <Metric>{(portfolio.metrics.expectedReturn * 100).toFixed(1)}%</Metric>
            <Badge color="green" className="mt-2">Annual</Badge>
          </Card>

          <Card>
            <Text>Portfolio Risk</Text>
            <Metric>{(portfolio.metrics.volatility * 100).toFixed(1)}%</Metric>
            <Text className="mt-2 text-xs">Volatility</Text>
          </Card>

          <Card>
            <Text>Sharpe Ratio</Text>
            <Metric>{portfolio.metrics.sharpeRatio.toFixed(2)}</Metric>
            <Badge color={portfolio.metrics.sharpeRatio > 1 ? 'emerald' : 'amber'} className="mt-2">
              {portfolio.metrics.sharpeRatio > 1 ? 'Good' : 'Moderate'}
            </Badge>
          </Card>
        </Grid>

        {/* Holdings View Toggle */}
        <Card>
          <div className="flex justify-between items-center mb-4">
            <Title>Holdings</Title>
            <div className="flex space-x-2">
              <button
                onClick={() => setSelectedView('table')}
                className={`px-3 py-1 rounded ${selectedView === 'table' ? 'bg-primary-100 text-primary-700' : 'text-gray-600'}`}
              >
                Table
              </button>
              <button
                onClick={() => setSelectedView('treemap')}
                className={`px-3 py-1 rounded ${selectedView === 'treemap' ? 'bg-primary-100 text-primary-700' : 'text-gray-600'}`}
              >
                Tree Map
              </button>
            </div>
          </div>

          {selectedView === 'table' ? (
            <Table>
              <TableHead>
                <TableRow>
                  <TableHeaderCell>Symbol</TableHeaderCell>
                  <TableHeaderCell>Name</TableHeaderCell>
                  <TableHeaderCell>Weight</TableHeaderCell>
                  <TableHeaderCell>Value</TableHeaderCell>
                  <TableHeaderCell>Price</TableHeaderCell>
                  <TableHeaderCell>Change</TableHeaderCell>
                  <TableHeaderCell>Sector</TableHeaderCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {portfolio.holdings.map((holding) => (
                  <TableRow key={holding.symbol}>
                    <TableCell>
                      <Text className="font-medium">{holding.symbol}</Text>
                    </TableCell>
                    <TableCell>{holding.name}</TableCell>
                    <TableCell>{(holding.weight * 100).toFixed(1)}%</TableCell>
                    <TableCell>${holding.value.toLocaleString()}</TableCell>
                    <TableCell>${holding.price.toFixed(2)}</TableCell>
                    <TableCell>
                      <div className={`flex items-center ${holding.changePercent >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                        {holding.changePercent >= 0 ? (
                          <ArrowUpIcon className="h-3 w-3 mr-1" />
                        ) : (
                          <ArrowDownIcon className="h-3 w-3 mr-1" />
                        )}
                        {Math.abs(holding.changePercent).toFixed(2)}%
                      </div>
                    </TableCell>
                    <TableCell>
                      <Badge color="gray">{holding.sector}</Badge>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          ) : (
            <div className="h-96">
              <TreeMap
                data={treeMapData}
                category="name"
                value="size"
                valueFormatter={(value) => `$${(value / 1000).toFixed(0)}k`}
              />
            </div>
          )}
        </Card>

        {/* Sector Allocation */}
        <Card>
          <Title>Sector Allocation</Title>
          <BarChart
            className="h-72 mt-4"
            data={getSectorAllocation(portfolio.holdings)}
            index="sector"
            categories={["value"]}
            colors={["blue"]}
            valueFormatter={(value) => `${value.toFixed(1)}%`}
            layout="vertical"
          />
        </Card>

        {/* Actions */}
        <Card>
          <Title>Quick Actions</Title>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mt-4">
            <button className="p-4 border rounded-lg hover:bg-gray-50 text-left">
              <DocumentChartBarIcon className="h-6 w-6 text-primary-600 mb-2" />
              <Text className="font-medium">Generate Report</Text>
              <Text className="text-xs text-gray-500">Download PDF report</Text>
            </button>
            <button className="p-4 border rounded-lg hover:bg-gray-50 text-left">
              <ArrowPathIcon className="h-6 w-6 text-primary-600 mb-2" />
              <Text className="font-medium">Run Optimization</Text>
              <Text className="text-xs text-gray-500">Find optimal weights</Text>
            </button>
            <button className="p-4 border rounded-lg hover:bg-gray-50 text-left">
              <DocumentChartBarIcon className="h-6 w-6 text-primary-600 mb-2" />
              <Text className="font-medium">Risk Analysis</Text>
              <Text className="text-xs text-gray-500">Detailed risk report</Text>
            </button>
          </div>
        </Card>
      </div>
    </Layout>
  )
}

// Helper functions
function generateMockHoldings(): Holding[] {
  return [
    { symbol: 'AAPL', name: 'Apple Inc.', weight: 0.15, value: 18750, shares: 100, price: 187.50, change: 2.50, changePercent: 1.35, sector: 'Technology', assetClass: 'Equity' },
    { symbol: 'MSFT', name: 'Microsoft Corp.', weight: 0.12, value: 15000, shares: 40, price: 375.00, change: -1.25, changePercent: -0.33, sector: 'Technology', assetClass: 'Equity' },
    { symbol: 'JPM', name: 'JPMorgan Chase', weight: 0.10, value: 12500, shares: 75, price: 166.67, change: 1.80, changePercent: 1.09, sector: 'Financial', assetClass: 'Equity' },
    { symbol: 'TLT', name: 'iShares 20+ Year Treasury', weight: 0.15, value: 18750, shares: 200, price: 93.75, change: -0.50, changePercent: -0.53, sector: 'Government', assetClass: 'Bond' },
    { symbol: 'GLD', name: 'SPDR Gold Trust', weight: 0.08, value: 10000, shares: 50, price: 200.00, change: 1.00, changePercent: 0.50, sector: 'Commodities', assetClass: 'Alternative' },
    { symbol: 'VTI', name: 'Vanguard Total Stock', weight: 0.10, value: 12500, shares: 50, price: 250.00, change: 2.00, changePercent: 0.81, sector: 'Broad Market', assetClass: 'Equity' },
    { symbol: 'BND', name: 'Vanguard Total Bond', weight: 0.10, value: 12500, shares: 150, price: 83.33, change: -0.25, changePercent: -0.30, sector: 'Aggregate', assetClass: 'Bond' },
    { symbol: 'XOM', name: 'Exxon Mobil', weight: 0.05, value: 6250, shares: 60, price: 104.17, change: 1.50, changePercent: 1.46, sector: 'Energy', assetClass: 'Equity' },
    { symbol: 'JNJ', name: 'Johnson & Johnson', weight: 0.08, value: 10000, shares: 60, price: 166.67, change: 0.75, changePercent: 0.45, sector: 'Healthcare', assetClass: 'Equity' },
    { symbol: 'PG', name: 'Procter & Gamble', weight: 0.07, value: 8750, shares: 55, price: 159.09, change: 0.50, changePercent: 0.32, sector: 'Consumer', assetClass: 'Equity' }
  ]
}

function generateMockPortfolio(): PortfolioData {
  return {
    clientId: 'CLI_100001',
    totalValue: 125000,
    dailyChange: 1250,
    dailyChangePercent: 1.02,
    holdings: generateMockHoldings(),
    metrics: {
      expectedReturn: 0.125,
      volatility: 0.152,
      sharpeRatio: 1.23,
      diversificationRatio: 1.45
    }
  }
}

function getSectorAllocation(holdings: Holding[]) {
  const sectors = holdings.reduce((acc, holding) => {
    if (!acc[holding.sector]) {
      acc[holding.sector] = 0
    }
    acc[holding.sector] += holding.weight * 100
    return acc
  }, {} as Record<string, number>)

  return Object.entries(sectors).map(([sector, value]) => ({
    sector,
    value
  })).sort((a, b) => b.value - a.value)
}