import { useState, useEffect } from 'react'
import Head from 'next/head'
import { 
  Card, Title, Text, Metric, ProgressBar, Badge, Grid, Col, 
  AreaChart, BarChart, DonutChart, LineChart, Tab, TabGroup, TabList, TabPanel, TabPanels 
} from '@tremor/react'
import { 
  CurrencyDollarIcon, TrendingUpIcon, ShieldCheckIcon, 
  UserGroupIcon, ChartPieIcon, BellIcon 
} from '@heroicons/react/24/outline'
import Layout from '@/components/Layout'
import { fetchAPI } from '@/services/api'
import toast from 'react-hot-toast'

interface DashboardData {
  totalClients: number
  totalAUM: number
  avgReturn: number
  avgRisk: number
  topPerformers: Array<{name: string, return: number}>
  assetAllocation: Array<{name: string, value: number}>
  performanceHistory: Array<{date: string, value: number}>
  riskMetrics: {
    portfolioVolatility: number
    sharpeRatio: number
    maxDrawdown: number
    var95: number
  }
}

export default function Dashboard() {
  const [data, setData] = useState<DashboardData | null>(null)
  const [loading, setLoading] = useState(true)
  const [selectedClientId, setSelectedClientId] = useState('CLI_100001')

  useEffect(() => {
    fetchDashboardData()
  }, [selectedClientId])

  const fetchDashboardData = async () => {
    try {
      setLoading(true)
      // Fetch portfolio data
      const portfolio = await fetchAPI(`/portfolio/${selectedClientId}`)
      
      // Mock additional dashboard data (in production, these would be separate API calls)
      const mockData: DashboardData = {
        totalClients: 100,
        totalAUM: 125000000,
        avgReturn: 12.5,
        avgRisk: 15.2,
        topPerformers: [
          { name: 'AAPL', return: 25.3 },
          { name: 'MSFT', return: 22.1 },
          { name: 'JPM', return: 18.7 },
          { name: 'TLT', return: 15.2 },
          { name: 'GLD', return: 12.8 }
        ],
        assetAllocation: [
          { name: 'Equity', value: 65 },
          { name: 'Bonds', value: 25 },
          { name: 'Alternatives', value: 10 }
        ],
        performanceHistory: generateMockPerformance(),
        riskMetrics: {
          portfolioVolatility: portfolio.volatility || 0.152,
          sharpeRatio: portfolio.sharpe_ratio || 1.23,
          maxDrawdown: -0.087,
          var95: -0.023
        }
      }

      setData(mockData)
    } catch (error) {
      console.error('Failed to fetch dashboard data:', error)
      toast.error('Failed to load dashboard data')
    } finally {
      setLoading(false)
    }
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

  if (!data) {
    return (
      <Layout>
        <div className="text-center py-12">
          <Text>No data available</Text>
        </div>
      </Layout>
    )
  }

  return (
    <Layout>
      <Head>
        <title>Dashboard - Neo4j Investment Platform</title>
      </Head>

      <div className="space-y-6">
        {/* Header */}
        <div className="flex justify-between items-center">
          <div>
            <Title>Investment Dashboard</Title>
            <Text>Monitor portfolio performance and risk metrics</Text>
          </div>
          <div className="flex items-center space-x-4">
            <Badge color="green">Live</Badge>
            <button className="p-2 hover:bg-gray-100 rounded-lg">
              <BellIcon className="h-5 w-5 text-gray-600" />
            </button>
          </div>
        </div>

        {/* Key Metrics */}
        <Grid numItems={1} numItemsSm={2} numItemsLg={4} className="gap-6">
          <Card decoration="top" decorationColor="blue">
            <div className="flex items-center justify-between">
              <div>
                <Text>Total AUM</Text>
                <Metric>${(data.totalAUM / 1000000).toFixed(1)}M</Metric>
              </div>
              <CurrencyDollarIcon className="h-8 w-8 text-blue-600" />
            </div>
            <ProgressBar value={75} className="mt-2" />
          </Card>

          <Card decoration="top" decorationColor="green">
            <div className="flex items-center justify-between">
              <div>
                <Text>Avg Return</Text>
                <Metric>{data.avgReturn}%</Metric>
              </div>
              <TrendingUpIcon className="h-8 w-8 text-green-600" />
            </div>
            <Badge color="green" className="mt-2">+2.3% vs Market</Badge>
          </Card>

          <Card decoration="top" decorationColor="amber">
            <div className="flex items-center justify-between">
              <div>
                <Text>Risk Level</Text>
                <Metric>{data.avgRisk}%</Metric>
              </div>
              <ShieldCheckIcon className="h-8 w-8 text-amber-600" />
            </div>
            <Text className="mt-2 text-xs">Portfolio Volatility</Text>
          </Card>

          <Card decoration="top" decorationColor="indigo">
            <div className="flex items-center justify-between">
              <div>
                <Text>Total Clients</Text>
                <Metric>{data.totalClients}</Metric>
              </div>
              <UserGroupIcon className="h-8 w-8 text-indigo-600" />
            </div>
            <Text className="mt-2 text-xs">Active portfolios</Text>
          </Card>
        </Grid>

        {/* Charts Section */}
        <TabGroup>
          <TabList>
            <Tab>Performance</Tab>
            <Tab>Allocation</Tab>
            <Tab>Risk Analysis</Tab>
            <Tab>Top Holdings</Tab>
          </TabList>
          <TabPanels>
            <TabPanel>
              <Card>
                <Title>Portfolio Performance</Title>
                <AreaChart
                  className="h-72 mt-4"
                  data={data.performanceHistory}
                  index="date"
                  categories={["value"]}
                  colors={["blue"]}
                  valueFormatter={(value) => `$${(value / 1000).toFixed(0)}k`}
                />
              </Card>
            </TabPanel>

            <TabPanel>
              <Grid numItems={1} numItemsLg={2} className="gap-6">
                <Card>
                  <Title>Asset Allocation</Title>
                  <DonutChart
                    className="h-72 mt-4"
                    data={data.assetAllocation}
                    category="value"
                    index="name"
                    valueFormatter={(value) => `${value}%`}
                    colors={["blue", "green", "amber"]}
                  />
                </Card>
                <Card>
                  <Title>Allocation Targets</Title>
                  <div className="mt-4 space-y-4">
                    {data.assetAllocation.map((asset) => (
                      <div key={asset.name}>
                        <div className="flex justify-between mb-1">
                          <Text>{asset.name}</Text>
                          <Text>{asset.value}%</Text>
                        </div>
                        <ProgressBar value={asset.value} />
                      </div>
                    ))}
                  </div>
                </Card>
              </Grid>
            </TabPanel>

            <TabPanel>
              <Grid numItems={1} numItemsLg={2} className="gap-6">
                <Card>
                  <Title>Risk Metrics</Title>
                  <div className="mt-4 space-y-4">
                    <div className="flex justify-between py-2 border-b">
                      <Text>Portfolio Volatility</Text>
                      <Badge color={data.riskMetrics.portfolioVolatility > 0.20 ? "red" : "green"}>
                        {(data.riskMetrics.portfolioVolatility * 100).toFixed(1)}%
                      </Badge>
                    </div>
                    <div className="flex justify-between py-2 border-b">
                      <Text>Sharpe Ratio</Text>
                      <Badge color={data.riskMetrics.sharpeRatio > 1 ? "green" : "amber"}>
                        {data.riskMetrics.sharpeRatio.toFixed(2)}
                      </Badge>
                    </div>
                    <div className="flex justify-between py-2 border-b">
                      <Text>Max Drawdown</Text>
                      <Badge color="red">
                        {(data.riskMetrics.maxDrawdown * 100).toFixed(1)}%
                      </Badge>
                    </div>
                    <div className="flex justify-between py-2">
                      <Text>VaR (95%)</Text>
                      <Badge color="amber">
                        {(data.riskMetrics.var95 * 100).toFixed(1)}%
                      </Badge>
                    </div>
                  </div>
                </Card>
                <Card>
                  <Title>Risk Distribution</Title>
                  <LineChart
                    className="h-72 mt-4"
                    data={generateRiskDistribution()}
                    index="percentile"
                    categories={["return"]}
                    colors={["indigo"]}
                    valueFormatter={(value) => `${value.toFixed(1)}%`}
                  />
                </Card>
              </Grid>
            </TabPanel>

            <TabPanel>
              <Card>
                <Title>Top Performing Holdings</Title>
                <BarChart
                  className="h-72 mt-4"
                  data={data.topPerformers}
                  index="name"
                  categories={["return"]}
                  colors={["emerald"]}
                  valueFormatter={(value) => `${value.toFixed(1)}%`}
                  layout="vertical"
                />
              </Card>
            </TabPanel>
          </TabPanels>
        </TabGroup>

        {/* Recent Activity */}
        <Card>
          <Title>Recent Activity</Title>
          <div className="mt-4 space-y-3">
            {generateRecentActivity().map((activity, idx) => (
              <div key={idx} className="flex items-center justify-between py-2 border-b last:border-0">
                <div className="flex items-center space-x-3">
                  <div className={`p-2 rounded-lg ${activity.type === 'optimization' ? 'bg-blue-100' : 'bg-green-100'}`}>
                    <ChartPieIcon className={`h-5 w-5 ${activity.type === 'optimization' ? 'text-blue-600' : 'text-green-600'}`} />
                  </div>
                  <div>
                    <Text className="font-medium">{activity.title}</Text>
                    <Text className="text-xs text-gray-500">{activity.time}</Text>
                  </div>
                </div>
                <Badge color={activity.status === 'completed' ? 'green' : 'amber'}>
                  {activity.status}
                </Badge>
              </div>
            ))}
          </div>
        </Card>
      </div>
    </Layout>
  )
}

// Helper functions
function generateMockPerformance() {
  const data = []
  const startValue = 100000
  let currentValue = startValue
  const today = new Date()
  
  for (let i = 365; i >= 0; i--) {
    const date = new Date(today)
    date.setDate(date.getDate() - i)
    
    // Add some random variation
    const dailyReturn = (Math.random() - 0.48) * 0.02
    currentValue = currentValue * (1 + dailyReturn)
    
    if (i % 7 === 0) { // Weekly data points
      data.push({
        date: date.toISOString().split('T')[0],
        value: Math.round(currentValue)
      })
    }
  }
  
  return data
}

function generateRiskDistribution() {
  return [
    { percentile: '5%', return: -2.3 },
    { percentile: '10%', return: -1.5 },
    { percentile: '25%', return: -0.5 },
    { percentile: '50%', return: 0.8 },
    { percentile: '75%', return: 1.8 },
    { percentile: '90%', return: 2.5 },
    { percentile: '95%', return: 3.2 }
  ]
}

function generateRecentActivity() {
  return [
    {
      type: 'optimization',
      title: 'Portfolio rebalanced using Markowitz model',
      time: '2 hours ago',
      status: 'completed'
    },
    {
      type: 'trade',
      title: 'Executed 5 trades for risk reduction',
      time: '5 hours ago',
      status: 'completed'
    },
    {
      type: 'optimization',
      title: 'Risk parity optimization scheduled',
      time: '1 day ago',
      status: 'pending'
    }
  ]
}