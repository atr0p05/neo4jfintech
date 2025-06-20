import { useState } from 'react'
import Head from 'next/head'
import { 
  Card, Title, Text, Metric, Badge, Grid, Col, Button,
  Select, SelectItem, NumberInput, ProgressBar,
  AreaChart, BarChart, Tab, TabGroup, TabList, TabPanel, TabPanels
} from '@tremor/react'
import { 
  CpuChipIcon, ChartBarIcon, AdjustmentsHorizontalIcon,
  PlayIcon, DocumentArrowDownIcon
} from '@heroicons/react/24/outline'
import Layout from '@/components/Layout'
import { optimizationAPI } from '@/services/api'
import toast from 'react-hot-toast'

interface OptimizationResult {
  model: string
  expectedReturn: number
  volatility: number
  sharpeRatio: number
  weights: Record<string, number>
  constraints: any
  timestamp: string
}

const OPTIMIZATION_MODELS = [
  { value: 'markowitz', label: 'Markowitz Mean-Variance' },
  { value: 'black_litterman', label: 'Black-Litterman' },
  { value: 'risk_parity', label: 'Risk Parity' },
  { value: 'cvar', label: 'CVaR Optimization' },
  { value: 'factor_based', label: 'Factor-Based' },
  { value: 'robust', label: 'Robust Optimization' },
  { value: 'kelly', label: 'Kelly Criterion' }
]

export default function Optimization() {
  const [selectedModel, setSelectedModel] = useState('markowitz')
  const [targetReturn, setTargetReturn] = useState<number | undefined>(10)
  const [maxRisk, setMaxRisk] = useState<number | undefined>(15)
  const [minWeight, setMinWeight] = useState<number | undefined>(2)
  const [maxWeight, setMaxWeight] = useState<number | undefined>(25)
  const [loading, setLoading] = useState(false)
  const [results, setResults] = useState<OptimizationResult | null>(null)
  const [comparison, setComparison] = useState<any[]>([])
  
  const clientId = 'CLI_100001' // In production, from user context

  const handleOptimize = async () => {
    try {
      setLoading(true)
      toast.loading('Running optimization...')
      
      const params = {
        target_return: targetReturn ? targetReturn / 100 : undefined,
        max_risk: maxRisk ? maxRisk / 100 : undefined,
        constraints: {
          min_weight: minWeight ? minWeight / 100 : 0.02,
          max_weight: maxWeight ? maxWeight / 100 : 0.25
        }
      }
      
      const result = await optimizationAPI.optimize(clientId, selectedModel, params)
      
      setResults({
        model: selectedModel,
        expectedReturn: result.expected_return,
        volatility: result.volatility,
        sharpeRatio: result.sharpe_ratio,
        weights: result.weights,
        constraints: result.constraints,
        timestamp: new Date().toISOString()
      })
      
      toast.dismiss()
      toast.success('Optimization completed successfully!')
      
      // Fetch comparison data
      const comparisonData = await optimizationAPI.compareModels(clientId)
      setComparison(comparisonData)
      
    } catch (error) {
      toast.dismiss()
      toast.error('Optimization failed. Please try again.')
      
      // Use mock data for demo
      setResults(generateMockResults())
      setComparison(generateMockComparison())
    } finally {
      setLoading(false)
    }
  }

  const handleExportResults = () => {
    if (!results) return
    
    const csv = convertToCSV(results.weights)
    const blob = new Blob([csv], { type: 'text/csv' })
    const url = window.URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `optimization_${selectedModel}_${new Date().toISOString()}.csv`
    a.click()
    window.URL.revokeObjectURL(url)
    
    toast.success('Results exported successfully!')
  }

  return (
    <Layout>
      <Head>
        <title>Portfolio Optimization - Neo4j Investment Platform</title>
      </Head>

      <div className="space-y-6">
        {/* Header */}
        <div className="flex justify-between items-center">
          <div>
            <Title>Portfolio Optimization</Title>
            <Text>Run advanced optimization models to find optimal portfolio weights</Text>
          </div>
          <Badge color="blue">
            <CpuChipIcon className="h-4 w-4 mr-1" />
            7 Models Available
          </Badge>
        </div>

        {/* Configuration */}
        <Card>
          <Title>Optimization Configuration</Title>
          <Grid numItems={1} numItemsSm={2} numItemsLg={4} className="gap-4 mt-4">
            <Col>
              <Text>Optimization Model</Text>
              <Select value={selectedModel} onValueChange={setSelectedModel} className="mt-2">
                {OPTIMIZATION_MODELS.map(model => (
                  <SelectItem key={model.value} value={model.value}>
                    {model.label}
                  </SelectItem>
                ))}
              </Select>
            </Col>
            
            <Col>
              <Text>Target Return (%)</Text>
              <NumberInput
                value={targetReturn}
                onValueChange={setTargetReturn}
                min={0}
                max={50}
                step={0.5}
                className="mt-2"
                placeholder="e.g., 10"
              />
            </Col>
            
            <Col>
              <Text>Max Risk (%)</Text>
              <NumberInput
                value={maxRisk}
                onValueChange={setMaxRisk}
                min={0}
                max={50}
                step={0.5}
                className="mt-2"
                placeholder="e.g., 15"
              />
            </Col>
            
            <Col>
              <Text>Weight Range (%)</Text>
              <div className="flex space-x-2 mt-2">
                <NumberInput
                  value={minWeight}
                  onValueChange={setMinWeight}
                  min={0}
                  max={10}
                  step={0.5}
                  placeholder="Min"
                />
                <NumberInput
                  value={maxWeight}
                  onValueChange={setMaxWeight}
                  min={10}
                  max={50}
                  step={1}
                  placeholder="Max"
                />
              </div>
            </Col>
          </Grid>

          <div className="mt-6 flex justify-end space-x-3">
            <Button
              icon={AdjustmentsHorizontalIcon}
              variant="secondary"
              onClick={() => toast.info('Advanced settings coming soon!')}
            >
              Advanced Settings
            </Button>
            <Button
              icon={PlayIcon}
              onClick={handleOptimize}
              loading={loading}
              loadingText="Optimizing..."
              color="blue"
            >
              Run Optimization
            </Button>
          </div>
        </Card>

        {/* Results */}
        {results && (
          <TabGroup>
            <TabList>
              <Tab>Results Overview</Tab>
              <Tab>Weight Distribution</Tab>
              <Tab>Model Comparison</Tab>
              <Tab>Efficient Frontier</Tab>
            </TabList>
            <TabPanels>
              <TabPanel>
                <Grid numItems={1} numItemsLg={2} className="gap-6">
                  <Card>
                    <Title>Optimization Metrics</Title>
                    <Grid numItems={2} className="gap-4 mt-4">
                      <div>
                        <Text>Expected Return</Text>
                        <Metric>{(results.expectedReturn * 100).toFixed(2)}%</Metric>
                      </div>
                      <div>
                        <Text>Portfolio Risk</Text>
                        <Metric>{(results.volatility * 100).toFixed(2)}%</Metric>
                      </div>
                      <div>
                        <Text>Sharpe Ratio</Text>
                        <Metric>{results.sharpeRatio.toFixed(3)}</Metric>
                      </div>
                      <div>
                        <Text>Diversification</Text>
                        <Metric>{Object.keys(results.weights).filter(k => results.weights[k] > 0.01).length}</Metric>
                      </div>
                    </Grid>
                    
                    <div className="mt-6">
                      <Button
                        icon={DocumentArrowDownIcon}
                        onClick={handleExportResults}
                        variant="secondary"
                        className="w-full"
                      >
                        Export Results
                      </Button>
                    </div>
                  </Card>

                  <Card>
                    <Title>Top Holdings</Title>
                    <div className="space-y-3 mt-4">
                      {Object.entries(results.weights)
                        .sort(([,a], [,b]) => b - a)
                        .slice(0, 5)
                        .map(([symbol, weight]) => (
                          <div key={symbol}>
                            <div className="flex justify-between mb-1">
                              <Text>{symbol}</Text>
                              <Text>{(weight * 100).toFixed(1)}%</Text>
                            </div>
                            <ProgressBar value={weight * 100} color="blue" />
                          </div>
                        ))}
                    </div>
                  </Card>
                </Grid>
              </TabPanel>

              <TabPanel>
                <Card>
                  <Title>Optimal Weight Distribution</Title>
                  <BarChart
                    className="h-96 mt-4"
                    data={Object.entries(results.weights)
                      .filter(([,weight]) => weight > 0.01)
                      .map(([symbol, weight]) => ({
                        symbol,
                        weight: weight * 100
                      }))
                      .sort((a, b) => b.weight - a.weight)}
                    index="symbol"
                    categories={["weight"]}
                    colors={["blue"]}
                    valueFormatter={(value) => `${value.toFixed(1)}%`}
                    layout="vertical"
                  />
                </Card>
              </TabPanel>

              <TabPanel>
                <Card>
                  <Title>Model Performance Comparison</Title>
                  <BarChart
                    className="h-72 mt-4"
                    data={comparison}
                    index="model"
                    categories={["sharpeRatio"]}
                    colors={["emerald"]}
                    valueFormatter={(value) => value.toFixed(3)}
                    layout="vertical"
                  />
                  
                  <div className="mt-6">
                    <Title>Detailed Comparison</Title>
                    <div className="overflow-x-auto mt-4">
                      <table className="min-w-full divide-y divide-gray-200">
                        <thead>
                          <tr>
                            <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase">Model</th>
                            <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase">Return</th>
                            <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase">Risk</th>
                            <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase">Sharpe</th>
                            <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase">Holdings</th>
                          </tr>
                        </thead>
                        <tbody className="divide-y divide-gray-200">
                          {comparison.map((model) => (
                            <tr key={model.model} className={model.model === selectedModel ? 'bg-blue-50' : ''}>
                              <td className="px-4 py-2 text-sm">{model.model}</td>
                              <td className="px-4 py-2 text-sm">{(model.expectedReturn * 100).toFixed(1)}%</td>
                              <td className="px-4 py-2 text-sm">{(model.volatility * 100).toFixed(1)}%</td>
                              <td className="px-4 py-2 text-sm">{model.sharpeRatio.toFixed(3)}</td>
                              <td className="px-4 py-2 text-sm">{model.numHoldings}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                </Card>
              </TabPanel>

              <TabPanel>
                <Card>
                  <Title>Efficient Frontier</Title>
                  <AreaChart
                    className="h-96 mt-4"
                    data={generateEfficientFrontier()}
                    index="risk"
                    categories={["portfolio", "current"]}
                    colors={["blue", "red"]}
                    valueFormatter={(value) => `${value.toFixed(1)}%`}
                  />
                  <Text className="mt-4 text-sm text-gray-600">
                    The efficient frontier shows the set of optimal portfolios that offer the highest expected return for each level of risk.
                    Your optimized portfolio is shown on the frontier.
                  </Text>
                </Card>
              </TabPanel>
            </TabPanels>
          </TabGroup>
        )}

        {/* Model Information */}
        <Card>
          <Title>About {OPTIMIZATION_MODELS.find(m => m.value === selectedModel)?.label}</Title>
          <Text className="mt-2">{getModelDescription(selectedModel)}</Text>
        </Card>
      </div>
    </Layout>
  )
}

// Helper functions
function generateMockResults(): OptimizationResult {
  return {
    model: 'markowitz',
    expectedReturn: 0.125,
    volatility: 0.152,
    sharpeRatio: 1.234,
    weights: {
      'AAPL': 0.15,
      'MSFT': 0.12,
      'JPM': 0.10,
      'TLT': 0.20,
      'GLD': 0.08,
      'VTI': 0.10,
      'BND': 0.15,
      'XOM': 0.05,
      'JNJ': 0.05
    },
    constraints: {
      min_weight: 0.02,
      max_weight: 0.25
    },
    timestamp: new Date().toISOString()
  }
}

function generateMockComparison() {
  return [
    { model: 'Markowitz', expectedReturn: 0.125, volatility: 0.152, sharpeRatio: 1.234, numHoldings: 9 },
    { model: 'Black-Litterman', expectedReturn: 0.135, volatility: 0.158, sharpeRatio: 1.298, numHoldings: 8 },
    { model: 'Risk Parity', expectedReturn: 0.105, volatility: 0.125, sharpeRatio: 1.152, numHoldings: 10 },
    { model: 'CVaR', expectedReturn: 0.115, volatility: 0.142, sharpeRatio: 1.189, numHoldings: 7 },
    { model: 'Factor-Based', expectedReturn: 0.128, volatility: 0.155, sharpeRatio: 1.245, numHoldings: 12 }
  ]
}

function generateEfficientFrontier() {
  const data = []
  for (let risk = 5; risk <= 25; risk += 1) {
    const baseReturn = risk * 0.6 + Math.random() * 2
    data.push({
      risk,
      portfolio: baseReturn,
      current: risk === 15 ? baseReturn : null
    })
  }
  return data
}

function getModelDescription(model: string): string {
  const descriptions: Record<string, string> = {
    markowitz: "The Markowitz Mean-Variance model is the foundation of modern portfolio theory. It finds the optimal balance between expected return and risk by minimizing portfolio variance for a given expected return.",
    black_litterman: "Black-Litterman combines market equilibrium with investor views to generate more stable, intuitive portfolio weights. It's particularly useful when you have specific market opinions.",
    risk_parity: "Risk Parity allocates capital so that each asset contributes equally to portfolio risk. This approach tends to result in more balanced, diversified portfolios.",
    cvar: "Conditional Value at Risk (CVaR) optimization focuses on minimizing tail risk - the average loss in worst-case scenarios. Ideal for risk-averse investors.",
    factor_based: "Factor-based optimization targets specific risk factors (value, momentum, quality) to achieve desired factor exposures while maintaining diversification.",
    robust: "Robust optimization accounts for uncertainty in expected returns and covariances, producing portfolios that perform well across various market scenarios.",
    kelly: "The Kelly Criterion maximizes long-term wealth growth by optimizing position sizes based on edge and probability of success."
  }
  return descriptions[model] || "Advanced portfolio optimization model."
}

function convertToCSV(weights: Record<string, number>): string {
  const headers = ['Symbol', 'Weight (%)', 'Suggested Value ($125,000 portfolio)']
  const rows = Object.entries(weights)
    .filter(([,weight]) => weight > 0.01)
    .sort(([,a], [,b]) => b - a)
    .map(([symbol, weight]) => [
      symbol,
      (weight * 100).toFixed(2),
      (weight * 125000).toFixed(0)
    ])
  
  return [headers, ...rows].map(row => row.join(',')).join('\n')
}