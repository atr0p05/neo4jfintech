import { useState, useEffect } from 'react'
import Head from 'next/head'
import Link from 'next/link'
import { ArrowRightIcon, ChartBarIcon, ShieldCheckIcon, LightBulbIcon, ChartPieIcon } from '@heroicons/react/24/outline'
import { Card, Title, Text, Metric, ProgressBar, Badge, Grid, Col } from '@tremor/react'
import { motion } from 'framer-motion'
import toast from 'react-hot-toast'

export default function Home() {
  const [stats, setStats] = useState({
    totalClients: 0,
    totalAUM: 0,
    avgReturn: 0,
    portfoliosOptimized: 0
  })

  useEffect(() => {
    // Simulate loading stats
    setTimeout(() => {
      setStats({
        totalClients: 100,
        totalAUM: 125000000,
        avgReturn: 12.5,
        portfoliosOptimized: 87
      })
    }, 1000)
  }, [])

  return (
    <>
      <Head>
        <title>Neo4j Investment Platform</title>
      </Head>

      <div className="min-h-screen bg-gray-50">
        {/* Navigation */}
        <nav className="bg-white shadow-sm">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div className="flex justify-between h-16">
              <div className="flex items-center">
                <h1 className="text-xl font-bold text-gray-900">Neo4j Investment Platform</h1>
              </div>
              <div className="flex items-center space-x-4">
                <Link href="/dashboard" className="text-gray-700 hover:text-primary-600 px-3 py-2 rounded-md text-sm font-medium">
                  Dashboard
                </Link>
                <Link href="/portfolio" className="text-gray-700 hover:text-primary-600 px-3 py-2 rounded-md text-sm font-medium">
                  Portfolios
                </Link>
                <Link href="/optimization" className="text-gray-700 hover:text-primary-600 px-3 py-2 rounded-md text-sm font-medium">
                  Optimization
                </Link>
              </div>
            </div>
          </div>
        </nav>

        {/* Hero Section */}
        <div className="relative bg-primary-800 overflow-hidden">
          <div className="max-w-7xl mx-auto">
            <div className="relative z-10 pb-8 sm:pb-16 md:pb-20 lg:max-w-2xl lg:w-full lg:pb-28 xl:pb-32">
              <main className="mt-10 mx-auto max-w-7xl px-4 sm:mt-12 sm:px-6 md:mt-16 lg:mt-20 lg:px-8 xl:mt-28">
                <motion.div 
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.5 }}
                  className="sm:text-center lg:text-left"
                >
                  <h1 className="text-4xl tracking-tight font-extrabold text-white sm:text-5xl md:text-6xl">
                    <span className="block">Advanced Portfolio</span>
                    <span className="block text-primary-200">Management Platform</span>
                  </h1>
                  <p className="mt-3 text-base text-gray-200 sm:mt-5 sm:text-lg sm:max-w-xl sm:mx-auto md:mt-5 md:text-xl lg:mx-0">
                    Harness the power of graph analytics and machine learning to optimize investment portfolios, 
                    manage risk, and deliver superior returns.
                  </p>
                  <div className="mt-5 sm:mt-8 sm:flex sm:justify-center lg:justify-start">
                    <div className="rounded-md shadow">
                      <Link
                        href="/dashboard"
                        className="w-full flex items-center justify-center px-8 py-3 border border-transparent text-base font-medium rounded-md text-primary-700 bg-white hover:bg-gray-50 md:py-4 md:text-lg md:px-10"
                      >
                        Get Started
                        <ArrowRightIcon className="ml-2 h-5 w-5" />
                      </Link>
                    </div>
                  </div>
                </motion.div>
              </main>
            </div>
          </div>
        </div>

        {/* Stats Section */}
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
          <Grid numItems={1} numItemsSm={2} numItemsLg={4} className="gap-6">
            <Col>
              <Card>
                <Text>Total Clients</Text>
                <Metric>{stats.totalClients}</Metric>
                <ProgressBar value={75} className="mt-2" />
              </Card>
            </Col>
            <Col>
              <Card>
                <Text>Assets Under Management</Text>
                <Metric>${(stats.totalAUM / 1000000).toFixed(0)}M</Metric>
                <Badge color="green" className="mt-2">+15% YoY</Badge>
              </Card>
            </Col>
            <Col>
              <Card>
                <Text>Average Return</Text>
                <Metric>{stats.avgReturn}%</Metric>
                <Badge color="emerald" className="mt-2">Above Market</Badge>
              </Card>
            </Col>
            <Col>
              <Card>
                <Text>Portfolios Optimized</Text>
                <Metric>{stats.portfoliosOptimized}</Metric>
                <Text className="mt-2 text-xs">This month</Text>
              </Card>
            </Col>
          </Grid>
        </div>

        {/* Features Section */}
        <div className="py-16 bg-white">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div className="text-center">
              <h2 className="text-3xl font-extrabold text-gray-900">
                Powerful Features for Modern Portfolio Management
              </h2>
            </div>

            <div className="mt-12">
              <Grid numItems={1} numItemsSm={2} numItemsLg={3} className="gap-8">
                <motion.div whileHover={{ scale: 1.05 }} transition={{ type: "spring", stiffness: 300 }}>
                  <Card>
                    <div className="flex items-center mb-4">
                      <ChartPieIcon className="h-8 w-8 text-primary-600 mr-3" />
                      <Title>Portfolio Optimization</Title>
                    </div>
                    <Text>
                      Advanced optimization algorithms including Markowitz, Black-Litterman, and Risk Parity 
                      to maximize returns while managing risk.
                    </Text>
                  </Card>
                </motion.div>

                <motion.div whileHover={{ scale: 1.05 }} transition={{ type: "spring", stiffness: 300 }}>
                  <Card>
                    <div className="flex items-center mb-4">
                      <ShieldCheckIcon className="h-8 w-8 text-primary-600 mr-3" />
                      <Title>Risk Analytics</Title>
                    </div>
                    <Text>
                      Comprehensive risk assessment with VaR, CVaR, factor exposures, and correlation analysis 
                      powered by graph analytics.
                    </Text>
                  </Card>
                </motion.div>

                <motion.div whileHover={{ scale: 1.05 }} transition={{ type: "spring", stiffness: 300 }}>
                  <Card>
                    <div className="flex items-center mb-4">
                      <LightBulbIcon className="h-8 w-8 text-primary-600 mr-3" />
                      <Title>AI Recommendations</Title>
                    </div>
                    <Text>
                      GraphRAG-powered investment recommendations combining Neo4j graph traversal with 
                      GPT-4 for intelligent insights.
                    </Text>
                  </Card>
                </motion.div>

                <motion.div whileHover={{ scale: 1.05 }} transition={{ type: "spring", stiffness: 300 }}>
                  <Card>
                    <div className="flex items-center mb-4">
                      <ChartBarIcon className="h-8 w-8 text-primary-600 mr-3" />
                      <Title>Real-time Market Data</Title>
                    </div>
                    <Text>
                      Live market data integration with technical indicators, fundamental analysis, and 
                      news sentiment tracking.
                    </Text>
                  </Card>
                </motion.div>

                <motion.div whileHover={{ scale: 1.05 }} transition={{ type: "spring", stiffness: 300 }}>
                  <Card>
                    <div className="flex items-center mb-4">
                      <ChartPieIcon className="h-8 w-8 text-primary-600 mr-3" />
                      <Title>Backtesting Engine</Title>
                    </div>
                    <Text>
                      Test investment strategies with historical data using our comprehensive backtesting 
                      framework with QuantStats reporting.
                    </Text>
                  </Card>
                </motion.div>

                <motion.div whileHover={{ scale: 1.05 }} transition={{ type: "spring", stiffness: 300 }}>
                  <Card>
                    <div className="flex items-center mb-4">
                      <ShieldCheckIcon className="h-8 w-8 text-primary-600 mr-3" />
                      <Title>Compliance Monitoring</Title>
                    </div>
                    <Text>
                      Automated compliance checks ensuring portfolios adhere to regulatory requirements 
                      and client constraints.
                    </Text>
                  </Card>
                </motion.div>
              </Grid>
            </div>
          </div>
        </div>

        {/* CTA Section */}
        <div className="bg-primary-700">
          <div className="max-w-7xl mx-auto py-12 px-4 sm:px-6 lg:px-8">
            <div className="text-center">
              <h2 className="text-3xl font-extrabold text-white">
                Ready to transform your investment management?
              </h2>
              <p className="mt-4 text-lg text-primary-200">
                Start optimizing portfolios with the power of graph analytics today.
              </p>
              <div className="mt-8">
                <Link
                  href="/dashboard"
                  className="inline-flex items-center justify-center px-5 py-3 border border-transparent text-base font-medium rounded-md text-primary-600 bg-white hover:bg-primary-50"
                >
                  Access Dashboard
                  <ArrowRightIcon className="ml-2 h-5 w-5" />
                </Link>
              </div>
            </div>
          </div>
        </div>
      </div>
    </>
  )
}