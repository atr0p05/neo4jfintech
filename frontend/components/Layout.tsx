import { ReactNode } from 'react'
import Link from 'next/link'
import { useRouter } from 'next/router'
import { 
  HomeIcon, ChartPieIcon, ChartBarIcon, 
  CpuChipIcon, BeakerIcon, Cog6ToothIcon 
} from '@heroicons/react/24/outline'

interface LayoutProps {
  children: ReactNode
}

const navigation = [
  { name: 'Dashboard', href: '/dashboard', icon: HomeIcon },
  { name: 'Portfolios', href: '/portfolio', icon: ChartPieIcon },
  { name: 'Optimization', href: '/optimization', icon: CpuChipIcon },
  { name: 'Risk Analysis', href: '/risk', icon: ChartBarIcon },
  { name: 'Backtesting', href: '/backtest', icon: BeakerIcon },
  { name: 'Settings', href: '/settings', icon: Cog6ToothIcon },
]

export default function Layout({ children }: LayoutProps) {
  const router = useRouter()

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Sidebar */}
      <div className="fixed inset-y-0 left-0 z-50 w-64 bg-white shadow-lg">
        <div className="flex h-16 items-center justify-center border-b">
          <Link href="/" className="text-xl font-bold text-primary-600">
            Neo4j Platform
          </Link>
        </div>
        <nav className="mt-5 px-2">
          {navigation.map((item) => {
            const isActive = router.pathname === item.href
            return (
              <Link
                key={item.name}
                href={item.href}
                className={`
                  group flex items-center px-2 py-2 text-sm font-medium rounded-md mb-1
                  ${isActive 
                    ? 'bg-primary-100 text-primary-900' 
                    : 'text-gray-600 hover:bg-gray-50 hover:text-gray-900'}
                `}
              >
                <item.icon
                  className={`
                    mr-3 h-5 w-5 flex-shrink-0
                    ${isActive ? 'text-primary-600' : 'text-gray-400 group-hover:text-gray-500'}
                  `}
                  aria-hidden="true"
                />
                {item.name}
              </Link>
            )
          })}
        </nav>
      </div>

      {/* Main content */}
      <div className="pl-64">
        <main className="py-6">
          <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
            {children}
          </div>
        </main>
      </div>
    </div>
  )
}