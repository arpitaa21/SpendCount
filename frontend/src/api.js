import axios from 'axios'

const getBaseUrl = () => {
  const custom = import.meta.env.VITE_API_BASE_URL || import.meta.env.VITE_API_URL
  if (custom) return custom.replace(/\/+$/, '')
  if (typeof window !== 'undefined' && (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1')) {
    return 'http://localhost:8000/api'
  }
  return '/api'
}

export const api = axios.create({ baseURL: getBaseUrl() })

api.interceptors.request.use(config => {
  const token = localStorage.getItem('spendcount_token')
  if (token) config.headers.Authorization = `Bearer ${token}`
  return config
})

api.interceptors.response.use(
  r => r,
  e => {
    const requestUrl = String(e.config?.url || '')
    const isPublicAuthRequest = /\/auth\/(login|register)(?:\?|$)/.test(requestUrl)
    const sentToken = Boolean(e.config?.headers?.Authorization || e.config?.headers?.authorization)
    if (e.response?.status === 401 && sentToken && !isPublicAuthRequest) {
      localStorage.removeItem('spendcount_token')
      if (location.pathname !== '/login') location.href = '/login'
    }
    return Promise.reject(e)
  }
)

export const money = n => `₹${Number(n || 0).toLocaleString('en-IN', { maximumFractionDigits: 0 })}`

export const localDate = (date = new Date()) => {
  const year = date.getFullYear()
  const month = String(date.getMonth() + 1).padStart(2, '0')
  const day = String(date.getDate()).padStart(2, '0')
  return `${year}-${month}-${day}`
}

export const errorText = e => e.response?.data?.detail || e.message || 'Something went wrong'
