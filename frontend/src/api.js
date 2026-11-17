import axios from 'axios'
export const api = axios.create({ baseURL: import.meta.env.VITE_API_URL || 'http://localhost:8000/api' })
api.interceptors.request.use(config => { const token = localStorage.getItem('spendcount_token'); if(token) config.headers.Authorization = `Bearer ${token}`; return config })
api.interceptors.response.use(r => r, e => { if(e.response?.status===401){ localStorage.removeItem('spendcount_token'); if(location.pathname!='/login') location.href='/login' } return Promise.reject(e) })
export const money = n => `₹${Number(n||0).toLocaleString('en-IN',{maximumFractionDigits:0})}`
export const errorText = e => e.response?.data?.detail || e.message || 'Something went wrong'
