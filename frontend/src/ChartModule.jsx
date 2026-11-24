import React from 'react'
import {
  AreaChart, Area, BarChart, Bar, PieChart, Pie, Cell, ResponsiveContainer,
  Tooltip, XAxis, YAxis, CartesianGrid, Line,
} from 'recharts'
import { money } from './api'

export default function ChartModule({ kind, data = [], colors = [] }) {
  if (kind === 'dashboard-trend') return (
    <ResponsiveContainer width="100%" height="100%">
      <AreaChart data={data}>
        <defs><linearGradient id="expfill" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stopColor={colors[0]} stopOpacity={.2}/><stop offset="95%" stopColor={colors[0]} stopOpacity={0}/></linearGradient></defs>
        <CartesianGrid strokeDasharray="3 4" vertical={false} stroke="#edf0f5"/>
        <XAxis dataKey="date" tickFormatter={v=>new Date(v+'T00:00').getDate()} tickLine={false} axisLine={false} tick={{fontSize:11,fill:'#9aa3b8'}} interval={Math.max(0,Math.floor(data.length/6)-1)}/>
        <YAxis hide/>
        <Tooltip formatter={(v,k)=>[money(v),k==='expenses'?'Expenses':'Income']} labelFormatter={v=>new Date(v+'T00:00').toLocaleDateString('en-IN',{day:'numeric',month:'short'})}/>
        <Area type="monotone" dataKey="expenses" stroke={colors[0]} strokeWidth={2.3} fill="url(#expfill)"/>
        <Area type="monotone" dataKey="income" stroke={colors[1]} strokeWidth={2} fill="transparent"/>
      </AreaChart>
    </ResponsiveContainer>
  )

  if (kind === 'dashboard-breakdown') return (
    <ResponsiveContainer width="100%" height="100%">
      <PieChart><Pie data={data} dataKey="amount" nameKey="category" innerRadius={67} outerRadius={91} paddingAngle={3} stroke="none">{data.map((_,i)=><Cell key={i} fill={colors[i%colors.length]}/>)}</Pie><Tooltip formatter={v=>money(v)}/></PieChart>
    </ResponsiveContainer>
  )

  if (kind === 'forecast') return (
    <ResponsiveContainer width="100%" height="100%">
      <AreaChart data={data}>
        <defs><linearGradient id="forecastFill" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stopColor={colors[0]} stopOpacity={.19}/><stop offset="100%" stopColor={colors[0]} stopOpacity={0}/></linearGradient></defs>
        <CartesianGrid strokeDasharray="3 4" vertical={false} stroke="#edf0f5"/>
        <XAxis dataKey="date" tickFormatter={v=>new Date(v+'T00:00').toLocaleDateString('en-IN',{day:'numeric',month:'short'})} tickLine={false} axisLine={false}/>
        <YAxis tickFormatter={v=>`₹${Math.round(v/1000)}k`} tickLine={false} axisLine={false}/>
        <Tooltip formatter={v=>money(v)} labelFormatter={v=>new Date(v+'T00:00').toLocaleDateString('en-IN',{day:'numeric',month:'short'})}/>
        <Area dataKey="projected_balance" name="Without expected income" type="monotone" stroke={colors[0]} strokeWidth={2.5} fill="url(#forecastFill)"/>
        <Line dataKey="with_expected_income" name="Including expected income" type="monotone" stroke={colors[1]} strokeWidth={2} strokeDasharray="5 4" dot={false}/>
      </AreaChart>
    </ResponsiveContainer>
  )

  if (kind === 'weekly-spending') return (
    <ResponsiveContainer width="100%" height="100%">
      <BarChart data={data}><CartesianGrid strokeDasharray="3 4" vertical={false} stroke="#edf0f5"/><XAxis dataKey="week" tickLine={false} axisLine={false}/><YAxis hide/><Tooltip formatter={v=>money(v)}/><Bar dataKey="amount" name="Spending" fill={colors[0]} radius={[5,5,0,0]}/></BarChart>
    </ResponsiveContainer>
  )

  return null
}
