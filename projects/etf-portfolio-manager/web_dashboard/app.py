#!/usr/bin/env python3
"""
Web Dashboard for ETF Portfolio Manager
Real-time visualization and monitoring interface.
"""

from flask import Flask, render_template, jsonify, request
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os
import threading
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Sample data for demonstration
class PortfolioData:
    """Manage portfolio data for the dashboard."""
    
    def __init__(self):
        self.portfolio_value = 100000
        self.allocation = {
            "VOO": {"weight": 0.45, "value": 45000, "return": 0.105},
            "VGT": {"weight": 0.20, "value": 20000, "return": 0.152},
            "XLP": {"weight": 0.15, "value": 15000, "return": 0.085},
            "XLU": {"weight": 0.10, "value": 10000, "return": 0.072},
            "FXI": {"weight": 0.05, "value": 5000, "return": 0.128},
            "KBA": {"weight": 0.05, "value": 5000, "return": 0.135}
        }
        
        self.performance_history = self._generate_performance_history()
        self.alerts = []
        self.last_update = datetime.now()
        
        # Start background update thread
        self.update_thread = threading.Thread(target=self._background_updater, daemon=True)
        self.update_thread.start()
    
    def _generate_performance_history(self):
        """Generate sample performance history."""
        dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
        base_value = 100000
        
        # Generate random walk with drift
        returns = np.random.normal(0.0005, 0.01, len(dates))
        values = base_value * (1 + returns).cumprod()
        
        history = []
        for date, value in zip(dates, values):
            history.append({
                "date": date.strftime("%Y-%m-%d"),
                "value": float(value),
                "daily_return": float(returns[len(history)] if len(history) < len(returns) else 0)
            })
        
        return history
    
    def _background_updater(self):
        """Background thread to update portfolio data."""
        while True:
            try:
                self._update_portfolio_values()
                self._check_alerts()
                time.sleep(60)  # Update every minute
            except Exception as e:
                logger.error(f"Background update error: {e}")
                time.sleep(300)  # Wait 5 minutes on error
    
    def _update_portfolio_values(self):
        """Update portfolio values with simulated market movements."""
        import yfinance as yf
        
        for symbol in self.allocation.keys():
            try:
                ticker = yf.Ticker(symbol)
                info = ticker.info
                
                current_price = info.get("regularMarketPrice") or info.get("currentPrice")
                if current_price:
                    # Simulate price change
                    weight = self.allocation[symbol]["weight"]
                    current_value = self.portfolio_value * weight
                    shares = current_value / current_price if current_price > 0 else 0
                    
                    # Update with small random movement for demo
                    price_change = np.random.normal(0.001, 0.005)
                    new_price = current_price * (1 + price_change)
                    new_value = shares * new_price
                    
                    self.allocation[symbol]["value"] = new_value
                    self.allocation[symbol]["return"] = price_change * 252  # Annualized
                
            except Exception as e:
                logger.warning(f"Error updating {symbol}: {e}")
        
        # Update total portfolio value
        self.portfolio_value = sum(etf["value"] for etf in self.allocation.values())
        
        # Add to history
        self.performance_history.append({
            "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "value": float(self.portfolio_value),
            "daily_return": float(np.random.normal(0.0005, 0.01))
        })
        
        # Keep only last 1000 points
        if len(self.performance_history) > 1000:
            self.performance_history = self.performance_history[-1000:]
        
        self.last_update = datetime.now()
        logger.info(f"Portfolio updated: ${self.portfolio_value:,.2f}")
    
    def _check_alerts(self):
        """Check for alert conditions."""
        # Price change alerts
        for symbol, data in self.allocation.items():
            daily_return = data["return"] / 252  # Convert annual to daily
            
            if abs(daily_return) > 0.05:  # 5% daily change
                alert = {
                    "type": "price_alert",
                    "symbol": symbol,
                    "message": f"{symbol}: {daily_return*100:+.1f}% daily change",
                    "severity": "high" if abs(daily_return) > 0.10 else "medium",
                    "timestamp": datetime.now().isoformat()
                }
                
                if alert not in self.alerts:
                    self.alerts.append(alert)
        
        # Portfolio value alerts
        if len(self.performance_history) >= 2:
            latest = self.performance_history[-1]["value"]
            previous = self.performance_history[-2]["value"]
            daily_change = (latest - previous) / previous
            
            if abs(daily_change) > 0.03:  # 3% portfolio change
                alert = {
                    "type": "portfolio_alert",
                    "message": f"Portfolio: {daily_change*100:+.1f}% daily change",
                    "severity": "high" if abs(daily_change) > 0.05 else "medium",
                    "timestamp": datetime.now().isoformat()
                }
                
                if alert not in self.alerts:
                    self.alerts.append(alert)
        
        # Keep only recent alerts (last 24 hours)
        cutoff = datetime.now() - timedelta(hours=24)
        self.alerts = [
            alert for alert in self.alerts 
            if datetime.fromisoformat(alert["timestamp"].replace('Z', '+00:00')) > cutoff
        ]
    
    def get_dashboard_data(self):
        """Get data for dashboard."""
        return {
            "portfolio_value": self.portfolio_value,
            "allocation": self.allocation,
            "performance_history": self.performance_history[-30:],  # Last 30 days
            "alerts": self.alerts[-10:],  # Last 10 alerts
            "last_update": self.last_update.isoformat(),
            "metrics": self._calculate_metrics()
        }
    
    def _calculate_metrics(self):
        """Calculate portfolio metrics."""
        if len(self.performance_history) < 2:
            return {}
        
        values = [point["value"] for point in self.performance_history]
        returns = [point["daily_return"] for point in self.performance_history]
        
        # Calculate metrics
        total_return = (values[-1] - values[0]) / values[0]
        annual_return = (1 + total_return) ** (252 / len(values)) - 1
        
        volatility = np.std(returns) * np.sqrt(252)
        sharpe_ratio = (annual_return - 0.02) / volatility if volatility > 0 else 0
        
        # Drawdown
        cumulative = (1 + np.array(returns)).cumprod()
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = np.min(drawdown) if len(drawdown) > 0 else 0
        
        return {
            "annual_return": annual_return,
            "volatility": volatility,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "total_return": total_return
        }

# Initialize portfolio data
portfolio_data = PortfolioData()

# Routes
@app.route('/')
def index():
    """Main dashboard page."""
    return render_template('dashboard.html')

@app.route('/api/dashboard')
def get_dashboard():
    """Get dashboard data."""
    data = portfolio_data.get_dashboard_data()
    return jsonify(data)

@app.route('/api/allocation')
def get_allocation():
    """Get allocation data for charts."""
    allocation = portfolio_data.allocation
    
    # Prepare data for pie chart
    labels = list(allocation.keys())
    values = [allocation[symbol]["value"] for symbol in labels]
    weights = [allocation[symbol]["weight"] for symbol in labels]
    returns = [allocation[symbol]["return"] for symbol in labels]
    
    return jsonify({
        "labels": labels,
        "values": values,
        "weights": weights,
        "returns": returns
    })

@app.route('/api/performance')
def get_performance():
    """Get performance history."""
    history = portfolio_data.performance_history
    
    dates = [point["date"] for point in history]
    values = [point["value"] for point in history]
    returns = [point["daily_return"] for point in history]
    
    return jsonify({
        "dates": dates,
        "values": values,
        "returns": returns
    })

@app.route('/api/alerts')
def get_alerts():
    """Get current alerts."""
    return jsonify({
        "alerts": portfolio_data.alerts,
        "count": len(portfolio_data.alerts)
    })

@app.route('/api/metrics')
def get_metrics():
    """Get portfolio metrics."""
    return jsonify(portfolio_data._calculate_metrics())

@app.route('/api/update', methods=['POST'])
def manual_update():
    """Trigger manual update."""
    portfolio_data._update_portfolio_values()
    return jsonify({"status": "updated", "timestamp": datetime.now().isoformat()})

@app.route('/api/backtest', methods=['POST'])
def run_backtest():
    """Run backtest with custom parameters."""
    try:
        data = request.json
        allocation = data.get('allocation', portfolio_data.allocation)
        initial_capital = data.get('initial_capital', 100000)
        years = data.get('years', 5)
        
        # Here you would call the actual backtest engine
        # For now, return simulated results
        
        results = {
            "total_return": np.random.normal(0.15, 0.05),
            "annual_return": np.random.normal(0.15, 0.05),
            "max_drawdown": np.random.normal(-0.15, 0.05),
            "sharpe_ratio": np.random.normal(0.7, 0.2),
            "allocation": allocation
        }
        
        return jsonify(results)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/api/weekly_scan', methods=['POST'])
def weekly_scan():
    """Run weekly portfolio scan and adjustment recommendations."""
    try:
        # Get current allocation
        current_allocation = portfolio_data.allocation
        
        # Analyze market conditions
        market_analysis = analyze_market_conditions()
        
        # Generate recommendations
        recommendations = generate_recommendations(current_allocation, market_analysis)
        
        return jsonify({
            "timestamp": datetime.now().isoformat(),
            "market_analysis": market_analysis,
            "recommendations": recommendations,
            "action_required": len(recommendations.get("actions", [])) > 0
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 400

def analyze_market_conditions():
    """Analyze current market conditions."""
    import yfinance as yf
    
    # Get market indicators
    try:
        spy = yf.Ticker("SPY")
        vix = yf.Ticker("^VIX")
        
        spy_info = spy.info
        vix_info = vix.info
        
        market_conditions = {
            "sp500_level": spy_info.get("regularMarketPrice"),
            "vix_level": vix_info.get("regularMarketPrice"),
            "market_sentiment": "neutral",
            "risk_level": "medium"
        }
        
        # Determine sentiment based on VIX
        vix_level = market_conditions["vix_level"]
        if vix_level:
            if vix_level > 30:
                market_conditions["market_sentiment"] = "fearful"
                market_conditions["risk_level"] = "high"
            elif vix_level < 15:
                market_conditions["market_sentiment"] = "greedy"
                market_conditions["risk_level"] = "low"
        
        return market_conditions
        
    except Exception as e:
        logger.error(f"Market analysis error: {e}")
        return {
            "market_sentiment": "unknown",
            "risk_level": "unknown",
            "error": str(e)
        }

def generate_recommendations(current_allocation, market_analysis):
    """Generate investment recommendations."""
    recommendations = {
        "actions": [],
        "adjustments": [],
        "holdings": [],
        "cash_action": "maintain",
        "risk_adjustment": "none"
    }
    
    # Analyze current allocation vs target
    target_allocation = {
        "VOO": 0.45,
        "VGT": 0.20,
        "XLP": 0.15,
        "XLU": 0.10,
        "FXI": 0.05,
        "KBA": 0.05
    }
    
    # Check for allocation drift
    for symbol, target_weight in target_allocation.items():
        if symbol in current_allocation:
            current_weight = current_allocation[symbol]["weight"]
            drift = current_weight - target_weight
            
            if abs(drift) > 0.02:  # More than 2% drift
                action = "reduce" if drift > 0 else "increase"
                amount = abs(drift) * portfolio_data.portfolio_value
                
                recommendations["adjustments"].append({
                    "symbol": symbol,
                    "action": action,
                    "amount": amount,
                    "current_weight": current_weight,
                    "target_weight": target_weight,
                    "drift": drift
                })
    
    # Market condition-based recommendations
    sentiment = market_analysis.get("market_sentiment", "neutral")
    risk_level = market_analysis.get("risk_level", "medium")
    
    if sentiment == "fearful" or risk_level == "high":
        recommendations["cash_action"] = "increase"
        recommendations["risk_adjustment"] = "reduce_equity"
        
        recommendations["actions"].append({
            "type": "defensive",
            "message": "Market fearful - increase defensive holdings",
            "details": "Consider increasing XLP and XLU, reducing VGT"
        })
    
    elif sentiment == "greedy" or risk_level == "low":
        recommendations["cash_action"] = "reduce"
        recommendations["risk_adjustment"] = "increase_equity"
        
        recommendations["actions"].append({
            "type": "aggressive",
            "message": "Market greedy - increase growth holdings",
            "details": "Consider increasing VGT, reducing XLP and XLU"
        })
    
    # China exposure check
    china_exposure = sum(
        current_allocation[symbol]["weight"] 
        for symbol in ["FXI", "KBA"] 
        if symbol in current_allocation
    )
    
    if china_exposure > 0.25:
        recommendations["actions"].append({
            "type": "risk_management",
            "message": "China exposure above 25% limit",
            "details": f"Current China exposure: {china_exposure*100:.1f}%. Consider reducing."
        })
    
    # Generate summary
    if not recommendations["actions"] and not recommendations["adjustments"]:
        recommendations["actions"].append({
            "type": "maintain",
            "message": "Portfolio well-balanced, maintain current allocation",
            "details": "No significant adjustments needed this week"
        })
    
    return recommendations

# Create templates directory and HTML file
os.makedirs("web_dashboard/templates", exist_ok=True)

# Create HTML template
html_template = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ETF Portfolio Manager Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/luxon"></script>
    <script src="https://cdn.jsdelivr.net/npm/chartjs-adapter-luxon"></script>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        
        .container {
            max-width: 1400px;
            margin: 0 auto;
        }
        
        .header {
            background: white;
            border-radius: 15px;
            padding: 25px;
            margin-bottom: 20px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        }
        
        .header h1 {
            color: #333;
            font-size: 28px;
            margin-bottom: 10px;
        }
        
        .header .subtitle {
            color: #666;
            font-size: 16px;
        }
        
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }
        
        .stat-card {
            background: white;
            border-radius: 12px;
            padding: 20px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.08);
            transition: transform 0.3s ease;
        }
        
        .stat-card:hover {
            transform: translateY(-5px);
        }
        
        .stat-card h3 {
            color: #666;
            font-size: 14px;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 10px;
        }
        
        .stat-card .value {
            font-size: 32px;
            font-weight: bold;
            color: #333;
        }
        
        .stat-card .change {
            font-size: 14px;
            margin-top: 5px;
        }
        
        .change.positive {
            color: #10b981;
        }
        
        .change.negative {
            color: #ef4444;
        }
        
        .charts-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }
        
        .chart-container {
            background: white;
            border-radius: 12px;
            padding: 20px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.08);
        }
        
        .chart-container h2 {
            color: #333;
            font-size: 18px;
            margin-bottom: 15px;
            padding-bottom: 10px;
            border-bottom: 2px solid #f3f4f6;
        }
        
        .alerts-container {
            background: white;
            border-radius: 12px;
            padding: 20px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.08);
            margin-bottom: 20px;
        }
        
        .alert {
            padding: 12px 15px;
            margin-bottom: 10px;
            border-radius: 8px;
            border-left: 4px solid;
        }
        
        .alert.high {
            background: #fee2e2;
            border-left-color: #ef4444;
        }
        
        .alert.medium {
            background: #fef3c7;
            border-left-color: #f59e0b;
        }
        
        .alert.low {
            background: #d1fae5;
            border-left-color: #10b981;
        }
        
        .alert .time {
            font-size: 12px;
            color: #666;
            margin-top: 5px;
        }
        
        .controls {
            display: flex;
            gap: 10px;
            margin-top: 20px;
        }
        
        .btn {
            padding: 10px 20px;
            border: none;
            border-radius: 8px;
            background: #667eea;
            color: white;
            font-weight: 600;
            cursor: pointer;
            transition: background 0.3s ease;
        }
        
        .btn:hover {
            background: #5a67d8;
        }
        
        .btn.secondary {
            background: #9ca3af;
        }
        
        .btn.secondary:hover {
            background: #6b7280;
        }
        
        .last-update {
            color: #666;
            font-size: 14px;
            margin-top: 10px;
        }
        
        @media (max-width: 768px) {
            .charts-grid {
                grid-template-columns: 1fr;
            }
            
            .stat-card .value {
                font-size: 24px;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📈 ETF Portfolio Manager Dashboard</h1>
            <div class="subtitle">Real-time monitoring and optimization for 15% annual return target</div>
            <div class="controls">
                <button class="btn" onclick="updateData()">🔄 Update Now</button>
                <button class="btn secondary" onclick="runWeeklyScan()">📊 Weekly Scan</button>
                <button class="btn secondary" onclick="runBacktest()">🧪 Run Backtest</button>
            </div>
            <div class="last-update" id="lastUpdate">Last update: Loading...</div>
        </div>
        
        <div class="stats-grid" id="statsGrid">
            <!-- Stats will be populated by JavaScript -->
        </div>
        
        <div class="charts-grid">
            <div class="chart-container">
                <h2>Portfolio Allocation</h2>
                <canvas id="allocationChart"></canvas>
            </div>
            
            <div class="chart-container">
                <h2>Portfolio Performance</h2>
                <canvas id="performanceChart"></canvas>
            </div>
            
            <div class="chart-container">
                <h2>ETF Returns Comparison</h2>
                <canvas id="returnsChart"></canvas>
            </div>
            
            <div class="chart-container">
                <h2>Risk Metrics</h2>
                <canvas id="riskChart"></canvas>
            </div>
        </div>
        
        <div class="alerts-container">
            <h2>🔔 Alerts & Notifications</h2>
            <div id="alertsList">
                <!-- Alerts will be populated by JavaScript -->
            </div>
        </div>
    </div>
    
    <script>
        // Charts
        let allocationChart, performanceChart, returnsChart, riskChart;
        
        // Initialize charts
        function initCharts() {
            // Allocation Chart (Pie)
            const allocationCtx = document.getElementById('allocationChart').getContext('2d');
            allocationChart = new Chart(allocationCtx, {
                type: 'pie',
                data: {
                    labels: [],
                    datasets: [{
                        data: [],
                        backgroundColor: [
                            '#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899'
                        ]
                    }]
                },
                options: {
                    responsive: true,
                    plugins: {
                        legend: {
                            position: 'right',
                        },
                        tooltip: {
                            callbacks: {
                                label: function(context) {
                                    const label = context.label || '';
                                    const value = context.raw || 0;
                                    const percentage = context.parsed || 0;
                                    return `${label}: $${value.toLocaleString()} (${(percentage*100).toFixed(1)}%)`;
                                }
                            }
                        }
                    }
                }
            });
            
            // Performance Chart (Line)
            const performanceCtx = document.getElementById('performanceChart').getContext('2d');
            performanceChart = new Chart(performanceCtx, {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [{
                        label: 'Portfolio Value',
                        data: [],
                        borderColor: '#3b82f6',
                        backgroundColor: 'rgba(59, 130, 246, 0.1)',
                        fill: true,
                        tension: 0.4
                    }]
                },
                options: {
                    responsive: true,
                    scales: {
                        x: {
                            type: 'time',
                            time: {
                                unit: 'day'
                            }
                        },
                        y: {
                            beginAtZero: false,
                            ticks: {
                                callback: function(value) {
                                    return '$' + value.toLocaleString();
                                }
                            }
                        }
                    },
                    plugins: {
                        tooltip: {
                            callbacks: {
                                label: function(context) {
                                    return `$${context.parsed.y.toLocaleString()}`;
                                }
                            }
                        }
                    }
                }
            });
            
            // Returns Chart (Bar)
            const returnsCtx = document.getElementById('returnsChart').getContext('2d');
            returnsChart = new Chart(returnsCtx, {
                type: 'bar',
                data: {
                    labels: [],
                    datasets: [{
                        label: 'Annual Return',
                        data: [],
                        backgroundColor: function(context) {
                            const value = context.raw;
                            return value >= 0 ? '#10b981' : '#ef4444';
                        }
                    }]
                },
                options: {
                    responsive: true,
                    scales: {
                        y: {
                            ticks: {
                                callback: function(value) {
                                    return (value * 100).toFixed(1) + '%';
                                }
                            }
                        }
                    },
                    plugins: {
                        tooltip: {
                            callbacks: {
                                label: function(context) {
                                    return (context.raw * 100).toFixed(2) + '%';
                                }
                            }
                        }
                    }
                }
            });
            
            // Risk Chart (Radar)
            const riskCtx = document.getElementById('riskChart').getContext('2d');
            riskChart = new Chart(riskCtx, {
                type: 'radar',
                data: {
                    labels: ['Return', 'Volatility', 'Sharpe Ratio', 'Max Drawdown', 'Win Rate'],
                    datasets: [{
                        label: 'Portfolio Metrics',
                        data: [0, 0, 0, 0, 0],
                        backgroundColor: 'rgba(59, 130, 246, 0.2)',
                        borderColor: '#3b82f6',
                        pointBackgroundColor: '#3b82f6'
                    }]
                },
                options: {
                    responsive: true,
                    scales: {
                        r: {
                            beginAtZero: true,
                            ticks: {
                                display: false
                            }
                        }
                    }
                }
            });
        }
        
        // Update dashboard data
        async function updateData() {
            try {
                const response = await fetch('/api/dashboard');
                const data = await response.json();
                
                updateStats(data);
                updateCharts(data);
                updateAlerts(data.alerts);
                updateLastUpdate(data.last_update);
                
            } catch (error) {
                console.error('Error updating data:', error);
                showAlert('Error updating data. Please try again.', 'high');
            }
        }
        
        // Update statistics
        function updateStats(data) {
            const statsGrid = document.getElementById('statsGrid');
            const metrics = data.metrics || {};
            
            const stats = [
                {
                    title: 'Portfolio Value',
                    value: `$${data.portfolio_value.toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2})}`,
                    change: metrics.annual_return ? `${(metrics.annual_return * 100).toFixed(2)}%` : 'N/A',
                    changeClass: metrics.annual_return >= 0 ? 'positive' : 'negative'
                },
                {
                    title: 'Annual Return',
                    value: metrics.annual_return ? `${(metrics.annual_return * 100).toFixed(2)}%` : 'N/A',
                    change: 'Target: 15.00%',
                    changeClass: metrics.annual_return >= 0.15 ? 'positive' : 'negative'
                },
                {
                    title: 'Sharpe Ratio',
                    value: metrics.sharpe_ratio ? metrics.sharpe_ratio.toFixed(3) : 'N/A',
                    change: metrics.sharpe_ratio >= 0.7 ? 'Good' : 'Needs improvement',
                    changeClass: metrics.sharpe_ratio >= 0.7 ? 'positive' : 'negative'
                },
                {
                    title: 'Max Drawdown',
                    value: metrics.max_drawdown ? `${(metrics.max_drawdown * 100).toFixed(2)}%` : 'N/A',
                    change: 'Limit: -20.00%',
                    changeClass: Math.abs(metrics.max_drawdown) <= 0.2 ? 'positive' : 'negative'
                },
                {
                    title: 'Volatility',
                    value: metrics.volatility ? `${(metrics.volatility * 100).toFixed(2)}%` : 'N/A',
                    change: 'Target: 18.00%',
                    changeClass: metrics.volatility <= 0.18 ? 'positive' : 'negative'
                },
                {
                    title: 'Active Alerts',
                    value: data.alerts ? data.alerts.length : 0,
                    change: data.alerts && data.alerts.length > 0 ? 'Needs attention' : 'All clear',
                    changeClass: data.alerts && data.alerts.length > 0 ? 'negative' : 'positive'
                }
            ];
            
            statsGrid.innerHTML = stats.map(stat => `
                <div class="stat-card">
                    <h3>${stat.title}</h3>
                    <div class="value">${stat.value}</div>
                    <div class="change ${stat.changeClass}">${stat.change}</div>
                </div>
            `).join('');
        }
        
        // Update charts
        function updateCharts(data) {
            // Update allocation chart
            const allocationLabels = Object.keys(data.allocation || {});
            const allocationValues = allocationLabels.map(symbol => data.allocation[symbol].value);
            
            allocationChart.data.labels = allocationLabels;
            allocationChart.data.datasets[0].data = allocationValues;
            allocationChart.update();
            
            // Update performance chart
            const performanceData = data.performance_history || [];
            performanceChart.data.labels = performanceData.map(point => point.date);
            performanceChart.data.datasets[0].data = performanceData.map(point => point.value);
            performanceChart.update();
            
            // Update returns chart
            returnsChart.data.labels = allocationLabels;
            returnsChart.data.datasets[0].data = allocationLabels.map(symbol => 
                data.allocation[symbol].return || 0
            );
            returnsChart.update();
            
            // Update risk chart
            const metrics = data.metrics || {};
            riskChart.data.datasets[0].data = [
                metrics.annual_return || 0,
                metrics.volatility || 0,
                metrics.sharpe_ratio || 0,
                Math.abs(metrics.max_drawdown) || 0,
                metrics.win_rate || 0.5
            ];
            riskChart.update();
        }
        
        // Update alerts
        function updateAlerts(alerts) {
            const alertsList = document.getElementById('alertsList');
            
            if (!alerts || alerts.length === 0) {
                alertsList.innerHTML = '<div class="alert low">No active alerts. Portfolio is healthy.</div>';
                return;
            }
            
            alertsList.innerHTML = alerts.map(alert => `
                <div class="alert ${alert.severity || 'medium'}">
                    <strong>${alert.type === 'price_alert' ? '📈 ' : '📊 '}${alert.message}</strong>
                    <div class="time">${formatTime(alert.timestamp)}</div>
                </div>
            `).join('');
        }
        
        // Update last update time
        function updateLastUpdate(timestamp) {
            const lastUpdate = document.getElementById('lastUpdate');
            if (timestamp) {
                const date = new Date(timestamp);
                lastUpdate.textContent = `Last update: ${date.toLocaleTimeString()}`;
            }
        }
        
        // Format time
        function formatTime(timestamp) {
            const date = new Date(timestamp);
            return date.toLocaleString();
        }
        
        // Show temporary alert
        function showAlert(message, severity = 'medium') {
            const alertsList = document.getElementById('alertsList');
            const alertDiv = document.createElement('div');
            alertDiv.className = `alert ${severity}`;
            alertDiv.innerHTML = `
                <strong>${message}</strong>
                <div class="time">Just now</div>
            `;
            
            alertsList.insertBefore(alertDiv, alertsList.firstChild);
            
            // Remove after 5 seconds
            setTimeout(() => {
                if (alertDiv.parentNode === alertsList) {
                    alertsList.removeChild(alertDiv);
                }
            }, 5000);
        }
        
        // Run weekly scan
        async function runWeeklyScan() {
            try {
                showAlert('Running weekly portfolio scan...', 'low');
                
                const response = await fetch('/api/weekly_scan', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    }
                });
                
                const result = await response.json();
                
                if (result.recommendations) {
                    const recs = result.recommendations;
                    let message = 'Weekly scan complete. ';
                    
                    if (recs.actions && recs.actions.length > 0) {
                        message += `Found ${recs.actions.length} recommended actions.`;
                        recs.actions.forEach(action => {
                            showAlert(`${action.type}: ${action.message}`, 'medium');
                        });
                    } else {
                        message += 'No significant adjustments needed.';
                    }
                    
                    showAlert(message, 'low');
                }
                
            } catch (error) {
                console.error('Error running weekly scan:', error);
                showAlert('Error running weekly scan. Please try again.', 'high');
            }
        }
        
        // Run backtest
        async function runBacktest() {
            try {
                showAlert('Running 5-year backtest...', 'low');
                
                const response = await fetch('/api/backtest', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        years: 5,
                        initial_capital: 100000
                    })
                });
                
                const result = await response.json();
                
                showAlert(`Backtest complete: ${(result.annual_return * 100).toFixed(2)}% annual return`, 'low');
                
            } catch (error) {
                console.error('Error running backtest:', error);
                showAlert('Error running backtest. Please try again.', 'high');
            }
        }
        
        // Manual update
        async function manualUpdate() {
            try {
                const response = await fetch('/api/update', {
                    method: 'POST'
                });
                
                const result = await response.json();
                showAlert('Portfolio data updated manually', 'low');
                updateData();
                
            } catch (error) {
                console.error('Error updating manually:', error);
                showAlert('Error updating manually. Please try again.', 'high');
            }
        }
        
        // Initialize on load
        document.addEventListener('DOMContentLoaded', function() {
            initCharts();
            updateData();
            
            // Auto-update every 60 seconds
            set
            // Auto-update every 60 seconds
            setInterval(updateData, 60000);
        });
    </script>
</body>
</html>
'''

# Create template file
with open("web_dashboard/templates/dashboard.html", "w") as f:
    f.write(html_template)

if __name__ == '__main__':
    print("Starting ETF Portfolio Manager Dashboard...")
    print("Dashboard available at: http://localhost:5000")
    print("Press Ctrl+C to stop")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
