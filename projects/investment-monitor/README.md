# Investment Monitor 📈

A Python-based system for monitoring ETFs (SPY, QQQ) and Tencent stock, providing daily investment insights with a focus on achieving 15% annual returns through SP500 + Tencent portfolio strategy.

## 🎯 Objectives

- Monitor SPY, QQQ, and Tencent (0700.HK) stock prices
- Analyze market trends and technical indicators
- Generate daily investment strategy reports
- Backtest SP500 + Tencent portfolio for 15% annual returns
- Provide risk alerts and market insights

## 📊 Target Assets

1. **SPY** - SPDR S&P 500 ETF Trust
   - Tracks S&P 500 index
   - Core holding for stable growth

2. **QQQ** - Invesco QQQ Trust
   - Tracks NASDAQ-100 index
   - Technology sector exposure

3. **Tencent (0700.HK)** - Tencent Holdings Limited
   - Chinese tech giant
   - Growth potential component

## 🏗️ Architecture

```
investment-monitor/
├── data/              # Historical and real-time data
├── src/
│   ├── collectors/    # Data collection from APIs
│   ├── analyzers/     # Technical and fundamental analysis
│   ├── strategies/    # Investment strategies
│   └── reporters/     # Report generation
├── config/            # Configuration files
├── tests/             # Unit and integration tests
└── docs/              # Documentation
```

## 📈 Analysis Features

### Technical Analysis
- Moving averages (SMA, EMA)
- RSI (Relative Strength Index)
- MACD (Moving Average Convergence Divergence)
- Bollinger Bands
- Volume analysis
- Support and resistance levels

### Portfolio Strategy
- SP500 (70-80%) + Tencent (20-30%) allocation
- Dynamic rebalancing based on market conditions
- Risk-adjusted return optimization
- Drawdown analysis

### Daily Reports
- Market summary
- Asset performance
- Technical signals
- Investment recommendations
- Risk warnings

## 🎯 15% Annual Return Strategy

### Core Principles
1. **Long-term focus**: Hold through market cycles
2. **Quality assets**: SP500 blue chips + Tencent growth
3. **Disciplined allocation**: 70-80% SPY, 20-30% Tencent
4. **Contrarian opportunities**: Buy during market fear
5. **Risk management**: Stop losses and position sizing

### Implementation
- Regular portfolio rebalancing
- Dollar-cost averaging during volatility
- Technical entry/exit signals
- Fundamental valuation checks

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- Virtual environment
- API keys (Yahoo Finance, Alpha Vantage, etc.)

### Installation
```bash
# Clone and setup
cd projects/investment-monitor
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Configuration
1. Copy `config/config.example.yaml` to `config/config.yaml`
2. Add your API keys
3. Configure monitoring preferences

### Usage
```bash
# Collect data
python src/collectors/fetch_data.py

# Run analysis
python src/analyzers/daily_analysis.py

# Generate report
python src/reporters/daily_report.py
```

## 📅 Development Roadmap

### Phase 1: Foundation (Week 1)
- [ ] Basic data collection
- [ ] Simple technical indicators
- [ ] Daily report template

### Phase 2: Core Analysis (Week 2)
- [ ] Advanced technical analysis
- [ ] Portfolio backtesting
- [ ] Risk metrics calculation

### Phase 3: Automation (Week 3)
- [ ] Scheduled tasks
- [ ] Email/SMS alerts
- [ ] Web dashboard

### Phase 4: Optimization (Week 4)
- [ ] Machine learning predictions
- [ ] Advanced risk management
- [ ] Performance optimization

## 🔧 Technology Stack

- **Python**: Core programming language
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computations
- **Matplotlib/Plotly**: Data visualization
- **yfinance**: Yahoo Finance API wrapper
- **Alpha Vantage**: Alternative financial data
- **SQLite/PostgreSQL**: Data storage
- **FastAPI**: Optional web interface
- **Docker**: Containerization

## 📝 License

This project is for educational and personal use only. Not financial advice.

## ⚠️ Disclaimer

This software is for informational purposes only. Past performance does not guarantee future results. Investing involves risk, including possible loss of principal. Always conduct your own research and consult with a qualified financial advisor before making investment decisions.

---

*Created by OpenClaw for @netxeye's investment monitoring needs*