# ETF Portfolio Optimizer 📊

An advanced ETF portfolio optimization system focused on achieving 5-year average 15% annual returns through strategic allocation across sectors with lowest management fees.

## 🎯 Investment Philosophy

### Core Principles
1. **ETF-Only**: No individual stocks, only diversified ETFs
2. **Lowest Fees**: Prioritize ETFs with minimal expense ratios
3. **Sector Diversification**: Spread across defensive and growth sectors
4. **5-Year Horizon**: Optimize for medium-term (5-year) performance
5. **15% Target**: Aim for 15% average annual return

### Target Sectors
- **SP500 Core** (Foundation)
- **Technology** (Growth)
- **Consumer Staples** (Defensive)
- **Utilities** (Stability + Dividends)
- **China Markets** (Growth + Diversification)
- **Consumer Discretionary** (Cyclical growth)

## 📈 Recommended ETF Universe

| Sector | Primary ETF | Ticker | Expense Ratio | Alternative | Why Chosen |
|--------|-------------|--------|---------------|-------------|------------|
| **SP500 Core** | Vanguard S&P 500 ETF | VOO | 0.03% | SPY (0.09%) | Lowest fee SP500 ETF |
| **Technology** | Vanguard Info Tech ETF | VGT | 0.10% | XLK (0.10%) | Pure tech, low fee |
| **Consumer Staples** | Consumer Staples SPDR | XLP | 0.10% | VDC (0.10%) | Defensive, dividends |
| **Utilities** | Utilities SPDR | XLU | 0.10% | VPU (0.10%) | Stable, dividend income |
| **China Large-Cap** | iShares China Large-Cap | FXI | 0.74% | MCHI (0.57%) | Large-cap China exposure |
| **China A-Shares** | KraneShares MSCI China A | KBA | 0.79% | ASHR (0.65%) | Direct A-share access |
| **Consumer Discretionary** | Consumer Disc. SPDR | XLY | 0.10% | VCR (0.10%) | Cyclical growth |

## 🏗️ System Architecture

```
etf-portfolio-optimizer/
├── data/                    # Historical and optimized data
├── src/
│   ├── etf_screener/       # ETF screening and selection
│   ├── portfolio_optimizer/ # MPT and risk optimization
│   ├── backtest_engine/    # 5-year historical backtesting
│   ├── performance_analyzer/ # Risk-adjusted metrics
│   └── report_generator/   # Optimization reports
├── config/                  # Optimization parameters
├── notebooks/              # Jupyter analysis notebooks
└── outputs/                # Optimization results
```

## 🔧 Core Features

### 1. ETF Screener & Selector
- Screen ETFs by sector, expense ratio, AUM, liquidity
- Automatically select lowest-fee options
- Validate tracking error and historical performance

### 2. Modern Portfolio Theory (MPT) Optimizer
- Mean-variance optimization
- Efficient frontier calculation
- Risk-adjusted return maximization
- Constraint handling (min/max allocations)

### 3. 5-Year Backtesting Engine
- Historical performance simulation (2019-2024)
- Dividend reinvestment
- Transaction cost modeling
- Rebalancing strategy testing

### 4. Risk Management
- Sharpe ratio optimization
- Maximum drawdown control
- Volatility targeting
- Correlation analysis

### 5. 15% Return Target Strategy
- Dynamic allocation adjustment
- Sector rotation signals
- Market regime detection
- Risk-on/risk-off positioning

## 📊 Optimization Methodology

### Objective Function
Maximize: `Expected Return - λ × Risk`

Where:
- **Expected Return**: 5-year projected annual return
- **Risk**: Portfolio volatility (standard deviation)
- **λ**: Risk aversion parameter (adjustable)

### Constraints
- Sum of weights = 100%
- Minimum allocation per ETF: 5% (for diversification)
- Maximum allocation per ETF: 30% (risk control)
- Maximum China exposure: 25% (country risk limit)
- Minimum SP500 core: 40% (foundation)

### Optimization Process
1. **Data Collection**: 5 years of daily returns
2. **Correlation Analysis**: Inter-ETF relationships
3. **Return Forecasting**: Historical + momentum factors
4. **Efficient Frontier**: Calculate optimal portfolios
5. **15% Target Selection**: Choose portfolio targeting 15% return
6. **Risk Assessment**: Validate risk metrics
7. **Rebalancing Plan**: Create actionable allocation

## 🚀 Getting Started

### Prerequisites
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Configuration
Edit `config/optimization.yaml`:
```yaml
target_return: 0.15  # 15% annual
time_horizon: 5      # years
risk_free_rate: 0.02 # 2%
max_drawdown: 0.20   # 20% maximum
min_expense_ratio: 0.01  # Maximum 1% fee
```

### Run Optimization
```bash
# Run complete optimization
python src/main.py --optimize

# Backtest specific allocation
python src/main.py --backtest --allocation "VOO:50,VGT:20,XLP:10,XLU:10,FXI:5,KBA:5"

# Generate report
python src/main.py --report
```

## 📈 Expected Output

### Optimization Report Includes:
1. **Recommended Allocation** with exact percentages
2. **Expected Return & Risk** projections
3. **5-Year Backtest Results** with historical performance
4. **Risk Metrics**: Sharpe, Sortino, Max Drawdown
5. **Sector Exposure Analysis**
6. **Rebalancing Calendar** and triggers
7. **Alternative Scenarios** (bull/bear markets)

### Sample Allocation Target:
```
TARGET PORTFOLIO (15% Annual Return Target)
===========================================
VOO (SP500 Core):       45%   | Expense: 0.03%
VGT (Technology):       20%   | Expense: 0.10%
XLP (Consumer Staples): 15%   | Expense: 0.10%
XLU (Utilities):        10%   | Expense: 0.10%
FXI (China Large):       5%   | Expense: 0.74%
KBA (China A-Shares):    5%   | Expense: 0.79%
-------------------------------------------
Total Expenses:         0.21% | Target Return: 15.0%
Expected Volatility:   18.2%  | Sharpe Ratio: 0.71
```

## 🔄 Rebalancing Strategy

### Triggers:
1. **Quarterly Calendar**: Rebalance every 3 months
2. **Allocation Drift**: >5% deviation from target
3. **Market Regime Change**: Significant trend shift
4. **Risk Threshold**: Volatility exceeds 25%

### Rules:
- Maintain core SP500 allocation (40-50%)
- Adjust tech exposure based on momentum
- Increase defensive sectors in bear markets
- Trim winners, add to laggards (contrarian)

## 📅 Development Roadmap

### Phase 1: Foundation (Week 1)
- [x] ETF data collection and screening
- [ ] Basic portfolio optimization
- [ ] 5-year backtesting framework

### Phase 2: Advanced Optimization (Week 2)
- [ ] Mean-variance optimization with constraints
- [ ] Risk-adjusted return maximization
- [ ] Monte Carlo simulation

### Phase 3: Strategy Enhancement (Week 3)
- [ ] Dynamic asset allocation
- [ ] Market regime detection
- [ ] Tactical overlays

### Phase 4: Production & Monitoring (Week 4)
- [ ] Automated reporting
- [ ] Performance tracking
- [ ] Alert system

## ⚠️ Risk Considerations

### Key Risks:
1. **Market Risk**: General market declines
2. **China Risk**: Geopolitical and regulatory
3. **Sector Risk**: Technology cyclicality
4. **Currency Risk**: USD/HKD/CNY fluctuations
5. **Tracking Error**: ETF vs. index performance

### Mitigations:
- Diversification across sectors and regions
- Defensive allocation (staples, utilities)
- Regular rebalancing
- Stop-loss mechanisms in backtesting

## 📝 Disclaimer

**This is an optimization tool, not financial advice.** 
- Past performance does not guarantee future results
- All investments carry risk
- 15% return target is aggressive and not guaranteed
- Consult with a qualified financial advisor

---

*Optimized for @netxeye's ETF portfolio with 5-year 15% return target*