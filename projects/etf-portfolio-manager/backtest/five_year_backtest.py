#!/usr/bin/env python3
"""
5-Year Backtesting Engine for ETF Portfolio
Complete historical simulation with dividends, rebalancing, and transaction costs.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FiveYearBacktest:
    """Complete 5-year backtesting engine."""
    
    def __init__(self, initial_capital: float = 100000):
        """
        Initialize backtest engine.
        
        Args:
            initial_capital: Initial investment amount (default $100,000)
        """
        self.initial_capital = initial_capital
        self.results = {}
        
        # Backtest parameters
        self.params = {
            "transaction_cost": 0.001,      # 0.1% transaction cost
            "dividend_tax_rate": 0.15,      # 15% dividend tax
            "rebalancing_frequency": 90,    # Rebalance every 90 days
            "slippage": 0.0005,             # 0.05% slippage
            "start_date": "2019-01-01",
            "end_date": "2024-01-01"
        }
        
        logger.info(f"5-Year Backtest Engine initialized: ${initial_capital:,} initial capital")
    
    def load_historical_data(self, symbols: List[str]) -> Dict:
        """
        Load 5 years of historical data for ETFs.
        
        Args:
            symbols: List of ETF symbols
            
        Returns:
            Dictionary with OHLCV and dividend data
        """
        logger.info(f"Loading 5-year historical data for {len(symbols)} ETFs...")
        
        import yfinance as yf
        
        historical_data = {}
        
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                
                # Get price data
                hist = ticker.history(
                    start=self.params["start_date"],
                    end=self.params["end_date"],
                    interval="1d"
                )
                
                # Get dividend data
                dividends = ticker.dividends
                if not dividends.empty:
                    dividends = dividends[
                        (dividends.index >= self.params["start_date"]) & 
                        (dividends.index <= self.params["end_date"])
                    ]
                
                # Get splits
                splits = ticker.splits
                
                historical_data[symbol] = {
                    "prices": hist,
                    "dividends": dividends,
                    "splits": splits,
                    "info": ticker.info
                }
                
                logger.info(f"  {symbol}: {len(hist)} trading days, "
                          f"{len(dividends)} dividend payments")
                
            except Exception as e:
                logger.error(f"Error loading data for {symbol}: {e}")
                historical_data[symbol] = {"error": str(e)}
        
        return historical_data
    
    def run_backtest(self, allocation: Dict, historical_data: Dict) -> Dict:
        """
        Run complete 5-year backtest.
        
        Args:
            allocation: Portfolio allocation {symbol: weight}
            historical_data: Historical price and dividend data
            
        Returns:
            Backtest results
        """
        logger.info("Running 5-year backtest...")
        
        # Initialize portfolio
        portfolio = self._initialize_portfolio(allocation)
        
        # Get common date range
        common_dates = self._get_common_dates(historical_data)
        
        if len(common_dates) < 100:
            logger.error("Insufficient common trading days for backtest")
            return {}
        
        # Run simulation day by day
        results = self._run_daily_simulation(portfolio, allocation, historical_data, common_dates)
        
        # Calculate performance metrics
        metrics = self._calculate_performance_metrics(results)
        
        # Generate detailed reports
        reports = self._generate_backtest_reports(results, metrics)
        
        self.results = {
            "backtest_results": results,
            "performance_metrics": metrics,
            "reports": reports,
            "allocation": allocation,
            "parameters": self.params
        }
        
        logger.info(f"Backtest completed: {metrics.get('total_return', 0)*100:.2f}% total return")
        return self.results
    
    def _initialize_portfolio(self, allocation: Dict) -> Dict:
        """Initialize portfolio with initial capital."""
        portfolio = {
            "cash": self.initial_capital,
            "positions": {},
            "total_value": self.initial_capital,
            "transaction_history": [],
            "dividend_history": [],
            "daily_values": [],
            "dates": []
        }
        
        # Calculate initial positions
        for symbol, weight in allocation.items():
            portfolio["positions"][symbol] = {
                "weight_target": weight,
                "shares": 0,
                "cost_basis": 0,
                "current_value": 0
            }
        
        return portfolio
    
    def _get_common_dates(self, historical_data: Dict) -> pd.DatetimeIndex:
        """Get common trading dates across all ETFs."""
        common_dates = None
        
        for symbol, data in historical_data.items():
            if "prices" in data and not data["prices"].empty:
                dates = data["prices"].index
                if common_dates is None:
                    common_dates = dates
                else:
                    common_dates = common_dates.intersection(dates)
        
        return common_dates if common_dates is not None else pd.DatetimeIndex([])
    
    def _run_daily_simulation(self, portfolio: Dict, allocation: Dict, 
                            historical_data: Dict, dates: pd.DatetimeIndex) -> Dict:
        """Run daily simulation over 5-year period."""
        logger.info(f"Simulating {len(dates)} trading days...")
        
        results = {
            "dates": [],
            "portfolio_values": [],
            "cash_balances": [],
            "position_values": {},
            "transactions": [],
            "dividends": [],
            "rebalancing_events": []
        }
        
        # Initialize position values dictionary
        for symbol in allocation.keys():
            results["position_values"][symbol] = []
        
        last_rebalance_date = None
        
        for i, date in enumerate(dates):
            # Update portfolio value for this day
            daily_result = self._process_daily_update(
                portfolio, allocation, historical_data, date, 
                last_rebalance_date, i, len(dates)
            )
            
            # Check if rebalancing needed
            if self._should_rebalance(portfolio, allocation, date, last_rebalance_date):
                self._rebalance_portfolio(portfolio, allocation, historical_data, date)
                last_rebalance_date = date
                results["rebalancing_events"].append({
                    "date": date,
                    "portfolio_value": portfolio["total_value"],
                    "transactions": portfolio["transaction_history"][-5:]  # Last 5 transactions
                })
            
            # Record daily results
            results["dates"].append(date)
            results["portfolio_values"].append(portfolio["total_value"])
            results["cash_balances"].append(portfolio["cash"])
            
            for symbol in allocation.keys():
                if symbol in portfolio["positions"]:
                    results["position_values"][symbol].append(
                        portfolio["positions"][symbol]["current_value"]
                    )
            
            # Record transactions and dividends from this day
            if portfolio["transaction_history"]:
                recent_tx = [tx for tx in portfolio["transaction_history"] 
                           if tx["date"] == date]
                results["transactions"].extend(recent_tx)
            
            if portfolio["dividend_history"]:
                recent_div = [div for div in portfolio["dividend_history"]
                            if div["date"] == date]
                results["dividends"].extend(recent_div)
            
            # Progress logging
            if i % 252 == 0:  # Every year
                logger.info(f"  Year {i//252 + 1}: ${portfolio['total_value']:,.2f}")
        
        return results
    
    def _process_daily_update(self, portfolio: Dict, allocation: Dict, 
                            historical_data: Dict, date: datetime,
                            last_rebalance_date: datetime, day_index: int, 
                            total_days: int) -> Dict:
        """Process daily portfolio update."""
        daily_total = portfolio["cash"]
        
        for symbol, position in portfolio["positions"].items():
            if symbol in historical_data and "prices" in historical_data[symbol]:
                prices = historical_data[symbol]["prices"]
                
                if date in prices.index:
                    price = prices.loc[date, "Close"]
                    
                    # Update position value
                    position["current_value"] = position["shares"] * price
                    daily_total += position["current_value"]
                    
                    # Process dividends
                    if "dividends" in historical_data[symbol]:
                        dividends = historical_data[symbol]["dividends"]
                        if date in dividends.index:
                            dividend_per_share = dividends.loc[date]
                            total_dividend = position["shares"] * dividend_per_share
                            after_tax = total_dividend * (1 - self.params["dividend_tax_rate"])
                            
                            portfolio["cash"] += after_tax
                            portfolio["dividend_history"].append({
                                "date": date,
                                "symbol": symbol,
                                "amount": total_dividend,
                                "after_tax": after_tax,
                                "shares": position["shares"]
                            })
        
        portfolio["total_value"] = daily_total
        portfolio["daily_values"].append(daily_total)
        portfolio["dates"].append(date)
        
        return {"date": date, "portfolio_value": daily_total}
    
    def _should_rebalance(self, portfolio: Dict, allocation: Dict, 
                         current_date: datetime, last_rebalance_date: datetime) -> bool:
        """Check if portfolio should be rebalanced."""
        # Time-based rebalancing
        if last_rebalance_date is None:
            return True
        
        days_since_rebalance = (current_date - last_rebalance_date).days
        if days_since_rebalance >= self.params["rebalancing_frequency"]:
            return True
        
        # Drift-based rebalancing
        total_value = portfolio["total_value"]
        max_drift = 0
        
        for symbol, position in portfolio["positions"].items():
            if total_value > 0:
                current_weight = position["current_value"] / total_value
                target_weight = allocation.get(symbol, 0)
                drift = abs(current_weight - target_weight)
                max_drift = max(max_drift, drift)
        
        # Rebalance if any position drifts more than 5%
        if max_drift > 0.05:
            return True
        
        return False
    
    def _rebalance_portfolio(self, portfolio: Dict, allocation: Dict, 
                           historical_data: Dict, date: datetime):
        """Rebalance portfolio to target allocation."""
        logger.debug(f"Rebalancing portfolio on {date.strftime('%Y-%m-%d')}")
        
        total_value = portfolio["total_value"]
        
        for symbol, position in portfolio["positions"].items():
            if symbol in historical_data and "prices" in historical_data[symbol]:
                prices = historical_data[symbol]["prices"]
                
                if date in prices.index:
                    price = prices.loc[date, "Close"]
                    target_value = total_value * allocation[symbol]
                    current_value = position["current_value"]
                    
                    # Calculate shares to buy/sell
                    shares_needed = target_value / price
                    shares_difference = shares_needed - position["shares"]
                    
                    if abs(shares_difference) > 0.001:  # Avoid tiny transactions
                        # Calculate transaction cost
                        transaction_value = abs(shares_difference) * price
                        transaction_cost = transaction_value * self.params["transaction_cost"]
                        
                        # Apply slippage
                        if shares_difference > 0:  # Buying
                            execution_price = price * (1 + self.params["slippage"])
                        else:  # Selling
                            execution_price = price * (1 - self.params["slippage"])
                        
                        # Update position
                        position["shares"] = shares_needed
                        position["current_value"] = position["shares"] * price
                        
                        # Update cash
                        cash_impact = -shares_difference * execution_price - transaction_cost
                        portfolio["cash"] += cash_impact
                        
                        # Record transaction
                        portfolio["transaction_history"].append({
                            "date": date,
                            "symbol": symbol,
                            "action": "BUY" if shares_difference > 0 else "SELL",
                            "shares": abs(shares_difference),
                            "price": execution_price,
                            "value": transaction_value,
                            "cost": transaction_cost,
                            "cash_impact": cash_impact
                        })
    
    def _calculate_performance_metrics(self, results: Dict) -> Dict:
        """Calculate comprehensive performance metrics."""
        if not results["portfolio_values"]:
            return {}
        
        portfolio_values = np.array(results["portfolio_values"])
        dates = results["dates"]
        
        # Calculate returns
        daily_returns = np.diff(portfolio_values) / portfolio_values[:-1]
        
        # Basic metrics
        total_return = (portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0]
        annual_return = (1 + total_return) ** (252 / len(daily_returns)) - 1
        
        # Volatility (annualized)
        daily_volatility = np.std(daily_returns)
        annual_volatility = daily_volatility * np.sqrt(252)
        
        # Sharpe ratio (assuming 2% risk-free rate)
        risk_free_rate = 0.02
        excess_returns = daily_returns - risk_free_rate/252
        sharpe_ratio = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
        
        # Sortino ratio (downside risk)
        downside_returns = daily_returns[daily_returns < risk_free_rate/252]
        if len(downside_returns) > 0:
            downside_std = np.std(downside_returns)
            sortino_ratio = (annual_return - risk_free_rate) / (downside_std * np.sqrt(252))
        else:
            sortino_ratio = sharpe_ratio
        
        # Maximum drawdown
        cumulative = (1 + daily_returns).cumprod()
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = np.min(drawdown)
        
        # Calmar ratio
        calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Alpha and Beta (vs SPY)
        # Note: Would need benchmark returns for accurate calculation
        
        # Win rate
        winning_days = np.sum(daily_returns > 0)
        total_days = len(daily_returns)
        win_rate = winning_days / total_days
        
        # Average win/loss
        winning_returns = daily_returns[daily_returns > 0]
        losing_returns = daily_returns[daily_returns < 0]
        avg_win = np.mean(winning_returns) if len(winning_returns) > 0 else 0
        avg_loss = np.mean(losing_returns) if len(losing_returns) > 0 else 0
        profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else 0
        
        # Transaction metrics
        total_transactions = len(results["transactions"])
        total_transaction_costs = sum(tx.get("cost", 0) for tx in results["transactions"])
        
        # Dividend metrics
        total_dividends = sum(div.get("after_tax", 0) for div in results["dividends"])
        
        metrics = {
            "total_return": total_return,
            "annual_return": annual_return,
            "annual_volatility": annual_volatility,
            "sharpe_ratio": sharpe_ratio,
            "sortino_ratio": sortino_ratio,
            "max_drawdown": max_drawdown,
            "calmar_ratio": calmar_ratio,
            "win_rate": win_rate,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "profit_factor": profit_factor,
            "total_transactions": total_transactions,
            "total_transaction_costs": total_transaction_costs,
            "total_dividends": total_dividends,
            "final_portfolio_value": portfolio_values[-1],
            "initial_portfolio_value": portfolio_values[0],
            "simulation_days": len(daily_returns),
            "start_date": dates[0] if dates else None,
            "end_date": dates[-1] if dates else None
        }
        
        return metrics
    
    def _generate_backtest_reports(self, results: Dict, metrics: Dict) -> Dict:
        """Generate detailed backtest reports."""
        reports = {
            "summary": self._generate_summary_report(metrics),
            "monthly_returns": self._calculate_monthly_returns(results),
            "yearly_returns": self._calculate_yearly_returns(results),
            "drawdown_analysis": self._analyze_drawdowns(results),
            "sector_performance": self._analyze_sector_performance(results),
            "transaction_analysis": self._analyze_transactions(results),
            "dividend_analysis": self._analyze_dividends(results)
        }
        
        return reports
    
    def _generate_summary_report(self, metrics: Dict) -> str:
        """Generate summary report."""
        report = []
        report.append("=" * 70)
        report.append("5-YEAR BACKTEST SUMMARY REPORT")
        report.append("=" * 70)
        report.append("")
        
        report.append("PERFORMANCE METRICS")
        report.append("-" * 40)
        report.append(f"Total Return:          {metrics.get('total_return', 0)*100:8.2f}%")
        report.append(f"Annualized Return:     {metrics.get('annual_return', 0)*100:8.2f}%")
        report.append(f"Annual Volatility:     {metrics.get('annual_volatility', 0)*        0)*100:8.2f}%")
        report.append(f"Sharpe Ratio:          {metrics.get('sharpe_ratio', 0):8.3f}")
        report.append(f"Sortino Ratio:         {metrics.get('sortino_ratio', 0):8.3f}")
        report.append(f"Maximum Drawdown:      {metrics.get('max_drawdown', 0)*100:8.2f}%")
        report.append(f"Calmar Ratio:          {metrics.get('calmar_ratio', 0):8.3f}")
        report.append("")
        
        report.append("TRADING METRICS")
        report.append("-" * 40)
        report.append(f"Win Rate:              {metrics.get('win_rate', 0)*100:8.2f}%")
        report.append(f"Average Win:           {metrics.get('avg_win', 0)*100:8.2f}%")
        report.append(f"Average Loss:          {metrics.get('avg_loss', 0)*100:8.2f}%")
        report.append(f"Profit Factor:         {metrics.get('profit_factor', 0):8.2f}")
        report.append(f"Total Transactions:    {metrics.get('total_transactions', 0):8d}")
        report.append(f"Transaction Costs:     ${metrics.get('total_transaction_costs', 0):8,.2f}")
        report.append(f"Total Dividends:       ${metrics.get('total_dividends', 0):8,.2f}")
        report.append("")
        
        report.append("PORTFOLIO VALUES")
        report.append("-" * 40)
        report.append(f"Initial Capital:       ${metrics.get('initial_portfolio_value', 0):12,.2f}")
        report.append(f"Final Value:           ${metrics.get('final_portfolio_value', 0):12,.2f}")
        report.append(f"Net Gain:              ${metrics.get('final_portfolio_value', 0) - metrics.get('initial_portfolio_value', 0):12,.2f}")
        report.append("")
        
        report.append("SIMULATION DETAILS")
        report.append("-" * 40)
        report.append(f"Start Date:            {metrics.get('start_date', 'N/A')}")
        report.append(f"End Date:              {metrics.get('end_date', 'N/A')}")
        report.append(f"Trading Days:          {metrics.get('simulation_days', 0):8d}")
        report.append(f"Years Simulated:       {metrics.get('simulation_days', 0)/252:.1f}")
        report.append("")
        
        # Performance assessment
        report.append("PERFORMANCE ASSESSMENT")
        report.append("-" * 40)
        annual_return = metrics.get('annual_return', 0)
        
        if annual_return >= 0.15:
            report.append("✅ EXCELLENT: Exceeded 15% annual return target")
        elif annual_return >= 0.10:
            report.append("⚠️  GOOD: Met 10% annual return, below 15% target")
        elif annual_return >= 0.05:
            report.append("⚠️  MODERATE: 5-10% annual return, consider adjustments")
        else:
            report.append("❌ POOR: Below 5% annual return, significant changes needed")
        
        max_dd = metrics.get('max_drawdown', 0)
        if abs(max_dd) <= 0.15:
            report.append("✅ GOOD: Maximum drawdown within 15% limit")
        elif abs(max_dd) <= 0.25:
            report.append("⚠️  MODERATE: Maximum drawdown 15-25%, monitor closely")
        else:
            report.append("❌ HIGH RISK: Maximum drawdown >25%, risk management needed")
        
        report.append("")
        report.append("=" * 70)
        report.append("Note: Past performance does not guarantee future results.")
        report.append("=" * 70)
        
        return "\n".join(report)
    
    def _calculate_monthly_returns(self, results: Dict) -> pd.DataFrame:
        """Calculate monthly returns."""
        if not results["dates"] or not results["portfolio_values"]:
            return pd.DataFrame()
        
        # Create DataFrame with dates and values
        df = pd.DataFrame({
            "date": results["dates"],
            "value": results["portfolio_values"]
        })
        df.set_index("date", inplace=True)
        
        # Resample to monthly
        monthly = df.resample('M').last()
        monthly["return"] = monthly["value"].pct_change()
        
        return monthly
    
    def _calculate_yearly_returns(self, results: Dict) -> pd.DataFrame:
        """Calculate yearly returns."""
        if not results["dates"] or not results["portfolio_values"]:
            return pd.DataFrame()
        
        df = pd.DataFrame({
            "date": results["dates"],
            "value": results["portfolio_values"]
        })
        df.set_index("date", inplace=True)
        
        # Resample to yearly
        yearly = df.resample('Y').last()
        yearly["return"] = yearly["value"].pct_change()
        
        return yearly
    
    def _analyze_drawdowns(self, results: Dict) -> Dict:
        """Analyze drawdown periods."""
        if not results["portfolio_values"]:
            return {}
        
        values = np.array(results["portfolio_values"])
        dates = results["dates"]
        
        # Calculate drawdowns
        cumulative = (1 + np.diff(values) / values[:-1]).cumprod()
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        
        # Find major drawdowns (>5%)
        major_drawdowns = []
        in_drawdown = False
        drawdown_start = None
        drawdown_depth = 0
        
        for i, dd in enumerate(drawdown):
            if dd < -0.05 and not in_drawdown:
                in_drawdown = True
                drawdown_start = dates[i] if i < len(dates) else None
                drawdown_depth = dd
            elif dd >= -0.05 and in_drawdown:
                in_drawdown = False
                drawdown_end = dates[i-1] if i > 0 else None
                major_drawdowns.append({
                    "start": drawdown_start,
                    "end": drawdown_end,
                    "depth": drawdown_depth,
                    "duration": (drawdown_end - drawdown_start).days if drawdown_end and drawdown_start else 0
                })
                drawdown_depth = 0
        
        # Handle ongoing drawdown
        if in_drawdown and drawdown_start:
            major_drawdowns.append({
                "start": drawdown_start,
                "end": dates[-1] if dates else None,
                "depth": drawdown_depth,
                "duration": (dates[-1] - drawdown_start).days if dates else 0,
                "ongoing": True
            })
        
        return {
            "max_drawdown": np.min(drawdown) if len(drawdown) > 0 else 0,
            "avg_drawdown": np.mean(drawdown[drawdown < 0]) if np.any(drawdown < 0) else 0,
            "major_drawdowns": major_drawdowns,
            "drawdown_dates": dates[1:],  # Skip first date (no return)
            "drawdown_values": drawdown
        }
    
    def _analyze_sector_performance(self, results: Dict) -> Dict:
        """Analyze performance by sector."""
        # This would require sector mapping for each ETF
        # For now, return placeholder
        return {
            "note": "Sector performance analysis requires sector mapping data"
        }
    
    def _analyze_transactions(self, results: Dict) -> Dict:
        """Analyze transaction patterns."""
        transactions = results.get("transactions", [])
        
        if not transactions:
            return {"total_transactions": 0}
        
        # Convert to DataFrame for analysis
        df = pd.DataFrame(transactions)
        
        analysis = {
            "total_transactions": len(transactions),
            "buy_transactions": len(df[df["action"] == "BUY"]),
            "sell_transactions": len(df[df["action"] == "SELL"]),
            "total_transaction_costs": df["cost"].sum() if "cost" in df.columns else 0,
            "avg_transaction_size": df["value"].mean() if "value" in df.columns else 0,
            "most_traded_symbol": df["symbol"].mode()[0] if not df.empty and "symbol" in df.columns else "N/A"
        }
        
        # Monthly transaction frequency
        if "date" in df.columns and not df.empty:
            df["date"] = pd.to_datetime(df["date"])
            monthly_tx = df.resample('M', on="date").size()
            analysis["monthly_transaction_frequency"] = monthly_tx.mean()
        
        return analysis
    
    def _analyze_dividends(self, results: Dict) -> Dict:
        """Analyze dividend income."""
        dividends = results.get("dividends", [])
        
        if not dividends:
            return {"total_dividends": 0}
        
        df = pd.DataFrame(dividends)
        
        analysis = {
            "total_dividends": len(dividends),
            "total_dividend_income": df["after_tax"].sum() if "after_tax" in df.columns else 0,
            "avg_dividend_per_payment": df["after_tax"].mean() if "after_tax" in df.columns else 0,
            "top_dividend_payers": df.groupby("symbol")["after_tax"].sum().nlargest(3).to_dict()
        }
        
        # Annual dividend income
        if "date" in df.columns and not df.empty:
            df["date"] = pd.to_datetime(df["date"])
            df["year"] = df["date"].dt.year
            annual_dividends = df.groupby("year")["after_tax"].sum()
            analysis["annual_dividend_income"] = annual_dividends.to_dict()
        
        return analysis
    
    def save_backtest_results(self, results: Dict, filename_prefix: str = None):
        """
        Save backtest results to files.
        
        Args:
            results: Backtest results dictionary
            filename_prefix: Prefix for output files
        """
        if filename_prefix is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename_prefix = f"backtest_results_{timestamp}"
        
        import os
        os.makedirs("backtest_outputs", exist_ok=True)
        
        # Save JSON results
        import json
        json_file = f"backtest_outputs/{filename_prefix}.json"
        with open(json_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save summary report
        if "reports" in results and "summary" in results["reports"]:
            report_file = f"backtest_outputs/{filename_prefix}_summary.txt"
            with open(report_file, 'w') as f:
                f.write(results["reports"]["summary"])
        
        # Save monthly returns CSV
        if "reports" in results and "monthly_returns" in results["reports"]:
            monthly_df = results["reports"]["monthly_returns"]
            if not monthly_df.empty:
                csv_file = f"backtest_outputs/{filename_prefix}_monthly.csv"
                monthly_df.to_csv(csv_file)
        
        logger.info(f"Backtest results saved to backtest_outputs/{filename_prefix}_*")


def main():
    """Test the backtest engine."""
    print("=" * 70)
    print("5-YEAR BACKTEST ENGINE TEST")
    print("=" * 70)
    
    # Sample allocation
    allocation = {
        "VOO": 0.45,  # SP500 Core
        "VGT": 0.20,  # Technology
        "XLP": 0.15,  # Consumer Staples
        "XLU": 0.10,  # Utilities
        "FXI": 0.05,  # China Large-Cap
        "KBA": 0.05   # China A-Shares
    }
    
    # Initialize backtest engine
    backtest = FiveYearBacktest(initial_capital=100000)
    
    # Load historical data
    print("\nLoading historical data...")
    symbols = list(allocation.keys())
    historical_data = backtest.load_historical_data(symbols)
    
    # Check data availability
    available_symbols = [s for s in symbols if "prices" in historical_data.get(s, {})]
    print(f"Data available for {len(available_symbols)}/{len(symbols)} symbols")
    
    if len(available_symbols) < 3:
        print("Insufficient data for backtest. Using synthetic data for demonstration.")
        # Create synthetic data for demonstration
        historical_data = {}
        np.random.seed(42)
        n_days = 252 * 5  # 5 years of trading days
        
        for symbol in symbols:
            # Generate synthetic price data
            dates = pd.date_range(start='2019-01-01', periods=n_days, freq='B')
            base_price = 100 if symbol == "VOO" else 50
            returns = np.random.normal(0.0005, 0.01, n_days)
            prices = base_price * (1 + returns).cumprod()
            
            historical_data[symbol] = {
                "prices": pd.DataFrame({
                    "Open": prices * 0.99,
                    "High": prices * 1.01,
                    "Low": prices * 0.98,
                    "Close": prices,
                    "Volume": np.random.randint(1000000, 5000000, n_days)
                }, index=dates)
            }
    
    # Run backtest
    print("\nRunning backtest...")
    results = backtest.run_backtest(allocation, historical_data)
    
    if results:
        # Print summary
        if "reports" in results and "summary" in results["reports"]:
            print("\n" + results["reports"]["summary"])
        
        # Save results
        backtest.save_backtest_results(results)
        
        print("\n" + "=" * 70)
        print("BACKTEST COMPLETE")
        print("=" * 70)
        print(f"Results saved to backtest_outputs/ directory")
    else:
        print("\nBacktest failed.")


if __name__ == "__main__":
    main()
