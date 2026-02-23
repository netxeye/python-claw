#!/usr/bin/env python3
"""
ETF Portfolio Optimizer - Modern Portfolio Theory implementation
for 5-year 15% annual return target with lowest fee ETFs.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Tuple, Optional
import json
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PortfolioOptimizer:
    """Modern Portfolio Theory optimizer for ETF portfolios."""
    
    def __init__(self, target_return: float = 0.15, risk_free_rate: float = 0.02):
        """
        Initialize portfolio optimizer.
        
        Args:
            target_return: Target annual return (default 15%)
            risk_free_rate: Risk-free rate (default 2%)
        """
        self.target_return = target_return
        self.risk_free_rate = risk_free_rate
        
        # Default constraints
        self.constraints = {
            "min_allocation": 0.05,  # Minimum 5% per ETF
            "max_allocation": 0.30,  # Maximum 30% per ETF
            "max_china": 0.25,       # Maximum 25% China exposure
            "min_sp500": 0.40,       # Minimum 40% SP500 core
            "max_tech": 0.25,        # Maximum 25% technology
            "max_utilities": 0.15,   # Maximum 15% utilities
        }
        
        logger.info(f"Portfolio Optimizer initialized: {target_return*100:.1f}% target return")
    
    def calculate_portfolio_stats(self, weights: np.ndarray, 
                                 expected_returns: np.ndarray, 
                                 cov_matrix: np.ndarray) -> Dict:
        """
        Calculate portfolio statistics.
        
        Args:
            weights: Portfolio weights
            expected_returns: Expected annual returns
            cov_matrix: Covariance matrix
            
        Returns:
            Dictionary with portfolio statistics
        """
        # Portfolio return
        port_return = np.sum(weights * expected_returns)
        
        # Portfolio volatility (annualized)
        port_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        
        # Sharpe ratio
        sharpe_ratio = (port_return - self.risk_free_rate) / port_volatility if port_volatility > 0 else 0
        
        # Sortino ratio (downside risk)
        downside_returns = expected_returns[expected_returns < self.risk_free_rate]
        if len(downside_returns) > 0:
            downside_std = np.std(downside_returns)
            sortino_ratio = (port_return - self.risk_free_rate) / downside_std if downside_std > 0 else 0
        else:
            sortino_ratio = sharpe_ratio
        
        return {
            "return": port_return,
            "volatility": port_volatility,
            "sharpe_ratio": sharpe_ratio,
            "sortino_ratio": sortino_ratio,
            "weights": weights
        }
    
    def portfolio_variance(self, weights: np.ndarray, cov_matrix: np.ndarray) -> float:
        """Calculate portfolio variance."""
        return np.dot(weights.T, np.dot(cov_matrix, weights))
    
    def portfolio_return(self, weights: np.ndarray, expected_returns: np.ndarray) -> float:
        """Calculate portfolio return."""
        return np.dot(weights, expected_returns)
    
    def optimize_portfolio(self, etf_data: Dict, years: int = 5) -> Dict:
        """
        Optimize portfolio allocation for target return.
        
        Args:
            etf_data: Dictionary with ETF data including returns
            years: Years of historical data to use
            
        Returns:
            Optimized portfolio allocation
        """
        logger.info(f"Optimizing portfolio for {self.target_return*100:.1f}% target return...")
        
        # Extract symbols and returns
        symbols = list(etf_data.keys())
        returns_data = []
        
        for symbol in symbols:
            if "returns" in etf_data[symbol]:
                returns_data.append(etf_data[symbol]["returns"])
            else:
                logger.warning(f"No returns data for {symbol}")
                returns_data.append(pd.Series([0.0]))
        
        # Create returns DataFrame
        returns_df = pd.DataFrame({symbol: returns for symbol, returns in zip(symbols, returns_data)})
        returns_df = returns_df.dropna()
        
        if returns_df.empty:
            logger.error("No valid returns data for optimization")
            return {}
        
        # Calculate expected returns and covariance
        expected_returns = returns_df.mean() * 252  # Annualize
        cov_matrix = returns_df.cov() * 252  # Annualize
        
        n_assets = len(symbols)
        
        # Define optimization constraints
        constraints = []
        
        # 1. Sum of weights = 1
        constraints.append({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        
        # 2. Target return constraint
        constraints.append({'type': 'eq', 'fun': lambda x: self.portfolio_return(x, expected_returns) - self.target_return})
        
        # Define bounds for each asset (min 5%, max 30%)
        bounds = [(self.constraints["min_allocation"], self.constraints["max_allocation"]) for _ in range(n_assets)]
        
        # Additional sector-specific constraints
        def china_constraint(x):
            """China exposure constraint (max 25%)."""
            china_indices = [i for i, sym in enumerate(symbols) if 'FXI' in sym or 'KBA' in sym or 'MCHI' in sym]
            china_exposure = sum(x[i] for i in china_indices)
            return self.constraints["max_china"] - china_exposure
        
        def sp500_constraint(x):
            """SP500 minimum constraint (min 40%)."""
            sp500_indices = [i for i, sym in enumerate(symbols) if 'VOO' in sym or 'SPY' in sym or 'IVV' in sym]
            sp500_exposure = sum(x[i] for i in sp500_indices) if sp500_indices else 0
            return sp500_exposure - self.constraints["min_sp500"]
        
        def tech_constraint(x):
            """Technology maximum constraint (max 25%)."""
            tech_indices = [i for i, sym in enumerate(symbols) if 'VGT' in sym or 'XLK' in sym or 'FTEC' in sym]
            tech_exposure = sum(x[i] for i in tech_indices) if tech_indices else 0
            return self.constraints["max_tech"] - tech_exposure
        
        def utilities_constraint(x):
            """Utilities maximum constraint (max 15%)."""
            utilities_indices = [i for i, sym in enumerate(symbols) if 'XLU' in sym or 'VPU' in sym or 'FUTY' in sym]
            utilities_exposure = sum(x[i] for i in utilities_indices) if utilities_indices else 0
            return self.constraints["max_utilities"] - utilities_exposure
        
        # Add sector constraints
        constraints.append({'type': 'ineq', 'fun': china_constraint})
        constraints.append({'type': 'ineq', 'fun': sp500_constraint})
        constraints.append({'type': 'ineq', 'fun': tech_constraint})
        constraints.append({'type': 'ineq', 'fun': utilities_constraint})
        
        # Initial guess (equal weights)
        initial_weights = np.array([1/n_assets] * n_assets)
        
        # Define objective function (minimize variance)
        def objective(weights):
            return self.portfolio_variance(weights, cov_matrix)
        
        # Run optimization
        try:
            result = minimize(
                fun=objective,
                x0=initial_weights,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 1000, 'ftol': 1e-9}
            )
            
            if result.success:
                optimized_weights = result.x
                
                # Calculate portfolio statistics
                stats = self.calculate_portfolio_stats(optimized_weights, expected_returns, cov_matrix)
                
                # Create allocation dictionary
                allocation = {}
                for i, symbol in enumerate(symbols):
                    allocation[symbol] = {
                        "weight": optimized_weights[i],
                        "expected_return": expected_returns[symbol],
                        "volatility": np.sqrt(cov_matrix.iloc[i, i]),
                        "expense_ratio": etf_data[symbol].get("expense_ratio", 0),
                        "aum": etf_data[symbol].get("aum", 0)
                    }
                
                # Calculate portfolio expense ratio
                portfolio_expense = sum(allocation[sym]["weight"] * allocation[sym]["expense_ratio"] 
                                      for sym in allocation)
                
                # Calculate sector exposures
                sector_exposures = self.calculate_sector_exposures(allocation)
                
                optimized_portfolio = {
                    "allocation": allocation,
                    "statistics": stats,
                    "portfolio_expense_ratio": portfolio_expense,
                    "sector_exposures": sector_exposures,
                    "optimization_success": True,
                    "symbols": symbols,
                    "constraints_used": self.constraints,
                    "target_return": self.target_return
                }
                
                logger.info(f"Optimization successful: {stats['return']*100:.2f}% return, "
                          f"{stats['volatility']*100:.2f}% volatility, "
                          f"Sharpe: {stats['sharpe_ratio']:.3f}")
                
                return optimized_portfolio
                
            else:
                logger.error(f"Optimization failed: {result.message}")
                return {"optimization_success": False, "message": result.message}
                
        except Exception as e:
            logger.error(f"Optimization error: {e}")
            return {"optimization_success": False, "error": str(e)}
    
    def calculate_sector_exposures(self, allocation: Dict) -> Dict:
        """Calculate sector exposures from allocation."""
        sector_mapping = {
            "SP500 Core": ["VOO", "SPY", "IVV"],
            "Technology": ["VGT", "XLK", "FTEC"],
            "Consumer Staples": ["XLP", "VDC", "FSTA"],
            "Utilities": ["XLU", "VPU", "FUTY"],
            "China Large-Cap": ["FXI", "MCHI", "GXC"],
            "China A-Shares": ["KBA", "ASHR", "CHIQ"],
            "Consumer Discretionary": ["XLY", "VCR", "FDIS"]
        }
        
        exposures = {}
        for sector, symbols in sector_mapping.items():
            sector_weight = 0
            for symbol in symbols:
                if symbol in allocation:
                    sector_weight += allocation[symbol]["weight"]
            if sector_weight > 0:
                exposures[sector] = sector_weight
        
        return exposures
    
    def generate_efficient_frontier(self, etf_data: Dict, n_points: int = 50) -> Dict:
        """
        Generate efficient frontier for visualization.
        
        Args:
            etf_data: ETF data with returns
            n_points: Number of points on frontier
            
        Returns:
            Efficient frontier data
        """
        logger.info("Generating efficient frontier...")
        
        symbols = list(etf_data.keys())
        returns_data = []
        
        for symbol in symbols:
            if "returns" in etf_data[symbol]:
                returns_data.append(etf_data[symbol]["returns"])
        
        returns_df = pd.DataFrame({symbol: returns for symbol, returns in zip(symbols, returns_data)})
        returns_df = returns_df.dropna()
        
        if returns_df.empty:
            return {}
        
        expected_returns = returns_df.mean() * 252
        cov_matrix = returns_df.cov() * 252
        
        n_assets = len(symbols)
        
        # Target returns for frontier
        min_return = expected_returns.min()
        max_return = expected_returns.max()
        target_returns = np.linspace(min_return, max_return * 1.2, n_points)
        
        frontier_weights = []
        frontier_returns = []
        frontier_volatilities = []
        frontier_sharpes = []
        
        for target in target_returns:
            # Constraints
            constraints = [
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
                {'type': 'eq', 'fun': lambda x: self.portfolio_return(x, expected_returns) - target}
            ]
            
            # Bounds
            bounds = [(0.01, 0.40) for _ in range(n_assets)]
            
            # Initial guess
            initial_weights = np.array([1/n_assets] * n_assets)
            
            # Optimize for minimum variance at this return level
            result = minimize(
                fun=lambda x: self.portfolio_variance(x, cov_matrix),
                x0=initial_weights,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 1000}
            )
            
            if result.success:
                weights = result.x
                port_return = self.portfolio_return(weights, expected_returns)
                port_vol = np.sqrt(self.portfolio_variance(weights, cov_matrix))
                sharpe = (port_return - self.risk_free_rate) / port_vol if port_vol > 0 else 0
                
                frontier_weights.append(weights)
                frontier_returns.append(port_return)
                frontier_volatilities.append(port_vol)
                frontier_sharpes.append(sharpe)
        
        # Find maximum Sharpe portfolio
        if frontier_sharpes:
            max_sharpe_idx = np.argmax(frontier_sharpes)
            max_sharpe_portfolio = {
                "return": frontier_returns[max_sharpe_idx],
                "volatility": frontier_volatilities[max_sharpe_idx],
                "sharpe": frontier_sharpes[max_sharpe_idx],
                "weights": frontier_weights[max_sharpe_idx]
            }
        else:
            max_sharpe_portfolio = {}
        
        # Find minimum variance portfolio
        if frontier_volatilities:
            min_var_idx = np.argmin(frontier_volatilities)
            min_var_portfolio = {
                "return": frontier_returns[min_var_idx],
                "volatility": frontier_volatilities[min_var_idx],
                "sharpe": frontier_sharpes[min_var_idx],
                "weights": frontier_weights[min_var_idx]
            }
        else:
            min_var_portfolio = {}
        
        frontier_data = {
            "returns": frontier_returns,
            "volatilities": frontier_volatilities,
            "sharpes": frontier_sharpes,
            "target_returns": target_returns.tolist(),
            "max_sharpe": max_sharpe_portfolio,
            "min_variance": min_var_portfolio,
            "symbols": symbols
        }
        
        logger.info(f"Efficient frontier generated with {len(frontier_returns)} points")
        return frontier_data
    
    def generate_optimization_report(self, optimized_portfolio: Dict, etf_data: Dict) -> str:
        """
        Generate human-readable optimization report.
        
        Args:
            optimized_portfolio: Optimized portfolio data
            etf_data: Original ETF data
            
        Returns:
            Formatted report string
        """
        if not optimized_portfolio.get("optimization_success", False):
            return "Optimization failed. No report generated."
        
        allocation = optimized_portfolio["allocation"]
        stats = optimized_portfolio["statistics"]
        sector_exposures = optimized_portfolio["sector_exposures"]
        
        report = []
        report.append("=" * 70)
        report.append("ETF PORTFOLIO OPTIMIZATION REPORT")
        report.append("=" * 70)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Target Return: {self.target_return*100:.1f}% annual")
        report.append("=" * 70)
        report.append("")
        
        # Portfolio Summary
        report.append("PORTFOLIO SUMMARY")
        report.append("-" * 40)
        report.append(f"Expected Return:      {stats['return']*100:6.2f}%")
        report.append(f"Expected Volatility:  {stats['volatility']*100:6.2f}%")
        report.append(f"Sharpe Ratio:         {stats['sharpe_ratio']:6.3f}")
        report.append(f"Sortino Ratio:        {stats['sortino_ratio']:6.3f}")
        report.append(f"Portfolio Expense:    {optimized_portfolio['portfolio_expense_ratio']*100:6.2f}%")
        report.append("")
        
        # Allocation Details
        report.append("ALLOCATION DETAILS")
        report.append("-" * 40)
        report.append(f"{'ETF':10} {'Weight':>10} {'Exp Return':>12} {'Volatility':>12} {'Expense':>10}")
        report.append("-" * 40)
        
        total_weight = 0
        for symbol, data in sorted(allocation.items(), key=lambda x: x[1]["weight"], reverse=True):
            weight = data["weight"]
            exp_return = data["expected_return"]
            volatility = data["volatility"]
            expense = data["expense_ratio"]
            
            report.append(f"{symbol:10} {weight*100:10.2f}% {exp_return*100:12.2f}% {volatility*100:12.2f}% {expense*100:10.2f}%")
            total_weight += weight
        
        report.append("-" * 40)
        report.append(f"{'Total':10} {total_weight*100:10.2f}%")
        report.append("")
        
        # Sector Exposures
        report.append("SECTOR EXPOSURES")
        report.append("-" * 40)
        for sector, exposure in sorted(sector_exposures.items(), key=lambda x: x[1], reverse=True):
            report.append(f"{sector:25} {exposure*100:6.2f}%")
        report.append("")
        
        # Constraints Summary
        report.append("CONSTRAINTS APPLIED")
        report.append("-" * 40)
        for constraint, value in self.constraints.items():
            report.append(f"{constraint.replace('_', ' ').title():25} {value*100:6.1f}%")
        report.append("")
        
        # Investment Recommendations
        report.append("INVESTMENT RECOMMENDATIONS")
        report.append("-" * 40)
        
        # Analyze allocation
        china_exposure = sum(exposure for sector, exposure in sector_exposures.items() 
                           if 'China' in sector)
        tech_exposure = sector_exposures.get('Technology', 0)
        sp500_exposure = sector_exposures.get('SP500 Core', 0)
        
        if china_exposure > 0.20:
            report.append("⚠️  China exposure >20% - Consider reducing if risk tolerance is low")
        if tech_exposure > 0.20:
            report.append("⚠️  Technology exposure >20% - Monitor for volatility")
        if sp500_exposure < 0.40:
            report.append("⚠️  SP500 exposure <40% - Below minimum, consider increasing")
        
        # Check if target achieved
        if stats['return'] >= self.target_return:
            report.append(f"✅ Target return of {self.target_return*100:.1f}% achieved")
        else:
            report.append(f"⚠️  Target return of {self.target_return*100:.1f}% not fully achieved")
            report.append(f"   Current projection: {stats['return']*100:.2f}%")
        
        # Rebalancing advice
        report.append("")
        report.append("REBALANCING ADVICE")
        report.append("-" * 40)
        report.append("• Rebalance quarterly or when allocations drift >5%")
        report.append("• Consider tax implications of selling")
        report.append("• Monitor China regulatory environment")
        report.append("• Review technology sector momentum monthly")
        report.append("")
        
        report.append("=" * 70)
        report.append("DISCLAIMER: This is an optimization model, not financial advice.")
        report.append("Past performance does not guarantee future results.")
        report.append("All investments carry risk. Consult a financial advisor.")
        report.append("=" * 70)
        
        return "\n".join(report)
    
    def save_optimization_results(self, optimized_portfolio: Dict, report: str, 
                                 filename_prefix: str = None):
        """
        Save optimization results to files.
        
        Args:
            optimized_portfolio: Optimized portfolio data
            report: Text report
            filename_prefix: Prefix for output files
        """
        if filename_prefix is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename_prefix = f"portfolio_optimization_{timestamp}"
        
        os.makedirs("outputs", exist_ok=True)
        
        # Save JSON data
        json_file = f"outputs/{filename_prefix}.json"
        with open(json_file, 'w') as f:
            json.dump(optimized_portfolio, f, indent=2, default=str)
        
        # Save text report
        report_file = f"outputs/{filename_prefix}_report.txt"
        with open(report_file, 'w') as f:
            f.write(report)
        
        # Save allocation CSV
        if "allocation" in optimized_portfolio:
            allocation_df = pd.DataFrame.from_dict(
                optimized_portfolio["allocation"], 
                orient='index'
            )
            csv_file = f"outputs/{filename_prefix}_allocation.csv"
            allocation_df.to_csv(csv_file)
        
        logger.info(f"Optimization results saved to outputs/{filename_prefix}_*")


def main():
    """Test the portfolio optimizer."""
    # Create sample ETF data for testing
    np.random.seed(42)
    
    # Sample ETFs with expected returns and volatilities
    sample_etfs = {
        "VOO": {
            "returns": pd.Series(np.random.normal(0.0005, 0.01, 1000)),
            "expense_ratio": 0.0003,
            "aum": 800_000_000_000
        },
        "VGT": {
            "returns": pd.Series(np.random.normal(0.0008, 0.015, 1000)),
            "expense_ratio": 0.0010,
            "aum": 50_000_000_000
        },
        "XLP": {
            "returns": pd.Series(np.random.normal(0.0003, 0.008, 1000)),
            "expense_ratio": 0.0010,
            "aum": 15_000_000_000
        },
        "XLU": {
            "returns": pd.Series(np.random.normal(0.0002, 0.007, 1000)),
            "expense_ratio": 0.0010,
            "aum": 12_000_000_000
        },
        "FXI": {
            "returns": pd.Series(np.random.normal(0.0006, 0.018, 1000)),
            "expense_ratio": 0.0074,
            "aum": 5_000_000_000
        },
        "KBA": {
            "returns": pd.Series(np.random.normal(0.0007, 0.020, 1000)),
            "expense_ratio": 0.0079,
            "aum": 1_000_000_000
        },
        "XLY": {
            "returns": pd.Series(np.random.normal(0.0006, 0.012, 1000)),
            "expense_ratio": 0.0010,
            "aum": 20_000_000_000
        }
    }
    
    print("=" * 70)
    print("PORTFOLIO OPTIMIZER TEST")
    print("=" * 70)
    
    # Initialize optimizer with 15% target
    optimizer = PortfolioOptimizer(target_return=0.15)
    
    # Run optimization
    optimized = optimizer.optimize_portfolio(sample_etfs, years=5)
    
    if optimized.get("optimization_success", False):
        # Generate report
        report = optimizer.generate_optimization_report(optimized, sample_etfs)
        print(report)
        
        # Save results
        optimizer.save_optimization_results(optimized, report)
        
        # Generate efficient frontier
        frontier = optimizer.generate_efficient_frontier(sample_etfs)
        if frontier:
            print(f"\nEfficient frontier generated with {len(frontier['returns'])} points")
            print(f"Max Sharpe portfolio: {frontier['max_sharpe']['return']*100:.2f}% return, "
                  f"{frontier['max_sharpe']['volatility']*100:.2f}% volatility")
    else:
        print("Optimization failed.")
        if "message" in optimized:
            print(f"Error: {optimized['message']}")
    
    print("\n" + "=" * 70)
    print("OPTIMIZATION TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
