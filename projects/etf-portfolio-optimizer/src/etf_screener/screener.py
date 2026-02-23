#!/usr/bin/env python3
"""
ETF Screener - Select lowest expense ratio ETFs for target sectors
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional, Tuple
import json
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ETFScreener:
    """Screen and select ETFs based on criteria."""
    
    # Target sectors and candidate ETFs
    SECTOR_ETFS = {
        "sp500_core": {
            "name": "S&P 500 Core",
            "candidates": [
                {"symbol": "VOO", "name": "Vanguard S&P 500 ETF", "target_fee": 0.0003},
                {"symbol": "SPY", "name": "SPDR S&P 500 ETF Trust", "target_fee": 0.0009},
                {"symbol": "IVV", "name": "iShares Core S&P 500 ETF", "target_fee": 0.0003},
            ]
        },
        "technology": {
            "name": "Technology",
            "candidates": [
                {"symbol": "VGT", "name": "Vanguard Information Technology ETF", "target_fee": 0.0010},
                {"symbol": "XLK", "name": "Technology Select Sector SPDR", "target_fee": 0.0010},
                {"symbol": "FTEC", "name": "Fidelity MSCI Information Technology ETF", "target_fee": 0.0008},
            ]
        },
        "consumer_staples": {
            "name": "Consumer Staples",
            "candidates": [
                {"symbol": "XLP", "name": "Consumer Staples Select Sector SPDR", "target_fee": 0.0010},
                {"symbol": "VDC", "name": "Vanguard Consumer Staples ETF", "target_fee": 0.0010},
                {"symbol": "FSTA", "name": "Fidelity MSCI Consumer Staples ETF", "target_fee": 0.0008},
            ]
        },
        "utilities": {
            "name": "Utilities",
            "candidates": [
                {"symbol": "XLU", "name": "Utilities Select Sector SPDR", "target_fee": 0.0010},
                {"symbol": "VPU", "name": "Vanguard Utilities ETF", "target_fee": 0.0010},
                {"symbol": "FUTY", "name": "Fidelity MSCI Utilities ETF", "target_fee": 0.0008},
            ]
        },
        "china_largecap": {
            "name": "China Large-Cap",
            "candidates": [
                {"symbol": "FXI", "name": "iShares China Large-Cap ETF", "target_fee": 0.0074},
                {"symbol": "MCHI", "name": "iShares MSCI China ETF", "target_fee": 0.0057},
                {"symbol": "GXC", "name": "SPDR S&P China ETF", "target_fee": 0.0059},
            ]
        },
        "china_ashares": {
            "name": "China A-Shares",
            "candidates": [
                {"symbol": "KBA", "name": "KraneShares Bosera MSCI China A ETF", "target_fee": 0.0079},
                {"symbol": "ASHR", "name": "Xtrackers Harvest CSI 300 China A-Shares ETF", "target_fee": 0.0065},
                {"symbol": "CHIQ", "name": "Global X MSCI China Consumer Disc ETF", "target_fee": 0.0065},
            ]
        },
        "consumer_discretionary": {
            "name": "Consumer Discretionary",
            "candidates": [
                {"symbol": "XLY", "name": "Consumer Discretionary Select Sector SPDR", "target_fee": 0.0010},
                {"symbol": "VCR", "name": "Vanguard Consumer Discretionary ETF", "target_fee": 0.0010},
                {"symbol": "FDIS", "name": "Fidelity MSCI Consumer Discretionary ETF", "target_fee": 0.0008},
            ]
        }
    }
    
    def __init__(self, min_aum: float = 100_000_000, min_volume: float = 100_000):
        """
        Initialize ETF screener.
        
        Args:
            min_aum: Minimum assets under management (default $100M)
            min_volume: Minimum average daily volume (default 100,000 shares)
        """
        self.min_aum = min_aum
        self.min_volume = min_volume
        self.selected_etfs = {}
        
        logger.info(f"ETF Screener initialized (min AUM: ${min_aum:,}, min volume: {min_volume:,})")
    
    def screen_etf(self, symbol: str, candidate_info: dict) -> Optional[dict]:
        """
        Screen a single ETF against criteria.
        
        Args:
            symbol: ETF symbol
            candidate_info: Candidate information
            
        Returns:
            Screened ETF data or None if fails criteria
        """
        try:
            logger.info(f"Screening {symbol}...")
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            # Extract key metrics
            etf_data = {
                "symbol": symbol,
                "name": info.get("longName", info.get("shortName", symbol)),
                "sector": candidate_info.get("sector", "Unknown"),
                "expense_ratio": info.get("expenseRatio"),
                "aum": info.get("totalAssets"),
                "avg_volume": info.get("averageVolume"),
                "dividend_yield": info.get("dividendYield"),
                "pe_ratio": info.get("trailingPE"),
                "ytd_return": info.get("ytdReturn"),
                "beta": info.get("beta"),
                "holdings": info.get("holdings"),
                "inception_date": info.get("fundInceptionDate"),
                "last_updated": datetime.now().isoformat()
            }
            
            # Check criteria
            passes = True
            failures = []
            
            # Check AUM
            if etf_data["aum"] and etf_data["aum"] < self.min_aum:
                passes = False
                failures.append(f"AUM ${etf_data['aum']:,} < ${self.min_aum:,}")
            
            # Check volume
            if etf_data["avg_volume"] and etf_data["avg_volume"] < self.min_volume:
                passes = False
                failures.append(f"Volume {etf_data['avg_volume']:,} < {self.min_volume:,}")
            
            # Check expense ratio exists
            if etf_data["expense_ratio"] is None:
                logger.warning(f"{symbol}: No expense ratio data")
                # Don't fail for missing expense ratio, but note it
            
            if passes:
                logger.info(f"  ✓ {symbol}: ${etf_data['aum']:,} AUM, {etf_data['expense_ratio'] or 'N/A'} expense")
                return etf_data
            else:
                logger.info(f"  ✗ {symbol}: Failed - {', '.join(failures)}")
                return None
                
        except Exception as e:
            logger.error(f"Error screening {symbol}: {e}")
            return None
    
    def screen_all_sectors(self) -> Dict[str, List[dict]]:
        """
        Screen ETFs for all target sectors.
        
        Returns:
            Dictionary mapping sectors to list of qualifying ETFs
        """
        logger.info("Screening ETFs across all sectors...")
        
        sector_results = {}
        
        for sector_key, sector_info in self.SECTOR_ETFS.items():
            logger.info(f"\nScreening {sector_info['name']} sector...")
            
            screened_etfs = []
            for candidate in sector_info["candidates"]:
                etf_data = self.screen_etf(candidate["symbol"], candidate)
                if etf_data:
                    # Add target fee for comparison
                    etf_data["target_fee"] = candidate.get("target_fee")
                    screened_etfs.append(etf_data)
            
            # Sort by expense ratio (lowest first)
            screened_etfs.sort(key=lambda x: x.get("expense_ratio", float('inf')))
            
            sector_results[sector_key] = {
                "sector_name": sector_info["name"],
                "etfs": screened_etfs,
                "screened_count": len(screened_etfs)
            }
            
            logger.info(f"  Found {len(screened_etfs)} qualifying ETFs")
        
        self.sector_results = sector_results
        return sector_results
    
    def select_lowest_fee_etfs(self, sector_results: Dict = None) -> Dict[str, dict]:
        """
        Select the lowest expense ratio ETF from each sector.
        
        Args:
            sector_results: Screened sector results (uses cached if None)
            
        Returns:
            Dictionary mapping sector to selected ETF
        """
        if sector_results is None:
            if not hasattr(self, 'sector_results'):
                sector_results = self.screen_all_sectors()
            else:
                sector_results = self.sector_results
        
        selected = {}
        
        for sector_key, sector_data in sector_results.items():
            etfs = sector_data.get("etfs", [])
            if etfs:
                # Select ETF with lowest expense ratio
                best_etf = min(etfs, key=lambda x: x.get("expense_ratio", float('inf')))
                selected[sector_key] = best_etf
                
                logger.info(f"Selected for {sector_data['sector_name']}: "
                          f"{best_etf['symbol']} ({best_etf.get('expense_ratio', 'N/A')} expense)")
            else:
                logger.warning(f"No ETFs qualified for {sector_data['sector_name']}")
        
        self.selected_etfs = selected
        return selected
    
    def get_historical_returns(self, symbols: List[str], years: int = 5) -> pd.DataFrame:
        """
        Get historical returns for selected ETFs.
        
        Args:
            symbols: List of ETF symbols
            years: Number of years of history
            
        Returns:
            DataFrame with daily returns
        """
        logger.info(f"Fetching {years}-year historical returns for {len(symbols)} ETFs...")
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=years * 365)
        
        returns_data = {}
        
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(start=start_date, end=end_date, interval="1d")
                
                if not hist.empty:
                    # Calculate daily returns
                    returns = hist['Close'].pct_change().dropna()
                    returns_data[symbol] = returns
                    logger.info(f"  {symbol}: {len(returns)} trading days")
                else:
                    logger.warning(f"  {symbol}: No historical data")
                    
            except Exception as e:
                logger.error(f"Error fetching returns for {symbol}: {e}")
        
        # Combine into DataFrame
        if returns_data:
            returns_df = pd.DataFrame(returns_data)
            logger.info(f"Return data shape: {returns_df.shape}")
            return returns_df
        else:
            logger.error("No return data collected")
            return pd.DataFrame()
    
    def calculate_correlations(self, returns_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate correlation matrix for ETFs.
        
        Args:
            returns_df: DataFrame with daily returns
            
        Returns:
            Correlation matrix
        """
        if returns_df.empty:
            logger.error("No return data for correlation calculation")
            return pd.DataFrame()
        
        correlation_matrix = returns_df.corr()
        
        logger.info("Correlation matrix calculated")
        logger.info(f"\n{correlation_matrix.round(3)}")
        
        return correlation_matrix
    
    def generate_screening_report(self, selected_etfs: Dict = None) -> dict:
        """
        Generate comprehensive screening report.
        
        Args:
            selected_etfs: Selected ETFs (uses cached if None)
            
        Returns:
            Screening report dictionary
        """
        if selected_etfs is None:
            if not self.selected_etfs:
                selected_etfs = self.select_lowest_fee_etfs()
            else:
                selected_etfs = self.selected_etfs
        
        report = {
            "generated_at": datetime.now().isoformat(),
            "screening_criteria": {
                "min_aum": self.min_aum,
                "min_volume": self.min_volume
            },
            "selected_etfs": selected_etfs,
            "summary_metrics": {}
        }
        
        # Calculate summary metrics
        total_etfs = len(selected_etfs)
        if total_etfs > 0:
            # Average expense ratio
            expense_ratios = [etf.get("expense_ratio") for etf in selected_etfs.values() 
                            if etf.get("expense_ratio") is not None]
            if expense_ratios:
                report["summary_metrics"]["avg_expense_ratio"] = np.mean(expense_ratios)
                report["summary_metrics"]["min_expense_ratio"] = min(expense_ratios)
                report["summary_metrics"]["max_expense_ratio"] = max(expense_ratios)
            
            # Total AUM
            total_aum = sum(etf.get("aum", 0) for etf in selected_etfs.values())
            report["summary_metrics"]["total_aum"] = total_aum
            
            # Average dividend yield
            dividend_yields = [etf.get("dividend_yield", 0) for etf in selected_etfs.values()]
            report["summary_metrics"]["avg_dividend_yield"] = np.mean(dividend_yields)
        
        logger.info(f"Screening report generated: {total_etfs} ETFs selected")
        return report
    
    def save_report(self, report: dict, filename: str = None):
        """
        Save screening report to file.
        
        Args:
            report: Report dictionary
            filename: Output filename
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"etf_screening_report_{timestamp}.json"
        
        os.makedirs("outputs", exist_ok=True)
        
        try:
            with open(f"outputs/{filename}", 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            logger.info(f"Report saved to outputs/{filename}")
            
        except Exception as e:
            logger.error(f"Error saving report: {e}")


def main():
    """Main function for testing the screener."""
    screener = ETFScreener(min_aum=50_000_000, min_volume=50_000)
    
    print("=" * 60)
    print("ETF Screener - Lowest Fee ETF Selection")
    print("=" * 60)
    
    # Screen all sectors
    sector_results = screener.screen_all_sectors()
    
    # Select lowest fee ETFs
    selected = screener.select_lowest_fee_etfs(sector_results)
    
    print("\n" + "=" * 60)
    print("SELECTED LOWEST-FEE ETFS")
    print("=" * 60)
    
    for sector, etf in selected.items():
        sector_name = screener.SECTOR_ETFS[sector]["name"]
        expense = etf.get("expense_ratio", "N/A")
        aum = etf.get("aum", 0)
        
        print(f"{sector_name:25} {etf['symbol']:6} "
              f"Expense: {expense if expense != 'N/A' else 'N/A':7} "
              f"AUM: ${aum/1e9:.1f}B")
    
    # Get historical returns for correlation analysis
    symbols = [etf["symbol"] for etf in selected.values()]
    returns_df = screener.get_historical_returns(symbols, years=3)
    
    if not returns_df.empty:
        print(f"\nHistorical returns collected: {returns_df.shape}")
        
        # Calculate correlations
        corr_matrix = screener.calculate_correlations(returns_df)
        
        print("\nTop correlations:")
        # Find highest correlations (excluding self-correlation)
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr = corr_matrix.iloc[i, j]
                if abs(corr) > 0.7:
                    print(f"  {corr_matrix.columns[i]} - {corr_matrix.columns[j]}: {corr:.3f}")
    
    # Generate and save report
    report = screener.generate_screening_report(selected)
    screener.save_report(report)
    
    print("\n" + "=" * 60)
    print("SCREENING COMPLETE")
    print("=" * 60)
    print(f"Selected {len(selected)} ETFs")
    print(f"Average expense ratio: {report['summary_metrics'].get('avg_expense_ratio', 'N/A'):.4f}")
    print(f"Total AUM: ${report['summary_metrics'].get('total_aum', 0)/1e9:.1f}B")
    print("Report saved to outputs/ directory")


if __name__ == "__main__":
    main()