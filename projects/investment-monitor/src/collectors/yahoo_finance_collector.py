#!/usr/bin/env python3
"""
Yahoo Finance Data Collector

This module collects financial data from Yahoo Finance API.
Supports SPY, QQQ, Tencent, and other assets.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import logging
from typing import Dict, List, Optional, Tuple, Union
import json
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class YahooFinanceCollector:
    """Collect financial data from Yahoo Finance."""
    
    def __init__(self, cache_dir: str = "./data/cache"):
        """
        Initialize the collector.
        
        Args:
            cache_dir: Directory to cache downloaded data
        """
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
        # Define assets to monitor
        self.assets = {
            "SPY": {
                "symbol": "SPY",
                "name": "SPDR S&P 500 ETF Trust",
                "currency": "USD",
                "description": "Tracks S&P 500 index"
            },
            "QQQ": {
                "symbol": "QQQ",
                "name": "Invesco QQQ Trust",
                "currency": "USD",
                "description": "Tracks NASDAQ-100 index"
            },
            "TENCENT": {
                "symbol": "0700.HK",
                "name": "Tencent Holdings Limited",
                "currency": "HKD",
                "description": "Chinese tech giant"
            }
        }
        
        # Additional reference assets
        self.reference_assets = {
            "VOO": "Vanguard S&P 500 ETF",
            "IVV": "iShares Core S&P 500 ETF",
            "BND": "Vanguard Total Bond Market ETF",
            "GLD": "SPDR Gold Shares",
            "TLT": "iShares 20+ Year Treasury Bond ETF"
        }
        
        logger.info(f"Initialized YahooFinanceCollector with {len(self.assets)} assets")
    
    def get_asset_info(self, symbol: str) -> Dict:
        """
        Get basic information about an asset.
        
        Args:
            symbol: Stock/ETF symbol
            
        Returns:
            Dictionary with asset information
        """
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            asset_info = {
                "symbol": symbol,
                "name": info.get("longName", info.get("shortName", symbol)),
                "currency": info.get("currency", "USD"),
                "sector": info.get("sector", "Unknown"),
                "industry": info.get("industry", "Unknown"),
                "market_cap": info.get("marketCap"),
                "pe_ratio": info.get("trailingPE"),
                "dividend_yield": info.get("dividendYield"),
                "52_week_high": info.get("fiftyTwoWeekHigh"),
                "52_week_low": info.get("fiftyTwoWeekLow"),
                "volume_avg": info.get("averageVolume"),
                "beta": info.get("beta"),
                "last_updated": datetime.now().isoformat()
            }
            
            logger.info(f"Retrieved info for {symbol}: {asset_info['name']}")
            return asset_info
            
        except Exception as e:
            logger.error(f"Error getting info for {symbol}: {e}")
            return {
                "symbol": symbol,
                "name": symbol,
                "currency": "Unknown",
                "error": str(e)
            }
    
    def download_historical_data(
        self,
        symbol: str,
        start_date: Union[str, datetime] = "2010-01-01",
        end_date: Union[str, datetime] = None,
        interval: str = "1d",
        force_download: bool = False
    ) -> pd.DataFrame:
        """
        Download historical price data.
        
        Args:
            symbol: Stock/ETF symbol
            start_date: Start date for historical data
            end_date: End date (defaults to today)
            interval: Data interval (1d, 1wk, 1mo)
            force_download: Force re-download even if cached
            
        Returns:
            DataFrame with historical data
        """
        if end_date is None:
            end_date = datetime.now()
        
        # Check cache first
        cache_file = os.path.join(
            self.cache_dir, 
            f"{symbol}_{interval}_{start_date}_{end_date}.parquet"
        )
        
        if not force_download and os.path.exists(cache_file):
            try:
                logger.info(f"Loading cached data for {symbol}")
                df = pd.read_parquet(cache_file)
                return df
            except Exception as e:
                logger.warning(f"Error loading cache for {symbol}: {e}")
        
        try:
            logger.info(f"Downloading historical data for {symbol} ({start_date} to {end_date})")
            
            # Download data
            ticker = yf.Ticker(symbol)
            df = ticker.history(
                start=start_date,
                end=end_date,
                interval=interval,
                auto_adjust=True
            )
            
            if df.empty:
                logger.warning(f"No data returned for {symbol}")
                return pd.DataFrame()
            
            # Add additional calculated columns
            df['Returns'] = df['Close'].pct_change()
            df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
            df['Volume_Ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()
            
            # Calculate simple moving averages
            df['SMA_20'] = df['Close'].rolling(window=20).mean()
            df['SMA_50'] = df['Close'].rolling(window=50).mean()
            df['SMA_200'] = df['Close'].rolling(window=200).mean()
            
            # Calculate daily range
            df['Daily_Range'] = (df['High'] - df['Low']) / df['Close'] * 100
            
            # Add symbol column
            df['Symbol'] = symbol
            
            # Cache the data
            df.to_parquet(cache_file)
            logger.info(f"Cached data for {symbol} to {cache_file}")
            
            logger.info(f"Downloaded {len(df)} records for {symbol}")
            return df
            
        except Exception as e:
            logger.error(f"Error downloading data for {symbol}: {e}")
            return pd.DataFrame()
    
    def download_multiple_assets(
        self,
        symbols: List[str],
        start_date: Union[str, datetime] = "2020-01-01",
        end_date: Union[str, datetime] = None,
        interval: str = "1d"
    ) -> Dict[str, pd.DataFrame]:
        """
        Download historical data for multiple assets.
        
        Args:
            symbols: List of stock/ETF symbols
            start_date: Start date for historical data
            end_date: End date (defaults to today)
            interval: Data interval
            
        Returns:
            Dictionary mapping symbols to DataFrames
        """
        results = {}
        
        for symbol in symbols:
            logger.info(f"Downloading data for {symbol}...")
            df = self.download_historical_data(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                interval=interval
            )
            
            if not df.empty:
                results[symbol] = df
            else:
                logger.warning(f"No data for {symbol}")
            
            # Be nice to the API
            time.sleep(0.5)
        
        logger.info(f"Downloaded data for {len(results)}/{len(symbols)} assets")
        return results
    
    def get_current_price(self, symbol: str) -> Dict:
        """
        Get current price and market data.
        
        Args:
            symbol: Stock/ETF symbol
            
        Returns:
            Dictionary with current price data
        """
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            # Get latest data
            hist = ticker.history(period="1d", interval="1m")
            
            if hist.empty:
                # Try daily data
                hist = ticker.history(period="5d", interval="1d")
            
            current_data = {
                "symbol": symbol,
                "timestamp": datetime.now().isoformat(),
                "price": info.get("regularMarketPrice") or info.get("currentPrice"),
                "previous_close": info.get("previousClose"),
                "open": info.get("open"),
                "day_high": info.get("dayHigh"),
                "day_low": info.get("dayLow"),
                "volume": info.get("volume"),
                "avg_volume": info.get("averageVolume"),
                "market_cap": info.get("marketCap"),
                "pe_ratio": info.get("trailingPE"),
                "dividend_yield": info.get("dividendYield"),
                "52_week_high": info.get("fiftyTwoWeekHigh"),
                "52_week_low": info.get("fiftyTwoWeekLow"),
                "currency": info.get("currency", "USD"),
                "exchange": info.get("exchange"),
                "quote_type": info.get("quoteType"),
                "data_source": "Yahoo Finance"
            }
            
            # Calculate daily change if we have data
            if current_data["price"] and current_data["previous_close"]:
                price = current_data["price"]
                prev_close = current_data["previous_close"]
                current_data["change"] = price - prev_close
                current_data["change_percent"] = (price - prev_close) / prev_close * 100
            
            logger.info(f"Current price for {symbol}: ${current_data.get('price', 'N/A')}")
            return current_data
            
        except Exception as e:
            logger.error(f"Error getting current price for {symbol}: {e}")
            return {
                "symbol": symbol,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def get_dividend_history(self, symbol: str, years: int = 5) -> pd.DataFrame:
        """
        Get dividend history for an asset.
        
        Args:
            symbol: Stock/ETF symbol
            years: Number of years of history
            
        Returns:
            DataFrame with dividend history
        """
        try:
            ticker = yf.Ticker(symbol)
            
            # Get dividend history
            dividends = ticker.dividends
            
            if dividends.empty:
                logger.info(f"No dividend history for {symbol}")
                return pd.DataFrame()
            
            # Filter for last N years
            cutoff_date = datetime.now() - timedelta(days=years*365)
            dividends = dividends[dividends.index >= cutoff_date]
            
            # Calculate annual dividends
            annual_dividends = dividends.resample('Y').sum()
            
            logger.info(f"Retrieved {len(dividends)} dividend records for {symbol}")
            return dividends
            
        except Exception as e:
            logger.error(f"Error getting dividend history for {symbol}: {e}")
            return pd.DataFrame()
    
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate technical indicators for price data.
        
        Args:
            df: DataFrame with price data (must have OHLC columns)
            
        Returns:
            DataFrame with added technical indicators
        """
        if df.empty:
            return df
        
        df = df.copy()
        
        # Calculate RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Calculate MACD
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
        
        # Calculate Bollinger Bands
        df['BB_Middle'] = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
        df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
        df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']
        
        # Calculate ATR (Average True Range)
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        df['ATR'] = true_range.rolling(window=14).mean()
        
        # Calculate Volume indicators
        df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
        
        # Calculate Price channels
        df['20d_High'] = df['High'].rolling(window=20).max()
        df['20d_Low'] = df['Low'].rolling(window=20).min()
        
        logger.info(f"Calculated technical indicators for {len(df)} records")
        return df
    
    def generate_summary_report(self, symbols: List[str] = None) -> Dict:
        """
        Generate a summary report for monitored assets.
        
        Args:
            symbols: List of symbols to include (defaults to all monitored assets)
            
        Returns:
            Dictionary with summary report
        """
        if symbols is None:
            symbols = list(self.assets.keys())
        
        report = {
            "generated_at": datetime.now().isoformat(),
            "assets": {},
            "market_summary": {},
            "recommendations": []
        }
        
        total_assets = len(symbols)
        successful = 0
        
        for symbol in symbols:
            try:
                # Get current data
                current_data = self.get_current_price(symbol)
                
                # Get recent historical data for analysis
                hist_data = self.download_historical_data(
                    symbol=symbol,
                    start_date=datetime.now() - timedelta(days=60),
                    end_date=datetime.now(),
                    interval="1d"
                )
                
                if not hist_data.empty:
                    # Calculate recent performance
                    recent_prices = hist_data['Close']
                    if len(recent_prices) >= 5:
                        week_change = (recent_prices.iloc[-1] - recent_prices.iloc[-5]) / recent_prices.iloc[-5] * 100
                        month_change = (recent_prices.iloc[-1] - recent_prices.iloc[-20]) / recent_prices.iloc[-20] * 100
                    else:
                        week_change = month_change = None
                    
                    # Add to report
                    asset_report = {
                        "current_price": current_data.get("price"),
                        "currency": current_data.get("currency"),
                        "daily_change": current_data.get("change_percent"),
                        "week_change": week_change,
                        "month_change": month_change,
                        "volume": current_data.get("volume"),
                        "avg_volume": current_data.get("avg_volume"),
                        "pe_ratio": current_data.get("pe_ratio"),
                        "dividend_yield": current_data.get("dividend_yield"),
                        "market_cap": current_data.get("market_cap"),
                        "last_updated": current_data.get("timestamp")
                    }
                    
                    report["assets"][symbol] = asset_report
                    successful += 1
                    
            except Exception as e:
                logger.error(f"Error generating report for {symbol}: {e}")
                report["assets"][symbol] = {"error": str(e)}
        
        # Calculate market summary
        if successful > 0:
            daily_changes = []
            for asset_data in report["assets"].values():
                if "daily_change" in asset_data and asset_data["daily_change"] is not None:
                    daily_changes.append(asset_data["daily_change"])
            
            if daily_changes:
                report["market_summary"] = {
                    "assets_monitored": total_assets,
                    "assets_successful": successful,
                    "avg_daily_change": np.mean(daily_changes),
                    "positive_assets": sum(1 for change in daily_changes if change > 0),
                    "negative_assets": sum(1 for change in daily_changes if change < 0),
                    "update_time": datetime.now().isoformat()
                }
        
        logger.info(f"Generated summary report for {successful}/{total_assets} assets")
        return report


def main():
    """Main function for testing the collector."""
    collector = YahooFinanceCollector()
    
    # Test with SPY
    print("Testing Yahoo Finance Collector...")
    
    # Get asset info
    spy_info = collector.get_asset_info("SPY")
    print(f"\nSPY Info: {spy_info['name']}")
    print(f"Market Cap: ${spy_info.get('market_cap', 'N/A'):,}")
    print(f"PE Ratio: {spy_info.get('pe_ratio', 'N/A')}")
    
    # Get current price
    spy_current = collector.get_current_price("SPY")
    print(f"\nCurrent Price: ${spy_current.get('price', 'N/A')}")
    print(f"Daily Change: {spy_current.get('change_percent', 'N/A'):.2f}%")
    
    # Download historical data
    print("\nDownloading historical data...")
    spy_hist = collector.download_historical_data(
        symbol="SPY",
        start_date="2024-01-01",
        end_date=datetime.now(),

    # Download historical data
    print("\nDownloading historical data...")
    spy_hist = collector.download_historical_data(
        symbol="SPY",
        start_date="2024-01-01",
        end_date=datetime.now(),
        interval="1d"
    )
    
    if not spy_hist.empty:
        print(f"Downloaded {len(spy_hist)} records")
        print(f"Latest close: ${spy_hist['Close'].iloc[-1]:.2f}")
        print(f"Date range: {spy_hist.index[0].date()} to {spy_hist.index[-1].date()}")
    
    # Generate summary report
    print("\nGenerating summary report...")
    report = collector.generate_summary_report(["SPY", "QQQ", "0700.HK"])
    
    print(f"\nReport generated at: {report['generated_at']}")
    for symbol, data in report['assets'].items():
        if 'current_price' in data:
            print(f"{symbol}: ${data['current_price']:.2f} ({data.get('daily_change', 0):.2f}%)")
    
    print("\nCollector test completed successfully!")


if __name__ == "__main__":
    main()
