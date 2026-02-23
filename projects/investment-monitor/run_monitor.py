#!/usr/bin/env python3
"""
Investment Monitor - Main Entry Point

This script runs the investment monitoring system for SPY, QQQ, and Tencent.
"""

import sys
import os
import logging
from datetime import datetime, timedelta
import json
import yaml

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from collectors.yahoo_finance_collector import YahooFinanceCollector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('investment_monitor.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class InvestmentMonitor:
    """Main investment monitoring system."""
    
    def __init__(self, config_path: str = None):
        """
        Initialize the investment monitor.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self.load_config(config_path)
        self.collector = YahooFinanceCollector()
        
        # Assets to monitor
        self.target_assets = ["SPY", "QQQ", "0700.HK"]
        
        logger.info("Investment Monitor initialized")
        logger.info(f"Monitoring assets: {', '.join(self.target_assets)}")
    
    def load_config(self, config_path: str = None) -> dict:
        """
        Load configuration from YAML file.
        
        Args:
            config_path: Path to config file
            
        Returns:
            Configuration dictionary
        """
        default_config = {
            "assets": {
                "SPY": {"weight": 0.75, "enabled": True},
                "QQQ": {"weight": 0.00, "enabled": True},
                "0700.HK": {"weight": 0.25, "enabled": True}
            },
            "analysis": {
                "target_return": 0.15,
                "risk_free_rate": 0.02,
                "rebalance_threshold": 0.05
            },
            "data": {
                "historical_days": 365 * 3,  # 3 years
                "update_interval": 300  # 5 minutes
            }
        }
        
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    user_config = yaml.safe_load(f)
                
                # Merge with default config
                merged_config = {**default_config, **user_config}
                logger.info(f"Loaded configuration from {config_path}")
                return merged_config
                
            except Exception as e:
                logger.error(f"Error loading config from {config_path}: {e}")
                logger.info("Using default configuration")
                return default_config
        else:
            logger.info("No config file found, using default configuration")
            return default_config
    
    def collect_market_data(self) -> dict:
        """
        Collect current market data for all assets.
        
        Returns:
            Dictionary with market data
        """
        logger.info("Collecting market data...")
        
        market_data = {
            "timestamp": datetime.now().isoformat(),
            "assets": {},
            "summary": {}
        }
        
        for symbol in self.target_assets:
            try:
                logger.info(f"Fetching data for {symbol}...")
                
                # Get current price
                current_data = self.collector.get_current_price(symbol)
                
                # Get recent historical data (last 30 days)
                hist_data = self.collector.download_historical_data(
                    symbol=symbol,
                    start_date=datetime.now() - timedelta(days=30),
                    end_date=datetime.now(),
                    interval="1d"
                )
                
                if not hist_data.empty:
                    # Calculate performance metrics
                    recent_prices = hist_data['Close']
                    
                    # Daily change
                    if len(recent_prices) >= 2:
                        daily_change_pct = (recent_prices.iloc[-1] - recent_prices.iloc[-2]) / recent_prices.iloc[-2] * 100
                    else:
                        daily_change_pct = None
                    
                    # Weekly change (5 trading days)
                    if len(recent_prices) >= 5:
                        weekly_change_pct = (recent_prices.iloc[-1] - recent_prices.iloc[-5]) / recent_prices.iloc[-5] * 100
                    else:
                        weekly_change_pct = None
                    
                    # Monthly change (20 trading days)
                    if len(recent_prices) >= 20:
                        monthly_change_pct = (recent_prices.iloc[-1] - recent_prices.iloc[-20]) / recent_prices.iloc[-20] * 100
                    else:
                        monthly_change_pct = None
                    
                    # Calculate volatility (20-day std)
                    if len(recent_prices) >= 20:
                        returns = recent_prices.pct_change().dropna()
                        volatility = returns.std() * (252 ** 0.5)  # Annualized
                    else:
                        volatility = None
                    
                    asset_data = {
                        "symbol": symbol,
                        "name": current_data.get("name", symbol),
                        "current_price": current_data.get("price"),
                        "currency": current_data.get("currency", "USD"),
                        "daily_change": daily_change_pct,
                        "weekly_change": weekly_change_pct,
                        "monthly_change": monthly_change_pct,
                        "volatility": volatility,
                        "volume": current_data.get("volume"),
                        "market_cap": current_data.get("market_cap"),
                        "pe_ratio": current_data.get("pe_ratio"),
                        "dividend_yield": current_data.get("dividend_yield"),
                        "data_points": len(hist_data),
                        "last_updated": current_data.get("timestamp")
                    }
                    
                    market_data["assets"][symbol] = asset_data
                    logger.info(f"  {symbol}: ${asset_data['current_price']:.2f} ({asset_data.get('daily_change', 0):.2f}%)")
                    
                else:
                    logger.warning(f"No historical data for {symbol}")
                    market_data["assets"][symbol] = {
                        "symbol": symbol,
                        "error": "No data available"
                    }
                    
            except Exception as e:
                logger.error(f"Error collecting data for {symbol}: {e}")
                market_data["assets"][symbol] = {
                    "symbol": symbol,
                    "error": str(e)
                }
        
        # Calculate portfolio summary
        self.calculate_portfolio_summary(market_data)
        
        logger.info(f"Market data collection complete: {len(market_data['assets'])} assets")
        return market_data
    
    def calculate_portfolio_summary(self, market_data: dict):
        """
        Calculate portfolio-level summary metrics.
        
        Args:
            market_data: Dictionary with asset data
        """
        assets = market_data["assets"]
        
        # Only calculate if we have price data
        valid_assets = [a for a in assets.values() if a.get("current_price") is not None]
        
        if not valid_assets:
            market_data["summary"] = {"error": "No valid asset data"}
            return
        
        # Get target weights from config
        target_weights = {}
        for symbol in self.target_assets:
            if symbol in self.config.get("assets", {}):
                target_weights[symbol] = self.config["assets"][symbol].get("weight", 0)
        
        # Calculate weighted returns
        weighted_daily_return = 0
        weighted_weekly_return = 0
        weighted_monthly_return = 0
        
        for asset in valid_assets:
            symbol = asset["symbol"]
            weight = target_weights.get(symbol, 0)
            
            if weight > 0:
                if asset.get("daily_change") is not None:
                    weighted_daily_return += asset["daily_change"] * weight
                if asset.get("weekly_change") is not None:
                    weighted_weekly_return += asset["weekly_change"] * weight
                if asset.get("monthly_change") is not None:
                    weighted_monthly_return += asset["monthly_change"] * weight
        
        # Calculate portfolio value (assuming $100,000 initial)
        initial_capital = 100000
        portfolio_value = 0
        
        for asset in valid_assets:
            symbol = asset["symbol"]
            weight = target_weights.get(symbol, 0)
            
            if weight > 0 and asset.get("current_price"):
                # Calculate position value
                position_value = initial_capital * weight
                portfolio_value += position_value
        
        market_data["summary"] = {
            "portfolio_value": portfolio_value,
            "initial_capital": initial_capital,
            "total_return": ((portfolio_value - initial_capital) / initial_capital * 100) if portfolio_value > 0 else 0,
            "weighted_daily_return": weighted_daily_return,
            "weighted_weekly_return": weighted_weekly_return,
            "weighted_monthly_return": weighted_monthly_return,
            "assets_count": len(valid_assets),
            "target_annual_return": self.config["analysis"].get("target_return", 0.15) * 100,
            "calculation_time": datetime.now().isoformat()
        }
    
    def generate_investment_insights(self, market_data: dict) -> dict:
        """
        Generate investment insights based on market data.
        
        Args:
            market_data: Dictionary with market data
            
        Returns:
            Dictionary with investment insights
        """
        logger.info("Generating investment insights...")
        
        insights = {
            "timestamp": datetime.now().isoformat(),
            "market_condition": "NEUTRAL",
            "recommendations": [],
            "warnings": [],
            "opportunities": [],
            "portfolio_action": "HOLD"
        }
        
        assets = market_data.get("assets", {})
        summary = market_data.get("summary", {})
        
        # Analyze each asset
        for symbol, asset in assets.items():
            if asset.get("error"):
                continue
            
            price = asset.get("current_price")
            daily_change = asset.get("daily_change", 0)
            weekly_change = asset.get("weekly_change", 0)
            
            # Generate asset-specific insights
            asset_insights = []
            
            # Price movement analysis
            if daily_change is not None:
                if daily_change > 2:
                    asset_insights.append(f"Strong daily gain of {daily_change:.2f}%")
                elif daily_change < -2:
                    asset_insights.append(f"Significant daily drop of {daily_change:.2f}%")
            
            if weekly_change is not None:
                if weekly_change > 5:
                    asset_insights.append(f"Strong weekly performance: {weekly_change:.2f}%")
                elif weekly_change < -5:
                    asset_insights.append(f"Weak weekly performance: {weekly_change:.2f}%")
            
            # Volume analysis
            volume = asset.get("volume")
            if volume and volume > 10000000:  # 10 million shares
                asset_insights.append("High trading volume detected")
            
            # PE ratio analysis
            pe_ratio = asset.get("pe_ratio")
            if pe_ratio:
                if pe_ratio < 15:
                    asset_insights.append("Low PE ratio - potentially undervalued")
                elif pe_ratio > 30:
                    asset_insights.append("High PE ratio - growth expectations priced in")
            
            # Add to recommendations if insights exist
            if asset_insights:
                insights["recommendations"].append({
                    "asset": symbol,
                    "insights": asset_insights,
                    "action": "MONITOR"  # Default action
                })
        
        # Portfolio-level insights
        target_return = self.config["analysis"].get("target_return", 0.15)
        weighted_monthly = summary.get("weighted_monthly_return", 0)
        
        # Calculate implied annual return from monthly
        implied_annual = ((1 + weighted_monthly/100) ** 12 - 1) * 100 if weighted_monthly else 0
        
        if implied_annual > target_return * 100:
            insights["portfolio_action"] = "CONSIDER_TAKING_PROFITS"
            insights["opportunities"].append(
                f"Portfolio performing above target ({implied_annual:.1f}% vs {target_return*100:.1f}% target)"
            )
        elif implied_annual < target_return * 100 * 0.5:  # Less than half of target
            insights["portfolio_action"] = "CONSIDER_ADDING"
            insights["opportunities"].append(
                f"Portroom for improvement ({implied_annual:.1f}% vs {target_return*100:.1f}% target)"
            )
        
        # Market condition assessment
        positive_assets = sum(1 for a in assets.values() 
                            if a.get("daily_change", 0) > 0 and not a.get("error"))
        total_assets = len([a for a in assets.values() if not a.get("error")])
        
        if total_assets > 0:
            positive_ratio = positive_assets / total_assets
            
            if positive_ratio > 0.7:
                insights["market_condition"] = "BULLISH"
            elif positive_ratio < 0.3:
                insights["market_condition"] = "BEARISH"
            else:
                insights["market_condition"] = "NEUTRAL"
        
        logger.info(f"Generated insights: {insights['market_condition']} market, {len(insights['recommendations'])} recommendations")
        return insights
    
    def generate_daily_report(self, market_data: dict, insights: dict) -> str:
        """
        Generate a formatted daily report.
        
        Args:
            market_data: Market data dictionary
            insights: Insights dictionary
            
        Returns:
            Formatted report string
        """
        logger.info("Generating daily report...")
        
        report = []
        report.append("# 📈 Daily Investment Report")
        report.append(f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
        report.append("")
        
        # Market Summary
        report.append("## 📊 Market Summary")
        summary = market_data.get("summary", {})
        
        report.append(f"- **Portfolio Value**: ${summary.get('portfolio_value', 0):,.2f}")
        report.append(f"- **Total Return**: {summary.get('total_return', 0):.2f}%")
        report.append(f"- **Target Annual Return**: {summary.get('target_annual_return', 15):.1f}%")
        report.append(f"- **Market Condition**: {insights.get('market_condition', 'NEUTRAL')}")
        report.append(f"- **Recommended Action**: {insights.get('portfolio_action', 'HOLD')}")
        report.append("")
        
        # Asset Performance
        report.append("## 📈 Asset Performance")
        assets = market_data.get("assets", {})
        
        for symbol, asset in assets.items():
            if asset.get("error"):
                report.append(f"### {symbol}: Error - {asset['error']}")
                continue
            
            report.append(f"### {symbol} - {asset.get('name', symbol)}")
            report.append(f"- **Price**: ${asset.get('current_price', 'N/A'):.2f} {asset.get('currency', '')}")
            
            changes = []
            if asset.get("daily_change") is not None:
                changes.append(f"Day: {asset['daily_change']:+.2f}%")
            if asset.get("weekly_change") is not None:
                changes.append(f"Week: {asset['weekly_change']:+.2f}%")
            if asset.get("monthly_change") is not None:
                changes.append(f"Month: {asset['monthly_change']:+.2f}%")
            
            if changes:
                report.append(f"- **Performance**: {', '.join(changes)}")
            
            metrics = []
            if asset.get("pe_ratio"):
                metrics.append(f"PE: {asset['pe_ratio']:.2f}")
            if asset.get("dividend_yield"):
                metrics.append(f"Yield: {asset['dividend_yield']*100:.2f}%")
            if asset.get("volatility"):
                metrics.append(f"Vol: {asset['volatility']:.2f}")
            
            if metrics:
                report.append(f"- **Metrics**: {', '.join(metrics)}")
            
            report.append("")
        
        # Investment Insights
        report.append("## 💡 Investment Insights")
        
        if insights.get("recommendations"):
            report.append("### Asset-Specific Insights")
            for rec in insights["recommendations"]:
                report.append(f"- **{rec['asset']}**: {', '.join(rec['insights'][:3])}")
        
        if insights.get("opportunities"):
            report.append("### Opportunities")
            for opp in insights["opportunities"]:
                report.append(f"- {opp}")
        
        if insights.get("warnings"):
            report.append("### ⚠️ Warnings")
            for warning in insights["warnings"]:
                report.append(f"- {warning}")
        
        report.append("")
        
        # Strategy Reminder
        report.append("## 🎯 15% Annual Return Strategy Reminder")
        report.append("""
**Core Principles:**
1. **SP500 Foundation**: 70-80% in SPY for stable growth
2. **Tencent Growth**: 20-30% in Tencent for upside potential
3. **Long-term Focus**: Hold through market cycles
4. **Disciplined Rebalancing**: Maintain target allocations
5. **Contrarian Mindset**: Buy during fear, trim during greed

**Current Allocation Target:**
- SPY (SP500): 75%
- Tencent: 25%
- QQQ (Monitor only): 0%
""")
        
        report.append("")
        report.append("---")
        report.append("*This report is for informational purposes only. Past performance does not guarantee future results.*")
        report.append("*Always conduct your own research and consider consulting a financial advisor.*")
        
        full_report = "\n".join(report)
        logger.info(f"Generated daily report ({len(full_report)} characters)")
        
        return full_report
    
    def save_report(self, report: str, filename: str = None):
        """
        Save report to file.
        
        Args:
            report: Report string
            filename            filename: Output filename (defaults to timestamp)
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"reports/daily_report_{timestamp}.md"
        
        os.makedirs("reports", exist_ok=True)
        
        try:
            with open(filename, 'w') as f:
                f.write(report)
            logger.info(f"Report saved to {filename}")
            
            # Also save to latest report
            latest_file = "reports/latest_report.md"
            with open(latest_file, 'w') as f:
                f.write(report)
            logger.info(f"Latest report updated: {latest_file}")
            
        except Exception as e:
            logger.error(f"Error saving report: {e}")
    
    def run_daily_analysis(self):
        """Run complete daily analysis."""
        logger.info("=" * 60)
        logger.info("Starting Daily Investment Analysis")
        logger.info("=" * 60)
        
        try:
            # Step 1: Collect market data
            market_data = self.collect_market_data()
            
            # Step 2: Generate insights
            insights = self.generate_investment_insights(market_data)
            
            # Step 3: Generate report
            report = self.generate_daily_report(market_data, insights)
            
            # Step 4: Save report
            self.save_report(report)
            
            # Step 5: Print summary
            print("\n" + "=" * 60)
            print("DAILY ANALYSIS COMPLETE")
            print("=" * 60)
            print(f"Market Condition: {insights['market_condition']}")
            print(f"Portfolio Action: {insights['portfolio_action']}")
            print(f"Recommendations: {len(insights['recommendations'])}")
            print(f"Report saved to: reports/latest_report.md")
            print("=" * 60)
            
            # Also print asset summary
            print("\nAsset Summary:")
            for symbol, asset in market_data.get('assets', {}).items():
                if asset.get('current_price'):
                    print(f"  {symbol}: ${asset['current_price']:.2f} ({asset.get('daily_change', 0):+.2f}%)")
            
            return True
            
        except Exception as e:
            logger.error(f"Error in daily analysis: {e}")
            import traceback
            traceback.print_exc()
            return False


def main():
    """Main function to run the investment monitor."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Investment Monitor for SPY, QQQ, and Tencent")
    parser.add_argument("--config", "-c", help="Path to configuration file")
    parser.add_argument("--test", "-t", action="store_true", help="Run in test mode")
    parser.add_argument("--quick", "-q", action="store_true", help="Quick run (minimal data)")
    
    args = parser.parse_args()
    
    # Initialize monitor
    monitor = InvestmentMonitor(config_path=args.config)
    
    if args.test:
        logger.info("Running in test mode...")
        # Test data collection
        market_data = monitor.collect_market_data()
        print(f"\nTest completed. Collected data for {len(market_data['assets'])} assets.")
        
        # Print test results
        for symbol, asset in market_data['assets'].items():
            if asset.get('current_price'):
                print(f"{symbol}: ${asset['current_price']:.2f}")
        
    else:
        # Run full daily analysis
        success = monitor.run_daily_analysis()
        
        if success:
            logger.info("Daily analysis completed successfully")
            sys.exit(0)
        else:
            logger.error("Daily analysis failed")
            sys.exit(1)


if __name__ == "__main__":
    main()
