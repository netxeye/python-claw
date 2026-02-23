#!/usr/bin/env python3
"""
Weekly Portfolio Scanner
Analyzes portfolio weekly and provides adjustment recommendations.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Tuple, Optional
import yfinance as yf
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WeeklyPortfolioScanner:
    """Weekly portfolio analysis and adjustment scanner."""
    
    def __init__(self):
        """Initialize weekly scanner."""
        self.target_allocation = {
            "VOO": 0.45,  # SP500 Core
            "VGT": 0.20,  # Technology
            "XLP": 0.15,  # Consumer Staples
            "XLU": 0.10,  # Utilities
            "FXI": 0.05,  # China Large-Cap
            "KBA": 0.05   # China A-Shares
        }
        
        self.rebalance_threshold = 0.05  # 5% drift triggers rebalance
        self.risk_parameters = {
            "max_china_exposure": 0.25,
            "min_sp500_exposure": 0.40,
            "max_tech_exposure": 0.25,
            "max_drawdown_alert": 0.10,  # 10% weekly drawdown
            "volatility_alert": 0.25     # 25% annualized volatility
        }
        
        logger.info("Weekly Portfolio Scanner initialized")
    
    def run_weekly_scan(self, current_portfolio: Dict) -> Dict:
        """
        Run complete weekly portfolio scan.
        
        Args:
            current_portfolio: Current portfolio data
            
        Returns:
            Scan results with recommendations
        """
        logger.info("Running weekly portfolio scan...")
        
        scan_results = {
            "scan_date": datetime.now().isoformat(),
            "market_analysis": {},
            "portfolio_analysis": {},
            "risk_assessment": {},
            "recommendations": {},
            "action_summary": {}
        }
        
        try:
            # Step 1: Analyze market conditions
            scan_results["market_analysis"] = self.analyze_market_conditions()
            
            # Step 2: Analyze current portfolio
            scan_results["portfolio_analysis"] = self.analyze_portfolio(current_portfolio)
            
            # Step 3: Assess risks
            scan_results["risk_assessment"] = self.assess_risks(
                scan_results["portfolio_analysis"],
                scan_results["market_analysis"]
            )
            
            # Step 4: Generate recommendations
            scan_results["recommendations"] = self.generate_recommendations(
                scan_results["portfolio_analysis"],
                scan_results["market_analysis"],
                scan_results["risk_assessment"]
            )
            
            # Step 5: Create action summary
            scan_results["action_summary"] = self.create_action_summary(
                scan_results["recommendations"]
            )
            
            logger.info("Weekly scan completed successfully")
            
        except Exception as e:
            logger.error(f"Error in weekly scan: {e}")
            scan_results["error"] = str(e)
        
        return scan_results
    
    def analyze_market_conditions(self) -> Dict:
        """Analyze current market conditions."""
        logger.info("Analyzing market conditions...")
        
        market_data = {
            "timestamp": datetime.now().isoformat(),
            "indicators": {},
            "sentiment": "neutral",
            "risk_level": "medium",
            "recommended_action": "monitor"
        }
        
        try:
            # Get key market indicators
            indicators = {
                "SPY": "S&P 500",
                "QQQ": "NASDAQ",
                "VIX": "Volatility Index",
                "TLT": "20+ Year Treasury",
                "GLD": "Gold"
            }
            
            for symbol, name in indicators.items():
                try:
                    ticker = yf.Ticker(symbol)
                    info = ticker.info
                    
                    market_data["indicators"][symbol] = {
                        "name": name,
                        "price": info.get("regularMarketPrice"),
                        "change": info.get("regularMarketChangePercent"),
                        "volume": info.get("regularMarketVolume"),
                        "52_week_high": info.get("fiftyTwoWeekHigh"),
                        "52_week_low": info.get("fiftyTwoWeekLow")
                    }
                    
                except Exception as e:
                    logger.warning(f"Error fetching {symbol}: {e}")
            
            # Analyze VIX for market sentiment
            vix_data = market_data["indicators"].get("VIX", {})
            vix_level = vix_data.get("price")
            
            if vix_level:
                if vix_level > 30:
                    market_data["sentiment"] = "fearful"
                    market_data["risk_level"] = "high"
                    market_data["recommended_action"] = "defensive"
                elif vix_level < 15:
                    market_data["sentiment"] = "greedy"
                    market_data["risk_level"] = "low"
                    market_data["recommended_action"] = "opportunistic"
            
            # Analyze SPY trend
            spy_data = market_data["indicators"].get("SPY", {})
            spy_price = spy_data.get("price")
            spy_52w_high = spy_data.get("52_week_high")
            
            if spy_price and spy_52w_high:
                from_high = (spy_price - spy_52w_high) / spy_52w_high
                if from_high < -0.10:  # More than 10% from high
                    market_data["sentiment"] = "correcting"
                    market_data["risk_level"] = "elevated"
            
            # Analyze bond yields (inverse relationship with TLT)
            tlt_data = market_data["indicators"].get("TLT", {})
            tlt_change = tlt_data.get("change")
            
            if tlt_change and tlt_change > 0.02:  # TLT up >2%
                market_data["sentiment"] = "defensive"
                market_data["recommended_action"] = "caution"
            
            logger.info(f"Market sentiment: {market_data['sentiment']}, Risk level: {market_data['risk_level']}")
            
        except Exception as e:
            logger.error(f"Error analyzing market conditions: {e}")
            market_data["error"] = str(e)
        
        return market_data
    
    def analyze_portfolio(self, portfolio: Dict) -> Dict:
        """Analyze current portfolio state."""
        logger.info("Analyzing portfolio...")
        
        analysis = {
            "current_allocation": {},
            "vs_target": {},
            "performance": {},
            "concentration": {},
            "drift_analysis": {}
        }
        
        try:
            # Extract current allocation
            total_value = portfolio.get("total_value", 0)
            
            for symbol, data in portfolio.get("allocation", {}).items():
                current_value = data.get("value", 0)
                current_weight = current_value / total_value if total_value > 0 else 0
                
                analysis["current_allocation"][symbol] = {
                    "value": current_value,
                    "weight": current_weight,
                    "target_weight": self.target_allocation.get(symbol, 0),
                    "return": data.get("return", 0)
                }
            
            # Calculate vs target
            for symbol, data in analysis["current_allocation"].items():
                current_weight = data["weight"]
                target_weight = data["target_weight"]
                drift = current_weight - target_weight
                
                analysis["vs_target"][symbol] = {
                    "current": current_weight,
                    "target": target_weight,
                    "drift": drift,
                    "drift_percent": drift / target_weight if target_weight > 0 else 0,
                    "action_needed": abs(drift) > self.rebalance_threshold
                }
            
            # Calculate performance metrics
            if "performance_history" in portfolio:
                history = portfolio["performance_history"]
                if len(history) >= 5:  # At least 5 days
                    values = [point["value"] for point in history[-5:]]  # Last 5 days
                    weekly_return = (values[-1] - values[0]) / values[0] if values[0] > 0 else 0
                    
                    analysis["performance"]["weekly_return"] = weekly_return
                    analysis["performance"]["weekly_volatility"] = np.std([
                        point.get("daily_return", 0) for point in history[-5:]
                    ])
            
            # Concentration analysis
            weights = [data["weight"] for data in analysis["current_allocation"].values()]
            if weights:
                analysis["concentration"] = {
                    "herfindahl_index": sum(w**2 for w in weights),  # HHI
                    "top_3_concentration": sum(sorted(weights, reverse=True)[:3]),
                    "max_single_weight": max(weights) if weights else 0
                }
            
            # Drift analysis
            drifts = [abs(data["drift"]) for data in analysis["vs_target"].values()]
            analysis["drift_analysis"] = {
                "max_drift": max(drifts) if drifts else 0,
                "avg_drift": np.mean(drifts) if drifts else 0,
                "drift_above_threshold": sum(1 for d in drifts if d > self.rebalance_threshold),
                "total_portfolio_drift": np.sqrt(sum(d**2 for d in drifts)) if drifts else 0
            }
            
            logger.info(f"Portfolio analysis: {analysis['drift_analysis']['drift_above_threshold']} ETFs need rebalancing")
            
        except Exception as e:
            logger.error(f"Error analyzing portfolio: {e}")
            analysis["error"] = str(e)
        
        return analysis
    
    def assess_risks(self, portfolio_analysis: Dict, market_analysis: Dict) -> Dict:
        """Assess portfolio risks."""
        logger.info("Assessing portfolio risks...")
        
        risk_assessment = {
            "market_risk": {},
            "concentration_risk": {},
            "sector_risk": {},
            "country_risk": {},
            "liquidity_risk": {},
            "overall_risk_score": 0
        }
        
        try:
            # Market risk
            market_sentiment = market_analysis.get("sentiment", "neutral")
            market_risk_level = market_analysis.get("risk_level", "medium")
            
            risk_assessment["market_risk"] = {
                "sentiment": market_sentiment,
                "risk_level": market_risk_level,
                "vix_level": market_analysis.get("indicators", {}).get("VIX", {}).get("price"),
                "recommendation": market_analysis.get("recommended_action", "monitor")
            }
            
            # Concentration risk
            concentration = portfolio_analysis.get("concentration", {})
            hhi = concentration.get("herfindahl_index", 0)
            top_3 = concentration.get("top_3_concentration", 0)
            
            concentration_risk = "low"
            if hhi > 0.25:
                concentration_risk = "high"
            elif hhi > 0.15:
                concentration_risk = "medium"
            
            risk_assessment["concentration_risk"] = {
                "herfindahl_index": hhi,
                "top_3_concentration": top_3,
                "risk_level": concentration_risk,
                "recommendation": "diversify" if concentration_risk == "high" else "monitor"
            }
            
            # Sector risk (simplified)
            current_allocation = portfolio_analysis.get("current_allocation", {})
            
            # Calculate sector exposures
            sector_exposures = {
                "SP500": sum(data["weight"] for symbol, data in current_allocation.items() 
                           if symbol in ["VOO", "SPY", "IVV"]),
                "Technology": sum(data["weight"] for symbol, data in current_allocation.items()
                                if symbol in ["VGT", "XLK", "FTEC"]),
                "China": sum(data["weight"] for symbol, data in current_allocation.items()
                           if symbol in ["FXI", "KBA", "MCHI"]),
                "Defensive": sum(data["weight"] for symbol, data in current_allocation.items()
                               if symbol in ["XLP", "XLU", "VDC", "VPU"])
            }
            
            # Check sector limits
            sector_risks = []
            if sector_exposures["Technology"] > self.risk_parameters["max_tech_exposure"]:
                sector_risks.append("Technology exposure too high")
            if sector_exposures["China"] > self.risk_parameters["max_china_exposure"]:
                sector_risks.append("China exposure too high")
            if sector_exposures["SP500"] < self.risk_parameters["min_sp500_exposure"]:
                sector_risks.append("SP500 exposure too low")
            
            risk_assessment["sector_risk"] = {
                "exposures": sector_exposures,
                "risks": sector_risks,
                "risk_level": "high" if sector_risks else "low",
                "recommendation": "rebalance" if sector_risks else "maintain"
            }
            
            # Country risk (focus on China)
            china_exposure = sector_exposures["China"]
            country_risk = "medium" if china_exposure > 0.15 else "low"
            
            risk_assessment["country_risk"] = {
                "china_exposure": china_exposure,
                "risk_level": country_risk,
                "recommendation": "reduce" if china_exposure > 0.20 else "monitor"
            }
            
            # Liquidity risk (simplified - check if ETFs are liquid)
            liquidity_risk = "low"  # Assuming major ETFs are liquid
            
            risk_assessment["liquidity_risk"] = {
                "risk_level": liquidity_risk,
                "recommendation": "monitor"
            }
            
            # Calculate overall risk score (0-100, higher = riskier)
            risk_score = 50  # Base score
            
            # Adjust based on factors
            if market_risk_level == "high":
                risk_score += 20
            elif market_risk_level == "low":
                risk_score -= 10
            
            if concentration_risk == "high":
                risk_score += 15
            elif concentration_risk == "medium":
                risk_score += 5
            
            if risk_assessment["sector_risk"]["risk_level"] == "high":
                risk_score += 15
            
            if country_risk == "high":
                risk_score += 10
            
            risk_score = max(0, min(100, risk_score))
            risk_assessment["overall_risk_score"] = risk_score
            
            logger.info(f"Overall risk score: {risk_score}/100")
            
        except Exception as e:
            logger.error(f"Error assessing risks: {e}")
            risk_assessment["error"] = str(e)
        
        return risk_assessment
    
    def generate_recommendations(self, portfolio_analysis: Dict, 
                                market_analysis: Dict, risk_assessment: Dict) -> Dict:
        """Generate investment recommendations."""
        logger.info("Generating recommendations...")
        
        recommendations = {
            "rebalancing": [],
            "market_timing": [],
            "risk_management": [],
            "cash_management": [],
            "summary": {}
        }
        
        try:
            # Rebalancing recommendations
            vs_target = portfolio_analysis.get("vs_target", {})
            
            for symbol, data in vs_target.items():
                if data.get("action_needed", False):
                    drift = data["drift"]
                    current = data["current"]
                    target = data["target"]
                    
                    action = "REDUCE" if drift > 0 else "INCREASE"
                    amount_pct = abs(drift)
                    
                    recommendations["rebalancing"].append({
                        "symbol": symbol,
                        "action": action,
                        "current_weight": f"{current*100:.1f}%",
                        "target_weight": f"{target*100:.1f}%",
                        "adjustment": f"{amount_pct*100:.1f}%",
                        "priority": "high" if abs(drift) > 0.10 else "medium",
                        "reason": f"Allocation drift: {drift*100:+.1f}% from target"
                    })
            
            # Market timing recommendations
            market_sentiment = market_analysis.get("sentiment", "neutral")
            market_action = market_analysis.get("recommended_action", "monitor")
            
            if market_sentiment == "fearful":
                recommendations["market_timing"].append({
                    "action": "INCREASE_DEFENSIVE",
                    "details": "Market fearful - increase defensive holdings (XLP, XLU)",
                    "priority": "high",
                    "etfs": ["XLP", "XLU"]
                })
                recommendations["market_timing"].append({
                    "action": "REDUCE_TECH",
                    "details": "Reduce technology exposure during high volatility",
                    "priority": "medium",
                    "etfs": ["VGT"]
                })
            
            elif market_sentiment == "greedy":
                recommendations["market_timing"].append({
                    "action": "INCREASE_GROWTH",
                    "details": "Market greedy - increase growth holdings",
                    "priority": "medium",
                    "etfs": ["VGT"]
                })
            
            # Risk management recommendations
            risk_score = risk_assessment.get("overall_risk_score", 50)
            
            if risk_score > 70:
                recommendations["risk_management"].append({
                    "action": "REDUCE_RISK",
                    "details": f"High risk score ({risk_score}/100) - reduce overall risk",
                    "priority": "high",
                    "suggestions": [
                        "Increase cash position",
                        "Reduce China exposure",
                        "Add defensive sectors"
                    ]
                })
            
            # Check China exposure
            china_exposure = risk_assessment.get("country_risk", {}).get("china_exposure", 0)
            if china_exposure > self.risk_parameters["max            _china_exposure"]:
                recommendations["risk_management"].append({
                    "action": "REDUCE_CHINA",
                    "details": f"China exposure {china_exposure*100:.1f}% exceeds limit",
                    "priority": "medium",
                    "suggestions": [
                        "Reduce FXI and KBA positions",
                        "Reallocate to SP500 or defensive sectors"
                    ]
                })
            
            # Cash management recommendations
            market_risk = risk_assessment.get("market_risk", {}).get("risk_level", "medium")
            
            if market_risk == "high":
                recommendations["cash_management"].append({
                    "action": "INCREASE_CASH",
                    "details": "High market risk - increase cash position",
                    "priority": "high",
                    "target_cash": "10-15%",
                    "reason": "Dry powder for buying opportunities"
                })
            elif market_risk == "low":
                recommendations["cash_management"].append({
                    "action": "DEPLOY_CASH",
                    "details": "Low market risk - deploy excess cash",
                    "priority": "medium",
                    "target_cash": "5%",
                    "reason": "Take advantage of favorable conditions"
                })
            
            # Create summary
            total_recommendations = (
                len(recommendations["rebalancing"]) +
                len(recommendations["market_timing"]) +
                len(recommendations["risk_management"]) +
                len(recommendations["cash_management"])
            )
            
            high_priority = sum(1 for category in recommendations.values() 
                              if isinstance(category, list) 
                              for rec in category if rec.get("priority") == "high")
            
            recommendations["summary"] = {
                "total_recommendations": total_recommendations,
                "high_priority": high_priority,
                "action_required": total_recommendations > 0,
                "primary_focus": self._determine_primary_focus(recommendations),
                "next_steps": self._generate_next_steps(recommendations)
            }
            
            logger.info(f"Generated {total_recommendations} recommendations ({high_priority} high priority)")
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            recommendations["error"] = str(e)
        
        return recommendations
    
    def _determine_primary_focus(self, recommendations: Dict) -> str:
        """Determine primary focus area."""
        if not recommendations:
            return "MONITOR"
        
        # Check for high priority items
        high_priority_items = []
        for category in ["rebalancing", "market_timing", "risk_management", "cash_management"]:
            if category in recommendations:
                high_priority_items.extend([
                    rec for rec in recommendations[category] 
                    if rec.get("priority") == "high"
                ])
        
        if high_priority_items:
            # Categorize high priority items
            actions = [item.get("action", "") for item in high_priority_items]
            
            if any("REDUCE_RISK" in action for action in actions):
                return "RISK_REDUCTION"
            elif any("INCREASE_CASH" in action for action in actions):
                return "CASH_BUILDING"
            elif any("REBALANCE" in action for action in actions):
                return "REBALANCING"
            else:
                return "ADJUSTMENT"
        
        # No high priority, check medium priority
        medium_priority_items = []
        for category in ["rebalancing", "market_timing", "risk_management", "cash_management"]:
            if category in recommendations:
                medium_priority_items.extend([
                    rec for rec in recommendations[category] 
                    if rec.get("priority") == "medium"
                ])
        
        if medium_priority_items:
            return "OPTIMIZATION"
        
        return "MONITOR"
    
    def _generate_next_steps(self, recommendations: Dict) -> List[str]:
        """Generate actionable next steps."""
        next_steps = []
        
        # Add high priority steps first
        high_priority_items = []
        for category in ["rebalancing", "market_timing", "risk_management", "cash_management"]:
            if category in recommendations:
                high_priority_items.extend([
                    rec for rec in recommendations[category] 
                    if rec.get("priority") == "high"
                ])
        
        for item in high_priority_items:
            action = item.get("action", "")
            details = item.get("details", "")
            next_steps.append(f"🚨 {action}: {details}")
        
        # Add medium priority steps
        medium_priority_items = []
        for category in ["rebalancing", "market_timing", "risk_management", "cash_management"]:
            if category in recommendations:
                medium_priority_items.extend([
                    rec for rec in recommendations[category] 
                    if rec.get("priority") == "medium"
                ])
        
        for item in medium_priority_items[:3]:  # Limit to top 3
            action = item.get("action", "")
            details = item.get("details", "")
            next_steps.append(f"📊 {action}: {details}")
        
        # If no specific steps, add monitoring step
        if not next_steps:
            next_steps.append("👁️  Monitor portfolio - no immediate actions needed")
        
        return next_steps
    
    def create_action_summary(self, recommendations: Dict) -> Dict:
        """Create actionable summary for quick review."""
        summary = {
            "timestamp": datetime.now().isoformat(),
            "action_required": recommendations.get("summary", {}).get("action_required", False),
            "primary_focus": recommendations.get("summary", {}).get("primary_focus", "MONITOR"),
            "key_actions": [],
            "quick_decisions": [],
            "timeline": "this_week"
        }
        
        try:
            # Extract key actions
            for category in ["rebalancing", "market_timing", "risk_management", "cash_management"]:
                if category in recommendations:
                    for rec in recommendations[category][:2]:  # Top 2 per category
                        summary["key_actions"].append({
                            "category": category,
                            "action": rec.get("action"),
                            "priority": rec.get("priority"),
                            "details": rec.get("details")
                        })
            
            # Create quick decisions
            if recommendations.get("summary", {}).get("high_priority", 0) > 0:
                summary["quick_decisions"].append("Execute high-priority rebalancing this week")
            
            market_timing = recommendations.get("market_timing", [])
            if any(rec.get("priority") == "high" for rec in market_timing):
                summary["quick_decisions"].append("Adjust allocation based on market sentiment")
            
            risk_items = recommendations.get("risk_management", [])
            if any("REDUCE_RISK" in rec.get("action", "") for rec in risk_items):
                summary["quick_decisions"].append("Implement risk reduction measures")
            
            # If no specific decisions, add monitoring
            if not summary["quick_decisions"]:
                summary["quick_decisions"].append("Continue monitoring - portfolio is well-positioned")
                summary["timeline"] = "next_week"
            
        except Exception as e:
            logger.error(f"Error creating action summary: {e}")
            summary["error"] = str(e)
        
        return summary
    
    def save_scan_results(self, results: Dict, filename_prefix: str = None):
        """Save scan results to file."""
        if filename_prefix is None:
            timestamp = datetime.now().strftime("%Y%m%d")
            filename_prefix = f"weekly_scan_{timestamp}"
        
        import os
        os.makedirs("weekly_scans", exist_ok=True)
        
        # Save full results
        json_file = f"weekly_scans/{filename_prefix}_full.json"
        with open(json_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save action summary
        if "action_summary" in results:
            summary_file = f"weekly_scans/{filename_prefix}_summary.json"
            with open(summary_file, 'w') as f:
                json.dump(results["action_summary"], f, indent=2, default=str)
        
        # Save human-readable report
        report_file = f"weekly_scans/{filename_prefix}_report.txt"
        self._generate_text_report(results, report_file)
        
        logger.info(f"Scan results saved to weekly_scans/{filename_prefix}_*")
    
    def _generate_text_report(self, results: Dict, filename: str):
        """Generate human-readable text report."""
        report = []
        
        report.append("=" * 80)
        report.append("WEEKLY PORTFOLIO SCAN REPORT")
        report.append("=" * 80)
        report.append(f"Scan Date: {results.get('scan_date', 'N/A')}")
        report.append("")
        
        # Market Analysis
        market = results.get("market_analysis", {})
        report.append("MARKET ANALYSIS")
        report.append("-" * 40)
        report.append(f"Sentiment: {market.get('sentiment', 'N/A')}")
        report.append(f"Risk Level: {market.get('risk_level', 'N/A')}")
        report.append(f"Recommended Action: {market.get('recommended_action', 'N/A')}")
        report.append("")
        
        # Portfolio Analysis
        portfolio = results.get("portfolio_analysis", {})
        drift = portfolio.get("drift_analysis", {})
        report.append("PORTFOLIO ANALYSIS")
        report.append("-" * 40)
        report.append(f"ETFs needing rebalance: {drift.get('drift_above_threshold', 0)}")
        report.append(f"Max allocation drift: {drift.get('max_drift', 0)*100:.1f}%")
        report.append(f"Total portfolio drift: {drift.get('total_portfolio_drift', 0)*100:.1f}%")
        report.append("")
        
        # Risk Assessment
        risk = results.get("risk_assessment", {})
        report.append("RISK ASSESSMENT")
        report.append("-" * 40)
        report.append(f"Overall Risk Score: {risk.get('overall_risk_score', 0)}/100")
        
        market_risk = risk.get("market_risk", {})
        report.append(f"Market Risk: {market_risk.get('risk_level', 'N/A')}")
        
        sector_risk = risk.get("sector_risk", {})
        if sector_risk.get("risks"):
            report.append("Sector Risks:")
            for risk_msg in sector_risk["risks"]:
                report.append(f"  • {risk_msg}")
        report.append("")
        
        # Recommendations Summary
        recs = results.get("recommendations", {})
        summary = recs.get("summary", {})
        report.append("RECOMMENDATIONS SUMMARY")
        report.append("-" * 40)
        report.append(f"Total Recommendations: {summary.get('total_recommendations', 0)}")
        report.append(f"High Priority: {summary.get('high_priority', 0)}")
        report.append(f"Primary Focus: {summary.get('primary_focus', 'N/A')}")
        report.append("")
        
        # Key Recommendations
        report.append("KEY RECOMMENDATIONS")
        report.append("-" * 40)
        
        categories = ["rebalancing", "market_timing", "risk_management", "cash_management"]
        for category in categories:
            if category in recs and recs[category]:
                report.append(f"\n{category.upper()}:")
                for rec in recs[category][:3]:  # Top 3 per category
                    priority = rec.get("priority", "").upper()
                    action = rec.get("action", "")
                    details = rec.get("details", "")
                    report.append(f"  [{priority}] {action}: {details}")
        
        report.append("")
        
        # Action Summary
        action = results.get("action_summary", {})
        report.append("ACTION SUMMARY")
        report.append("-" * 40)
        report.append(f"Action Required: {'YES' if action.get('action_required') else 'NO'}")
        report.append(f"Primary Focus: {action.get('primary_focus', 'N/A')}")
        report.append(f"Timeline: {action.get('timeline', 'N/A')}")
        report.append("")
        
        if action.get("quick_decisions"):
            report.append("QUICK DECISIONS:")
            for decision in action["quick_decisions"]:
                report.append(f"  • {decision}")
        
        report.append("")
        report.append("=" * 80)
        report.append("END OF REPORT")
        report.append("=" * 80)
        
        # Write to file
        with open(filename, 'w') as f:
            f.write("\n".join(report))


def main():
    """Test the weekly scanner."""
    print("=" * 80)
    print("WEEKLY PORTFOLIO SCANNER TEST")
    print("=" * 80)
    
    # Create sample portfolio data
    sample_portfolio = {
        "total_value": 115000,
        "allocation": {
            "VOO": {"value": 50000, "return": 0.12},
            "VGT": {"value": 25000, "return": 0.18},
            "XLP": {"value": 15000, "return": 0.08},
            "XLU": {"value": 10000, "return": 0.07},
            "FXI": {"value": 10000, "return": 0.10},
            "KBA": {"value": 5000, "return": 0.15}
        },
        "performance_history": [
            {"date": "2024-01-01", "value": 100000, "daily_return": 0.001},
            {"date": "2024-01-02", "value": 101000, "daily_return": 0.002},
            {"date": "2024-01-03", "value": 102500, "daily_return": 0.0015},
            {"date": "2024-01-04", "value": 103000, "daily_return": 0.0005},
            {"date": "2024-01-05", "value": 104000, "daily_return": 0.001},
            {"date": "2024-01-06", "value": 105000, "daily_return": 0.001},
            {"date": "2024-01-07", "value": 115000, "daily_return": 0.010}
        ]
    }
    
    # Initialize scanner
    scanner = WeeklyPortfolioScanner()
    
    # Run scan
    print("\nRunning weekly scan...")
    results = scanner.run_weekly_scan(sample_portfolio)
    
    if "error" not in results:
        # Print summary
        action_summary = results.get("action_summary", {})
        print(f"\nAction Required: {'YES' if action_summary.get('action_required') else 'NO'}")
        print(f"Primary Focus: {action_summary.get('primary_focus', 'N/A')}")
        
        # Print key recommendations
        recs = results.get("recommendations", {})
        print(f"\nTotal Recommendations: {recs.get('summary', {}).get('total_recommendations', 0)}")
        
        # Save results
        scanner.save_scan_results(results)
        
        print("\n" + "=" * 80)
        print("SCAN COMPLETE")
        print("=" * 80)
        print("Results saved to weekly_scans/ directory")
        print("Check the report for detailed recommendations")
    else:
        print(f"\nScan failed: {results.get('error')}")


if __name__ == "__main__":
    main()
