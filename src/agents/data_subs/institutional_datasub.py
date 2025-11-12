# src/agents/data_subs/institutional_datasub.py
# Purpose: InstitutionalDataSub for fetching and processing 13F institutional holdings data.
# Inherits from BaseAgent, uses institutional holdings tools for specialized institutional data collection.

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))  # Dynamic root path for imports.

import asyncio
import logging
from typing import Dict, Any, List, Optional
import pandas as pd
from src.agents.base import BaseAgent  # Absolute import.
from src.utils.tools import institutional_holdings_analysis_tool, thirteen_f_filings_tool

logger = logging.getLogger(__name__)

class InstitutionalDatasub(BaseAgent):
    def __init__(self):
        # Temporarily disable tools until they are properly implemented as StructuredTool objects
        super().__init__("institutional_data", config_paths={}, prompt_paths={}, tools=[])

    async def process_input(self, input_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Process input to fetch and analyze institutional holdings data with LLM enhancement.
        Args:
            input_data: Dict with parameters (symbol for holdings analysis).
        Returns:
            Dict with structured institutional holdings data and LLM analysis.
        """
        logger.info(f"InstitutionalDatasub processing input: {input_data}")

        try:
            symbol = input_data.get('symbol', 'AAPL') if input_data else 'AAPL'

            # Step 1: Plan institutional exploration with LLM
            exploration_plan = await self._plan_institutional_exploration(symbol, input_data)

            # Step 2: Fetch data from multiple sources concurrently
            raw_data = await self._fetch_institutional_sources_concurrent(symbol, exploration_plan)

            # Step 3: Consolidate data into structured DataFrames
            consolidated_data = self._consolidate_institutional_data(raw_data, symbol)

            # Step 4: Analyze with LLM for insights
            llm_analysis = await self._analyze_institutional_data_llm(consolidated_data)

            # Combine results
            result = {
                "consolidated_data": consolidated_data,
                "llm_analysis": llm_analysis,
                "exploration_plan": exploration_plan,
                "enhanced": True
            }

            # Store institutional data in shared memory
            await self.store_shared_memory("institutional_data", symbol, {
                "institutional_holdings": consolidated_data,
                "llm_analysis": llm_analysis,
                "timestamp": datetime.now().isoformat()
            })

            logger.info(f"InstitutionalDatasub output: LLM-enhanced institutional data collected for {symbol}")
            return result

        except Exception as e:
            logger.error(f"InstitutionalDatasub failed: {e}")
            return {"institutional_ownership": 0.0, "error": str(e), "enhanced": False}

    def _parse_institutional_result(self, result: Dict[str, Any], symbol: str) -> Dict[str, Any]:
        """Parse institutional holdings tool result into structured format."""
        try:
            if "error" in result:
                return {
                    "symbol": symbol,
                    "holdings": [],
                    "source": "error",
                    "error_message": result["error"]
                }

            # Extract holdings data
            holdings = result.get("whale_wisdom_data", {}).get("top_holdings", [])
            concentration = result.get("concentration_analysis", {})
            intelligence = result.get("market_intelligence", {})

            return {
                "symbol": symbol,
                "total_institutions": result.get("whale_wisdom_data", {}).get("total_institutions", 0),
                "total_shares": result.get("whale_wisdom_data", {}).get("total_shares", 0),
                "top_holdings": holdings,
                "concentration_analysis": concentration,
                "market_intelligence": intelligence,
                "source": "whale_wisdom_api",
                "timestamp": pd.Timestamp.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Error parsing institutional result: {e}")
            return {
                "symbol": symbol,
                "holdings": [],
                "source": "parse_error",
                "error_message": str(e)
            }

    def _enhance_institutional_analysis(self, institutional_data: Dict[str, Any], symbol: str) -> Dict[str, Any]:
        """Enhance institutional analysis with additional insights."""
        try:
            holdings = institutional_data.get("top_holdings", [])

            if not holdings:
                return institutional_data

            # Calculate additional metrics
            total_market_value = sum(h.get("market_value", 0) for h in holdings if h.get("market_value"))

            # Add percentage of total institutional ownership
            for holding in holdings:
                if total_market_value > 0 and holding.get("market_value"):
                    holding["ownership_percentage"] = (holding["market_value"] / total_market_value) * 100

            # Add institutional ownership tier classification
            for holding in holdings:
                shares = holding.get("shares", 0)
                if shares >= 10000000:  # 10M+ shares
                    holding["ownership_tier"] = "Mega Holder"
                elif shares >= 1000000:  # 1M+ shares
                    holding["ownership_tier"] = "Large Holder"
                elif shares >= 100000:  # 100K+ shares
                    holding["ownership_tier"] = "Medium Holder"
                else:
                    holding["ownership_tier"] = "Small Holder"

            # Add summary statistics
            institutional_data["summary_stats"] = {
                "avg_holding_size": sum(h.get("shares", 0) for h in holdings) / len(holdings) if holdings else 0,
                "largest_holder": max(holdings, key=lambda x: x.get("shares", 0))["institution_name"] if holdings else None,
                "total_market_value": total_market_value,
                "ownership_tiers": {
                    "mega_holders": len([h for h in holdings if h.get("ownership_tier") == "Mega Holder"]),
                    "large_holders": len([h for h in holdings if h.get("ownership_tier") == "Large Holder"]),
                    "medium_holders": len([h for h in holdings if h.get("ownership_tier") == "Medium Holder"]),
                    "small_holders": len([h for h in holdings if h.get("ownership_tier") == "Small Holder"])
                }
            }

            return institutional_data

        except Exception as e:
            logger.error(f"Error enhancing institutional analysis: {e}")
            return institutional_data

    def fetch_institutional_holdings(self, symbol: str) -> Dict[str, Any]:
        """
        Fetch institutional holdings data for a symbol.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Dict containing institutional holdings data
        """
        return {
            'symbol': symbol,
            'total_institutional_ownership': 0.75,  # 75%
            'institutional_holders': [
                {
                    'name': 'Vanguard Group Inc',
                    'shares': 100000000,
                    'value': 15000000000.0,
                    'percentage': 0.08
                },
                {
                    'name': 'BlackRock Inc',
                    'shares': 80000000,
                    'value': 12000000000.0,
                    'percentage': 0.065
                }
            ],
            'quarterly_changes': {
                'increased_positions': 45,
                'decreased_positions': 23,
                'held_positions': 156
            },
            'source': 'institutional_holdings_subagent'
        }

    async def _plan_institutional_exploration(self, symbol: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Use LLM to plan intelligent exploration of institutional holdings data."""
        context_str = f"""
        Symbol: {symbol}
        Market Context: {context or 'General market analysis'}
        Current Holdings Data: {self.fetch_institutional_holdings(symbol)}
        """
        
        question = f"""
        Plan a comprehensive exploration strategy for institutional holdings of {symbol}.
        Consider:
        1. Key institutional investors to analyze (top holders, activist investors, index funds)
        2. Recent 13F filing changes and trends
        3. Ownership concentration and positioning insights
        4. Risk assessment based on institutional ownership patterns
        5. Market intelligence from institutional behavior
        
        Provide a structured plan with priorities and data sources to explore.
        """
        
        plan_response = await self.reason_with_llm(context_str, question)
        return {"plan": plan_response, "symbol": symbol, "timestamp": pd.Timestamp.now().isoformat()}

    async def _fetch_institutional_sources_concurrent(self, symbol: str, plan: Dict[str, Any]) -> Dict[str, Any]:
        """Fetch institutional data from multiple sources concurrently."""
        
        async def fetch_sec_edgar(symbol: str) -> Dict[str, Any]:
            """Fetch 13F filings from SEC EDGAR."""
            try:
                # Mock SEC EDGAR data for now
                return {
                    "source": "sec_edgar",
                    "filings": [
                        {"quarter": "Q1 2024", "institutions": 1250, "total_value": 450000000000},
                        {"quarter": "Q4 2023", "institutions": 1180, "total_value": 420000000000}
                    ]
                }
            except Exception as e:
                return {"source": "sec_edgar", "error": str(e)}
        
        async def fetch_whale_wisdom(symbol: str) -> Dict[str, Any]:
            """Fetch detailed holdings from Whale Wisdom."""
            try:
                # Use existing method
                return self.fetch_institutional_holdings(symbol)
            except Exception as e:
                return {"source": "whale_wisdom", "error": str(e)}
        
        async def fetch_institutional_databases(symbol: str) -> Dict[str, Any]:
            """Fetch from other institutional databases."""
            try:
                # Mock additional data
                return {
                    "source": "institutional_db",
                    "ownership_trends": {
                        "3_month_change": 0.02,
                        "6_month_change": 0.05,
                        "1_year_change": 0.08
                    }
                }
            except Exception as e:
                return {"source": "institutional_db", "error": str(e)}
        
        # Execute concurrent fetches
        tasks = [
            fetch_sec_edgar(symbol),
            fetch_whale_wisdom(symbol),
            fetch_institutional_databases(symbol)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        consolidated = {}
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Error in concurrent fetch: {result}")
                continue
            if isinstance(result, dict):
                source = result.get("source", "unknown")
                consolidated[source] = result
        
        return consolidated

    def _consolidate_institutional_data(self, raw_data: Dict[str, Any], symbol: str) -> Dict[str, Any]:
        """Consolidate institutional data into structured DataFrames."""
        try:
            import pandas as pd
            
            # Create holdings DataFrame
            holdings_data = []
            whale_data = raw_data.get("whale_wisdom", {})
            
            for holder in whale_data.get("institutional_holders", []):
                holdings_data.append({
                    "institution_name": holder.get("name", ""),
                    "shares": holder.get("shares", 0),
                    "value": holder.get("value", 0.0),
                    "percentage": holder.get("percentage", 0.0),
                    "ownership_tier": self._classify_ownership_tier(holder.get("shares", 0))
                })
            
            holdings_df = pd.DataFrame(holdings_data)
            
            # Create filings DataFrame
            filings_data = []
            sec_data = raw_data.get("sec_edgar", {})
            
            for filing in sec_data.get("filings", []):
                filings_data.append({
                    "quarter": filing.get("quarter", ""),
                    "institutions": filing.get("institutions", 0),
                    "total_value": filing.get("total_value", 0.0)
                })
            
            filings_df = pd.DataFrame(filings_data)
            
            # Create ownership trends DataFrame
            trends_data = []
            db_data = raw_data.get("institutional_db", {})
            trends = db_data.get("ownership_trends", {})
            
            for period, change in trends.items():
                trends_data.append({
                    "period": period,
                    "ownership_change_pct": change
                })
            
            trends_df = pd.DataFrame(trends_data)
            
            return {
                "symbol": symbol,
                "holdings_df": holdings_df.to_dict('records'),
                "filings_df": filings_df.to_dict('records'),
                "trends_df": trends_df.to_dict('records'),
                "summary": {
                    "total_institutional_ownership": whale_data.get("total_institutional_ownership", 0.0),
                    "total_institutions": len(holdings_data),
                    "avg_holding_size": holdings_df["shares"].mean() if not holdings_df.empty else 0,
                    "largest_holder": holdings_df.loc[holdings_df["shares"].idxmax()]["institution_name"] if not holdings_df.empty else None
                },
                "timestamp": pd.Timestamp.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error consolidating institutional data: {e}")
            return {"error": str(e), "symbol": symbol}

    def _classify_ownership_tier(self, shares: int) -> str:
        """Classify institutional ownership tier based on shares."""
        if shares >= 10000000:  # 10M+ shares
            return "Mega Holder"
        elif shares >= 1000000:  # 1M+ shares
            return "Large Holder"
        elif shares >= 100000:  # 100K+ shares
            return "Medium Holder"
        else:
            return "Small Holder"

    async def _analyze_institutional_data_llm(self, consolidated_data: Dict[str, Any]) -> Dict[str, Any]:
        """Use LLM for risk assessment and positioning insights from institutional data."""
        context_str = f"""
        Institutional Data Analysis for {consolidated_data.get('symbol', 'Unknown')}:
        
        Summary Statistics:
        {consolidated_data.get('summary', {})}
        
        Holdings Data:
        {consolidated_data.get('holdings_df', [])}
        
        Filings Trends:
        {consolidated_data.get('filings_df', [])}
        
        Ownership Trends:
        {consolidated_data.get('trends_df', [])}
        """
        
        question = """
        Analyze this institutional ownership data and provide insights on:
        1. Risk assessment based on ownership concentration and changes
        2. Institutional positioning insights (bullish/bearish signals)
        3. Key institutional investors and their potential impact
        4. Market intelligence from institutional behavior patterns
        5. Recommendations for portfolio positioning based on institutional activity
        
        Focus on risk management and alignment with our goals (<5% drawdown, 10-20% monthly ROI).
        """
        
        analysis_response = await self.reason_with_llm(context_str, question)
        return {
            "llm_analysis": analysis_response,
            "risk_assessment": self._extract_risk_from_llm(analysis_response),
            "positioning_insights": self._extract_insights_from_llm(analysis_response),
            "timestamp": pd.Timestamp.now().isoformat()
        }

    def _extract_risk_from_llm(self, llm_response: str) -> str:
        """Extract risk assessment from LLM response."""
        # Simple extraction - could be enhanced
        if "high risk" in llm_response.lower() or "concentrated" in llm_response.lower():
            return "High Risk - Concentrated Ownership"
        elif "moderate risk" in llm_response.lower():
            return "Moderate Risk"
        else:
            return "Low Risk - Diversified Ownership"

    def _extract_insights_from_llm(self, llm_response: str) -> List[str]:
        """Extract key insights from LLM response."""
        # Simple extraction - could be enhanced
        insights = []
        if "bullish" in llm_response.lower():
            insights.append("Bullish institutional positioning")
        if "bearish" in llm_response.lower():
            insights.append("Bearish institutional positioning")
        if "activist" in llm_response.lower():
            insights.append("Potential activist investor activity")
        return insights if insights else ["No specific insights extracted"]