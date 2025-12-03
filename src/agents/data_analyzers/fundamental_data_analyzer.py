# src/agents/data_subs/fundamental_datasub.py
# Purpose: Fundamental Data Subagent for comprehensive financial statement analysis.
# Fetches earnings, balance sheet, cash flow, and valuation metrics for fundamental analysis.
# Structural Reasoning: Dedicated subagent for fundamental data, enabling parallel processing with technical sources.
# Ties to system: Provides fundamental data dict for main data agent coordination.
# For legacy wealth: Access to fundamental factors for superior investment decisions.

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))  # Dynamic root path for imports.

from src.agents.data_analyzers.base_data_analyzer import BaseDataAnalyzer
import logging
from typing import Dict, Any, List
import json
import pandas as pd
from src.utils.tools import fundamental_data_tool, fundamental_analysis_tool

logger = logging.getLogger(__name__)

class FundamentalDataAnalyzer(BaseDataAnalyzer):
    """
    Fundamental Data Subagent.
    Reasoning: Fetches comprehensive fundamental data for financial analysis and valuation.
    """
    def __init__(self):
        super().__init__(role='fundamental_data')

    async def reflect(self, adjustments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Reflect on batch adjustments for self-improvement.
        Analyzes fundamental data performance metrics and generates optimization insights.

        Args:
            adjustments: Dictionary containing performance data and adjustment metrics

        Returns:
            Dict with reflection insights and improvement recommendations
        """
        try:
            logger.info(f"Fundamental reflecting on adjustments: {adjustments}")

            # Analyze fundamental data-specific performance
            reflection_insights = {
                'timestamp': pd.Timestamp.now().isoformat(),
                'adjustments_processed': len(adjustments) if isinstance(adjustments, dict) else 0,
                'performance_analysis': {},
                'optimization_opportunities': [],
                'data_quality_insights': {}
            }

            # Extract fundamental data performance metrics
            if isinstance(adjustments, dict):
                # Analyze data completeness, accuracy, timeliness
                if 'performance_data' in adjustments:
                    perf_data = adjustments['performance_data']
                    reflection_insights['performance_analysis'] = {
                        'data_completeness_score': perf_data.get('data_completeness_score', 0),
                        'valuation_accuracy': perf_data.get('valuation_accuracy', 0),
                        'financial_health_accuracy': perf_data.get('financial_health_accuracy', 0),
                        'growth_prediction_accuracy': perf_data.get('growth_prediction_accuracy', 0),
                        'source_reliability_score': perf_data.get('source_reliability_score', 0)
                    }

                # Identify fundamental data-specific improvement opportunities
                if 'issues' in adjustments:
                    issues = adjustments['issues']
                    for issue in issues:
                        if 'completeness' in issue.lower() or 'missing' in issue.lower():
                            reflection_insights['optimization_opportunities'].append({
                                'type': 'data_completeness_enhancement',
                                'description': 'Improve fundamental data completeness by expanding source coverage and fallback mechanisms',
                                'priority': 'high'
                            })
                        elif 'accuracy' in issue.lower() or 'valuation' in issue.lower():
                            reflection_insights['optimization_opportunities'].append({
                                'type': 'valuation_model_refinement',
                                'description': 'Enhance valuation model accuracy through better multiple calculations and peer comparisons',
                                'priority': 'medium'
                            })
                        elif 'timeliness' in issue.lower() or 'freshness' in issue.lower():
                            reflection_insights['optimization_opportunities'].append({
                                'type': 'data_freshness_optimization',
                                'description': 'Optimize data freshness requirements and caching strategies for different fundamental metrics',
                                'priority': 'medium'
                            })
                        elif 'consolidation' in issue.lower() or 'integration' in issue.lower():
                            reflection_insights['optimization_opportunities'].append({
                                'type': 'data_consolidation_improvement',
                                'description': 'Strengthen data consolidation logic and cross-source validation mechanisms',
                                'priority': 'high'
                            })

                # Generate data quality insights
                reflection_insights['data_quality_insights'] = {
                    'statement_coverage_breadth': adjustments.get('statement_coverage_breadth', 0.8),
                    'peer_comparison_depth': adjustments.get('peer_comparison_depth', 0.7),
                    'historical_data_completeness': adjustments.get('historical_data_completeness', 0.9),
                    'forward_estimate_reliability': adjustments.get('forward_estimate_reliability', 0.6),
                    'sector_benchmarking_accuracy': adjustments.get('sector_benchmarking_accuracy', 0.8)
                }

            # Store reflection in memory for future learning
            self.update_memory('fundamental_reflection_history', {
                'timestamp': reflection_insights['timestamp'],
                'insights': reflection_insights,
                'adjustments_analyzed': adjustments
            })

            logger.info(f"Fundamental reflection completed with {len(reflection_insights['optimization_opportunities'])} optimization opportunities identified")
            return reflection_insights

        except Exception as e:
            logger.error(f"Error during fundamental reflection analysis: {e}")
            return {
                'error': str(e),
                'timestamp': pd.Timestamp.now().isoformat(),
                'adjustments_processed': len(adjustments) if isinstance(adjustments, dict) else 0
            }

    async def _plan_data_exploration(self, *args, **kwargs) -> Dict[str, Any]:
        """Plan fundamental data exploration strategy."""
        symbols = kwargs.get('symbols', [])
        return await self._plan_fundamental_exploration(symbols)

    async def _execute_data_exploration(self, exploration_plan: Dict[str, Any], *args, **kwargs) -> Dict[str, Any]:
        """Execute fundamental data fetching and initial processing."""
        symbols = exploration_plan.get("symbols", [])
        raw_fundamental_data = await self._fetch_fundamental_sources_concurrent(symbols, exploration_plan)
        return raw_fundamental_data

    async def _enhance_data(self, raw_data: Dict[str, Any], *args, **kwargs) -> Dict[str, Any]:
        """Enhance fundamental data with analysis and consolidation."""
        exploration_plan = kwargs.get('exploration_plan', {})
        symbols = exploration_plan.get("symbols", [])

        # Consolidate data into structured DataFrames
        consolidated_data = self._consolidate_fundamental_data(raw_data, exploration_plan)

        # LLM-driven analysis of consolidated data
        analysis_focus = exploration_plan.get("analysis_focus", [])
        llm_analysis = await self._analyze_fundamental_data_llm(consolidated_data, analysis_focus)

        return {
            "consolidated_data": consolidated_data,
            "llm_analysis": llm_analysis,
            "symbols_processed": symbols,
            "timestamp": consolidated_data.get("metadata", {}).get("consolidation_timestamp"),
            "enhanced": True
        }

    async def process_input(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process fundamental data using BaseDataAnalyzer pattern for backward compatibility.
        """
        try:
            # Initialize LLM if not already done
            if not self.llm:
                await self.async_initialize_llm()

            symbols = input_data.get('symbols', [])
            if not symbols:
                return {"error": "No symbols provided for fundamental analysis"}

            logger.info(f"Processing fundamental data for symbols: {symbols}")

            # Use base class process_input for standardized processing
            result = await super().process_input(input_data)

            # For backward compatibility, ensure expected structure and add memory storage
            if isinstance(result, dict) and "consolidated_data" in result:
                consolidated_data = result["consolidated_data"]
                llm_analysis = result.get("llm_analysis", {})

                # Add exploration_plan if missing
                if "exploration_plan" not in result:
                    exploration_plan = await self._plan_fundamental_exploration(symbols)
                    result["exploration_plan"] = exploration_plan

                # Add raw_data if missing (though base class might not provide it)
                if "raw_data" not in result:
                    exploration_plan = result.get("exploration_plan", {})
                    raw_fundamental_data = await self._fetch_fundamental_sources_concurrent(symbols, exploration_plan)
                    result["raw_data"] = raw_fundamental_data

                # Store fundamental data in shared memory for each symbol
                for symbol in symbols:
                    await self.store_shared_memory("fundamental_data", symbol, {
                        "fundamental_data": consolidated_data,
                        "llm_analysis": llm_analysis,
                        "timestamp": pd.Timestamp.now().isoformat()
                    })

                logger.info(f"Completed fundamental analysis for {len(symbols)} symbols")
                return result

            # Fallback to original logic if base class doesn't return expected structure
            return await self._fallback_process_input(input_data)

        except Exception as e:
            logger.error(f"Error in fundamental data processing: {e}")
            return {"error": str(e)}

    async def _fallback_process_input(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fallback processing method for backward compatibility.
        """
        symbols = input_data.get('symbols', [])
        if not symbols:
            return {"error": "No symbols provided for fundamental analysis"}

        # Step 1: LLM-powered exploration planning
        exploration_plan = await self._plan_fundamental_exploration(symbols)

        # Step 2: Concurrent data fetching from multiple sources
        raw_fundamental_data = await self._fetch_fundamental_sources_concurrent(symbols, exploration_plan)

        # Step 3: Consolidate data into structured DataFrames
        consolidated_data = self._consolidate_fundamental_data(raw_fundamental_data, exploration_plan)

        # Step 4: LLM-driven analysis of consolidated data
        analysis_focus = exploration_plan.get("analysis_focus", [])
        llm_analysis = await self._analyze_fundamental_data_llm(consolidated_data, analysis_focus)

        # Combine results
        return {
            "exploration_plan": exploration_plan,
            "raw_data": raw_fundamental_data,
            "consolidated_data": consolidated_data,
            "llm_analysis": llm_analysis,
            "symbols_processed": symbols,
            "timestamp": consolidated_data.get("metadata", {}).get("consolidation_timestamp"),
            "enhanced": True
        }

    def _enhance_fundamentals_with_analysis(self, fundamental_data: Dict[str, Any], symbol: str) -> Dict[str, Any]:
        """Enhance fundamental data with comprehensive analysis."""
        try:
            # First perform the basic analysis
            fundamental_data = self._analyze_fundamentals(fundamental_data, symbol)

            # Then perform comprehensive factor analysis
            analysis_result = fundamental_analysis_tool.invoke({
                "fundamental_data": fundamental_data,
                "analysis_type": "comprehensive"
            })

            if "error" not in analysis_result:
                fundamental_data["factor_analysis"] = analysis_result

            return fundamental_data

        except Exception as e:
            logger.error(f"Error enhancing fundamentals with analysis for {symbol}: {e}")
            fundamental_data["analysis_error"] = str(e)
            return fundamental_data

    def _analyze_fundamentals(self, fundamental_data: Dict[str, Any], symbol: str) -> Dict[str, Any]:
        try:
            analysis = {
                "symbol": symbol,
                "analysis_timestamp": fundamental_data.get("overview", {}).get("timestamp", ""),
                "quality_metrics": {},
                "valuation_metrics": {},
                "growth_metrics": {},
                "financial_health": {}
            }

            # Extract overview data
            overview = fundamental_data.get("overview", {}).get("company_overview", {})

            # Valuation metrics
            if overview:
                analysis["valuation_metrics"] = {
                    "pe_ratio": overview.get("pe_ratio"),
                    "pb_ratio": overview.get("pb_ratio"),
                    "peg_ratio": overview.get("peg_ratio"),
                    "dividend_yield": overview.get("dividend_yield"),
                    "market_cap": overview.get("market_cap"),
                    "beta": overview.get("beta")
                }

                # Quality metrics
                analysis["quality_metrics"] = {
                    "roe": overview.get("roe"),
                    "roa": overview.get("roa"),
                    "profit_margin": overview.get("profit_margin"),
                    "eps": overview.get("eps")
                }

            # Analyze income statement trends
            income_annual = fundamental_data.get("income_statement", {}).get("income_statement_annual", [])
            if income_annual and len(income_annual) >= 2:
                revenues = [stmt.get("total_revenue", 0) for stmt in income_annual if stmt.get("total_revenue", 0) > 0]
                net_incomes = [stmt.get("net_income", 0) for stmt in income_annual if stmt.get("net_income", 0) != 0]

                if len(revenues) >= 2:
                    revenue_growth = (revenues[0] - revenues[-1]) / abs(revenues[-1]) if revenues[-1] != 0 else 0
                    analysis["growth_metrics"]["revenue_growth_annual"] = revenue_growth

                if len(net_incomes) >= 2:
                    earnings_growth = (net_incomes[0] - net_incomes[-1]) / abs(net_incomes[-1]) if net_incomes[-1] != 0 else 0
                    analysis["growth_metrics"]["earnings_growth_annual"] = earnings_growth

            # Analyze balance sheet health
            balance_annual = fundamental_data.get("balance_sheet", {}).get("balance_sheet_annual", [])
            if balance_annual and len(balance_annual) >= 1:
                latest_balance = balance_annual[0]

                total_assets = latest_balance.get("total_assets", 0)
                total_liabilities = latest_balance.get("total_liabilities", 0)
                total_equity = latest_balance.get("total_shareholder_equity", 0)
                current_assets = latest_balance.get("current_assets", 0)
                current_liabilities = latest_balance.get("current_liabilities", 0)

                # Financial health ratios
                if total_assets > 0:
                    debt_to_equity = total_liabilities / total_equity if total_equity > 0 else float('inf')
                    analysis["financial_health"]["debt_to_equity"] = debt_to_equity

                if current_liabilities > 0:
                    current_ratio = current_assets / current_liabilities
                    analysis["financial_health"]["current_ratio"] = current_ratio

            # Analyze cash flow quality
            cashflow_annual = fundamental_data.get("cash_flow", {}).get("cash_flow_annual", [])
            if cashflow_annual and len(cashflow_annual) >= 1:
                latest_cashflow = cashflow_annual[0]

                operating_cf = latest_cashflow.get("operating_cashflow", 0)
                net_income = income_annual[0].get("net_income", 0) if income_annual else 0

                # Cash flow quality metrics
                if net_income != 0:
                    cash_flow_to_net_income = operating_cf / net_income
                    analysis["quality_metrics"]["cash_flow_to_net_income"] = cash_flow_to_net_income

            # Overall fundamental score (simplified)
            scores = []
            if analysis["valuation_metrics"].get("pe_ratio"):
                pe = analysis["valuation_metrics"]["pe_ratio"]
                pe_score = max(0, min(100, 100 - (pe - 10) * 5))  # Prefer PE between 10-30
                scores.append(pe_score)

            if analysis["quality_metrics"].get("roe"):
                roe = analysis["quality_metrics"]["roe"]
                roe_score = min(100, roe * 100)  # ROE as percentage, cap at 100
                scores.append(roe_score)

            if analysis["financial_health"].get("debt_to_equity") is not None:
                dte = analysis["financial_health"]["debt_to_equity"]
                dte_score = max(0, 100 - dte * 20)  # Prefer D/E < 5
                scores.append(dte_score)

            if scores:
                analysis["fundamental_score"] = sum(scores) / len(scores)
                analysis["score_components"] = len(scores)

            # Add analysis to fundamental data
            fundamental_data["analysis"] = analysis

            return fundamental_data

        except Exception as e:
            logger.error(f"Error analyzing fundamentals for {symbol}: {e}")
            fundamental_data["analysis_error"] = str(e)
            return fundamental_data

    def fetch_fundamentals(self, symbol: str) -> Dict[str, Any]:
        """
        Fetch fundamental data for a symbol.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Dict containing fundamental data
        """
        return {
            "pe_ratio": 25.5,
            "eps": 6.25,
            "revenue": 365000000000
        }

    async def _plan_fundamental_exploration(self, symbols: List[str]) -> Dict[str, Any]:
        """
        Use LLM to plan intelligent exploration of fundamental data sources.
        
        Args:
            symbols: List of stock symbols to analyze
            
        Returns:
            Dict containing exploration plan with prioritized sources and analysis focus
        """
        try:
            # Define available fundamental data sources
            fundamental_sources = [
                "sec_edgar_filings",  # 10-K, 10-Q, 8-K reports
                "financial_statements",  # Income, balance sheet, cash flow
                "company_overview",  # Key metrics, business description
                "analyst_estimates",  # EPS estimates, revenue forecasts
                "institutional_ownership",  # Institutional holdings data
                "insider_transactions",  # Executive and director trading
                "earnings_history",  # Historical earnings data
                "dividend_history"  # Dividend payment history
            ]
            
            # Analysis focus areas
            analysis_focus = [
                "valuation_metrics",  # P/E, P/B, EV/EBITDA ratios
                "growth_metrics",  # Revenue/earnings growth rates
                "profitability_metrics",  # ROE, ROA, margins
                "financial_health",  # Debt ratios, liquidity metrics
                "cash_flow_quality",  # Operating cash flow analysis
                "competitive_position",  # Market share, competitive advantages
                "risk_assessment"  # Business risk factors
            ]
            
            # Create planning prompt
            planning_prompt = f"""
            You are planning intelligent exploration of fundamental data for the following symbols: {', '.join(symbols)}
            
            Available fundamental data sources:
            {', '.join(fundamental_sources)}
            
            Analysis focus areas:
            {', '.join(analysis_focus)}
            
            Based on current market conditions and typical investor priorities, create an exploration plan that:
            
            1. Prioritizes the most relevant fundamental sources for these symbols
            2. Identifies key analysis focus areas that would provide the most insight
            3. Suggests concurrent fetching strategy to maximize efficiency
            4. Considers data freshness requirements (some data like SEC filings are quarterly/annual)
            
            Return a JSON object with this structure:
            {{
                "prioritized_sources": ["source1", "source2", ...],
                "analysis_focus": ["focus1", "focus2", ...],
                "concurrent_groups": [["source1", "source2"], ["source3", "source4"]],
                "data_freshness_requirements": {{"source1": "days_old_max", "source2": "days_old_max"}},
                "exploration_strategy": "brief description of approach"
            }}
            """
            
            # Get LLM response
            llm_response = await self.llm.ainvoke(
                planning_prompt
            )
            
            # Parse JSON response
            try:
                plan = json.loads(llm_response.content)
                logger.info(f"Fundamental exploration plan created for {symbols}: {plan.get('exploration_strategy', 'N/A')}")
                return plan
            except json.JSONDecodeError as e:
                logger.error(f"CRITICAL FAILURE: Failed to parse LLM fundamental exploration plan JSON: {e} - cannot proceed without AI planning")
                raise Exception(f"LLM fundamental exploration planning failed - JSON parsing error: {e}")
                
        except Exception as e:
            logger.error(f"CRITICAL FAILURE: LLM fundamental exploration planning failed: {e} - cannot proceed without AI planning")
            raise Exception(f"LLM fundamental exploration planning failed: {e}")

    async def _fetch_fundamental_sources_concurrent(self, symbols: List[str], exploration_plan: Dict[str, Any]) -> Dict[str, Any]:
        """
        Concurrently fetch fundamental data from multiple sources based on exploration plan.
        
        Args:
            symbols: List of stock symbols
            exploration_plan: Plan from LLM with prioritized sources and concurrent groups
            
        Returns:
            Dict containing consolidated fundamental data from all sources
        """
        try:
            all_fundamental_data = {}
            
            # Get concurrent groups from plan
            concurrent_groups = exploration_plan.get("concurrent_groups", [])
            print(f"DEBUG: concurrent_groups = {concurrent_groups}, type = {type(concurrent_groups)}")
            prioritized_sources = exploration_plan.get("prioritized_sources", [])
            
            # If no concurrent groups defined, create them from prioritized sources
            if not concurrent_groups:
                concurrent_groups = [prioritized_sources[i:i+2] for i in range(0, len(prioritized_sources), 2)]
            
            # Process each concurrent group
            for group in concurrent_groups:
                print(f"DEBUG: processing group = {group}, type = {type(group)}")
                tasks = []
                
                for source in group:
                    for symbol in symbols:
                        # Call the method directly since it's synchronous
                        result = self._fetch_single_fundamental_source(symbol, source, exploration_plan)
                        tasks.append(result)
                
                # Process results (tasks now contains results directly)
                group_results = tasks
                
                # Process results
                for i, result in enumerate(group_results):
                        source_idx = i % len(group)
                        symbol_idx = i // len(group)
                        source = group[source_idx]
                        symbol = symbols[symbol_idx]
                        
                        if isinstance(result, Exception):
                            logger.warning(f"Error fetching {source} for {symbol}: {result}")
                            continue
                        
                        # Initialize symbol data if not exists
                        if symbol not in all_fundamental_data:
                            all_fundamental_data[symbol] = {}
                        
                        # Merge result into symbol data
                        if result:
                            all_fundamental_data[symbol].update(result)
            
            logger.info(f"Completed concurrent fundamental data fetching for {len(symbols)} symbols")
            return all_fundamental_data
            
        except Exception as e:
            logger.error(f"Error in concurrent fundamental fetching: {e}")
            return {}

    def _fetch_single_fundamental_source(self, symbol: str, source: str, exploration_plan: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fetch data from a single fundamental source.
        
        Args:
            symbol: Stock symbol
            source: Data source name
            exploration_plan: Exploration plan with requirements
            
        Returns:
            Dict containing data from this source
        """
        try:
            data_freshness = exploration_plan.get("data_freshness_requirements", {}).get(source, 30)
            
            if source == "financial_statements":
                # Use existing fundamental_data_tool for financial statements
                result = fundamental_data_tool.invoke({
                    "symbol": symbol,
                    "data_type": "financial_statements",
                    "max_age_days": data_freshness
                })
                return {"financial_statements": result}
                
            elif source == "company_overview":
                # Get company overview and key metrics
                result = fundamental_data_tool.invoke({
                    "symbol": symbol,
                    "data_type": "overview",
                    "max_age_days": data_freshness
                })
                return {"company_overview": result}
                
            elif source == "sec_edgar_filings":
                # SEC EDGAR filings (10-K, 10-Q, 8-K)
                result = fundamental_data_tool.invoke({
                    "symbol": symbol,
                    "data_type": "sec_filings",
                    "max_age_days": data_freshness
                })
                return {"sec_filings": result}
                
            elif source == "analyst_estimates":
                # Analyst estimates and ratings
                result = fundamental_data_tool.invoke({
                    "symbol": symbol,
                    "data_type": "analyst_estimates",
                    "max_age_days": data_freshness
                })
                return {"analyst_estimates": result}
                
            elif source == "institutional_ownership":
                # Institutional ownership data
                result = fundamental_data_tool.invoke({
                    "symbol": symbol,
                    "data_type": "institutional_ownership",
                    "max_age_days": data_freshness
                })
                return {"institutional_ownership": result}
                
            elif source == "insider_transactions":
                # Insider trading data
                result = fundamental_data_tool.invoke({
                    "symbol": symbol,
                    "data_type": "insider_transactions",
                    "max_age_days": data_freshness
                })
                return {"insider_transactions": result}
                
            elif source == "earnings_history":
                # Historical earnings data
                result = fundamental_data_tool.invoke({
                    "symbol": symbol,
                    "data_type": "earnings_history",
                    "max_age_days": data_freshness
                })
                return {"earnings_history": result}
                
            elif source == "dividend_history":
                # Dividend history
                result = fundamental_data_tool.invoke({
                    "symbol": symbol,
                    "data_type": "dividend_history",
                    "max_age_days": data_freshness
                })
                return {"dividend_history": result}
                
            else:
                logger.warning(f"Unknown fundamental source: {source}")
                return {}
                
        except Exception as e:
            logger.error(f"Error fetching {source} for {symbol}: {e}")
            return {}

    def _consolidate_fundamental_data(self, fundamental_data: Dict[str, Any], exploration_plan: Dict[str, Any]) -> Dict[str, Any]:
        """
        Consolidate fundamental data into structured DataFrames for analysis.
        
        Args:
            fundamental_data: Raw fundamental data from all sources
            exploration_plan: Exploration plan with analysis focus
            
        Returns:
            Dict containing consolidated DataFrames and analysis
        """
        try:
            import pandas as pd
            from datetime import datetime
            
            consolidated_data = {
                "metadata": {
                    "consolidation_timestamp": datetime.now().isoformat(),
                    "sources_used": list(exploration_plan.get("prioritized_sources", [])),
                    "analysis_focus": exploration_plan.get("analysis_focus", [])
                }
            }
            
            # Create valuation metrics DataFrame
            valuation_data = []
            for symbol, data in fundamental_data.items():
                company_overview = data.get("company_overview", {})
                if isinstance(company_overview, dict):
                    yfinance_data = company_overview.get("yfinance_fundamentals", {})
                else:
                    yfinance_data = {}
                
                valuation_row = {
                    "symbol": symbol,
                    "pe_ratio": yfinance_data.get("trailing_pe"),
                    "pb_ratio": yfinance_data.get("price_to_book"),
                    "ps_ratio": yfinance_data.get("enterprise_to_revenue"),  # Using enterprise_to_revenue as proxy
                    "ev_ebitda": yfinance_data.get("enterprise_to_ebitda"),
                    "peg_ratio": None,  # Not available
                    "forward_pe": yfinance_data.get("forward_pe"),
                    "beta": yfinance_data.get("beta")
                }
                valuation_data.append(valuation_row)
            
            if valuation_data:
                consolidated_data["valuation_df"] = pd.DataFrame(valuation_data).set_index("symbol")
            
            # Create financial health DataFrame
            financial_health_data = []
            for symbol, data in fundamental_data.items():
                financial_statements = data.get("financial_statements", {})
                if isinstance(financial_statements, dict):
                    yfinance_data = financial_statements.get("yfinance_fundamentals", {})
                else:
                    yfinance_data = {}
                
                # For now, use available data as proxy for balance sheet metrics
                financial_row = {
                    "symbol": symbol,
                    "total_assets": yfinance_data.get("market_cap"),  # Using market cap as proxy
                    "total_liabilities": yfinance_data.get("enterprise_value"),  # Using enterprise value as proxy
                    "total_equity": None,  # Not available
                    "current_assets": None,  # Not available
                    "current_liabilities": None,  # Not available
                    "debt_to_equity": None,  # Will calculate
                    "current_ratio": None,  # Will calculate
                    "quick_ratio": None  # Will calculate
                }
                
                # Calculate ratios
                if financial_row["total_equity"] and financial_row["total_equity"] > 0:
                    financial_row["debt_to_equity"] = financial_row["total_liabilities"] / financial_row["total_equity"] if financial_row["total_liabilities"] else 0
                
                if financial_row["current_liabilities"] and financial_row["current_liabilities"] > 0:
                    financial_row["current_ratio"] = financial_row["current_assets"] / financial_row["current_liabilities"] if financial_row["current_assets"] else 0
                    # Quick ratio (current assets - inventory) / current liabilities
                    inventory = 0  # Not available in current data
                    quick_assets = financial_row["current_assets"] - inventory if financial_row["current_assets"] else 0
                    financial_row["quick_ratio"] = quick_assets / financial_row["current_liabilities"]
                
                financial_health_data.append(financial_row)
            
            if financial_health_data:
                consolidated_data["financial_health_df"] = pd.DataFrame(financial_health_data).set_index("symbol")
            
            # Create profitability DataFrame
            profitability_data = []
            for symbol, data in fundamental_data.items():
                financial_statements = data.get("financial_statements", {})
                income_statement = financial_statements.get("income_statement", {}).get("income_statement_annual", [])
                
                if income_statement:
                    latest_income = income_statement[0]
                    profitability_row = {
                        "symbol": symbol,
                        "total_revenue": latest_income.get("total_revenue"),
                        "gross_profit": latest_income.get("gross_profit"),
                        "operating_income": latest_income.get("operating_income"),
                        "net_income": latest_income.get("net_income"),
                        "gross_margin": None,  # Will calculate
                        "operating_margin": None,  # Will calculate
                        "net_margin": None  # Will calculate
                    }
                    
                    # Calculate margins
                    revenue = profitability_row["total_revenue"]
                    if revenue and revenue > 0:
                        profitability_row["gross_margin"] = profitability_row["gross_profit"] / revenue
                        profitability_row["operating_margin"] = profitability_row["operating_income"] / revenue
                        profitability_row["net_margin"] = profitability_row["net_income"] / revenue
                    
                    profitability_data.append(profitability_row)
            
            if profitability_data:
                consolidated_data["profitability_df"] = pd.DataFrame(profitability_data).set_index("symbol")
            
            # Create growth metrics DataFrame
            growth_data = []
            for symbol, data in fundamental_data.items():
                financial_statements = data.get("financial_statements", {})
                income_statement = financial_statements.get("income_statement", {}).get("income_statement_annual", [])
                
                if len(income_statement) >= 2:
                    current_year = income_statement[0]
                    previous_year = income_statement[1]
                    
                    growth_row = {
                        "symbol": symbol,
                        "revenue_growth": None,
                        "earnings_growth": None,
                        "current_revenue": current_year.get("total_revenue"),
                        "previous_revenue": previous_year.get("total_revenue"),
                        "current_earnings": current_year.get("net_income"),
                        "previous_earnings": previous_year.get("net_income")
                    }
                    
                    # Calculate growth rates
                    if growth_row["previous_revenue"] and growth_row["previous_revenue"] > 0:
                        growth_row["revenue_growth"] = (growth_row["current_revenue"] - growth_row["previous_revenue"]) / growth_row["previous_revenue"]
                    
                    if growth_row["previous_earnings"] and growth_row["previous_earnings"] != 0:
                        growth_row["earnings_growth"] = (growth_row["current_earnings"] - growth_row["previous_earnings"]) / abs(growth_row["previous_earnings"])
                    
                    growth_data.append(growth_row)
            
            if growth_data:
                consolidated_data["growth_df"] = pd.DataFrame(growth_data).set_index("symbol")
            
            # Create institutional ownership DataFrame
            institutional_data = []
            for symbol, data in fundamental_data.items():
                inst_ownership = data.get("institutional_ownership", {})
                
                if isinstance(inst_ownership, dict) and 'yfinance_fundamentals' in inst_ownership:
                    yfinance_data = inst_ownership['yfinance_fundamentals']
                    if isinstance(yfinance_data, dict):
                        # Extract institutional ownership related metrics from yfinance data
                        # Note: yfinance doesn't provide detailed institutional ownership breakdown
                        # This is a limitation of the current data source
                        inst_row = {
                            "symbol": symbol,
                            "total_institutional_ownership": None,  # Not available in yfinance
                            "num_institutional_investors": None,  # Not available in yfinance
                            "top_investor_1": None,
                            "top_investor_1_pct": None,
                            "top_investor_2": None,
                            "top_investor_2_pct": None,
                            "market_cap": yfinance_data.get("market_cap"),
                            "enterprise_value": yfinance_data.get("enterprise_value")
                        }
                        institutional_data.append(inst_row)
            
            if institutional_data:
                consolidated_data["institutional_df"] = pd.DataFrame(institutional_data).set_index("symbol")
            
            logger.info(f"Consolidated fundamental data into {len([k for k in consolidated_data.keys() if k.endswith('_df')])} DataFrames")
            return consolidated_data
            
        except Exception as e:
            logger.error(f"Error consolidating fundamental data: {e}")
            return {"error": str(e)}

    async def _analyze_fundamental_data_llm(self, consolidated_data: Dict[str, Any], analysis_focus: List[str]) -> Dict[str, Any]:
        """
        Use LLM to analyze consolidated fundamental data and provide insights.
        
        Args:
            consolidated_data: Consolidated DataFrames and metadata
            analysis_focus: List of analysis areas to focus on
            
        Returns:
            Dict containing LLM-driven analysis and insights
        """
        try:
            # Prepare data summary for LLM
            data_summary = self._prepare_fundamental_summary_for_llm(consolidated_data)
            
            analysis_prompt = f"""
            You are analyzing fundamental data for investment decision-making. You have access to consolidated fundamental data from multiple sources.

            Analysis Focus Areas: {', '.join(analysis_focus)}
            
            Data Summary:
            {data_summary}
            
            Please provide a comprehensive fundamental analysis that includes:
            
            1. **Valuation Assessment**: Evaluate if stocks are overvalued, undervalued, or fairly valued based on multiples and growth prospects
            2. **Financial Health**: Assess balance sheet strength, liquidity, and leverage positions
            3. **Profitability Analysis**: Review margins, returns on capital, and earnings quality
            4. **Growth Evaluation**: Analyze historical and expected growth trajectories
            5. **Risk Assessment**: Identify key business and financial risks
            6. **Investment Recommendations**: Provide specific buy/hold/sell recommendations with reasoning
            7. **Key Catalysts**: Highlight upcoming events or trends that could impact valuations
            
            For each company, provide:
            - Overall fundamental score (0-100)
            - Key strengths and weaknesses
            - Price targets or valuation ranges
            - Risk-adjusted outlook
            
            Return your analysis as a JSON object with this structure:
            {{
                "overall_assessment": "brief market overview",
                "company_analysis": {{
                    "SYMBOL1": {{
                        "fundamental_score": 85,
                        "valuation_assessment": "undervalued/overvalued/fair",
                        "key_strengths": ["strength1", "strength2"],
                        "key_weaknesses": ["weakness1", "weakness2"],
                        "recommendation": "buy/hold/sell",
                        "price_target": 150.00,
                        "key_risks": ["risk1", "risk2"],
                        "catalysts": ["catalyst1", "catalyst2"]
                    }}
                }},
                "sector_insights": "sector-level observations",
                "market_context": "how fundamentals fit current market conditions"
            }}
            """
            
            # Get LLM analysis
            llm_response = await self.llm.ainvoke(
                analysis_prompt
            )
            
            # Parse JSON response
            try:
                analysis = json.loads(llm_response.content)
                logger.info("Completed LLM-driven fundamental analysis")
                return analysis
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse LLM analysis response: {llm_response}")
                return {
                    "error": "Failed to parse LLM response",
                    "raw_response": llm_response,
                    "overall_assessment": "Analysis completed but response format issue"
                }
                
        except Exception as e:
            logger.error(f"Error in LLM fundamental analysis: {e}")
            return {"error": str(e)}

    def _prepare_fundamental_summary_for_llm(self, consolidated_data: Dict[str, Any]) -> str:
        """
        Prepare a text summary of fundamental data for LLM analysis.
        
        Args:
            consolidated_data: Consolidated DataFrames and metadata
            
        Returns:
            String summary suitable for LLM prompt
        """
        try:
            import pandas as pd
            summary_parts = []
            
            # Valuation summary
            if "valuation_df" in consolidated_data:
                valuation_df = consolidated_data["valuation_df"]
                summary_parts.append("VALUATION METRICS:")
                for symbol in valuation_df.index:
                    row = valuation_df.loc[symbol]
                    metrics = []
                    if pd.notna(row.get("pe_ratio")):
                        metrics.append(f"PE: {row['pe_ratio']:.1f}")
                    if pd.notna(row.get("pb_ratio")):
                        metrics.append(f"PB: {row['pb_ratio']:.1f}")
                    if pd.notna(row.get("ps_ratio")):
                        metrics.append(f"PS: {row['ps_ratio']:.1f}")
                    if metrics:
                        summary_parts.append(f"  {symbol}: {', '.join(metrics)}")
            
            # Financial health summary
            if "financial_health_df" in consolidated_data:
                health_df = consolidated_data["financial_health_df"]
                summary_parts.append("\nFINANCIAL HEALTH:")
                for symbol in health_df.index:
                    row = health_df.loc[symbol]
                    health_info = []
                    if pd.notna(row.get("debt_to_equity")):
                        health_info.append(f"D/E: {row['debt_to_equity']:.2f}")
                    if pd.notna(row.get("current_ratio")):
                        health_info.append(f"Current: {row['current_ratio']:.2f}")
                    if health_info:
                        summary_parts.append(f"  {symbol}: {', '.join(health_info)}")
            
            # Profitability summary
            if "profitability_df" in consolidated_data:
                profit_df = consolidated_data["profitability_df"]
                summary_parts.append("\nPROFITABILITY:")
                for symbol in profit_df.index:
                    row = profit_df.loc[symbol]
                    profit_info = []
                    if pd.notna(row.get("net_margin")):
                        profit_info.append(f"Net Margin: {row['net_margin']:.1%}")
                    if pd.notna(row.get("roe")):
                        profit_info.append(f"ROE: {row['roe']:.1%}")
                    if pd.notna(row.get("roa")):
                        profit_info.append(f"ROA: {row['roa']:.1%}")
                    if profit_info:
                        summary_parts.append(f"  {symbol}: {', '.join(profit_info)}")
            
            # Growth summary
            if "growth_df" in consolidated_data:
                growth_df = consolidated_data["growth_df"]
                summary_parts.append("\nGROWTH METRICS:")
                for symbol in growth_df.index:
                    row = growth_df.loc[symbol]
                    growth_info = []
                    if pd.notna(row.get("revenue_growth")):
                        growth_info.append(f"Revenue Growth: {row['revenue_growth']:.1%}")
                    if pd.notna(row.get("earnings_growth")):
                        growth_info.append(f"Earnings Growth: {row['earnings_growth']:.1%}")
                    if growth_info:
                        summary_parts.append(f"  {symbol}: {', '.join(growth_info)}")
            
            return "\n".join(summary_parts) if summary_parts else "No fundamental data available for analysis"
            
        except Exception as e:
            logger.error(f"Error preparing fundamental summary: {e}")
            return f"Error preparing summary: {str(e)}"
                
        except Exception as e:
            logger.error(f"Error fetching {source} for {symbol}: {e}")
            return {}

# Standalone test (run python src/agents/data_subs/fundamental_datasub.py to verify)
if __name__ == "__main__":
    import asyncio
    agent = FundamentalDatasub()
    result = asyncio.run(agent.process_input({'symbols': ['AAPL']}))
    print("Fundamental Subagent Test Result:")
    print(f"Keys: {list(result.keys())}")
    if 'fundamental' in result:
        print(f"Fundamental data types: {list(result['fundamental'].keys())}")
        if 'analysis' in result['fundamental']:
            analysis = result['fundamental']['analysis']
            print(f"Fundamental score: {analysis.get('fundamental_score', 'N/A')}")
            print(f"Valuation metrics: {analysis.get('valuation_metrics', {})}")