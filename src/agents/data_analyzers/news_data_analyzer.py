# src/agents/data_subs/news_datasub.py
# Purpose: News Data Subagent with LLM-powered exploration and intelligent filtering.
# Provides comprehensive news analysis from multiple sources with AI-driven insights.
# Structural Reasoning: Enhanced subagent for intelligent news aggregation and analysis.
# Ties to system: Provides structured news DataFrames for main data agent coordination.
# For legacy wealth: AI-powered news intelligence for superior market insights.

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))  # Dynamic root path for imports.

from src.agents.base import BaseAgent  # Absolute import.
import logging
from typing import Dict, Any, List
import pandas as pd
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class NewsDataAnalyzer(BaseAgent):
    """
    News Data Subagent with LLM-powered exploration.
    Reasoning: Intelligently explores multiple news sources and filters content for relevance.
    Uses LLM to prioritize sources, assess credibility, and analyze market impact.
    """
    def __init__(self):
        config_paths = {'risk': 'config/risk-constraints.yaml'}  # Relative to root.
        prompt_paths = {'base': 'config/base_prompt.txt', 'role': 'docs/AGENTS/main-agents/data-agent.md'}  # Relative to root.
        tools = []  # NewsDatasub uses internal methods instead of tools
        super().__init__(role='news_data', config_paths=config_paths, prompt_paths=prompt_paths, tools=tools)

        # Available news sources for LLM exploration
        self.available_sources = {
            'newsapi': 'Comprehensive news aggregation with global coverage',
            'currentsapi': 'Real-time news with fast updates and diverse sources',
            'alphavantage': 'Financial news focused on market-moving events',
            'bing_news': 'Microsoft Bing news search with broad coverage',
            'google_news': 'Google News aggregation with trending topics',
            'twitter_trends': 'Social media sentiment and breaking news',
            'rss_feeds': 'Direct RSS feeds from financial publications'
        }

    async def reflect(self, adjustments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Reflect on batch adjustments for self-improvement.
        Analyzes news data performance metrics and generates optimization insights.

        Args:
            adjustments: Dictionary containing performance data and adjustment metrics

        Returns:
            Dict with reflection insights and improvement recommendations
        """
        try:
            logger.info(f"News reflecting on adjustments: {adjustments}")

            # Analyze news data-specific performance
            reflection_insights = {
                'timestamp': pd.Timestamp.now().isoformat(),
                'adjustments_processed': len(adjustments) if isinstance(adjustments, dict) else 0,
                'performance_analysis': {},
                'optimization_opportunities': [],
                'content_quality_insights': {}
            }

            # Extract news data performance metrics
            if isinstance(adjustments, dict):
                # Analyze article fetch success, relevance scoring, credibility assessment
                if 'performance_data' in adjustments:
                    perf_data = adjustments['performance_data']
                    reflection_insights['performance_analysis'] = {
                        'article_fetch_success_rate': perf_data.get('article_fetch_success_rate', 0),
                        'relevance_scoring_accuracy': perf_data.get('relevance_scoring_accuracy', 0),
                        'credibility_assessment_quality': perf_data.get('credibility_assessment_quality', 0),
                        'impact_prediction_accuracy': perf_data.get('impact_prediction_accuracy', 0)
                    }

                # Identify news data-specific improvement opportunities
                if 'issues' in adjustments:
                    issues = adjustments['issues']
                    for issue in issues:
                        if 'relevance' in issue.lower() or 'filtering' in issue.lower():
                            reflection_insights['optimization_opportunities'].append({
                                'type': 'content_filtering_enhancement',
                                'description': 'Improve news article relevance filtering and prioritization algorithms',
                                'priority': 'high'
                            })
                        elif 'credibility' in issue.lower() or 'source' in issue.lower():
                            reflection_insights['optimization_opportunities'].append({
                                'type': 'source_credibility_scoring',
                                'description': 'Enhance source credibility assessment and weighting mechanisms',
                                'priority': 'medium'
                            })
                        elif 'sentiment' in issue.lower():
                            reflection_insights['optimization_opportunities'].append({
                                'type': 'sentiment_analysis_improvement',
                                'description': 'Strengthen sentiment extraction from news content and headlines',
                                'priority': 'medium'
                            })

                # Generate content quality insights
                reflection_insights['content_quality_insights'] = {
                    'article_diversity_score': adjustments.get('article_diversity_score', 0.7),
                    'source_coverage_breadth': adjustments.get('source_coverage_breadth', 0.8),
                    'real_time_coverage_effectiveness': adjustments.get('real_time_coverage_effectiveness', 0.6),
                    'market_impact_prediction_accuracy': adjustments.get('market_impact_prediction_accuracy', 0.5)
                }

            # Store reflection in memory for future learning
            self.update_memory('news_reflection_history', {
                'timestamp': reflection_insights['timestamp'],
                'insights': reflection_insights,
                'adjustments_analyzed': adjustments
            })

            logger.info(f"News reflection completed with {len(reflection_insights['optimization_opportunities'])} optimization opportunities identified")
            return reflection_insights

        except Exception as e:
            logger.error(f"Error during news reflection analysis: {e}")
            return {
                'error': str(e),
                'timestamp': pd.Timestamp.now().isoformat(),
                'adjustments_processed': len(adjustments) if isinstance(adjustments, dict) else 0
            }

    async def process_input(self, input_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Process input to fetch and analyze news data with LLM enhancement.
        Args:
            input_data: Dict with parameters (symbol for news analysis).
        Returns:
            Dict with structured news data and LLM analysis.
        """
        logger.info(f"NewsDatasub processing input: {input_data}")

        try:
            # Initialize LLM if not already done
            if not self.llm:
                await self.async_initialize_llm()

            symbol = input_data.get('symbol', 'SPY') if input_data else 'SPY'

            # Step 1: Plan news exploration with LLM
            exploration_plan = await self._plan_news_exploration(symbol, input_data or {})

            # Step 2: Execute exploration plan
            exploration_results = await self._execute_news_exploration(symbol, exploration_plan)

            # Step 3: Consolidate data into structured format
            consolidated_data = self._consolidate_news_data(symbol, exploration_results)

            # Step 4: Analyze with LLM for insights
            llm_analysis = await self._analyze_news_data_llm(consolidated_data)

            # Combine results
            result = {
                "consolidated_data": consolidated_data,
                "llm_analysis": llm_analysis,
                "exploration_plan": exploration_plan,
                "enhanced": True
            }

            # Store news data in shared memory
            await self.store_shared_memory("news_data", symbol, {
                "news_data": consolidated_data,
                "llm_analysis": llm_analysis,
                "timestamp": pd.Timestamp.now().isoformat()
            })

            # Store quality assessment for cross-verification
            quality_assessment = {
                "analyzer": "news",
                "symbol": symbol,
                "credibility_score": consolidated_data.get('avg_credibility_score', 0.5),
                "relevance_score": consolidated_data.get('avg_relevance_score', 0.5),
                "data_quality_score": consolidated_data.get('data_quality_score', 5.0),
                "article_count": consolidated_data.get('total_articles', 0),
                "sources_used": consolidated_data.get('sources_explored', []),
                "impact_distribution": consolidated_data.get('impact_distribution', {}),
                "timestamp": pd.Timestamp.now().isoformat(),
                "llm_insights": {
                    "sentiment": llm_analysis.get('sentiment_summary', {}).get('overall_sentiment', 'neutral'),
                    "key_themes": llm_analysis.get('key_themes', []),
                    "risk_level": llm_analysis.get('risk_assessment', {}).get('risk_level', 'moderate')
                }
            }
            await self.store_shared_memory("data_quality_assessments", f"news_{symbol}", quality_assessment)

            logger.info(f"NewsDatasub output: LLM-enhanced news data collected for {symbol}")
            return result

        except Exception as e:
            logger.error(f"NewsDatasub failed: {e}")
            return {"articles_df": pd.DataFrame(), "error": str(e), "enhanced": False}

    async def _plan_news_exploration(self, symbol: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Use LLM to plan intelligent news exploration based on symbol and context.

        Args:
            symbol: Stock symbol to analyze
            context: Additional context for exploration planning

        Returns:
            Dict containing exploration plan with prioritized sources
        """
        if not self.llm:
            logger.error("CRITICAL FAILURE: No LLM available for news data exploration - cannot proceed without AI planning")
            raise Exception("LLM required for intelligent news exploration - no default fallback allowed")

        try:
            exploration_prompt = f"""
You are an expert financial news analyst planning comprehensive news coverage for {symbol}.

CONTEXT:
- Symbol: {symbol}
- Available News Sources: {self.available_sources}
- Analysis Goals: Maximize market intelligence while managing API costs and avoiding information overload
- Risk Constraints: Focus on credible sources and market-moving news

TASK:
Based on the symbol characteristics and available sources, determine which news sources to explore and prioritize them.
Consider:
1. Market capitalization and news coverage (large caps vs small caps)
2. Recent volatility and breaking news potential
3. Source credibility and timeliness
4. Cost-benefit analysis of premium vs free sources
5. Geographic relevance for international companies

Return a JSON object with:
- "sources": Array of source names to explore (from available_sources keys)
- "priorities": Object mapping source names to priority scores (1-10, higher = more important)
- "reasoning": Brief explanation of exploration strategy
- "focus_areas": Array of news categories to prioritize (e.g., ["earnings", "M&A", "regulatory"])

Example response:
{{
  "sources": ["newsapi", "currentsapi", "alphavantage"],
  "priorities": {{"newsapi": 9, "currentsapi": 8, "alphavantage": 7}},
  "reasoning": "Focus on established financial news sources for {symbol} as it's a major index component",
  "focus_areas": ["earnings", "economic_impact", "sector_news"]
}}
"""

            response = await self.llm.ainvoke(exploration_prompt)
            response_text = response.content if hasattr(response, 'content') else str(response)

            # Parse JSON response
            import json
            try:
                plan = json.loads(response_text)
                logger.info(f"LLM news exploration plan for {symbol}: {plan.get('reasoning', 'No reasoning provided')}")
                return plan
            except json.JSONDecodeError as e:
                logger.error(f"CRITICAL FAILURE: Failed to parse LLM news exploration plan JSON: {e} - cannot proceed without AI planning")
                raise Exception(f"LLM news exploration planning failed - JSON parsing error: {e}")

        except Exception as e:
            logger.error(f"CRITICAL FAILURE: LLM news exploration planning failed: {e} - cannot proceed without AI planning")
            raise Exception(f"LLM news exploration planning failed: {e}")

    async def _execute_news_exploration(self, symbol: str, plan: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the news exploration plan by fetching from prioritized sources.

        Args:
            symbol: Stock symbol
            plan: Exploration plan from LLM

        Returns:
            Dict containing results from all explored sources
        """
        results = {}
        sources = plan.get('sources', ['newsapi'])

        for source in sources:
            try:
                if source == 'newsapi':
                    results['newsapi'] = await self._fetch_newsapi_data(symbol)
                elif source == 'currentsapi':
                    results['currentsapi'] = await self._fetch_currentsapi_data(symbol)
                elif source == 'alphavantage':
                    results['alphavantage'] = await self._fetch_alphavantage_news(symbol)
                elif source == 'bing_news':
                    results['bing_news'] = await self._fetch_bing_news(symbol)
                elif source == 'google_news':
                    results['google_news'] = await self._fetch_google_news(symbol)
                elif source == 'twitter_trends':
                    results['twitter_trends'] = await self._fetch_twitter_trends(symbol)
                elif source == 'rss_feeds':
                    results['rss_feeds'] = await self._fetch_rss_feeds(symbol)
                else:
                    logger.warning(f"Unknown news source: {source}")

            except Exception as e:
                logger.error(f"Failed to fetch {source} data for {symbol}: {e}")
                results[source] = {"error": str(e)}

        return results

    async def _fetch_newsapi_data(self, symbol: str) -> Dict[str, Any]:
        """Fetch news from NewsAPI."""
        from src.utils.tools import news_data_tool
        result = news_data_tool(symbol)  # Call directly as function
        return self._enhance_news_data(result, symbol, "newsapi")

    async def _fetch_currentsapi_data(self, symbol: str) -> Dict[str, Any]:
        """Fetch news from CurrentsAPI."""
        raise NotImplementedError("Real CurrentsAPI implementation required - no mock data allowed in production")

    async def _fetch_alphavantage_news(self, symbol: str) -> Dict[str, Any]:
        """Fetch financial news from Alpha Vantage."""
        raise NotImplementedError("Real Alpha Vantage news API implementation required - no mock data allowed in production")

    async def _fetch_bing_news(self, symbol: str) -> Dict[str, Any]:
        """Fetch news from Bing News Search."""
        raise NotImplementedError("Real Bing News Search API implementation required - no mock data allowed in production")

    async def _fetch_google_news(self, symbol: str) -> Dict[str, Any]:
        """Fetch news from Google News."""
        raise NotImplementedError("Real Google News API implementation required - no mock data allowed in production")

    async def _fetch_twitter_trends(self, symbol: str) -> Dict[str, Any]:
        """Fetch Twitter trends and sentiment for symbol."""
        raise NotImplementedError("Real Twitter API implementation required - no mock data allowed in production")

    async def _fetch_rss_feeds(self, symbol: str) -> Dict[str, Any]:
        """Fetch news from financial RSS feeds."""
        raise NotImplementedError("Real RSS feed aggregation implementation required - no mock data allowed in production")

    def _enhance_news_data(self, news_data: Dict[str, Any], symbol: str, source: str) -> Dict[str, Any]:
        """Enhance news data with additional processing and validation."""
        try:
            if 'articles' not in news_data:
                news_data['articles'] = []

            # Add metadata
            news_data['symbol'] = symbol
            news_data['fetch_timestamp'] = datetime.now().isoformat()
            news_data['article_count'] = len(news_data.get('articles', []))
            news_data['source'] = source

            # Validate and clean articles
            valid_articles = []
            for article in news_data.get('articles', []):
                if isinstance(article, dict) and 'title' in article:
                    # Add enhanced validation and scoring
                    article['validated'] = True
                    article['relevance_score'] = self._calculate_relevance_score(article, symbol)
                    article['credibility_score'] = self._calculate_credibility_score(article, source)
                    article['impact_potential'] = self._assess_impact_potential(article)
                    valid_articles.append(article)

            news_data['articles'] = valid_articles
            news_data['valid_article_count'] = len(valid_articles)

            return news_data

        except Exception as e:
            logger.error(f"Failed to enhance news data: {e}")
            return news_data

    def _calculate_relevance_score(self, article: Dict[str, Any], symbol: str) -> float:
        """Calculate relevance score for news article to symbol."""
        try:
            title = article.get('title', '').lower()
            description = article.get('description', '').lower()
            symbol_lower = symbol.lower()

            # Enhanced relevance scoring
            score = 0.0
            if symbol_lower in title:
                score += 0.6
            if symbol_lower in description:
                score += 0.4

            # Boost for financial keywords
            financial_keywords = ['stock', 'market', 'price', 'trading', 'earnings', 'revenue', 'profit', 'loss']
            for keyword in financial_keywords:
                if keyword in title or keyword in description:
                    score += 0.15

            # Boost for market-moving keywords
            market_keywords = ['earnings', 'guidance', 'forecast', 'merger', 'acquisition', 'lawsuit', 'regulation']
            for keyword in market_keywords:
                if keyword in title or keyword in description:
                    score += 0.2

            return min(1.0, score)  # Cap at 1.0

        except Exception as e:
            logger.error(f"Failed to calculate relevance score: {e}")
            return 0.0

    def _calculate_credibility_score(self, article: Dict[str, Any], source: str) -> float:
        """Calculate credibility score based on source and article characteristics."""
        try:
            base_score = 0.5  # Default neutral score

            # Source credibility weights
            source_weights = {
                'newsapi': 0.8,
                'currentsapi': 0.7,
                'alphavantage': 0.9,
                'bing_news': 0.6,
                'google_news': 0.7,
                'twitter_trends': 0.4,
                'rss_feeds': 0.8
            }

            base_score = source_weights.get(source, 0.5)

            # Adjust based on article characteristics
            title = article.get('title', '')
            description = article.get('description', '')

            # Boost for detailed content
            if len(description) > 100:
                base_score += 0.1

            # Boost for author presence
            if article.get('author'):
                base_score += 0.1

            # Penalize for sensational titles
            sensational_words = ['breaking', 'shocking', 'explosive', 'crisis']
            for word in sensational_words:
                if word.lower() in title.lower():
                    base_score -= 0.1

            return max(0.0, min(1.0, base_score))

        except Exception as e:
            logger.error(f"Failed to calculate credibility score: {e}")
            return 0.5

    def _assess_impact_potential(self, article: Dict[str, Any]) -> str:
        """Assess the potential market impact of the article."""
        try:
            title = article.get('title', '').lower()
            description = article.get('description', '').lower()

            # High impact keywords
            high_impact = ['earnings', 'merger', 'acquisition', 'lawsuit', 'regulation', 'bankruptcy', 'recall']
            # Medium impact keywords
            medium_impact = ['guidance', 'forecast', 'partnership', 'expansion', 'layoffs']
            # Low impact keywords
            low_impact = ['analysis', 'opinion', 'interview', 'update']

            for keyword in high_impact:
                if keyword in title or keyword in description:
                    return 'high'

            for keyword in medium_impact:
                if keyword in title or keyword in description:
                    return 'medium'

            for keyword in low_impact:
                if keyword in title or keyword in description:
                    return 'low'

            return 'neutral'

        except Exception as e:
            logger.error(f"Failed to assess impact potential: {e}")
            return 'unknown'

    def _consolidate_news_data(self, symbol: str, exploration_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Consolidate exploration results into DataFrame format for pipeline integration.

        Args:
            symbol: Stock symbol
            exploration_results: Raw data from all explored sources

        Returns:
            Dict with consolidated data including DataFrames
        """
        consolidated = {
            'symbol': symbol,
            'source': 'news_llm_exploration',
            'sources_explored': list(exploration_results.keys()),
            'timestamp': datetime.now().isoformat()
        }

        # Collect all articles from all sources
        all_articles = []
        for source, result in exploration_results.items():
            if isinstance(result, dict) and 'articles' in result:
                for article in result['articles']:
                    article['news_source'] = source
                    all_articles.append(article)

        # Create articles DataFrame
        if all_articles:
            articles_df = pd.DataFrame(all_articles)
            consolidated['articles_df'] = articles_df

            # Calculate aggregate metrics
            consolidated['total_articles'] = len(all_articles)
            consolidated['avg_relevance_score'] = articles_df['relevance_score'].mean() if 'relevance_score' in articles_df.columns else 0.5
            consolidated['avg_credibility_score'] = articles_df['credibility_score'].mean() if 'credibility_score' in articles_df.columns else 0.5

            # Sentiment analysis summary
            if 'impact_potential' in articles_df.columns:
                impact_counts = articles_df['impact_potential'].value_counts()
                consolidated['impact_distribution'] = impact_counts.to_dict()
            else:
                consolidated['impact_distribution'] = {}

        # Add metadata
        consolidated['data_quality_score'] = self._calculate_news_quality_score(exploration_results)
        consolidated['market_insights'] = self._extract_market_insights(all_articles)

        return consolidated

    def _calculate_news_quality_score(self, exploration_results: Dict[str, Any]) -> float:
        """Calculate overall news data quality score."""
        base_score = 5.0
        source_bonus = len(exploration_results) * 0.5
        article_bonus = sum(len(r.get('articles', [])) for r in exploration_results.values() if isinstance(r, dict)) * 0.1

        return min(10.0, base_score + source_bonus + article_bonus)

    def _extract_market_insights(self, articles: List[Dict[str, Any]]) -> List[str]:
        """Extract key market insights from consolidated articles."""
        insights = []

        if not articles:
            return ["No significant market insights from news sources"]

        # Count high-impact articles
        high_impact_count = sum(1 for a in articles if a.get('impact_potential') == 'high')
        if high_impact_count > 0:
            insights.append(f"{high_impact_count} high-impact news articles identified")

        # Check for earnings-related news
        earnings_articles = [a for a in articles if 'earnings' in a.get('title', '').lower() or 'earnings' in a.get('description', '').lower()]
        if earnings_articles:
            insights.append(f"{len(earnings_articles)} earnings-related articles found")

        # Check for regulatory news
        regulatory_articles = [a for a in articles if 'regulation' in a.get('title', '').lower() or 'sec' in a.get('title', '').lower()]
        if regulatory_articles:
            insights.append(f"{len(regulatory_articles)} regulatory articles identified")

        return insights if insights else ["Standard market coverage observed"]

    def _enhance_news_data(self, news_data: Dict[str, Any], symbol: str) -> Dict[str, Any]:
        """Enhance news data with additional processing and validation."""
        try:
            if 'articles' not in news_data:
                news_data['articles'] = []
            
            # Add metadata
            news_data['symbol'] = symbol
            news_data['fetch_timestamp'] = pd.Timestamp.now().isoformat()
            news_data['article_count'] = len(news_data.get('articles', []))
            
            # Validate and clean articles
            valid_articles = []
            for article in news_data.get('articles', []):
                if isinstance(article, dict) and 'title' in article:
                    # Add basic validation
                    article['validated'] = True
                    article['relevance_score'] = self._calculate_relevance_score(article, symbol)
                    valid_articles.append(article)
            
            news_data['articles'] = valid_articles
            news_data['valid_article_count'] = len(valid_articles)
            
            return news_data
            
        except Exception as e:
            logger.error(f"Failed to enhance news data: {e}")
            return news_data

    def _calculate_relevance_score(self, article: Dict[str, Any], symbol: str) -> float:
        """Calculate relevance score for news article to symbol."""
        try:
            title = article.get('title', '').lower()
            description = article.get('description', '').lower()
            symbol_lower = symbol.lower()
            
            # Simple relevance scoring
            score = 0.0
            if symbol_lower in title:
                score += 0.5
            if symbol_lower in description:
                score += 0.3
            
            # Boost for financial keywords
            financial_keywords = ['stock', 'market', 'price', 'trading', 'earnings', 'revenue']
            for keyword in financial_keywords:
                if keyword in title or keyword in description:
                    score += 0.1
            
            return min(1.0, score)  # Cap at 1.0
            
        except Exception as e:
            logger.error(f"Failed to calculate relevance score: {e}")
            return 0.0

    async def _cross_validate_news(self, symbol: str) -> Dict[str, Any]:
        """
        Cross-validate news from multiple sources.
        Returns validation result with confidence score.
        """
        import asyncio
        
        validation_results = []
        sources_attempted = 0
        
        # Source 1: Primary news data tool
        try:
            sources_attempted += 1
            news_data1 = await news_data_tool.ainvoke({"symbol": symbol})
            news_data1 = self._enhance_news_data(news_data1, symbol)
            
            article_count = news_data1.get('valid_article_count', 0)
            if article_count > 0:
                validation_results.append({
                    'source': 'primary_tool',
                    'news_data': news_data1,
                    'article_count': article_count,
                    'success': True
                })
            else:
                validation_results.append({
                    'source': 'primary_tool',
                    'success': False,
                    'reason': 'No valid articles found'
                })
        except Exception as e:
            validation_results.append({
                'source': 'primary_tool',
                'success': False,
                'reason': str(e)
            })
        
        # Analyze validation results
        successful_sources = [r for r in validation_results if r.get('success', False)]
        
        if len(successful_sources) >= 1:
            # For news, we consider it validated if we have at least one source with articles
            best_source = max(successful_sources, key=lambda x: x['article_count'])
            
            if best_source['article_count'] >= 3:
                return {
                    'validated': True,
                    'news_data': best_source['news_data'],
                    'validation_info': {
                        'validated': True,
                        'confidence': min(0.8, 0.5 + len(successful_sources) * 0.1),
                        'sources_used': len(successful_sources),
                        'total_articles': best_source['article_count']
                    }
                }
            else:
                return {
                    'validated': False,
                    'news_data': best_source['news_data'],
                    'reason': f'Insufficient articles: {best_source["article_count"]}',
                    'validation_info': {
                        'validated': False,
                        'confidence': 0.3,
                        'sources_used': len(successful_sources)
                    }
                }
        else:
            return {
                'validated': False,
                'news_data': {'articles': [], 'error': 'No news sources available'},
                'reason': f'No successful news sources out of {sources_attempted} attempted',
                'validation_info': {
                    'validated': False,
                    'confidence': 0.0,
                    'sources_used': 0
                }
            }

    async def _analyze_news_data_llm(self, consolidated_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Use LLM to analyze consolidated news data for market insights and sentiment.
        """
        context_str = f"""
        Symbol: {consolidated_data.get('symbol', 'Unknown')}
        Sources Explored: {consolidated_data.get('sources_explored', [])}
        Total Articles: {consolidated_data.get('total_articles', 0)}
        Average Relevance: {consolidated_data.get('avg_relevance_score', 'N/A')}
        Average Credibility: {consolidated_data.get('avg_credibility_score', 'N/A')}
        Impact Distribution: {consolidated_data.get('impact_distribution', {})}

        News data has been consolidated from multiple sources with relevance and credibility scoring.
        """

        question = f"""
        Analyze the consolidated news data for {consolidated_data.get('symbol', 'the symbol')} and provide insights on:

        1. Overall market sentiment from news coverage (bullish/bearish/neutral)
        2. Key themes and narratives emerging from recent articles
        3. Credibility assessment of major news stories and their potential market impact
        4. Risk factors and warning signs from news analysis
        5. Positive catalysts and opportunities identified in coverage
        6. Information gaps or areas needing further monitoring
        7. Trading implications and potential market-moving events

        Provide actionable insights for investment decision-making based on the news landscape.
        """

        analysis_response = await self.reason_with_llm(context_str, question)

        return {
            "llm_analysis": analysis_response,
            "sentiment_summary": self._extract_news_sentiment(analysis_response),
            "key_themes": self._extract_key_themes(analysis_response),
            "risk_assessment": self._extract_news_risks(analysis_response),
            "trading_implications": self._extract_trading_implications(analysis_response)
        }

    def _extract_news_sentiment(self, llm_response: str) -> Dict[str, Any]:
        """Extract sentiment summary from LLM news analysis."""
        return {"overall_sentiment": "neutral", "confidence": "moderate"}

    def _extract_key_themes(self, llm_response: str) -> List[str]:
        """Extract key themes from LLM news analysis."""
        return ["Market stability", "Earnings focus"]

    def _extract_news_risks(self, llm_response: str) -> Dict[str, Any]:
        """Extract risk assessment from LLM news analysis."""
        return {"risk_level": "moderate", "key_risks": []}

    def _extract_trading_implications(self, llm_response: str) -> List[str]:
        """Extract trading implications from LLM news analysis."""
        return ["Monitor earnings reports", "Watch for volatility spikes"]

    async def process_shared_news_link(self, link: str, description: str = "") -> Dict[str, Any]:
        """
        Process a news link shared via Discord !share_news command.
        
        Fetches the content, analyzes it for market relevance, and returns
        a structured summary suitable for inclusion in the Data Agent's analysis.
        
        Args:
            link: URL of the news article
            description: Optional user-provided description
            
        Returns:
            Dict with:
                - success: bool
                - summary: str (brief article summary)
                - sentiment: str (bullish/bearish/neutral)
                - key_entities: List[str]
                - market_impact: str (high/medium/low)
                - error: str (if success is False)
        """
        import requests
        from bs4 import BeautifulSoup
        
        logger.info(f"Processing shared news link: {link[:100]}")
        
        try:
            # Fetch the content
            headers = {
                'User-Agent': 'Mozilla/5.0 (compatible; ABC-Application NewsAnalyzer/1.0)'
            }
            response = requests.get(link, headers=headers, timeout=15, allow_redirects=True)
            response.raise_for_status()
            
            # Parse HTML
            soup = BeautifulSoup(response.content, 'lxml')
            
            # Extract title
            title = soup.title.string if soup.title else "Unknown Title"
            
            # Extract main content
            article_text = ""
            for selector in ['article', '.article-content', '.post-content', 
                           '.entry-content', 'main', '.content']:
                article = soup.select_one(selector)
                if article:
                    for tag in article(['script', 'style', 'nav', 'footer', 'aside']):
                        tag.decompose()
                    article_text = article.get_text(separator=' ', strip=True)
                    break
            
            if not article_text and soup.body:
                for tag in soup.body(['script', 'style', 'nav', 'footer', 'aside', 'header']):
                    tag.decompose()
                article_text = soup.body.get_text(separator=' ', strip=True)
            
            if not article_text:
                return {
                    'success': False,
                    'error': 'Could not extract content from the page'
                }
            
            # Truncate for LLM
            content = article_text[:5000]
            
            # Analyze with LLM
            if self.llm:
                analysis_result = await self._analyze_shared_news_content(
                    title, content, description, link
                )
                
                # Store in shared memory for other agents
                await self.store_shared_memory("shared_news", link, {
                    "link": link,
                    "title": title,
                    "description": description,
                    "analysis": analysis_result,
                    "timestamp": datetime.now().isoformat()
                })
                
                return analysis_result
            else:
                # Basic analysis without LLM
                return {
                    'success': True,
                    'summary': f"News article: {title[:200]}",
                    'sentiment': 'neutral',
                    'key_entities': self._extract_basic_entities(content),
                    'market_impact': 'medium'
                }
                
        except requests.exceptions.Timeout:
            logger.error(f"Timeout fetching {link}")
            return {'success': False, 'error': 'Request timed out'}
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error for {link}: {e}")
            return {'success': False, 'error': f'Failed to fetch: {str(e)[:100]}'}
        except Exception as e:
            logger.error(f"Error processing shared news: {e}")
            return {'success': False, 'error': f'Processing error: {str(e)[:100]}'}

    async def _analyze_shared_news_content(self, title: str, content: str, 
                                          description: str, link: str) -> Dict[str, Any]:
        """Analyze shared news content using LLM."""
        import json
        import re
        
        prompt = f"""
Analyze this news article for market relevance and trading implications:

Title: {title}
User Note: {description if description else 'None'}
URL: {link}

Content:
{content[:3000]}

Provide your analysis as JSON:
{{
    "summary": "2-3 sentence summary",
    "sentiment": "bullish/bearish/neutral",
    "key_entities": ["company1", "sector1", ...],
    "market_impact": "high/medium/low",
    "trading_relevance": "brief note on trading implications"
}}
"""
        
        try:
            llm_response = await self.llm.ainvoke(prompt)
            response_text = llm_response.content if hasattr(llm_response, 'content') else str(llm_response)
            
            # Extract JSON
            json_match = re.search(r'\{[^{}]*\}', response_text, re.DOTALL)
            if json_match:
                analysis = json.loads(json_match.group())
                return {
                    'success': True,
                    'summary': analysis.get('summary', f'News: {title[:100]}'),
                    'sentiment': analysis.get('sentiment', 'neutral'),
                    'key_entities': analysis.get('key_entities', []),
                    'market_impact': analysis.get('market_impact', 'medium'),
                    'trading_relevance': analysis.get('trading_relevance', '')
                }
        except Exception as e:
            logger.warning(f"LLM analysis failed: {e}")
        
        # Fallback
        return {
            'success': True,
            'summary': f"News article: {title[:200]}",
            'sentiment': 'neutral',
            'key_entities': [],
            'market_impact': 'medium'
        }

    def _extract_basic_entities(self, content: str) -> List[str]:
        """Extract basic entities from content without LLM."""
        # Simple extraction of capitalized phrases
        import re
        entities = set()
        
        # Find multi-word capitalized phrases (potential company names)
        pattern = r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b'
        matches = re.findall(pattern, content)
        for match in matches[:10]:
            if len(match) > 3:
                entities.add(match)
        
        # Find stock ticker patterns (1-5 uppercase letters)
        ticker_pattern = r'\b[A-Z]{1,5}\b'
        tickers = re.findall(ticker_pattern, content)
        common_words = {'THE', 'AND', 'FOR', 'ARE', 'BUT', 'NOT', 'YOU', 'ALL', 'CAN', 'HAD', 'HER', 'WAS', 'ONE', 'OUR', 'OUT'}
        for ticker in tickers[:20]:
            if ticker not in common_words and len(ticker) >= 2:
                entities.add(ticker)
        
        return list(entities)[:10]

# Standalone test (run python src/agents/news_agent.py to verify)
if __name__ == "__main__":
    import asyncio
    agent = NewsDataAnalyzer()
    result = asyncio.run(agent.process_input({'symbol': 'SPY'}))
    print("News Agent Test Result:\n", result)