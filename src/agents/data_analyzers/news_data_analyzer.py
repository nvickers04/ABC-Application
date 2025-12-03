# src/agents/data_analyzers/news_data_analyzer.py
# News Data Analyzer Subagent
# Fetches and analyzes news data from multiple sources
# Handles sentiment analysis, event detection, and news impact assessment

import asyncio
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime, timedelta
import pandas as pd
from dataclasses import dataclass
import json
import numpy as np

from src.agents.data_analyzers.base_data_analyzer import BaseDataAnalyzer
from src.utils.redis_cache import get_redis_cache_manager
from src.utils.news_tools import NewsDataTool
from src.utils.memory_manager import get_memory_manager  # Assume this exists
# from src.utils.llm import get_llm  # Assume LLM initializer

logger = logging.getLogger(__name__)

@dataclass
class NewsArticle:
    """Data class for news article representation."""
    title: str
    content: str
    source: str
    published_at: datetime
    url: str
    sentiment: Optional[float] = None
    relevance_score: Optional[float] = None
    topics: Optional[List[str]] = None
    symbol_relevance: Optional[Dict[str, float]] = None
    named_entities: Optional[Dict[str, List[str]]] = None
    keywords: Optional[List[str]] = None
    readability_score: Optional[float] = None
    quality_score: Optional[float] = None

class NewsDataAnalyzer(BaseDataAnalyzer):
    """
    Subagent for fetching and analyzing news data from multiple sources.
    Handles news aggregation, sentiment analysis, and event detection.
    """

    def __init__(self):
        super().__init__(role="news_data")
        self.cache_manager = get_redis_cache_manager()
        if self.cache_manager is None:
            raise RuntimeError("Redis cache manager not available")
        assert self.cache_manager is not None
        self.max_articles_per_query = 20
        self.relevance_threshold = 0.3
        self.sources_priority = {
            "newsapi": 9,
            "currentsapi": 8,
            "financial_times": 7,
            "reuters": 6,
            "bloomberg": 5
        }
        self.memory_manager = get_memory_manager()
        self.news_tool = NewsDataTool()
        self.llm = None  # TODO: Initialize LLM

    async def _plan_data_exploration(self, *args, **kwargs) -> Dict[str, Any]:
        """
        Plan news data exploration.

        Args:
            input_data: Input parameters

        Returns:
            Exploration plan
        """
        symbols = kwargs.get("symbols", [])
        focus_areas = kwargs.get("focus_areas", ["earnings", "mergers", "regulations"])
        return await self._get_news_exploration_plan(symbols, focus_areas)

    async def _execute_data_exploration(self, exploration_plan: Dict[str, Any], *args, **kwargs) -> Dict[str, Any]:
        """
        Execute news data exploration.

        Args:
            exploration_plan: Plan from _plan_data_exploration

        Returns:
            Raw news data
        """
        symbols = exploration_plan.get("symbols", [])
        time_period = exploration_plan.get("time_period", "last_24h")
        articles = await self._fetch_news_concurrently(symbols, exploration_plan, time_period)
        return {"articles": articles, "symbols": symbols, "focus_areas": exploration_plan.get("focus_areas", [])}

    async def _enhance_data(self, validated_data: Dict[str, Any], *args, **kwargs) -> Dict[str, Any]:
        """
        Enhance news data with analysis.

        Args:
            validated_data: Validated data

        Returns:
            Enhanced data with analysis
        """
        # Filter and rank articles
        relevant_articles = self._filter_relevant_articles(validated_data.get("articles", []), validated_data.get("symbols", []), validated_data.get("focus_areas", []))

        # Perform sentiment analysis
        sentiment_results = await self._analyze_news_sentiment(relevant_articles)

        # Detect events and themes
        event_detection = await self._detect_news_events(relevant_articles, sentiment_results)

        # Validate data quality
        quality_score = await self._validate_news_quality(relevant_articles, sentiment_results)

        # Convert to DataFrame for analysis
        articles_df = pd.DataFrame([article.__dict__ for article in relevant_articles]) if relevant_articles else pd.DataFrame()

        return {
            "articles_df": articles_df,
            "sentiment_summary": sentiment_results,
            "event_detection": event_detection,
            "sources_used": list(set(article.source for article in relevant_articles)) if relevant_articles else [],
            "quality_score": quality_score,
            "total_articles_fetched": len(validated_data.get("articles", [])),
            "relevant_articles_count": len(relevant_articles)
        }

    async def process_input(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        # Use base class process_input
        base_result = await super().process_input(input_data)
        # Add any news-specific processing if needed
        return base_result

    async def _validate_news_quality(self, articles: List[NewsArticle], sentiment_results: Dict[str, Any]) -> float:
        """Validate news data quality asynchronously."""
        quality_scores = []
        for article in articles:
            scores = {
                "source_credibility": self.sources_priority.get(article.source, 5) / 10,
                "content_length": min(len(article.content) / 500, 1.0),
                "recency": min(1.0, (datetime.now() - article.published_at).days / 7),
                "sentiment_consistency": abs(article.sentiment - sentiment_results["average_sentiment"]) < 0.5 if article.sentiment else 0.8,
                "content_consistency": await self._check_content_consistency(article),
                "fact_check_score": await self._perform_fact_check(article),
            }
            article.quality_score = float(np.mean(list(scores.values())))
            quality_scores.append(article.quality_score)

        return float(np.mean(quality_scores)) if quality_scores else 0.0

    async def _fetch_news_concurrently(self, symbols: List[str], plan: Dict, time_period: str) -> List[NewsArticle]:
        """
        Fetch news concurrently from multiple sources based on exploration plan.
        
        Args:
            symbols: List of symbols to search news for
            plan: LLM exploration plan
            time_period: Time period for news search
            
        Returns:
            List of news articles
        """
        tasks = []
        all_articles = []
        
        for source in plan.get("sources", []):
            if source in self.sources_priority:
                priority = self.sources_priority[source]
                if priority >= plan.get("priorities", {}).get(source, 0):
                    task = self._fetch_from_source(source, symbols, time_period, plan)
                    tasks.append((priority, task))
        
        # Sort by priority and execute concurrently
        tasks.sort(key=lambda x: x[0], reverse=True)
        concurrent_tasks = [task for _, task in tasks[:self.max_concurrent_requests]]
        
        if concurrent_tasks:
            results = await asyncio.gather(*concurrent_tasks, return_exceptions=True)
            for result in results:
                if isinstance(result, list):
                    all_articles.extend(result)
        
        return all_articles

    async def _fetch_from_source(self, source: str, symbols: List[str], time_period: str, plan: Dict) -> List[NewsArticle]:
        """
        Fetch news from a specific source.
        
        Args:
            source: News source name
            symbols: Symbols to search for
            time_period: Time period
            plan: Exploration plan
            
        Returns:
            List of news articles from the source
        """
        try:
            # Check cache first
            cache_key = f"{source}:{symbols[0] if symbols else 'general'}:{time_period}"
            cached_articles = self.cache_manager.get("news", cache_key)  # type: ignore
            if cached_articles:
                logger.info(f"Cache hit for news from {source}")
                return [NewsArticle(**article) for article in json.loads(cached_articles)]
            
            # Use the appropriate tool for the source
            if source == "newsapi":
                articles = await self._fetch_newsapi_articles(symbols, time_period)
            elif source == "currentsapi":
                articles = await self._fetch_currentsapi_articles(symbols, time_period)
            elif source == "financial_times":
                articles = await self._fetch_ft_articles(symbols, time_period)
            else:
                logger.warning(f"Source {source} not implemented")
                articles = []
            
            # Convert to NewsArticle objects
            news_articles = []
            for article_data in articles:
                try:
                    news_article = NewsArticle(
                        title=article_data.get("title", ""),
                        content=article_data.get("content", article_data.get("description", "")),
                        source=source,
                        published_at=datetime.fromisoformat(article_data.get("published_at", datetime.now().isoformat())),
                        url=article_data.get("url", ""),
                        sentiment=None,  # Will be analyzed later
                        relevance_score=None  # Will be calculated later
                    )
                    news_articles.append(news_article)
                except Exception as e:
                    logger.error(f"Failed to parse article from {source}: {e}")
                    continue
            
            # Cache the results
            cache_data = [article.__dict__ for article in news_articles]
            self.cache_manager.set("news", cache_key, json.dumps(cache_data), ttl_seconds=1800)  # type: ignore  # 30 minutes
            
            logger.info(f"Fetched {len(news_articles)} articles from {source}")
            return news_articles
            
        except Exception as e:
            logger.error(f"Failed to fetch news from {source}: {e}")
            return []

    async def _fetch_newsapi_articles(self, symbols: List[str], time_period: str) -> List[Dict]:
        """
        Fetch articles from NewsAPI.
        
        Args:
            symbols: Symbols to search
            time_period: Time period
            
        Returns:
            List of article dictionaries
        """
        try:
            from src.utils.tools import news_data_tool
            query = " OR ".join(symbols)
            articles = await self.news_tool.ainvoke({"query": query, "language": "en", "page_size": self.max_articles_per_query})
            return articles.get("articles", [])
        except Exception as e:
            logger.error(f"NewsAPI fetch failed: {e}")
            return []

    async def _fetch_currentsapi_articles(self, symbols: List[str], time_period: str) -> List[Dict]:
        """
        Fetch articles from CurrentsAPI.
        
        Args:
            symbols: Symbols to search
            time_period: Time period
            
        Returns:
            List of article dictionaries
        """
        try:
            from src.utils.tools import news_data_tool
            query = " OR ".join(symbols)
            articles = await self.news_tool.ainvoke({"query": query, "language": "en", "page_size": self.max_articles_per_query})
            return articles.get("articles", [])
        except Exception as e:
            logger.error(f"CurrentsAPI fetch failed: {e}")
            return []

    async def _fetch_ft_articles(self, symbols: List[str], time_period: str) -> List[Dict]:
        """
        Fetch articles from Financial Times (placeholder).
        
        Args:
            symbols: Symbols to search
            time_period: Time period
            
        Returns:
            List of article dictionaries
        """
        # FT API implementation would go here
        logger.info("Financial Times API integration pending")
        return []

    def _filter_relevant_articles(self, articles: List[NewsArticle], symbols: List[str], focus_areas: List[str]) -> List[NewsArticle]:
        """
        Filter and rank news articles by relevance to symbols and focus areas.
        
        Args:
            articles: All fetched articles
            symbols: Target symbols
            focus_areas: Focus areas for analysis
            
        Returns:
            List of relevant articles
        """
        relevant_articles = []
        
        for article in articles:
            # Calculate relevance score
            relevance_score = self._calculate_relevance_score(article, symbols, focus_areas)
            
            if relevance_score > self.relevance_threshold:
                article.relevance_score = relevance_score
                # Extract topics
                article.topics = self._extract_topics(article.content, focus_areas)
                relevant_articles.append(article)
        
        # Sort by relevance
        relevant_articles.sort(key=lambda x: x.relevance_score, reverse=True)
        return relevant_articles[:self.max_articles_per_query]

    def _calculate_relevance_score(self, article: NewsArticle, symbols: List[str], focus_areas: List[str]) -> float:
        """
        Calculate relevance score for an article.
        
        Args:
            article: News article
            symbols: Target symbols
            focus_areas: Focus areas
            
        Returns:
            Relevance score between 0-1
        """
        score = 0.0
        
        # Symbol mention score
        content_lower = article.content.lower()
        title_lower = article.title.lower()
        symbol_mentions = sum(1 for symbol in symbols if symbol.lower() in content_lower or symbol.lower() in title_lower)
        symbol_score = min(symbol_mentions / len(symbols), 1.0)
        score += symbol_score * 0.4
        
        # Focus area score
        focus_matches = sum(1 for area in focus_areas if area.lower() in content_lower or area.lower() in title_lower)
        focus_score = focus_matches / len(focus_areas)
        score += focus_score * 0.3
        
        # Source quality score
        source_score = self.sources_priority.get(article.source, 0) / 10.0
        score += source_score * 0.2
        
        # Recency score
        recency_hours = (datetime.now() - article.published_at).total_seconds() / 3600
        recency_score = max(0, 1 - (recency_hours / 24))  # 24 hour window
        score += recency_score * 0.1
        
        return min(score, 1.0)

    def _extract_topics(self, content: str, focus_areas: List[str]) -> List[str]:
        """
        Extract relevant topics from article content.
        
        Args:
            content: Article content
            focus_areas: Focus areas
            
        Returns:
            List of extracted topics
        """
        topics = []
        content_lower = content.lower()
        
        for area in focus_areas:
            if area.lower() in content_lower:
                topics.append(area)
        
        # Add general financial topics if relevant
        if any(word in content_lower for word in ["earnings", "revenue", "profit", "loss"]):
            topics.append("earnings")
        if any(word in content_lower for word in ["merger", "acquisition", "deal"]):
            topics.append("m&a")
        if any(word in content_lower for word in ["regulation", "compliance", "law"]):
            topics.append("regulatory")
        
        return list(set(topics))  # Remove duplicates

    async def _analyze_news_sentiment(self, articles: List[NewsArticle]) -> Dict[str, Any]:
        """Perform sentiment analysis on news articles."""
        sentiments = []
        for article in articles:
            # Simple fallback sentiment if vader not available
            score = 0.0  # Neutral
            article.sentiment = score
            sentiments.append(score)

        return {
            "average_sentiment": np.mean(sentiments) if sentiments else 0.0,
            "sentiment_distribution": pd.Series(sentiments).describe().to_dict() if sentiments else {}
        }

    async def _detect_news_events(self, articles: List[NewsArticle], sentiment_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Detect events and themes from news articles.
        """
        events = {
            "earnings_events": [],
            "m_and_a_events": [],
            "regulatory_events": [],
            "significant_sentiment_shifts": [],
            "event_confidence": {}
        }
        
        for i, article in enumerate(articles):
            content_lower = article.content.lower()
            
            # Earnings events
            if any(word in content_lower for word in ["earnings", "quarterly results", "eps", "revenue"]):
                events["earnings_events"].append({
                    "article_index": i,
                    "title": article.title,
                    "sentiment": article.sentiment,
                    "confidence": 0.8
                })
            
            # M&A events
            if any(word in content_lower for word in ["acquisition", "merger", "deal", "buyout"]):
                events["m_and_a_events"].append({
                    "article_index": i,
                    "title": article.title,
                    "sentiment": article.sentiment,
                    "confidence": 0.7
                })
            
            # Regulatory events
            if any(word in content_lower for word in ["regulation", "sec", "fda", "compliance", "lawsuit"]):
                events["regulatory_events"].append({
                    "article_index": i,
                    "title": article.title,
                    "sentiment": article.sentiment,
                    "confidence": 0.6
                })
        
        # Sentiment shift detection (compare to historical baseline)
        if sentiment_results.get("average_compound"):
            historical_sentiment = await self.memory_manager.retrieve_memory("market_sentiment_baseline")
            if historical_sentiment:
                shift = sentiment_results["average_compound"] - historical_sentiment
                if abs(shift) > 0.1:  # Significant shift
                    events["significant_sentiment_shifts"].append({
                        "shift_magnitude": shift,
                        "new_sentiment": sentiment_results["overall_sentiment"],
                        "confidence": 0.75
                    })
        
        return events

    async def _get_news_exploration_plan(self, symbols: List[str], focus_areas: List[str]) -> Dict:
        """
        Use LLM to determine optimal news exploration strategy.
        
        Args:
            symbols: List of symbols
            focus_areas: Focus areas for news search
            
        Returns:
            Exploration plan dictionary
        """
        prompt = f"""
        For symbols {symbols} and focus areas {focus_areas}, create a news exploration plan.
        Consider source priorities, concurrent search opportunities, and content analysis needs.
        Return JSON with sources, priorities, concurrent_groups, search_parameters, and exploration_strategy.
        Sources available: newsapi, currentsapi, financial_times, reuters, bloomberg.
        """
        
        try:
            if self.llm:
                response = await self.llm.ainvoke(prompt)  # type: ignore
                plan = self._parse_llm_response(response.content)
                return plan
            else:
                logger.warning("LLM not available, using fallback plan")
                return {
                    "sources": ["newsapi", "currentsapi"],
                    "priorities": {"newsapi": 9, "currentsapi": 8},
                    "concurrent_groups": [["newsapi"], ["currentsapi"]],
                    "search_parameters": {
                        "language": "en",
                        "limit": self.max_articles_per_query,
                        "time_period": "last_24h"
                    },
                    "focus_areas": focus_areas,
                    "symbols": symbols
                }
        except Exception as e:
            logger.error(f"LLM news exploration plan failed: {e}")
            return {
                "sources": ["newsapi", "currentsapi"],
                "priorities": {"newsapi": 9, "currentsapi": 8},
                "concurrent_groups": [["newsapi"], ["currentsapi"]],
                "search_parameters": {
                    "language": "en",
                    "limit": self.max_articles_per_query,
                    "time_period": "last_24h"
                },
                "exploration_strategy": "source_priority"
            }

    def _parse_llm_response(self, response: str) -> Dict:
        """
        Parse LLM response into structured exploration plan.
        
        Args:
            response: LLM response string
            
        Returns:
            Structured exploration plan
        """
        try:
            # Extract JSON from response
            start = response.find('{')
            end = response.rfind('}') + 1
            if start == -1 or end == 0:
                raise ValueError("No JSON found in response")
            json_str = response[start:end]
            return json.loads(json_str)
        except Exception as e:
            logger.error(f"Failed to parse LLM news response: {e}")
            return {}

    async def _cross_validate_news(self, articles: List[NewsArticle], symbols: List[str]) -> Dict[str, Any]:
        """
        Cross-validate news articles across multiple sources and methods.
        
        Args:
            articles: List of news articles
            symbols: Target symbols
            
        Returns:
            Validation results
        """
        try:
            # Validate article authenticity and source credibility
            validation_results = []
            
            for article in articles:
                validation = {
                    "article_id": id(article),
                    "source_credible": self._validate_source_credibility(article.source),
                    "content_consistency": await self._check_content_consistency(article),
                    "fact_check_score": await self._perform_fact_check(article),
                    "duplicate_check": self._check_for_duplicates(article, articles)
                }
                
                # Overall validation score
                validation["overall_score"] = sum([
                    validation["source_credible"],
                    validation["content_consistency"],
                    validation["fact_check_score"],
                    1 if not validation["duplicate_check"] else 0
                ]) / 4
                
                validation_results.append(validation)
            
            # Overall validation summary
            valid_articles = sum(1 for v in validation_results if v["overall_score"] > 0.7)
            validation_summary = {
                "total_articles": len(articles),
                "valid_articles": valid_articles,
                "validation_rate": valid_articles / len(articles) if articles else 0,
                "common_issues": self._summarize_validation_issues(validation_results),
                "validated": valid_articles > 0
            }
            
            return {
                "validation_results": validation_results,
                "summary": validation_summary,
                "news_data": [article.__dict__ for article in articles]
            }
            
        except Exception as e:
            logger.error(f"News cross-validation failed: {e}")
            return {
                "error": str(e),
                "validated": False,
                "news_data": []
            }

    def _validate_source_credibility(self, source: str) -> float:
        """
        Validate the credibility of a news source.
        
        Args:
            source: News source name
            
        Returns:
            Credibility score 0-1
        """
        credible_sources = {
            "reuters": 0.95,
            "bloomberg": 0.95,
            "financial_times": 0.92,
            "wsj": 0.90,
            "nytimes": 0.88,
            "newsapi": 0.75,  # Aggregator
            "currentsapi": 0.70  # Aggregator
        }
        
        return credible_sources.get(source.lower(), 0.5)

    async def _check_content_consistency(self, article: NewsArticle) -> float:
        """
        Check content consistency and factual accuracy.
        
        Args:
            article: News article
            
        Returns:
            Consistency score 0-1
        """
        # Placeholder for content analysis
        # Would use NLP models to check for inconsistencies, bias, etc.
        return 0.8  # Default score

    async def _perform_fact_check(self, article: NewsArticle) -> float:
        """
        Perform fact-checking on article content.
        
        Args:
            article: News article
            
        Returns:
            Fact-check score 0-1
        """
        # Placeholder for fact-checking service integration
        # Would use services like Google Fact Check Tools, ClaimBuster, etc.
        return 0.85  # Default score

    def _check_for_duplicates(self, article: NewsArticle, all_articles: List[NewsArticle]) -> bool:
        """
        Check if article is duplicate of existing articles.
        
        Args:
            article: Article to check
            all_articles: All articles
            
        Returns:
            True if duplicate found
        """
        article_signature = (article.title.lower(), article.content[:100].lower())
        
        for other_article in all_articles:
            if other_article != article:
                other_signature = (other_article.title.lower(), other_article.content[:100].lower())
                if article_signature == other_signature:
                    return True
        
        return False

    def _summarize_validation_issues(self, validation_results: List[Dict]) -> List[str]:
        """
        Summarize common validation issues.
        
        Args:
            validation_results: List of validation results
            
        Returns:
            List of common issues
        """
        issues = []
        issue_counts = {
            "low_source_credibility": 0,
            "content_inconsistency": 0,
            "fact_check_failures": 0,
            "duplicates": 0
        }
        
        for result in validation_results:
            if result["source_credible"] < 0.7:
                issue_counts["low_source_credibility"] += 1
            if result["content_consistency"] < 0.7:
                issue_counts["content_inconsistency"] += 1
            if result["fact_check_score"] < 0.7:
                issue_counts["fact_check_failures"] += 1
            if result["duplicate_check"]:
                issue_counts["duplicates"] += 1
        
        for issue, count in issue_counts.items():
            if count > 0:
                percentage = (count / len(validation_results)) * 100
                issues.append(f"{count} articles ({percentage:.1f}%) with {issue.replace('_', ' ')}")
        
        return issues

    async def _enhance_news_data(self, articles: List[NewsArticle], symbols: List[str]) -> List[NewsArticle]:
        """
        Enhance news articles with additional analysis and metadata.
        
        Args:
            articles: List of news articles
            symbols: Target symbols
            
        Returns:
            Enhanced list of news articles
        """
        enhanced_articles = []
        
        for article in articles:
            # Add symbol relevance
            article.symbol_relevance = self._calculate_symbol_relevance(article, symbols)
            
            # Add entity recognition
            article.named_entities = await self._extract_named_entities(article.content)
            
            # Add keyword extraction
            article.keywords = self._extract_keywords(article.content)
            
            # Add readability score
            article.readability_score = self._calculate_readability(article.content)
            
            enhanced_articles.append(article)
        
        return enhanced_articles

    def _calculate_symbol_relevance(self, article: NewsArticle, symbols: List[str]) -> Dict[str, float]:
        """Calculate relevance to symbols."""
        relevance = {}
        for symbol in symbols:
            relevance[symbol] = article.content.lower().count(symbol.lower()) / len(article.content.split())
        return relevance

    async def _extract_named_entities(self, content: str) -> Dict[str, List[str]]:
        """Extract named entities."""
        # Placeholder implementation
        return {"entities": []}

    def _extract_keywords(self, content: str) -> List[str]:
        """Extract keywords."""
        # Placeholder
        return []

    def _calculate_readability(self, content: str) -> float:
        """Calculate readability score."""
        # Placeholder
        return 0.8

    def _analyze_symbol_context(self, content: str, symbol: str) -> float:
        """
        Analyze the context around symbol mentions to determine sentiment.
        
        Args:
            content: Article content
            symbol: Symbol to analyze
            
        Returns:
            Context sentiment score -1 to 1
        """
        # Placeholder for context analysis
        # Would use dependency parsing and sentiment around mentions
        return 0.0  # Neutral context



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
            title = soup.title.string if soup.title and soup.title.string else "Unknown Title"
            
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

    async def _analyze_shared_news_content(self, title: str, content: str, description: str, link: str) -> Dict[str, Any]:
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
            if self.llm:
                llm_response = await self.llm.ainvoke(prompt)  # type: ignore
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
    test_input = {"symbols": ["AAPL"], "time_period": "last_24h"}
    result = asyncio.run(agent.process_input(test_input))
    print("News Analyzer Test Result:\n", result)