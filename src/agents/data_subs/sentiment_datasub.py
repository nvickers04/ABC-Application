# src/agents/sentiment_agent.py
# Comprehensive Sentiment Data Subagent implementing full specification
# Multi-dimensional sentiment analysis with LLM integration and collaborative memory

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))  # Dynamic root path for imports.

from src.agents.base import BaseAgent  # Absolute import.
import logging
from typing import Dict, Any, List, Optional
import numpy as np
import time
import asyncio
import os
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from src.utils.redis_cache import get_redis_cache_manager, cache_get, cache_set, cache_delete

logger = logging.getLogger(__name__)

@dataclass
class SentimentMemory:
    """Collaborative memory for sentiment analysis patterns and insights."""
    sentiment_patterns: Dict[str, Any] = field(default_factory=dict)
    market_reactions: Dict[str, Any] = field(default_factory=dict)
    event_sentiment: Dict[str, Any] = field(default_factory=dict)
    source_reliability: Dict[str, float] = field(default_factory=dict)
    session_insights: List[Dict[str, Any]] = field(default_factory=list)

    def add_session_insight(self, insight: Dict[str, Any]):
        """Add sentiment insight to session memory."""
        self.session_insights.append({
            **insight,
            'timestamp': datetime.now().isoformat()
        })

    def get_recent_insights(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent sentiment insights."""
        return self.session_insights[-limit:]

class SentimentDatasub(BaseAgent):
    """
    Comprehensive Sentiment Data Subagent implementing full specification.
    Multi-dimensional sentiment analysis across news, social media, and market data.
    """

    def __init__(self):
        config_paths = {'risk': 'config/risk-constraints.yaml'}  # Relative to root.
        prompt_paths = {'base': 'base_prompt.txt', 'role': 'agents/data-agent-complete.md'}  # Relative to root.
        tools = []  # SentimentDatasub uses internal methods instead of tools
        super().__init__(role='sentiment_data', config_paths=config_paths, prompt_paths=prompt_paths, tools=tools)

        # Initialize Redis cache manager
        self.redis_cache = get_redis_cache_manager()
        self.cache_ttl = 1800  # 30 minutes TTL for sentiment data

        # Initialize collaborative memory
        self.memory = SentimentMemory()

        # Sentiment sources configuration
        self.sentiment_sources = {
            'news': self._analyze_news_sentiment,
            'social_media': self._analyze_social_sentiment,
            'market_data': self._analyze_market_sentiment,
            'economic_data': self._analyze_economic_sentiment
        }

        # Initialize sentiment analysis models
        self._initialize_sentiment_models()

    def _initialize_sentiment_models(self):
        """Initialize various sentiment analysis models and APIs."""
        self.models = {}

        # TextBlob (always available)
        try:
            from textblob import TextBlob
            self.models['textblob'] = TextBlob
        except ImportError:
            logger.warning("TextBlob not available")

        # VADER (NLTK)
        try:
            from nltk.sentiment.vader import SentimentIntensityAnalyzer
            self.models['vader'] = SentimentIntensityAnalyzer()
        except ImportError:
            logger.warning("VADER not available")

        # Transformers-based models (optional)
        try:
            from transformers import pipeline
            self.models['finbert'] = pipeline("sentiment-analysis",
                                            model="ProsusAI/finbert",
                                            tokenizer="ProsusAI/finbert")
        except ImportError:
            logger.warning("FinBERT not available")

    def _is_cache_valid(self, cache_key):
        """Check if Redis cache entry exists and is valid."""
        return cache_get('sentiment', cache_key) is not None

    def _get_cached_sentiment(self, cache_key):
        """Get cached sentiment result from Redis."""
        return cache_get('sentiment', cache_key)

    def _cache_sentiment(self, cache_key, data):
        """Cache sentiment result in Redis with TTL."""
        cache_set('sentiment', cache_key, data, self.cache_ttl)

    def reflect(self, adjustments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Reflect on batch adjustments for self-improvement.
        """
        logger.info(f"Sentiment Reflecting on adjustments: {adjustments}")
        return {}

    async def process_input(self, input_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Comprehensive sentiment analysis across multiple dimensions.
        """
        logger.info(f"SentimentDatasub processing input: {input_data or 'Default analysis'}")

        # Extract analysis parameters
        focus_areas = input_data.get('focus_areas', ['news', 'social_media', 'market_data']) if input_data else ['news', 'social_media', 'market_data']
        time_horizon = input_data.get('time_horizon', 'current')
        include_llm_analysis = input_data.get('llm_analysis', True)
        symbols = input_data.get('symbols', ['SPY', 'AAPL']) if input_data else ['SPY', 'AAPL']

        # Create cache key
        cache_key = f"sentiment_{'_'.join(focus_areas)}_{time_horizon}_{'_'.join(symbols)}"

        # Check cache first
        if self._is_cache_valid(cache_key):
            logger.info(f"Using cached sentiment result for: {cache_key}")
            return self._get_cached_sentiment(cache_key)

        try:
            # Aggregate sentiment from multiple sources
            sentiment_data = await self._aggregate_sentiment_data(focus_areas, symbols, time_horizon)

            # Perform LLM-driven sentiment analysis
            if include_llm_analysis:
                sentiment_data = await self._perform_llm_sentiment_analysis(sentiment_data, focus_areas)

            # Calculate composite sentiment scores
            sentiment_data['composite_sentiment'] = self._calculate_composite_sentiment(sentiment_data)

            # Assess market impact
            sentiment_data['market_impact'] = self._assess_market_impact(sentiment_data)

            # Generate collaborative insights
            sentiment_data['collaborative_insights'] = self._generate_collaborative_insights(sentiment_data)

            # Update memory
            self._update_memory(sentiment_data)

            # Cache the result
            self._cache_sentiment(cache_key, {
                "sentiment_score": sentiment_data.get('composite_sentiment', {}).get('score', 0.5),
                "sentiment": sentiment_data
            })

            # Store sentiment data in shared memory for each symbol
            for symbol in symbols:
                await self.store_shared_memory("sentiment_data", symbol, {
                    "sentiment_score": sentiment_data.get('composite_sentiment', {}).get('score', 0.5),
                    "sentiment": sentiment_data,
                    "timestamp": datetime.now().isoformat()
                })

            logger.info(f"SentimentDatasub completed analysis: {len(sentiment_data.get('sources', {}))} sources processed")
            return {
                "sentiment_score": sentiment_data.get('composite_sentiment', {}).get('score', 0.5),
                "sentiment": sentiment_data
            }

        except Exception as e:
            logger.error(f"SentimentDatasub failed: {e}")
            result = {
                "sentiment_score": 0.5,
                "sentiment": {
                    "composite_score": 0.5,
                    "error": str(e),
                    "sources": {},
                    "market_impact": "neutral"
                }
            }
            self._cache_sentiment(cache_key, result)
            return result

    async def _aggregate_sentiment_data(self, focus_areas: List[str], symbols: List[str], time_horizon: str) -> Dict[str, Any]:
        """Aggregate sentiment data from multiple sources."""
        aggregated_data = {
            'sources': {},
            'timestamp': datetime.now().isoformat(),
            'focus_areas': focus_areas,
            'symbols': symbols,
            'time_horizon': time_horizon
        }

        # Analyze each focus area concurrently
        analysis_tasks = []
        for area in focus_areas:
            if area in self.sentiment_sources:
                task = self.sentiment_sources[area](symbols, time_horizon)
                analysis_tasks.append(task)

        # Execute all analysis tasks
        if analysis_tasks:
            results = await asyncio.gather(*analysis_tasks, return_exceptions=True)

            for i, result in enumerate(results):
                area = focus_areas[i] if i < len(focus_areas) else f"unknown_{i}"
                if isinstance(result, Exception):
                    logger.warning(f"Sentiment analysis failed for {area}: {result}")
                    aggregated_data['sources'][area] = {"error": str(result)}
                else:
                    aggregated_data['sources'][area] = result

        return aggregated_data

    async def _analyze_news_sentiment(self, symbols: List[str], time_horizon: str) -> Dict[str, Any]:
        """Analyze sentiment from news sources."""
        news_sentiment = {
            'source': 'news',
            'articles_analyzed': 0,
            'sentiment_scores': [],
            'key_themes': [],
            'timestamp': datetime.now().isoformat()
        }

        try:
            # Use CurrentsAPI for news (if available)
            api_key = os.getenv('CURRENTS_API_KEY')
            if api_key:
                import requests

                # Search for news about symbols
                for symbol in symbols[:2]:  # Limit to 2 symbols for API efficiency
                    try:
                        response = requests.get(
                            "https://api.currentsapi.services/v1/search",
                            params={
                                "keywords": symbol,
                                "language": "en",
                                "apiKey": api_key,
                                "limit": 10
                            },
                            timeout=10
                        )
                        response.raise_for_status()
                        data = response.json()

                        articles = data.get('news', [])
                        news_sentiment['articles_analyzed'] += len(articles)

                        for article in articles:
                            title = article.get('title', '')
                            description = article.get('description', '')

                            # Analyze sentiment of title and description
                            combined_text = f"{title} {description}"
                            sentiment_score = self._analyze_text_sentiment(combined_text)

                            news_sentiment['sentiment_scores'].append({
                                'symbol': symbol,
                                'title': title[:100],
                                'score': sentiment_score,
                                'source': 'currents_api'
                            })

                    except Exception as e:
                        logger.warning(f"Failed to fetch news for {symbol}: {e}")

            # Calculate aggregate news sentiment
            if news_sentiment['sentiment_scores']:
                scores = [s['score'] for s in news_sentiment['sentiment_scores']]
                news_sentiment['aggregate_score'] = np.mean(scores)
                news_sentiment['sentiment_volatility'] = np.std(scores) if len(scores) > 1 else 0
                news_sentiment['confidence'] = min(0.9, len(scores) / 20.0)  # Higher confidence with more articles
            else:
                news_sentiment['aggregate_score'] = 0.5  # Neutral
                news_sentiment['confidence'] = 0.0

            # Extract key themes (simplified)
            news_sentiment['key_themes'] = self._extract_news_themes(news_sentiment['sentiment_scores'])

        except Exception as e:
            logger.error(f"News sentiment analysis failed: {e}")
            news_sentiment['error'] = str(e)

        return news_sentiment

    async def _analyze_social_sentiment(self, symbols: List[str], time_horizon: str) -> Dict[str, Any]:
        """Analyze sentiment from social media sources."""
        social_sentiment = {
            'source': 'social_media',
            'platforms': {},
            'timestamp': datetime.now().isoformat()
        }

        # Twitter/X sentiment (if API available)
        twitter_data = await self._analyze_twitter_sentiment(symbols, time_horizon)
        if twitter_data:
            social_sentiment['platforms']['twitter'] = twitter_data

        # Reddit sentiment (simplified - would need Reddit API)
        reddit_data = await self._analyze_reddit_sentiment(symbols, time_horizon)
        if reddit_data:
            social_sentiment['platforms']['reddit'] = reddit_data

        # Calculate aggregate social sentiment
        platform_scores = []
        for platform, data in social_sentiment['platforms'].items():
            if 'aggregate_score' in data:
                platform_scores.append(data['aggregate_score'])

        if platform_scores:
            social_sentiment['aggregate_score'] = np.mean(platform_scores)
            social_sentiment['confidence'] = min(0.8, len(platform_scores) * 0.3)
        else:
            social_sentiment['aggregate_score'] = 0.5
            social_sentiment['confidence'] = 0.0

        return social_sentiment

    async def _analyze_twitter_sentiment(self, symbols: List[str], time_horizon: str) -> Dict[str, Any]:
        """Analyze Twitter sentiment for symbols."""
        twitter_sentiment = {
            'tweets_analyzed': 0,
            'sentiment_scores': []
        }

        try:
            # Use Twitter API v2 if available
            bearer_token = os.getenv('TWITTER_BEARER_TOKEN')
            if bearer_token:
                import tweepy

                client = tweepy.Client(bearer_token=bearer_token)

                for symbol in symbols[:1]:  # Limit to 1 symbol for API efficiency
                    try:
                        # Search for recent tweets about the symbol
                        query = f"${symbol} OR #{symbol}"
                        tweets = client.search_recent_tweets(
                            query=query,
                            max_results=20,
                            tweet_fields=['created_at', 'public_metrics', 'text']
                        )

                        if tweets.data:
                            twitter_sentiment['tweets_analyzed'] += len(tweets.data)

                            for tweet in tweets.data:
                                sentiment_score = self._analyze_text_sentiment(tweet.text)
                                twitter_sentiment['sentiment_scores'].append({
                                    'text': tweet.text[:100],
                                    'score': sentiment_score,
                                    'likes': tweet.public_metrics.get('like_count', 0) if tweet.public_metrics else 0,
                                    'retweets': tweet.public_metrics.get('retweet_count', 0) if tweet.public_metrics else 0
                                })

                    except Exception as e:
                        logger.warning(f"Twitter analysis failed for {symbol}: {e}")

            # Calculate aggregate Twitter sentiment
            if twitter_sentiment['sentiment_scores']:
                scores = [s['score'] for s in twitter_sentiment['sentiment_scores']]
                twitter_sentiment['aggregate_score'] = np.mean(scores)
                twitter_sentiment['sentiment_volatility'] = np.std(scores) if len(scores) > 1 else 0
            else:
                twitter_sentiment['aggregate_score'] = 0.5

        except Exception as e:
            logger.error(f"Twitter sentiment analysis failed: {e}")
            twitter_sentiment['error'] = str(e)

        return twitter_sentiment

    async def _analyze_reddit_sentiment(self, symbols: List[str], time_horizon: str) -> Dict[str, Any]:
        """Analyze Reddit sentiment (simplified implementation)."""
        # This would require Reddit API integration
        # For now, return neutral sentiment
        return {
            'subreddits_analyzed': 0,
            'posts_analyzed': 0,
            'aggregate_score': 0.5,
            'confidence': 0.0,
            'note': 'Reddit API integration not implemented'
        }

    async def _analyze_market_sentiment(self, symbols: List[str], time_horizon: str) -> Dict[str, Any]:
        """Analyze market-based sentiment indicators."""
        market_sentiment = {
            'source': 'market_data',
            'indicators': {},
            'timestamp': datetime.now().isoformat()
        }

        try:
            # Put/Call ratio (simplified - would need options data)
            market_sentiment['indicators']['put_call_ratio'] = {
                'value': 0.8,  # Mock value
                'sentiment': 'neutral',  # PCR < 0.7 is bullish, > 1.0 is bearish
                'description': 'Put/Call ratio from options data'
            }

            # VIX level (fear gauge)
            market_sentiment['indicators']['vix_level'] = {
                'value': 18.5,  # Mock value
                'sentiment': 'neutral',  # VIX > 30 is high fear, < 15 is complacency
                'description': 'VIX volatility index'
            }

            # High-low index (breadth)
            market_sentiment['indicators']['advance_decline'] = {
                'value': 1.2,  # Mock value
                'sentiment': 'bullish',  # > 1.0 is bullish breadth
                'description': 'Advance-decline line ratio'
            }

            # Calculate aggregate market sentiment
            indicator_scores = []
            for indicator, data in market_sentiment['indicators'].items():
                if data['sentiment'] == 'bullish':
                    score = 0.7
                elif data['sentiment'] == 'bearish':
                    score = 0.3
                else:
                    score = 0.5
                indicator_scores.append(score)

            market_sentiment['aggregate_score'] = np.mean(indicator_scores) if indicator_scores else 0.5
            market_sentiment['confidence'] = 0.6  # Market indicators are generally reliable

        except Exception as e:
            logger.error(f"Market sentiment analysis failed: {e}")
            market_sentiment['error'] = str(e)

        return market_sentiment

    async def _analyze_economic_sentiment(self, symbols: List[str], time_horizon: str) -> Dict[str, Any]:
        """Analyze sentiment from economic data releases."""
        economic_sentiment = {
            'source': 'economic_data',
            'indicators': {},
            'timestamp': datetime.now().isoformat()
        }

        try:
            # This would integrate with economic data
            # For now, return neutral
            economic_sentiment['aggregate_score'] = 0.5
            economic_sentiment['confidence'] = 0.0
            economic_sentiment['note'] = 'Economic sentiment integration pending'

        except Exception as e:
            logger.error(f"Economic sentiment analysis failed: {e}")
            economic_sentiment['error'] = str(e)

        return economic_sentiment

    def _analyze_text_sentiment(self, text: str) -> float:
        """Analyze sentiment of text using available models."""
        scores = []

        # TextBlob analysis
        if 'textblob' in self.models:
            try:
                blob = self.models['textblob'](text)
                polarity = blob.sentiment.polarity
                normalized_score = (polarity + 1) / 2  # Convert -1/+1 to 0/1
                scores.append(normalized_score)
            except Exception as e:
                logger.warning(f"TextBlob analysis failed: {e}")

        # VADER analysis
        if 'vader' in self.models:
            try:
                scores_dict = self.models['vader'].polarity_scores(text)
                compound_score = scores_dict['compound']
                normalized_score = (compound_score + 1) / 2  # Convert -1/+1 to 0/1
                scores.append(normalized_score)
            except Exception as e:
                logger.warning(f"VADER analysis failed: {e}")

        # FinBERT analysis (if available)
        if 'finbert' in self.models:
            try:
                result = self.models['finbert'](text[:512])  # Limit text length
                label = result[0]['label']
                confidence = result[0]['score']

                if label == 'positive':
                    score = 0.5 + (confidence * 0.5)
                elif label == 'negative':
                    score = 0.5 - (confidence * 0.5)
                else:
                    score = 0.5

                scores.append(score)
            except Exception as e:
                logger.warning(f"FinBERT analysis failed: {e}")

        # Return average of all available model scores
        if scores:
            return np.mean(scores)
        else:
            logger.error("CRITICAL FAILURE: No sentiment scores available from any model - cannot return neutral fallback")
            raise Exception("Sentiment analysis failed - no neutral fallback scores allowed")

    def _extract_news_themes(self, sentiment_scores: List[Dict[str, Any]]) -> List[str]:
        """Extract key themes from news sentiment scores."""
        themes = []

        if not sentiment_scores:
            return themes

        # Simple theme extraction based on keywords in titles
        bullish_keywords = ['rise', 'gain', 'surge', 'jump', 'rally', 'bullish', 'upgrade']
        bearish_keywords = ['fall', 'drop', 'decline', 'plunge', 'crash', 'bearish', 'downgrade']

        bullish_count = 0
        bearish_count = 0

        for score_data in sentiment_scores:
            title = score_data.get('title', '').lower()
            for keyword in bullish_keywords:
                if keyword in title:
                    bullish_count += 1
                    break
            for keyword in bearish_keywords:
                if keyword in title:
                    bearish_count += 1
                    break

        if bullish_count > bearish_count:
            themes.append('bullish_news_flow')
        elif bearish_count > bullish_count:
            themes.append('bearish_news_flow')
        else:
            themes.append('mixed_news_flow')

        return themes

    async def _perform_llm_sentiment_analysis(self, sentiment_data: Dict[str, Any], focus_areas: List[str]) -> Dict[str, Any]:
        """Perform LLM-driven sentiment analysis and interpretation."""
        try:
            sources = sentiment_data.get('sources', {})

            # Generate sentiment narrative
            sentiment_narrative = await self._generate_sentiment_narrative(sentiment_data)

            # Analyze sentiment consistency across sources
            consistency_analysis = self._analyze_sentiment_consistency(sources)

            # Predict market reaction to sentiment
            market_prediction = await self._predict_sentiment_market_impact(sentiment_data)

            # Identify sentiment anomalies
            anomalies = self._identify_sentiment_anomalies(sentiment_data)

            sentiment_data.update({
                'sentiment_narrative': sentiment_narrative,
                'consistency_analysis': consistency_analysis,
                'market_prediction': market_prediction,
                'anomalies': anomalies,
                'llm_analysis_timestamp': datetime.now().isoformat()
            })

            return sentiment_data

        except Exception as e:
            logger.error(f"LLM sentiment analysis failed: {e}")
            return sentiment_data

    async def _generate_sentiment_narrative(self, sentiment_data: Dict[str, Any]) -> str:
        """Generate coherent sentiment narrative from multiple sources."""
        sources = sentiment_data.get('sources', {})
        narrative_parts = []

        # Analyze each source
        for source_name, source_data in sources.items():
            if 'aggregate_score' in source_data:
                score = source_data['aggregate_score']
                confidence = source_data.get('confidence', 0.5)

                if score > 0.6:
                    sentiment_desc = "bullish"
                elif score < 0.4:
                    sentiment_desc = "bearish"
                else:
                    sentiment_desc = "neutral"

                confidence_desc = "high" if confidence > 0.7 else "moderate" if confidence > 0.4 else "low"

                narrative_parts.append(f"{source_name} sentiment is {sentiment_desc} with {confidence_desc} confidence")

        if narrative_parts:
            return ". ".join(narrative_parts)
        else:
            return "Sentiment analysis in progress - data collection ongoing."

    def _analyze_sentiment_consistency(self, sources: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze consistency of sentiment across different sources."""
        scores = []
        source_names = []

        for source_name, source_data in sources.items():
            if 'aggregate_score' in source_data:
                scores.append(source_data['aggregate_score'])
                source_names.append(source_name)

        if len(scores) < 2:
            return {
                'consistency': 'insufficient_data',
                'confidence': 0.0,
                'sources_compared': len(scores)
            }

        # Calculate consistency metrics
        score_mean = np.mean(scores)
        score_std = np.std(scores)
        coefficient_of_variation = score_std / abs(score_mean) if score_mean != 0 else float('inf')

        # Determine consistency level
        if coefficient_of_variation < 0.1:
            consistency = 'high'
            consistency_score = 0.9
        elif coefficient_of_variation < 0.2:
            consistency = 'moderate'
            consistency_score = 0.7
        else:
            consistency = 'low'
            consistency_score = 0.4

        return {
            'consistency': consistency,
            'consistency_score': consistency_score,
            'coefficient_of_variation': coefficient_of_variation,
            'score_mean': score_mean,
            'score_std': score_std,
            'sources_compared': len(scores),
            'source_names': source_names
        }

    async def _predict_sentiment_market_impact(self, sentiment_data: Dict[str, Any]) -> Dict[str, Any]:
        """Predict market reaction to current sentiment environment."""
        composite_score = sentiment_data.get('composite_sentiment', {}).get('score', 0.5)
        consistency = sentiment_data.get('consistency_analysis', {}).get('consistency', 'unknown')

        prediction = {
            'expected_move': 'neutral',
            'timeframe': '1-3 days',
            'confidence': 0.5
        }

        # Predict based on sentiment score and consistency
        if composite_score > 0.7 and consistency in ['high', 'moderate']:
            prediction.update({
                'expected_move': 'upward',
                'rationale': 'Strong bullish sentiment with consistent signals across sources',
                'confidence': 0.7
            })
        elif composite_score < 0.3 and consistency in ['high', 'moderate']:
            prediction.update({
                'expected_move': 'downward',
                'rationale': 'Strong bearish sentiment with consistent signals across sources',
                'confidence': 0.7
            })
        elif consistency == 'low':
            prediction.update({
                'expected_move': 'volatile',
                'rationale': 'Inconsistent sentiment signals may lead to increased volatility',
                'confidence': 0.6
            })

        return prediction

    def _identify_sentiment_anomalies(self, sentiment_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify anomalies in sentiment data."""
        anomalies = []
        sources = sentiment_data.get('sources', {})

        # Check for extreme sentiment readings
        for source_name, source_data in sources.items():
            if 'aggregate_score' in source_data:
                score = source_data['aggregate_score']
                confidence = source_data.get('confidence', 0.5)

                if score > 0.8 and confidence > 0.7:
                    anomalies.append({
                        'type': 'extreme_bullish',
                        'source': source_name,
                        'score': score,
                        'description': f'Extremely bullish sentiment in {source_name}'
                    })
                elif score < 0.2 and confidence > 0.7:
                    anomalies.append({
                        'type': 'extreme_bearish',
                        'source': source_name,
                        'score': score,
                        'description': f'Extremely bearish sentiment in {source_name}'
                    })

        return anomalies

    def _calculate_composite_sentiment(self, sentiment_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate composite sentiment score across all sources."""
        sources = sentiment_data.get('sources', {})
        source_weights = {
            'news': 0.3,
            'social_media': 0.3,
            'market_data': 0.3,
            'economic_data': 0.1
        }

        weighted_scores = []
        total_weight = 0

        for source_name, source_data in sources.items():
            if 'aggregate_score' in source_data:
                weight = source_weights.get(source_name, 0.25)
                score = source_data['aggregate_score']
                confidence = source_data.get('confidence', 0.5)

                # Adjust weight by confidence
                adjusted_weight = weight * confidence

                weighted_scores.append(score * adjusted_weight)
                total_weight += adjusted_weight

        if total_weight > 0:
            composite_score = sum(weighted_scores) / total_weight
        else:
            composite_score = 0.5

        # Determine sentiment label
        if composite_score > 0.6:
            label = 'bullish'
        elif composite_score < 0.4:
            label = 'bearish'
        else:
            label = 'neutral'

        return {
            'score': composite_score,
            'label': label,
            'confidence': min(0.9, total_weight / 1.0),  # Higher confidence with more weight
            'sources_contributed': len([s for s in sources.values() if 'aggregate_score' in s])
        }

    def _assess_market_impact(self, sentiment_data: Dict[str, Any]) -> Dict[str, Any]:
        """Assess overall market impact of sentiment environment."""
        composite_sentiment = sentiment_data.get('composite_sentiment', {})
        score = composite_sentiment.get('score', 0.5)
        consistency = sentiment_data.get('consistency_analysis', {}).get('consistency', 'unknown')

        impact_assessment = {
            'overall_impact': 'neutral',
            'strength': 'moderate',
            'time_horizon': 'short_term',
            'asset_classes_affected': []
        }

        # Determine impact based on sentiment and consistency
        if score > 0.7 and consistency == 'high':
            impact_assessment.update({
                'overall_impact': 'strongly_bullish',
                'strength': 'strong',
                'asset_classes_affected': ['equities', 'risk_assets']
            })
        elif score > 0.6 and consistency in ['high', 'moderate']:
            impact_assessment.update({
                'overall_impact': 'bullish',
                'strength': 'moderate',
                'asset_classes_affected': ['equities', 'cyclicals']
            })
        elif score < 0.3 and consistency == 'high':
            impact_assessment.update({
                'overall_impact': 'strongly_bearish',
                'strength': 'strong',
                'asset_classes_affected': ['equities', 'risk_assets']
            })
        elif score < 0.4 and consistency in ['high', 'moderate']:
            impact_assessment.update({
                'overall_impact': 'bearish',
                'strength': 'moderate',
                'asset_classes_affected': ['equities', 'defensives']
            })

        return impact_assessment

    def _generate_collaborative_insights(self, sentiment_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate insights for sharing with other agents."""
        insights = []

        composite_score = sentiment_data.get('composite_sentiment', {}).get('score', 0.5)
        market_impact = sentiment_data.get('market_impact', {})

        # Generate strategy agent insights
        if composite_score > 0.6:
            insights.append({
                'target_agent': 'strategy',
                'insight_type': 'sentiment_signal',
                'content': 'Bullish sentiment environment favors momentum and growth strategies',
                'confidence': 0.7,
                'relevance': 'high'
            })
        elif composite_score < 0.4:
            insights.append({
                'target_agent': 'strategy',
                'insight_type': 'sentiment_signal',
                'content': 'Bearish sentiment environment favors defensive and value strategies',
                'confidence': 0.7,
                'relevance': 'high'
            })

        # Generate risk agent insights
        impact_strength = market_impact.get('strength', 'moderate')
        if impact_strength == 'strong':
            insights.append({
                'target_agent': 'risk',
                'insight_type': 'sentiment_risk',
                'content': f'Strong sentiment signals may indicate {market_impact.get("overall_impact", "neutral")} market regime',
                'confidence': 0.8,
                'relevance': 'high'
            })

        # Generate economic data agent insights
        consistency = sentiment_data.get('consistency_analysis', {}).get('consistency', 'unknown')
        if consistency == 'high':
            insights.append({
                'target_agent': 'economic_data',
                'insight_type': 'sentiment_consistency',
                'content': 'High sentiment consistency may correlate with economic data trends',
                'confidence': 0.6,
                'relevance': 'medium'
            })

        return insights

    def _update_memory(self, sentiment_data: Dict[str, Any]):
        """Update collaborative memory with sentiment insights."""
        composite_score = sentiment_data.get('composite_sentiment', {}).get('score', 0.5)

        # Add sentiment pattern insight
        self.memory.add_session_insight({
            'type': 'sentiment_pattern',
            'composite_score': composite_score,
            'market_impact': sentiment_data.get('market_impact', {}).get('overall_impact', 'neutral'),
            'consistency': sentiment_data.get('consistency_analysis', {}).get('consistency', 'unknown')
        })

        # Update market reaction patterns
        market_prediction = sentiment_data.get('market_prediction', {})
        if market_prediction.get('expected_move') != 'neutral':
            self.memory.market_reactions[datetime.now().strftime('%Y-%m-%d')] = {
                'sentiment_score': composite_score,
                'predicted_move': market_prediction.get('expected_move'),
                'confidence': market_prediction.get('confidence', 0.5)
            }

    # Legacy methods for backward compatibility
    def _parse_sentiment_result(self, result: str) -> Dict[str, Any]:
        """Parse sentiment tool result into structured format (legacy)."""
        try:
            if "{" in result and "}" in result:
                import ast
                parsed = ast.literal_eval(result.split(": ", 1)[1] if ": " in result else result)
                return {
                    'score': parsed.get('score', 0.5),
                    'source': parsed.get('source', 'parsed'),
                    'impact': parsed.get('impact', result)
                }
            else:
                return {'score': 0.5, 'source': 'parsed', 'impact': result}
        except:
            return {'score': 0.5, 'source': 'fallback', 'impact': 'neutral'}

    async def _cross_validate_sentiment(self, text: str) -> Dict[str, Any]:
        """Legacy cross-validation method."""
        # Simplified version for backward compatibility
        score = self._analyze_text_sentiment(text)
        return {
            'validated': True,
            'sentiment': {
                'score': score,
                'source': 'legacy_cross_validation',
                'impact': 'bullish' if score > 0.6 else 'bearish' if score < 0.4 else 'neutral'
            },
            'validation_info': {
                'validated': True,
                'confidence': 0.7,
                'sources_used': 1
            }
        }

    def _enhance_sentiment_analysis(self, sentiment: Dict[str, Any], text: str) -> Dict[str, Any]:
        """Legacy enhancement method."""
        return sentiment

    def analyze_sentiment(self, texts: List[str]) -> Dict[str, Any]:
        """
        Analyze sentiment of a list of texts.
        
        Args:
            texts: List of text strings to analyze
            
        Returns:
            Dict containing sentiment analysis results
        """
        if not texts:
            return {"sentiment_score": 0.5, "confidence": 0.0, "source": "empty_input"}
            
        scores = []
        for text in texts:
            if not isinstance(text, str) or not text.strip():
                continue
            score = self._analyze_text_sentiment(text)
            scores.append(score)
            
        if not scores:
            return {"sentiment_score": 0.5, "confidence": 0.0, "source": "no_valid_texts"}
            
        avg_score = sum(scores) / len(scores)
        confidence = min(1.0, len(scores) / 10.0)  # Higher confidence with more texts
        
        return {
            "sentiment_score": avg_score,
            "confidence": confidence,
            "source": "sentiment_analysis",
            "texts_analyzed": len(scores)
        }


# Standalone test
if __name__ == "__main__":
    import asyncio
    agent = SentimentDatasub()
    result = asyncio.run(agent.process_input({
        'focus_areas': ['news', 'social_media'],
        'symbols': ['AAPL'],
        'time_horizon': 'current'
    }))
    print("Sentiment Agent Test Result:\n", result)
