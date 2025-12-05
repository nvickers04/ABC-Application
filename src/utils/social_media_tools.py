#!/usr/bin/env python3
"""
Social media data collection and analysis tools.
Provides tools for Twitter sentiment analysis and social media monitoring.
"""

import os
import json
import logging
from typing import Dict, Any, Optional
import requests
from langchain_core.tools import tool

from .validation import circuit_breaker, DataValidator
from .constants import DEFAULT_API_TIMEOUT, ERROR_API_KEY_NOT_FOUND, ERROR_NO_DATA_FOUND

logger = logging.getLogger(__name__)


@tool
def twitter_sentiment_tool(query: str, max_tweets: int = 100) -> Dict[str, Any]:
    """
    Analyze Twitter sentiment for a given query.
    Args:
        query: Search query for tweets
        max_tweets: Maximum number of tweets to analyze
    Returns:
        dict: Sentiment analysis results
    """
    bearer_token = os.getenv('TWITTER_BEARER_TOKEN')
    if not bearer_token:
        return {
            "error": ERROR_API_KEY_NOT_FOUND,
            "step_1": "Go to https://developer.twitter.com/",
            "step_2": "Create a Twitter Developer account",
            "step_3": "Create a new app and get Bearer Token",
            "step_4": "Add TWITTER_BEARER_TOKEN=your_token_here to .env file"
        }

    try:
        query = DataValidator.sanitize_text_input(query)
        if not query:
            return {"error": "No valid query provided"}

        # Twitter API v2 search endpoint
        url = "https://api.twitter.com/2/tweets/search/recent"
        headers = {"Authorization": f"Bearer {bearer_token}"}

        params = {
            "query": query,
            "max_results": min(max_tweets, 100),  # Twitter API limit
            "tweet.fields": "created_at,public_metrics,text,author_id"
        }

        response = requests.get(url, headers=headers, params=params, timeout=DEFAULT_API_TIMEOUT)
        response.raise_for_status()

        data = response.json()

        if not data.get("data"):
            return {
                "query": query,
                "tweets_found": 0,
                "sentiment_score": 0.0,
                "confidence": 0.0,
                "message": "No tweets found for the given query"
            }

        tweets = data["data"]

        # Analyze sentiment of tweets
        sentiments = []
        total_likes = 0
        total_retweets = 0

        for tweet in tweets:
            text = tweet.get("text", "")
            metrics = tweet.get("public_metrics", {})

            # Simple sentiment analysis (could be enhanced with ML models)
            try:
                from textblob import TextBlob
                blob = TextBlob(text)
                sentiment = blob.sentiment.polarity
                sentiments.append(sentiment)
            except Exception:
                # Fallback: basic keyword analysis
                positive_words = ['good', 'great', 'excellent', 'amazing', 'love', 'like', 'best', 'awesome']
                negative_words = ['bad', 'terrible', 'awful', 'hate', 'worst', 'suck', 'disappointing']

                text_lower = text.lower()
                positive_count = sum(1 for word in positive_words if word in text_lower)
                negative_count = sum(1 for word in negative_words if word in text_lower)

                if positive_count > negative_count:
                    sentiment = 0.5
                elif negative_count > positive_count:
                    sentiment = -0.5
                else:
                    sentiment = 0.0

                sentiments.append(sentiment)

            total_likes += metrics.get("like_count", 0)
            total_retweets += metrics.get("retweet_count", 0)

        # Calculate aggregate sentiment
        if sentiments:
            avg_sentiment = sum(sentiments) / len(sentiments)

            # Calculate engagement-weighted sentiment
            engagement_scores = []
            for i, tweet in enumerate(tweets):
                metrics = tweet.get("public_metrics", {})
                engagement = metrics.get("like_count", 0) + metrics.get("retweet_count", 0) * 2
                engagement_scores.append((sentiments[i], engagement))

            if engagement_scores:
                total_engagement = sum(engagement for _, engagement in engagement_scores)
                if total_engagement > 0:
                    weighted_sentiment = sum(sent * eng for sent, eng in engagement_scores) / total_engagement
                else:
                    weighted_sentiment = avg_sentiment
            else:
                weighted_sentiment = avg_sentiment

            # Determine confidence based on sample size and variance
            import statistics
            try:
                variance = statistics.variance(sentiments) if len(sentiments) > 1 else 0
                confidence = min(0.9, len(sentiments) / 100) * (1 - variance)  # Higher confidence with more tweets and less variance
            except:
                confidence = min(0.8, len(sentiments) / 50)

            return {
                "query": query,
                "tweets_analyzed": len(tweets),
                "average_sentiment": avg_sentiment,
                "weighted_sentiment": weighted_sentiment,
                "confidence": confidence,
                "sentiment_distribution": {
                    "positive": sum(1 for s in sentiments if s > 0.1),
                    "negative": sum(1 for s in sentiments if s < -0.1),
                    "neutral": sum(1 for s in sentiments if -0.1 <= s <= 0.1)
                },
                "total_engagement": {
                    "likes": total_likes,
                    "retweets": total_retweets
                },
                "source": "twitter_api",
                "time_range": "recent_7_days"
            }

        return {"error": "Failed to analyze tweet sentiments"}

    except (requests.RequestException, json.JSONDecodeError, ValueError, KeyError) as e:
        return {"error": f"Twitter API failed: {str(e)}"}


def social_media_monitor_tool(queries: list, platforms: list = None) -> Dict[str, Any]:
    """
    Monitor social media sentiment across multiple platforms.
    Args:
        queries: List of search queries
        platforms: List of platforms to monitor
    Returns:
        dict: Aggregated social media sentiment
    """
    if platforms is None:
        platforms = ["twitter"]

    results = {}

    for platform in platforms:
        if platform.lower() == "twitter":
            platform_results = {}
            for query in queries:
                result = twitter_sentiment_tool(query, max_tweets=50)
                if "error" not in result:
                    platform_results[query] = result
                else:
                    platform_results[query] = {"error": result["error"]}

            results["twitter"] = platform_results

    # Aggregate results
    if results:
        total_sentiment = 0
        total_confidence = 0
        count = 0

        for platform, queries_data in results.items():
            for query, data in queries_data.items():
                if "weighted_sentiment" in data:
                    total_sentiment += data["weighted_sentiment"]
                    total_confidence += data.get("confidence", 0)
                    count += 1

        if count > 0:
            return {
                "overall_sentiment": total_sentiment / count,
                "average_confidence": total_confidence / count,
                "platforms_monitored": list(results.keys()),
                "queries_analyzed": queries,
                "detailed_results": results,
                "source": "social_media_monitor"
            }

    return {
        "error": ERROR_NO_DATA_FOUND,
        "platforms_attempted": platforms,
        "queries": queries
    }


def reddit_sentiment_tool(subreddit: str, query: str = "", limit: int = 25) -> Dict[str, Any]:
    """
    Analyze sentiment from Reddit posts.
    Note: Requires Reddit API credentials.
    Args:
        subreddit: Subreddit name
        query: Search query
        limit: Maximum posts to analyze
    Returns:
        dict: Reddit sentiment analysis
    """
    # Reddit API requires OAuth2 setup which is complex
    # This is a placeholder for future implementation

    return {
        "error": "Reddit API integration not yet implemented",
        "note": "Reddit API requires OAuth2 setup and API credentials",
        "subreddit": subreddit,
        "query": query
    }


def news_sentiment_aggregation_tool(news_data: Dict[str, Any], social_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Aggregate sentiment from news and social media sources.
    Args:
        news_data: News sentiment data
        social_data: Social media sentiment data
    Returns:
        dict: Aggregated sentiment analysis
    """
    try:
        # Extract sentiment scores
        news_sentiment = 0
        social_sentiment = 0

        if "articles" in news_data:
            # Simple news sentiment based on titles/descriptions
            articles = news_data["articles"]
            if articles:
                sentiments = []
                for article in articles[:10]:  # Analyze first 10 articles
                    title = article.get("title", "")
                    desc = article.get("description", "")

                    try:
                        from textblob import TextBlob
                        text = f"{title} {desc}"
                        blob = TextBlob(text)
                        sentiments.append(blob.sentiment.polarity)
                    except:
                        sentiments.append(0.0)

                news_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0

        if "overall_sentiment" in social_data:
            social_sentiment = social_data["overall_sentiment"]

        # Weighted aggregation (news might be more reliable than social)
        if news_sentiment != 0 and social_sentiment != 0:
            combined_sentiment = (news_sentiment * 0.6) + (social_sentiment * 0.4)
            confidence = 0.8
        elif news_sentiment != 0:
            combined_sentiment = news_sentiment
            confidence = 0.7
        elif social_sentiment != 0:
            combined_sentiment = social_sentiment
            confidence = 0.6
        else:
            combined_sentiment = 0.0
            confidence = 0.0

        return {
            "combined_sentiment": combined_sentiment,
            "news_sentiment": news_sentiment,
            "social_sentiment": social_sentiment,
            "confidence": confidence,
            "sources": ["news", "social_media"] if news_sentiment != 0 and social_sentiment != 0 else ["news"] if news_sentiment != 0 else ["social_media"],
            "source": "sentiment_aggregation"
        }

    except Exception as e:
        return {"error": f"Sentiment aggregation failed: {str(e)}"}