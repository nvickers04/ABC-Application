#!/usr/bin/env python3
"""
News data collection and analysis tools.
Provides tools for fetching news from various APIs with fallbacks.
"""

import os
import json
import logging
from typing import Dict, Any, Optional, List
import requests
import pandas as pd

from .validation import circuit_breaker, DataValidator
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class NewsDataInput(BaseModel):
    query: str = Field(default="stock market", description="Search query for news")
    language: str = Field(default="en", description="Language code (e.g., 'en')")
    page_size: int = Field(default=10, description="Number of articles to fetch")


class NewsDataTool(BaseTool):
    name: str = "news_data_tool"
    description: str = "Fetch news data from NewsAPI with comprehensive fallbacks."
    args_schema: type = NewsDataInput

    def _run(self, query: str = "stock market", language: str = "en", page_size: int = 10) -> Dict[str, Any]:
        """
        Fetch news data from NewsAPI with comprehensive fallbacks.
        Args:
            query: Search query for news
            language: Language code (e.g., 'en')
            page_size: Number of articles to fetch
        Returns:
            dict: News data with articles and metadata
        """
        query = DataValidator.sanitize_text_input(query)

        # Primary: NewsAPI
        news_api_key = os.getenv('NEWS_API_KEY')
        if news_api_key:
            try:
                url = "https://newsapi.org/v2/everything"
                params = {
                    "q": query,
                    "language": language,
                    "pageSize": min(page_size, 100),  # NewsAPI limit
                    "sortBy": "publishedAt",
                    "apiKey": news_api_key
                }

                response = requests.get(url, params=params, timeout=30)
                response.raise_for_status()

                data = response.json()

                if data.get("status") == "ok" and data.get("articles"):
                    articles = data["articles"]

                    # Validate and process articles
                    processed_articles = []
                    for article in articles[:page_size]:
                        if article.get("title") and article.get("description"):
                            processed_article = {
                                "title": DataValidator.sanitize_text_input(article.get("title", "")),
                                "description": DataValidator.sanitize_text_input(article.get("description", "")),
                                "url": article.get("url", ""),
                                "source": article.get("source", {}).get("name", "Unknown"),
                                "published_at": article.get("publishedAt", ""),
                                "author": article.get("author", "Unknown")
                            }
                            processed_articles.append(processed_article)

                    # Check for anomalies
                    anomalies = DataValidator.detect_data_anomalies(processed_articles)

                    return {
                        "status": "success",
                        "query": query,
                        "total_results": data.get("totalResults", 0),
                        "articles_returned": len(processed_articles),
                        "articles": processed_articles,
                        "source": "newsapi",
                        "free_tier_limit": "100 requests/day",
                        "anomalies": anomalies
                    }

            except (requests.RequestException, json.JSONDecodeError, ValueError, KeyError) as e:
                logger.warning(f"NewsAPI failed: {e}")

        # Backup 1: Currents API
        currents_key = os.getenv('CURRENTS_API_KEY')
        if currents_key:
            try:
                url = "https://api.currentsapi.services/v1/search"
                params = {
                    "keywords": query,
                    "language": language,
                    "apiKey": currents_key
                }

                response = requests.get(url, params=params, timeout=30)
                response.raise_for_status()

                data = response.json()

                if data.get("news") and len(data["news"]) > 0:
                    articles = data["news"]

                    processed_articles = []
                    for article in articles[:page_size]:
                        processed_article = {
                            "title": DataValidator.sanitize_text_input(article.get("title", "")),
                            "description": DataValidator.sanitize_text_input(article.get("description", "")),
                            "url": article.get("url", ""),
                            "source": article.get("author", "Unknown"),
                            "published_at": article.get("published", ""),
                            "author": article.get("author", "Unknown")
                        }
                        processed_articles.append(processed_article)

                    anomalies = DataValidator.detect_data_anomalies(processed_articles)

                    return {
                        "status": "success",
                        "query": query,
                        "total_results": len(articles),
                        "articles_returned": len(processed_articles),
                        "articles": processed_articles,
                        "source": "currents",
                        "free_tier_limit": "600 requests/month",
                        "anomalies": anomalies
                    }

            except (requests.RequestException, json.JSONDecodeError, ValueError, KeyError) as e:
                logger.warning(f"Currents API failed: {e}")

        # Backup 2: NewsData.io
        newsdata_key = os.getenv('NEWSDATA_API_KEY')
        if newsdata_key:
            try:
                url = "https://newsdata.io/api/1/news"
                params = {
                    "q": query,
                    "language": language,
                    "size": min(page_size, 10),  # NewsData limit
                    "apikey": newsdata_key
                }

                response = requests.get(url, params=params, timeout=30)
                response.raise_for_status()

                data = response.json()

                if data.get("results") and len(data["results"]) > 0:
                    articles = data["results"]

                    processed_articles = []
                    for article in articles[:page_size]:
                        processed_article = {
                            "title": DataValidator.sanitize_text_input(article.get("title", "")),
                            "description": DataValidator.sanitize_text_input(article.get("description", "")),
                            "url": article.get("link", ""),
                            "source": article.get("source_id", "Unknown"),
                            "published_at": article.get("pubDate", ""),
                            "author": article.get("creator", ["Unknown"])[0] if article.get("creator") else "Unknown"
                        }
                        processed_articles.append(processed_article)

                    anomalies = DataValidator.detect_data_anomalies(processed_articles)

                    return {
                        "status": "success",
                        "query": query,
                        "total_results": data.get("totalResults", len(articles)),
                        "articles_returned": len(processed_articles),
                        "articles": processed_articles,
                        "source": "newsdata",
                        "free_tier_limit": "200 requests/month",
                        "anomalies": anomalies
                    }

            except (requests.RequestException, json.JSONDecodeError, ValueError, KeyError) as e:
                logger.warning(f"NewsData API failed: {e}")

        # Final fallback: Return error with setup instructions
        return {
            "status": "error",
            "error": "All news APIs failed or are not configured",
            "setup_instructions": {
                "newsapi": "Set NEWS_API_KEY environment variable (100 requests/day free)",
                "currents": "Set CURRENTS_API_KEY environment variable (600 requests/month free)",
                "newsdata": "Set NEWSDATA_API_KEY environment variable (200 requests/month free)"
            },
            "query": query
        }


class EconomicDataInput(BaseModel):
    indicators: str = Field(default="UNRATE,GDP,PCEPI,FEDFUNDS", description="Comma-separated FRED series IDs")


class EconomicDataTool(BaseTool):
    name: str = "economic_data_tool"
    description: str = "Fetch economic data from FRED API."
    args_schema: type = EconomicDataInput

    def __init__(self):
        super().__init__()

    def _run(self, indicators: str = "UNRATE,GDP,PCEPI,FEDFUNDS") -> Dict[str, Any]:
        """
        Fetch economic data from FRED API.
        Args:
            indicators: Comma-separated FRED series IDs
        Returns:
            dict: Economic data series
        """
        try:
            from fredapi import Fred

            fred_key = os.getenv('FRED_API_KEY')
            if not fred_key:
                return {"error": "FRED API key not found. Please set FRED_API_KEY in environment variables."}

            fred = Fred(api_key=fred_key)

            series_list = [s.strip() for s in indicators.split(',')]
            results = {}

            for series_id in series_list:
                try:
                    data = fred.get_series(series_id)
                    if data is not None and not data.empty:
                        # Convert to dict with dates as strings
                        data_dict = {str(date): float(value) for date, value in data.items() if pd.notna(value)}
                        results[series_id] = {
                            "data": data_dict,
                            "latest_value": data.iloc[-1] if not data.empty else None,
                            "latest_date": str(data.index[-1]) if not data.empty else None,
                            "count": len(data_dict)
                        }
                    else:
                        results[series_id] = {"error": f"No data found for series {series_id}"}

                except Exception as e:
                    results[series_id] = {"error": f"Failed to fetch {series_id}: {str(e)}"}

            return {
                "status": "success",
                "series": results,
                "source": "FRED",
                "note": "Federal Reserve Economic Data (FRED) API"
            }

        except Exception as e:
            return {"error": f"Economic data fetch failed: {str(e)}"}


economic_data_tool = EconomicDataTool()


class CurrentsNewsInput(BaseModel):
    query: str = Field(default="", description="Search query")
    language: str = Field(default="en", description="Language code")
    page_size: int = Field(default=10, description="Number of articles")


class CurrentsNewsTool(BaseTool):
    name: str = "currents_news_tool"
    description: str = "Fetch news from Currents API (standalone function for specific use)."
    args_schema: type = CurrentsNewsInput

    def __init__(self):
        super().__init__()

    def _run(self, query: str = "", language: str = "en", page_size: int = 10) -> Dict[str, Any]:
        """
        Fetch news from Currents API (standalone function for specific use).
        Args:
            query: Search query
            language: Language code
            page_size: Number of articles
        Returns:
            dict: News data
        """
        currents_key = os.getenv('CURRENTS_API_KEY')
        if not currents_key:
            return {
                "error": "Currents API key not found",
                "setup": "Add CURRENTS_API_KEY=your_key_here to .env file"
            }

        try:
            url = "https://api.currentsapi.services/v1/search"
            params = {
                "keywords": query or "latest news",
                "language": language,
                "apiKey": currents_key
            }

            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()

            data = response.json()

            if not data.get("news"):
                return {"error": "No news found", "query": query}

            articles = data["news"][:page_size]

            processed_articles = []
            for article in articles:
                processed_articles.append({
                    "title": DataValidator.sanitize_text_input(article.get("title", "")),
                    "description": DataValidator.sanitize_text_input(article.get("description", "")),
                    "url": article.get("url", ""),
                    "published_at": article.get("published", ""),
                    "author": article.get("author", "Unknown"),
                    "category": article.get("category", [])
                })

            return {
                "status": "success",
                "articles": processed_articles,
                "count": len(processed_articles),
                "source": "currents",
                "query": query
            }

        except (requests.RequestException, json.JSONDecodeError, ValueError, KeyError) as e:
            return {"error": f"Currents API failed: {str(e)}"}


news_data_tool = NewsDataTool()
economic_data_tool = EconomicDataTool()
currents_news_tool = CurrentsNewsTool()