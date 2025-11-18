#!/usr/bin/env python3
import sys
from pathlib import Path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from unittest.mock import patch, Mock

# Test importing with mocks
try:
    with patch('src.agents.data_analyzers.yfinance_data_analyzer.YfinanceDataAnalyzer', return_value=Mock()):
        with patch('src.agents.data_analyzers.sentiment_data_analyzer.SentimentDataAnalyzer', return_value=Mock()):
            with patch('src.agents.data_analyzers.news_data_analyzer.NewsDataAnalyzer', return_value=Mock()):
                with patch('src.agents.data_analyzers.economic_data_analyzer.EconomicDataAnalyzer', return_value=Mock()):
                    with patch('src.agents.data_analyzers.institutional_data_analyzer.InstitutionalDataAnalyzer', return_value=Mock()):
                        with patch('src.agents.data_analyzers.fundamental_data_analyzer.FundamentalDataAnalyzer', return_value=Mock()):
                            with patch('src.agents.data_analyzers.microstructure_data_analyzer.MicrostructureDataAnalyzer', return_value=Mock()):
                                with patch('src.agents.data_analyzers.kalshi_data_analyzer.KalshiDataAnalyzer', return_value=Mock()):
                                    with patch('src.agents.data_analyzers.options_data_analyzer.OptionsDataAnalyzer', return_value=Mock()):
                                        from src.agents.data import DataAgent
                                        print("DataAgent imported successfully with mocks")
except Exception as e:
    print(f"Failed to import DataAgent: {e}")