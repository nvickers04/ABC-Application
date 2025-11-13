#!/usr/bin/env python3
import sys
from pathlib import Path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from unittest.mock import patch, Mock

# Test importing with mocks
try:
    with patch('src.agents.data_subs.yfinance_datasub.YfinanceDatasub', return_value=Mock()):
        with patch('src.agents.data_subs.sentiment_datasub.SentimentDatasub', return_value=Mock()):
            with patch('src.agents.data_subs.news_datasub.NewsDatasub', return_value=Mock()):
                with patch('src.agents.data_subs.economic_datasub.EconomicDatasub', return_value=Mock()):
                    with patch('src.agents.data_subs.institutional_datasub.InstitutionalDatasub', return_value=Mock()):
                        with patch('src.agents.data_subs.fundamental_datasub.FundamentalDatasub', return_value=Mock()):
                            with patch('src.agents.data_subs.microstructure_datasub.MicrostructureDatasub', return_value=Mock()):
                                with patch('src.agents.data_subs.kalshi_datasub.KalshiDatasub', return_value=Mock()):
                                    with patch('src.agents.data_subs.options_datasub.OptionsDatasub', return_value=Mock()):
                                        from src.agents.data import DataAgent
                                        print("DataAgent imported successfully with mocks")
except Exception as e:
    print(f"Failed to import DataAgent: {e}")