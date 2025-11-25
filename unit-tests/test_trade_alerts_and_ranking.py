#!/usr/bin/env python3
"""
Unit tests for trade alerts and ranking functionality in LiveWorkflowOrchestrator.
"""

import unittest
from unittest.mock import AsyncMock, MagicMock, patch
import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.agents.live_workflow_orchestrator import LiveWorkflowOrchestrator


class TestTradeAlertsAndRanking(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures"""
        self.orchestrator = LiveWorkflowOrchestrator()
        # Mock Discord channels
        self.orchestrator.alerts_channel = AsyncMock()
        self.orchestrator.ranked_trades_channel = AsyncMock()
        self.orchestrator.channel = AsyncMock()  # General channel fallback

    def test_rank_trade_proposals_by_confidence(self):
        """Test ranking proposals by confidence descending"""
        proposals = [
            {'instrument': 'AAPL', 'action': 'BUY', 'confidence': 0.7, 'expected_return': 0.05},
            {'instrument': 'GOOG', 'action': 'SELL', 'confidence': 0.9, 'expected_return': 0.03},
            {'instrument': 'TSLA', 'action': 'HOLD', 'confidence': 0.8, 'expected_return': 0.04}
        ]

        ranked = self.orchestrator.rank_trade_proposals(proposals)

        # Should be sorted by confidence descending
        self.assertEqual(ranked[0]['instrument'], 'GOOG')  # 0.9
        self.assertEqual(ranked[1]['instrument'], 'TSLA')  # 0.8
        self.assertEqual(ranked[2]['instrument'], 'AAPL')  # 0.7

    def test_rank_trade_proposals_by_expected_return_tiebreaker(self):
        """Test ranking with expected return as tiebreaker"""
        proposals = [
            {'instrument': 'AAPL', 'action': 'BUY', 'confidence': 0.8, 'expected_return': 0.03},
            {'instrument': 'GOOG', 'action': 'SELL', 'confidence': 0.8, 'expected_return': 0.05}
        ]

        ranked = self.orchestrator.rank_trade_proposals(proposals)

        # Same confidence, higher expected return first
        self.assertEqual(ranked[0]['instrument'], 'GOOG')  # 0.05
        self.assertEqual(ranked[1]['instrument'], 'AAPL')  # 0.03

    def test_extract_trade_alert_info_structured(self):
        """Test extracting alerts from structured dict responses"""
        response_data = {
            'agent': 'strategy',
            'response': {
                'trade_proposals': [
                    {'instrument': 'AAPL', 'action': 'BUY', 'confidence': 0.9},
                    {'instrument': 'GOOG', 'action': 'SELL', 'confidence': 0.7}
                ]
            }
        }

        alert = self.orchestrator._extract_trade_alert_info(response_data)

        self.assertIsNotNone(alert)
        self.assertIn('**Strategy Agent** generated 2 ranked trade proposal(s)', alert)  # type: ignore
        self.assertIn('#1 BUY AAPL', alert)  # type: ignore
        self.assertIn('#2 SELL GOOG', alert)  # type: ignore

    def test_extract_trade_alert_info_string_parsing(self):
        """Test extracting alerts from string responses with regex"""
        response_data = {
            'agent': 'strategy',
            'response': 'I recommend BUY AAPL Confidence: 0.8 and SELL GOOG Confidence: 0.6'
        }

        alert = self.orchestrator._extract_trade_alert_info(response_data)

        self.assertIn('**Strategy Agent** has 2 trade proposal(s) in text', alert)  # type: ignore
        self.assertIn('BUY aapl', alert)  # type: ignore
        self.assertIn('SELL goog', alert)  # type: ignore

    @patch('asyncio.sleep', new_callable=AsyncMock)
    def test_send_trade_alert_success(self, mock_sleep):
        """Test successful trade alert send"""
        async def inner():
            await self.orchestrator.send_trade_alert("Test alert", "trade")

            self.orchestrator.alerts_channel.send.assert_called_once()  # type: ignore
            self.orchestrator.channel.send.assert_not_called()  # type: ignore

        asyncio.run(inner())

    @patch('asyncio.sleep', new_callable=AsyncMock)
    def test_send_trade_alert_retry_on_failure(self, mock_sleep):
        """Test trade alert retry on failure"""
        async def inner():
            self.orchestrator.alerts_channel.send.side_effect = [Exception("Network error"), None]  # type: ignore

            await self.orchestrator.send_trade_alert("Test alert", "trade")

            # Should retry once
            self.assertEqual(self.orchestrator.alerts_channel.send.call_count, 2)  # type: ignore
            self.orchestrator.channel.send.assert_not_called()  # type: ignore

        asyncio.run(inner())

    @patch('asyncio.sleep', new_callable=AsyncMock)
    def test_send_trade_alert_fallback_after_retries(self, mock_sleep):
        """Test fallback to general channel after all retries fail"""
        async def inner():
            self.orchestrator.alerts_channel.send.side_effect = Exception("Persistent error")  # type: ignore

            await self.orchestrator.send_trade_alert("Test alert", "trade")

            # Should retry 2 times then fallback
            self.assertEqual(self.orchestrator.alerts_channel.send.call_count, 3)  # type: ignore
            self.orchestrator.channel.send.assert_called_once()  # type: ignore

        asyncio.run(inner())

    @patch('asyncio.sleep', new_callable=AsyncMock)
    def test_send_ranked_trade_info_success(self, mock_sleep):
        """Test successful ranked trade info send"""
        async def inner():
            await self.orchestrator.send_ranked_trade_info("Test proposals", "proposal")

            self.orchestrator.ranked_trades_channel.send.assert_called_once()  # type: ignore
            self.orchestrator.channel.send.assert_not_called()  # type: ignore

        asyncio.run(inner())

    @patch('asyncio.sleep', new_callable=AsyncMock)
    def test_send_ranked_trade_info_fallback_no_channel(self, mock_sleep):
        """Test fallback when ranked trades channel not configured"""
        async def inner():
            self.orchestrator.ranked_trades_channel = None

            await self.orchestrator.send_ranked_trade_info("Test proposals", "proposal")

            self.orchestrator.channel.send.assert_called_once()  # type: ignore

        asyncio.run(inner())


if __name__ == '__main__':
    unittest.main()