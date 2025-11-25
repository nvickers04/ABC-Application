import time
import random
import json
from locust import HttpUser, task, between, constant, constant_pacing
from locust.exception import StopUser
import asyncio
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.agents.data import DataAgent
from src.agents.strategy import StrategyAgent
from src.agents.risk import RiskAgent
from src.agents.execution import ExecutionAgent

class TradingSystemUser(HttpUser):
    """
    Locust user class for load testing the ABC trading system.
    Simulates realistic 24/6 trading workloads with concurrent operations.
    """

    # Simulate realistic trading hours (24/6 operation)
    wait_time = constant_pacing(1)  # 1 request per second per user

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data_agent = DataAgent()
        self.strategy_agent = StrategyAgent()
        self.risk_agent = RiskAgent()
        self.execution_agent = ExecutionAgent()
        self.session_id = f"session_{random.randint(1000, 9999)}"

    @task(30)  # 30% of requests - data collection (most frequent)
    def data_collection_task(self):
        """Simulate data collection requests"""
        symbols = random.choice([
            ['AAPL'],  # Single stock
            ['AAPL', 'MSFT', 'GOOGL'],  # Tech portfolio
            ['SPY', 'QQQ', 'IWM'],  # ETF portfolio
            ['TSLA', 'NVDA', 'AMD', 'INTC'],  # High volatility
            ['JPM', 'BAC', 'WFC', 'C']  # Banking sector
        ])

        periods = random.choice(['1d', '5d', '1mo', '3mo'])

        request_data = {
            'symbols': symbols,
            'period': periods,
            'session_id': self.session_id
        }

        start_time = time.time()

        try:
            # Simulate async call (in real scenario, this would be an API call)
            result = asyncio.run(self.data_agent.process_input(request_data))
            response_time = (time.time() - start_time) * 1000  # Convert to ms

            if result and 'symbols_processed' in result:
                self.environment.events.request.fire(
                    request_type="DATA",
                    name="data_collection",
                    response_time=response_time,
                    response_length=len(str(result)),
                    exception=None
                )
            else:
                raise Exception("Invalid data response")

        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            self.environment.events.request.fire(
                request_type="DATA",
                name="data_collection",
                response_time=response_time,
                response_length=0,
                exception=e
            )

    @task(20)  # 20% of requests - strategy analysis
    def strategy_analysis_task(self):
        """Simulate strategy analysis requests"""
        import pandas as pd
        import numpy as np

        # Generate realistic market data
        dates = pd.date_range('2024-01-01', periods=random.randint(50, 200), freq='D')
        prices = np.random.normal(100, 10, len(dates)).cumsum() + 100  # Random walk

        df = pd.DataFrame({
            'Close': prices,
            'High': prices * np.random.uniform(1.01, 1.05, len(dates)),
            'Low': prices * np.random.uniform(0.95, 0.99, len(dates)),
            'Open': prices * np.random.uniform(0.98, 1.02, len(dates))
        })

        sentiments = ['bullish', 'bearish', 'neutral']
        sentiment = random.choice(sentiments)
        confidence = random.uniform(0.5, 0.95)

        request_data = {
            'dataframe': df,
            'sentiment': {'sentiment': sentiment, 'confidence': confidence},
            'symbols': ['SPY'],
            'session_id': self.session_id
        }

        start_time = time.time()

        try:
            result = asyncio.run(self.strategy_agent.process_input(request_data))
            response_time = (time.time() - start_time) * 1000

            if result and 'strategy_type' in result:
                self.environment.events.request.fire(
                    request_type="STRATEGY",
                    name="strategy_analysis",
                    response_time=response_time,
                    response_length=len(str(result)),
                    exception=None
                )
            else:
                raise Exception("Invalid strategy response")

        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            self.environment.events.request.fire(
                request_type="STRATEGY",
                name="strategy_analysis",
                response_time=response_time,
                response_length=0,
                exception=e
            )

    @task(15)  # 15% of requests - risk analysis
    def risk_analysis_task(self):
        """Simulate risk analysis requests"""
        # Generate realistic portfolio returns
        num_returns = random.randint(100, 500)
        returns = np.random.normal(0.001, 0.02, num_returns)  # Daily returns

        request_data = {
            'portfolio_returns': returns.tolist(),
            'portfolio_value': random.uniform(50000, 500000),
            'symbols': ['AAPL', 'MSFT', 'GOOGL'],
            'analysis_type': random.choice(['basic', 'comprehensive', 'stress_test']),
            'session_id': self.session_id
        }

        start_time = time.time()

        try:
            result = asyncio.run(self.risk_agent.process_input(request_data))
            response_time = (time.time() - start_time) * 1000

            if result and 'risk_score' in result:
                self.environment.events.request.fire(
                    request_type="RISK",
                    name="risk_analysis",
                    response_time=response_time,
                    response_length=len(str(result)),
                    exception=None
                )
            else:
                raise Exception("Invalid risk response")

        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            self.environment.events.request.fire(
                request_type="RISK",
                name="risk_analysis",
                response_time=response_time,
                response_length=0,
                exception=e
            )

    @task(10)  # 10% of requests - trade execution
    def trade_execution_task(self):
        """Simulate trade execution requests"""
        actions = ['buy', 'sell']
        action = random.choice(actions)

        symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA', 'SPY', 'QQQ']
        symbol = random.choice(symbols)

        # Realistic order sizes
        if symbol in ['SPY', 'QQQ']:
            quantity = random.randint(10, 100)  # ETF shares
        else:
            quantity = random.randint(1, 50)   # Stock shares

        request_data = {
            'action': action,
            'symbol': symbol,
            'quantity': quantity,
            'order_type': random.choice(['market', 'limit', 'stop']),
            'session_id': self.session_id
        }

        start_time = time.time()

        try:
            result = asyncio.run(self.execution_agent.process_input(request_data))
            response_time = (time.time() - start_time) * 1000

            if result and 'order_status' in result:
                self.environment.events.request.fire(
                    request_type="EXECUTION",
                    name="trade_execution",
                    response_time=response_time,
                    response_length=len(str(result)),
                    exception=None
                )
            else:
                raise Exception("Invalid execution response")

        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            self.environment.events.request.fire(
                request_type="EXECUTION",
                name="trade_execution",
                response_time=response_time,
                response_length=0,
                exception=e
            )

    @task(5)  # 5% of requests - concurrent multi-agent workflow
    def multi_agent_workflow_task(self):
        """Simulate complex multi-agent workflow under load"""
        start_time = time.time()

        try:
            # Simulate a complete trading workflow
            symbols = ['AAPL', 'MSFT']

            # Step 1: Data collection
            data_result = asyncio.run(self.data_agent.process_input({
                'symbols': symbols,
                'period': '1mo',
                'session_id': self.session_id
            }))

            # Step 2: Strategy analysis (simplified)
            strategy_result = asyncio.run(self.strategy_agent.process_input({
                'dataframe': None,  # Would normally use data_result
                'sentiment': {'sentiment': 'neutral', 'confidence': 0.7},
                'symbols': symbols,
                'session_id': self.session_id
            }))

            # Step 3: Risk assessment
            risk_result = asyncio.run(self.risk_agent.process_input({
                'portfolio_returns': [0.01, 0.005, -0.002, 0.008],
                'portfolio_value': 100000,
                'symbols': symbols,
                'analysis_type': 'basic',
                'session_id': self.session_id
            }))

            # Step 4: Trade execution (if conditions met)
            if strategy_result.get('validation_confidence', 0) > 0.6:
                execution_result = asyncio.run(self.execution_agent.process_input({
                    'action': 'buy',
                    'symbol': random.choice(symbols),
                    'quantity': 10,
                    'session_id': self.session_id
                }))
            else:
                execution_result = {"order_status": "not_executed"}

            response_time = (time.time() - start_time) * 1000

            self.environment.events.request.fire(
                request_type="WORKFLOW",
                name="multi_agent_workflow",
                response_time=response_time,
                response_length=len(str({
                    'data': data_result,
                    'strategy': strategy_result,
                    'risk': risk_result,
                    'execution': execution_result
                })),
                exception=None
            )

        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            self.environment.events.request.fire(
                request_type="WORKFLOW",
                name="multi_agent_workflow",
                response_time=response_time,
                response_length=0,
                exception=e
            )

    @task(20)  # 20% of requests - API health checks and monitoring
    def health_check_task(self):
        """Simulate system health monitoring requests"""
        check_types = ['agent_status', 'memory_usage', 'api_connectivity', 'database_health']

        request_data = {
            'check_type': random.choice(check_types),
            'session_id': self.session_id
        }

        start_time = time.time()

        try:
            # Simulate health check (in real system, this would call monitoring endpoints)
            time.sleep(random.uniform(0.01, 0.1))  # Simulate API call

            result = {
                'status': 'healthy',
                'response_time': random.uniform(10, 100),
                'check_type': request_data['check_type']
            }

            response_time = (time.time() - start_time) * 1000

            self.environment.events.request.fire(
                request_type="HEALTH",
                name="health_check",
                response_time=response_time,
                response_length=len(str(result)),
                exception=None
            )

        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            self.environment.events.request.fire(
                request_type="HEALTH",
                name="health_check",
                response_time=response_time,
                response_length=0,
                exception=e
            )

class HighFrequencyUser(TradingSystemUser):
    """User class simulating high-frequency trading patterns"""
    wait_time = constant_pacing(0.1)  # 10 requests per second

    @task
    def rapid_data_requests(self):
        """High-frequency data requests"""
        self.data_collection_task()

class BatchProcessingUser(TradingSystemUser):
    """User class simulating batch processing workloads"""
    wait_time = between(5, 15)  # Less frequent but heavier requests

    @task
    def batch_analysis_task(self):
        """Batch processing of multiple symbols"""
        symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA', 'JPM', 'BAC', 'WFC']
        random.shuffle(symbols)
        batch_size = random.randint(3, 8)

        request_data = {
            'symbols': symbols[:batch_size],
            'period': '3mo',
            'batch_size': batch_size,
            'session_id': self.session_id
        }

        start_time = time.time()

        try:
            result = asyncio.run(self.data_agent.process_input(request_data))
            response_time = (time.time() - start_time) * 1000

            self.environment.events.request.fire(
                request_type="BATCH",
                name="batch_processing",
                response_time=response_time,
                response_length=len(str(result)),
                exception=None
            )

        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            self.environment.events.request.fire(
                request_type="BATCH",
                name="batch_processing",
                response_time=response_time,
                response_length=0,
                exception=e
            )