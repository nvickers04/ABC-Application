# Testing Framework

## Overview

The ABC Application system implements a comprehensive testing framework that covers unit testing, integration testing, performance testing, and backtesting validation. This ensures system reliability, performance, and correctness across all components and trading scenarios.

## Testing Architecture

### Test Categories

1. **Unit Tests** (`tests/unit/`)
   - Individual function/method testing
   - Mock external dependencies
   - Fast execution (< 100ms per test)

2. **Integration Tests** (`tests/integration/`)
   - Multi-component interaction testing
   - Real database and message queue connections
   - Medium execution time (100ms - 10s per test)

3. **System Tests** (`tests/system/`)
   - End-to-end workflow testing
   - Full system deployment simulation
   - Slow execution (10s - 5min per test)

4. **Performance Tests** (`tests/performance/`)
   - Load testing and benchmarking
   - Memory and CPU profiling
   - Stress testing under extreme conditions

5. **Backtesting Tests** (`tests/backtesting/`)
   - Historical strategy validation
   - Risk metric verification
   - Performance attribution testing

## Unit Testing Framework

### Agent Unit Tests

```python
# tests/unit/test_data_agent.py
import pytest
from unittest.mock import Mock, patch, MagicMock
from src.agents.data_agent import DataAgent
from src.data.market_data import MarketDataProvider

class TestDataAgent:
    @pytest.fixture
    def mock_data_provider(self):
        """Mock data provider for testing"""
        provider = Mock(spec=MarketDataProvider)
        provider.get_historical_data.return_value = {
            'AAPL': {
                'prices': [150.0, 151.0, 152.0],
                'volumes': [1000000, 1100000, 1050000],
                'timestamps': ['2024-01-01', '2024-01-02', '2024-01-03']
            }
        }
        return provider

    @pytest.fixture
    def data_agent(self, mock_data_provider):
        """Data agent with mocked dependencies"""
        agent = DataAgent()
        agent.data_provider = mock_data_provider
        agent.memory_agent = Mock()
        return agent

    def test_fetch_market_data_success(self, data_agent, mock_data_provider):
        """Test successful market data fetching"""
        result = data_agent.fetch_market_data('AAPL', '1d', days=3)

        assert result['symbol'] == 'AAPL'
        assert len(result['prices']) == 3
        assert result['data_quality'] > 0.9
        mock_data_provider.get_historical_data.assert_called_once_with('AAPL', '1d', 3)

    def test_fetch_market_data_provider_failure(self, data_agent, mock_data_provider):
        """Test handling of data provider failures"""
        mock_data_provider.get_historical_data.side_effect = ConnectionError("API unavailable")

        with pytest.raises(DataFetchError):
            data_agent.fetch_market_data('AAPL', '1d', days=3)

        # Verify fallback logic is triggered
        assert data_agent.fallback_provider.get_historical_data.called

    @patch('src.agents.data_agent.DataAgent.store_in_memory')
    def test_data_processing_and_storage(self, mock_store, data_agent, mock_data_provider):
        """Test data processing pipeline and memory storage"""
        # Execute data fetch and processing
        result = data_agent.fetch_market_data('AAPL', '1d', days=3)

        # Verify data processing
        assert 'technical_indicators' in result
        assert 'sentiment_score' in result
        assert 'volatility' in result

        # Verify storage call
        mock_store.assert_called_once()
        stored_data = mock_store.call_args[0][0]
        assert stored_data['symbol'] == 'AAPL'
        assert 'processed_at' in stored_data

    @pytest.mark.parametrize("symbol,expected_quality", [
        ("AAPL", 0.95),
        ("TSLA", 0.92),
        ("UNKNOWN", 0.0)
    ])
    def test_data_quality_assessment(self, data_agent, symbol, expected_quality):
        """Test data quality assessment for different symbols"""
        if symbol == "UNKNOWN":
            with pytest.raises(InvalidSymbolError):
                data_agent.assess_data_quality(symbol)
        else:
            quality = data_agent.assess_data_quality(symbol)
            assert abs(quality - expected_quality) < 0.05
```

### Strategy Testing

```python
# tests/unit/test_strategy_agent.py
import pytest
import numpy as np
from src.agents.strategy_agent import StrategyAgent
from src.strategies.options_strategy import CoveredCallStrategy

class TestStrategyAgent:
    @pytest.fixture
    def sample_market_data(self):
        """Sample market data for testing"""
        return {
            'underlying_price': 150.0,
            'volatility': 0.25,
            'risk_free_rate': 0.05,
            'dividend_yield': 0.02,
            'options_chain': {
                'calls': [
                    {'strike': 145, 'bid': 8.50, 'ask': 8.75, 'volume': 1000},
                    {'strike': 150, 'bid': 5.20, 'ask': 5.45, 'volume': 1500},
                    {'strike': 155, 'bid': 2.80, 'ask': 3.05, 'volume': 800}
                ],
                'puts': [
                    {'strike': 145, 'bid': 1.20, 'ask': 1.35, 'volume': 500},
                    {'strike': 150, 'bid': 3.80, 'ask': 4.05, 'volume': 1200},
                    {'strike': 155, 'bid': 7.50, 'ask': 7.75, 'volume': 600}
                ]
            }
        }

    def test_covered_call_strategy_generation(self, sample_market_data):
        """Test covered call strategy generation"""
        strategy_agent = StrategyAgent()
        strategy = CoveredCallStrategy()

        # Generate strategy recommendation
        recommendation = strategy_agent.generate_options_strategy(
            symbol='AAPL',
            market_data=sample_market_data,
            risk_tolerance='moderate',
            time_horizon_days=30
        )

        # Verify strategy structure
        assert recommendation['strategy_type'] == 'covered_call'
        assert 'legs' in recommendation
        assert len(recommendation['legs']) == 2  # Stock + call option

        # Verify position sizing
        total_exposure = sum(abs(leg['quantity']) * leg['price'] for leg in recommendation['legs'])
        assert total_exposure <= 100000  # Max position size

        # Verify risk metrics
        assert 'max_loss' in recommendation
        assert 'max_profit' in recommendation
        assert 'breakeven' in recommendation
        assert recommendation['max_loss'] < 0  # Debit strategy
        assert recommendation['max_profit'] > 0

    def test_strategy_risk_assessment(self, sample_market_data):
        """Test strategy risk assessment"""
        strategy_agent = StrategyAgent()

        strategy = {
            'legs': [
                {'type': 'stock', 'quantity': 1000, 'price': 150.0},
                {'type': 'call', 'quantity': -10, 'strike': 155, 'price': 3.05}
            ]
        }

        risk_assessment = strategy_agent.assess_strategy_risk(strategy, sample_market_data)

        # Verify risk metrics
        assert 'delta' in risk_assessment
        assert 'gamma' in risk_assessment
        assert 'theta' in risk_assessment
        assert 'vega' in risk_assessment
        assert 'rho' in risk_assessment

        # Verify delta is slightly negative (covered call)
        assert -0.1 < risk_assessment['delta'] < 0.1

        # Verify theta is positive (time decay benefits strategy)
        assert risk_assessment['theta'] > 0

    @pytest.mark.parametrize("volatility,expected_premium", [
        (0.15, 2.50),  # Low vol scenario
        (0.25, 3.05),  # Normal vol scenario
        (0.35, 4.20)   # High vol scenario
    ])
    def test_volatility_impact_on_premium(self, volatility, expected_premium):
        """Test volatility impact on option premiums"""
        strategy_agent = StrategyAgent()

        market_data = {
            'underlying_price': 150.0,
            'volatility': volatility,
            'risk_free_rate': 0.05,
            'time_to_expiry': 30/365
        }

        premium = strategy_agent.calculate_option_premium(
            strike=155,
            option_type='call',
            market_data=market_data
        )

        assert abs(premium - expected_premium) < 0.50  # Allow reasonable tolerance
```

## Integration Testing

### Agent Communication Testing

```python
# tests/integration/test_agent_communication.py
import pytest
import asyncio
from src.agents.data_agent import DataAgent
from src.agents.strategy_agent import StrategyAgent
from src.communication.a2a_protocol import A2AProtocol
from tests.fixtures.redis_fixture import redis_server

class TestAgentCommunication:
    @pytest.fixture
    async def communication_setup(self, redis_server):
        """Setup agents with real communication layer"""
        protocol = A2AProtocol(redis_url=redis_server)

        data_agent = DataAgent(agent_id="data_agent_001", protocol=protocol)
        strategy_agent = StrategyAgent(agent_id="strategy_agent_001", protocol=protocol)

        await protocol.initialize()
        await data_agent.initialize()
        await strategy_agent.initialize()

        yield data_agent, strategy_agent, protocol

        # Cleanup
        await protocol.shutdown()
        await data_agent.shutdown()
        await strategy_agent.shutdown()

    @pytest.mark.asyncio
    async def test_data_request_response_cycle(self, communication_setup):
        """Test complete data request and response cycle"""
        data_agent, strategy_agent, protocol = communication_setup

        # Strategy agent requests market data
        request = {
            'message_type': 'query',
            'query_type': 'market_data',
            'parameters': {
                'symbols': ['AAPL', 'GOOGL'],
                'timeframe': '1d',
                'days': 30
            }
        }

        # Send request and wait for response
        response_future = await strategy_agent.query_data_agent(request)
        response = await response_future

        # Verify response structure
        assert response['status'] == 'success'
        assert 'data' in response
        assert len(response['data']) == 2  # Two symbols
        assert all(symbol in response['data'] for symbol in ['AAPL', 'GOOGL'])

        # Verify data quality
        for symbol_data in response['data'].values():
            assert 'prices' in symbol_data
            assert 'volumes' in symbol_data
            assert len(symbol_data['prices']) == 30

    @pytest.mark.asyncio
    async def test_broadcast_market_update(self, communication_setup):
        """Test broadcasting market updates to multiple agents"""
        data_agent, strategy_agent, protocol = communication_setup

        # Add another strategy agent
        strategy_agent_2 = StrategyAgent(agent_id="strategy_agent_002", protocol=protocol)
        await strategy_agent_2.initialize()

        # Data agent broadcasts market update
        market_update = {
            'symbol': 'AAPL',
            'price': 185.50,
            'volume': 1500000,
            'timestamp': '2024-01-15T10:30:00Z',
            'significant_move': True
        }

        await data_agent.broadcast_market_update(market_update)

        # Wait for agents to receive update
        await asyncio.sleep(0.1)

        # Verify both strategy agents received the update
        assert strategy_agent.last_market_update == market_update
        assert strategy_agent_2.last_market_update == market_update

        await strategy_agent_2.shutdown()

    @pytest.mark.asyncio
    async def test_error_handling_and_recovery(self, communication_setup):
        """Test error handling and recovery in communication"""
        data_agent, strategy_agent, protocol = communication_setup

        # Simulate data provider failure
        original_provider = data_agent.data_provider
        data_agent.data_provider = None  # Simulate failure

        # Attempt request
        request = {
            'message_type': 'query',
            'query_type': 'market_data',
            'parameters': {'symbols': ['AAPL']}
        }

        response_future = await strategy_agent.query_data_agent(request)
        response = await response_future

        # Verify error handling
        assert response['status'] == 'error'
        assert 'error_message' in response
        assert 'retry_suggested' in response

        # Restore provider and retry
        data_agent.data_provider = original_provider
        response_future = await strategy_agent.query_data_agent(request)
        response = await response_future

        # Verify recovery
        assert response['status'] == 'success'
```

### Database Integration Testing

```python
# tests/integration/test_database_operations.py
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from src.database.models import Trade, Position, Portfolio
from src.database.connection import DatabaseConnection
from tests.fixtures.database_fixture import test_database

class TestDatabaseOperations:
    @pytest.fixture
    def db_session(self, test_database):
        """Database session for testing"""
        engine = create_engine(test_database.url)
        Session = sessionmaker(bind=engine)
        session = Session()
        yield session
        session.close()

    def test_trade_persistence(self, db_session):
        """Test trade data persistence"""
        # Create test trade
        trade = Trade(
            symbol='AAPL',
            quantity=100,
            price=185.50,
            side='buy',
            order_type='market',
            timestamp='2024-01-15T10:30:00Z',
            execution_id='exec_001',
            commission=5.00
        )

        # Persist trade
        db_session.add(trade)
        db_session.commit()

        # Retrieve and verify
        retrieved_trade = db_session.query(Trade).filter_by(execution_id='exec_001').first()

        assert retrieved_trade.symbol == 'AAPL'
        assert retrieved_trade.quantity == 100
        assert retrieved_trade.price == 185.50
        assert retrieved_trade.side == 'buy'

    def test_portfolio_position_updates(self, db_session):
        """Test portfolio position updates"""
        # Initial position
        position = Position(
            symbol='AAPL',
            quantity=1000,
            average_price=180.00,
            current_price=185.50,
            market_value=185500.00,
            unrealized_pnl=5500.00
        )

        db_session.add(position)
        db_session.commit()

        # Update position (additional purchase)
        position.quantity += 500
        position.average_price = ((1000 * 180.00) + (500 * 185.50)) / 1500
        position.market_value = 1500 * 185.50
        position.unrealized_pnl = position.market_value - (1500 * position.average_price)

        db_session.commit()

        # Verify update
        updated_position = db_session.query(Position).filter_by(symbol='AAPL').first()
        assert updated_position.quantity == 1500
        assert abs(updated_position.average_price - 181.70) < 0.01  # Weighted average
        assert updated_position.market_value == 1500 * 185.50

    def test_portfolio_risk_calculations(self, db_session):
        """Test portfolio-level risk calculations"""
        # Create multiple positions
        positions = [
            Position(symbol='AAPL', quantity=1000, current_price=185.50, volatility=0.25),
            Position(symbol='MSFT', quantity=500, current_price=380.00, volatility=0.22),
            Position(symbol='GOOGL', quantity=200, current_price=2750.00, volatility=0.28)
        ]

        for pos in positions:
            pos.market_value = pos.quantity * pos.current_price
            db_session.add(pos)

        db_session.commit()

        # Calculate portfolio metrics
        portfolio = Portfolio()
        portfolio.calculate_risk_metrics(db_session)

        # Verify calculations
        assert portfolio.total_value > 0
        assert 0 < portfolio.volatility < 1  # Reasonable volatility range
        assert portfolio.var_95 > 0  # Value at Risk should be positive
        assert portfolio.sharpe_ratio is not None

        # Verify diversification
        assert portfolio.diversification_ratio > 1  # Should be diversified
```

## Performance Testing

### Load Testing Framework

```python
# tests/performance/test_load_performance.py
import pytest
import asyncio
import time
from locust import HttpUser, task, between
from src.performance.load_tester import LoadTester
from src.agents.data_agent import DataAgent

class TestLoadPerformance:
    def test_data_agent_concurrent_requests(self):
        """Test data agent performance under concurrent load"""
        load_tester = LoadTester()

        async def run_load_test():
            # Configure load test
            config = {
                'concurrent_users': 50,
                'ramp_up_time': 30,
                'test_duration': 300,  # 5 minutes
                'request_rate': 10  # requests per second
            }

            # Run load test
            results = await load_tester.run_data_agent_load_test(config)

            # Verify performance metrics
            assert results['avg_response_time'] < 500  # ms
            assert results['95th_percentile'] < 1000  # ms
            assert results['error_rate'] < 0.05  # 5%
            assert results['throughput'] > 8  # requests per second

            return results

        results = asyncio.run(run_load_test())

        # Log performance results
        print(f"Average response time: {results['avg_response_time']}ms")
        print(f"95th percentile: {results['95th_percentile']}ms")
        print(f"Error rate: {results['error_rate']*100}%")
        print(f"Throughput: {results['throughput']} req/s")

    def test_memory_usage_under_load(self):
        """Test memory usage during high load"""
        import psutil
        import os

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Simulate high load
        load_tester = LoadTester()
        asyncio.run(load_tester.generate_high_load(duration=60))

        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory

        # Verify memory usage is reasonable
        assert memory_increase < 500  # Less than 500MB increase
        assert final_memory < 2000  # Less than 2GB total usage

    def test_database_performance_under_load(self):
        """Test database performance during concurrent operations"""
        load_tester = LoadTester()

        async def run_db_load_test():
            config = {
                'concurrent_writers': 20,
                'concurrent_readers': 30,
                'test_duration': 120,
                'batch_size': 100
            }

            results = await load_tester.run_database_load_test(config)

            # Verify database performance
            assert results['write_latency_avg'] < 50  # ms
            assert results['read_latency_avg'] < 20  # ms
            assert results['deadlock_rate'] < 0.01  # 1%
            assert results['connection_pool_utilization'] < 0.9  # 90%

            return results

        results = asyncio.run(run_db_load_test())

        print(f"Write latency: {results['write_latency_avg']}ms")
        print(f"Read latency: {results['read_latency_avg']}ms")
        print(f"Deadlock rate: {results['deadlock_rate']*100}%")

# Locust load testing script
class ApiUser(HttpUser):
    wait_time = between(1, 3)

    @task
    def get_market_data(self):
        self.client.get("/api/v1/market-data?symbols=AAPL,GOOGL,MSFT")

    @task
    def get_portfolio_status(self):
        self.client.get("/api/v1/portfolio/status")

    @task
    def submit_trade(self):
        trade_data = {
            "symbol": "AAPL",
            "quantity": 100,
            "side": "buy",
            "order_type": "market"
        }
        self.client.post("/api/v1/trades", json=trade_data)
```

### Memory and CPU Profiling

```python
# tests/performance/test_memory_profiling.py
import pytest
import tracemalloc
from memory_profiler import profile
from src.agents.strategy_agent import StrategyAgent

class TestMemoryProfiling:
    def test_strategy_generation_memory_usage(self):
        """Profile memory usage during strategy generation"""
        tracemalloc.start()

        strategy_agent = StrategyAgent()

        # Generate multiple strategies
        for i in range(100):
            market_data = {
                'underlying_price': 150.0 + i,
                'volatility': 0.25,
                'risk_free_rate': 0.05
            }

            strategy = strategy_agent.generate_options_strategy(
                symbol=f'TEST{i}',
                market_data=market_data,
                risk_tolerance='moderate'
            )

        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # Convert to MB
        current_mb = current / 1024 / 1024
        peak_mb = peak / 1024 / 1024

        print(f"Current memory usage: {current_mb:.2f} MB")
        print(f"Peak memory usage: {peak_mb:.2f} MB")

        # Assert reasonable memory usage
        assert peak_mb < 100  # Less than 100MB peak
        assert current_mb < 50  # Less than 50MB final usage

    @profile
    def profiled_strategy_calculation(self):
        """Memory profiled strategy calculation"""
        strategy_agent = StrategyAgent()

        # Complex strategy calculation
        market_data = {
            'underlying_price': 150.0,
            'volatility': 0.25,
            'risk_free_rate': 0.05,
            'options_chain': {
                'calls': [{'strike': strike, 'price': 5.0} for strike in range(140, 170, 5)],
                'puts': [{'strike': strike, 'price': 5.0} for strike in range(140, 170, 5)]
            }
        }

        strategy = strategy_agent.generate_complex_strategy(market_data)

        return strategy
```

## Backtesting Validation

### Strategy Backtesting Tests

```python
# tests/backtesting/test_strategy_backtesting.py
import pytest
import pandas as pd
from src.backtesting.engine import BacktestingEngine
from src.strategies.momentum_strategy import MomentumStrategy
from src.strategies.mean_reversion_strategy import MeanReversionStrategy

class TestStrategyBacktesting:
    @pytest.fixture
    def historical_data(self):
        """Load historical test data"""
        # Load test data covering multiple market conditions
        data = pd.read_csv('tests/data/historical_prices.csv', index_col='date', parse_dates=True)

        # Ensure data quality
        assert len(data) > 500  # At least 500 trading days
        assert 'AAPL' in data.columns
        assert 'SPY' in data.columns

        return data

    def test_momentum_strategy_backtest(self, historical_data):
        """Test momentum strategy backtesting"""
        engine = BacktestingEngine()
        strategy = MomentumStrategy(lookback_period=20, hold_period=5)

        # Configure backtest
        config = {
            'start_date': '2020-01-01',
            'end_date': '2023-12-31',
            'initial_capital': 100000,
            'commission': 0.001,  # 0.1% per trade
            'slippage': 0.0005    # 0.05% slippage
        }

        # Run backtest
        results = engine.run_backtest(strategy, historical_data, config)

        # Verify results structure
        assert 'total_return' in results
        assert 'sharpe_ratio' in results
        assert 'max_drawdown' in results
        assert 'win_rate' in results
        assert 'profit_factor' in results

        # Verify reasonable performance metrics
        assert results['total_return'] > -0.5  # Not a total loss
        assert results['sharpe_ratio'] > 0     # Positive risk-adjusted return
        assert results['max_drawdown'] < 0.5   # Less than 50% drawdown
        assert results['win_rate'] > 0.4       # At least 40% win rate

        print(f"Momentum Strategy Results:")
        print(f"  Total Return: {results['total_return']:.2%}")
        print(f"  Sharpe Ratio: {results['sharpe_ratio']:.2f}")
        print(f"  Max Drawdown: {results['max_drawdown']:.2%}")
        print(f"  Win Rate: {results['win_rate']:.2%}")

    def test_mean_reversion_strategy_backtest(self, historical_data):
        """Test mean reversion strategy backtesting"""
        engine = BacktestingEngine()
        strategy = MeanReversionStrategy(
            lookback_period=50,
            entry_threshold=2.0,
            exit_threshold=0.5,
            max_holding_period=20
        )

        config = {
            'start_date': '2020-01-01',
            'end_date': '2023-12-31',
            'initial_capital': 100000,
            'commission': 0.001,
            'slippage': 0.0005
        }

        results = engine.run_backtest(strategy, historical_data, config)

        # Verify strategy performed as expected for mean reversion
        assert results['total_return'] > -0.3  # Reasonable performance
        assert results['sharpe_ratio'] > -1    # Not extremely negative
        assert results['max_drawdown'] < 0.4   # Controlled drawdown

        # Mean reversion should have more frequent trades
        assert results['total_trades'] > 50

    def test_strategy_comparison(self, historical_data):
        """Compare multiple strategies on same data"""
        engine = BacktestingEngine()

        strategies = [
            ('Momentum', MomentumStrategy(lookback_period=20)),
            ('Mean Reversion', MeanReversionStrategy(lookback_period=50)),
        ]

        config = {
            'start_date': '2021-01-01',
            'end_date': '2023-12-31',
            'initial_capital': 100000
        }

        comparison_results = {}

        for name, strategy in strategies:
            results = engine.run_backtest(strategy, historical_data, config)
            comparison_results[name] = results

        # Verify strategies have different performance profiles
        momentum_return = comparison_results['Momentum']['total_return']
        mr_return = comparison_results['Mean Reversion']['total_return']

        # Strategies should have different returns (not identical)
        assert abs(momentum_return - mr_return) > 0.05

        print("Strategy Comparison:")
        for name, results in comparison_results.items():
            print(f"  {name}: {results['total_return']:.2%} return, "
                  f"{results['sharpe_ratio']:.2f} Sharpe")

    def test_risk_metrics_validation(self, historical_data):
        """Validate risk metrics calculation"""
        engine = BacktestingEngine()
        strategy = MomentumStrategy()

        config = {
            'start_date': '2020-01-01',
            'end_date': '2023-12-31',
            'initial_capital': 100000
        }

        results = engine.run_backtest(strategy, historical_data, config)

        # Validate VaR calculation
        assert 'var_95' in results
        assert 0 < results['var_95'] < 0.5  # Reasonable VaR range

        # Validate drawdown calculations
        assert 'max_drawdown' in results
        assert results['max_drawdown'] >= 0

        # Validate Sharpe ratio bounds
        assert -5 < results['sharpe_ratio'] < 5  # Reasonable Sharpe range

        # Cross-validate metrics
        # If Sharpe is high, returns should be high relative to volatility
        if results['sharpe_ratio'] > 2:
            assert results['total_return'] > results['volatility'] * results['sharpe_ratio'] * 0.8
```

## Test Automation and CI/CD

### GitHub Actions CI Pipeline

```yaml
# .github/workflows/ci.yml
name: CI Pipeline

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.9, 3.10, 3.11]

    steps:
    - uses: actions/checkout@v3

    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install -r requirements-dev.txt

    - name: Run unit tests
      run: |
        pytest tests/unit/ -v --cov=src --cov-report=xml --cov-report=html

    - name: Run integration tests
      run: |
        pytest tests/integration/ -v --tb=short
      env:
        REDIS_URL: redis://localhost:6379
        DATABASE_URL: postgresql://test:test@localhost:5432/test_db

    - name: Run performance tests
      run: |
        pytest tests/performance/ -v --durations=10

    - name: Upload coverage reports
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml

  backtesting-validation:
    runs-on: ubuntu-latest
    needs: test

    steps:
    - uses: actions/checkout@v3

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'

    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install -r requirements-dev.txt

    - name: Run backtesting validation
      run: |
        pytest tests/backtesting/ -v --tb=short
      env:
        BACKTEST_DATA_PATH: tests/data/

  security-scan:
    runs-on: ubuntu-latest
    needs: test

    steps:
    - uses: actions/checkout@v3

    - name: Run security scan
      uses: securecodewarrior/github-action-gosec@master
      with:
        args: './...'

    - name: Run dependency vulnerability scan
      run: |
        pip install safety
        safety check
```

### Test Data Management

```python
# tests/fixtures/test_data_manager.py
import pytest
import pandas as pd
from pathlib import Path

class TestDataManager:
    """Manages test data for consistent testing"""

    def __init__(self, data_dir: str = "tests/data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)

    def get_historical_prices(self, symbols: list = None, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """Get historical price data for testing"""
        if symbols is None:
            symbols = ['AAPL', 'GOOGL', 'MSFT', 'SPY']

        # Load or generate test data
        data_file = self.data_dir / "historical_prices.csv"

        if data_file.exists():
            data = pd.read_csv(data_file, index_col='date', parse_dates=True)
        else:
            # Generate synthetic data for testing
            data = self._generate_synthetic_data(symbols, start_date, end_date)
            data.to_csv(data_file)

        return data

    def _generate_synthetic_data(self, symbols: list, start_date: str, end_date: str) -> pd.DataFrame:
        """Generate synthetic price data"""
        if start_date is None:
            start_date = '2020-01-01'
        if end_date is None:
            end_date = '2024-01-01'

        dates = pd.date_range(start_date, end_date, freq='B')  # Business days

        data = {}
        for symbol in symbols:
            # Generate random walk prices
            np.random.seed(hash(symbol) % 2**32)  # Deterministic seed per symbol

            # Start with reasonable price
            start_price = 100 + hash(symbol) % 200

            # Generate daily returns (random walk with drift)
            daily_returns = np.random.normal(0.0005, 0.02, len(dates))  # 0.05% drift, 2% vol

            # Calculate prices
            prices = start_price * np.exp(np.cumsum(daily_returns))

            # Generate volumes
            base_volume = 1000000 + hash(symbol) % 5000000
            volumes = np.random.lognormal(np.log(base_volume), 0.5, len(dates))

            data[symbol] = prices
            data[f"{symbol}_volume"] = volumes.astype(int)

        df = pd.DataFrame(data, index=dates)
        df.index.name = 'date'

        return df

    def get_options_chain_data(self, symbol: str, expiration: str = None) -> dict:
        """Get options chain data for testing"""
        # Return realistic options chain structure
        return {
            'underlying_price': 150.0,
            'calls': [
                {'strike': 140, 'bid': 12.50, 'ask': 12.75, 'volume': 1500, 'open_interest': 5000},
                {'strike': 145, 'bid': 8.20, 'ask': 8.45, 'volume': 2200, 'open_interest': 7500},
                {'strike': 150, 'bid': 4.80, 'ask': 5.05, 'volume': 3100, 'open_interest': 10000},
                {'strike': 155, 'bid': 2.45, 'ask': 2.70, 'volume': 1800, 'open_interest': 6200},
                {'strike': 160, 'bid': 1.15, 'ask': 1.40, 'volume': 950, 'open_interest': 3800}
            ],
            'puts': [
                {'strike': 140, 'bid': 1.05, 'ask': 1.30, 'volume': 1200, 'open_interest': 4100},
                {'strike': 145, 'bid': 2.80, 'ask': 3.05, 'volume': 1950, 'open_interest': 6800},
                {'strike': 150, 'bid': 5.60, 'ask': 5.85, 'volume': 2800, 'open_interest': 9200},
                {'strike': 155, 'bid': 9.75, 'ask': 10.00, 'volume': 1650, 'open_interest': 5500},
                {'strike': 160, 'bid': 14.80, 'ask': 15.05, 'volume': 820, 'open_interest': 3200}
            ]
        }

@pytest.fixture
def test_data_manager():
    """Fixture for test data manager"""
    return TestDataManager()

@pytest.fixture
def historical_prices(test_data_manager):
    """Fixture for historical price data"""
    return test_data_manager.get_historical_prices()
```

## Test Reporting and Analytics

### Test Results Dashboard

```python
# tests/reporting/test_dashboard.py
import pytest
import json
from datetime import datetime
from pathlib import Path

class TestResultsDashboard:
    def __init__(self, results_dir: str = "test_results"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)

    def save_test_results(self, session_results):
        """Save comprehensive test results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        results = {
            'timestamp': timestamp,
            'summary': {
                'total_tests': session_results.testscollected,
                'passed': session_results.tests_passed,
                'failed': len(session_results.failed),
                'skipped': len(session_results.skipped),
                'duration': session_results.duration
            },
            'performance_metrics': self._collect_performance_metrics(),
            'coverage': self._collect_coverage_metrics(),
            'failures': [self._format_failure(f) for f in session_results.failed]
        }

        filename = f"test_results_{timestamp}.json"
        with open(self.results_dir / filename, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        return results

    def _collect_performance_metrics(self):
        """Collect performance test metrics"""
        # Implementation would collect from performance tests
        return {
            'avg_response_time': 245.5,
            '95th_percentile': 450.2,
            'error_rate': 0.023,
            'throughput': 12.5
        }

    def _collect_coverage_metrics(self):
        """Collect code coverage metrics"""
        # Implementation would parse coverage reports
        return {
            'line_coverage': 87.3,
            'branch_coverage': 82.1,
            'function_coverage': 91.5
        }

    def _format_failure(self, failure):
        """Format test failure for reporting"""
        return {
            'test_name': failure.nodeid,
            'error_type': failure.errortype.__name__,
            'error_message': str(failure.errortrace),
            'duration': failure.duration
        }

    def generate_html_report(self, results):
        """Generate HTML test report"""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>ABC Application Test Results</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .summary {{ background: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .metric {{ display: inline-block; margin: 10px; text-align: center; }}
                .passed {{ color: green; }}
                .failed {{ color: red; }}
                .failures {{ margin-top: 20px; }}
                .failure {{ background: #ffe6e6; margin: 10px 0; padding: 10px; border-left: 4px solid red; }}
            </style>
        </head>
        <body>
            <h1>ABC Application Test Results</h1>
            <div class="summary">
                <h2>Test Summary</h2>
                <div class="metric">
                    <div style="font-size: 24px; font-weight: bold;">{results['summary']['total_tests']}</div>
                    <div>Total Tests</div>
                </div>
                <div class="metric">
                    <div style="font-size: 24px; font-weight: bold; color: green;">{results['summary']['passed']}</div>
                    <div>Passed</div>
                </div>
                <div class="metric">
                    <div style="font-size: 24px; font-weight: bold; color: red;">{results['summary']['failed']}</div>
                    <div>Failed</div>
                </div>
                <div class="metric">
                    <div style="font-size: 24px; font-weight: bold; color: orange;">{results['summary']['skipped']}</div>
                    <div>Skipped</div>
                </div>
            </div>

            {'<div class="failures"><h2>Test Failures</h2>' + ''.join(f'<div class="failure"><h3>{f["test_name"]}</h3><p>{f["error_message"]}</p></div>' for f in results['failures']) + '</div>' if results['failures'] else ''}
        </body>
        </html>
        """

        report_file = self.results_dir / f"test_report_{results['timestamp']}.html"
        with open(report_file, 'w') as f:
            f.write(html_content)

        return report_file

# Pytest plugin for custom reporting
@pytest.hookimpl(tryfirst=True)
def pytest_sessionfinish(session, exitstatus):
    """Hook to generate custom test reports"""
    dashboard = TestResultsDashboard()

    # Mock session results for demonstration
    class MockResults:
        def __init__(self):
            self.testscollected = 150
            self.tests_passed = 142
            self.failed = []
            self.skipped = []
            self.duration = 45.2

    results = dashboard.save_test_results(MockResults())
    html_report = dashboard.generate_html_report(results)

    print(f"Test report generated: {html_report}")
```

This comprehensive testing framework ensures the ABC Application system maintains high quality, performance, and reliability across all components and trading scenarios.

---

*For configuration details, see IMPLEMENTATION/configuration.md. For API health monitoring, see REFERENCE/api-health-monitoring.md.*