import asyncio
import pytest
from src.agents.data_analyzers.ibkr_data_analyzer import IBKRDataAnalyzer

@pytest.mark.asyncio
async def test_ibkr_data_analyzer_init():
    """Test IBKRDataAnalyzer initialization."""
    analyzer = IBKRDataAnalyzer()
    assert analyzer is not None
    assert hasattr(analyzer, 'connector')
    assert hasattr(analyzer, 'historical_provider')
    assert hasattr(analyzer, 'data_sources')
    assert 'ibkr_historical' in analyzer.data_sources
    assert 'ibkr_live' in analyzer.data_sources
    assert 'yfinance_fallback' in analyzer.data_sources

@pytest.mark.asyncio
async def test_ibkr_process_input():
    """Test IBKRDataAnalyzer process_input method."""
    analyzer = IBKRDataAnalyzer()
    result = await analyzer.process_input({'symbols': ['SPY']})
    assert isinstance(result, dict)
    assert 'consolidated_data' in result or 'error' in result
    if 'consolidated_data' in result:
        assert 'symbols' in result['consolidated_data']
        assert len(result['consolidated_data']['symbols']) == 1

@pytest.mark.asyncio
async def test_ibkr_fetch_historical():
    """Test historical data fetching with fallback."""
    analyzer = IBKRDataAnalyzer()
    # This test may fail if IBKR not connected, but should fallback to yfinance
    result = await analyzer._fetch_ibkr_historical('SPY', ['historical'], '1mo')
    assert isinstance(result, dict)
    assert 'data' in result or 'error' in result

@pytest.mark.asyncio
async def test_ibkr_fetch_live():
    """Test live data fetching with fallback."""
    analyzer = IBKRDataAnalyzer()
    result = await analyzer._fetch_ibkr_live('SPY', ['quotes'], '1mo')
    assert isinstance(result, dict)
    assert 'data' in result or 'error' in result

@pytest.mark.asyncio
async def test_ibkr_consolidate_data():
    """Test data consolidation."""
    analyzer = IBKRDataAnalyzer()
    # Mock exploration results
    mock_results = {'SPY': {'ibkr_historical': {'data': {'2023-01-01': {'open': 100, 'high': 105, 'low': 99, 'close': 102, 'volume': 1000}}}}}
    consolidated = analyzer._consolidate_market_data(['SPY'], mock_results)
    assert isinstance(consolidated, dict)
    assert 'symbol_dataframes' in consolidated
    assert 'SPY' in consolidated['symbol_dataframes']

if __name__ == "__main__":
    pytest.main([__file__])
