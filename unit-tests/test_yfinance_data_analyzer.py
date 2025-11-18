import pytest
from src.agents.data_analyzers.yfinance_data_analyzer import YfinanceDataAnalyzer

@pytest.mark.asyncio
async def test_process_input():
    agent = YfinanceDataAnalyzer()
    input_data = {'symbols': ['AAPL'], 'data_types': ['quotes']}
    result = await agent.process_input(input_data)
    assert 'consolidated_data' in result
    assert 'llm_analysis' in result