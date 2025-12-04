# [LABEL:TEST:pyramiding] [LABEL:TEST:unit] [LABEL:FRAMEWORK:pytest]
# [LABEL:AUTHOR:cline] [LABEL:UPDATED:2025-11-26] [LABEL:REVIEWED:pending]
#
# Purpose: Unit tests for PyramidingEngine class
# Dependencies: pytest, numpy, pandas, src.utils.pyramiding
# Related: src/utils/pyramiding.py

import pytest
import numpy as np
import pandas as pd
from src.utils.pyramiding import PyramidingEngine

@pytest.fixture
def engine():
    return PyramidingEngine(max_tiers=5, base_risk_pct=0.02)

def test_calculate_pyramiding_plan(engine):
    plan = engine.calculate_pyramiding_plan(
        current_price=105.0,
        entry_price=100.0,
        volatility=0.2,
        trend_strength=0.6,
        current_pnl_pct=0.05,
        max_drawdown_pct=0.1,
        portfolio_value=100000.0
    )
    assert isinstance(plan, dict)
    assert 'base_position_size' in plan
    assert 'tiers' in plan
    assert 'scaling_factors' in plan

def test_calculate_stops(engine):
    stops = engine.calculate_stops(
        entry_price=100.0,
        current_price=105.0,
        max_drawdown_pct=0.1,
        vol_regime='normal'
    )
    assert isinstance(stops, dict)
    assert 'initial_stop' in stops
    assert 'trailing_stop' in stops

def test_should_add_to_position(engine):
    result = engine.should_add_to_position(
        current_price=110.0,
        last_tier_price=100.0,
        current_pnl_pct=0.10,
        volatility=0.15
    )
    assert isinstance(result, bool)

# Add more test cases as needed

if __name__ == "__main__":
    pytest.main([__file__])
