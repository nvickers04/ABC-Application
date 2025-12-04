#!/usr/bin/env python3
"""Test script to check RiskAgent import and instantiation."""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

try:
    print("Attempting to import RiskAgent...")
    from src.agents.risk import RiskAgent
    print("✅ RiskAgent imported successfully")

    print("Attempting to instantiate RiskAgent...")
    ra = RiskAgent()
    print("✅ RiskAgent instantiated successfully")

except Exception as e:
    print(f"❌ Error importing/instantiating RiskAgent: {e}")
    import traceback
    traceback.print_exc()