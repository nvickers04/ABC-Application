#!/usr/bin/env python3
"""
Test script for enhanced data subagents with LLM capabilities.
Tests that all 7 subagents can be imported and have the required LLM methods.
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_subagent_imports():
    """Test that all enhanced subagents can be imported."""
    print("🧪 Testing Enhanced Data Subagents Import...")

    subagents = [
        ('institutional_datasub', 'InstitutionalDatasub'),
        ('microstructure_datasub', 'MicrostructureDatasub'),
        ('options_datasub', 'OptionsDatasub'),
        ('kalshi_datasub', 'KalshiDatasub'),
        ('news_datasub', 'NewsDatasub'),
        ('yfinance_datasub', 'YfinanceDatasub'),
        ('fundamental_datasub', 'FundamentalDatasub')
    ]

    results = {}

    for module_name, class_name in subagents:
        try:
            module = __import__(f'src.agents.data_subs.{module_name}', fromlist=[class_name])
            cls = getattr(module, class_name)
            results[class_name] = {'import': True, 'class': cls}
            print(f"✅ {class_name} imported successfully")
        except Exception as e:
            results[class_name] = {'import': False, 'error': str(e)}
            print(f"❌ {class_name} import failed: {e}")

    return results

def test_llm_methods(subagent_results):
    """Test that all subagents have the required LLM methods."""
    print("\n🧠 Testing LLM Methods...")

    # Define the actual method names for each subagent
    required_methods_by_subagent = {
        'InstitutionalDatasub': [
            '_plan_institutional_exploration',
            '_fetch_institutional_sources_concurrent',
            '_consolidate_institutional_data',
            '_analyze_institutional_data_llm'
        ],
        'MicrostructureDatasub': [
            '_plan_microstructure_exploration',
            '_fetch_microstructure_sources_concurrent',
            '_consolidate_microstructure_data',
            '_analyze_microstructure_data_llm'
        ],
        'OptionsDatasub': [
            '_plan_options_exploration',
            '_fetch_options_sources_concurrent',
            '_consolidate_options_data',
            '_analyze_options_data_llm'
        ],
        'KalshiDatasub': [
            '_plan_kalshi_exploration',
            '_fetch_kalshi_sources_concurrent',
            '_consolidate_kalshi_data',
            '_analyze_kalshi_data_llm'
        ],
        'NewsDatasub': [
            '_plan_news_exploration',
            '_execute_news_exploration',  # This subagent uses different naming
            '_consolidate_news_data',
            '_analyze_news_data_llm'
        ],
        'YfinanceDatasub': [
            '_plan_data_exploration',  # This subagent uses different naming
            '_execute_data_exploration',
            '_consolidate_market_data',
            '_analyze_market_data_llm'
        ],
        'FundamentalDatasub': [
            '_plan_fundamental_exploration',
            '_fetch_fundamental_sources_concurrent',
            '_consolidate_fundamental_data',
            '_analyze_fundamental_data_llm'
        ]
    }

    method_results = {}

    for subagent_name, result in subagent_results.items():
        if not result['import']:
            method_results[subagent_name] = {'available': False, 'error': 'Import failed'}
            continue

        cls = result['class']
        required_methods = required_methods_by_subagent.get(subagent_name, [])

        methods_found = []
        methods_missing = []

        for method_name in required_methods:
            if hasattr(cls, method_name):
                methods_found.append(method_name)
            else:
                methods_missing.append(method_name)

        method_results[subagent_name] = {
            'available': len(methods_missing) == 0,
            'found': methods_found,
            'missing': methods_missing
        }

        if methods_missing:
            print(f"❌ {subagent_name} missing methods: {methods_missing}")
        else:
            print(f"✅ {subagent_name} has all LLM methods")

    return method_results

def test_process_input_structure(subagent_results):
    """Test that process_input methods follow the enhanced structure."""
    print("\n🔄 Testing Process Input Structure...")

    structure_results = {}

    for subagent_name, result in subagent_results.items():
        if not result['import']:
            structure_results[subagent_name] = {'valid': False, 'error': 'Import failed'}
            continue

        cls = result['class']

        # Check if process_input exists
        if not hasattr(cls, 'process_input'):
            structure_results[subagent_name] = {'valid': False, 'error': 'No process_input method'}
            print(f"❌ {subagent_name} missing process_input method")
            continue

        # Check method signature (basic check)
        import inspect
        try:
            sig = inspect.signature(cls.process_input)
            params = list(sig.parameters.keys())

            # Should have 'self' and 'input_data'
            has_self = 'self' in params or len(params) >= 1
            has_input_data = 'input_data' in params

            structure_results[subagent_name] = {
                'valid': has_self and has_input_data,
                'params': params,
                'async': inspect.iscoroutinefunction(cls.process_input)
            }

            if has_self and has_input_data and inspect.iscoroutinefunction(cls.process_input):
                print(f"✅ {subagent_name} process_input structure valid")
            else:
                print(f"❌ {subagent_name} process_input structure invalid")

        except Exception as e:
            structure_results[subagent_name] = {'valid': False, 'error': str(e)}
            print(f"❌ {subagent_name} process_input inspection failed: {e}")

    return structure_results

def main():
    """Run all tests."""
    print("🚀 Enhanced Data Subagents Testing Suite")
    print("=" * 50)

    # Test imports
    import_results = test_subagent_imports()

    # Test LLM methods
    method_results = test_llm_methods(import_results)

    # Test process_input structure
    structure_results = test_process_input_structure(import_results)

    # Summary
    print("\n📊 Test Results Summary")
    print("=" * 50)

    total_subagents = len(import_results)
    import_success = sum(1 for r in import_results.values() if r['import'])
    method_success = sum(1 for r in method_results.values() if r.get('available', False))
    structure_success = sum(1 for r in structure_results.values() if r.get('valid', False))

    print(f"Imports: {import_success}/{total_subagents} ✅")
    print(f"LLM Methods: {method_success}/{total_subagents} ✅")
    print(f"Process Input: {structure_success}/{total_subagents} ✅")

    if import_success == total_subagents and method_success == total_subagents and structure_success == total_subagents:
        print("\n🎉 ALL TESTS PASSED! Enhanced data subagents are ready.")
        return True
    else:
        print("\n⚠️  Some tests failed. Check output above for details.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)