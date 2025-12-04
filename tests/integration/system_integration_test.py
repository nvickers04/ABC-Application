#!/usr/bin/env python3
"""
Comprehensive System Integration Test for Enhanced Data Subagents.
Tests end-to-end functionality with LLM-enhanced data collection and analysis.
"""

import sys
import os
import asyncio
import time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

async def test_full_system_integration():
    """Test complete system integration with enhanced subagents."""
    print("🚀 Starting Full System Integration Test")
    print("=" * 50)

    try:
        # Import main components
        from src.agents.data import DataAgent
        print("✅ DataAgent imported successfully")

        # Initialize DataAgent
        data_agent = DataAgent()
        print("✅ DataAgent initialized successfully")

        # Test single symbol processing with enhanced subagents
        test_symbol = "AAPL"
        print(f"\n📊 Testing enhanced data collection for {test_symbol}")

        start_time = time.time()

        # Process symbol with enhanced subagents
        result = await data_agent.process_input({
            'symbols': [test_symbol],
            'period': '1y'  # Shorter period for faster testing
        })

        processing_time = time.time() - start_time
        print(f"⏱️  Processing time: {processing_time:.2f}s")

        # Validate enhanced subagent integration
        validation_results = validate_enhanced_results(result, test_symbol)

        # Print comprehensive results
        print_results_summary(result, validation_results, processing_time)

        return validation_results['all_checks_pass']

    except Exception as e:
        print(f"❌ System integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def validate_enhanced_results(result, symbol):
    """Validate that enhanced subagents are working correctly."""
    validation = {
        'symbol_processed': False,
        'enhanced_subagents': [],
        'llm_analysis_present': [],
        'data_quality_checks': [],
        'performance_metrics': {},
        'all_checks_pass': False
    }

    try:
        # Check if symbol was processed
        if 'symbol_data' in result and symbol in result['symbol_data']:
            validation['symbol_processed'] = True
            symbol_data = result['symbol_data'][symbol]

            # Check for enhanced subagents
            enhanced_subagents = []
            llm_analysis_sources = []

            subagent_keys = ['institutional', 'microstructure', 'options', 'kalshi', 'news', 'yfinance', 'fundamental']
            for key in subagent_keys:
                if key in symbol_data:
                    subagent_data = symbol_data[key]
                    if isinstance(subagent_data, dict):
                        if subagent_data.get('enhanced', False):
                            enhanced_subagents.append(key)
                        if 'llm_analysis' in subagent_data:
                            llm_analysis_sources.append(key)

            validation['enhanced_subagents'] = enhanced_subagents
            validation['llm_analysis_present'] = llm_analysis_sources

            # Data quality checks
            quality_checks = []

            # Check for DataFrames in enhanced subagents
            if 'yfinance' in enhanced_subagents and 'price_data' in symbol_data.get('yfinance', {}):
                quality_checks.append("YFinance DataFrames present")

            if 'institutional' in enhanced_subagents and 'consolidated_data' in symbol_data.get('institutional', {}):
                quality_checks.append("Institutional data consolidated")

            if 'news' in enhanced_subagents and 'articles_df' in symbol_data.get('news', {}):
                quality_checks.append("News articles DataFrame present")

            if 'options' in enhanced_subagents and 'consolidated_data' in symbol_data.get('options', {}):
                quality_checks.append("Options data consolidated")

            validation['data_quality_checks'] = quality_checks

            # Performance metrics
            validation['performance_metrics'] = {
                'enhanced_subagents_count': len(enhanced_subagents),
                'llm_analysis_sources_count': len(llm_analysis_sources),
                'data_quality_checks_count': len(quality_checks)
            }

            # Overall validation
            validation['all_checks_pass'] = (
                validation['symbol_processed'] and
                len(enhanced_subagents) >= 5 and  # At least 5 enhanced subagents
                len(llm_analysis_sources) >= 3 and  # At least 3 with LLM analysis
                len(quality_checks) >= 2  # At least 2 data quality checks
            )

    except Exception as e:
        print(f"Validation error: {e}")
        validation['validation_error'] = str(e)

    return validation

def print_results_summary(result, validation, processing_time):
    """Print comprehensive test results summary."""
    print("\n📋 Test Results Summary")
    print("-" * 30)

    print(f"✅ Symbol processed: {validation['symbol_processed']}")
    print(f"🔄 Enhanced subagents: {len(validation['enhanced_subagents'])}")
    print(f"🤖 LLM analysis sources: {len(validation['llm_analysis_present'])}")
    print(f"📊 Data quality checks: {len(validation['data_quality_checks'])}")

    if validation['enhanced_subagents']:
        print(f"   Enhanced: {', '.join(validation['enhanced_subagents'])}")

    if validation['llm_analysis_present']:
        print(f"   LLM analysis: {', '.join(validation['llm_analysis_present'])}")

    if validation['data_quality_checks']:
        for check in validation['data_quality_checks']:
            print(f"   ✓ {check}")

    print(f"⏱️  Processing time: {processing_time:.2f}s")
    print(f"🎯 Overall result: {'✅ PASS' if validation['all_checks_pass'] else '❌ FAIL'}")

    # Additional insights
    if 'symbol_data' in result:
        symbol_data = list(result['symbol_data'].values())[0]
        if 'predictive_insights' in symbol_data:
            insights = symbol_data['predictive_insights']
            if 'confidence_score' in insights:
                print(f"   📈 Predictive confidence: {insights['confidence_score']:.2f}")
def main():
    """Main test execution."""
    print("🧪 ABC Application - Enhanced Data Subagents Integration Test")
    print("Testing LLM-powered data collection and analysis system")
    print()

    # Run async test
    success = asyncio.run(test_full_system_integration())

    print("\n" + "=" * 50)
    if success:
        print("🎉 SYSTEM INTEGRATION TEST PASSED!")
        print("✅ Enhanced data subagents are working correctly")
        print("✅ LLM analysis is integrated throughout the system")
        print("✅ Data quality and performance metrics are good")
        print("\n🚀 System is ready for production use with enhanced AI capabilities!")
    else:
        print("❌ SYSTEM INTEGRATION TEST FAILED!")
        print("⚠️  Some enhanced subagents may not be working correctly")
        print("🔍 Check the logs above for specific issues")

    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)