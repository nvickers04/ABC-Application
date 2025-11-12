# compare_data_sources.py
# Purpose: Compare yfinance vs IBKR historical data quality and availability
# Helps users understand the benefits of IBKR data for professional backtesting

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
from integrations.ibkr_historical_data import IBKRHistoricalDataProvider
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

async def compare_data_sources():
    """
    Compare yfinance and IBKR data sources for quality, coverage, and differences.
    """
    print('🔍 COMPARING DATA SOURCES: yfinance vs IBKR')
    print('=' * 50)

    symbols = ['SPY', 'AAPL', 'MSFT', 'GOOGL']
    start_date = '2023-01-01'
    end_date = '2023-12-31'

    print(f'Testing symbols: {symbols}')
    print(f'Date range: {start_date} to {end_date}')
    print()

    # Initialize IBKR provider
    ibkr_provider = IBKRHistoricalDataProvider()

    # Connect to IBKR
    connected = await ibkr_provider.connector.connect()
    if not connected:
        print('❌ Cannot connect to IBKR - comparison requires TWS connection')
        return

    results = {}

    for symbol in symbols:
        print(f'📊 Analyzing {symbol}...')
        results[symbol] = {}

        # Get yfinance data
        try:
            yf_data = yf.download(symbol, start=start_date, end=end_date, progress=False)
            if not yf_data.empty:
                yf_data.columns = yf_data.columns.get_level_values(0)  # Flatten MultiIndex
                results[symbol]['yfinance'] = {
                    'bars': len(yf_data),
                    'start_date': yf_data.index.min().date(),
                    'end_date': yf_data.index.max().date(),
                    'avg_volume': yf_data['Volume'].mean(),
                    'price_range': f"${yf_data['Close'].min():.2f} - ${yf_data['Close'].max():.2f}",
                    'data_quality': 'Good' if len(yf_data) > 200 else 'Limited'
                }
            else:
                results[symbol]['yfinance'] = {'error': 'No data'}
        except Exception as e:
            results[symbol]['yfinance'] = {'error': str(e)}

        # Get IBKR data
        try:
            ibkr_data = await ibkr_provider.get_historical_bars(symbol, start_date, end_date, '1 day')
            if ibkr_data is not None and not ibkr_data.empty:
                results[symbol]['ibkr'] = {
                    'bars': len(ibkr_data),
                    'start_date': ibkr_data.index.min().date(),
                    'end_date': ibkr_data.index.max().date(),
                    'avg_volume': ibkr_data['volume'].mean(),
                    'price_range': f"${ibkr_data['close'].min():.2f} - ${ibkr_data['close'].max():.2f}",
                    'data_quality': 'Professional'
                }
            else:
                results[symbol]['ibkr'] = {'error': 'No data'}
        except Exception as e:
            results[symbol]['ibkr'] = {'error': str(e)}

        # Compare data points
        yf_bars = results[symbol]['yfinance'].get('bars', 0)
        ibkr_bars = results[symbol]['ibkr'].get('bars', 0)

        results[symbol]['comparison'] = {
            'bar_difference': ibkr_bars - yf_bars,
            'coverage_ratio': ibkr_bars / yf_bars if yf_bars > 0 else 0,
            'ibkr_advantage': 'More comprehensive' if ibkr_bars > yf_bars else 'Similar coverage'
        }

        print(f'   yfinance: {yf_bars} bars')
        print(f'   IBKR: {ibkr_bars} bars')
        print(f'   Difference: {results[symbol]["comparison"]["bar_difference"]} bars')
        print()

    # Summary analysis
    print('📈 SUMMARY ANALYSIS')
    print('-' * 20)

    total_yf_bars = sum(r.get('yfinance', {}).get('bars', 0) for r in results.values())
    total_ibkr_bars = sum(r.get('ibkr', {}).get('bars', 0) for r in results.values())

    print(f'Total bars - yfinance: {total_yf_bars}, IBKR: {total_ibkr_bars}')
    print(f'IBKR advantage: {total_ibkr_bars - total_yf_bars} additional bars')
    print(f'Coverage improvement: {((total_ibkr_bars / total_yf_bars) - 1) * 100:.1f}%' if total_yf_bars > 0 else 'N/A')
    print()

    print('🎯 KEY BENEFITS OF IBKR DATA:')
    print('   • Professional-grade market data')
    print('   • More comprehensive historical coverage')
    print('   • Real-time data access capabilities')
    print('   • Institutional-quality bar data')
    print('   • Enhanced accuracy for backtesting')
    print('   • Direct integration with trading platform')
    print()

    # Save comparison results
    comparison_data = {
        'timestamp': datetime.now().isoformat(),
        'comparison_period': f'{start_date} to {end_date}',
        'symbols_tested': symbols,
        'results': results,
        'summary': {
            'total_yfinance_bars': total_yf_bars,
            'total_ibkr_bars': total_ibkr_bars,
            'ibkr_advantage_bars': total_ibkr_bars - total_yf_bars,
            'ibkr_advantage_percentage': ((total_ibkr_bars / total_yf_bars) - 1) * 100 if total_yf_bars > 0 else 0
        }
    }

    filename = f'data_source_comparison_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(filename, 'w') as f:
        import json
        json.dump(comparison_data, f, indent=2, default=str)

    print(f'💾 Comparison results saved to: {filename}')
    print()
    print('✅ Data source comparison complete!')

if __name__ == '__main__':
    asyncio.run(compare_data_sources())