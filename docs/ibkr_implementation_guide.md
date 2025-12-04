# IBKR Implementation Guide

## Overview

This document outlines the IBKR (Interactive Brokers) integration architecture and implementation choices for the ABC-Application trading system.

## Architecture Decision

**Primary Implementation**: Direct IBKR Connector (`src/integrations/ibkr_connector.py`)

**Rationale**: After comprehensive evaluation, the Direct Connector was chosen over the Bridge implementation due to:
- **Performance**: 43x faster initialization (0.19s vs 8+ seconds)
- **Simplicity**: Lower complexity and maintenance overhead
- **Reliability**: Proven stability with fewer failure modes
- **Sufficiency**: Meets all current trading requirements

## Implementation Choices

### 1. Connection Management
- **Choice**: Singleton pattern with connection pooling
- **Rationale**: Ensures single connection to IBKR API, prevents connection conflicts
- **Implementation**: Thread-safe singleton in `IBKRConnector` class

### 2. Error Handling Strategy
- **Choice**: Hierarchical exception classes with circuit breaker pattern
- **Rationale**: Provides robust error recovery and prevents cascade failures
- **Implementation**: Custom exceptions in `src/utils/exceptions.py` with automatic retry logic

### 3. Data Synchronization
- **Choice**: Event-driven updates with caching
- **Rationale**: Balances real-time data needs with performance
- **Implementation**: Async event handlers for market data and account updates

### 4. Order Management
- **Choice**: State-based order tracking with validation
- **Rationale**: Ensures order integrity and provides audit trail
- **Implementation**: Order state machine with pre-trade validation

## Usage Patterns

### Basic Connection
```python
from src.integrations.ibkr_connector import IBKRConnector

# Get singleton instance
connector = IBKRConnector()

# Connect to IBKR (async)
await connector.connect()

# Check connection status
status = connector.get_connection_status()
```

### Market Data Retrieval
```python
# Get real-time market data
market_data = await connector.get_market_data('AAPL', bar_size='1 min', duration='1 D')

# Data structure:
{
    'symbol': 'AAPL',
    'timestamp': '2025-12-04T10:30:00Z',
    'open': 150.25,
    'high': 151.00,
    'low': 149.75,
    'close': 150.80,
    'volume': 1250000
}
```

### Account Information
```python
# Get account summary
account = await connector.get_account_summary()

# Data structure:
{
    'cash': 50000.00,
    'total_value': 75000.00,
    'buying_power': 100000.00,
    'maintenance_margin': 25000.00
}
```

### Position Tracking
```python
# Get current positions
positions = await connector.get_positions()

# Data structure:
[
    {
        'symbol': 'AAPL',
        'quantity': 100,
        'avg_cost': 145.50,
        'current_price': 150.25,
        'unrealized_pnl': 475.00
    }
]
```

### Order Placement
```python
# Place market order
order = {
    'symbol': 'AAPL',
    'quantity': 100,
    'action': 'BUY',
    'order_type': 'MKT'
}

result = await connector.place_order(order)

# Result structure:
{
    'order_id': '123456789',
    'status': 'submitted',
    'timestamp': '2025-12-04T10:30:00Z'
}
```

### Order Status Checking
```python
# Check order status
status = await connector.get_order_status('123456789')

# Status structure:
{
    'order_id': '123456789',
    'status': 'filled',  # submitted, filled, cancelled, rejected
    'filled_quantity': 100,
    'avg_fill_price': 150.25,
    'timestamp': '2025-12-04T10:30:05Z'
}
```

## Configuration

### Environment Variables
```bash
# IBKR Connection Settings
IBKR_HOST=127.0.0.1
IBKR_PORT=7497
IBKR_CLIENT_ID=1

# Account Settings
IBKR_ACCOUNT_ID=DU1234567  # Paper trading account

# Risk Settings
MAX_ORDER_SIZE=1000
MAX_PORTFOLIO_RISK=0.05
```

### Configuration Files
- `config/ibkr_config.ini`: IBKR-specific settings
- `config/risk_config.yaml`: Risk management parameters
- `config/trading_permissions.yaml`: Trading constraints

## Error Handling Patterns

### Connection Errors
```python
try:
    await connector.connect()
except IBKRConnectionError as e:
    logger.error(f"IBKR connection failed: {e}")
    # Implement retry logic or fallback
```

### Order Errors
```python
try:
    result = await connector.place_order(order)
except OrderError as e:
    logger.error(f"Order failed: {e}")
    # Check order parameters or account status
```

### Market Data Errors
```python
try:
    data = await connector.get_market_data('AAPL')
except MarketDataError as e:
    logger.error(f"Market data error: {e}")
    # Use cached data or alternative source
```

## Circuit Breaker Pattern

The system implements automatic circuit breakers to prevent cascade failures:

- **Connection Failures**: Automatic retry with exponential backoff
- **Rate Limiting**: Prevents API abuse with request throttling
- **Account Alerts**: Monitors for margin calls or account restrictions

## Testing Strategy

### Unit Tests
- Mock IBKR connections for isolated testing
- Test error scenarios and edge cases
- Validate data transformations

### Integration Tests
- Test complete Data → Strategy → Risk → Execution workflow
- Validate component interactions
- Performance benchmarking

### Paper Trading Validation
- End-to-end testing with paper trading account
- Realistic market conditions simulation
- Performance monitoring and optimization

## Monitoring and Observability

### Key Metrics
- Connection uptime and latency
- Order execution success rate
- API call frequency and errors
- Account balance and P&L tracking

### Logging
- Structured logging with correlation IDs
- Error tracking with context
- Performance metrics collection

### Alerts
- Connection failures
- Order rejections
- Account margin alerts
- System performance degradation

## Maintenance Guidelines

### Regular Tasks
- Monitor IBKR API changes and update accordingly
- Review and update risk parameters quarterly
- Validate backup and recovery procedures monthly

### Performance Optimization
- Profile API call patterns
- Optimize data caching strategies
- Monitor memory usage and connection pooling

### Security Updates
- Keep IBKR API client updated
- Review account permissions regularly
- Monitor for security vulnerabilities

## Troubleshooting

### Common Issues

**Connection Refused**
- Verify TWS/Gateway is running
- Check firewall settings
- Validate API permissions

**Order Rejection**
- Check account permissions
- Verify market hours
- Validate order parameters

**Data Delays**
- Check network connectivity
- Verify market data subscriptions
- Monitor API rate limits

### Debug Mode
Enable debug logging for detailed troubleshooting:
```python
import logging
logging.getLogger('ibkr_connector').setLevel(logging.DEBUG)
```

## Future Considerations

### Bridge Implementation
The Bridge implementation (`nautilus_ibkr_bridge.py`) provides enhanced features:
- Advanced risk management
- Portfolio analytics
- Multi-asset support

Consider migration when:
- Advanced risk features are required
- Performance overhead becomes acceptable
- Enhanced monitoring needs grow

### API Evolution
Monitor IBKR API changes and plan updates accordingly. The Direct Connector architecture allows for easier adaptation to API changes.

## Conclusion

The Direct IBKR Connector provides a robust, performant foundation for trading operations. The implementation choices prioritize reliability, maintainability, and performance while providing comprehensive error handling and monitoring capabilities.