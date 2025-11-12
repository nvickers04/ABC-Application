@echo off
REM ABC Application Continuous Trading Launcher
REM Runs the AI trading system continuously during market hours

echo 🚀 Starting ABC Application Continuous Paper Trading System
echo ========================================================
echo.

REM Set Python path
set PYTHONPATH=%~dp0

REM Check if TWS is running
echo 📊 Checking IBKR TWS connection...
python -c "
import asyncio
from ib_insync import IB
import time

async def check_connection():
    ib = IB()
    try:
        ib.connect('127.0.0.1', 7497, clientId=1)
        timeout = 10
        for i in range(timeout):
            if ib.isConnected():
                accounts = ib.managedAccounts()
                print(f'✅ TWS Connected - Account: {accounts[0] if accounts else \"Unknown\"}')
                ib.disconnect()
                return True
            time.sleep(1)
        print('❌ TWS not responding')
        return False
    except Exception as e:
        print(f'❌ Connection failed: {e}')
        return False

result = asyncio.run(check_connection())
if not result:
    echo.
    echo ❌ TWS connection failed. Please ensure:
    echo    1. IBKR Trader Workstation is running
    echo    2. API is enabled ^(File -^> Global Configuration -^> API^)
    echo    3. Socket port is 7497
    echo    4. You are logged into your paper trading account
    echo.
    pause
    exit /b 1
"

if errorlevel 1 goto :error

echo.
echo ✅ TWS connection verified
echo 🔄 Starting continuous trading system...
echo.
echo Press Ctrl+C to stop the system gracefully
echo.

REM Start the continuous trading system
python continuous_trading.py

goto :end

:error
echo.
echo ❌ Failed to start continuous trading system
echo Please check the error messages above
pause

:end
echo.
echo 🏁 Continuous trading system stopped
echo Check continuous_trading.log for detailed logs
pause