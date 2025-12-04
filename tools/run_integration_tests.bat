@echo off
REM Automated Integration Test Runner for ABC-Application (Windows)
REM Runs integration tests with proper setup and teardown

echo ABC-Application Integration Test Runner
echo ======================================

REM Check if Redis is running
echo Checking Redis...
redis-cli ping >nul 2>&1
if %errorlevel% neq 0 (
    echo Warning: Redis not running. Some tests may be skipped.
)

REM Check Python environment
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: Python not found
    exit /b 1
)

REM Set PYTHONPATH
set PYTHONPATH=%~dp0..;%PYTHONPATH%

REM Run integration tests
echo Running integration tests...
python tools\run_integration_tests.py %*

REM Check result
if %errorlevel% equ 0 (
    echo.
    echo ^✅ All integration tests passed!
) else (
    echo.
    echo ^❌ Integration tests failed with exit code %errorlevel%
)

exit /b %errorlevel%