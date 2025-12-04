@echo off
REM Unified Workflow Starter Batch Script
REM Provides easy Windows interface to run the Unified Workflow Orchestrator

echo 🚀 Starting Unified Workflow Orchestrator
echo.

REM Check if virtual environment exists
if not exist "%~dp0..\myenv\Scripts\activate.bat" (
    echo ❌ Virtual environment not found at %~dp0..\myenv
    echo Please run setup scripts first
    pause
    exit /b 1
)

REM Activate virtual environment
call "%~dp0..\myenv\Scripts\activate.bat"

REM Change to project root
cd "%~dp0.."

REM Run the unified workflow starter
python tools\start_unified_workflow.py %*

REM Deactivate virtual environment
call "%~dp0..\myenv\Scripts\deactivate.bat"

echo.
echo 🛑 Unified Workflow Orchestrator stopped
pause