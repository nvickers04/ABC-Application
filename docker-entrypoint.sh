#!/bin/bash
# Docker entrypoint script for ABC Application

echo "🚀 Starting ABC Application..."

# Wait for Redis to be ready
echo "⏳ Waiting for Redis to be ready..."
while ! redis-cli -h redis ping > /dev/null 2>&1; do
    echo "Redis not ready, waiting..."
    sleep 2
done

echo "✅ Redis is ready!"

# Run the application
cd /app
python -c "
import asyncio
import sys
sys.path.insert(0, '.')
from src.agents.live_workflow_orchestrator import LiveWorkflowOrchestrator

async def main():
    print('🚀 Starting Live Workflow Orchestrator...')
    orchestrator = LiveWorkflowOrchestrator()
    await orchestrator.run_orchestrator()

asyncio.run(main())
"