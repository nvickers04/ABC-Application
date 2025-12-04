#!/usr/bin/env python3
"""
Automated Integration Test Runner for ABC-Application
Runs integration tests with proper setup and teardown.
"""

import sys
import os
import subprocess
import argparse
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def check_dependencies():
    """Check if required dependencies are available."""
    required_services = {
        'redis': check_redis,
        'health_server': check_health_server
    }

    missing = []
    for service, check_func in required_services.items():
        if not check_func():
            missing.append(service)

    if missing:
        print(f"Warning: Missing services: {', '.join(missing)}")
        print("Some integration tests may be skipped.")

    return len(missing) == 0

def check_redis():
    """Check if Redis is running."""
    try:
        import redis
        r = redis.Redis(host='localhost', port=6379, db=0)
        r.ping()
        return True
    except:
        return False

def check_health_server():
    """Check if health server is running."""
    try:
        import requests
        response = requests.get('http://localhost:8080/health', timeout=5)
        return response.status_code in [200, 503]
    except:
        return False

def start_health_server():
    """Start the health server for testing."""
    print("Starting health server...")
    import subprocess
    import signal
    import atexit

    cmd = [sys.executable, str(project_root / 'tools' / 'health_server.py'), '--host', '127.0.0.1', '--port', '8080']
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # Wait for server to start
    time.sleep(2)

    def cleanup():
        print("Stopping health server...")
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()

    atexit.register(cleanup)
    return process

def run_pytest(args):
    """Run pytest with given arguments."""
    cmd = [sys.executable, '-m', 'pytest'] + args
    print(f"Running: {' '.join(cmd)}")

    result = subprocess.run(cmd, cwd=str(project_root))
    return result.returncode

def main():
    parser = argparse.ArgumentParser(description='Run ABC-Application integration tests')
    parser.add_argument('--ibkr', action='store_true', help='Run IBKR integration tests')
    parser.add_argument('--health-server', action='store_true', help='Start health server for testing')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--coverage', action='store_true', help='Generate coverage report')
    parser.add_argument('--parallel', action='store_true', help='Run tests in parallel')
    parser.add_argument('--fail-fast', action='store_true', help='Stop on first failure')

    args, unknown = parser.parse_known_args()

    print("ABC-Application Integration Test Runner")
    print("=" * 50)

    # Check dependencies
    deps_ok = check_dependencies()

    # Start health server if requested
    health_process = None
    if args.health_server:
        health_process = start_health_server()

    try:
        # Build pytest arguments
        pytest_args = ['integration-tests/']

        if args.ibkr:
            pytest_args.extend(['--run-ibkr-tests'])

        if args.verbose:
            pytest_args.append('-v')
            pytest_args.append('-s')

        if args.coverage:
            pytest_args.extend(['--cov=src', '--cov-report=html', '--cov-report=term'])

        if args.parallel:
            pytest_args.extend(['-n', 'auto'])

        if args.fail_fast:
            pytest_args.append('--tb=short')
        else:
            pytest_args.append('--tb=long')

        # Add any additional arguments
        pytest_args.extend(unknown)

        # Run tests
        print(f"Running integration tests...")
        exit_code = run_pytest(pytest_args)

        if exit_code == 0:
            print("✅ All integration tests passed!")
        else:
            print(f"❌ Integration tests failed with exit code {exit_code}")

        return exit_code

    finally:
        # Cleanup
        if health_process:
            print("Cleaning up health server...")
            health_process.terminate()
            try:
                health_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                health_process.kill()

if __name__ == '__main__':
    sys.exit(main())</content>
</xai:function_call name="create_file">
<parameter name="filePath">c:\Users\nvick\ABC-Application\.github\workflows\integration-tests.yml