#!/usr/bin/env python3
"""
Test Environment Setup Script
Sets up test environments mirroring production for integration testing.
"""

import os
import sys
import subprocess
import argparse
import yaml
from pathlib import Path

def load_config(environment='test'):
    """Load environment configuration."""
    config_path = Path(__file__).parent.parent / 'config' / 'environments' / f'{environment}.yaml'
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def check_docker():
    """Check if Docker is available."""
    try:
        subprocess.run(['docker', '--version'], check=True, capture_output=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def check_docker_compose():
    """Check if Docker Compose is available."""
    try:
        subprocess.run(['docker-compose', '--version'], check=True, capture_output=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        # Try docker compose (new syntax)
        try:
            subprocess.run(['docker', 'compose', 'version'], check=True, capture_output=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False

def start_test_environment(config):
    """Start the test environment using Docker Compose."""
    compose_file = Path(__file__).parent / 'docker-compose.test.yml'
    project_dir = Path(__file__).parent.parent

    if not compose_file.exists():
        print(f"Error: Docker Compose file not found: {compose_file}")
        return False

    print("Starting test environment with Docker Compose...")

    # Use docker compose (new syntax) if available, otherwise docker-compose
    try:
        cmd = ['docker', 'compose', '-f', str(compose_file), 'up', '-d']
        subprocess.run(cmd, cwd=str(project_dir), check=True)
        print("✅ Test environment started successfully")
        return True
    except subprocess.CalledProcessError:
        try:
            cmd = ['docker-compose', '-f', str(compose_file), 'up', '-d']
            subprocess.run(cmd, cwd=str(project_dir), check=True)
            print("✅ Test environment started successfully")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to start test environment: {e}")
            return False

def stop_test_environment():
    """Stop the test environment."""
    compose_file = Path(__file__).parent / 'docker-compose.test.yml'
    project_dir = Path(__file__).parent.parent

    print("Stopping test environment...")

    try:
        cmd = ['docker', 'compose', '-f', str(compose_file), 'down']
        subprocess.run(cmd, cwd=str(project_dir), check=True)
    except subprocess.CalledProcessError:
        try:
            cmd = ['docker-compose', '-f', str(compose_file), 'down']
            subprocess.run(cmd, cwd=str(project_dir), check=True)
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to stop test environment: {e}")
            return False

    print("✅ Test environment stopped successfully")
    return True

def wait_for_services(config, timeout=120):
    """Wait for all services to be healthy."""
    import time
    import requests

    services = {
        'redis': f"redis-cli -h {config['redis']['host']} -p {config['redis']['port']} ping",
        'postgres': f"pg_isready -h {config['database']['host']} -p {config['database']['port']} -U {config['database']['user']} -d {config['database']['name']}",
        'mock_ibkr': f"nc -z {config['ibkr']['host']} {config['ibkr']['port']}",
        'health_monitor': f"curl -f http://{config['health']['host']}:{config['health']['port']}/health"
    }

    print("Waiting for services to be ready...")

    start_time = time.time()
    while time.time() - start_time < timeout:
        all_ready = True

        for service_name, check_cmd in services.items():
            try:
                if service_name == 'health_monitor':
                    # Use requests for HTTP check
                    response = requests.get(f"http://{config['health']['host']}:{config['health']['port']}/health", timeout=5)
                    if response.status_code not in [200, 503]:
                        all_ready = False
                else:
                    # Use subprocess for other checks
                    result = subprocess.run(check_cmd.split(), capture_output=True, timeout=5)
                    if result.returncode != 0:
                        all_ready = False
            except (subprocess.TimeoutExpired, requests.RequestException):
                all_ready = False

        if all_ready:
            print("✅ All services are ready!")
            return True

        time.sleep(5)

    print("❌ Timeout waiting for services to be ready")
    return False

def run_tests(config, test_args=None):
    """Run integration tests."""
    if test_args is None:
        test_args = []

    project_dir = Path(__file__).parent.parent

    print("Running integration tests...")

    cmd = [sys.executable, 'tools/run_integration_tests.py'] + test_args
    result = subprocess.run(cmd, cwd=str(project_dir))

    return result.returncode == 0

def main():
    parser = argparse.ArgumentParser(description='Set up ABC-Application test environment')
    parser.add_argument('--environment', '-e', default='test',
                       choices=['test', 'staging'],
                       help='Environment to set up')
    parser.add_argument('--action', '-a', default='start',
                       choices=['start', 'stop', 'restart', 'test'],
                       help='Action to perform')
    parser.add_argument('--no-wait', action='store_true',
                       help='Skip waiting for services to be ready')
    parser.add_argument('--test-args', nargs='*', default=[],
                       help='Additional arguments for test execution')

    args = parser.parse_args()

    # Load configuration
    try:
        config = load_config(args.environment)
    except FileNotFoundError:
        print(f"❌ Configuration file for environment '{args.environment}' not found")
        return 1

    # Check Docker availability
    if not check_docker():
        print("❌ Docker is not available. Please install Docker to use test environments.")
        return 1

    if not check_docker_compose():
        print("❌ Docker Compose is not available. Please install Docker Compose.")
        return 1

    if args.action == 'stop':
        return 0 if stop_test_environment() else 1

    elif args.action == 'start':
        if not start_test_environment(config):
            return 1

        if not args.no_wait:
            if not wait_for_services(config):
                return 1

        print("
Test environment is ready!"        print(f"  Redis: {config['redis']['host']}:{config['redis']['port']}")
        print(f"  PostgreSQL: {config['database']['host']}:{config['database']['port']}")
        print(f"  Mock IBKR: {config['ibkr']['host']}:{config['ibkr']['port']}")
        print(f"  Health Monitor: {config['health']['host']}:{config['health']['port']}")

        return 0

    elif args.action == 'restart':
        stop_test_environment()
        return main()  # Recursive call with start action

    elif args.action == 'test':
        # Ensure environment is running
        if not start_test_environment(config):
            return 1

        if not args.no_wait:
            if not wait_for_services(config):
                return 1

        # Run tests
        if run_tests(config, args.test_args):
            print("✅ All tests passed!")
            return 0
        else:
            print("❌ Some tests failed!")
            return 1

if __name__ == '__main__':
    sys.exit(main())</content>
</xai:function_call name="create_file">
<parameter name="filePath">c:\Users\nvick\ABC-Application\setup\test_environment_readme.md