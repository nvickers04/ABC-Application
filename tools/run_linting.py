#!/usr/bin/env python3
"""
Import and Code Linting Script for ABC-Application
Runs isort, flake8, and other linting tools to ensure code quality.
"""

import sys
import subprocess
import argparse
from pathlib import Path

def run_command(cmd, description):
    """Run a command and return success status."""
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ {description} passed")
            return True
        else:
            print(f"❌ {description} failed")
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            return False
    except Exception as e:
        print(f"❌ Error running {description}: {e}")
        return False

def check_imports():
    """Check import sorting and consistency."""
    print("🔍 Checking import sorting with isort...")

    # Check if isort is available
    try:
        import isort
    except ImportError:
        print("❌ isort not installed. Install with: pip install isort")
        return False

    # Run isort check
    cmd = [sys.executable, "-m", "isort", "--check-only", "--diff", "src/", "tools/", "integration-tests/"]
    return run_command(cmd, "isort import sorting check")

def fix_imports():
    """Fix import sorting."""
    print("🔧 Fixing import sorting with isort...")

    try:
        import isort
    except ImportError:
        print("❌ isort not installed. Install with: pip install isort")
        return False

    # Run isort fix
    cmd = [sys.executable, "-m", "isort", "src/", "tools/", "integration-tests/"]
    return run_command(cmd, "isort import sorting fix")

def check_flake8():
    """Check code style with flake8."""
    print("🔍 Checking code style with flake8...")

    try:
        import flake8
    except ImportError:
        print("❌ flake8 not installed. Install with: pip install flake8")
        return False

    # Run flake8
    cmd = [sys.executable, "-m", "flake8", "src/", "tools/", "integration-tests/"]
    return run_command(cmd, "flake8 code style check")

def check_unused_imports():
    """Check for unused imports using pylint or similar."""
    print("🔍 Checking for unused imports...")

    # This would require more advanced tools like pylint or vulture
    # For now, we'll use a simple approach
    print("ℹ️  Advanced unused import detection requires pylint or vulture")
    print("   Consider installing: pip install pylint vulture")
    return True

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Run code linting for ABC-Application')
    parser.add_argument('--fix', action='store_true', help='Automatically fix issues where possible')
    parser.add_argument('--imports-only', action='store_true', help='Only check import-related issues')
    parser.add_argument('--strict', action='store_true', help='Fail on any warnings')

    args = parser.parse_args()

    print("ABC-Application Code Linting")
    print("=" * 40)

    success = True

    # Check imports
    if not check_imports():
        if args.fix:
            print("Attempting to fix import issues...")
            if not fix_imports():
                success = False
        else:
            success = False

    if not args.imports_only:
        # Check code style
        if not check_flake8():
            success = False

        # Check unused imports
        if not check_unused_imports():
            success = False

    if success:
        print("\n✅ All linting checks passed!")
        return 0
    else:
        print("\n❌ Some linting checks failed!")
        print("Run with --fix to automatically fix import sorting issues.")
        return 1

if __name__ == '__main__':
    sys.exit(main())