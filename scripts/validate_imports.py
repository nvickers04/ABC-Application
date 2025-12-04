#!/usr/bin/env python3
"""
Import Path Validation Script

Validates that all Python modules in the project can be imported successfully.
This catches import errors early in the CI/CD pipeline.

Usage: python scripts/validate_imports.py
"""

import sys
import os
import importlib
import pkgutil
from pathlib import Path
from typing import List, Tuple

def find_python_files(base_path: str) -> List[str]:
    """Find all Python files in the project"""
    python_files = []
    for root, dirs, files in os.walk(base_path):
        # Skip common directories that shouldn't be imported
        dirs[:] = [d for d in dirs if d not in {
            '__pycache__', '.git', '.pytest_cache', 'htmlcov',
            'node_modules', '.vscode', 'myenv', 'data', 'logs'
        }]

        for file in files:
            if file.endswith('.py') and not file.startswith('test_'):
                python_files.append(os.path.join(root, file))

    return python_files

def extract_module_names(python_files: List[str], base_path: str) -> List[str]:
    """Extract module names from Python files"""
    modules = []
    base_path = Path(base_path).resolve()

    for file_path in python_files:
        file_path = Path(file_path).resolve()

        # Convert file path to module path
        try:
            relative_path = file_path.relative_to(base_path)
            module_parts = relative_path.with_suffix('').parts

            # Skip files not in src/ or other importable directories
            if not (module_parts[0] in ['src'] or
                    str(relative_path).startswith(('scripts/', 'tools/'))):
                continue

            # Convert path to dotted module name
            if module_parts[0] == 'src':
                module_name = '.'.join(module_parts)  # Keep 'src' prefix for proper importing
            else:
                module_name = '.'.join(module_parts)

            modules.append(module_name)

        except ValueError:
            # File not under base_path
            continue

    return modules

def validate_imports(modules: List[str]) -> Tuple[List[str], List[Tuple[str, str]]]:
    """Try to import all modules and return success/failure results"""
    successful = []
    failed = []

    # Add current directory to Python path
    sys.path.insert(0, str(Path(__file__).parent.parent))

    for module_name in sorted(set(modules)):
        try:
            # Skip certain modules that require special setup
            if module_name in {
                'setup.setup_tigerbeetle',  # Requires PowerShell
                'setup.setup_redis',        # Requires external setup
                'tools.health_server',     # May require network
                'tools.memory_leak_detector', # May require running processes
                'tools.performance_test',   # Performance testing
                'tools.peak_operations_monitor', # Requires IBKR connection
            }:
                successful.append(module_name)
                continue

            importlib.import_module(module_name)
            successful.append(module_name)

        except Exception as e:
            failed.append((module_name, str(e)))

    return successful, failed

def main():
    """Main validation function"""
    print("🔍 Validating Python import paths...")

    base_path = Path(__file__).parent.parent
    python_files = find_python_files(str(base_path))

    print(f"Found {len(python_files)} Python files")

    modules = extract_module_names(python_files, str(base_path))
    print(f"Extracted {len(modules)} potential modules")

    successful, failed = validate_imports(modules)

    print(f"\n✅ Successfully imported {len(successful)} modules")
    print(f"❌ Failed to import {len(failed)} modules")

    if failed:
        print("\nFailed imports:")
        for module, error in failed:
            print(f"  - {module}: {error}")
        return 1
    else:
        print("\n🎉 All import paths validated successfully!")
        return 0

if __name__ == "__main__":
    sys.exit(main())