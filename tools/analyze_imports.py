#!/usr/bin/env python3
"""
Import Dependencies Analyzer for ABC-Application
Analyzes import statements across the codebase for consistency and issues.
"""

import os
import sys
import ast
import re
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Set, Tuple

class ImportAnalyzer:
    """Analyzes Python import statements in a codebase."""

    def __init__(self, root_path: str):
        self.root_path = Path(root_path)
        self.imports_by_file: Dict[str, List[Tuple[str, str]]] = {}
        self.all_imports: Set[str] = set()
        self.standard_libs: Set[str] = self._get_standard_libs()
        self.third_party_imports: Counter = Counter()
        self.local_imports: Counter = Counter()
        self.unused_imports: Dict[str, List[str]] = {}

    def _get_standard_libs(self) -> Set[str]:
        """Get Python standard library modules."""
        # Common stdlib modules
        return {
            'sys', 'os', 'ast', 're', 'collections', 'typing', 'pathlib',
            'tempfile', 'subprocess', 'argparse', 'json', 'logging', 'asyncio',
            'threading', 'socket', 'time', 'datetime', 'uuid', 'math', 'random',
            'string', 'hashlib', 'base64', 'urllib', 'http', 'email', 'xml',
            'html', 'csv', 'configparser', 'sqlite3', 'gzip', 'zipfile', 'tarfile',
            'shutil', 'glob', 'fnmatch', 'linecache', 'pickle', 'copyreg', 'copy',
            'pprint', 'reprlib', 'enum', 'numbers', 'functools', 'operator',
            'itertools', 'contextlib', 'warnings', 'weakref', 'gc', 'inspect',
            'site', 'sysconfig', 'platform', 'errno', 'ctypes', 'mmap',
            'contextvars', 'concurrent', 'multiprocessing', 'queue', 'sched',
            'abc', 'io', 'codecs', 'unicodedata', 'locale', 'calendar',
            'optparse', 'getopt', 'readline', 'rlcompleter', 'sqlite3',
            'zlib', 'bz2', 'lzma', 'zipimport', 'pkgutil', 'runpy', 'importlib',
            'imp', 'importlib', 'zipimport', 'pkgutil', 'modulefinder',
            'runpy', 'pkg_resources', 'setuptools'
        }

    def analyze_file(self, file_path: Path) -> None:
        """Analyze imports in a single Python file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            tree = ast.parse(content, filename=str(file_path))
            imports = []

            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append(('import', alias.name))
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ''
                    for alias in node.names:
                        imports.append(('from', f"{module}.{alias.name}" if module else alias.name))

            self.imports_by_file[str(file_path)] = imports

            for import_type, import_name in imports:
                self.all_imports.add(import_name)
                base_module = import_name.split('.')[0]

                if base_module in self.standard_libs:
                    continue  # Skip stdlib
                elif import_name.startswith(('src.', 'tools.', 'integration_tests.', 'unit_tests.')):
                    self.local_imports[base_module] += 1
                else:
                    self.third_party_imports[base_module] += 1

        except Exception as e:
            print(f"Error analyzing {file_path}: {e}")

    def analyze_codebase(self) -> None:
        """Analyze all Python files in the codebase."""
        for py_file in self.root_path.rglob('*.py'):
            if not any(skip in str(py_file) for skip in ['__pycache__', '.git', 'myenv', 'htmlcov']):
                self.analyze_file(py_file)

    def check_import_consistency(self) -> Dict[str, List[str]]:
        """Check for import consistency issues."""
        issues = defaultdict(list)

        # Check for mixed import styles
        for file_path, imports in self.imports_by_file.items():
            stdlib_imports = []
            third_party_imports = []
            local_imports = []

            for import_type, import_name in imports:
                base_module = import_name.split('.')[0]

                if base_module in self.standard_libs:
                    stdlib_imports.append((import_type, import_name))
                elif import_name.startswith(('src.', 'tools.')):
                    local_imports.append((import_type, import_name))
                else:
                    third_party_imports.append((import_type, import_name))

            # Check ordering (stdlib, third-party, local)
            all_imports_ordered = stdlib_imports + third_party_imports + local_imports
            if imports != all_imports_ordered:
                issues['import_order'].append(file_path)

        return dict(issues)

    def generate_report(self) -> str:
        """Generate analysis report."""
        report = []
        report.append("# Import Dependencies Analysis Report")
        report.append("=" * 50)
        report.append("")

        # Summary
        report.append("## Summary")
        report.append(f"- Total Python files analyzed: {len(self.imports_by_file)}")
        report.append(f"- Total unique imports: {len(self.all_imports)}")
        report.append(f"- Third-party packages: {len(self.third_party_imports)}")
        report.append(f"- Local modules: {len(self.local_imports)}")
        report.append("")

        # Third-party dependencies
        report.append("## Third-Party Dependencies")
        for package, count in sorted(self.third_party_imports.items()):
            report.append(f"- {package}: {count} imports")
        report.append("")

        # Local imports
        report.append("## Local Module Imports")
        for module, count in sorted(self.local_imports.items()):
            report.append(f"- {module}: {count} imports")
        report.append("")

        # Import consistency issues
        consistency_issues = self.check_import_consistency()
        if consistency_issues:
            report.append("## Import Consistency Issues")
            for issue_type, files in consistency_issues.items():
                report.append(f"### {issue_type.replace('_', ' ').title()}")
                for file_path in files:
                    report.append(f"- {file_path}")
                report.append("")
        else:
            report.append("## Import Consistency: ✅ All Good")
            report.append("No import ordering issues found.")
            report.append("")

        # Files with most imports
        report.append("## Files with Most Imports")
        file_import_counts = [(file, len(imports)) for file, imports in self.imports_by_file.items()]
        file_import_counts.sort(key=lambda x: x[1], reverse=True)

        for file_path, count in file_import_counts[:10]:
            report.append(f"- {file_path}: {count} imports")
        report.append("")

        return "\n".join(report)

def main():
    """Main entry point."""
    if len(sys.argv) != 2:
        print("Usage: python analyze_imports.py <project_root>")
        sys.exit(1)

    project_root = sys.argv[1]

    analyzer = ImportAnalyzer(project_root)
    print("Analyzing import dependencies...")
    analyzer.analyze_codebase()

    report = analyzer.generate_report()
    print(report)

    # Save report
    report_file = Path(project_root) / "docs" / "import_analysis_report.md"
    report_file.parent.mkdir(exist_ok=True)
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"\nReport saved to: {report_file}")

if __name__ == '__main__':
    main()