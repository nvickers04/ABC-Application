#!/usr/bin/env python3
"""
Security scanning script for ABC Application
Runs automated security tests and vulnerability scanning
"""

import subprocess
import sys
import os
import json
from pathlib import Path
from datetime import datetime
import argparse

class SecurityScanner:
    """Comprehensive security scanner for the ABC Application"""

    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.results = {}
        self.passed = True

    def run_bandit_scan(self) -> dict:
        """Run Bandit static security analysis"""
        print("🔍 Running Bandit security scan...")

        config_file = self.project_root / "tests" / "security" / "bandit_config.yaml"

        if not config_file.exists():
            return {"error": "Bandit config file not found"}

        cmd = [
            sys.executable, "-m", "bandit",
            "-c", str(config_file),
            "-r", str(self.project_root / "src"),
            "--format", "json"
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

            if result.returncode in [0, 1]:  # 0 = no issues, 1 = issues found
                bandit_results = json.loads(result.stdout)
                issues = bandit_results.get("results", {})

                # Count issues by severity
                high_count = 0
                medium_count = 0
                low_count = 0

                for filename, file_issues in issues.items():
                    for issue in file_issues:
                        severity = issue.get("issue_severity", "UNKNOWN")
                        if severity == "HIGH":
                            high_count += 1
                        elif severity == "MEDIUM":
                            medium_count += 1
                        elif severity == "LOW":
                            low_count += 1

                return {
                    "status": "completed",
                    "high_severity": high_count,
                    "medium_severity": medium_count,
                    "low_severity": low_count,
                    "total_issues": high_count + medium_count + low_count,
                    "details": issues
                }
            else:
                return {"error": f"Bandit scan failed: {result.stderr}"}

        except subprocess.TimeoutExpired:
            return {"error": "Bandit scan timed out"}
        except Exception as e:
            return {"error": f"Bandit scan error: {str(e)}"}

    def run_safety_check(self) -> dict:
        """Check for known security vulnerabilities in dependencies"""
        print("🔍 Checking dependencies for vulnerabilities...")

        cmd = [sys.executable, "-m", "safety", "check", "--json"]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.project_root, timeout=120)

            if result.returncode == 0:
                safety_results = json.loads(result.stdout)
                vulnerabilities = safety_results.get("vulnerabilities", [])

                return {
                    "status": "completed",
                    "vulnerabilities_found": len(vulnerabilities),
                    "details": vulnerabilities
                }
            else:
                return {"error": f"Safety check failed: {result.stderr}"}

        except subprocess.TimeoutExpired:
            return {"error": "Safety check timed out"}
        except Exception as e:
            return {"error": f"Safety check error: {str(e)}"}

    def run_pytest_security(self) -> dict:
        """Run security-focused pytest tests"""
        print("🔍 Running security unit tests...")

        security_test_dir = self.project_root / "tests" / "security"

        if not security_test_dir.exists():
            return {"error": "Security tests directory not found"}

        cmd = [
            sys.executable, "-m", "pytest",
            str(security_test_dir),
            "--tb=short",
            "--disable-warnings",
            "-q",
            "--json-report",
            "--json-report-file", "security_test_results.json"
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.project_root, timeout=300)

            # Try to read the JSON results
            json_file = self.project_root / "security_test_results.json"
            test_results = {}

            if json_file.exists():
                with open(json_file, 'r') as f:
                    test_results = json.load(f)

                # Clean up
                json_file.unlink(missing_ok=True)

            return {
                "status": "completed",
                "return_code": result.returncode,
                "passed": result.returncode == 0,
                "output": result.stdout,
                "errors": result.stderr,
                "test_results": test_results
            }

        except subprocess.TimeoutExpired:
            return {"error": "Security tests timed out"}
        except Exception as e:
            return {"error": f"Security tests error: {str(e)}"}

    def check_file_permissions(self) -> dict:
        """Check file permissions for sensitive files"""
        print("🔍 Checking file permissions...")

        sensitive_files = [
            "config/ibkr_config.ini",
            "config/api_cost_reference.yaml",
            "vault-config/",
            "vault-data/",
            "ABCSSH",
            "ABCSSH.pub"
        ]

        issues = []

        for file_path in sensitive_files:
            full_path = self.project_root / file_path

            if full_path.exists():
                if os.name != 'nt':  # Unix-like systems
                    import stat
                    file_stat = os.stat(full_path)

                    # Check if world-readable
                    if file_stat.st_mode & stat.S_IROTH:
                        issues.append(f"{file_path} is world-readable")

                    # Check if executable when it shouldn't be
                    if file_stat.st_mode & stat.S_IXOTH and not file_path.endswith(('.sh', '.py', '.exe')):
                        issues.append(f"{file_path} is world-executable")
                else:
                    # Windows - basic check for hidden files
                    if not file_path.startswith('.') and 'password' in file_path.lower():
                        issues.append(f"Potentially sensitive file {file_path} may need protection")

        return {
            "status": "completed",
            "issues_found": len(issues),
            "details": issues
        }

    def check_hardcoded_secrets(self) -> dict:
        """Check for hardcoded secrets in source code"""
        print("🔍 Checking for hardcoded secrets...")

        secret_patterns = [
            r'api_key\s*=\s*["\'][^"\']*["\']',
            r'secret\s*=\s*["\'][^"\']*["\']',
            r'password\s*=\s*["\'][^"\']*["\']',
            r'token\s*=\s*["\'][^"\']*["\']',
            r'sk-\w+',  # OpenAI API keys
            r'xoxb-\w+',  # Slack tokens
            r'ghp_\w+',  # GitHub tokens
        ]

        issues = []

        # Scan Python files in src/
        src_dir = self.project_root / "src"
        if src_dir.exists():
            for py_file in src_dir.rglob("*.py"):
                try:
                    with open(py_file, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()

                        for pattern in secret_patterns:
                            import re
                            matches = re.findall(pattern, content, re.IGNORECASE)
                            if matches:
                                issues.append(f"Potential hardcoded secret in {py_file}: {matches[:3]}")  # Show first 3 matches

                except Exception as e:
                    issues.append(f"Error scanning {py_file}: {str(e)}")

        return {
            "status": "completed",
            "issues_found": len(issues),
            "details": issues
        }

    def run_all_scans(self) -> dict:
        """Run all security scans"""
        print("🚀 Starting comprehensive security scan...")
        print("=" * 60)

        self.results = {
            "timestamp": datetime.now().isoformat(),
            "scans": {}
        }

        # Run all scans
        scans = [
            ("bandit", self.run_bandit_scan),
            ("safety", self.run_safety_check),
            ("pytest_security", self.run_pytest_security),
            ("file_permissions", self.check_file_permissions),
            ("hardcoded_secrets", self.check_hardcoded_secrets)
        ]

        for scan_name, scan_func in scans:
            print(f"\n📋 Running {scan_name} scan...")
            try:
                result = scan_func()
                self.results["scans"][scan_name] = result

                if "error" in result:
                    print(f"❌ {scan_name}: ERROR - {result['error']}")
                    self.passed = False
                elif scan_name == "bandit" and result.get("high_severity", 0) > 0:
                    print(f"❌ {scan_name}: FAILED - {result['high_severity']} high severity issues")
                    self.passed = False
                elif scan_name == "safety" and result.get("vulnerabilities_found", 0) > 0:
                    print(f"❌ {scan_name}: FAILED - {result['vulnerabilities_found']} vulnerabilities")
                    self.passed = False
                elif scan_name == "pytest_security" and not result.get("passed", False):
                    print(f"❌ {scan_name}: FAILED - tests did not pass")
                    self.passed = False
                elif scan_name in ["file_permissions", "hardcoded_secrets"] and result.get("issues_found", 0) > 0:
                    print(f"❌ {scan_name}: FAILED - {result['issues_found']} issues found")
                    self.passed = False
                else:
                    print(f"✅ {scan_name}: PASSED")

            except Exception as e:
                print(f"❌ {scan_name}: EXCEPTION - {str(e)}")
                self.results["scans"][scan_name] = {"error": str(e)}
                self.passed = False

        self.results["overall_status"] = "PASSED" if self.passed else "FAILED"

        print("\n" + "=" * 60)
        print(f"🎯 Security Scan Result: {self.results['overall_status']}")

        return self.results

    def save_results(self, output_file: str = None):
        """Save scan results to file"""
        if not output_file:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"security_scan_results_{timestamp}.json"

        output_path = self.project_root / output_file

        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2)

        print(f"📄 Results saved to: {output_path}")
        return str(output_path)

def main():
    parser = argparse.ArgumentParser(description="ABC Application Security Scanner")
    parser.add_argument("--output", "-o", help="Output file for results")
    parser.add_argument("--project-root", "-p", default=".", help="Project root directory")

    args = parser.parse_args()

    scanner = SecurityScanner(args.project_root)
    results = scanner.run_all_scans()
    output_file = scanner.save_results(args.output)

    # Exit with appropriate code
    sys.exit(0 if scanner.passed else 1)

if __name__ == "__main__":
    main()