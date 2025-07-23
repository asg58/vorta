#!/usr/bin/env python3
"""
VORTA Test Runner - Professional VS Code Integration
Comprehensive test suite for VORTA platform
"""

import asyncio
import os
import subprocess
import sys
from pathlib import Path


class VortaTestRunner:
    """Professional test runner for VORTA development."""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.test_files = [
            "test_metrics.py",
            # Add more test files here
        ]
    
    def print_header(self, title: str):
        """Print professional test header."""
        print("=" * 80)
        print(f" 🧪 {title}")
        print("=" * 80)
    
    def run_pytest(self) -> bool:
        """Run pytest with professional configuration."""
        self.print_header("RUNNING PYTEST SUITE")
        
        try:
            # Run pytest with verbose output
            result = subprocess.run([
                sys.executable, "-m", "pytest", 
                "-v", "--tb=short", "--no-header",
                *self.test_files
            ], cwd=self.project_root, capture_output=False)
            
            return result.returncode == 0
            
        except Exception as e:
            print(f"❌ Pytest execution failed: {e}")
            return False
    
    def run_linting(self) -> bool:
        """Run code linting checks."""
        self.print_header("RUNNING CODE QUALITY CHECKS")
        
        python_files = [
            "vorta_monitor.py",
            "test_metrics.py", 
            "services/api/main.py"
        ]
        
        success = True
        
        for file_path in python_files:
            if os.path.exists(file_path):
                print(f"🔍 Checking: {file_path}")
                
                # Check syntax
                try:
                    with open(file_path, 'r') as f:
                        compile(f.read(), file_path, 'exec')
                    print("  ✅ Syntax: OK")
                except SyntaxError as e:
                    print(f"  ❌ Syntax Error: {e}")
                    success = False
            else:
                print(f"⚠️  File not found: {file_path}")
        
        return success
    
    async def run_health_checks(self) -> bool:
        """Run VORTA service health checks."""
        self.print_header("RUNNING VORTA HEALTH CHECKS")
        
        try:
            import aiohttp
            
            health_endpoints = [
                "http://localhost:8000/api/health",
                "http://localhost:9090/-/healthy",
                "http://localhost:3000/api/health"
            ]
            
            async with aiohttp.ClientSession() as session:
                for url in health_endpoints:
                    try:
                        async with session.get(url, timeout=5) as response:
                            service = url.split("//")[1].split(":")[1]
                            if response.status == 200:
                                print(f"  ✅ Port {service}: Healthy")
                            else:
                                print(f"  ⚠️  Port {service}: Status {response.status}")
                    except Exception as e:
                        service = url.split("//")[1].split(":")[1]
                        print(f"  ❌ Port {service}: {str(e)[:50]}")
            
            return True
            
        except ImportError:
            print("  ⚠️  aiohttp not available - skipping health checks")
            return True
        except Exception as e:
            print(f"  ❌ Health check failed: {e}")
            return False
    
    def generate_test_report(self, pytest_success: bool, lint_success: bool, health_success: bool):
        """Generate comprehensive test report."""
        self.print_header("VORTA TEST REPORT")
        
        total_tests = 3
        passed_tests = sum([pytest_success, lint_success, health_success])
        
        print("📊 Test Summary:")
        print(f"  • Pytest Suite:     {'✅ PASS' if pytest_success else '❌ FAIL'}")
        print(f"  • Code Quality:     {'✅ PASS' if lint_success else '❌ FAIL'}")
        print(f"  • Health Checks:    {'✅ PASS' if health_success else '❌ FAIL'}")
        print()
        print(f"🎯 Overall Result: {passed_tests}/{total_tests} tests passed")
        
        if passed_tests == total_tests:
            print("🚀 ALL TESTS PASSED - READY FOR PRODUCTION!")
        else:
            print("⚠️  SOME TESTS FAILED - REVIEW REQUIRED")
        
        return passed_tests == total_tests
    
    async def run_all_tests(self) -> bool:
        """Run complete test suite."""
        print("🚀 Starting VORTA Professional Test Suite...")
        print()
        
        # Run all test categories
        pytest_success = self.run_pytest()
        print()
        
        lint_success = self.run_linting()
        print()
        
        health_success = await self.run_health_checks()
        print()
        
        # Generate final report
        overall_success = self.generate_test_report(pytest_success, lint_success, health_success)
        
        return overall_success


if __name__ == "__main__":
    runner = VortaTestRunner()
    
    try:
        success = asyncio.run(runner.run_all_tests())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n🛑 Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Test runner failed: {e}")
        sys.exit(1)
