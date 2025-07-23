#!/usr/bin/env python3
"""
VORTA Quick Check Tool
=====================

🔍 Quick validation tool voor alle VORTA services en tools
Professional VS Code integration met intelligent checks

Author: VORTA Development Team
Version: 2.0.0 - Enterprise Edition
"""

import subprocess
import time
from datetime import datetime
from pathlib import Path

import requests


class VortaQuickCheck:
    def __init__(self):
        self.start_time = time.time()
        self.checks = []
        
    def print_header(self):
        """Print professional header"""
        print("=" * 80)
        print("🔍 VORTA QUICK CHECK - Intelligent Validation")
        print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Enterprise Tools")
        print("=" * 80)
        
    def check_service(self, name, url, timeout=2):
        """Check if service is responding"""
        try:
            response = requests.get(url, timeout=timeout)
            status = "🟢 HEALTHY" if response.status_code == 200 else "🟡 WARNING"
            latency = f"{response.elapsed.total_seconds():.3f}s"
            self.checks.append(True)
            return f"{status:<12} | {latency}"
        except:
            self.checks.append(False)
            return "🔴 DOWN     | N/A"
    
    def check_docker(self):
        """Check Docker containers"""
        try:
            result = subprocess.run(
                ["docker", "ps", "--format", "table {{.Names}}\\t{{.Status}}"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                containers = [line for line in result.stdout.split('\n')[1:] if 'vorta-ultra' in line]
                return len(containers), containers
            return 0, []
        except:
            return 0, []
    
    def check_file_exists(self, path):
        """Check if file exists"""
        return "✅" if Path(path).exists() else "❌"
    
    def run_checks(self):
        """Run all checks"""
        self.print_header()
        
        # Service Checks
        print("\n🔧 CORE SERVICES:")
        print("-" * 60)
        api_status = self.check_service("VORTA API", "http://localhost:8000/api/health")
        print(f"🚀 VORTA API       | {api_status}")
        
        prometheus_status = self.check_service("Prometheus", "http://localhost:9090/-/healthy")
        print(f"📊 Prometheus      | {prometheus_status}")
        
        grafana_status = self.check_service("Grafana", "http://localhost:3000/api/health")
        print(f"📈 Grafana         | {grafana_status}")
        
        # Docker Check
        print("\n🐳 DOCKER STATUS:")
        print("-" * 60)
        container_count, containers = self.check_docker()
        print(f"📦 Containers      | {container_count}/8 running")
        
        # Tools Check
        print("\n🛠️  TOOLS VERIFICATION:")
        print("-" * 60)
        tools = [
            ("Intelligence Engine", "tools/intelligence/intelligence_engine.py"),
            ("GitHub Workflows AI", "tools/intelligence/github_workflow_intelligence.py"),
            ("Smart Startup", "tools/development/smart_startup.py"),
            ("Dev Dashboard", "tools/monitoring/dev_dashboard.py"),
            ("Quick Check", "tools/intelligence/quick_check.py")
        ]
        
        for name, path in tools:
            status = self.check_file_exists(path)
            print(f"{status} {name:<20} | {path}")
        
        # GitHub Workflows Check
        print("\n🔄 GITHUB WORKFLOWS:")
        print("-" * 60)
        workflows_dir = Path(".github/workflows")
        if workflows_dir.exists():
            workflow_files = list(workflows_dir.glob("*.yml"))
            print(f"📋 Workflow Files  | {len(workflow_files)} found")
            for workflow in workflow_files[:5]:  # Show first 5
                print(f"   ✅ {workflow.name}")
            if len(workflow_files) > 5:
                print(f"   ... and {len(workflow_files) - 5} more")
        else:
            print("❌ Workflows Dir   | Not found")
        
        # Summary
        print("\n🎯 VALIDATION SUMMARY:")
        print("-" * 60)
        healthy_services = sum(self.checks)
        total_checks = len(self.checks)
        success_rate = (healthy_services / total_checks * 100) if total_checks > 0 else 0
        
        print(f"✅ Healthy Services: {healthy_services}/{total_checks}")
        print(f"📊 Success Rate: {success_rate:.1f}%")
        print(f"⏱️  Check Duration: {time.time() - self.start_time:.2f}s")
        
        if success_rate >= 80:
            print("🚀 VORTA STATUS: READY FOR DEVELOPMENT!")
        elif success_rate >= 60:
            print("⚠️  VORTA STATUS: Some issues detected")
        else:
            print("🔴 VORTA STATUS: Multiple issues - investigate required")
        
        print("=" * 80)

def main():
    """Main function"""
    checker = VortaQuickCheck()
    checker.run_checks()

if __name__ == "__main__":
    main()
