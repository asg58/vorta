#!/usr/bin/env python3
"""
VORTA Auto-Startup Manager - Intelligent VS Code Integration
Automatically manages VORTA services and determines when to use tools
"""

import asyncio
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False


class VortaAutoManager:
    """Intelligent auto-management for VORTA development environment."""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.status_file = self.project_root / "vorta_status.json"
        
        # Service detection patterns
        self.required_services = [
            {"name": "API", "port": 8000, "health": "/api/health"},
            {"name": "Prometheus", "port": 9090, "health": "/-/healthy"}, 
            {"name": "Grafana", "port": 3000, "health": "/api/health"}
        ]
        
        # Auto-startup conditions
        self.startup_conditions = {
            "on_code_change": True,      # Start when code files change
            "on_test_run": True,         # Start before tests
            "on_debug_start": True,      # Start when debugging
            "on_vscode_open": False,     # Start when VS Code opens (disabled by default)
            "smart_detection": True      # Use intelligent detection
        }
    
    def save_status(self, status: Dict[str, Any]):
        """Save current VORTA status to file."""
        try:
            with open(self.status_file, 'w') as f:
                json.dump({
                    **status,
                    "last_updated": datetime.now().isoformat(),
                    "auto_manager_version": "1.0"
                }, f, indent=2)
        except Exception as e:
            print(f"⚠️  Could not save status: {e}")
    
    def load_status(self) -> Dict[str, Any]:
        """Load previous VORTA status."""
        try:
            if self.status_file.exists():
                with open(self.status_file, 'r') as f:
                    return json.load(f)
        except Exception:
            pass
        return {"services": {}, "last_started": None}
    
    async def check_service_availability(self) -> Dict[str, bool]:
        """Quick check if services are running."""
        if not AIOHTTP_AVAILABLE:
            return {service["name"]: False for service in self.required_services}
        
        results = {}
        
        async with aiohttp.ClientSession() as session:
            for service in self.required_services:
                url = f"http://localhost:{service['port']}{service['health']}"
                try:
                    async with session.get(url, timeout=2) as response:
                        results[service["name"]] = response.status == 200
                except:
                    results[service["name"]] = False
        
        return results
    
    def check_docker_services(self) -> Dict[str, bool]:
        """Check if Docker containers are running."""
        try:
            result = subprocess.run([
                'docker', 'ps', '--format', '{{.Names}}\t{{.Status}}'
            ], capture_output=True, text=True, timeout=5)
            
            if result.returncode != 0:
                return {}
            
            running_containers = {}
            for line in result.stdout.strip().split('\n'):
                if 'vorta' in line.lower():
                    name = line.split('\t')[0]
                    status = line.split('\t')[1] if '\t' in line else ""
                    running_containers[name] = "healthy" in status.lower() or "up" in status.lower()
            
            return running_containers
            
        except Exception:
            return {}
    
    def determine_startup_need(self) -> Dict[str, Any]:
        """Intelligently determine if services need to be started."""
        print("🤖 VORTA Auto-Manager: Analyzing environment...")
        
        # Check current status
        service_status = asyncio.run(self.check_service_availability())
        docker_status = self.check_docker_services()
        previous_status = self.load_status()
        
        # Decision logic
        services_needed = sum(1 for running in service_status.values() if not running)
        containers_needed = sum(1 for running in docker_status.values() if not running)
        
        # Intelligent decisions
        decisions = {
            "services_running": services_needed == 0,
            "containers_running": len(docker_status) >= 3,  # At least 3 containers
            "needs_startup": services_needed > 0 or len(docker_status) < 3,
            "startup_reason": [],
            "recommended_action": "none"
        }
        
        # Determine reasons and actions
        if services_needed > 0:
            decisions["startup_reason"].append(f"{services_needed} services not responding")
        
        if len(docker_status) < 3:
            decisions["startup_reason"].append("Docker containers not running")
        
        # Recommendations
        if decisions["needs_startup"]:
            if len(docker_status) == 0:
                decisions["recommended_action"] = "full_startup"
            else:
                decisions["recommended_action"] = "service_restart"
        else:
            decisions["recommended_action"] = "monitoring_only"
        
        return {
            "decision": decisions,
            "service_status": service_status,
            "docker_status": docker_status,
            "timestamp": datetime.now().isoformat()
        }
    
    def execute_startup_sequence(self, action: str) -> bool:
        """Execute the appropriate startup sequence."""
        print(f"🚀 Executing: {action}")
        
        try:
            if action == "full_startup":
                print("  → Starting Docker Compose stack...")
                result = subprocess.run([
                    'docker-compose', 'up', '-d'
                ], cwd=self.project_root, timeout=60)
                
                if result.returncode == 0:
                    print("  ✅ Docker stack started")
                    # Wait for services to be ready
                    print("  → Waiting for services to be ready...")
                    time.sleep(10)
                    return True
                else:
                    print("  ❌ Docker startup failed")
                    return False
            
            elif action == "service_restart":
                print("  → Restarting specific services...")
                containers_to_restart = ["vorta-ultra-api", "vorta-ultra-prometheus"]
                
                for container in containers_to_restart:
                    subprocess.run(['docker', 'restart', container], timeout=30)
                    print(f"  ✅ Restarted {container}")
                
                return True
            
            elif action == "monitoring_only":
                print("  → Services already running, starting monitoring...")
                return True
            
        except Exception as e:
            print(f"  ❌ Startup failed: {e}")
            return False
    
    def create_vscode_integration(self):
        """Create VS Code integration for auto-startup."""
        
        # Create tasks for auto-management
        auto_tasks = {
            "version": "2.0.0",
            "tasks": [
                {
                    "label": "VORTA: Smart Auto-Start",
                    "type": "shell",
                    "command": "python",
                    "args": ["auto_manager.py", "--auto-start"],
                    "group": {
                        "kind": "build",
                        "isDefault": True
                    },
                    "presentation": {
                        "echo": true,
                        "reveal": "always",
                        "focus": false,
                        "panel": "shared"
                    },
                    "runOptions": {
                        "runOn": "folderOpen"  # Auto-run when folder opens
                    }
                },
                {
                    "label": "VORTA: Quick Status Check",
                    "type": "shell", 
                    "command": "python",
                    "args": ["auto_manager.py", "--status"],
                    "group": "test"
                }
            ]
        }
        
        # Save auto-tasks
        tasks_file = self.project_root / ".vscode" / "tasks-auto.json"
        with open(tasks_file, 'w') as f:
            json.dump(auto_tasks, f, indent=2)
        
        print(f"✅ Created VS Code auto-tasks: {tasks_file}")
    
    def print_smart_recommendations(self, analysis: Dict[str, Any]):
        """Print intelligent recommendations."""
        print("\n" + "=" * 70)
        print("🧠 VORTA SMART AUTO-MANAGER RECOMMENDATIONS")
        print("=" * 70)
        
        decision = analysis["decision"]
        
        # Status overview
        print("\n📊 CURRENT STATUS:")
        if decision["services_running"]:
            print("  ✅ All services responding")
        else:
            print("  ⚠️  Services need attention")
        
        if decision["containers_running"]:
            print("  ✅ Docker containers running")
        else:
            print("  ⚠️  Docker containers need startup")
        
        # Recommendations
        print(f"\n🎯 RECOMMENDATION: {decision['recommended_action'].upper()}")
        
        if decision["startup_reason"]:
            print("📝 REASONS:")
            for reason in decision["startup_reason"]:
                print(f"  • {reason}")
        
        # When to use each tool
        print("\n🛠️  TOOL USAGE STRATEGY:")
        print("  🔄 Auto-Manager: Use when:")
        print("     • Starting development session")
        print("     • After system restart") 
        print("     • When services are down")
        
        print("  🧪 Test Suite: Use when:")
        print("     • Before commits")
        print("     • After code changes")
        print("     • CI/CD validation")
        
        print("  📊 Dashboard: Use when:")
        print("     • Active development")
        print("     • Debugging issues")
        print("     • Performance monitoring")
        
        print("  🔍 Quick Check: Use when:")
        print("     • Rapid status verification")
        print("     • Health validation")
        print("     • Before important operations")
    
    async def run_smart_analysis(self):
        """Run complete smart analysis and recommendations."""
        analysis = self.determine_startup_need()
        
        # Save current status
        self.save_status(analysis)
        
        # Print recommendations
        self.print_smart_recommendations(analysis)
        
        # Auto-execute if needed
        if analysis["decision"]["needs_startup"]:
            action = analysis["decision"]["recommended_action"]
            
            print(f"\n🤖 AUTO-EXECUTING: {action}")
            success = self.execute_startup_sequence(action)
            
            if success:
                print("✅ Auto-startup completed successfully!")
                
                # Wait and verify
                print("🔍 Verifying services...")
                await asyncio.sleep(5)
                
                final_status = await self.check_service_availability()
                healthy_services = sum(1 for status in final_status.values() if status)
                
                print(f"📊 Final Status: {healthy_services}/{len(final_status)} services healthy")
            else:
                print("❌ Auto-startup failed - manual intervention required")
        else:
            print("\n✅ All services ready - no startup needed!")
        
        # Create VS Code integration
        self.create_vscode_integration()


async def main():
    """Main auto-manager function."""
    manager = VortaAutoManager()
    
    # Check command line arguments
    import sys
    
    if "--auto-start" in sys.argv:
        print("🤖 VORTA Auto-Manager: Smart startup mode")
        await manager.run_smart_analysis()
    elif "--status" in sys.argv:
        print("🔍 VORTA Quick Status Check")
        status = await manager.check_service_availability()
        for service, healthy in status.items():
            icon = "✅" if healthy else "❌"
            print(f"  {icon} {service}")
    else:
        # Full interactive analysis
        await manager.run_smart_analysis()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Auto-manager stopped")
    except Exception as e:
        print(f"\n❌ Auto-manager error: {e}")
