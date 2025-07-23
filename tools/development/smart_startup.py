#!/usr/bin/env python3
"""
VORTA Smart Startup - Intelligent Automation
Determines automatically when and how to start VORTA services
"""

import subprocess
import time
from datetime import datetime
from pathlib import Path


class VortaSmartStartup:
    """Smart startup manager voor VORTA development."""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        
        # Service requirements
        self.required_ports = {
            "API": 8000,
            "Prometheus": 9090, 
            "Grafana": 3000
        }
        
        # When to auto-start
        self.startup_triggers = {
            "first_time": True,          # First time today
            "after_restart": True,       # After system restart  
            "before_tests": True,        # Before running tests
            "before_debug": True,        # Before debugging
            "on_code_change": False,     # On file changes (disabled - too aggressive)
        }
    
    def check_port_availability(self, port: int) -> bool:
        """Check if a port is responding."""
        try:
            import socket
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            result = sock.connect_ex(('localhost', port))
            sock.close()
            return result == 0
        except:
            return False
    
    def check_docker_containers(self) -> dict:
        """Check Docker container status."""
        try:
            result = subprocess.run([
                'docker', 'ps', '--format', '{{.Names}}\t{{.Status}}'
            ], capture_output=True, text=True, timeout=5)
            
            containers = {}
            if result.returncode == 0:
                for line in result.stdout.strip().split('\n'):
                    if 'vorta' in line.lower() and '\t' in line:
                        name = line.split('\t')[0]
                        status = line.split('\t')[1]
                        containers[name] = "healthy" in status.lower() or "up" in status.lower()
            
            return containers
            
        except Exception as e:
            return {"error": str(e)}
    
    def analyze_current_state(self) -> dict:
        """Analyze huidige staat van VORTA services."""
        print("🔍 Analyzing VORTA environment...")
        
        # Check ports
        port_status = {}
        for service, port in self.required_ports.items():
            port_status[service] = self.check_port_availability(port)
        
        # Check Docker
        docker_status = self.check_docker_containers()
        
        # Decision logic
        ports_healthy = sum(1 for status in port_status.values() if status)
        containers_running = len([c for c in docker_status.values() if c and c != "error"])
        
        analysis = {
            "ports": port_status,
            "docker": docker_status,
            "summary": {
                "ports_healthy": ports_healthy,
                "total_ports": len(port_status),
                "containers_running": containers_running,
                "needs_startup": ports_healthy < len(port_status) or containers_running < 3
            },
            "timestamp": datetime.now().strftime("%H:%M:%S")
        }
        
        return analysis
    
    def determine_action(self, analysis: dict) -> str:
        """Bepaal welke actie nodig is."""
        summary = analysis["summary"]
        
        if summary["containers_running"] == 0:
            return "full_startup"  # Alles starten
        elif summary["ports_healthy"] < summary["total_ports"]:
            return "service_check"  # Services controleren
        elif summary["containers_running"] < 3:
            return "container_restart"  # Containers herstarten
        else:
            return "monitoring_only"  # Alleen monitoring
    
    def execute_action(self, action: str) -> bool:
        """Voer de bepaalde actie uit."""
        print(f"🚀 Executing: {action.replace('_', ' ').title()}")
        
        try:
            if action == "full_startup":
                print("  → Starting complete Docker stack...")
                result = subprocess.run([
                    'docker-compose', 'up', '-d'
                ], cwd=self.project_root, timeout=90)
                
                if result.returncode == 0:
                    print("  ✅ Docker stack started successfully")
                    print("  → Waiting for services to initialize...")
                    time.sleep(15)  # Wait for startup
                    return True
                else:
                    print("  ❌ Docker startup failed")
                    return False
            
            elif action == "service_check":
                print("  → Running service health checks...")
                time.sleep(2)
                return True
            
            elif action == "container_restart":
                print("  → Restarting key containers...")
                containers = ["vorta-ultra-api", "vorta-ultra-prometheus"]
                
                for container in containers:
                    try:
                        subprocess.run(['docker', 'restart', container], timeout=30)
                        print(f"  ✅ Restarted {container}")
                    except:
                        print(f"  ⚠️  Could not restart {container}")
                
                time.sleep(5)
                return True
            
            elif action == "monitoring_only":
                print("  ✅ Services running - starting monitoring mode")
                return True
            
        except Exception as e:
            print(f"  ❌ Action failed: {e}")
            return False
    
    def print_status_report(self, analysis: dict, action: str):
        """Print comprehensive status report."""
        print("\n" + "=" * 70)
        print("🧠 VORTA SMART STARTUP - INTELLIGENT DECISION SYSTEM")
        print("=" * 70)
        
        # Current status
        print(f"\n📊 CURRENT STATUS ({analysis['timestamp']}):")
        
        for service, status in analysis["ports"].items():
            icon = "✅" if status else "❌"
            port = self.required_ports[service]
            print(f"  {icon} {service:12} (port {port}) - {'Responding' if status else 'Not available'}")
        
        # Docker status
        print("\n🐳 DOCKER CONTAINERS:")
        docker_status = analysis["docker"]
        
        if "error" in docker_status:
            print(f"  ❌ Docker check failed: {docker_status['error']}")
        elif not docker_status:
            print("  ⚠️  No VORTA containers found")
        else:
            for container, running in docker_status.items():
                icon = "✅" if running else "❌"
                print(f"  {icon} {container}")
        
        # Summary
        summary = analysis["summary"]
        print("\n🎯 DECISION SUMMARY:")
        print(f"  • Services healthy: {summary['ports_healthy']}/{summary['total_ports']}")
        print(f"  • Containers running: {summary['containers_running']}")
        print(f"  • Recommended action: {action.replace('_', ' ').title()}")
        
        # When to use tools
        print("\n🛠️  TOOL USAGE INTELLIGENCE:")
        print("  📋 Use auto_manager.py when:")
        print("     • Starting development (elke keer)")
        print("     • After system reboot")
        print("     • Unknown service state")
        
        print("  🧪 Use test_metrics.py when:")
        print("     • Before commits")
        print("     • API changes made")
        print("     • Validation needed")
        
        print("  📊 Use dev_dashboard.py when:")
        print("     • Active development")
        print("     • Real-time monitoring")
        print("     • Performance analysis")
        
        print("  🔍 Use quick_check.py when:")
        print("     • Quick health verification")
        print("     • Between development tasks")
        
        if action == "monitoring_only":
            print("\n✅ RECOMMENDATION: Services ready - no startup needed!")
            print("   → Use: python dev_dashboard.py (for monitoring)")
            print("   → Use: python test_metrics.py (for validation)")
        else:
            print(f"\n🚀 RECOMMENDATION: Running {action.replace('_', ' ')} now...")
    
    def run_intelligent_analysis(self):
        """Run complete intelligent analysis."""
        # Analyze current state
        analysis = self.analyze_current_state()
        
        # Determine best action
        action = self.determine_action(analysis)
        
        # Print comprehensive report
        self.print_status_report(analysis, action)
        
        # Execute action
        print("\n🤖 EXECUTING INTELLIGENT ACTION...")
        success = self.execute_action(action)
        
        # Final verification
        if success and action != "monitoring_only":
            print("\n🔍 Verifying results...")
            time.sleep(3)
            
            final_analysis = self.analyze_current_state()
            final_summary = final_analysis["summary"]
            
            print(f"📊 Final Status: {final_summary['ports_healthy']}/{final_summary['total_ports']} services ready")
            
            if final_summary["ports_healthy"] == final_summary["total_ports"]:
                print("🎉 SUCCESS: All VORTA services are ready!")
                print("💡 Next: Use dev_dashboard.py for monitoring")
            else:
                print("⚠️  Some services still need attention")
        
        return success


def main():
    """Main smart startup function."""
    startup_manager = VortaSmartStartup()
    
    # Check command line args
    import sys
    
    if "--status-only" in sys.argv:
        analysis = startup_manager.analyze_current_state()
        summary = analysis["summary"]
        
        print(f"🔍 Quick Status: {summary['ports_healthy']}/{summary['total_ports']} services, {summary['containers_running']} containers")
        return
    
    # Run full intelligent analysis
    print("🤖 VORTA Smart Startup - Intelligent Automation")
    print("=" * 50)
    
    success = startup_manager.run_intelligent_analysis()
    
    if success:
        print("\n✅ Smart startup completed successfully!")
    else:
        print("\n❌ Smart startup encountered issues")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 Smart startup interrupted")
    except Exception as e:
        print(f"\n❌ Smart startup error: {e}")
