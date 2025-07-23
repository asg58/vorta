#!/usr/bin/env python3
"""
VORTA Development Dashboard - VS Code Professional Integration
Complete development status overview for VORTA platform
"""

import asyncio
import subprocess
import time
from datetime import datetime
from typing import Any, Dict, List

try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False


class VortaDevelopmentDashboard:
    """Professional development dashboard for VORTA platform."""
    
    def __init__(self):
        self.services = {
            "VORTA API": "http://localhost:8000/api/health",
            "Prometheus": "http://localhost:9090/-/healthy",
            "Grafana": "http://localhost:3000/api/health"
        }
        self.metrics_url = "http://localhost:8000/api/metrics"
    
    def print_banner(self):
        """Print professional dashboard banner."""
        print("\n" + "=" * 80)
        print("🚀 VORTA ULTRA DEVELOPMENT DASHBOARD")
        print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | VS Code Professional Integration")
        print("=" * 80)
    
    async def get_service_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all VORTA services."""
        if not AIOHTTP_AVAILABLE:
            return {"error": "aiohttp not available"}
        
        results = {}
        
        async with aiohttp.ClientSession() as session:
            for service_name, url in self.services.items():
                start_time = time.time()
                try:
                    async with session.get(url, timeout=5) as response:
                        duration = time.time() - start_time
                        results[service_name] = {
                            "status": "healthy" if response.status == 200 else "degraded",
                            "status_code": response.status,
                            "response_time": f"{duration:.3f}s",
                            "icon": "🟢" if response.status == 200 else "🟡"
                        }
                except Exception as e:
                    results[service_name] = {
                        "status": "down",
                        "error": str(e)[:50],
                        "icon": "🔴"
                    }
        
        return results
    
    async def get_metrics_summary(self) -> Dict[str, str]:
        """Get VORTA metrics summary."""
        if not AIOHTTP_AVAILABLE:
            return {"error": "aiohttp not available"}
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(self.metrics_url, timeout=5) as response:
                    if response.status == 200:
                        metrics_text = await response.text()
                        
                        # Parse key metrics
                        lines = metrics_text.split('\n')
                        metrics = {
                            "total_requests": "0",
                            "health_checks": "0", 
                            "metrics_requests": "0",
                            "active_connections": "0"
                        }
                        
                        for line in lines:
                            if 'vorta_ultra_http_requests_total{' in line:
                                if 'endpoint="/api/health"' in line:
                                    metrics["health_checks"] = line.split()[-1]
                                elif 'endpoint="/api/metrics"' in line:
                                    metrics["metrics_requests"] = line.split()[-1]
                            elif 'vorta_ultra_active_connections' in line and not line.startswith('#'):
                                metrics["active_connections"] = line.split()[-1]
                        
                        return metrics
                    else:
                        return {"error": f"HTTP {response.status}"}
                        
        except Exception as e:
            return {"error": str(e)[:50]}
    
    def get_docker_status(self) -> List[Dict[str, str]]:
        """Get Docker containers status."""
        try:
            result = subprocess.run([
                'docker', 'ps', '--format', 
                'table {{.Names}}\t{{.Status}}\t{{.Ports}}'
            ], capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')[1:]  # Skip header
                containers = []
                
                for line in lines:
                    if 'vorta' in line.lower():
                        parts = line.split('\t')
                        status = parts[1] if len(parts) > 1 else "Unknown"
                        icon = "🟢" if "healthy" in status.lower() else "🟡"
                        
                        containers.append({
                            'name': parts[0][:25] if len(parts) > 0 else 'Unknown',
                            'status': status[:40],
                            'ports': parts[2][:30] if len(parts) > 2 else 'None',
                            'icon': icon
                        })
                
                return containers
            else:
                return [{"error": "Docker command failed"}]
                
        except Exception as e:
            return [{"error": str(e)[:50]}]
    
    def print_service_status(self, services: Dict[str, Dict[str, Any]]):
        """Print service status section."""
        print("\n🔧 SERVICE STATUS:")
        print("-" * 60)
        
        if "error" in services:
            print(f"❌ Error: {services['error']}")
            return
        
        for service_name, data in services.items():
            icon = data.get('icon', '❓')
            status = data.get('status', 'unknown').upper()
            response_time = data.get('response_time', 'N/A')
            
            print(f"{icon} {service_name:15} | {status:8} | {response_time}")
            
            if 'error' in data:
                print(f"   └─ Error: {data['error']}")
    
    def print_metrics_summary(self, metrics: Dict[str, str]):
        """Print metrics summary section."""
        print("\n📊 VORTA METRICS:")
        print("-" * 60)
        
        if "error" in metrics:
            print(f"❌ Metrics Error: {metrics['error']}")
            return
        
        metric_labels = {
            "health_checks": "Health Checks",
            "metrics_requests": "Metrics Requests", 
            "active_connections": "Active Connections"
        }
        
        for key, label in metric_labels.items():
            value = metrics.get(key, "0")
            print(f"📈 {label:20} | {value}")
    
    def print_docker_status(self, containers: List[Dict[str, str]]):
        """Print Docker containers status."""
        print("\n🐳 DOCKER CONTAINERS:")
        print("-" * 60)
        
        if not containers:
            print("❌ No VORTA containers found")
            return
        
        for container in containers:
            if "error" in container:
                print(f"❌ Docker Error: {container['error']}")
                continue
            
            icon = container.get('icon', '❓')
            name = container['name']
            status = container['status']
            
            print(f"{icon} {name:25} | {status}")
    
    def print_quick_links(self):
        """Print quick access links."""
        print("\n🔗 QUICK ACCESS:")
        print("-" * 60)
        print("🌐 Grafana Dashboard: http://localhost:3000")
        print("📊 Prometheus: http://localhost:9090")
        print("📖 API Documentation: http://localhost:8000/docs")
        print("💚 Health Check: http://localhost:8000/api/health")
        print("📈 Metrics: http://localhost:8000/api/metrics")
    
    def print_development_info(self):
        """Print development workflow information."""
        print("\n⚡ VS CODE DEVELOPMENT:")
        print("-" * 60)
        print("🧪 Run Tests: python test_metrics.py")
        print("🔍 Quick Check: python quick_check.py")
        print("📊 Monitor: python vorta_monitor.py")
        print("🐳 Docker Logs: docker logs -f vorta-ultra-api")
    
    async def generate_complete_dashboard(self):
        """Generate complete development dashboard."""
        self.print_banner()
        
        # Get all data
        services = await self.get_service_status()
        metrics = await self.get_metrics_summary()
        containers = self.get_docker_status()
        
        # Print all sections
        self.print_service_status(services)
        self.print_metrics_summary(metrics)
        self.print_docker_status(containers)
        self.print_quick_links()
        self.print_development_info()
        
        # Final status
        if AIOHTTP_AVAILABLE and "error" not in services:
            healthy_count = sum(1 for s in services.values() if s.get('status') == 'healthy')
            total_services = len(services)
            
            print(f"\n🎯 OVERALL STATUS: {healthy_count}/{total_services} services healthy")
            
            if healthy_count == total_services:
                print("✅ VORTA PLATFORM READY FOR DEVELOPMENT!")
            else:
                print("⚠️  Some services need attention")
        else:
            print("\n⚠️  Limited status information available")
        
        print("=" * 80 + "\n")


async def main():
    """Main dashboard function."""
    dashboard = VortaDevelopmentDashboard()
    await dashboard.generate_complete_dashboard()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Dashboard stopped by user")
    except Exception as e:
        print(f"\n❌ Dashboard error: {e}")
