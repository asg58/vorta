#!/usr/bin/env python3
"""
VORTA Development Monitor - VS Code Integration
Real-time monitoring tool voor development workflow
"""

import asyncio
import os
import subprocess
from datetime import datetime
from typing import Any, Dict, List

import aiohttp


class VortaDevelopmentMonitor:
    """Professional monitoring tool voor VS Code development."""
    
    def __init__(self):
        self.base_url = "http://localhost:8000"
        self.grafana_url = "http://localhost:3000"
        self.prometheus_url = "http://localhost:9090"
        
    def clear_screen(self):
        """Clear terminal voor clean output."""
        os.system('cls' if os.name == 'nt' else 'clear')
    
    def print_header(self):
        """Print professional header."""
        print("=" * 80)
        print(" 🚀 VORTA ULTRA - DEVELOPMENT MONITORING DASHBOARD")
        print(f" 📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)
    
    async def check_service_health(self) -> Dict[str, Any]:
        """Check health of all VORTA services."""
        services = {
            "VORTA API": f"{self.base_url}/api/health",
            "Prometheus": f"{self.prometheus_url}/-/healthy", 
            "Grafana": f"{self.grafana_url}/api/health"
        }
        
        results = {}
        
        async with aiohttp.ClientSession() as session:
            for service_name, url in services.items():
                try:
                    async with session.get(url, timeout=5) as response:
                        if response.status == 200:
                            results[service_name] = {
                                "status": "🟢 HEALTHY",
                                "response_time": f"{response.headers.get('X-Response-Time', 'N/A')}",
                                "status_code": response.status
                            }
                        else:
                            results[service_name] = {
                                "status": "🟡 DEGRADED",
                                "status_code": response.status
                            }
                except Exception as e:
                    results[service_name] = {
                        "status": "🔴 DOWN",
                        "error": str(e)[:50]
                    }
        
        return results
    
    async def get_metrics_summary(self) -> Dict[str, Any]:
        """Get VORTA metrics summary."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.base_url}/api/metrics") as response:
                    if response.status == 200:
                        metrics_text = await response.text()
                        
                        # Parse key metrics
                        lines = metrics_text.split('\n')
                        metrics = {}
                        
                        for line in lines:
                            if 'vorta_ultra_http_requests_total{' in line:
                                # Extract endpoint and count
                                if 'endpoint="/api/health"' in line:
                                    count = line.split()[-1]
                                    metrics['health_checks'] = count
                                elif 'endpoint="/api/metrics"' in line:
                                    count = line.split()[-1] 
                                    metrics['metrics_requests'] = count
                            
                            elif 'vorta_ultra_active_connections' in line:
                                if not line.startswith('#'):
                                    count = line.split()[-1]
                                    metrics['active_connections'] = count
                        
                        return metrics
                    
        except Exception as e:
            return {"error": str(e)}
    
    def get_docker_status(self) -> List[Dict[str, str]]:
        """Get Docker containers status."""
        try:
            result = subprocess.run(
                ['docker', 'ps', '--format', 'table {{.Names}}\\t{{.Status}}\\t{{.Ports}}'],
                capture_output=True, text=True, timeout=10
            )
            
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')[1:]  # Skip header
                containers = []
                
                for line in lines:
                    if 'vorta' in line.lower():
                        parts = line.split('\t')
                        containers.append({
                            'name': parts[0] if len(parts) > 0 else 'Unknown',
                            'status': parts[1] if len(parts) > 1 else 'Unknown',
                            'ports': parts[2] if len(parts) > 2 else 'None'
                        })
                
                return containers
        except Exception as e:
            return [{"error": str(e)}]
    
    async def run_monitoring_loop(self):
        """Main monitoring loop for VS Code terminal."""
        while True:
            self.clear_screen()
            self.print_header()
            
            # Service Health Check
            print("\n📊 SERVICE HEALTH STATUS:")
            print("-" * 50)
            
            health_results = await self.check_service_health()
            for service, data in health_results.items():
                status = data.get('status', 'Unknown')
                print(f" {service:15} | {status}")
                if 'error' in data:
                    print(f"                  └─ Error: {data['error']}")
            
            # Metrics Summary
            print("\n⚡ VORTA METRICS SUMMARY:")
            print("-" * 50)
            
            metrics = await self.get_metrics_summary()
            if 'error' not in metrics:
                for key, value in metrics.items():
                    print(f" {key.replace('_', ' ').title():20} | {value}")
            else:
                print(f" Error: {metrics['error']}")
            
            # Docker Status
            print("\n🐳 DOCKER CONTAINERS:")
            print("-" * 50)
            
            containers = self.get_docker_status()
            for container in containers:
                if 'error' not in container:
                    name = container['name'][:25]
                    status_icon = "🟢" if "healthy" in container['status'].lower() else "🟡"
                    print(f" {status_icon} {name:25} | {container['status'][:30]}")
                else:
                    print(f" ❌ Docker Error: {container['error']}")
            
            # Quick Actions
            print("\n🎯 QUICK ACTIONS (VS Code Integrated):")
            print("-" * 50)
            print(" • Grafana Dashboard: http://localhost:3000")
            print(" • Prometheus: http://localhost:9090")
            print(" • API Docs: http://localhost:8000/docs")
            print(" • Health Check: http://localhost:8000/api/health")
            
            print("\n⏰ Refreshing in 30 seconds... (Ctrl+C to stop)")
            
            try:
                await asyncio.sleep(30)
            except KeyboardInterrupt:
                print("\n\n🛑 Monitoring stopped. Thank you for using VORTA Monitor!")
                break


if __name__ == "__main__":
    print("🚀 Starting VORTA Development Monitor...")
    monitor = VortaDevelopmentMonitor()
    
    try:
        asyncio.run(monitor.run_monitoring_loop())
    except KeyboardInterrupt:
        print("\n👋 Monitor stopped by user.")
