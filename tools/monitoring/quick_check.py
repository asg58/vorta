#!/usr/bin/env python3
"""
VORTA Status Check - Professional VS Code Integration
Quick health check tool for VORTA development
"""

import asyncio
import time
from datetime import datetime

import aiohttp


class VortaQuickCheck:
    """Quick status check for VORTA services."""
    
    async def check_all_services(self):
        """Check all VORTA services quickly."""
        print("🚀 VORTA ULTRA - QUICK STATUS CHECK")
        print(f"📅 {datetime.now().strftime('%H:%M:%S')}")
        print("=" * 50)
        
        services = {
            "API": "http://localhost:8000/api/health",
            "Prometheus": "http://localhost:9090/-/healthy",
            "Grafana": "http://localhost:3000/api/health"
        }
        
        results = {}
        
        async with aiohttp.ClientSession() as session:
            for name, url in services.items():
                start_time = time.time()
                try:
                    async with session.get(url, timeout=3) as response:
                        duration = time.time() - start_time
                        if response.status == 200:
                            results[name] = {
                                "status": "✅",
                                "duration": f"{duration:.3f}s"
                            }
                        else:
                            results[name] = {
                                "status": "⚠️",
                                "code": response.status
                            }
                except Exception as e:
                    results[name] = {
                        "status": "❌",
                        "error": str(e)[:30]
                    }
        
        # Print results
        for service, data in results.items():
            status = data["status"]
            if "duration" in data:
                print(f"{service:12} {status} ({data['duration']})")
            elif "code" in data:
                print(f"{service:12} {status} (HTTP {data['code']})")
            else:
                print(f"{service:12} {status} ({data.get('error', 'Unknown')})")
        
        # Quick metrics check
        try:
            async with session.get("http://localhost:8000/api/metrics", timeout=3) as response:
                if response.status == 200:
                    text = await response.text()
                    lines = len(text.split('\n'))
                    print(f"Metrics      ✅ ({lines} lines)")
                else:
                    print(f"Metrics      ⚠️ (HTTP {response.status})")
        except:
            print("Metrics      ❌ (Not available)")
        
        print("=" * 50)
        
        # Count healthy services
        healthy = sum(1 for r in results.values() if r["status"] == "✅")
        total = len(results) + 1  # +1 for metrics
        
        if healthy == len(results):
            print(f"🎉 ALL SERVICES HEALTHY ({healthy}/{len(results)})")
        else:
            print(f"⚠️  {healthy}/{len(results)} services healthy")


async def main():
    """Main function."""
    checker = VortaQuickCheck()
    await checker.check_all_services()


if __name__ == "__main__":
    asyncio.run(main())
