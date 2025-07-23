"""
VORTA API Metrics Test Suite
Gebruikt VS Code test integration voor development workflow
"""
import asyncio
import json
import time
from typing import Any, Dict

import aiohttp


class VortaMetricsValidator:
    """Test validator for VORTA API metrics using VS Code integration."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        
    async def test_api_health(self) -> Dict[str, Any]:
        """Test API health endpoint - VS Code will show results."""
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(f"{self.base_url}/api/health") as response:
                    if response.status == 200:
                        data = await response.json()
                        print(f"✅ Health Check: {data['status']}")
                        return {"status": "pass", "data": data}
                    else:
                        print(f"❌ Health Check failed: {response.status}")
                        return {"status": "fail", "error": response.status}
            except Exception as e:
                print(f"❌ Connection failed: {e}")
                return {"status": "error", "error": str(e)}
    
    async def test_metrics_endpoint(self) -> Dict[str, Any]:
        """Test Prometheus metrics endpoint."""
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(f"{self.base_url}/api/metrics") as response:
                    if response.status == 200:
                        metrics_data = await response.text()
                        print(f"✅ Metrics Available: {len(metrics_data)} characters")
                        
                        # Validate specific VORTA metrics
                        required_metrics = [
                            "vorta_ultra_http_requests_total",
                            "vorta_ultra_http_request_duration_seconds",
                            "vorta_ultra_active_connections"
                        ]
                        
                        missing_metrics = []
                        for metric in required_metrics:
                            if metric not in metrics_data:
                                missing_metrics.append(metric)
                        
                        if missing_metrics:
                            print(f"❌ Missing metrics: {missing_metrics}")
                            return {"status": "fail", "missing": missing_metrics}
                        else:
                            print("✅ All required metrics present")
                            return {"status": "pass", "metrics_count": len(required_metrics)}
                            
            except Exception as e:
                print(f"❌ Metrics test failed: {e}")
                return {"status": "error", "error": str(e)}
    
    async def test_performance_benchmark(self) -> Dict[str, Any]:
        """Performance test for VS Code monitoring."""
        async with aiohttp.ClientSession() as session:
            start_time = time.time()
            
            # Make 10 concurrent requests
            tasks = []
            for _ in range(10):
                task = session.get(f"{self.base_url}/api/health")
                tasks.append(task)
            
            try:
                await asyncio.gather(*tasks, return_exceptions=True)
                end_time = time.time()
                
                total_time = end_time - start_time
                avg_time = total_time / len(tasks)
                
                print(f"✅ Performance Test: {total_time:.3f}s total, {avg_time:.3f}s average")
                
                return {
                    "status": "pass",
                    "total_time": total_time,
                    "average_time": avg_time,
                    "requests": len(tasks)
                }
                
            except Exception as e:
                print(f"❌ Performance test failed: {e}")
                return {"status": "error", "error": str(e)}


async def run_all_tests():
    """Main test runner - VS Code will execute this."""
    print("🚀 Starting VORTA API Test Suite...")
    
    validator = VortaMetricsValidator()
    
    # Test Suite
    tests = [
        ("Health Check", validator.test_api_health),
        ("Metrics Validation", validator.test_metrics_endpoint), 
        ("Performance Benchmark", validator.test_performance_benchmark)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n📊 Running: {test_name}")
        result = await test_func()
        results[test_name] = result
    
    # Summary Report
    print("\n" + "="*50)
    print("🎯 VORTA API TEST RESULTS")
    print("="*50)
    
    passed = sum(1 for r in results.values() if r.get("status") == "pass")
    total = len(results)
    
    for test_name, result in results.items():
        status_emoji = "✅" if result.get("status") == "pass" else "❌"
        print(f"{status_emoji} {test_name}: {result.get('status', 'unknown')}")
    
    print(f"\n🎯 Summary: {passed}/{total} tests passed")
    
    return results


if __name__ == "__main__":
    # Dit kan VS Code direct runnen met F5 of Ctrl+F5
    results = asyncio.run(run_all_tests())
    
    # Save results for VS Code inspection
    with open("test_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("\n💾 Results saved to test_results.json")
