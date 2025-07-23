#!/usr/bin/env python3
"""
VORTA Phase 5.6 Performance Optimization - Standalone Demo
Enterprise AI Voice Agent - Performance Optimization System Demonstration
"""

import asyncio
import time
import threading
from typing import Dict, Any, List, Optional, Callable
from collections import defaultdict
from dataclasses import dataclass
import weakref
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/performance_demo.log', mode='w')
    ]
)
logger = logging.getLogger('VortaPerformanceDemo')

@dataclass
class PerformanceMetrics:
    """Performance metrics for factory operations"""
    operation_name: str
    execution_time: float
    memory_usage: int
    cache_hit_rate: float
    pool_utilization: float
    creation_count: int
    reuse_count: int

class MockComponent:
    """Mock component for performance testing"""
    
    def __init__(self, component_type: str, config: Dict[str, Any] = None):
        self.type = component_type
        self.config = config or {}
        self.created_at = time.time()
        self.usage_count = 0
        self.is_busy = False
        
        # Simulate initialization time
        time.sleep(0.001)  # 1ms initialization
        
    def process(self, data: Any) -> Any:
        """Mock processing method"""
        self.usage_count += 1
        # Simulate processing time
        time.sleep(0.0005)  # 0.5ms processing
        return f"Processed {data} with {self.type}"
    
    def reset(self):
        """Reset component state for reuse"""
        self.is_busy = False
        
    def __del__(self):
        logger.debug(f"Component {self.type} destroyed after {self.usage_count} uses")

class OptimizedComponentPool:
    """High-performance component pool with intelligent reuse"""
    
    def __init__(self, max_size: int = 100):
        self.max_size = max_size
        self.pools: Dict[str, List[MockComponent]] = defaultdict(list)
        self.usage_stats: Dict[str, int] = defaultdict(int)
        self.hit_count = 0
        self.miss_count = 0
        self.lock = threading.Lock()
        
    def get_component(self, component_type: str, config: Dict[str, Any] = None) -> MockComponent:
        """Get component from pool or create new one"""
        with self.lock:
            pool = self.pools[component_type]
            
            # Try to reuse existing component
            for component in pool:
                if not component.is_busy:
                    component.is_busy = True
                    component.reset()
                    self.hit_count += 1
                    self.usage_stats[component_type] += 1
                    return component
            
            # Create new component if pool not full
            if len(pool) < self.max_size:
                component = MockComponent(component_type, config)
                component.is_busy = True
                pool.append(component)
                self.miss_count += 1
                self.usage_stats[component_type] += 1
                return component
            
            # Pool is full, wait or create temporary
            logger.warning(f"Pool full for {component_type}, creating temporary component")
            self.miss_count += 1
            return MockComponent(component_type, config)
    
    def return_component(self, component: MockComponent):
        """Return component to pool"""
        with self.lock:
            component.is_busy = False
    
    def get_cache_hit_rate(self) -> float:
        """Calculate cache hit rate"""
        total = self.hit_count + self.miss_count
        return (self.hit_count / total * 100) if total > 0 else 0.0
    
    def get_pool_utilization(self) -> float:
        """Calculate average pool utilization"""
        with self.lock:
            total_capacity = len(self.pools) * self.max_size
            total_used = sum(len(pool) for pool in self.pools.values())
            return (total_used / total_capacity * 100) if total_capacity > 0 else 0.0

class IntelligentCache:
    """LRU Cache with TTL and intelligent prefetching"""
    
    def __init__(self, max_size: int = 1000, ttl: float = 300.0):
        self.max_size = max_size
        self.ttl = ttl
        self.cache: Dict[str, Any] = {}
        self.access_times: Dict[str, float] = {}
        self.access_count: Dict[str, int] = defaultdict(int)
        self.lock = threading.Lock()
        
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        with self.lock:
            current_time = time.time()
            
            if key in self.cache:
                # Check TTL
                if current_time - self.access_times[key] < self.ttl:
                    self.access_times[key] = current_time
                    self.access_count[key] += 1
                    return self.cache[key]
                else:
                    # Expired, remove
                    self._remove_key(key)
            
            return None
    
    def set(self, key: str, value: Any):
        """Set value in cache"""
        with self.lock:
            current_time = time.time()
            
            # Remove oldest if at capacity
            if len(self.cache) >= self.max_size and key not in self.cache:
                self._evict_oldest()
            
            self.cache[key] = value
            self.access_times[key] = current_time
            self.access_count[key] = 1
    
    def _remove_key(self, key: str):
        """Remove key from all tracking structures"""
        self.cache.pop(key, None)
        self.access_times.pop(key, None)
        self.access_count.pop(key, None)
    
    def _evict_oldest(self):
        """Evict least recently used item"""
        if not self.access_times:
            return
            
        oldest_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
        self._remove_key(oldest_key)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self.lock:
            return {
                'size': len(self.cache),
                'max_size': self.max_size,
                'utilization': len(self.cache) / self.max_size * 100,
                'most_accessed': max(self.access_count.items(), key=lambda x: x[1]) if self.access_count else ('none', 0)
            }

class PerformanceOptimizedFactory:
    """High-performance factory with pooling, caching, and async support"""
    
    def __init__(self):
        self.component_pool = OptimizedComponentPool(max_size=50)
        self.result_cache = IntelligentCache(max_size=500, ttl=300.0)
        self.metrics: List[PerformanceMetrics] = []
        self.creation_count = 0
        self.reuse_count = 0
        
    def create_component(self, component_type: str, config: Dict[str, Any] = None, use_cache: bool = True) -> MockComponent:
        """Create component with optimization"""
        start_time = time.time()
        
        # Try cache first
        cache_key = f"{component_type}:{hash(str(config))}" if config else component_type
        if use_cache:
            cached_component = self.result_cache.get(cache_key)
            if cached_component and not cached_component.is_busy:
                cached_component.is_busy = True
                self.reuse_count += 1
                execution_time = time.time() - start_time
                self._record_metrics("cached_create", execution_time)
                return cached_component
        
        # Get from pool
        component = self.component_pool.get_component(component_type, config)
        
        # Cache for future use
        if use_cache:
            self.result_cache.set(cache_key, component)
        
        self.creation_count += 1
        execution_time = time.time() - start_time
        self._record_metrics("pooled_create", execution_time)
        
        return component
    
    async def create_component_async(self, component_type: str, config: Dict[str, Any] = None) -> MockComponent:
        """Async component creation"""
        # Run in thread pool to avoid blocking
        return await asyncio.get_event_loop().run_in_executor(
            None, self.create_component, component_type, config
        )
    
    def return_component(self, component: MockComponent):
        """Return component to pool"""
        self.component_pool.return_component(component)
    
    def _record_metrics(self, operation: str, execution_time: float):
        """Record performance metrics"""
        metric = PerformanceMetrics(
            operation_name=operation,
            execution_time=execution_time,
            memory_usage=0,  # Simplified
            cache_hit_rate=self.component_pool.get_cache_hit_rate(),
            pool_utilization=self.component_pool.get_pool_utilization(),
            creation_count=self.creation_count,
            reuse_count=self.reuse_count
        )
        self.metrics.append(metric)
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        if not self.metrics:
            return {"status": "No metrics recorded"}
        
        recent_metrics = self.metrics[-100:]  # Last 100 operations
        avg_execution_time = sum(m.execution_time for m in recent_metrics) / len(recent_metrics)
        
        return {
            "total_operations": len(self.metrics),
            "avg_execution_time_ms": round(avg_execution_time * 1000, 3),
            "cache_hit_rate": round(self.component_pool.get_cache_hit_rate(), 2),
            "pool_utilization": round(self.component_pool.get_pool_utilization(), 2),
            "creation_count": self.creation_count,
            "reuse_count": self.reuse_count,
            "reuse_ratio": round((self.reuse_count / (self.creation_count + self.reuse_count)) * 100, 2) if (self.creation_count + self.reuse_count) > 0 else 0,
            "cache_stats": self.result_cache.get_stats()
        }

def benchmark_performance():
    """Comprehensive performance benchmark"""
    logger.info("🚀 Starting VORTA Phase 5.6 Performance Optimization Benchmark")
    
    # Test configurations
    test_configurations = [
        {"name": "Standard Factory", "optimized": False, "iterations": 100},
        {"name": "Optimized Factory", "optimized": True, "iterations": 100},
        {"name": "Bulk Operations", "optimized": True, "iterations": 500},
        {"name": "Concurrent Operations", "optimized": True, "iterations": 200}
    ]
    
    results = {}
    
    for config in test_configurations:
        logger.info(f"📊 Testing: {config['name']}")
        
        if config["optimized"]:
            factory = PerformanceOptimizedFactory()
        else:
            # Simulate standard factory (no optimizations)
            factory = type('StandardFactory', (), {
                'create_component': lambda self, ct, cfg=None, uc=True: MockComponent(ct, cfg)
            })()
        
        # Benchmark
        start_time = time.time()
        components = []
        
        if config["name"] == "Concurrent Operations":
            # Test async operations
            async def create_concurrent():
                tasks = []
                for i in range(config["iterations"]):
                    if hasattr(factory, 'create_component_async'):
                        task = factory.create_component_async(f"component_{i % 5}")
                        tasks.append(task)
                    else:
                        # Fallback for standard factory
                        component = factory.create_component(f"component_{i % 5}")
                        tasks.append(asyncio.create_task(asyncio.sleep(0)))
                        components.append(component)
                
                if hasattr(factory, 'create_component_async'):
                    return await asyncio.gather(*tasks)
                else:
                    return components
            
            if hasattr(factory, 'create_component_async'):
                components = asyncio.run(create_concurrent())
        else:
            # Standard synchronous test
            for i in range(config["iterations"]):
                component = factory.create_component(f"component_{i % 10}")
                components.append(component)
                
                # Return to pool for optimized factory
                if config["optimized"] and hasattr(factory, 'return_component'):
                    factory.return_component(component)
        
        end_time = time.time()
        total_time = end_time - start_time
        avg_time_per_op = (total_time / config["iterations"]) * 1000  # ms
        
        # Get performance summary for optimized factory
        if config["optimized"] and hasattr(factory, 'get_performance_summary'):
            perf_summary = factory.get_performance_summary()
        else:
            perf_summary = {"avg_execution_time_ms": avg_time_per_op}
        
        results[config["name"]] = {
            "total_time_s": round(total_time, 3),
            "avg_time_per_op_ms": round(avg_time_per_op, 3),
            "operations_per_second": round(config["iterations"] / total_time, 1),
            "performance_summary": perf_summary
        }
        
        logger.info(f"✅ {config['name']}: {avg_time_per_op:.3f}ms per operation")
    
    return results

def print_performance_report(results: Dict[str, Any]):
    """Print comprehensive performance report"""
    print("\n" + "="*80)
    print("🏆 VORTA Phase 5.6 Performance Optimization Report")
    print("="*80)
    
    for name, result in results.items():
        print(f"\n📈 {name}:")
        print(f"   Total Time: {result['total_time_s']}s")
        print(f"   Avg Time/Op: {result['avg_time_per_op_ms']}ms")
        print(f"   Ops/Second: {result['operations_per_second']}")
        
        if "performance_summary" in result and isinstance(result["performance_summary"], dict):
            summary = result["performance_summary"]
            if "cache_hit_rate" in summary:
                print(f"   Cache Hit Rate: {summary['cache_hit_rate']}%")
            if "pool_utilization" in summary:
                print(f"   Pool Utilization: {summary['pool_utilization']}%")
            if "reuse_ratio" in summary:
                print(f"   Component Reuse: {summary['reuse_ratio']}%")
    
    # Calculate improvements
    if "Standard Factory" in results and "Optimized Factory" in results:
        standard_time = results["Standard Factory"]["avg_time_per_op_ms"]
        optimized_time = results["Optimized Factory"]["avg_time_per_op_ms"]
        improvement_factor = standard_time / optimized_time if optimized_time > 0 else 1
        improvement_percent = ((standard_time - optimized_time) / standard_time) * 100
        
        print(f"\n🚀 Performance Improvement:")
        print(f"   Speed Improvement: {improvement_factor:.1f}x faster")
        print(f"   Time Reduction: {improvement_percent:.1f}%")
        print(f"   From {standard_time:.3f}ms to {optimized_time:.3f}ms per operation")
    
    print("\n" + "="*80)
    print("✅ Phase 5.6 Performance Optimization: COMPLETE")
    print("="*80)

def main():
    """Main performance demonstration"""
    print("🎯 VORTA Phase 5.6 Performance Optimization Demo")
    print("Enterprise AI Voice Agent - Performance Optimization System")
    
    try:
        # Create logs directory
        import os
        os.makedirs('logs', exist_ok=True)
        
        # Run benchmarks
        results = benchmark_performance()
        
        # Print report
        print_performance_report(results)
        
        logger.info("🎉 Performance optimization demonstration completed successfully")
        
    except Exception as e:
        logger.error(f"❌ Error during performance demo: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
