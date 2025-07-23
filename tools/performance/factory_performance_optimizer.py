# ⚡ VORTA Factory Pattern Performance Optimizer Integration

"""
Advanced Performance Optimization for VORTA Factory Pattern
Integrates with existing factory_manager.py to provide:
- Component pooling and reuse
- Intelligent caching strategies  
- Memory usage optimization
- Async/await performance enhancement
- Real-time performance monitoring

Phase 5.6: Performance Optimization - IN PROGRESS
"""

import asyncio
import time
import gc
import threading
import weakref
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Any, Union, Callable, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
from contextlib import contextmanager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class FactoryPerformanceConfig:
    """Configuration for factory performance optimization"""
    enable_component_pooling: bool = True
    enable_result_caching: bool = True
    enable_memory_optimization: bool = True
    max_pool_size: int = 50
    max_cache_size: int = 200
    cache_ttl_seconds: int = 300  # 5 minutes
    memory_cleanup_interval: int = 60  # seconds
    pool_cleanup_threshold: float = 0.8  # Cleanup when 80% full

@dataclass
class ComponentMetrics:
    """Performance metrics for individual components"""
    component_name: str
    total_creations: int = 0
    pool_hits: int = 0
    cache_hits: int = 0
    avg_creation_time_ms: float = 0.0
    last_cleanup: Optional[datetime] = None
    memory_usage_mb: float = 0.0

class OptimizedComponentPool:
    """
    High-performance component pool with intelligent lifecycle management
    """
    
    def __init__(self, config: FactoryPerformanceConfig):
        self.config = config
        self.pools: Dict[str, List[Any]] = {}
        self.pool_metadata: Dict[str, List[datetime]] = {}
        self.metrics: Dict[str, ComponentMetrics] = {}
        self.lock = threading.RLock()
        
        # Weak references for automatic cleanup
        self.active_components: Set[weakref.ref] = set()
        
        # Background cleanup task
        self.cleanup_task: Optional[asyncio.Task] = None
        
    async def start_background_tasks(self):
        """Start background maintenance tasks"""
        if not self.cleanup_task:
            self.cleanup_task = asyncio.create_task(self._background_cleanup())
            logger.info("Started factory pool background cleanup task")
    
    async def stop_background_tasks(self):
        """Stop background maintenance tasks"""
        if self.cleanup_task:
            self.cleanup_task.cancel()
            try:
                await self.cleanup_task
            except asyncio.CancelledError:
                pass
            self.cleanup_task = None
    
    def get_or_create(self, component_type: str, factory_func: Callable) -> Any:
        """Get component from pool or create new one with performance tracking"""
        
        # Initialize metrics if needed
        if component_type not in self.metrics:
            self.metrics[component_type] = ComponentMetrics(component_name=component_type)
        
        metrics = self.metrics[component_type]
        
        with self.lock:
            # Try to get from pool first
            if (self.config.enable_component_pooling and 
                component_type in self.pools and 
                self.pools[component_type]):
                
                component = self.pools[component_type].pop()
                self.pool_metadata[component_type].pop()
                
                metrics.pool_hits += 1
                logger.debug(f"Pool hit for {component_type}")
                return component
        
        # Create new component with timing
        start_time = time.perf_counter()
        component = factory_func()
        end_time = time.perf_counter()
        
        creation_time = (end_time - start_time) * 1000
        
        # Update metrics
        metrics.total_creations += 1
        metrics.avg_creation_time_ms = (
            (metrics.avg_creation_time_ms * (metrics.total_creations - 1) + creation_time) /
            metrics.total_creations
        )
        
        # Track active component with weak reference
        if hasattr(component, '__weakref__'):
            weak_ref = weakref.ref(component, self._component_cleanup_callback)
            self.active_components.add(weak_ref)
        
        logger.debug(f"Created new {component_type} in {creation_time:.3f}ms")
        return component
    
    def return_to_pool(self, component_type: str, component: Any):
        """Return component to pool for reuse"""
        if not self.config.enable_component_pooling:
            return
            
        with self.lock:
            # Initialize pool if needed
            if component_type not in self.pools:
                self.pools[component_type] = []
                self.pool_metadata[component_type] = []
            
            # Check pool size limit
            if len(self.pools[component_type]) >= self.config.max_pool_size:
                logger.debug(f"Pool full for {component_type}, discarding component")
                return
            
            # Reset component if possible
            if hasattr(component, 'reset') and callable(component.reset):
                try:
                    component.reset()
                except Exception as e:
                    logger.warning(f"Failed to reset {component_type}: {e}")
                    return  # Don't pool if reset failed
            
            # Add to pool with timestamp
            self.pools[component_type].append(component)
            self.pool_metadata[component_type].append(datetime.now())
            
            logger.debug(f"Returned {component_type} to pool")
    
    def _component_cleanup_callback(self, weak_ref):
        """Callback when component is garbage collected"""
        self.active_components.discard(weak_ref)
    
    async def _background_cleanup(self):
        """Background task to cleanup expired pool entries"""
        while True:
            try:
                await asyncio.sleep(self.config.memory_cleanup_interval)
                await self._cleanup_expired_entries()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in background cleanup: {e}")
    
    async def _cleanup_expired_entries(self):
        """Remove expired entries from pools"""
        cleanup_count = 0
        cutoff_time = datetime.now() - timedelta(seconds=self.config.cache_ttl_seconds)
        
        with self.lock:
            for component_type in list(self.pools.keys()):
                if component_type not in self.pool_metadata:
                    continue
                
                # Find expired entries
                pool = self.pools[component_type]
                metadata = self.pool_metadata[component_type]
                
                # Keep only non-expired entries
                new_pool = []
                new_metadata = []
                
                for component, timestamp in zip(pool, metadata):
                    if timestamp > cutoff_time:
                        new_pool.append(component)
                        new_metadata.append(timestamp)
                    else:
                        cleanup_count += 1
                
                self.pools[component_type] = new_pool
                self.pool_metadata[component_type] = new_metadata
        
        if cleanup_count > 0:
            logger.info(f"Cleaned up {cleanup_count} expired pool entries")
    
    def get_pool_statistics(self) -> Dict[str, Any]:
        """Get comprehensive pool statistics"""
        with self.lock:
            pool_sizes = {k: len(v) for k, v in self.pools.items()}
            
            total_hits = sum(m.pool_hits for m in self.metrics.values())
            total_creations = sum(m.total_creations for m in self.metrics.values())
            
            return {
                "pool_sizes": pool_sizes,
                "total_pool_hits": total_hits,
                "total_creations": total_creations,
                "pool_hit_rate": total_hits / max(total_creations, 1),
                "active_components": len(self.active_components),
                "component_metrics": {
                    name: {
                        "creations": m.total_creations,
                        "pool_hits": m.pool_hits,
                        "avg_creation_ms": m.avg_creation_time_ms,
                        "hit_rate": m.pool_hits / max(m.total_creations, 1)
                    }
                    for name, m in self.metrics.items()
                }
            }

class FactoryResultCache:
    """
    Intelligent caching system for factory results
    """
    
    def __init__(self, config: FactoryPerformanceConfig):
        self.config = config
        self.cache: Dict[str, Any] = {}
        self.cache_metadata: Dict[str, datetime] = {}
        self.hit_count = 0
        self.miss_count = 0
        self.lock = threading.RLock()
    
    def get_cache_key(self, component_type: str, args: tuple, kwargs: dict) -> str:
        """Generate cache key for component creation parameters"""
        # Create a stable hash from arguments
        try:
            args_hash = hash(args) if args else 0
            kwargs_hash = hash(frozenset(kwargs.items())) if kwargs else 0
            return f"{component_type}:{args_hash}:{kwargs_hash}"
        except TypeError:
            # If arguments aren't hashable, don't cache
            return None
    
    def get(self, cache_key: str) -> Optional[Any]:
        """Get cached result if available and not expired"""
        if not cache_key or not self.config.enable_result_caching:
            return None
            
        with self.lock:
            if cache_key not in self.cache:
                self.miss_count += 1
                return None
            
            # Check if expired
            cached_time = self.cache_metadata.get(cache_key)
            if cached_time:
                age = (datetime.now() - cached_time).total_seconds()
                if age > self.config.cache_ttl_seconds:
                    # Remove expired entry
                    del self.cache[cache_key]
                    del self.cache_metadata[cache_key]
                    self.miss_count += 1
                    return None
            
            self.hit_count += 1
            return self.cache[cache_key]
    
    def put(self, cache_key: str, value: Any) -> bool:
        """Cache a value if caching is enabled and cache has space"""
        if not cache_key or not self.config.enable_result_caching:
            return False
            
        with self.lock:
            # Check cache size limit
            if len(self.cache) >= self.config.max_cache_size:
                # Remove oldest entries (simple LRU)
                oldest_keys = sorted(
                    self.cache_metadata.items(),
                    key=lambda x: x[1]
                )[:self.config.max_cache_size // 4]  # Remove 25%
                
                for key, _ in oldest_keys:
                    del self.cache[key]
                    del self.cache_metadata[key]
            
            self.cache[cache_key] = value
            self.cache_metadata[cache_key] = datetime.now()
            return True
    
    def get_cache_statistics(self) -> Dict[str, Any]:
        """Get cache performance statistics"""
        with self.lock:
            total_requests = self.hit_count + self.miss_count
            return {
                "size": len(self.cache),
                "max_size": self.config.max_cache_size,
                "hit_count": self.hit_count,
                "miss_count": self.miss_count,
                "hit_rate": self.hit_count / max(total_requests, 1),
                "utilization": len(self.cache) / self.config.max_cache_size
            }

class FactoryPerformanceManager:
    """
    Main performance management system for Factory Pattern
    """
    
    def __init__(self, config: Optional[FactoryPerformanceConfig] = None):
        self.config = config or FactoryPerformanceConfig()
        self.component_pool = OptimizedComponentPool(self.config)
        self.result_cache = FactoryResultCache(self.config)
        self.is_started = False
        
        # Performance monitoring
        self.optimization_start_time = time.time()
        self.total_optimizations = 0
        
    async def start(self):
        """Start the performance management system"""
        if not self.is_started:
            await self.component_pool.start_background_tasks()
            self.is_started = True
            logger.info("Factory Performance Manager started")
    
    async def stop(self):
        """Stop the performance management system"""
        if self.is_started:
            await self.component_pool.stop_background_tasks()
            self.is_started = False
            logger.info("Factory Performance Manager stopped")
    
    @contextmanager
    def optimize_factory_call(self, component_type: str, factory_func: Callable, *args, **kwargs):
        """Context manager for optimized factory calls"""
        start_time = time.perf_counter()
        component = None
        
        try:
            # Try cache first
            cache_key = self.result_cache.get_cache_key(component_type, args, kwargs)
            cached_result = self.result_cache.get(cache_key)
            
            if cached_result is not None:
                logger.debug(f"Cache hit for {component_type}")
                yield cached_result
                return
            
            # Get from pool or create
            component = self.component_pool.get_or_create(
                component_type, 
                lambda: factory_func(*args, **kwargs)
            )
            
            # Cache the result if appropriate
            if cache_key and self._should_cache_component(component_type):
                self.result_cache.put(cache_key, component)
            
            self.total_optimizations += 1
            yield component
            
        finally:
            end_time = time.perf_counter()
            operation_time = (end_time - start_time) * 1000
            
            # Return to pool if component supports it
            if component and self._should_pool_component(component_type):
                self.component_pool.return_to_pool(component_type, component)
            
            logger.debug(f"Factory call for {component_type} completed in {operation_time:.3f}ms")
    
    def _should_cache_component(self, component_type: str) -> bool:
        """Determine if component type should be cached"""
        # Cache expensive-to-create components
        expensive_components = {
            "multi_modal_processor",
            "conversation_orchestrator", 
            "voice_cloning_engine",
            "enterprise_security_layer"
        }
        return component_type in expensive_components
    
    def _should_pool_component(self, component_type: str) -> bool:
        """Determine if component type should be pooled"""
        # Pool frequently used, stateless components
        poolable_components = {
            "neural_vad_processor",
            "wake_word_detector",
            "audio_quality_analyzer",
            "intent_recognition_engine"
        }
        return component_type in poolable_components
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        uptime = time.time() - self.optimization_start_time
        
        return {
            "performance_summary": {
                "uptime_seconds": uptime,
                "total_optimizations": self.total_optimizations,
                "optimizations_per_second": self.total_optimizations / max(uptime, 1),
                "config": {
                    "pooling_enabled": self.config.enable_component_pooling,
                    "caching_enabled": self.config.enable_result_caching,
                    "max_pool_size": self.config.max_pool_size,
                    "max_cache_size": self.config.max_cache_size
                }
            },
            "component_pooling": self.component_pool.get_pool_statistics(),
            "result_caching": self.result_cache.get_cache_statistics(),
            "recommendations": self._generate_performance_recommendations()
        }
    
    def _generate_performance_recommendations(self) -> List[str]:
        """Generate performance optimization recommendations"""
        recommendations = []
        
        pool_stats = self.component_pool.get_pool_statistics()
        cache_stats = self.result_cache.get_cache_statistics()
        
        # Pool recommendations
        if pool_stats["pool_hit_rate"] < 0.3:
            recommendations.append("Pool hit rate is low - consider increasing pool sizes or adjusting pool retention")
        
        # Cache recommendations  
        if cache_stats["hit_rate"] < 0.4:
            recommendations.append("Cache hit rate is low - review caching strategy and TTL settings")
        
        if cache_stats["utilization"] > 0.9:
            recommendations.append("Cache utilization is high - consider increasing cache size")
        
        return recommendations

# Global performance manager (initialized when needed)
_performance_manager: Optional[FactoryPerformanceManager] = None

def get_performance_manager() -> FactoryPerformanceManager:
    """Get or create global performance manager"""
    global _performance_manager
    if _performance_manager is None:
        _performance_manager = FactoryPerformanceManager()
    return _performance_manager

def optimize_factory_call(component_type: str):
    """Decorator to optimize factory method calls"""
    def decorator(factory_func: Callable):
        async def optimized_wrapper(*args, **kwargs):
            perf_manager = get_performance_manager()
            if not perf_manager.is_started:
                await perf_manager.start()
            
            with perf_manager.optimize_factory_call(component_type, factory_func, *args, **kwargs) as component:
                return component
        
        return optimized_wrapper
    return decorator

async def main():
    """Demo of advanced factory performance optimization"""
    logger.info("=== VORTA Factory Performance Optimization Demo ===")
    
    # Initialize performance manager
    perf_manager = FactoryPerformanceManager()
    await perf_manager.start()
    
    try:
        # Simulate various factory calls
        def mock_expensive_factory():
            time.sleep(0.02)  # 20ms creation time
            return {"type": "expensive", "created": datetime.now()}
        
        def mock_cheap_factory():
            time.sleep(0.001)  # 1ms creation time  
            return {"type": "cheap", "created": datetime.now()}
        
        # Test different optimization strategies
        logger.info("Testing component pooling optimization")
        for i in range(10):
            with perf_manager.optimize_factory_call("neural_vad_processor", mock_cheap_factory):
                pass
        
        logger.info("Testing result caching optimization")
        for i in range(5):
            with perf_manager.optimize_factory_call("multi_modal_processor", mock_expensive_factory):
                pass
        
        # Generate performance report
        report = perf_manager.get_performance_report()
        
        logger.info("\n=== Performance Optimization Report ===")
        logger.info(f"Total optimizations: {report['performance_summary']['total_optimizations']}")
        logger.info(f"Pool hit rate: {report['component_pooling']['pool_hit_rate']:.1%}")
        logger.info(f"Cache hit rate: {report['result_caching']['hit_rate']:.1%}")
        
        if report['recommendations']:
            logger.info("Recommendations:")
            for rec in report['recommendations']:
                logger.info(f"  - {rec}")
        
    finally:
        await perf_manager.stop()

if __name__ == "__main__":
    asyncio.run(main())
