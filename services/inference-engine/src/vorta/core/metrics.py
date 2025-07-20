"""
VORTA Metrics Collector
Production-ready metrics collection and monitoring for the inference engine
"""

import logging
import threading
import time
from collections import defaultdict, deque
from typing import Any, Dict

from ..config.settings import Settings

logger = logging.getLogger(__name__)


class MetricsCollector:
    """Collects and manages metrics for the inference engine"""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.is_initialized = False
        self.lock = threading.RLock()
        
        # Request metrics
        self.request_count = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.request_latencies = deque(maxlen=1000)  # Keep last 1000 latencies
        
        # Model metrics
        self.model_requests: Dict[str, int] = defaultdict(int)
        self.model_latencies: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=100)
        )
        self.model_errors: Dict[str, int] = defaultdict(int)
        
        # Cache metrics
        self.cache_hits = 0
        self.cache_misses = 0
        
        # System metrics
        self.start_time = time.time()
        
    def initialize(self):
        """Initialize metrics collector"""
        try:
            logger.info("Initializing metrics collector...")
            self.is_initialized = True
            logger.info("Metrics collector initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize metrics collector: {e}")
            raise
    
    def is_ready(self) -> bool:
        """Check if metrics collector is ready"""
        return self.is_initialized
    
    def record_request(self, model: str, latency: float, success: bool):
        """Record a single inference request"""
        with self.lock:
            self.request_count += 1
            self.request_latencies.append(latency)
            self.model_requests[model] += 1
            self.model_latencies[model].append(latency)
            
            if success:
                self.successful_requests += 1
            else:
                self.failed_requests += 1
                self.model_errors[model] += 1
    
    def record_batch_request(
        self, model: str, batch_size: int, latency: float, success: bool
    ):
        """Record a batch inference request"""
        with self.lock:
            # Record as individual requests for overall stats
            avg_latency = latency / batch_size
            for _ in range(batch_size):
                self.request_count += 1
                self.request_latencies.append(avg_latency)
                self.model_requests[model] += 1
                self.model_latencies[model].append(avg_latency)
                
                if success:
                    self.successful_requests += 1
                else:
                    self.failed_requests += 1
                    self.model_errors[model] += 1
    
    def get_inference_metrics(self) -> Dict[str, Any]:
        """Get comprehensive inference metrics"""
        with self.lock:
            # Calculate average latency
            avg_latency = 0.0
            if self.request_latencies:
                total_latency = sum(self.request_latencies)
                avg_latency = total_latency / len(self.request_latencies)
            
            # Calculate cache hit rate
            total_cache_requests = self.cache_hits + self.cache_misses
            cache_hit_rate = 0.0
            if total_cache_requests > 0:
                cache_hit_rate = self.cache_hits / total_cache_requests
            
            return {
                "total_requests": self.request_count,
                "successful_requests": self.successful_requests,
                "failed_requests": self.failed_requests,
                "average_latency": avg_latency,
                "models_loaded": len(self.model_requests),
                "cache_hit_rate": cache_hit_rate
            }
    
    def shutdown(self):
        """Shutdown metrics collector"""
        logger.info("Shutting down metrics collector...")
        self.is_initialized = False
        logger.info("Metrics collector shutdown completed")
