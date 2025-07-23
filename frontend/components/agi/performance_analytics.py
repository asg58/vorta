# frontend/components/agi/performance_analytics.py
"""
VORTA: Performance Analytics Engine

This module provides tools for in-depth analysis of the AGI's performance,
covering technical metrics, user experience, and business KPIs. It helps in
identifying bottlenecks, optimizing resources, and understanding user behavior.
"""

import logging
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
import asyncio
from collections import deque
import numpy as np

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# --- Data Structures for Analytics ---

@dataclass
class MetricEvent:
    """A single performance metric event."""
    name: str
    value: float
    timestamp: float = field(default_factory=time.time)
    tags: Dict[str, str] = field(default_factory=dict) # e.g., {"endpoint": "/chat", "model": "gpt-4"}

@dataclass
class AnalyticsReport:
    """A summary report of performance analytics."""
    report_name: str
    time_window_seconds: int
    generated_at: str
    summary: Dict[str, Any] = field(default_factory=dict)
    detailed_stats: Dict[str, Dict[str, float]] = field(default_factory=dict)

# --- Analytics Engine Components ---

class MetricStore:
    """
    An in-memory store for recent metric events.
    Uses a deque for efficient fixed-size storage.
    """
    def __init__(self, max_size: int = 10000):
        self.store: Dict[str, deque] = {}
        self.max_size = max_size
        self.lock = asyncio.Lock()
        logger.info(f"MetricStore initialized with max size {max_size} per metric.")

    async def add_metric(self, event: MetricEvent):
        """Adds a metric event to the store."""
        async with self.lock:
            if event.name not in self.store:
                self.store[event.name] = deque(maxlen=self.max_size)
            self.store[event.name].append(event)

    async def query(self, metric_name: str, time_window_seconds: int) -> List[MetricEvent]:
        """Queries metric events within a specific time window."""
        async with self.lock:
            if metric_name not in self.store:
                return []
            
            now = time.time()
            cutoff = now - time_window_seconds
            
            # This is inefficient for large deques. A real system would use a time-series DB.
            return [event for event in self.store[metric_name] if event.timestamp >= cutoff]

class ReportGenerator:
    """Generates analytical reports from metric data."""
    
    async def calculate_stats(self, values: List[float]) -> Dict[str, float]:
        """Calculates descriptive statistics for a list of values."""
        if not values:
            return {}
        
        arr = np.array(values)
        stats = {
            "count": len(arr),
            "mean": np.mean(arr),
            "std_dev": np.std(arr),
            "min": np.min(arr),
            "max": np.max(arr),
            "p50": np.percentile(arr, 50),
            "p90": np.percentile(arr, 90),
            "p95": np.percentile(arr, 95),
            "p99": np.percentile(arr, 99),
        }
        return {k: round(v, 4) for k, v in stats.items()}

    async def generate_latency_report(self, metric_store: MetricStore, time_window: int) -> AnalyticsReport:
        """Generates a report on API and model latency."""
        logger.info(f"Generating latency report for the last {time_window} seconds.")
        
        api_latency_events = await metric_store.query("api_latency_ms", time_window)
        model_latency_events = await metric_store.query("model_latency_ms", time_window)

        report = AnalyticsReport(
            report_name="Latency Performance Report",
            time_window_seconds=time_window,
            generated_at=time.ctime()
        )

        api_latencies = [e.value for e in api_latency_events]
        model_latencies = [e.value for e in model_latency_events]

        report.detailed_stats["api_latency_ms"] = await self.calculate_stats(api_latencies)
        report.detailed_stats["model_latency_ms"] = await self.calculate_stats(model_latencies)

        report.summary = {
            "total_requests": len(api_latencies),
            "avg_api_latency_ms": report.detailed_stats.get("api_latency_ms", {}).get("mean", 0),
            "p95_api_latency_ms": report.detailed_stats.get("api_latency_ms", {}).get("p95", 0),
            "avg_model_latency_ms": report.detailed_stats.get("model_latency_ms", {}).get("mean", 0),
        }
        return report

# --- Main Performance Analytics Engine ---

class PerformanceAnalyticsEngine:
    """Orchestrates the collection, storage, and analysis of performance metrics."""
    def __init__(self):
        self.metric_store = MetricStore()
        self.report_generator = ReportGenerator()
        self.metric_queue = asyncio.Queue()
        self.worker_task = asyncio.create_task(self._metric_worker())
        logger.info("PerformanceAnalyticsEngine initialized and worker started.")

    async def _metric_worker(self):
        """Background worker that processes metrics from a queue."""
        logger.info("Metric worker started. Waiting for metrics...")
        while True:
            try:
                metric_event = await self.metric_queue.get()
                await self.metric_store.add_metric(metric_event)
                self.metric_queue.task_done()
            except Exception as e:
                logger.error(f"Error in metric worker: {e}", exc_info=True)

    async def log_metric(self, name: str, value: float, tags: Optional[Dict[str, str]] = None):
        """Public method to add a metric to the processing queue."""
        event = MetricEvent(name=name, value=value, tags=tags or {})
        await self.metric_queue.put(event)

    async def generate_report(self, report_type: str, time_window: int = 3600) -> Optional[AnalyticsReport]:
        """Generates a specific type of analytics report."""
        if report_type == "latency":
            return await self.report_generator.generate_latency_report(self.metric_store, time_window)
        else:
            logger.warning(f"Report type '{report_type}' is not supported.")
            return None

    async def stop(self):
        """Gracefully stops the metric worker."""
        logger.info("Stopping PerformanceAnalyticsEngine...")
        await self.metric_queue.join()
        self.worker_task.cancel()

# --- Example Usage ---

async def main():
    """Demonstrates the functionality of the PerformanceAnalyticsEngine."""
    logger.info("--- VORTA Performance Analytics Engine Demonstration ---")
    
    engine = PerformanceAnalyticsEngine()

    # 1. Simulate logging some performance metrics
    logger.info("\n--- Scenario 1: Logging simulated performance metrics ---")
    for i in range(100):
        # Simulate API latency
        api_latency = np.random.normal(loc=120, scale=30)
        await engine.log_metric("api_latency_ms", api_latency, tags={"endpoint": "/chat"})
        
        # Simulate model latency (part of the API latency)
        model_latency = api_latency * np.random.uniform(0.6, 0.8)
        await engine.log_metric("model_latency_ms", model_latency, tags={"model": "gpt-4"})
        
        await asyncio.sleep(0.01) # Simulate time passing

    # Wait for the queue to be processed
    await asyncio.sleep(0.2)

    # 2. Generate a performance report
    logger.info("\n--- Scenario 2: Generating a latency report ---")
    latency_report = await engine.generate_report("latency", time_window=60)
    
    if latency_report:
        print(f"\n--- {latency_report.report_name} ---")
        print(f"Generated at: {latency_report.generated_at}")
        print("\nSummary:")
        for key, value in latency_report.summary.items():
            print(f"  - {key}: {value:.2f}")
        
        print("\nDetailed Statistics (api_latency_ms):")
        for stat, value in latency_report.detailed_stats.get("api_latency_ms", {}).items():
            print(f"  - {stat}: {value:.2f}")

    # 3. Stop the engine
    await engine.stop()
    logger.info("\nDemonstration complete.")


if __name__ == "__main__":
    # To run this demonstration, you might need to install:
    # pip install numpy
    try:
        import numpy as np
    except ImportError:
        logger.warning("="*50)
        logger.warning("Running in limited functionality mode.")
        logger.warning("Please run 'pip install numpy' for full features.")
        logger.warning("="*50)
    
    asyncio.run(main())

# Alias for backward compatibility
PerformanceAnalytics = PerformanceAnalyticsEngine
