# services/orchestrator/load_balancer.py
"""
VORTA: Intelligent Load Balancer

This module provides intelligent load balancing and failover capabilities for the
VORTA platform. While a production deployment would typically use a dedicated
load balancer like Nginx, Traefik, or a cloud provider's LB, this internal
load balancer can handle application-level routing logic.

Features:
- Health-check-based routing (avoids sending requests to down services).
- Strategies: Round Robin, Least Connections (simulated).
- Automatic failover to a healthy instance.
- Service registry awareness.
"""

import logging
import asyncio
from typing import List, Dict, Optional
import random
from collections import defaultdict

# Configure logging
logger = logging.getLogger(__name__)

# --- Load Balancer State ---

class ServiceInstance:
    """Represents a single instance of a running service."""
    def __init__(self, url: str):
        self.url = url
        self.is_healthy = True
        self.active_connections = 0

    def set_health(self, status: bool):
        if self.is_healthy != status:
            logger.info(f"Service instance {self.url} health status changed to {'HEALTHY' if status else 'UNHEALTHY'}.")
            self.is_healthy = status

class ServicePool:
    """Manages a pool of instances for a single service."""
    def __init__(self, service_name: str):
        self.service_name = service_name
        self.instances: List[ServiceInstance] = []
        self.current_index = 0 # For Round Robin

    def add_instance(self, url: str):
        self.instances.append(ServiceInstance(url))
        logger.info(f"Added instance {url} to service pool '{self.service_name}'.")

    def get_healthy_instances(self) -> List[ServiceInstance]:
        return [inst for inst in self.instances if inst.is_healthy]

# --- Load Balancing Strategies ---

class LoadBalancer:
    """
    Selects the best service instance based on a chosen strategy.
    """
    def __init__(self, service_pool: ServicePool):
        self.service_pool = service_pool

    def get_next_instance(self, strategy: str = "round_robin") -> Optional[ServiceInstance]:
        """
        Returns a service instance based on the load balancing strategy.
        """
        healthy_instances = self.service_pool.get_healthy_instances()
        if not healthy_instances:
            logger.error(f"No healthy instances available for service '{self.service_pool.service_name}'.")
            return None

        if strategy == "round_robin":
            return self._round_robin(healthy_instances)
        elif strategy == "least_connections":
            return self._least_connections(healthy_instances)
        elif strategy == "random":
            return random.choice(healthy_instances)
        else:
            logger.warning(f"Unknown strategy '{strategy}'. Defaulting to round_robin.")
            return self._round_robin(healthy_instances)

    def _round_robin(self, instances: List[ServiceInstance]) -> ServiceInstance:
        """Selects instances in a circular order."""
        instance = instances[self.service_pool.current_index % len(instances)]
        self.service_pool.current_index = (self.service_pool.current_index + 1) % len(instances)
        return instance

    def _least_connections(self, instances: List[ServiceInstance]) -> ServiceInstance:
        """Selects the instance with the fewest active connections."""
        return min(instances, key=lambda inst: inst.active_connections)

# --- Health Monitor ---

class HealthMonitor:
    """
    Periodically checks the health of all service instances and updates their status.
    """
    def __init__(self, service_pools: Dict[str, ServicePool], check_interval_seconds: int = 30):
        self.service_pools = service_pools
        self.check_interval = check_interval_seconds
        self.monitor_task: Optional[asyncio.Task] = None

    async def _check_health(self, instance: ServiceInstance):
        """Performs a single health check on an instance."""
        try:
            # In a real system, you'd use httpx or aiohttp here.
            # This is a simulation.
            # A real check would be: `response = await client.get(f"{instance.url}/health")`
            # For simulation, we'll just assume it's healthy but randomly fail one.
            is_ok = random.random() > 0.1 # 10% chance of failure for demonstration
            instance.set_health(is_ok)
        except Exception as e:
            logger.error(f"Health check for {instance.url} failed with error: {e}")
            instance.set_health(False)

    async def start(self):
        """Starts the background health monitoring task."""
        if self.monitor_task is None:
            logger.info(f"Starting health monitor with a {self.check_interval}s interval.")
            self.monitor_task = asyncio.create_task(self._run())
        else:
            logger.warning("Health monitor is already running.")

    async def _run(self):
        while True:
            logger.info("Running periodic health checks...")
            tasks = []
            for pool in self.service_pools.values():
                for instance in pool.instances:
                    tasks.append(self._check_health(instance))
            await asyncio.gather(*tasks)
            await asyncio.sleep(self.check_interval)

    async def stop(self):
        """Stops the health monitor."""
        if self.monitor_task:
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                logger.info("Health monitor stopped successfully.")
            self.monitor_task = None

# --- Example Usage ---

async def main():
    """Demonstrates the load balancer and health monitor."""
    logger.info("--- Load Balancer Demonstration ---")

    # 1. Setup service pools
    inference_pool = ServicePool("inference_engine")
    inference_pool.add_instance("http://localhost:8001")
    inference_pool.add_instance("http://localhost:8002") # A second instance

    vector_pool = ServicePool("vector_store")
    vector_pool.add_instance("http://localhost:9001")

    all_pools = {"inference_engine": inference_pool, "vector_store": vector_pool}

    # 2. Start the health monitor
    monitor = HealthMonitor(all_pools, check_interval_seconds=5)
    await monitor.start()

    # 3. Create a load balancer for the inference service
    lb = LoadBalancer(inference_pool)

    # 4. Simulate making requests
    print("\n--- Simulating 10 requests with Round Robin ---")
    for i in range(10):
        instance = lb.get_next_instance("round_robin")
        if instance:
            print(f"Request {i+1} routed to: {instance.url}")
        else:
            print(f"Request {i+1} failed: No healthy instance.")
    
    # Let the monitor run for a bit
    print("\nWaiting for health checks to run...")
    await asyncio.sleep(6)

    # 5. Manually mark an instance as unhealthy to see failover
    print("\n--- Simulating an instance failure ---")
    inference_pool.instances[0].set_health(False)

    print("\n--- Simulating 5 more requests (failover expected) ---")
    for i in range(5):
        instance = lb.get_next_instance("round_robin")
        if instance:
            print(f"Request {i+1} routed to: {instance.url}")
        else:
            print(f"Request {i+1} failed: No healthy instance.")

    # 6. Stop the monitor
    await monitor.stop()
    logger.info("Demonstration complete.")

if __name__ == "__main__":
    asyncio.run(main())
