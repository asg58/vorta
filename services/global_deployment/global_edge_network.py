# services/global_deployment/global_edge_network.py
"""
VORTA AGI: Global Edge Network Manager

Ultra-low latency worldwide deployment system for enterprise scale
- Edge node management across global regions
- Intelligent request routing based on latency
- Auto-scaling edge infrastructure
- Real-time performance monitoring per region
"""

import asyncio
import time
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from enum import Enum
import random
import aiohttp
from aiohttp import web

# --- Basic Configuration ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# --- Data Models ---
class EdgeRegion(Enum):
    """Global edge regions for VORTA deployment."""

    US_EAST = "us-east-1"
    US_WEST = "us-west-2"
    EU_WEST = "eu-west-1"
    EU_CENTRAL = "eu-central-1"


@dataclass
class EdgeNode:
    """Represents an edge node in the global network."""

    region: EdgeRegion
    endpoint: str
    capacity: int = 1000
    current_load: int = 0
    latency_ms: float = float("inf")
    status: str = "unknown"
    last_health_check: float = field(default_factory=time.time)


@dataclass
class RoutingDecision:
    """Result of intelligent routing decision."""

    selected_node: EdgeNode
    reason: str
    backup_nodes: List[EdgeNode] = field(default_factory=list)


class GlobalEdgeNetworkError(Exception):
    """Custom exception for GlobalEdgeNetwork errors."""

    pass


# --- Core Network Class ---
class GlobalEdgeNetwork:
    """Manages the global edge network for ultra-low latency deployment."""

    def __init__(self, update_interval: int = 10):
        self.edge_nodes: Dict[EdgeRegion, EdgeNode] = {}
        self.session: Optional[aiohttp.ClientSession] = None
        self._monitor_task: Optional[asyncio.Task] = None
        self.update_interval = update_interval
        self._initialize_edge_nodes()

    def _initialize_edge_nodes(self):
        """Initialize edge nodes with mock local endpoints for demonstration."""
        base_port = 8001
        edge_configs = {
            EdgeRegion.US_EAST: {
                "endpoint": f"http://127.0.0.1:{base_port}",
                "capacity": 2000,
            },
            EdgeRegion.US_WEST: {
                "endpoint": f"http://127.0.0.1:{base_port+1}",
                "capacity": 1500,
            },
            EdgeRegion.EU_WEST: {
                "endpoint": f"http://127.0.0.1:{base_port+2}",
                "capacity": 1800,
            },
            EdgeRegion.EU_CENTRAL: {
                "endpoint": f"http://127.0.0.1:{base_port+3}",
                "capacity": 1200,
            },
        }

        for region, config in edge_configs.items():
            self.edge_nodes[region] = EdgeNode(region=region, **config)
        logger.info(f"Initialized {len(self.edge_nodes)} global edge nodes.")

    def start_monitoring(self):
        """Starts the background task for continuous health checks."""
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession()
        if self._monitor_task is None:
            self._monitor_task = asyncio.create_task(self._health_check_loop())
            logger.info("Edge node health monitoring started.")

    async def stop_monitoring(self):
        """Stops the background health check task."""
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                logger.info("Monitoring task successfully cancelled.")
            self._monitor_task = None
            logger.info("Edge node health monitoring stopped.")
        if self.session:
            await self.session.close()
            self.session = None

    def find_optimal_edge_node(self) -> RoutingDecision:
        """Finds the optimal edge node based on current health and latency data."""
        active_nodes = sorted(
            [node for node in self.edge_nodes.values() if node.status == "active"],
            key=lambda n: n.latency_ms,
        )

        if not active_nodes:
            return self._fallback_routing()

        best_node = active_nodes[0]
        backup_nodes = active_nodes[1:3]

        return RoutingDecision(
            selected_node=best_node,
            reason=f"Optimal latency: {best_node.latency_ms:.2f}ms",
            backup_nodes=backup_nodes,
        )

    def _fallback_routing(self) -> RoutingDecision:
        """Provides a fallback routing decision if no nodes are active."""
        all_nodes = list(self.edge_nodes.values())
        if not all_nodes:
            raise GlobalEdgeNetworkError("No edge nodes configured.")

        # Fallback to the node that was last seen, even if unreachable
        selected_node = min(all_nodes, key=lambda n: n.last_health_check)

        return RoutingDecision(
            selected_node=selected_node, reason="Fallback: No active nodes available."
        )

    async def _health_check_loop(self):
        """Periodically checks the health of all registered nodes."""
        while True:
            try:
                await self.health_check_all_nodes()
            except Exception as e:
                logger.error(f"Error in health check loop: {e}")
            await asyncio.sleep(self.update_interval)

    async def health_check_all_nodes(self):
        """Performs health checks on all nodes and updates their status."""
        tasks = {
            region: self._perform_node_health_check(node)
            for region, node in self.edge_nodes.items()
        }
        results = await asyncio.gather(*tasks.values(), return_exceptions=True)

        for region, result in zip(tasks.keys(), results):
            node = self.edge_nodes[region]
            node.last_health_check = time.time()
            if isinstance(result, Exception):
                node.status = "unreachable"
                node.latency_ms = float("inf")
                logger.warning(f"Node {node.region.value} is unreachable: {result}")
            else:
                node.status = result.get("status", "unknown")
                node.current_load = result.get("current_load", 0)
                node.latency_ms = result.get("response_time", float("inf"))

    async def _perform_node_health_check(self, node: EdgeNode) -> Dict[str, Any]:
        """Performs a single health check on a node."""
        if not self.session:
            raise GlobalEdgeNetworkError("Session not started.")

        start_time = time.time()
        try:
            async with self.session.get(
                f"{node.endpoint}/health", timeout=2
            ) as response:
                response_time = (time.time() - start_time) * 1000
                response.raise_for_status()
                health_data = await response.json()
                return {
                    "status": "active",
                    "response_time": response_time,
                    "current_load": health_data.get("load", {}).get("current", 0),
                }
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            raise GlobalEdgeNetworkError(
                f"Node {node.region.value} health check failed"
            ) from e


# --- Mock Server for Testing ---
async def create_mock_edge_server(port: int) -> web.AppRunner:
    """Creates and starts a mock aiohttp server for testing."""
    app = web.Application()

    async def health(request: web.Request) -> web.Response:
        # Simulate potential server unavailability
        if random.random() < 0.05:  # 5% chance to be "down"
            return web.Response(status=503, reason="Service Unavailable")

        await asyncio.sleep(random.uniform(0.01, 0.05))  # Simulate latency
        return web.json_response(
            {
                "status": "active",
                "load": {"current": random.randint(100, 500), "max": 1000},
            }
        )

    app.router.add_get("/health", health)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", port)
    await site.start()
    logger.info(f"Mock edge server started on http://127.0.0.1:{port}")
    return runner


# --- Main Execution Block ---
async def main():
    """Main function to demonstrate and test GlobalEdgeNetwork."""
    ports = [8001, 8002, 8003, 8004]
    runners = []
    try:
        for port in ports:
            runners.append(await create_mock_edge_server(port))
    except Exception as e:
        logger.error(f"Failed to start mock servers: {e}")
        return

    network = GlobalEdgeNetwork(update_interval=5)
    network.start_monitoring()

    try:
        for i in range(3):
            await asyncio.sleep(5.1)
            logger.info(f"\n--- Test Cycle {i+1} ---")
            decision = network.find_optimal_edge_node()
            logger.info(
                f"Routing decision: -> {decision.selected_node.region.value} "
                f"({decision.selected_node.latency_ms:.2f}ms). Reason: {decision.reason}"
            )
            for region, node in sorted(
                network.edge_nodes.items(), key=lambda item: item[1].latency_ms
            ):
                logger.info(
                    f"  Node {node.region.value:<12}: Status={node.status:<12} "
                    f"Latency={node.latency_ms:7.2f}ms"
                )
    finally:
        logger.info("Shutting down...")
        await network.stop_monitoring()
        for runner in runners:
            await runner.cleanup()
        logger.info("Shutdown complete.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Interrupted by user.")