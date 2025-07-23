# services/orchestrator/__init__.py
"""
VORTA Orchestrator Service

This package contains all modules related to the central service orchestrator.

Modules:
- service_coordinator: The main FastAPI application for routing and coordination.
- workflow_manager: Manages the execution of complex, multi-step workflows.
- load_balancer: Provides application-level load balancing and failover.
- request_queue: A system for queuing and throttling requests to prevent overload.
"""

from . import service_coordinator
from . import workflow_manager
from . import load_balancer
from . import request_queue

__all__ = [
    "service_coordinator",
    "workflow_manager",
    "load_balancer",
    "request_queue",
]

import logging

logging.getLogger(__name__).addHandler(logging.NullHandler())
