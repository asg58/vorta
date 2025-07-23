# services/orchestrator/workflow_manager.py
"""
VORTA: Workflow Manager

This module is responsible for defining and executing complex, multi-step
workflows that involve multiple services. It provides a structured way to
handle sequences of operations, ensuring they are executed in the correct
order and with proper error handling.

Example Workflow:
A user asks a question.
1. Orchestrator receives the request.
2. Workflow Manager is invoked.
3. Step 1: Transcribe audio (Inference Engine).
4. Step 2: Search knowledge base for context (Vector Store).
5. Step 3: Generate a response with context (Inference Engine).
6. Step 4: Synthesize the response to audio (Inference Engine).
7. Step 5: Log the interaction (Analytics Service).
"""

import logging
from typing import Dict, Any, List, Callable, Awaitable
from dataclasses import dataclass, field
import httpx

# Configure logging
logger = logging.getLogger(__name__)

# --- Data Structures for Workflows ---

@dataclass
class WorkflowStep:
    """Represents a single step in a workflow."""
    name: str
    service: str  # The name of the service to call (from the registry)
    endpoint: str # The API endpoint to hit on that service
    method: str = "POST"
    payload_transformer: Callable[[Dict[str, Any]], Dict[str, Any]] = lambda ctx: ctx.get('last_result', {})

@dataclass
class WorkflowContext:
    """Holds the state of a workflow as it executes."""
    workflow_id: str
    initial_payload: Dict[str, Any]
    results: Dict[str, Any] = field(default_factory=dict)
    last_result: Any = None
    current_step: int = 0
    status: str = "running"

# --- Workflow Definitions ---

class Workflow:
    """A class to define and execute a sequence of steps."""
    def __init__(self, name: str, steps: List[WorkflowStep], service_registry: Dict[str, str]):
        self.name = name
        self.steps = steps
        self.service_registry = service_registry
        logger.info(f"Workflow '{self.name}' initialized with {len(self.steps)} steps.")

    async def execute(self, context: WorkflowContext) -> WorkflowContext:
        """Executes the defined steps in sequence."""
        logger.info(f"Executing workflow '{self.name}' with ID {context.workflow_id}.")
        
        for i, step in enumerate(self.steps):
            context.current_step = i
            logger.info(f"Executing step {i+1}: '{step.name}' on service '{step.service}'.")
            
            service_url = self.service_registry.get(step.service)
            if not service_url:
                context.status = "failed"
                logger.error(f"Service '{step.service}' not found in registry.")
                return context

            try:
                async with httpx.AsyncClient() as client:
                    # Transform the payload from the previous step's result
                    payload = step.payload_transformer(context.results)
                    
                    response = await client.request(
                        method=step.method,
                        url=f"{service_url}{step.endpoint}",
                        json=payload,
                        timeout=60.0
                    )
                    response.raise_for_status()
                    
                    result = response.json()
                    context.results[step.name] = result
                    context.last_result = result
                    logger.info(f"Step '{step.name}' completed successfully.")

            except httpx.HTTPStatusError as e:
                context.status = "failed"
                error_details = e.response.text
                logger.error(f"Step '{step.name}' failed with HTTP error {e.response.status_code}: {error_details}")
                context.results['error'] = f"Step '{step.name}' failed: {error_details}"
                return context
            except Exception as e:
                context.status = "failed"
                logger.error(f"An unexpected error occurred during step '{step.name}': {e}", exc_info=True)
                context.results['error'] = f"Step '{step.name}' failed with an unexpected error."
                return context

        context.status = "completed"
        logger.info(f"Workflow '{self.name}' completed successfully.")
        return context

# --- Workflow Manager ---

class WorkflowManager:
    """Manages all available workflows."""
    def __init__(self, service_registry: Dict[str, str]):
        self.workflows: Dict[str, Workflow] = {}
        self.service_registry = service_registry
        self._register_default_workflows()

    def _register_default_workflows(self):
        """Defines and registers the default system workflows."""
        
        # Define a simple "process and store" workflow
        process_and_store_steps = [
            WorkflowStep(
                name="process_data",
                service="inference_engine",
                endpoint="/process",
                payload_transformer=lambda ctx: ctx.get('initial_payload', {})
            ),
            WorkflowStep(
                name="store_result",
                service="vector_store",
                endpoint="/store",
                # This transformer creates a new payload from the context
                payload_transformer=lambda ctx: {
                    "processed_data": ctx.get('process_data', {}),
                    "metadata": {"source": "workflow"}
                }
            )
        ]
        
        process_and_store_workflow = Workflow(
            name="process_and_store",
            steps=process_and_store_steps,
            service_registry=self.service_registry
        )
        self.workflows[process_and_store_workflow.name] = process_and_store_workflow

    async def run_workflow(self, workflow_name: str, initial_payload: Dict[str, Any]) -> WorkflowContext:
        """Runs a named workflow with a given payload."""
        workflow = self.workflows.get(workflow_name)
        if not workflow:
            raise ValueError(f"Workflow '{workflow_name}' not found.")
        
        context = WorkflowContext(
            workflow_id=f"wf_{hash(str(initial_payload))}", # Simple unique ID
            initial_payload=initial_payload,
            results={'initial_payload': initial_payload}
        )
        
        return await workflow.execute(context)

# Example of how this manager would be integrated into the main FastAPI app
# This part is for demonstration and would live in service_coordinator.py

async def example_integration():
    logger.info("--- WorkflowManager Demonstration ---")
    
    # This would be initialized at startup in the main app
    service_registry = {
        "inference_engine": "http://localhost:8001",
        "vector_store": "http://localhost:8002",
    }
    manager = WorkflowManager(service_registry)

    # This would be triggered by an API call
    initial_data = {"text": "Tell me about the VORTA project."}
    
    logger.info(f"Running workflow 'process_and_store' with data: {initial_data}")
    
    # In a real scenario, the services would need to be running.
    # Here, we'll just see the log output of the attempt.
    try:
        result_context = await manager.run_workflow("process_and_store", initial_data)
        print("\n--- Workflow Result ---")
        print(f"Workflow ID: {result_context.workflow_id}")
        print(f"Status: {result_context.status}")
        print("Results per step:")
        import json
        print(json.dumps(result_context.results, indent=2))
        print("-----------------------\n")
    except ValueError as e:
        logger.error(e)
    except Exception as e:
        logger.error(f"Demonstration failed. Are the downstream services running? Error: {e}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(example_integration())
