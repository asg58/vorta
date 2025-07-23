# services/orchestrator/service_coordinator.py
"""
VORTA: Service Coordinator

This is the central nervous system of the VORTA platform. It's a FastAPI-based
service responsible for receiving requests from the frontend, routing them to the
appropriate backend services (like the inference engine, vector store, etc.),
and coordinating complex, multi-step workflows.

Key Responsibilities:
- Expose a unified API to the frontend.
- Route requests to downstream services.
- Manage service discovery and health checks.
- Coordinate workflows that involve multiple services.
- Handle authentication and authorization.
"""

import logging
import httpx
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
import asyncio

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(
    title="VORTA Service Orchestrator",
    description="The central coordinator for all VORTA backend services.",
    version="1.0.0"
)

# --- Service Registry ---
# In a real production environment, this would be managed by a service discovery
# tool like Consul or etcd, or handled by Kubernetes services.
SERVICE_REGISTRY = {
    "inference_engine": "http://localhost:8001",
    "vector_store": "http://localhost:8002",
    "api_gateway": "http://localhost:8080",
    # Add other services here as they come online
}

# --- Health Check Monitoring ---
# This is a simplified health checker. A real system would use more robust checks.
async def check_service_health(service_name: str, url: str):
    try:
        async with httpx.AsyncClient() as client:
            # Assuming each service has a /health endpoint
            response = await client.get(f"{url}/health")
            if response.status_code == 200:
                logger.info(f"Health check PASSED for {service_name}.")
                return True
            else:
                logger.warning(f"Health check FAILED for {service_name} with status {response.status_code}.")
                return False
    except httpx.RequestError as e:
        logger.error(f"Health check FAILED for {service_name}. Cannot connect to {url}. Error: {e}")
        return False

@app.on_event("startup")
async def startup_event():
    logger.info("VORTA Orchestrator is starting up...")
    # Perform initial health checks on all registered services
    tasks = [check_service_health(name, url) for name, url in SERVICE_REGISTRY.items()]
    await asyncio.gather(*tasks)
    logger.info("Initial health checks complete.")

# --- Generic Proxy/Routing Logic ---

async def forward_request(service_name: str, request: Request):
    """
    Forwards an incoming request to a downstream service.
    """
    service_url = SERVICE_REGISTRY.get(service_name)
    if not service_url:
        raise HTTPException(status_code=404, detail=f"Service '{service_name}' not found.")

    # Construct the full path for the downstream service
    downstream_url = f"{service_url}{request.url.path}"
    
    async with httpx.AsyncClient() as client:
        try:
            # Prepare the request to be forwarded
            headers = dict(request.headers)
            # Host header needs to be updated for the downstream service
            headers["host"] = service_url.split("://")[1]
            
            data = await request.body()

            # Forward the request
            response = await client.request(
                method=request.method,
                url=downstream_url,
                headers=headers,
                params=request.query_params,
                content=data,
                timeout=30.0, # Set a reasonable timeout
            )
            
            # Return the response from the downstream service
            return JSONResponse(content=response.json(), status_code=response.status_code)

        except httpx.RequestError as e:
            logger.error(f"Error forwarding request to {service_name}: {e}")
            raise HTTPException(status_code=503, detail=f"Service '{service_name}' is unavailable.")

# --- API Endpoints ---

@app.get("/health")
async def health_check():
    """Provides a health check endpoint for the orchestrator itself."""
    return {"status": "ok", "service": "orchestrator"}

# Example of a specific route for the inference engine
@app.api_route("/inference/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def route_to_inference(request: Request, path: str):
    """Routes all /inference/* requests to the inference engine."""
    logger.info(f"Routing request for path '/inference/{path}' to inference_engine.")
    return await forward_request("inference_engine", request)

# Example of a specific route for the vector store
@app.api_route("/vector/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def route_to_vector_store(request: Request, path: str):
    """Routes all /vector/* requests to the vector store."""
    logger.info(f"Routing request for path '/vector/{path}' to vector_store.")
    return await forward_request("vector_store", request)

# --- Complex Workflow Example ---

@app.post("/workflows/process_and_store")
async def process_and_store_workflow(request: Request):
    """
    A complex workflow that first processes data via the inference engine,
    then stores the result in the vector store.
    """
    logger.info("Executing 'process_and_store' workflow...")
    
    try:
        # Step 1: Send data to the inference engine for processing
        inference_request_data = await request.json()
        inference_url = f"{SERVICE_REGISTRY['inference_engine']}/process" # Assuming a /process endpoint
        
        async with httpx.AsyncClient() as client:
            inference_response = await client.post(inference_url, json=inference_request_data, timeout=60.0)
            inference_response.raise_for_status()
            processed_data = inference_response.json()

            # Step 2: Send the processed data to the vector store
            vector_store_url = f"{SERVICE_REGISTRY['vector_store']}/store" # Assuming a /store endpoint
            vector_store_payload = {
                "original_query": inference_request_data,
                "processed_result": processed_data
            }
            vector_response = await client.post(vector_store_url, json=vector_store_payload, timeout=30.0)
            vector_response.raise_for_status()
            storage_confirmation = vector_response.json()

        logger.info("Workflow 'process_and_store' completed successfully.")
        return {
            "status": "success",
            "workflow": "process_and_store",
            "result": storage_confirmation
        }

    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error in workflow: {e.response.status_code} - {e.response.text}")
        raise HTTPException(status_code=e.response.status_code, detail=f"Downstream service error: {e.response.text}")
    except Exception as e:
        logger.error(f"An unexpected error occurred in the workflow: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error in workflow execution.")

if __name__ == "__main__":
    import uvicorn
    # To run this service:
    # uvicorn services.orchestrator.service_coordinator:app --reload --port 8000
    logger.info("Starting VORTA Orchestrator on http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
