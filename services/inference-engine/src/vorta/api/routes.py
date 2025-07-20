"""
VORTA Inference Engine API Routes
Production-ready REST API endpoints for AI model inference
"""

import json
import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from .dependencies import get_inference_engine, get_metrics_collector

# Local imports
from .models import BatchInferenceRequest, InferenceRequest, InferenceResponse

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(
    prefix="/inference",
    tags=["Inference"],
    responses={404: {"description": "Not found"}}
)

class HealthResponse(BaseModel):
    """Health check response model"""
    status: str
    timestamp: str
    service: str = "inference-engine"
    version: str = "1.0.0"

class InferenceMetrics(BaseModel):
    """Inference metrics response model"""
    total_requests: int
    successful_requests: int
    failed_requests: int
    average_latency: float
    models_loaded: int
    cache_hit_rate: float

@router.post("/predict", response_model=InferenceResponse)
async def predict(
    request: InferenceRequest,
    background_tasks: BackgroundTasks,
    inference_engine=Depends(get_inference_engine),
    metrics_collector=Depends(get_metrics_collector)
):
    """
    Single inference prediction endpoint
    
    Accepts text, image, or multimodal input and returns AI model predictions
    with caching, monitoring, and error handling.
    """
    start_time = datetime.utcnow()
    request_id = str(uuid.uuid4())
    
    logger.info(f"Inference request {request_id} - Model: {request.model}")
    
    try:
        # Validate request
        if not request.input_data:
            raise HTTPException(
                status_code=400, 
                detail="Input data is required"
            )
        
        # Check if model is loaded
        if not await inference_engine.is_model_loaded(request.model):
            # Background task to load model
            background_tasks.add_task(
                inference_engine.load_model, 
                request.model
            )
            raise HTTPException(
                status_code=503,
                detail=f"Model {request.model} is loading, please try again"
            )
        
        # Perform inference
        result = await inference_engine.predict(
            model_name=request.model,
            input_data=request.input_data,
            parameters=request.parameters,
            request_id=request_id
        )
        
        # Calculate latency
        end_time = datetime.utcnow()
        latency = (end_time - start_time).total_seconds()
        
        # Record metrics
        background_tasks.add_task(
            metrics_collector.record_request,
            model=request.model,
            latency=latency,
            success=True
        )
        
        # Build response
        response = InferenceResponse(
            request_id=request_id,
            model=request.model,
            predictions=result["predictions"],
            confidence=result.get("confidence"),
            latency=latency,
            timestamp=end_time.isoformat(),
            metadata={
                "cached": result.get("cached", False),
                "model_version": result.get("model_version"),
                "processing_time": result.get("processing_time")
            }
        )
        
        logger.info(f"Inference request {request_id} completed - Latency: {latency:.3f}s")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        # Calculate latency for failed request
        end_time = datetime.utcnow()
        latency = (end_time - start_time).total_seconds()
        
        # Record failed metrics
        background_tasks.add_task(
            metrics_collector.record_request,
            model=request.model,
            latency=latency,
            success=False
        )
        
        logger.error(f"Inference request {request_id} failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Inference failed: {str(e)}"
        )

@router.post("/batch", response_model=List[InferenceResponse])
async def batch_predict(
    request: BatchInferenceRequest,
    background_tasks: BackgroundTasks,
    inference_engine=Depends(get_inference_engine),
    metrics_collector=Depends(get_metrics_collector)
):
    """
    Batch inference prediction endpoint
    
    Processes multiple inputs in parallel for improved throughput
    """
    start_time = datetime.utcnow()
    batch_id = str(uuid.uuid4())
    
    logger.info(f"Batch inference request {batch_id} - {len(request.inputs)} items")
    
    try:
        # Validate batch size
        if len(request.inputs) > 100:  # Configurable limit
            raise HTTPException(
                status_code=400,
                detail="Batch size exceeds maximum limit of 100"
            )
        
        # Check if model is loaded
        if not await inference_engine.is_model_loaded(request.model):
            raise HTTPException(
                status_code=503,
                detail=f"Model {request.model} is not loaded"
            )
        
        # Process batch
        results = await inference_engine.batch_predict(
            model_name=request.model,
            inputs=request.inputs,
            parameters=request.parameters,
            batch_id=batch_id
        )
        
        # Calculate latency
        end_time = datetime.utcnow()
        total_latency = (end_time - start_time).total_seconds()
        avg_latency = total_latency / len(request.inputs)
        
        # Record metrics
        background_tasks.add_task(
            metrics_collector.record_batch_request,
            model=request.model,
            batch_size=len(request.inputs),
            latency=total_latency,
            success=True
        )
        
        # Build responses
        responses = []
        for i, result in enumerate(results):
            response = InferenceResponse(
                request_id=f"{batch_id}-{i}",
                model=request.model,
                predictions=result["predictions"],
                confidence=result.get("confidence"),
                latency=avg_latency,
                timestamp=end_time.isoformat(),
                metadata={
                    "batch_id": batch_id,
                    "batch_index": i,
                    "cached": result.get("cached", False)
                }
            )
            responses.append(response)
        
        logger.info(f"Batch inference {batch_id} completed - Total latency: {total_latency:.3f}s")
        return responses
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch inference {batch_id} failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Batch inference failed: {str(e)}"
        )

@router.post("/stream")
async def stream_predict(
    request: InferenceRequest,
    inference_engine=Depends(get_inference_engine)
):
    """
    Streaming inference endpoint for real-time predictions
    """
    request_id = str(uuid.uuid4())
    
    async def generate_stream():
        try:
            async for chunk in inference_engine.stream_predict(
                model_name=request.model,
                input_data=request.input_data,
                parameters=request.parameters,
                request_id=request_id
            ):
                yield f"data: {json.dumps(chunk)}\n\n"
        except Exception as e:
            error_chunk = {
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
            yield f"data: {json.dumps(error_chunk)}\n\n"
    
    return StreamingResponse(
        generate_stream(),
        media_type="text/stream-server-sent-events",
        headers={"Cache-Control": "no-cache"}
    )

@router.get("/models", response_model=List[Dict[str, Any]])
async def list_models(inference_engine=Depends(get_inference_engine)):
    """List available models and their status"""
    try:
        models = await inference_engine.list_models()
        return models
    except Exception as e:
        logger.error(f"Failed to list models: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve model list"
        )

@router.post("/models/{model_name}/load")
async def load_model(
    model_name: str,
    background_tasks: BackgroundTasks,
    inference_engine=Depends(get_inference_engine)
):
    """Load a specific model"""
    try:
        # Start loading in background
        background_tasks.add_task(inference_engine.load_model, model_name)
        
        return {
            "message": f"Loading model {model_name}",
            "status": "loading",
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to load model {model_name}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load model: {str(e)}"
        )

@router.delete("/models/{model_name}/unload")
async def unload_model(
    model_name: str,
    inference_engine=Depends(get_inference_engine)
):
    """Unload a specific model to free memory"""
    try:
        await inference_engine.unload_model(model_name)
        
        return {
            "message": f"Model {model_name} unloaded successfully",
            "status": "unloaded",
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to unload model {model_name}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to unload model: {str(e)}"
        )

@router.get("/metrics", response_model=InferenceMetrics)
async def get_inference_metrics(metrics_collector=Depends(get_metrics_collector)):
    """Get inference engine metrics"""
    try:
        metrics = await metrics_collector.get_inference_metrics()
        return InferenceMetrics(**metrics)
    except Exception as e:
        logger.error(f"Failed to get metrics: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve metrics"
        )

@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.utcnow().isoformat()
    )