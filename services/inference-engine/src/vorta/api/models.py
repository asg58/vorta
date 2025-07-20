"""
VORTA Inference Engine Models
Pydantic models for request/response validation and serialization
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, validator


class ModelType(str, Enum):
    """Available model types"""
    TEXT_GENERATION = "text-generation"
    IMAGE_CLASSIFICATION = "image-classification"
    OBJECT_DETECTION = "object-detection"
    SPEECH_RECOGNITION = "speech-recognition"
    MULTIMODAL = "multimodal"
    EMBEDDING = "embedding"
    TRANSLATION = "translation"

class InputType(str, Enum):
    """Input data types"""
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    MULTIMODAL = "multimodal"

class InferenceRequest(BaseModel):
    """Single inference request model"""
    model: str = Field(..., description="Model name to use for inference")
    input_data: Dict[str, Any] = Field(..., description="Input data for the model")
    parameters: Optional[Dict[str, Any]] = Field(default={}, description="Model-specific parameters")
    input_type: Optional[InputType] = Field(default=InputType.TEXT, description="Type of input data")
    timeout: Optional[int] = Field(default=30, ge=1, le=300, description="Request timeout in seconds")
    use_cache: Optional[bool] = Field(default=True, description="Whether to use caching")
    
    @validator('input_data')
    def validate_input_data(cls, v):
        if not v:
            raise ValueError("Input data cannot be empty")
        return v
    
    @validator('model')
    def validate_model_name(cls, v):
        if not v or not v.strip():
            raise ValueError("Model name cannot be empty")
        return v.strip()

class BatchInferenceRequest(BaseModel):
    """Batch inference request model"""
    model: str = Field(..., description="Model name to use for inference")
    inputs: List[Dict[str, Any]] = Field(..., description="List of input data for batch processing")
    parameters: Optional[Dict[str, Any]] = Field(default={}, description="Model-specific parameters")
    input_type: Optional[InputType] = Field(default=InputType.TEXT, description="Type of input data")
    timeout: Optional[int] = Field(default=60, ge=1, le=600, description="Request timeout in seconds")
    use_cache: Optional[bool] = Field(default=True, description="Whether to use caching")
    parallel_processing: Optional[bool] = Field(default=True, description="Whether to process in parallel")
    
    @validator('inputs')
    def validate_inputs(cls, v):
        if not v:
            raise ValueError("Inputs list cannot be empty")
        if len(v) > 100:  # Configurable limit
            raise ValueError("Batch size cannot exceed 100 items")
        return v
    
    @validator('model')
    def validate_model_name(cls, v):
        if not v or not v.strip():
            raise ValueError("Model name cannot be empty")
        return v.strip()

class Prediction(BaseModel):
    """Individual prediction result"""
    label: Optional[str] = Field(None, description="Predicted label")
    score: Optional[float] = Field(None, ge=0, le=1, description="Confidence score")
    bbox: Optional[List[float]] = Field(None, description="Bounding box coordinates [x, y, w, h]")
    text: Optional[str] = Field(None, description="Generated or recognized text")
    embedding: Optional[List[float]] = Field(None, description="Vector embedding")
    metadata: Optional[Dict[str, Any]] = Field(default={}, description="Additional prediction metadata")

class InferenceResponse(BaseModel):
    """Inference response model"""
    request_id: str = Field(..., description="Unique request identifier")
    model: str = Field(..., description="Model used for inference")
    predictions: List[Prediction] = Field(..., description="List of predictions")
    confidence: Optional[float] = Field(None, ge=0, le=1, description="Overall confidence score")
    latency: float = Field(..., ge=0, description="Processing latency in seconds")
    timestamp: str = Field(..., description="Response timestamp")
    metadata: Optional[Dict[str, Any]] = Field(default={}, description="Additional response metadata")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class ModelInfo(BaseModel):
    """Model information model"""
    name: str = Field(..., description="Model name")
    type: ModelType = Field(..., description="Model type")
    version: str = Field(..., description="Model version")
    description: Optional[str] = Field(None, description="Model description")
    input_types: List[InputType] = Field(..., description="Supported input types")
    output_format: str = Field(..., description="Output format description")
    loaded: bool = Field(..., description="Whether model is currently loaded")
    load_time: Optional[float] = Field(None, description="Model load time in seconds")
    memory_usage: Optional[int] = Field(None, description="Memory usage in MB")
    parameters: Optional[Dict[str, Any]] = Field(default={}, description="Model parameters")

class ErrorResponse(BaseModel):
    """Error response model"""
    error: str = Field(..., description="Error message")
    error_code: str = Field(..., description="Error code")
    details: Optional[Dict[str, Any]] = Field(default={}, description="Additional error details")
    timestamp: str = Field(..., description="Error timestamp")
    request_id: Optional[str] = Field(None, description="Request ID if available")