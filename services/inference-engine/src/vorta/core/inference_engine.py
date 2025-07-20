"""
VORTA Core Inference Engine
Production-ready AI model inference engine with caching, monitoring,
and optimization
"""

import asyncio
import hashlib
import json
import logging
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from typing import Any, AsyncGenerator, Dict, List, Optional

# ML/AI imports (would be actual libraries in production)
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None

try:
    import transformers  # noqa: F401
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

from ..api.models import ModelType, Prediction
from ..config.settings import Settings

logger = logging.getLogger(__name__)


class ModelManager:
    """Manages loading, unloading, and caching of AI models"""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.loaded_models: Dict[str, Any] = {}
        self.model_metadata: Dict[str, Dict] = {}
        self.load_times: Dict[str, float] = {}
        self.access_times: Dict[str, datetime] = {}
        self.lock = threading.RLock()
        self.executor = ThreadPoolExecutor(max_workers=2)
        
    async def load_model(self, model_name: str) -> bool:
        """Load a model asynchronously"""
        logger.info(f"Loading model: {model_name}")
        start_time = time.time()
        
        try:
            with self.lock:
                if model_name in self.loaded_models:
                    logger.info(f"Model {model_name} already loaded")
                    self.access_times[model_name] = datetime.now()
                    return True
                
                # Check if we need to unload models due to memory constraints
                max_models = self.settings.models.max_models_loaded
                if len(self.loaded_models) >= max_models:
                    await self._unload_oldest_model()
            
            # Load model in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            model_data = await loop.run_in_executor(
                self.executor,
                self._load_model_sync,
                model_name
            )
            
            if model_data:
                with self.lock:
                    self.loaded_models[model_name] = model_data
                    self.load_times[model_name] = time.time() - start_time
                    self.access_times[model_name] = datetime.now()
                    
                load_time = self.load_times[model_name]
                logger.info(f"Model {model_name} loaded successfully "
                            f"in {load_time:.2f}s")
                return True
            else:
                logger.error(f"Failed to load model {model_name}")
                return False
                
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {e}")
            logger.error(traceback.format_exc())
            return False
    
    def _load_model_sync(self, model_name: str) -> Optional[Dict]:
        """Synchronous model loading (runs in thread pool)"""
        try:
            # Simulate model loading (in production, use actual libraries)
            if HAS_TORCH and model_name.startswith("llama"):
                # Example for transformer model
                model_data = {
                    "type": "transformer",
                    "model": f"mock_model_{model_name}",  # Actual model
                    "tokenizer": f"mock_tokenizer_{model_name}",  # Actual tokenizer  # noqa: E501
                    "config": {
                        "max_length": 2048,
                        "temperature": 0.7,
                        "top_p": 0.9
                    }
                }
            else:
                # Generic model data
                model_data = {
                    "type": "generic",
                    "model": f"mock_model_{model_name}",
                    "config": {
                        "input_size": 1024,
                        "output_size": 512
                    }
                }
            
            # Store metadata
            self.model_metadata[model_name] = {
                "name": model_name,
                "type": self._get_model_type(model_name),
                "version": "1.0.0",
                "size_mb": 100,  # Would be actual size
                "loaded_at": datetime.now().isoformat()
            }
            
            return model_data
            
        except Exception as e:
            logger.error(f"Sync model loading failed for {model_name}: {e}")
            return None
    
    def _get_model_type(self, model_name: str) -> ModelType:
        """Determine model type from name"""
        text_models = ["llama", "gpt", "bert", "t5"]
        if any(x in model_name.lower() for x in text_models):
            return ModelType.TEXT_GENERATION
        
        image_models = ["resnet", "efficientnet", "vgg"]
        if any(x in model_name.lower() for x in image_models):
            return ModelType.IMAGE_CLASSIFICATION
        
        if "embed" in model_name.lower():
            return ModelType.EMBEDDING
        
        return ModelType.TEXT_GENERATION
    
    def unload_model(self, model_name: str) -> bool:
        """Unload a specific model"""
        try:
            with self.lock:
                if model_name in self.loaded_models:
                    del self.loaded_models[model_name]
                    del self.access_times[model_name]
                    if model_name in self.load_times:
                        del self.load_times[model_name]
                    if model_name in self.model_metadata:
                        del self.model_metadata[model_name]
                    
                    logger.info(f"Model {model_name} unloaded successfully")
                    return True
                else:
                    logger.warning(f"Model {model_name} was not loaded")
                    return False
        except Exception as e:
            logger.error(f"Error unloading model {model_name}: {e}")
            return False
    
    async def _unload_oldest_model(self):
        """Unload the least recently accessed model"""
        if not self.loaded_models:
            return
        
        oldest_model = min(self.access_times.items(), key=lambda x: x[1])
        await self.unload_model(oldest_model[0])
    
    def is_model_loaded(self, model_name: str) -> bool:
        """Check if model is loaded"""
        with self.lock:
            return model_name in self.loaded_models
    
    def get_loaded_models(self) -> List[Dict]:
        """Get list of loaded models with metadata"""
        with self.lock:
            return [
                {
                    "name": name,
                    "metadata": self.model_metadata.get(name, {}),
                    "load_time": self.load_times.get(name),
                    "last_accessed": self.access_times.get(name)
                }
                for name in self.loaded_models.keys()
            ]

class CacheManager:
    """Manages inference result caching"""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.cache: Dict[str, Any] = {}
        self.cache_times: Dict[str, datetime] = {}
        self.lock = threading.RLock()
        self.hits = 0
        self.misses = 0
    
    def _generate_cache_key(self, model_name: str, input_data: Dict, parameters: Dict) -> str:
        """Generate cache key from input"""
        key_data = {
            "model": model_name,
            "input": input_data,
            "params": parameters
        }
        key_string = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(key_string.encode()).hexdigest()
    
    def get(self, model_name: str, input_data: Dict, parameters: Dict) -> Optional[Dict]:
        """Get cached result if available and not expired"""
        if not self.settings.cache_enabled:
            return None
        
        cache_key = self._generate_cache_key(model_name, input_data, parameters)
        
        with self.lock:
            if cache_key in self.cache:
                cache_time = self.cache_times[cache_key]
                ttl = timedelta(seconds=self.settings.redis.ttl)
                if datetime.now() - cache_time < ttl:
                    self.hits += 1
                    result = self.cache[cache_key].copy()
                    result["cached"] = True
                    return result
                else:
                    # Expired
                    del self.cache[cache_key]
                    del self.cache_times[cache_key]
            
            self.misses += 1
            return None
    
    def set(self, model_name: str, input_data: Dict, parameters: Dict, result: Dict):
        """Cache result"""
        if not self.settings.cache_enabled:
            return
        
        cache_key = self._generate_cache_key(model_name, input_data, parameters)
        
        with self.lock:
            self.cache[cache_key] = result.copy()
            self.cache_times[cache_key] = datetime.now()
            
            # Simple cache eviction if too large
            if len(self.cache) > 1000:  # Configurable limit
                # Remove oldest entry
                oldest_item = min(self.cache_times.items(), key=lambda x: x[1])
                oldest_key = oldest_item[0]
                del self.cache[oldest_key]
                del self.cache_times[oldest_key]
    
    def get_stats(self) -> Dict:
        """Get cache statistics"""
        with self.lock:
            total = self.hits + self.misses
            hit_rate = self.hits / total if total > 0 else 0.0
            
            return {
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate": hit_rate,
                "cache_size": len(self.cache)
            }


class InferenceEngine:
    """Main inference engine class"""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.model_manager = ModelManager(settings)
        self.cache_manager = CacheManager(settings)
        self.is_initialized = False
        self.executor = ThreadPoolExecutor(max_workers=4)
        
    async def initialize(self):
        """Initialize the inference engine"""
        try:
            logger.info("Initializing VORTA Inference Engine...")
            
            # Auto-load configured models
            if self.settings.models.auto_load_models:
                for model_name in self.settings.models.text_generation_models[:2]:  # Load first 2
                    await self.model_manager.load_model(model_name)
            
            self.is_initialized = True
            logger.info("Inference engine initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize inference engine: {e}")
            raise
    
    def is_ready(self) -> bool:
        """Check if engine is ready"""
        return self.is_initialized
    
    async def predict(self, model_name: str, input_data: Dict[str, Any], 
                     parameters: Optional[Dict[str, Any]] = None, 
                     request_id: Optional[str] = None) -> Dict[str, Any]:
        """Single model prediction"""
        parameters = parameters or {}
        
        # Check cache first
        cached_result = self.cache_manager.get(model_name, input_data, parameters)
        if cached_result:
            logger.debug(f"Cache hit for request {request_id}")
            return cached_result
        
        # Ensure model is loaded
        if not self.model_manager.is_model_loaded(model_name):
            await self.model_manager.load_model(model_name)
        
        # Perform inference
        start_time = time.time()
        result = await self._run_inference(model_name, input_data, parameters)
        processing_time = time.time() - start_time
        
        # Add metadata
        result.update({
            "processing_time": processing_time,
            "model_version": "1.0.0",
            "cached": False
        })
        
        # Cache result
        self.cache_manager.set(model_name, input_data, parameters, result)
        
        return result
    
    async def _run_inference(self, model_name: str, input_data: Dict, parameters: Dict) -> Dict:
        """Run actual model inference"""
        try:
            # Simulate model inference (replace with actual model calls)
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self.executor,
                self._inference_sync,
                model_name, input_data, parameters
            )
            return result
            
        except Exception as e:
            logger.error(f"Inference failed for model {model_name}: {e}")
            raise
    
    def _inference_sync(self, model_name: str, input_data: Dict, parameters: Dict) -> Dict:
        """Synchronous inference (runs in thread pool)"""
        # Mock inference - replace with actual model inference
        model_data = self.model_manager.loaded_models[model_name]
        
        # Simulate processing time
        time.sleep(0.1)  # Remove in production
        
        # Generate mock predictions based on input type
        if "text" in input_data:
            predictions = [
                Prediction(
                    text=f"Generated response for: {input_data['text'][:50]}...",
                    score=0.95,
                    metadata={"tokens": 100, "model": model_name}
                ).dict()
            ]
        elif "image" in input_data:
            predictions = [
                Prediction(
                    label="object_detected",
                    score=0.89,
                    bbox=[10, 20, 100, 150],
                    metadata={"confidence": 0.89}
                ).dict()
            ]
        else:
            predictions = [
                Prediction(
                    text="Generic prediction",
                    score=0.8,
                    metadata={"type": "generic"}
                ).dict()
            ]
        
        return {
            "predictions": predictions,
            "confidence": predictions[0]["score"] if predictions else 0.0
        }
    
    async def batch_predict(self, model_name: str, inputs: List[Dict[str, Any]], 
                           parameters: Optional[Dict[str, Any]] = None,
                           batch_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Batch prediction processing"""
        parameters = parameters or {}
        
        # Ensure model is loaded
        if not self.model_manager.is_model_loaded(model_name):
            await self.model_manager.load_model(model_name)
        
        # Process in parallel
        tasks = []
        for i, input_data in enumerate(inputs):
            task = self.predict(
                model_name=model_name,
                input_data=input_data,
                parameters=parameters,
                request_id=f"{batch_id}-{i}" if batch_id else None
            )
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Batch item {i} failed: {result}")
                processed_results.append({
                    "predictions": [],
                    "confidence": 0.0,
                    "error": str(result)
                })
            else:
                processed_results.append(result)
        
        return processed_results
    
    async def stream_predict(self, model_name: str, input_data: Dict[str, Any],
                            parameters: Optional[Dict[str, Any]] = None,
                            request_id: Optional[str] = None) -> AsyncGenerator[Dict, None]:
        """Streaming prediction for real-time responses"""
        parameters = parameters or {}
        
        # Ensure model is loaded
        if not self.model_manager.is_model_loaded(model_name):
            await self.model_manager.load_model(model_name)
        
        # Simulate streaming response (replace with actual streaming)
        chunks = [
            {"chunk_id": 0, "data": {"partial_text": "Starting"}, "is_final": False},
            {"chunk_id": 1, "data": {"partial_text": "Starting generation"}, "is_final": False},
            {"chunk_id": 2, "data": {"partial_text": "Starting generation..."}, "is_final": False},
            {"chunk_id": 3, "data": {"final_text": "Complete response", "confidence": 0.92}, "is_final": True}
        ]
        
        for chunk in chunks:
            chunk["timestamp"] = datetime.utcnow().isoformat()
            chunk["request_id"] = request_id
            yield chunk
            await asyncio.sleep(0.2)  # Simulate streaming delay
    
    async def list_models(self) -> List[Dict[str, Any]]:
        """List all available models"""
        loaded_models = self.model_manager.get_loaded_models()
        
        # Add configured models that aren't loaded
        all_models = []
        for model_name in (self.settings.models.text_generation_models + 
                          self.settings.models.image_classification_models +
                          self.settings.models.embedding_models):
            is_loaded = any(m["name"] == model_name for m in loaded_models)
            
            model_info = {
                "name": model_name,
                "type": self.model_manager._get_model_type(model_name).value,
                "loaded": is_loaded,
                "status": "loaded" if is_loaded else "available"
            }
            
            if is_loaded:
                loaded_info = next(m for m in loaded_models if m["name"] == model_name)
                model_info.update(loaded_info["metadata"])
                model_info["load_time"] = loaded_info["load_time"]
            
            all_models.append(model_info)
        
        return all_models
    
    async def is_model_loaded(self, model_name: str) -> bool:
        """Check if model is loaded"""
        return self.model_manager.is_model_loaded(model_name)
    
    async def load_model(self, model_name: str) -> bool:
        """Load a model"""
        return await self.model_manager.load_model(model_name)
    
    async def unload_model(self, model_name: str) -> bool:
        """Unload a model"""
        return await self.model_manager.unload_model(model_name)
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get engine statistics"""
        cache_stats = self.cache_manager.get_stats()
        loaded_models = self.model_manager.get_loaded_models()
        
        return {
            "models_loaded": len(loaded_models),
            "cache_stats": cache_stats,
            "loaded_models": [m["name"] for m in loaded_models],
            "is_ready": self.is_ready()
        }
    
    async def shutdown(self):
        """Shutdown the inference engine"""
        logger.info("Shutting down inference engine...")
        
        # Unload all models
        for model_name in list(self.model_manager.loaded_models.keys()):
            await self.model_manager.unload_model(model_name)
        
        # Shutdown thread pools
        self.executor.shutdown(wait=True)
        self.model_manager.executor.shutdown(wait=True)
        
        self.is_initialized = False
        logger.info("Inference engine shutdown completed")