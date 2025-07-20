"""
VORTA Inference Engine Dependencies
Dependency injection for FastAPI endpoints
"""

import logging
from contextlib import asynccontextmanager
from functools import lru_cache
from typing import Optional

from fastapi import Depends, HTTPException, status

# Import core components
from ..core.inference_engine import InferenceEngine
from ..core.metrics import MetricsCollector

logger = logging.getLogger(__name__)

# Global instances (singleton pattern)
_inference_engine: Optional[InferenceEngine] = None
_metrics_collector: Optional[MetricsCollector] = None

@lru_cache()
def get_app_settings():
    """Get application settings (cached)"""
    from ..config.settings import Settings
    return Settings()

async def get_inference_engine() -> InferenceEngine:
    """
    Dependency to get the inference engine instance
    Creates singleton instance if not exists
    """
    global _inference_engine
    
    if _inference_engine is None:
        settings = get_app_settings()
        _inference_engine = InferenceEngine(settings)
        await _inference_engine.initialize()
        logger.info("Inference engine initialized")
    
    if not _inference_engine.is_ready():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Inference engine is not ready"
        )
    
    return _inference_engine

async def get_metrics_collector() -> MetricsCollector:
    """
    Dependency to get the metrics collector instance
    Creates singleton instance if not exists
    """
    global _metrics_collector
    
    if _metrics_collector is None:
        settings = get_app_settings()
        _metrics_collector = MetricsCollector(settings)
        await _metrics_collector.initialize()
        logger.info("Metrics collector initialized")
    
    return _metrics_collector

async def get_health_checker():
    """
    Dependency for health check endpoints
    Returns basic system health information
    """
    try:
        # Basic health checks
        settings = get_app_settings()
        
        health_data = {
            "service": "inference-engine",
            "status": "healthy",
            "version": getattr(settings, 'version', '1.0.0'),
            "checks": {
                "settings": True,
                "memory": True,  # Could add actual memory check
                "disk": True,    # Could add actual disk check
            }
        }
        
        # Check inference engine if available
        if _inference_engine is not None:
            health_data["checks"]["inference_engine"] = _inference_engine.is_ready()
        
        # Check metrics collector if available
        if _metrics_collector is not None:
            health_data["checks"]["metrics"] = _metrics_collector.is_ready()
        
        return health_data
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Health check failed: {str(e)}"
        )

async def validate_api_key(api_key: Optional[str] = None):
    """
    Dependency for API key validation
    Can be used to protect endpoints if needed
    """
    settings = get_app_settings()
    
    # If API key validation is disabled, allow all requests
    if not getattr(settings, 'require_api_key', False):
        return True
    
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key is required",
            headers={"WWW-Authenticate": "Bearer"}
        )
    
    # Validate API key against configured keys
    valid_keys = getattr(settings, 'api_keys', [])
    if api_key not in valid_keys:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "Bearer"}
        )
    
    return True

async def get_current_user(api_key: str = Depends(validate_api_key)):
    """
    Dependency to get current user information
    Can be extended for user-based access control
    """
    # For now, return basic user info
    # In production, this could query a user database
    return {
        "user_id": "anonymous",
        "permissions": ["inference"],
        "quota": {
            "requests_per_minute": 100,
            "requests_per_day": 10000
        }
    }

@asynccontextmanager
async def lifespan_context():
    """
    Context manager for application lifespan
    Handles initialization and cleanup
    """
    logger.info("Starting VORTA Inference Engine...")
    
    try:
        # Initialize all services using centralized dependencies
        from ..core.dependencies import initialize_all_services, shutdown_all_services
        
        await initialize_all_services()
        
        logger.info("🚀 VORTA Inference Engine started successfully")
        yield
        
    except Exception as e:
        logger.error(f"❌ Failed to start inference engine: {e}")
        raise
    finally:
        # Cleanup all services
        logger.info("🔄 Shutting down VORTA Inference Engine...")
        
        from ..core.dependencies import shutdown_all_services
        await shutdown_all_services()
        
        logger.info("✅ VORTA Inference Engine shutdown completed")

# Utility dependencies
def get_request_timeout(default: int = 30) -> int:
    """Get request timeout from settings or use default"""
    settings = get_app_settings()
    return getattr(settings, 'default_timeout', default)

def get_max_batch_size(default: int = 100) -> int:
    """Get maximum batch size from settings or use default"""
    settings = get_app_settings()
    return getattr(settings, 'max_batch_size', default)

async def check_rate_limit(user_id: str = "anonymous"):
    """
    Rate limiting dependency
    Can be used to limit requests per user/IP
    """
    # For now, just log the request
    # In production, implement actual rate limiting
    logger.debug(f"Request from user: {user_id}")
    return True