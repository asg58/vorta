"""
VORTA Core Dependencies
Centralized dependency injection for all services
"""

import logging
from functools import lru_cache
from typing import Optional

from ..config.settings import get_settings

# Import all service dependencies
from .inference_engine import InferenceEngine
from .metrics import MetricsCollector
from .speech_recognition import WhisperSpeechRecognizer
from .tts_service import VortaTTSService

logger = logging.getLogger(__name__)

# Global service instances
_inference_engine: Optional[InferenceEngine] = None
_speech_recognizer: Optional[WhisperSpeechRecognizer] = None
_tts_service: Optional[VortaTTSService] = None
_metrics_collector: Optional[MetricsCollector] = None

@lru_cache()
def get_inference_engine() -> InferenceEngine:
    """Get or create inference engine instance"""
    global _inference_engine
    
    if _inference_engine is None:
        settings = get_settings()
        _inference_engine = InferenceEngine(settings)
        logger.info("Inference engine instance created")
    
    return _inference_engine

@lru_cache()
def get_speech_recognizer() -> WhisperSpeechRecognizer:
    """Get or create speech recognition service instance"""
    global _speech_recognizer
    
    if _speech_recognizer is None:
        settings = get_settings()
        _speech_recognizer = WhisperSpeechRecognizer(settings)
        logger.info("Speech recognizer instance created")
    
    return _speech_recognizer

@lru_cache()
def get_tts_service() -> VortaTTSService:
    """Get or create TTS service instance"""
    global _tts_service
    
    if _tts_service is None:
        settings = get_settings()
        _tts_service = VortaTTSService(settings)
        logger.info("TTS service instance created")
    
    return _tts_service

@lru_cache()
def get_metrics_collector() -> MetricsCollector:
    """Get or create metrics collector instance"""
    global _metrics_collector
    
    if _metrics_collector is None:
        settings = get_settings()
        _metrics_collector = MetricsCollector(settings)
        logger.info("Metrics collector instance created")
    
    return _metrics_collector

async def initialize_all_services():
    """Initialize all core services"""
    logger.info("Initializing all VORTA services...")
    
    try:
        # Initialize metrics collector
        get_metrics_collector()
        logger.info("Metrics collector ready")
        
        # Initialize inference engine
        inference_engine = get_inference_engine()
        await inference_engine.initialize()
        logger.info("Inference engine initialized")
        
        # Initialize speech recognition
        recognizer = get_speech_recognizer()
        await recognizer.initialize()
        logger.info("Speech recognition initialized")
        
        # Initialize TTS service  
        tts_service = get_tts_service()
        await tts_service.initialize()
        logger.info("TTS service initialized")
        
        logger.info("🚀 All VORTA services initialized successfully!")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize VORTA services: {e}")
        raise

async def shutdown_all_services():
    """Shutdown all core services"""
    logger.info("Shutting down all VORTA services...")
    
    try:
        if _inference_engine:
            await _inference_engine.shutdown()
            logger.info("Inference engine shutdown")
        
        if _speech_recognizer:
            await _speech_recognizer.shutdown()
            logger.info("Speech recognition shutdown")
        
        if _tts_service:
            await _tts_service.shutdown()
            logger.info("TTS service shutdown")
        
        logger.info("🔄 All VORTA services shutdown completed")
        
    except Exception as e:
        logger.error(f"❌ Error during VORTA services shutdown: {e}")

def get_services_status() -> dict:
    """Get status of all services"""
    return {
        "inference_engine": _inference_engine is not None,
        "speech_recognizer": _speech_recognizer is not None and _speech_recognizer.is_initialized,
        "tts_service": _tts_service is not None and _tts_service.is_initialized,
        "metrics_collector": _metrics_collector is not None
    }
