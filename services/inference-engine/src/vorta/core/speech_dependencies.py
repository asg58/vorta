"""
VORTA Speech Service Dependencies
Dependency injection for speech recognition and TTS services
"""

import logging
from functools import lru_cache
from typing import Optional

from ..config.settings import get_settings
from .speech_recognition import WhisperSpeechRecognizer
from .tts_service import VortaTTSService

logger = logging.getLogger(__name__)

# Global service instances
_speech_recognizer: Optional[WhisperSpeechRecognizer] = None
_tts_service: Optional[VortaTTSService] = None

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

async def initialize_speech_services():
    """Initialize all speech services"""
    logger.info("Initializing speech services...")
    
    try:
        # Initialize speech recognition
        recognizer = get_speech_recognizer()
        await recognizer.initialize()
        logger.info("Speech recognition service initialized")
        
        # Initialize TTS service  
        tts_service = get_tts_service()
        await tts_service.initialize()
        logger.info("TTS service initialized")
        
        logger.info("All speech services initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize speech services: {e}")
        raise

async def shutdown_speech_services():
    """Shutdown all speech services"""
    logger.info("Shutting down speech services...")
    
    try:
        if _speech_recognizer:
            await _speech_recognizer.shutdown()
            logger.info("Speech recognition service shutdown")
        
        if _tts_service:
            await _tts_service.shutdown()
            logger.info("TTS service shutdown")
        
        logger.info("All speech services shutdown completed")
        
    except Exception as e:
        logger.error(f"Error during speech services shutdown: {e}")
