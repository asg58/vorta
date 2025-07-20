"""
VORTA Text-to-Speech Service
High-quality TTS with multiple provider support
"""

import asyncio
import base64
import logging
import time
from enum import Enum
from typing import Any, Dict, List

# TTS provider imports
try:
    import openai
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False
    logging.warning("OpenAI TTS not available")

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

from ..config.settings import Settings

logger = logging.getLogger(__name__)

class TTSProvider(str, Enum):
    """Supported TTS providers"""
    OPENAI = "openai"
    ELEVEN_LABS = "eleven_labs" 
    AZURE = "azure"
    LOCAL = "local"

class VoiceProfile:
    """Voice profile configuration"""
    def __init__(self, 
                 provider: TTSProvider,
                 voice_id: str,
                 name: str,
                 language: str = "en",
                 gender: str = "neutral",
                 description: str = ""):
        self.provider = provider
        self.voice_id = voice_id
        self.name = name
        self.language = language
        self.gender = gender
        self.description = description

class VortaTTSService:
    """High-quality Text-to-Speech service with multiple providers"""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.providers: Dict[TTSProvider, Any] = {}
        self.is_initialized = False
        
        # Voice profiles
        self.voices = self._initialize_voice_profiles()
        
        # TTS settings
        self.default_voice = "alloy"  # OpenAI default
        self.default_provider = TTSProvider.OPENAI
        self.output_format = "mp3"
        self.sample_rate = 22050
        
        # OpenAI TTS settings
        self.openai_client = None
        
    def _initialize_voice_profiles(self) -> Dict[str, VoiceProfile]:
        """Initialize available voice profiles"""
        voices = {}
        
        # OpenAI TTS voices
        openai_voices = [
            VoiceProfile(TTSProvider.OPENAI, "alloy", "Alloy", "en", "neutral", "Balanced, natural voice"),
            VoiceProfile(TTSProvider.OPENAI, "echo", "Echo", "en", "male", "Deep, resonant male voice"),
            VoiceProfile(TTSProvider.OPENAI, "fable", "Fable", "en", "neutral", "Warm, engaging storyteller"),
            VoiceProfile(TTSProvider.OPENAI, "onyx", "Onyx", "en", "male", "Authoritative male voice"),
            VoiceProfile(TTSProvider.OPENAI, "nova", "Nova", "en", "female", "Bright, energetic female"),
            VoiceProfile(TTSProvider.OPENAI, "shimmer", "Shimmer", "en", "female", "Gentle, soothing female")
        ]
        
        for voice in openai_voices:
            voices[voice.voice_id] = voice
            
        return voices
        
    async def initialize(self):
        """Initialize TTS service and providers"""
        try:
            logger.info("Initializing VORTA TTS Service...")
            
            # Debug OpenAI API key
            api_key = self.settings.openai_api_key
            logger.info(f"OpenAI API Key: {'Found' if api_key else 'NOT FOUND'}")
            logger.info(f"HAS_OPENAI: {HAS_OPENAI}")
            
            # Initialize OpenAI TTS
            if HAS_OPENAI and api_key:
                await self._initialize_openai_tts()
            else:
                logger.warning("OpenAI TTS initialization skipped - missing API key or OpenAI library")
            
            self.is_initialized = True
            logger.info("VORTA TTS Service initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize TTS service: {e}")
            raise
    
    async def _initialize_openai_tts(self):
        """Initialize OpenAI TTS provider"""
        try:
            logger.info(f"Initializing OpenAI TTS with key: {self.settings.openai_api_key[:20]}..." if self.settings.openai_api_key else "No OpenAI API key found")
            
            self.openai_client = openai.OpenAI(
                api_key=self.settings.openai_api_key
            )
            
            self.providers[TTSProvider.OPENAI] = {
                "client": self.openai_client,
                "models": ["tts-1", "tts-1-hd"],
                "default_model": "tts-1"  # Faster, good quality
            }
            
            logger.info("✅ OpenAI TTS provider initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize OpenAI TTS: {e}")
            # Don't raise exception, just log error
            pass
            
    async def synthesize_speech(self,
                               text: str,
                               voice: str = None,
                               provider: TTSProvider = None,
                               options: Dict = None) -> Dict[str, Any]:
        """
        Convert text to speech
        
        Args:
            text: Text to synthesize
            voice: Voice profile to use
            provider: TTS provider to use
            options: Additional synthesis options
            
        Returns:
            Synthesis result with audio data and metadata
        """
        try:
            voice = voice or self.default_voice
            provider = provider or self.default_provider
            options = options or {}
            
            logger.info(f"Synthesizing speech: '{text[:50]}...' with voice '{voice}'")
            
            start_time = time.time()
            
            # Route to appropriate provider
            if provider == TTSProvider.OPENAI:
                result = await self._synthesize_openai(text, voice, options)
            else:
                raise ValueError(f"Provider {provider} not implemented yet")
            
            processing_time = time.time() - start_time
            
            # Add metadata
            result.update({
                "processing_time": processing_time,
                "provider": provider.value,
                "voice_used": voice,
                "text_length": len(text),
                "character_count": len(text)
            })
            
            logger.info(f"Speech synthesis completed in {processing_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"Speech synthesis failed: {e}")
            return {
                "audio_data": None,
                "error": str(e),
                "processing_time": 0
            }
    
    async def _synthesize_openai(self, text: str, voice: str, options: Dict) -> Dict[str, Any]:
        """Synthesize speech using OpenAI TTS"""
        try:
            if TTSProvider.OPENAI not in self.providers:
                raise Exception("OpenAI TTS not available")
            
            provider_info = self.providers[TTSProvider.OPENAI]
            client = provider_info["client"]
            model = options.get("model", provider_info["default_model"])
            
            # Configure TTS request
            tts_options = {
                "model": model,
                "input": text,
                "voice": voice,
                "response_format": options.get("format", "mp3"),
                "speed": options.get("speed", 1.0)  # 0.25 to 4.0
            }
            
            # Run in executor to avoid blocking
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: client.audio.speech.create(**tts_options)
            )
            
            # Get audio data
            audio_bytes = response.content
            
            return {
                "audio_data": audio_bytes,
                "format": tts_options["response_format"],
                "model_used": model,
                "audio_size": len(audio_bytes),
                "base64_audio": base64.b64encode(audio_bytes).decode('utf-8'),
                "success": True
            }
            
        except Exception as e:
            logger.error(f"OpenAI TTS synthesis failed: {e}")
            return {
                "audio_data": None,
                "error": str(e),
                "success": False
            }
    
    async def synthesize_streaming(self,
                                  text_chunks: List[str],
                                  voice: str = None,
                                  provider: TTSProvider = None) -> List[Dict[str, Any]]:
        """
        Synthesize multiple text chunks for streaming
        
        Args:
            text_chunks: List of text chunks to synthesize
            voice: Voice profile to use
            provider: TTS provider to use
            
        Returns:
            List of synthesis results
        """
        try:
            tasks = []
            for i, chunk in enumerate(text_chunks):
                task = self.synthesize_speech(
                    chunk,
                    voice=voice,
                    provider=provider,
                    options={"chunk_index": i}
                )
                tasks.append(task)
            
            # Process chunks concurrently
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Filter successful results
            audio_chunks = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Chunk {i} synthesis failed: {result}")
                    continue
                    
                if result.get("success"):
                    audio_chunks.append({
                        "chunk_index": i,
                        "audio_data": result["audio_data"],
                        "format": result.get("format", "mp3"),
                        "base64_audio": result.get("base64_audio")
                    })
            
            return audio_chunks
            
        except Exception as e:
            logger.error(f"Streaming synthesis failed: {e}")
            return []
    
    async def get_available_voices(self, provider: TTSProvider = None) -> List[Dict[str, Any]]:
        """Get available voices for a provider"""
        try:
            if provider is None:
                # Return all voices
                voices = []
                for voice in self.voices.values():
                    voices.append({
                        "voice_id": voice.voice_id,
                        "name": voice.name,
                        "provider": voice.provider.value,
                        "language": voice.language,
                        "gender": voice.gender,
                        "description": voice.description
                    })
                return voices
            else:
                # Return voices for specific provider
                voices = []
                for voice in self.voices.values():
                    if voice.provider == provider:
                        voices.append({
                            "voice_id": voice.voice_id,
                            "name": voice.name,
                            "language": voice.language,
                            "gender": voice.gender,
                            "description": voice.description
                        })
                return voices
                
        except Exception as e:
            logger.error(f"Failed to get available voices: {e}")
            return []
    
    async def validate_voice(self, voice_id: str, provider: TTSProvider = None) -> bool:
        """Validate if a voice is available"""
        try:
            if voice_id in self.voices:
                voice = self.voices[voice_id]
                if provider is None or voice.provider == provider:
                    return True
            return False
            
        except Exception as e:
            logger.error(f"Voice validation failed: {e}")
            return False
    
    def split_text_for_synthesis(self, text: str, max_length: int = 4000) -> List[str]:
        """
        Split long text into chunks suitable for TTS synthesis
        
        Args:
            text: Text to split
            max_length: Maximum characters per chunk
            
        Returns:
            List of text chunks
        """
        if len(text) <= max_length:
            return [text]
        
        chunks = []
        sentences = text.split('. ')
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk + sentence + '. ') <= max_length:
                current_chunk += sentence + '. '
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + '. '
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    async def get_service_status(self) -> Dict[str, Any]:
        """Get TTS service status"""
        return {
            "initialized": self.is_initialized,
            "providers": {
                provider.value: {
                    "available": provider in self.providers,
                    "config": self.providers.get(provider, {})
                }
                for provider in TTSProvider
            },
            "total_voices": len(self.voices),
            "default_voice": self.default_voice,
            "default_provider": self.default_provider.value,
            "supported_formats": ["mp3", "opus", "aac", "flac"]
        }
    
    async def shutdown(self):
        """Cleanup and shutdown TTS service"""
        logger.info("Shutting down VORTA TTS Service...")
        self.providers.clear()
        self.is_initialized = False
        logger.info("VORTA TTS Service shutdown completed")
