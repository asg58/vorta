"""
VORTA Speech Recognition Service
OpenAI Whisper integration for real-time speech-to-text processing
"""

import asyncio
import logging
import tempfile
import time
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List

# Audio processing imports
try:
    import librosa
    import numpy as np
    import whisper
    HAS_AUDIO_LIBS = True
except ImportError as e:
    HAS_AUDIO_LIBS = False
    logging.warning(f"Audio libraries not available: {e}")

from ..config.settings import Settings

logger = logging.getLogger(__name__)

class WhisperSpeechRecognizer:
    """OpenAI Whisper speech recognition service"""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.models: Dict[str, Any] = {}
        self.is_initialized = False
        self.supported_formats = ['.wav', '.mp3', '.m4a', '.flac', '.ogg']
        
        # Default model configuration
        self.default_model = "base"  # balance between speed and accuracy
        self.sample_rate = 16000  # Whisper's preferred sample rate
        
    async def initialize(self):
        """Initialize Whisper models"""
        try:
            logger.info("Initializing Whisper Speech Recognition...")
            
            if not HAS_AUDIO_LIBS:
                raise Exception("Audio processing libraries not available")
            
            # Load default Whisper model
            await self.load_model(self.default_model)
            
            self.is_initialized = True
            logger.info("Whisper Speech Recognition initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Whisper: {e}")
            raise
    
    async def load_model(self, model_name: str = "base"):
        """Load a specific Whisper model"""
        try:
            if model_name in self.models:
                logger.info(f"Whisper model {model_name} already loaded")
                return True
            
            logger.info(f"Loading Whisper model: {model_name}")
            start_time = time.time()
            
            # Load model in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            model = await loop.run_in_executor(
                None, 
                whisper.load_model, 
                model_name
            )
            
            self.models[model_name] = model
            load_time = time.time() - start_time
            
            logger.info(f"Whisper model {model_name} loaded in {load_time:.2f}s")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load Whisper model {model_name}: {e}")
            return False
    
    async def transcribe_audio(self, audio_data: bytes, 
                              model_name: str = None,
                              language: str = None,
                              options: Dict = None) -> Dict[str, Any]:
        """
        Transcribe audio data to text
        
        Args:
            audio_data: Raw audio bytes
            model_name: Whisper model to use
            language: Source language (auto-detect if None)
            options: Additional transcription options
        
        Returns:
            Transcription result with text, confidence, segments, etc.
        """
        try:
            model_name = model_name or self.default_model
            options = options or {}
            
            # Ensure model is loaded
            if model_name not in self.models:
                await self.load_model(model_name)
            
            model = self.models[model_name]
            
            # Process audio data
            audio_array = await self._process_audio_data(audio_data)
            
            # Transcribe using Whisper
            start_time = time.time()
            
            transcribe_options = {
                "language": language,
                "task": "transcribe",
                **options
            }
            
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: model.transcribe(audio_array, **transcribe_options)
            )
            
            processing_time = time.time() - start_time
            
            # Format response
            response = {
                "text": result.get("text", "").strip(),
                "language": result.get("language", "unknown"),
                "segments": result.get("segments", []),
                "processing_time": processing_time,
                "model_used": model_name,
                "confidence": self._calculate_average_confidence(result.get("segments", [])),
                "metadata": {
                    "audio_duration": len(audio_array) / self.sample_rate,
                    "detected_language": result.get("language"),
                    "num_segments": len(result.get("segments", []))
                }
            }
            
            logger.info(f"Transcription completed in {processing_time:.2f}s: '{response['text'][:100]}...'")
            return response
            
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            return {
                "text": "",
                "error": str(e),
                "processing_time": 0,
                "confidence": 0.0
            }
    
    async def transcribe_file(self, file_path: str, **kwargs) -> Dict[str, Any]:
        """Transcribe audio from file"""
        try:
            # Read audio file
            with open(file_path, 'rb') as f:
                audio_data = f.read()
            
            return await self.transcribe_audio(audio_data, **kwargs)
            
        except Exception as e:
            logger.error(f"Failed to transcribe file {file_path}: {e}")
            return {"text": "", "error": str(e)}
    
    async def stream_transcribe(self, audio_chunks: AsyncGenerator[bytes, None],
                               model_name: str = None) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Stream transcription for real-time audio processing
        
        Args:
            audio_chunks: Async generator yielding audio chunks
            model_name: Whisper model to use
        
        Yields:
            Partial transcription results
        """
        try:
            model_name = model_name or self.default_model
            
            if model_name not in self.models:
                await self.load_model(model_name)
            
            buffer = bytearray()
            chunk_duration = 3.0  # seconds per chunk
            chunk_size = int(self.sample_rate * chunk_duration * 2)  # 16-bit audio
            
            async for chunk in audio_chunks:
                buffer.extend(chunk)
                
                # Process when we have enough data
                if len(buffer) >= chunk_size:
                    # Extract chunk for processing
                    chunk_data = bytes(buffer[:chunk_size])
                    buffer = buffer[chunk_size//2:]  # 50% overlap
                    
                    # Transcribe chunk
                    result = await self.transcribe_audio(
                        chunk_data,
                        model_name=model_name
                    )
                    
                    if result.get("text"):
                        yield {
                            "partial_text": result["text"],
                            "is_final": False,
                            "confidence": result.get("confidence", 0.0),
                            "timestamp": time.time()
                        }
            
            # Process remaining buffer
            if len(buffer) > 0:
                result = await self.transcribe_audio(
                    bytes(buffer),
                    model_name=model_name
                )
                
                yield {
                    "final_text": result.get("text", ""),
                    "is_final": True,
                    "confidence": result.get("confidence", 0.0),
                    "timestamp": time.time(),
                    "segments": result.get("segments", [])
                }
                
        except Exception as e:
            logger.error(f"Stream transcription failed: {e}")
            yield {
                "error": str(e),
                "is_final": True,
                "timestamp": time.time()
            }
    
    async def _process_audio_data(self, audio_data: bytes) -> np.ndarray:
        """Process raw audio bytes into format suitable for Whisper"""
        try:
            # Save to temporary file for processing
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                tmp_file.write(audio_data)
                tmp_path = tmp_file.name
            
            try:
                # Load and resample audio
                audio_array, sr = librosa.load(tmp_path, sr=self.sample_rate)
                
                # Ensure audio is the right format (float32, mono)
                if len(audio_array.shape) > 1:
                    audio_array = librosa.to_mono(audio_array)
                
                # Normalize audio
                audio_array = librosa.util.normalize(audio_array)
                
                return audio_array
                
            finally:
                # Clean up temp file
                Path(tmp_path).unlink(missing_ok=True)
                
        except Exception as e:
            logger.error(f"Audio processing failed: {e}")
            # Fallback: try to create a simple array from raw bytes
            return np.frombuffer(audio_data, dtype=np.float32)
    
    def _calculate_average_confidence(self, segments: List[Dict]) -> float:
        """Calculate average confidence score from segments"""
        if not segments:
            return 0.0
        
        # Whisper doesn't provide direct confidence scores
        # We estimate based on segment characteristics
        confidences = []
        for segment in segments:
            # Use probability if available, otherwise estimate
            if 'avg_logprob' in segment:
                # Convert log probability to confidence estimate
                confidence = min(1.0, max(0.0, np.exp(segment['avg_logprob']) * 2))
            else:
                # Fallback confidence based on text length and timing
                text_length = len(segment.get('text', ''))
                duration = segment.get('end', 0) - segment.get('start', 0)
                confidence = min(1.0, text_length / max(duration * 10, 1))
            
            confidences.append(confidence)
        
        return sum(confidences) / len(confidences) if confidences else 0.0
    
    def get_available_models(self) -> List[str]:
        """Get list of available Whisper models"""
        return [
            "tiny",      # 39 MB, fastest
            "base",      # 74 MB, good balance  
            "small",     # 244 MB, better accuracy
            "medium",    # 769 MB, high accuracy
            "large-v2",  # 1550 MB, best accuracy
            "large-v3"   # 1550 MB, latest best
        ]
    
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """Get information about a specific model"""
        model_info = {
            "tiny": {"size": "39 MB", "speed": "~32x", "accuracy": "Basic"},
            "base": {"size": "74 MB", "speed": "~16x", "accuracy": "Good"},
            "small": {"size": "244 MB", "speed": "~6x", "accuracy": "Better"},
            "medium": {"size": "769 MB", "speed": "~2x", "accuracy": "High"},
            "large-v2": {"size": "1550 MB", "speed": "~1x", "accuracy": "Best"},
            "large-v3": {"size": "1550 MB", "speed": "~1x", "accuracy": "Best (Latest)"}
        }
        
        return model_info.get(model_name, {"size": "Unknown", "speed": "Unknown", "accuracy": "Unknown"})
    
    async def get_status(self) -> Dict[str, Any]:
        """Get service status information"""
        return {
            "initialized": self.is_initialized,
            "models_loaded": list(self.models.keys()),
            "default_model": self.default_model,
            "sample_rate": self.sample_rate,
            "supported_formats": self.supported_formats,
            "available_models": self.get_available_models()
        }
    
    async def shutdown(self):
        """Cleanup and shutdown"""
        logger.info("Shutting down Whisper Speech Recognition...")
        self.models.clear()
        self.is_initialized = False
        logger.info("Whisper Speech Recognition shutdown completed")
