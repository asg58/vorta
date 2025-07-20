"""
VORTA Speech API Routes
Real-time speech recognition and text-to-speech endpoints
"""

import io
import json
import logging
import time
from typing import Any, Dict, List, Optional

from fastapi import (
    APIRouter,
    File,
    Form,
    HTTPException,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from ..core.dependencies import get_speech_recognizer, get_tts_service
from ..core.tts_service import TTSProvider

logger = logging.getLogger(__name__)

# Request/Response Models
class TranscriptionRequest(BaseModel):
    """Speech transcription request"""
    model: str = Field(default="base", description="Whisper model to use")
    language: Optional[str] = Field(default=None, description="Source language (auto-detect if None)")
    task: str = Field(default="transcribe", description="Task: transcribe or translate")
    temperature: float = Field(default=0.0, description="Sampling temperature")
    response_format: str = Field(default="json", description="Response format")

class TranscriptionResponse(BaseModel):
    """Speech transcription response"""
    text: str = Field(description="Transcribed text")
    language: str = Field(description="Detected language")
    confidence: float = Field(description="Confidence score")
    processing_time: float = Field(description="Processing time in seconds")
    segments: List[Dict[str, Any]] = Field(default=[], description="Detailed segments")
    metadata: Dict[str, Any] = Field(default={}, description="Additional metadata")

class TTSRequest(BaseModel):
    """Text-to-speech request"""
    text: str = Field(description="Text to synthesize", max_length=4000)
    voice: str = Field(default="alloy", description="Voice to use")
    provider: TTSProvider = Field(default=TTSProvider.OPENAI, description="TTS provider")
    model: str = Field(default="tts-1", description="TTS model")
    speed: float = Field(default=1.0, ge=0.25, le=4.0, description="Speech speed")
    response_format: str = Field(default="mp3", description="Audio format")

class TTSResponse(BaseModel):
    """Text-to-speech response"""
    success: bool = Field(description="Success status")
    audio_size: int = Field(description="Audio data size in bytes")
    format: str = Field(description="Audio format")
    processing_time: float = Field(description="Processing time in seconds")
    voice_used: str = Field(description="Voice used for synthesis")
    provider: str = Field(description="TTS provider used")

# API Router
router = APIRouter(prefix="/speech", tags=["Speech"])

@router.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe_audio(
    file: UploadFile = File(..., description="Audio file to transcribe"),
    model: str = Form(default="base"),
    language: Optional[str] = Form(default=None),
    temperature: float = Form(default=0.0)
):
    """
    Transcribe uploaded audio file to text using Whisper
    
    Supported formats: WAV, MP3, M4A, FLAC, OGG
    """
    try:
        # Validate file
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file provided")
        
        # Check file extension
        supported_extensions = ['.wav', '.mp3', '.m4a', '.flac', '.ogg']
        file_ext = '.' + file.filename.split('.')[-1].lower()
        
        if file_ext not in supported_extensions:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported file format. Supported: {supported_extensions}"
            )
        
        # Read audio data
        audio_data = await file.read()
        
        if len(audio_data) == 0:
            raise HTTPException(status_code=400, detail="Empty audio file")
        
        # Get speech recognizer
        recognizer = get_speech_recognizer()
        
        if not recognizer.is_initialized:
            raise HTTPException(status_code=503, detail="Speech recognition service not available")
        
        # Transcribe audio
        options = {
            "temperature": temperature,
        }
        
        result = await recognizer.transcribe_audio(
            audio_data,
            model_name=model,
            language=language,
            options=options
        )
        
        if "error" in result:
            raise HTTPException(status_code=500, detail=f"Transcription failed: {result['error']}")
        
        # Return response
        return TranscriptionResponse(
            text=result["text"],
            language=result.get("language", "unknown"),
            confidence=result.get("confidence", 0.0),
            processing_time=result.get("processing_time", 0.0),
            segments=result.get("segments", []),
            metadata=result.get("metadata", {})
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Audio transcription failed: {e}")
        raise HTTPException(status_code=500, detail=f"Transcription error: {str(e)}")

@router.post("/tts", response_model=TTSResponse)
async def synthesize_speech(request: TTSRequest):
    """
    Convert text to speech using high-quality TTS
    
    Returns audio data as base64 encoded string in response
    """
    try:
        # Get TTS service
        tts_service = get_tts_service()
        
        if not tts_service.is_initialized:
            raise HTTPException(status_code=503, detail="TTS service not available")
        
        # Validate voice
        if not await tts_service.validate_voice(request.voice, request.provider):
            raise HTTPException(status_code=400, detail=f"Invalid voice: {request.voice}")
        
        # Synthesize speech
        options = {
            "model": request.model,
            "speed": request.speed,
            "format": request.response_format
        }
        
        result = await tts_service.synthesize_speech(
            request.text,
            voice=request.voice,
            provider=request.provider,
            options=options
        )
        
        if "error" in result:
            raise HTTPException(status_code=500, detail=f"TTS synthesis failed: {result['error']}")
        
        if not result.get("success"):
            raise HTTPException(status_code=500, detail="TTS synthesis failed")
        
        return TTSResponse(
            success=True,
            audio_size=result.get("audio_size", 0),
            format=result.get("format", request.response_format),
            processing_time=result.get("processing_time", 0.0),
            voice_used=result.get("voice_used", request.voice),
            provider=result.get("provider", request.provider.value)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"TTS synthesis failed: {e}")
        raise HTTPException(status_code=500, detail=f"TTS error: {str(e)}")

@router.get("/tts/audio/{request_id}")
async def download_audio(request_id: str):
    """
    Download generated audio file
    (Implementation depends on audio storage strategy)
    """
    # TODO: Implement audio file retrieval
    raise HTTPException(status_code=501, detail="Audio download not implemented yet")

@router.post("/tts/stream")
async def synthesize_speech_stream(request: TTSRequest):
    """
    Convert text to speech and stream audio directly
    """
    try:
        # Get TTS service
        tts_service = get_tts_service()
        
        if not tts_service.is_initialized:
            raise HTTPException(status_code=503, detail="TTS service not available")
        
        # Synthesize speech
        options = {
            "model": request.model,
            "speed": request.speed,
            "format": request.response_format
        }
        
        result = await tts_service.synthesize_speech(
            request.text,
            voice=request.voice,
            provider=request.provider,
            options=options
        )
        
        if "error" in result or not result.get("success"):
            raise HTTPException(status_code=500, detail="TTS synthesis failed")
        
        # Stream audio data
        audio_data = result["audio_data"]
        
        return StreamingResponse(
            io.BytesIO(audio_data),
            media_type=f"audio/{request.response_format}",
            headers={
                "Content-Disposition": f"attachment; filename=speech.{request.response_format}"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"TTS streaming failed: {e}")
        raise HTTPException(status_code=500, detail=f"TTS streaming error: {str(e)}")

@router.websocket("/transcribe/stream")
async def websocket_transcribe_stream(websocket: WebSocket):
    """
    WebSocket endpoint for real-time speech transcription
    
    Client sends audio chunks, receives real-time transcriptions
    """
    await websocket.accept()
    logger.info("WebSocket connection established for speech transcription")
    
    try:
        # Get speech recognizer
        recognizer = get_speech_recognizer()
        
        if not recognizer.is_initialized:
            await websocket.send_json({"error": "Speech recognition service not available"})
            await websocket.close()
            return
        
        # Audio buffer for streaming
        audio_buffer = bytearray()
        
        while True:
            try:
                # Receive audio data from client
                data = await websocket.receive()
                
                if data["type"] == "websocket.receive":
                    if "bytes" in data:
                        # Audio data received
                        audio_chunk = data["bytes"]
                        audio_buffer.extend(audio_chunk)
                        
                        # Process when we have enough data (e.g., 1 second of audio)
                        chunk_size = 16000 * 2  # 1 second at 16kHz, 16-bit
                        
                        if len(audio_buffer) >= chunk_size:
                            # Extract chunk for processing
                            chunk_data = bytes(audio_buffer[:chunk_size])
                            audio_buffer = audio_buffer[chunk_size//2:]  # 50% overlap
                            
                            # Transcribe chunk
                            result = await recognizer.transcribe_audio(chunk_data)
                            
                            if result.get("text"):
                                # Send partial result
                                await websocket.send_json({
                                    "type": "partial_transcription",
                                    "text": result["text"],
                                    "confidence": result.get("confidence", 0.0),
                                    "timestamp": time.time()
                                })
                    
                    elif "text" in data:
                        # Control message from client
                        message = json.loads(data["text"])
                        
                        if message.get("command") == "finalize":
                            # Process remaining buffer
                            if len(audio_buffer) > 0:
                                result = await recognizer.transcribe_audio(bytes(audio_buffer))
                                
                                await websocket.send_json({
                                    "type": "final_transcription",
                                    "text": result.get("text", ""),
                                    "confidence": result.get("confidence", 0.0),
                                    "segments": result.get("segments", []),
                                    "timestamp": time.time()
                                })
                                
                                audio_buffer.clear()
                        
                        elif message.get("command") == "ping":
                            # Heartbeat response
                            await websocket.send_json({
                                "type": "pong",
                                "timestamp": time.time()
                            })
                
            except WebSocketDisconnect:
                break
            except Exception as e:
                logger.error(f"WebSocket transcription error: {e}")
                await websocket.send_json({
                    "type": "error", 
                    "message": str(e)
                })
    
    except Exception as e:
        logger.error(f"WebSocket connection failed: {e}")
    
    finally:
        logger.info("WebSocket connection closed")

@router.get("/voices")
async def get_available_voices(provider: Optional[TTSProvider] = None):
    """Get list of available TTS voices"""
    try:
        tts_service = get_tts_service()
        
        if not tts_service.is_initialized:
            raise HTTPException(status_code=503, detail="TTS service not available")
        
        voices = await tts_service.get_available_voices(provider)
        
        return {
            "voices": voices,
            "total": len(voices),
            "provider_filter": provider.value if provider else None
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get voices: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting voices: {str(e)}")

@router.get("/models")
async def get_available_models():
    """Get list of available Whisper models and TTS models"""
    try:
        recognizer = get_speech_recognizer()
        tts_service = get_tts_service()
        
        result = {
            "whisper_models": [],
            "tts_providers": {}
        }
        
        if recognizer.is_initialized:
            result["whisper_models"] = [
                {
                    "name": model,
                    "info": recognizer.get_model_info(model)
                }
                for model in recognizer.get_available_models()
            ]
        
        if tts_service.is_initialized:
            status = await tts_service.get_service_status()
            result["tts_providers"] = status.get("providers", {})
        
        return result
        
    except Exception as e:
        logger.error(f"Failed to get models: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting models: {str(e)}")

@router.get("/status")
async def get_speech_service_status():
    """Get speech services status"""
    try:
        recognizer = get_speech_recognizer()
        tts_service = get_tts_service()
        
        stt_status = await recognizer.get_status() if recognizer else {"initialized": False}
        tts_status = await tts_service.get_service_status() if tts_service else {"initialized": False}
        
        return {
            "speech_recognition": stt_status,
            "text_to_speech": tts_status,
            "overall_status": "operational" if stt_status.get("initialized") and tts_status.get("initialized") else "partial"
        }
        
    except Exception as e:
        logger.error(f"Failed to get speech service status: {e}")
        raise HTTPException(status_code=500, detail=f"Status error: {str(e)}")

# Health check for speech services
@router.get("/health")
async def health_check():
    """Quick health check for speech services"""
    try:
        recognizer = get_speech_recognizer()
        tts_service = get_tts_service()
        
        return {
            "status": "healthy",
            "services": {
                "speech_recognition": recognizer.is_initialized if recognizer else False,
                "text_to_speech": tts_service.is_initialized if tts_service else False
            },
            "timestamp": time.time()
        }
        
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": time.time()
        }
