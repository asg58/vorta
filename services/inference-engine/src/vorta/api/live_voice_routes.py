"""
🎤 VORTA Live Voice API Routes
WebSocket and HTTP endpoints for live voice conversations

This module provides:
- Live voice conversation endpoints
- WebSocket streaming for real-time audio
- Voice turn processing
- Conversation management
"""

import logging
import time
from typing import Optional

try:
    from fastapi import APIRouter, File, Form, HTTPException, UploadFile, WebSocket
    from fastapi.responses import JSONResponse, StreamingResponse
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False
    APIRouter = None

from ..core.live_voice_pipeline import (
    get_live_voice_pipeline,
    initialize_live_voice_pipeline,
)

logger = logging.getLogger(__name__)

if HAS_FASTAPI:
    router = APIRouter(prefix="/voice", tags=["Live Voice"])

    @router.post("/conversation/start")
    async def start_voice_conversation(
        user_id: Optional[str] = Form(default=None),
        voice: str = Form(default="nova"),
        tts_provider: str = Form(default="openai"),
        whisper_model: str = Form(default="base")
    ):
        """Start a new live voice conversation"""
        try:
            # Initialize pipeline if needed
            pipeline = await initialize_live_voice_pipeline()
            
            # Start conversation
            voice_settings = {
                "voice": voice,
                "tts_provider": tts_provider,
                "whisper_model": whisper_model
            }
            
            conversation_id = await pipeline.start_conversation(
                user_id=user_id,
                voice_settings=voice_settings
            )
            
            return JSONResponse({
                "success": True,
                "conversation_id": conversation_id,
                "message": "Voice conversation started",
                "settings": voice_settings,
                "timestamp": time.time()
            })
            
        except Exception as e:
            logger.error(f"❌ Failed to start voice conversation: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @router.post("/conversation/{conversation_id}/turn")
    async def process_voice_turn(
        conversation_id: str,
        audio_file: UploadFile = File(..., description="Audio file for voice turn"),
        audio_format: str = Form(default="wav")
    ):
        """Process a complete voice turn with uploaded audio"""
        try:
            # Get pipeline
            pipeline = get_live_voice_pipeline()
            if not pipeline.is_initialized:
                raise HTTPException(status_code=503, detail="Voice pipeline not initialized")
            
            # Read audio data
            audio_data = await audio_file.read()
            
            # Process voice turn
            result = await pipeline.process_voice_turn(
                conversation_id=conversation_id,
                audio_data=audio_data,
                audio_format=audio_format
            )
            
            if not result.get("success"):
                raise HTTPException(status_code=400, detail=result.get("error", "Voice processing failed"))
            
            return JSONResponse(result)
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"❌ Voice turn processing failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @router.get("/conversation/{conversation_id}/history")
    async def get_conversation_history(conversation_id: str):
        """Get conversation history and statistics"""
        try:
            pipeline = get_live_voice_pipeline()
            history = await pipeline.get_conversation_history(conversation_id)
            
            if history is None:
                raise HTTPException(status_code=404, detail="Conversation not found")
            
            return JSONResponse(history)
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"❌ Failed to get conversation history: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @router.get("/pipeline/metrics")
    async def get_pipeline_metrics():
        """Get comprehensive pipeline performance metrics"""
        try:
            pipeline = get_live_voice_pipeline()
            metrics = pipeline.get_pipeline_metrics()
            
            return JSONResponse(metrics)
            
        except Exception as e:
            logger.error(f"❌ Failed to get pipeline metrics: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @router.websocket("/conversation/live")
    async def websocket_live_conversation(websocket: WebSocket):
        """WebSocket endpoint for live voice conversations"""
        try:
            # Initialize pipeline if needed
            pipeline = await initialize_live_voice_pipeline()
            
            # Handle WebSocket conversation
            await pipeline.handle_websocket_conversation(websocket)
            
        except Exception as e:
            logger.error(f"❌ WebSocket live conversation failed: {e}")
            try:
                await websocket.close()
            except:
                pass

    @router.get("/status")
    async def get_voice_pipeline_status():
        """Get voice pipeline status and health"""
        try:
            pipeline = get_live_voice_pipeline()
            metrics = pipeline.get_pipeline_metrics()
            
            status = {
                "service": "VORTA Live Voice Pipeline",
                "status": "online" if metrics["pipeline_status"]["initialized"] else "offline",
                "initialized": metrics["pipeline_status"]["initialized"],
                "active_conversations": metrics["pipeline_status"]["active_conversations"],
                "components": metrics["pipeline_status"]["components_ready"],
                "performance": {
                    "total_conversations": metrics["performance_metrics"]["total_conversations"],
                    "total_turns": metrics["performance_metrics"]["total_turns"],
                    "average_end_to_end_latency_ms": int(metrics["performance_metrics"]["end_to_end_latency"] * 1000),
                },
                "timestamp": time.time()
            }
            
            return JSONResponse(status)
            
        except Exception as e:
            logger.error(f"❌ Failed to get pipeline status: {e}")
            return JSONResponse({
                "service": "VORTA Live Voice Pipeline",
                "status": "error",
                "error": str(e),
                "timestamp": time.time()
            }, status_code=500)

else:
    # Fallback when FastAPI is not available
    logger.warning("⚠️ FastAPI not available - Live Voice API routes disabled")
    router = None
