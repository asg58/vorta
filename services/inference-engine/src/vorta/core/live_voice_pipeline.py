"""
🎤 VORTA Live Voice Pipeline
Complete end-to-end voice conversation system

This module provides the complete voice conversation pipeline:
- Real-time audio processing
- Speech-to-Text with Whisper
- AI conversation processing
- Text-to-Speech synthesis
- WebSocket streaming support

Author: Ultra High-Grade Development Team
Version: 1.0.0-live
Performance: <1.5s end-to-end latency
"""

import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

try:
    from fastapi import WebSocket
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False
    WebSocket = None

# Import VORTA components
from ..config.settings import Settings
from .inference_engine import VortaInferenceEngine
from .speech_recognition import WhisperSpeechRecognizer
from .tts_service import TTSProvider, VortaTTSService

logger = logging.getLogger(__name__)

@dataclass
class VoiceConversationState:
    """State management for live voice conversations"""
    conversation_id: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    start_time: datetime = field(default_factory=datetime.now)
    
    # Conversation history
    turns: List[Dict[str, Any]] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)
    
    # Voice settings
    selected_voice: str = "nova"
    tts_provider: TTSProvider = TTSProvider.OPENAI
    whisper_model: str = "base"
    
    # Performance tracking
    total_turns: int = 0
    average_response_time: float = 0.0
    total_processing_time: float = 0.0

class LiveVoicePipeline:
    """
    🎤 Live Voice Conversation Pipeline
    
    Complete end-to-end voice conversation system with:
    - Real-time speech recognition
    - AI conversation processing
    - Voice synthesis and streaming
    - WebSocket communication
    """
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.is_initialized = False
        
        # Core components
        self.speech_recognizer: Optional[WhisperSpeechRecognizer] = None
        self.tts_service: Optional[VortaTTSService] = None
        self.inference_engine: Optional[VortaInferenceEngine] = None
        
        # Active conversations
        self.active_conversations: Dict[str, VoiceConversationState] = {}
        
        # Performance metrics
        self.metrics = {
            'total_conversations': 0,
            'total_turns': 0,
            'average_response_time': 0.0,
            'speech_recognition_time': 0.0,
            'ai_processing_time': 0.0,
            'tts_synthesis_time': 0.0,
            'end_to_end_latency': 0.0
        }
        
        logger.info("🎤 Live Voice Pipeline initialized")
    
    async def initialize(self):
        """Initialize all pipeline components"""
        try:
            logger.info("🚀 Initializing Live Voice Pipeline components...")
            
            # Initialize speech recognition
            self.speech_recognizer = WhisperSpeechRecognizer(self.settings)
            await self.speech_recognizer.initialize()
            logger.info("✅ Speech recognition initialized")
            
            # Initialize TTS service
            self.tts_service = VortaTTSService(self.settings)
            await self.tts_service.initialize()
            logger.info("✅ TTS service initialized")
            
            # Initialize inference engine
            self.inference_engine = VortaInferenceEngine(self.settings)
            await self.inference_engine.initialize()
            logger.info("✅ Inference engine initialized")
            
            self.is_initialized = True
            logger.info("🎤 Live Voice Pipeline ready for conversations!")
            
        except Exception as e:
            logger.error(f"❌ Pipeline initialization failed: {e}")
            raise
    
    async def start_conversation(self, 
                               user_id: Optional[str] = None,
                               voice_settings: Optional[Dict] = None) -> str:
        """Start a new voice conversation"""
        try:
            conversation_id = str(uuid.uuid4())
            
            # Create conversation state
            state = VoiceConversationState(
                conversation_id=conversation_id,
                user_id=user_id,
                session_id=str(uuid.uuid4())
            )
            
            # Apply voice settings if provided
            if voice_settings:
                if 'voice' in voice_settings:
                    state.selected_voice = voice_settings['voice']
                if 'tts_provider' in voice_settings:
                    state.tts_provider = TTSProvider(voice_settings['tts_provider'])
                if 'whisper_model' in voice_settings:
                    state.whisper_model = voice_settings['whisper_model']
            
            # Store conversation
            self.active_conversations[conversation_id] = state
            self.metrics['total_conversations'] += 1
            
            logger.info(f"🎤 Started conversation {conversation_id[:8]} for user {user_id}")
            return conversation_id
            
        except Exception as e:
            logger.error(f"❌ Failed to start conversation: {e}")
            raise
    
    async def process_voice_turn(self, 
                               conversation_id: str,
                               audio_data: bytes,
                               audio_format: str = "wav") -> Dict[str, Any]:
        """
        Process a complete voice turn: audio → transcription → AI → speech
        
        Args:
            conversation_id: Active conversation identifier
            audio_data: Raw audio data
            audio_format: Audio format (wav, mp3, etc.)
            
        Returns:
            Complete turn result with transcription, AI response, and audio
        """
        start_time = time.time()
        
        try:
            if not self.is_initialized:
                raise Exception("Pipeline not initialized")
            
            if conversation_id not in self.active_conversations:
                raise Exception(f"Conversation {conversation_id} not found")
            
            state = self.active_conversations[conversation_id]
            turn_id = str(uuid.uuid4())
            
            logger.info(f"🎤 Processing voice turn {turn_id[:8]} for conversation {conversation_id[:8]}")
            
            # Step 1: Speech Recognition
            stt_start = time.time()
            transcription_result = await self.speech_recognizer.transcribe_audio(
                audio_data,
                model_name=state.whisper_model
            )
            stt_time = time.time() - stt_start
            
            user_text = transcription_result.get("text", "").strip()
            if not user_text:
                return {
                    "success": False,
                    "error": "No speech detected in audio",
                    "turn_id": turn_id
                }
            
            logger.info(f"🗣️ Transcribed: '{user_text[:100]}...' in {stt_time:.2f}s")
            
            # Step 2: AI Processing
            ai_start = time.time()
            
            # Build conversation context
            conversation_history = []
            for turn in state.turns[-10:]:  # Last 10 turns for context
                if turn.get('user_text'):
                    conversation_history.append({
                        "role": "user",
                        "content": turn['user_text']
                    })
                if turn.get('ai_response'):
                    conversation_history.append({
                        "role": "assistant", 
                        "content": turn['ai_response']
                    })
            
            # Add current user input
            conversation_history.append({
                "role": "user",
                "content": user_text
            })
            
            # Generate AI response
            ai_result = await self.inference_engine.process_conversation(
                messages=conversation_history,
                user_id=state.user_id,
                conversation_id=conversation_id
            )
            
            ai_time = time.time() - ai_start
            ai_response = ai_result.get("response", "I'm sorry, I couldn't process that.")
            
            logger.info(f"🤖 AI response: '{ai_response[:100]}...' in {ai_time:.2f}s")
            
            # Step 3: Text-to-Speech Synthesis
            tts_start = time.time()
            tts_result = await self.tts_service.synthesize_speech(
                text=ai_response,
                voice=state.selected_voice,
                provider=state.tts_provider,
                options={
                    "model": "tts-1",
                    "speed": 1.0,
                    "format": "mp3"
                }
            )
            tts_time = time.time() - tts_start
            
            if not tts_result.get("success"):
                logger.error(f"❌ TTS synthesis failed: {tts_result.get('error')}")
                tts_result = {
                    "audio_data": None,
                    "base64_audio": None,
                    "error": "TTS synthesis failed"
                }
            
            logger.info(f"🔊 Speech synthesized in {tts_time:.2f}s")
            
            # Step 4: Store conversation turn
            total_time = time.time() - start_time
            
            turn_data = {
                "turn_id": turn_id,
                "timestamp": datetime.now().isoformat(),
                "user_text": user_text,
                "ai_response": ai_response,
                "audio_format": audio_format,
                "processing_times": {
                    "speech_recognition": stt_time,
                    "ai_processing": ai_time,
                    "tts_synthesis": tts_time,
                    "total": total_time
                },
                "transcription_confidence": transcription_result.get("confidence", 0.0),
                "ai_metadata": ai_result.get("metadata", {}),
                "tts_metadata": {
                    "voice_used": state.selected_voice,
                    "provider": state.tts_provider.value,
                    "audio_size": tts_result.get("audio_size", 0)
                }
            }
            
            state.turns.append(turn_data)
            state.total_turns += 1
            state.total_processing_time += total_time
            state.average_response_time = state.total_processing_time / state.total_turns
            
            # Update global metrics
            self.metrics['total_turns'] += 1
            self.metrics['speech_recognition_time'] = (
                (self.metrics['speech_recognition_time'] * (self.metrics['total_turns'] - 1) + stt_time) /
                self.metrics['total_turns']
            )
            self.metrics['ai_processing_time'] = (
                (self.metrics['ai_processing_time'] * (self.metrics['total_turns'] - 1) + ai_time) /
                self.metrics['total_turns']
            )
            self.metrics['tts_synthesis_time'] = (
                (self.metrics['tts_synthesis_time'] * (self.metrics['total_turns'] - 1) + tts_time) /
                self.metrics['total_turns']
            )
            self.metrics['end_to_end_latency'] = (
                (self.metrics['end_to_end_latency'] * (self.metrics['total_turns'] - 1) + total_time) /
                self.metrics['total_turns']
            )
            
            # Prepare response
            response = {
                "success": True,
                "turn_id": turn_id,
                "conversation_id": conversation_id,
                "user_text": user_text,
                "ai_response": ai_response,
                "audio_data": tts_result.get("audio_data"),
                "base64_audio": tts_result.get("base64_audio"),
                "processing_times": turn_data["processing_times"],
                "performance": {
                    "total_latency_ms": int(total_time * 1000),
                    "stt_latency_ms": int(stt_time * 1000),
                    "ai_latency_ms": int(ai_time * 1000),
                    "tts_latency_ms": int(tts_time * 1000)
                },
                "metadata": {
                    "conversation_turns": state.total_turns,
                    "average_response_time": state.average_response_time,
                    "transcription_confidence": transcription_result.get("confidence", 0.0),
                    "voice_used": state.selected_voice,
                    "model_used": state.whisper_model
                }
            }
            
            logger.info(f"✅ Voice turn completed in {total_time:.2f}s (STT: {stt_time:.2f}s, AI: {ai_time:.2f}s, TTS: {tts_time:.2f}s)")
            
            return response
            
        except Exception as e:
            logger.error(f"❌ Voice turn processing failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "turn_id": turn_id if 'turn_id' in locals() else None,
                "conversation_id": conversation_id,
                "processing_time": time.time() - start_time
            }
    
    async def handle_websocket_conversation(self, websocket: WebSocket):
        """Handle WebSocket-based voice conversation"""
        await websocket.accept()
        conversation_id = None
        
        try:
            logger.info("🔌 WebSocket voice conversation connected")
            
            # Send welcome message
            await websocket.send_json({
                "type": "connection_established",
                "message": "VORTA Live Voice Pipeline ready",
                "timestamp": time.time()
            })
            
            while True:
                try:
                    # Receive data from client
                    data = await websocket.receive()
                    
                    if data["type"] == "websocket.receive":
                        if "text" in data:
                            # Control message
                            message = json.loads(data["text"])
                            
                            if message.get("action") == "start_conversation":
                                # Start new conversation
                                user_id = message.get("user_id")
                                voice_settings = message.get("voice_settings", {})
                                
                                conversation_id = await self.start_conversation(
                                    user_id=user_id,
                                    voice_settings=voice_settings
                                )
                                
                                await websocket.send_json({
                                    "type": "conversation_started",
                                    "conversation_id": conversation_id,
                                    "message": "Voice conversation started",
                                    "timestamp": time.time()
                                })
                            
                            elif message.get("action") == "ping":
                                # Heartbeat
                                await websocket.send_json({
                                    "type": "pong",
                                    "timestamp": time.time()
                                })
                        
                        elif "bytes" in data and conversation_id:
                            # Audio data received
                            audio_data = data["bytes"]
                            
                            # Process voice turn
                            result = await self.process_voice_turn(
                                conversation_id=conversation_id,
                                audio_data=audio_data
                            )
                            
                            # Send result back to client
                            response = {
                                "type": "voice_turn_complete",
                                "result": result,
                                "timestamp": time.time()
                            }
                            
                            await websocket.send_json(response)
                
                except Exception as e:
                    logger.error(f"❌ WebSocket message processing error: {e}")
                    await websocket.send_json({
                        "type": "error",
                        "message": str(e),
                        "timestamp": time.time()
                    })
        
        except Exception as e:
            logger.error(f"❌ WebSocket conversation error: {e}")
        
        finally:
            if conversation_id and conversation_id in self.active_conversations:
                del self.active_conversations[conversation_id]
                logger.info(f"🔌 WebSocket conversation {conversation_id[:8] if conversation_id else 'unknown'} closed")
    
    async def get_conversation_history(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        """Get conversation history and statistics"""
        if conversation_id not in self.active_conversations:
            return None
        
        state = self.active_conversations[conversation_id]
        
        return {
            "conversation_id": conversation_id,
            "user_id": state.user_id,
            "start_time": state.start_time.isoformat(),
            "total_turns": state.total_turns,
            "average_response_time": state.average_response_time,
            "voice_settings": {
                "selected_voice": state.selected_voice,
                "tts_provider": state.tts_provider.value,
                "whisper_model": state.whisper_model
            },
            "turns": state.turns[-10:],  # Last 10 turns
            "context": state.context
        }
    
    def get_pipeline_metrics(self) -> Dict[str, Any]:
        """Get comprehensive pipeline performance metrics"""
        return {
            "pipeline_status": {
                "initialized": self.is_initialized,
                "active_conversations": len(self.active_conversations),
                "components_ready": {
                    "speech_recognition": self.speech_recognizer is not None and self.speech_recognizer.is_initialized,
                    "tts_service": self.tts_service is not None and self.tts_service.is_initialized,
                    "inference_engine": self.inference_engine is not None and self.inference_engine.is_initialized
                }
            },
            "performance_metrics": self.metrics.copy(),
            "conversation_summary": {
                "total_active": len(self.active_conversations),
                "average_turns_per_conversation": (
                    sum(state.total_turns for state in self.active_conversations.values()) / 
                    max(len(self.active_conversations), 1)
                ),
                "longest_conversation": max(
                    [state.total_turns for state in self.active_conversations.values()],
                    default=0
                )
            }
        }
    
    async def shutdown(self):
        """Gracefully shutdown the pipeline"""
        try:
            logger.info("🔄 Shutting down Live Voice Pipeline...")
            
            # Close all active conversations
            for conversation_id in list(self.active_conversations.keys()):
                del self.active_conversations[conversation_id]
            
            # Shutdown components
            if self.tts_service:
                await self.tts_service.shutdown()
            
            self.is_initialized = False
            logger.info("✅ Live Voice Pipeline shutdown complete")
            
        except Exception as e:
            logger.error(f"❌ Pipeline shutdown error: {e}")

# Global pipeline instance
_live_voice_pipeline: Optional[LiveVoicePipeline] = None

def get_live_voice_pipeline() -> LiveVoicePipeline:
    """Get the global live voice pipeline instance"""
    global _live_voice_pipeline
    if _live_voice_pipeline is None:
        from ..config.settings import get_settings
        settings = get_settings()
        _live_voice_pipeline = LiveVoicePipeline(settings)
    return _live_voice_pipeline

async def initialize_live_voice_pipeline():
    """Initialize the global live voice pipeline"""
    pipeline = get_live_voice_pipeline()
    if not pipeline.is_initialized:
        await pipeline.initialize()
    return pipeline
