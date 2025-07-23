# frontend/components/ai/factory.py

"""
Factory for creating AI and conversation components.

This module provides a centralized factory for instantiating all components
related to the AI conversation intelligence stack. It abstracts the creation
process, allowing for different implementations (e.g., real vs. mock) to be
used based on the environment configuration. This is crucial for decoupling
components and enabling robust testing.
"""

import os

from .context_memory_manager import ContextMemoryManager
from .conversation_orchestrator import ConversationOrchestrator
from .emotion_analysis_processor import EmotionAnalysisProcessor
from .intent_recognition_engine import IntentRecognitionEngine
from .multi_modal_processor import MultiModalProcessor
from .response_generation_engine import ResponseGenerationEngine
from .voice_personality_engine import VoicePersonalityEngine

# Environment variable to control implementation type
ENVIRONMENT = os.getenv("VORTA_ENVIRONMENT", "production")

class AIComponentFactory:
    """
    A factory for creating instances of AI components.
    
    This class centralizes the creation of all AI-related components,
    facilitating dependency injection and making it easy to switch
    between different component implementations.
    """

    @staticmethod
    def create_conversation_orchestrator(**kwargs) -> ConversationOrchestrator:
        """Creates a Conversation Orchestrator - PRODUCTION ONLY."""
        return ConversationOrchestrator(**kwargs)

    @staticmethod
    def create_intent_recognition_engine(**kwargs) -> IntentRecognitionEngine:
        """Creates an Intent Recognition Engine - PRODUCTION ONLY."""
        return IntentRecognitionEngine(**kwargs)

    @staticmethod
    def create_emotion_analysis_processor(**kwargs) -> EmotionAnalysisProcessor:
        """Creates an Emotion Analysis Processor - PRODUCTION ONLY."""
        return EmotionAnalysisProcessor(**kwargs)

    @staticmethod
    def create_context_memory_manager(**kwargs) -> ContextMemoryManager:
        """Creates a Context Memory Manager - PRODUCTION ONLY."""
        return ContextMemoryManager(**kwargs)

    @staticmethod
    def create_response_generation_engine(**kwargs) -> ResponseGenerationEngine:
        """Creates a Response Generation Engine - PRODUCTION ONLY."""
        return ResponseGenerationEngine(**kwargs)

    @staticmethod
    def create_voice_personality_engine(**kwargs) -> VoicePersonalityEngine:
        """Creates a Voice Personality Engine - PRODUCTION ONLY."""
        return VoicePersonalityEngine(**kwargs)

    @staticmethod
    def create_multi_modal_processor(**kwargs) -> MultiModalProcessor:
        """Creates a Multi-Modal Processor - PRODUCTION ONLY."""
        return MultiModalProcessor(**kwargs)

