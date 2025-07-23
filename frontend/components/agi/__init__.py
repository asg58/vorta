# frontend/components/agi/__init__.py
"""
VORTA: AGI Components Package

This package contains the advanced AGI features for the VORTA Enterprise system.
Each module represents a distinct capability, designed to be modular and robust.

Modules:
- adaptive_learning_engine: Personalizes user experience by learning from interactions.
- enterprise_security_layer: Provides security features like encryption and anonymization.
- multi_modal_processor: Fuses data from various sources (voice, text) into a unified model.
- performance_analytics: Collects and analyzes system performance metrics.
- predictive_conversation: Predicts future conversational turns and user intents.
- proactive_assistant: Offers context-aware suggestions and assistance.
- voice_biometrics: Handles speaker identification and verification.
"""

from .adaptive_learning_engine import AdaptiveLearningEngine
from .enterprise_security_layer import EnterpriseSecurityLayer
from .multi_modal_processor import MultiModalProcessor
from .performance_analytics import PerformanceAnalytics
from .predictive_conversation import PredictiveConversationEngine
from .proactive_assistant import ProactiveAssistant
from .voice_biometrics import VoiceBiometricsEngine

__all__ = [
    "AdaptiveLearningEngine",
    "EnterpriseSecurityLayer",
    "MultiModalProcessor",
    "PerformanceAnalytics",
    "PredictiveConversationEngine",
    "ProactiveAssistant",
    "VoiceBiometricsEngine",
]

import logging

logging.getLogger(__name__).addHandler(logging.NullHandler())
