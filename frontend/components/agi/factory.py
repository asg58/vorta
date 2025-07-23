# frontend/components/agi/factory.py

"""
Factory for creating advanced AGI components.

This factory is responsible for instantiating the high-level AGI components
that provide the core enterprise-grade features of the VORTA agent.
It ensures that these components are created with their correct, production-ready
dependencies, following the factory pattern to abstract away the instantiation logic.
"""

import os

from .adaptive_learning_engine import AdaptiveLearningEngine
from .enterprise_security_layer import EnterpriseSecurityLayer
from .multi_modal_processor import MultiModalProcessor as AGIMultiModalProcessor
from .performance_analytics import PerformanceAnalytics
from .predictive_conversation import PredictiveConversation
from .proactive_assistant import ProactiveAssistant
from .voice_biometrics import VoiceBiometrics

# Environment variable to control implementation type
ENVIRONMENT = os.getenv("VORTA_ENVIRONMENT", "production")

class AGIComponentFactory:
    """
    A factory for creating instances of AGI components.
    
    This class centralizes the creation of all AGI-related components,
    ensuring a consistent and decoupled architecture.
    """

    @staticmethod
    def create_multi_modal_processor(**kwargs) -> AGIMultiModalProcessor:
        """Creates an AGI Multi-Modal Processor - PRODUCTION ONLY."""
        return AGIMultiModalProcessor(**kwargs)

    @staticmethod
    def create_predictive_conversation(**kwargs) -> PredictiveConversation:
        """Creates a Predictive Conversation engine - PRODUCTION ONLY."""
        return PredictiveConversation(**kwargs)

    @staticmethod
    def create_adaptive_learning_engine(**kwargs) -> AdaptiveLearningEngine:
        """Creates an Adaptive Learning Engine - PRODUCTION ONLY."""
        return AdaptiveLearningEngine(**kwargs)

    @staticmethod
    def create_enterprise_security_layer(**kwargs) -> EnterpriseSecurityLayer:
        """Creates an Enterprise Security Layer - PRODUCTION ONLY."""
        return EnterpriseSecurityLayer(**kwargs)

    @staticmethod
    def create_performance_analytics(**kwargs) -> PerformanceAnalytics:
        """Creates a Performance Analytics engine - PRODUCTION ONLY."""
        return PerformanceAnalytics(**kwargs)

    @staticmethod
    def create_proactive_assistant(**kwargs) -> ProactiveAssistant:
        """Creates a Proactive Assistant - PRODUCTION ONLY."""
        return ProactiveAssistant(**kwargs)

    @staticmethod
    def create_voice_biometrics(**kwargs) -> VoiceBiometrics:
        """Creates a Voice Biometrics engine - PRODUCTION ONLY."""
        return VoiceBiometrics(**kwargs)
