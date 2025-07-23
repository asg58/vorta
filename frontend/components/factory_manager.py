# frontend/components/factory_manager.py

"""
VORTA Factory Manager - Centralized Component Factory Management

This module provides centralized management for all component factories across
the VORTA AGI Voice Agent. It implements the Factory Pattern as specified
in Phase 5.4 of the roadmap for transitioning from development mocks to 
production-ready implementations.

Author: Ultra High-Grade Development Team
Version: 3.0.0-agi
License: Enterprise
"""

import os
import logging
from typing import Dict, Any, Optional

# Import all component factories
from .audio.factory import AudioComponentFactory
from .ai.factory import AIComponentFactory
from .agi.factory import AGIComponentFactory
from .voice.factory import VoiceComponentFactory

logger = logging.getLogger(__name__)

class VORTAFactoryManager:
    """
    Centralized Factory Manager for all VORTA components.
    
    This class provides a single entry point for creating all components
    across the VORTA system, implementing proper dependency injection
    and environment-based component selection.
    """
    
    def __init__(self, environment: Optional[str] = None):
        """
        Initialize the Factory Manager.
        
        Args:
            environment: Override the environment setting (production, testing, development)
        """
        self.environment = environment or os.getenv("VORTA_ENVIRONMENT", "production")
        
        # Initialize all component factories
        self.audio_factory = AudioComponentFactory()
        self.ai_factory = AIComponentFactory()
        self.agi_factory = AGIComponentFactory()
        self.voice_factory = VoiceComponentFactory()
        
        logger.info(f"🏭 VORTA Factory Manager initialized (environment: {self.environment})")
    
    # Audio Component Creation Methods
    def create_neural_vad_processor(self, **kwargs):
        """Create Neural VAD Processor"""
        return self.audio_factory.create_vad_processor(**kwargs)
    
    def create_wake_word_detector(self, config: Optional[dict] = None, **kwargs):
        """Create Wake Word Detector"""
        return self.audio_factory.create_wake_word_detector(config=config, **kwargs)
    
    def create_noise_cancellation_engine(self, **kwargs):
        """Create Noise Cancellation Engine"""
        return self.audio_factory.create_noise_cancellation_engine(**kwargs)
    
    def create_audio_stream_manager(self, **kwargs):
        """Create Audio Stream Manager"""
        return self.audio_factory.create_audio_stream_manager(**kwargs)
    
    def create_dsp_enhancement_suite(self, **kwargs):
        """Create DSP Enhancement Suite"""
        return self.audio_factory.create_dsp_enhancement_suite(**kwargs)
    
    def create_audio_quality_analyzer(self, **kwargs):
        """Create Audio Quality Analyzer"""
        return self.audio_factory.create_audio_quality_analyzer(**kwargs)
    
    # AI Component Creation Methods
    def create_conversation_orchestrator(self, **kwargs):
        """Create Conversation Orchestrator"""
        return self.ai_factory.create_conversation_orchestrator(**kwargs)
    
    def create_intent_recognition_engine(self, **kwargs):
        """Create Intent Recognition Engine"""
        return self.ai_factory.create_intent_recognition_engine(**kwargs)
    
    def create_emotion_analysis_processor(self, **kwargs):
        """Create Emotion Analysis Processor"""
        return self.ai_factory.create_emotion_analysis_processor(**kwargs)
    
    def create_context_memory_manager(self, **kwargs):
        """Create Context Memory Manager"""
        return self.ai_factory.create_context_memory_manager(**kwargs)
    
    def create_response_generation_engine(self, **kwargs):
        """Create Response Generation Engine"""
        return self.ai_factory.create_response_generation_engine(**kwargs)
    
    def create_voice_personality_engine(self, **kwargs):
        """Create Voice Personality Engine"""
        return self.ai_factory.create_voice_personality_engine(**kwargs)
    
    def create_multi_modal_processor(self, **kwargs):
        """Create Multi-Modal Processor"""
        return self.ai_factory.create_multi_modal_processor(**kwargs)
    
    # Voice Component Creation Methods
    def create_real_time_audio_streamer(self, **kwargs):
        """Create Real-Time Audio Streamer"""
        return self.voice_factory.create_real_time_audio_streamer(**kwargs)
    
    def create_voice_cloning_engine(self, config: Optional[dict] = None, **kwargs):
        """Create Voice Cloning Engine"""
        return self.voice_factory.create_voice_cloning_engine(config=config, **kwargs)
    
    def create_advanced_wake_word_system(self, **kwargs):
        """Create Advanced Wake Word System"""
        return self.voice_factory.create_advanced_wake_word_system(**kwargs)
    
    def create_voice_biometrics_processor(self, **kwargs):
        """Create Voice Biometrics Processor"""
        return self.voice_factory.create_voice_biometrics_processor(**kwargs)
    
    def create_adaptive_noise_cancellation(self, **kwargs):
        """Create Adaptive Noise Cancellation"""
        return self.voice_factory.create_adaptive_noise_cancellation(**kwargs)
    
    def create_voice_quality_enhancer(self, **kwargs):
        """Create Voice Quality Enhancer"""
        return self.voice_factory.create_voice_quality_enhancer(**kwargs)
    
    # AGI Component Creation Methods
    def create_agi_multi_modal_processor(self, **kwargs):
        """Create AGI Multi-Modal Processor"""
        return self.agi_factory.create_multi_modal_processor(**kwargs)
    
    def create_predictive_conversation(self, **kwargs):
        """Create Predictive Conversation"""
        return self.agi_factory.create_predictive_conversation(**kwargs)
    
    def create_adaptive_learning_engine(self, **kwargs):
        """Create Adaptive Learning Engine"""
        return self.agi_factory.create_adaptive_learning_engine(**kwargs)
    
    def create_enterprise_security_layer(self, **kwargs):
        """Create Enterprise Security Layer"""
        return self.agi_factory.create_enterprise_security_layer(**kwargs)
    
    def create_performance_analytics(self, **kwargs):
        """Create Performance Analytics"""
        return self.agi_factory.create_performance_analytics(**kwargs)
    
    def create_proactive_assistant(self, **kwargs):
        """Create Proactive Assistant"""
        return self.agi_factory.create_proactive_assistant(**kwargs)
    
    def create_agi_voice_biometrics(self, **kwargs):
        """Create AGI Voice Biometrics"""
        return self.agi_factory.create_voice_biometrics(**kwargs)
    
    # Utility Methods
    def get_environment(self) -> str:
        """Get current environment setting"""
        return self.environment
    
    def set_environment(self, environment: str):
        """Set environment (production, testing, development)"""
        self.environment = environment
        os.environ["VORTA_ENVIRONMENT"] = environment
        logger.info(f"🔄 Environment changed to: {environment}")
    
    def create_full_voice_pipeline(self, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Create a complete voice processing pipeline with all components.
        
        Returns:
            Dict containing all initialized voice processing components
        """
        config = config or {}
        
        logger.info("🚀 Creating full VORTA voice processing pipeline...")
        
        pipeline = {
            # Phase 1: Audio Foundation
            'neural_vad': self.create_neural_vad_processor(**config.get('vad', {})),
            'wake_word_detector': self.create_wake_word_detector(**config.get('wake_word', {})),
            'noise_cancellation': self.create_noise_cancellation_engine(**config.get('noise_cancel', {})),
            'audio_stream_manager': self.create_audio_stream_manager(**config.get('stream', {})),
            'dsp_enhancement': self.create_dsp_enhancement_suite(**config.get('dsp', {})),
            'audio_quality_analyzer': self.create_audio_quality_analyzer(**config.get('quality', {})),
            
            # Phase 2: AI Intelligence
            'conversation_orchestrator': self.create_conversation_orchestrator(**config.get('orchestrator', {})),
            'intent_recognition': self.create_intent_recognition_engine(**config.get('intent', {})),
            'emotion_analysis': self.create_emotion_analysis_processor(**config.get('emotion', {})),
            'context_memory': self.create_context_memory_manager(**config.get('memory', {})),
            'response_generation': self.create_response_generation_engine(**config.get('response', {})),
            'voice_personality': self.create_voice_personality_engine(**config.get('personality', {})),
            'multi_modal': self.create_multi_modal_processor(**config.get('multimodal', {})),
            
            # Phase 3: Advanced Voice Processing
            'real_time_streamer': self.create_real_time_audio_streamer(**config.get('rt_stream', {})),
            'voice_cloning': self.create_voice_cloning_engine(**config.get('voice_clone', {})),
            'advanced_wake_word': self.create_advanced_wake_word_system(**config.get('adv_wake', {})),
            'voice_biometrics': self.create_voice_biometrics_processor(**config.get('biometrics', {})),
            'adaptive_noise_cancel': self.create_adaptive_noise_cancellation(**config.get('adv_noise', {})),
            'voice_quality_enhancer': self.create_voice_quality_enhancer(**config.get('voice_enhance', {})),
            
            # Phase 4: Enterprise AGI Features
            'agi_multi_modal': self.create_agi_multi_modal_processor(**config.get('agi_multi', {})),
            'predictive_conversation': self.create_predictive_conversation(**config.get('predictive', {})),
            'adaptive_learning': self.create_adaptive_learning_engine(**config.get('learning', {})),
            'enterprise_security': self.create_enterprise_security_layer(**config.get('security', {})),
            'performance_analytics': self.create_performance_analytics(**config.get('analytics', {})),
            'proactive_assistant': self.create_proactive_assistant(**config.get('proactive', {})),
            'agi_voice_biometrics': self.create_agi_voice_biometrics(**config.get('agi_bio', {}))
        }
        
        logger.info(f"✅ Full VORTA pipeline created with {len(pipeline)} components")
        return pipeline
    
    def get_component_status(self) -> Dict[str, Dict[str, Any]]:
        """
        Get status of all component categories.
        
        Returns:
            Dict with status information for each component category
        """
        return {
            'environment': self.environment,
            'factory_status': {
                'audio_factory': 'active',
                'ai_factory': 'active', 
                'voice_factory': 'active',
                'agi_factory': 'active'
            },
            'component_counts': {
                'audio_components': 6,
                'ai_components': 7,
                'voice_components': 6,
                'agi_components': 7,
                'total_components': 26
            },
            'mock_mode': self.environment == 'testing',
            'production_ready': self.environment == 'production'
        }

# Global factory manager instance
_factory_manager = None

def get_factory_manager(environment: Optional[str] = None) -> VORTAFactoryManager:
    """
    Get the global factory manager instance (singleton pattern).
    
    Args:
        environment: Optional environment override
        
    Returns:
        VORTAFactoryManager instance
    """
    global _factory_manager
    if _factory_manager is None or (environment and _factory_manager.environment != environment):
        _factory_manager = VORTAFactoryManager(environment)
    return _factory_manager

# Convenience functions for direct component creation
def create_component(component_type: str, component_name: str, **kwargs):
    """
    Create a component using the factory manager.
    
    Args:
        component_type: Type category (audio, ai, voice, agi)
        component_name: Specific component name
        **kwargs: Component configuration
        
    Returns:
        Initialized component instance
    """
    factory = get_factory_manager()
    method_name = f"create_{component_name}"
    
    if hasattr(factory, method_name):
        return getattr(factory, method_name)(**kwargs)
    else:
        raise ValueError(f"Unknown component: {component_type}.{component_name}")

__all__ = [
    'VORTAFactoryManager',
    'get_factory_manager', 
    'create_component'
]
