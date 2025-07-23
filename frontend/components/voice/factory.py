# frontend/components/voice/factory.py

"""
Factory for creating voice processing components.

This factory provides a centralized way to instantiate and configure
all components related to the voice processing pipeline. It ensures that
components are created with the correct dependencies and configurations,
switching between real and mock implementations based on the environment.
"""

import os
from typing import Optional

from .adaptive_noise_cancellation import (
    AdaptiveNoiseCancellationEngine as AdaptiveNoiseCancellation,
)
from .advanced_wake_word_system import AdvancedWakeWordSystem
from .real_time_audio_streamer import RealTimeAudioStreamer
from .voice_biometrics_processor import VoiceBiometricsProcessor
from .voice_cloning_engine import VoiceCloningEngine
from .voice_quality_enhancer import VoiceQualityEnhancer

# Environment variable to control implementation type
ENVIRONMENT = os.getenv("VORTA_ENVIRONMENT", "production")

class VoiceComponentFactory:
    """
    A factory for creating instances of voice processing components.
    
    This class centralizes the creation of all voice-related components,
    allowing for easy swapping of implementations (e.g., for testing or
    different deployment environments).
    """

    @staticmethod
    def create_real_time_audio_streamer(**kwargs) -> RealTimeAudioStreamer:
        """Creates a Real-Time Audio Streamer - PRODUCTION ONLY."""
        return RealTimeAudioStreamer(**kwargs)

    @staticmethod
    def create_voice_cloning_engine(
        config: Optional[dict] = None, 
        **kwargs
    ) -> VoiceCloningEngine:
        """Creates a Voice Cloning Engine - PRODUCTION ONLY."""
        return VoiceCloningEngine(config=config, **kwargs)

    @staticmethod
    def create_advanced_wake_word_system(**kwargs) -> AdvancedWakeWordSystem:
        """Creates an Advanced Wake Word System - PRODUCTION ONLY."""
        return AdvancedWakeWordSystem(**kwargs)

    @staticmethod
    def create_voice_biometrics_processor(**kwargs) -> VoiceBiometricsProcessor:
        """Creates a Voice Biometrics Processor - PRODUCTION ONLY."""
        return VoiceBiometricsProcessor(**kwargs)

    @staticmethod
    def create_adaptive_noise_cancellation(**kwargs) -> AdaptiveNoiseCancellation:
        """Creates an Adaptive Noise Cancellation engine - PRODUCTION ONLY."""
        return AdaptiveNoiseCancellation(**kwargs)

    @staticmethod
    def create_voice_quality_enhancer(**kwargs) -> VoiceQualityEnhancer:
        """Creates a Voice Quality Enhancer - PRODUCTION ONLY."""
        return VoiceQualityEnhancer(**kwargs)

# Example of how to use the factory:
#
# if __name__ == '__main__':
#     # Set the environment for testing
#     os.environ["VORTA_ENVIRONMENT"] = "testing"
#
#     # Create components using the factory
#     audio_streamer = VoiceComponentFactory.create_real_time_audio_streamer()
#     voice_cloning = VoiceComponentFactory.create_voice_cloning_engine()
#     wake_word = VoiceComponentFactory.create_advanced_wake_word_system()
#
#     # Set the environment for production
#     os.environ["VORTA_ENVIRONMENT"] = "production"
#
#     # Create production-ready components
#     prod_streamer = VoiceComponentFactory.create_real_time_audio_streamer()
