# frontend/components/audio/factory.py

"""
Factory for creating audio processing components.

This factory provides a centralized way to instantiate and configure
all components related to the audio processing pipeline. It ensures that
components are created with the correct dependencies and configurations,
switching between real and alternative implementations based on the environment.
"""

import os
from typing import Optional

from .audio_quality_analyzer import AudioQualityAnalyzer
from .audio_stream_manager import AudioStreamManager
from .dsp_enhancement_suite import DSPEnhancementSuite
from .neural_vad_processor import NeuralVADProcessor
from .noise_cancellation_engine import NoiseCancellationEngine
from .wake_word_detector import WakeWordDetector

# Environment variable to control implementation type
ENVIRONMENT = os.getenv("VORTA_ENVIRONMENT", "production")

class AudioComponentFactory:
    """
    A factory for creating instances of audio components.
    
    This class centralizes the creation of all audio-related components,
    allowing for easy swapping of implementations (e.g., for testing or
    different deployment environments).
    """

    @staticmethod
    def create_vad_processor(**kwargs) -> NeuralVADProcessor:
        """Creates a Voice Activity Detection (VAD) processor - PRODUCTION ONLY."""
        return NeuralVADProcessor(**kwargs)

    @staticmethod
    def create_wake_word_detector(
        config: Optional[dict] = None, 
        **kwargs
    ) -> WakeWordDetector:
        """Creates a wake word detection engine - PRODUCTION ONLY."""
        return WakeWordDetector(config=config, **kwargs)

    @staticmethod
    def create_noise_cancellation_engine(**kwargs) -> NoiseCancellationEngine:
        """Creates a noise cancellation engine - PRODUCTION ONLY."""
        return NoiseCancellationEngine(**kwargs)

    @staticmethod
    def create_audio_stream_manager(**kwargs) -> AudioStreamManager:
        """Creates an audio stream manager - PRODUCTION ONLY."""
        return AudioStreamManager(**kwargs)

    @staticmethod
    def create_dsp_enhancement_suite(**kwargs) -> DSPEnhancementSuite:
        """Creates a Digital Signal Processing (DSP) enhancement suite - PRODUCTION ONLY."""
        return DSPEnhancementSuite(**kwargs)

    @staticmethod
    def create_audio_quality_analyzer(**kwargs) -> AudioQualityAnalyzer:
        """Creates an audio quality analyzer - PRODUCTION ONLY."""
        return AudioQualityAnalyzer(**kwargs)

# Example of how to use the factory:
#
# if __name__ == '__main__':
#     # Set the environment for testing
#     os.environ["VORTA_ENVIRONMENT"] = "testing"
#
#     # Create components using the factory
#     vad_processor = AudioComponentFactory.create_vad_processor()
#     wake_word_detector = AudioComponentFactory.create_wake_word_detector()
#
#     print(f"Created VAD Processor: {type(vad_processor).__name__}")
#     print(f"Created Wake Word Detector: {type(wake_word_detector).__name__}")
#
#     # Switch back to production
#     os.environ["VORTA_ENVIRONMENT"] = "production"
#
#     vad_processor_prod = AudioComponentFactory.create_vad_processor()
#     print(f"Created Production VAD Processor: {type(vad_processor_prod).__name__}")

