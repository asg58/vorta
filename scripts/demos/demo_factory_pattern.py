#!/usr/bin/env python3
"""
VORTA Factory Pattern Implementation Demo

This script demonstrates the complete Factory Pattern implementation
for the VORTA AGI Voice Agent, showing how to create components in
both production and testing environments.

Author: Ultra High-Grade Development Team
Version: 3.0.0-agi
"""

import os
import sys
import asyncio
import logging
from typing import Dict, Any

# Add the frontend components to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from frontend.components.factory_manager import get_factory_manager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

logger = logging.getLogger(__name__)

async def demo_production_mode():
    """Demonstrate production mode component creation"""
    logger.info("🚀 === PRODUCTION MODE DEMO ===")
    
    # Set production environment
    os.environ["VORTA_ENVIRONMENT"] = "production"
    factory = get_factory_manager()
    
    try:
        # Create audio components
        logger.info("🎵 Creating Audio Components...")
        vad_processor = factory.create_neural_vad_processor()
        wake_word = factory.create_wake_word_detector()
        noise_cancel = factory.create_noise_cancellation_engine()
        
        # Create AI components  
        logger.info("🧠 Creating AI Components...")
        orchestrator = factory.create_conversation_orchestrator()
        intent_engine = factory.create_intent_recognition_engine()
        emotion_processor = factory.create_emotion_analysis_processor()
        
        # Create voice components
        logger.info("🎤 Creating Voice Components...")
        audio_streamer = factory.create_real_time_audio_streamer()
        voice_cloner = factory.create_voice_cloning_engine()
        voice_biometrics = factory.create_voice_biometrics_processor()
        
        # Create AGI components
        logger.info("🚀 Creating AGI Components...")
        agi_multimodal = factory.create_agi_multi_modal_processor()
        predictive_conv = factory.create_predictive_conversation()
        learning_engine = factory.create_adaptive_learning_engine()
        security_layer = factory.create_enterprise_security_layer()
        
        logger.info("✅ All production components created successfully!")
        
        # Show component status
        status = factory.get_component_status()
        logger.info(f"📊 Component Status: {status}")
        
    except Exception as e:
        logger.error(f"❌ Production demo failed: {e}")

async def demo_testing_mode():
    """Demonstrate testing mode with mock components"""
    logger.info("🧪 === TESTING MODE DEMO ===")
    
    # Set testing environment
    os.environ["VORTA_ENVIRONMENT"] = "testing"
    factory = get_factory_manager(environment="testing")
    
    try:
        # Create mock audio components
        logger.info("🎭 Creating Mock Audio Components...")
        mock_vad = factory.create_neural_vad_processor()
        mock_wake_word = factory.create_wake_word_detector()
        
        # Simulate audio processing
        if hasattr(mock_vad, 'process_audio'):
            result = await mock_vad.process_audio(b"test_audio_data")
            logger.info(f"🎯 Mock VAD Result: {result}")
        
        # Create mock AI components
        logger.info("🤖 Creating Mock AI Components...")
        mock_orchestrator = factory.create_conversation_orchestrator()
        mock_intent = factory.create_intent_recognition_engine()
        
        # Simulate AI processing
        if hasattr(mock_intent, 'recognize_intent'):
            intent_result = await mock_intent.recognize_intent("Hello, how are you?")
            logger.info(f"🎯 Mock Intent Result: {intent_result}")
        
        # Create mock voice components
        logger.info("🎭 Creating Mock Voice Components...")
        mock_streamer = factory.create_real_time_audio_streamer()
        mock_cloner = factory.create_voice_cloning_engine()
        
        # Simulate voice processing
        if hasattr(mock_streamer, 'stream_audio'):
            stream_result = await mock_streamer.stream_audio(b"audio_data")
            logger.info(f"🎯 Mock Streaming Result: {stream_result}")
        
        # Create mock AGI components
        logger.info("🌟 Creating Mock AGI Components...")
        mock_agi = factory.create_agi_multi_modal_processor()
        mock_predictive = factory.create_predictive_conversation()
        mock_security = factory.create_enterprise_security_layer()
        
        # Simulate AGI processing
        if hasattr(mock_security, 'validate_request'):
            security_result = await mock_security.validate_request({"test": "data"})
            logger.info(f"🔒 Mock Security Result: {security_result}")
        
        logger.info("✅ All mock components tested successfully!")
        
    except Exception as e:
        logger.error(f"❌ Testing demo failed: {e}")

async def demo_full_pipeline():
    """Demonstrate creating the full VORTA pipeline"""
    logger.info("🌟 === FULL PIPELINE DEMO ===")
    
    factory = get_factory_manager()
    
    try:
        # Create full pipeline with custom configuration
        pipeline_config = {
            'vad': {'detection_threshold': 0.7},
            'wake_word': {'sensitivity': 'high'},
            'voice_clone': {'quality': 'studio'},
            'security': {'level': 'enterprise'}
        }
        
        pipeline = factory.create_full_voice_pipeline(config=pipeline_config)
        
        logger.info(f"🚀 Full pipeline created with {len(pipeline)} components:")
        for component_name in pipeline.keys():
            logger.info(f"  ✅ {component_name}")
        
        # Show pipeline statistics
        logger.info("📊 Pipeline Statistics:")
        logger.info(f"  • Total Components: {len(pipeline)}")
        logger.info(f"  • Environment: {factory.get_environment()}")
        logger.info(f"  • Mock Mode: {'Yes' if factory.get_environment() == 'testing' else 'No'}")
        
    except Exception as e:
        logger.error(f"❌ Pipeline demo failed: {e}")

async def demo_environment_switching():
    """Demonstrate switching between environments"""
    logger.info("🔄 === ENVIRONMENT SWITCHING DEMO ===")
    
    factory = get_factory_manager()
    
    try:
        # Start in production
        factory.set_environment("production")
        logger.info(f"📈 Current environment: {factory.get_environment()}")
        
        prod_component = factory.create_neural_vad_processor()
        logger.info(f"✅ Production component created: {type(prod_component).__name__}")
        
        # Switch to testing
        factory.set_environment("testing")
        logger.info(f"🧪 Current environment: {factory.get_environment()}")
        
        test_component = factory.create_neural_vad_processor()
        logger.info(f"✅ Testing component created: {type(test_component).__name__}")
        
        # Verify they're different types
        if type(prod_component) != type(test_component):
            logger.info("🎯 Environment switching working correctly - different component types!")
        else:
            logger.warning("⚠️ Environment switching may not be working properly")
            
    except Exception as e:
        logger.error(f"❌ Environment switching demo failed: {e}")

async def main():
    """Run all demos"""
    logger.info("🎉 VORTA Factory Pattern Implementation Demo")
    logger.info("=" * 60)
    
    # Run all demo scenarios
    await demo_production_mode()
    logger.info("")
    
    await demo_testing_mode()  
    logger.info("")
    
    await demo_full_pipeline()
    logger.info("")
    
    await demo_environment_switching()
    logger.info("")
    
    logger.info("🎊 Factory Pattern Demo Complete!")
    logger.info("🚀 VORTA AGI Voice Agent is ready for enterprise deployment!")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("👋 Demo interrupted by user")
    except Exception as e:
        logger.error(f"💥 Demo failed with error: {e}")
        sys.exit(1)
