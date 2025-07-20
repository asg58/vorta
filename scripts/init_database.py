#!/usr/bin/env python3
"""
🚀 VORTA ULTRA - Database Initialization Script
Creates all necessary tables and indexes for optimal performance
"""

import asyncio
import logging
import os

import asyncpg

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://vorta_admin:VortaUltraAdmin2024@localhost:5432/vorta_ultra")

async def init_database():
    """Initialize the VORTA ULTRA database schema"""
    
    try:
        conn = await asyncpg.connect(DATABASE_URL)
        logger.info("✅ Connected to PostgreSQL database")
        
        # Create tables
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS tts_requests (
                id SERIAL PRIMARY KEY,
                text TEXT NOT NULL,
                voice VARCHAR(50) NOT NULL,
                quality VARCHAR(20) NOT NULL,
                processing_time_ms INTEGER NOT NULL,
                created_at TIMESTAMP DEFAULT NOW(),
                user_id VARCHAR(100),
                audio_url VARCHAR(500)
            );
        """)
        
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS voice_profiles (
                id SERIAL PRIMARY KEY,
                voice_id VARCHAR(50) UNIQUE NOT NULL,
                name VARCHAR(100) NOT NULL,
                gender VARCHAR(20),
                language VARCHAR(20),
                description TEXT,
                is_premium BOOLEAN DEFAULT FALSE,
                quality_score DECIMAL(3,2),
                latency_ms INTEGER,
                created_at TIMESTAMP DEFAULT NOW()
            );
        """)
        
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS api_metrics (
                id SERIAL PRIMARY KEY,
                endpoint VARCHAR(200) NOT NULL,
                method VARCHAR(10) NOT NULL,
                response_time_ms INTEGER NOT NULL,
                status_code INTEGER NOT NULL,
                timestamp TIMESTAMP DEFAULT NOW(),
                user_agent TEXT,
                ip_address INET
            );
        """)
        
        # Create indexes for performance
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_tts_created_at ON tts_requests(created_at);")
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_tts_voice ON tts_requests(voice);")
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_api_metrics_endpoint ON api_metrics(endpoint);")
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_api_metrics_timestamp ON api_metrics(timestamp);")
        
        # Insert default voice profiles
        voices_data = [
            ('alloy', 'Alloy Neural', 'neutral', 'en-US', 'Advanced neural voice with perfect clarity', False, 0.95, 125),
            ('echo', 'Echo Quantum', 'male', 'en-US', 'Deep resonant voice with quantum processing', True, 0.98, 89),
            ('fable', 'Fable Ultra', 'female', 'en-US', 'Storytelling voice with emotional intelligence', True, 0.97, 112),
            ('nova', 'Nova Hyperspace', 'female', 'en-US', 'Crystal clear voice with hyperspace processing', False, 0.96, 105),
            ('onyx', 'Onyx Professional', 'male', 'en-US', 'Business-grade voice for professional use', True, 0.99, 78),
            ('shimmer', 'Shimmer Ethereal', 'female', 'en-US', 'Light, ethereal voice with shimmer effects', False, 0.94, 134)
        ]
        
        for voice_data in voices_data:
            await conn.execute("""
                INSERT INTO voice_profiles (voice_id, name, gender, language, description, is_premium, quality_score, latency_ms)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                ON CONFLICT (voice_id) DO NOTHING;
            """, *voice_data)
        
        logger.info("✅ Database schema initialized successfully")
        logger.info("✅ Voice profiles inserted")
        logger.info("✅ Performance indexes created")
        
        await conn.close()
        
    except Exception as e:
        logger.error(f"❌ Database initialization failed: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(init_database())
