-- VORTA ULTRA PostgreSQL Database Initialization
-- Enterprise-grade database setup for VORTA AI Platform

-- Create main database if not exists
SELECT 'CREATE DATABASE vorta_ultra'
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'vorta_ultra');

-- Connect to the database
\c vorta_ultra;

-- Create extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";

-- Create users table
CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    is_active BOOLEAN DEFAULT TRUE,
    is_admin BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    last_login TIMESTAMP WITH TIME ZONE
);

-- Create voice sessions table
CREATE TABLE IF NOT EXISTS voice_sessions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    session_type VARCHAR(20) CHECK (session_type IN ('tts', 'stt', 'conversation')),
    input_text TEXT,
    output_text TEXT,
    voice_id VARCHAR(50),
    processing_time_ms INTEGER,
    quality_score FLOAT,
    confidence_score FLOAT,
    file_size_bytes INTEGER,
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create system metrics table
CREATE TABLE IF NOT EXISTS system_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    metric_type VARCHAR(50) NOT NULL,
    metric_name VARCHAR(100) NOT NULL,
    metric_value FLOAT NOT NULL,
    unit VARCHAR(20),
    labels JSONB,
    recorded_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create audio files table
CREATE TABLE IF NOT EXISTS audio_files (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id UUID REFERENCES voice_sessions(id) ON DELETE CASCADE,
    file_name VARCHAR(255) NOT NULL,
    file_path TEXT NOT NULL,
    file_size_bytes INTEGER NOT NULL,
    mime_type VARCHAR(100) NOT NULL,
    duration_seconds FLOAT,
    sample_rate INTEGER,
    channels INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create API logs table
CREATE TABLE IF NOT EXISTS api_logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE SET NULL,
    endpoint VARCHAR(255) NOT NULL,
    method VARCHAR(10) NOT NULL,
    status_code INTEGER NOT NULL,
    response_time_ms INTEGER NOT NULL,
    request_size_bytes INTEGER,
    response_size_bytes INTEGER,
    ip_address INET,
    user_agent TEXT,
    request_headers JSONB,
    response_headers JSONB,
    error_message TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_users_username ON users(username);
CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
CREATE INDEX IF NOT EXISTS idx_voice_sessions_user_id ON voice_sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_voice_sessions_created_at ON voice_sessions(created_at);
CREATE INDEX IF NOT EXISTS idx_voice_sessions_session_type ON voice_sessions(session_type);
CREATE INDEX IF NOT EXISTS idx_system_metrics_type_name ON system_metrics(metric_type, metric_name);
CREATE INDEX IF NOT EXISTS idx_system_metrics_recorded_at ON system_metrics(recorded_at);
CREATE INDEX IF NOT EXISTS idx_audio_files_session_id ON audio_files(session_id);
CREATE INDEX IF NOT EXISTS idx_api_logs_created_at ON api_logs(created_at);
CREATE INDEX IF NOT EXISTS idx_api_logs_endpoint ON api_logs(endpoint);
CREATE INDEX IF NOT EXISTS idx_api_logs_user_id ON api_logs(user_id);

-- Create triggers for updated_at
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_users_updated_at BEFORE UPDATE ON users
FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Create default admin user (change password in production!)
INSERT INTO users (username, email, password_hash, is_admin) VALUES
('admin', 'admin@vorta-ultra.com', crypt('VortaUltraAdmin2024', gen_salt('bf')), TRUE)
ON CONFLICT (username) DO NOTHING;

-- Create sample voice profiles data
INSERT INTO voice_sessions (user_id, session_type, input_text, voice_id, processing_time_ms, quality_score) VALUES
((SELECT id FROM users WHERE username = 'admin'), 'tts', 'Welcome to VORTA ULTRA!', 'alloy', 1250, 98.5),
((SELECT id FROM users WHERE username = 'admin'), 'tts', 'Enterprise AI Platform is ready', 'nova', 980, 99.2)
ON CONFLICT DO NOTHING;

-- Grant permissions
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO vorta_admin;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO vorta_admin;

-- Create read-only user for reporting
CREATE USER vorta_readonly WITH PASSWORD 'VortaReadOnly2024';
GRANT CONNECT ON DATABASE vorta_ultra TO vorta_readonly;
GRANT USAGE ON SCHEMA public TO vorta_readonly;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO vorta_readonly;

-- Success message
SELECT 'VORTA ULTRA PostgreSQL database initialized successfully!' as status;
