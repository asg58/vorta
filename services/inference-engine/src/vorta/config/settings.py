"""
VORTA Inference Engine Settings
Pydantic-based configuration management with environment variable support
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import Field, validator
from pydantic_settings import BaseSettings

# Load .env file
try:
    from dotenv import load_dotenv
    # Load .env from the service root directory
    env_path = Path(__file__).parent.parent.parent.parent / ".env"
    load_dotenv(env_path)
except ImportError:
    pass


class DatabaseSettings(BaseSettings):
    """Database configuration"""
    host: str = Field(default="localhost", env="DB_HOST")
    port: int = Field(default=5432, env="DB_PORT")
    database: str = Field(default="vorta", env="DB_NAME")
    username: str = Field(default="postgres", env="DB_USER")
    password: str = Field(default="", env="DB_PASSWORD")
    pool_size: int = Field(default=10, env="DB_POOL_SIZE")
    
    @property
    def url(self) -> str:
        return f"postgresql://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"

class RedisSettings(BaseSettings):
    """Redis configuration for caching"""
    host: str = Field(default="localhost", env="REDIS_HOST")
    port: int = Field(default=6379, env="REDIS_PORT")
    database: int = Field(default=0, env="REDIS_DB")
    password: Optional[str] = Field(default=None, env="REDIS_PASSWORD")
    ttl: int = Field(default=3600, env="REDIS_TTL")  # Cache TTL in seconds
    max_connections: int = Field(default=20, env="REDIS_MAX_CONNECTIONS")
    
    @property
    def url(self) -> str:
        auth = f":{self.password}@" if self.password else ""
        return f"redis://{auth}{self.host}:{self.port}/{self.database}"

class ModelSettings(BaseSettings):
    """Model configuration"""
    models_path: str = Field(default="./models", env="MODELS_PATH")
    default_model: str = Field(default="llama2-7b", env="DEFAULT_MODEL")
    max_models_loaded: int = Field(default=3, env="MAX_MODELS_LOADED")
    model_cache_size: int = Field(default=1024, env="MODEL_CACHE_SIZE")  # MB
    auto_load_models: bool = Field(default=True, env="AUTO_LOAD_MODELS")
    model_timeout: int = Field(default=300, env="MODEL_TIMEOUT")  # seconds
    
    # Model-specific configurations
    text_generation_models: List[str] = Field(
        default=["llama2-7b", "gpt-3.5-turbo", "mistral-7b"],
        env="TEXT_GENERATION_MODELS"
    )
    image_classification_models: List[str] = Field(
        default=["resnet50", "efficientnet-b0"],
        env="IMAGE_CLASSIFICATION_MODELS"
    )
    embedding_models: List[str] = Field(
        default=["sentence-transformers/all-MiniLM-L6-v2"],
        env="EMBEDDING_MODELS"
    )

class APISettings(BaseSettings):
    """API configuration"""
    host: str = Field(default="0.0.0.0", env="API_HOST")
    port: int = Field(default=8000, env="API_PORT")
    workers: int = Field(default=1, env="API_WORKERS")
    reload: bool = Field(default=False, env="API_RELOAD")
    
    # Security
    require_api_key: bool = Field(default=False, env="REQUIRE_API_KEY")
    api_keys: List[str] = Field(default=[], env="API_KEYS")
    cors_origins: List[str] = Field(
        default=["http://localhost:3000", "http://localhost:8080"],
        env="CORS_ORIGINS"
    )
    
    # Rate limiting
    rate_limit_enabled: bool = Field(default=True, env="RATE_LIMIT_ENABLED")
    requests_per_minute: int = Field(default=100, env="REQUESTS_PER_MINUTE")
    requests_per_day: int = Field(default=10000, env="REQUESTS_PER_DAY")
    
    # Timeouts
    default_timeout: int = Field(default=30, env="DEFAULT_TIMEOUT")
    max_timeout: int = Field(default=300, env="MAX_TIMEOUT")
    
    # Batch processing
    max_batch_size: int = Field(default=100, env="MAX_BATCH_SIZE")
    batch_timeout: int = Field(default=60, env="BATCH_TIMEOUT")

class LoggingSettings(BaseSettings):
    """Logging configuration"""
    level: str = Field(default="INFO", env="LOG_LEVEL")
    format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        env="LOG_FORMAT"
    )
    file_enabled: bool = Field(default=True, env="LOG_FILE_ENABLED")
    file_path: str = Field(default="logs/inference-engine.log", env="LOG_FILE_PATH")
    file_max_size: int = Field(default=10, env="LOG_FILE_MAX_SIZE")  # MB
    file_backup_count: int = Field(default=5, env="LOG_FILE_BACKUP_COUNT")
    
    # Structured logging
    json_logging: bool = Field(default=False, env="JSON_LOGGING")
    
    @validator('level')
    def validate_log_level(cls, v):
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if v.upper() not in valid_levels:
            raise ValueError(f'Log level must be one of {valid_levels}')
        return v.upper()

class MonitoringSettings(BaseSettings):
    """Monitoring and metrics configuration"""
    enabled: bool = Field(default=True, env="MONITORING_ENABLED")
    prometheus_enabled: bool = Field(default=True, env="PROMETHEUS_ENABLED")
    prometheus_port: int = Field(default=9090, env="PROMETHEUS_PORT")
    
    # Health checks
    health_check_interval: int = Field(default=30, env="HEALTH_CHECK_INTERVAL")  # seconds
    
    # Metrics collection
    collect_model_metrics: bool = Field(default=True, env="COLLECT_MODEL_METRICS")
    collect_system_metrics: bool = Field(default=True, env="COLLECT_SYSTEM_METRICS")
    metrics_retention_days: int = Field(default=30, env="METRICS_RETENTION_DAYS")

class Settings(BaseSettings):
    """Main application settings"""
    
    # Environment
    environment: str = Field(default="development", env="ENVIRONMENT")
    debug: bool = Field(default=False, env="DEBUG")
    version: str = Field(default="1.0.0", env="VERSION")
    service_name: str = Field(default="vorta-inference-engine", env="SERVICE_NAME")
    
    # External API keys
    openai_api_key: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    eleven_labs_api_key: Optional[str] = Field(default=None, env="ELEVEN_LABS_API_KEY")
    
    # Component settings
    database: DatabaseSettings = DatabaseSettings()
    redis: RedisSettings = RedisSettings()
    models: ModelSettings = ModelSettings()
    api: APISettings = APISettings()
    logging: LoggingSettings = LoggingSettings()
    monitoring: MonitoringSettings = MonitoringSettings()
    
    # GPU/Hardware settings
    use_gpu: bool = Field(default=True, env="USE_GPU")
    gpu_memory_fraction: float = Field(default=0.8, env="GPU_MEMORY_FRACTION")
    cpu_cores: Optional[int] = Field(default=None, env="CPU_CORES")
    
    # Advanced settings
    experimental_features: bool = Field(default=False, env="EXPERIMENTAL_FEATURES")
    cache_enabled: bool = Field(default=True, env="CACHE_ENABLED")
    async_processing: bool = Field(default=True, env="ASYNC_PROCESSING")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        
    @validator('environment')
    def validate_environment(cls, v):
        valid_envs = ['development', 'staging', 'production', 'testing']
        if v.lower() not in valid_envs:
            raise ValueError(f'Environment must be one of {valid_envs}')
        return v.lower()
    
    @validator('gpu_memory_fraction')
    def validate_gpu_memory(cls, v):
        if not 0.1 <= v <= 1.0:
            raise ValueError('GPU memory fraction must be between 0.1 and 1.0')
        return v
    
    @property
    def is_development(self) -> bool:
        return self.environment == "development"
    
    @property
    def is_production(self) -> bool:
        return self.environment == "production"
    
    @property
    def models_directory(self) -> Path:
        return Path(self.models.models_path)
    
    @property
    def logs_directory(self) -> Path:
        return Path(self.logging.file_path).parent
    
    def create_directories(self):
        """Create necessary directories"""
        self.models_directory.mkdir(parents=True, exist_ok=True)
        self.logs_directory.mkdir(parents=True, exist_ok=True)
    
    def load_model_config(self, config_file: str = "models.json") -> Dict[str, Any]:
        """Load model configuration from file"""
        config_path = self.models_directory / config_file
        if config_path.exists():
            with open(config_path, 'r') as f:
                return json.load(f)
        return {}
    
    def save_model_config(self, config: Dict[str, Any], config_file: str = "models.json"):
        """Save model configuration to file"""
        config_path = self.models_directory / config_file
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)

# Global settings instance
_settings: Optional[Settings] = None

def get_settings() -> Settings:
    """Get application settings (singleton)"""
    global _settings
    if _settings is None:
        _settings = Settings()
        _settings.create_directories()
    return _settings

def reload_settings():
    """Reload settings (useful for testing)"""
    global _settings
    _settings = None
    return get_settings()