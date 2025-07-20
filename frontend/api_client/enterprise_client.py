"""
VORTA Enterprise API Client
Professional-grade API communication layer for VORTA AI Platform

Author: High-Grade Development Team
Version: 2.0.0-enterprise
License: MIT
"""

import datetime
import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

# Professional import handling for dependencies
try:
    import requests
except ImportError:
    print("Warning: requests not installed. Run: pip install requests")
    requests = None


@dataclass
class APIResponse:
    """Standardized API response container"""
    success: bool
    data: Dict[str, Any]
    status_code: Optional[int] = None
    response_time_ms: Optional[float] = None
    error_message: Optional[str] = None
    timestamp: Optional[str] = None


@dataclass
class VoiceProfile:
    """Enterprise voice profile specification"""
    voice_id: str
    display_name: str
    language: str
    gender: str
    is_premium: bool
    quality_rating: float
    latency_ms: float


class VortaEnterpriseAPIClient:
    """
    Enterprise-grade API client for VORTA AI Platform
    
    Features:
    - Automatic retry logic with exponential backoff
    - Comprehensive error handling and logging
    - Performance metrics tracking
    - Circuit breaker pattern implementation
    - Request/response validation
    - Enterprise security headers
    """
    
    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        timeout: int = 30,
        max_retries: int = 3,
        retry_delay: float = 1.0
    ):
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        # Initialize HTTP session with enterprise configuration
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'VORTA-Enterprise-Client/2.0.0',
            'Accept': 'application/json',
            'Content-Type': 'application/json',
            'X-Client-Version': '2.0.0',
            'X-Platform': 'VORTA-AI-Platform'
        })
        
        # Enterprise logging configuration
        self.logger = self._setup_logger()
        
        # Performance tracking
        self.request_metrics = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'average_response_time': 0.0,
            'circuit_breaker_trips': 0
        }
        
        # Circuit breaker state
        self.circuit_breaker_open = False
        self.circuit_breaker_failures = 0
        self.circuit_breaker_threshold = 5
        self.circuit_breaker_reset_time = None
        
    def _setup_logger(self) -> logging.Logger:
        """Configure enterprise logging"""
        logger = logging.getLogger('VortaEnterpriseAPI')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger
        
    def _check_circuit_breaker(self) -> bool:
        """Circuit breaker pattern implementation"""
        if not self.circuit_breaker_open:
            return True
            
        # Check if reset time has passed
        if (self.circuit_breaker_reset_time and
                time.time() > self.circuit_breaker_reset_time):
            self.circuit_breaker_open = False
            self.circuit_breaker_failures = 0
            self.logger.info("Circuit breaker reset - attempting requests")
            return True
            
        return False
        
    def _handle_circuit_breaker_failure(self):
        """Handle circuit breaker failure logic"""
        self.circuit_breaker_failures += 1
        
        if self.circuit_breaker_failures >= self.circuit_breaker_threshold:
            self.circuit_breaker_open = True
            # 60 second reset timeout
            self.circuit_breaker_reset_time = time.time() + 60
            self.request_metrics['circuit_breaker_trips'] += 1
            self.logger.warning("Circuit breaker opened - blocking requests")
            
    def _execute_request(
        self,
        method: str,
        endpoint: str,
        **kwargs
    ) -> APIResponse:
        """
        Execute HTTP request with enterprise error handling
        
        Args:
            method: HTTP method (GET, POST, PUT, DELETE)
            endpoint: API endpoint path
            **kwargs: Additional request parameters
            
        Returns:
            APIResponse: Standardized response object
        """
        if not self._check_circuit_breaker():
            return APIResponse(
                success=False,
                data={},
                error_message="Circuit breaker is open - requests blocked"
            )
            
        self.request_metrics['total_requests'] += 1
        start_time = time.time()
        
        url = f"{self.base_url}{endpoint}"
        
        for attempt in range(self.max_retries + 1):
            try:
                self.logger.debug(
                    f"Executing {method} request to {url} (attempt {attempt + 1})"
                )
                
                response = self.session.request(
                    method=method,
                    url=url,
                    timeout=self.timeout,
                    **kwargs
                )
                
                response_time = (time.time() - start_time) * 1000
                
                # Update performance metrics
                self._update_metrics(response_time, True)
                
                if response.status_code == 200:
                    try:
                        data = response.json()
                    except json.JSONDecodeError:
                        data = {'raw_response': response.text}
                        
                    return APIResponse(
                        success=True,
                        data=data,
                        status_code=response.status_code,
                        response_time_ms=round(response_time, 2),
                        timestamp=datetime.datetime.now().isoformat()
                    )
                else:
                    self.logger.warning(
                        f"HTTP {response.status_code}: {response.text}"
                    )
                    
            except requests.exceptions.Timeout:
                self.logger.warning(
                    f"Request timeout on attempt {attempt + 1}"
                )
                if attempt < self.max_retries:
                    # Exponential backoff
                    time.sleep(self.retry_delay * (2 ** attempt))
                    continue
                    
            except requests.exceptions.ConnectionError:
                self.logger.warning(
                    f"Connection error on attempt {attempt + 1}"
                )
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay * (2 ** attempt))
                    continue
                    
            except Exception as e:
                self.logger.error(f"Unexpected error: {e}")
                break
                
        # All retries failed
        self._update_metrics(0, False)
        self._handle_circuit_breaker_failure()
        
        return APIResponse(
            success=False,
            data={},
            error_message="Request failed after all retry attempts"
        )
        
    def _update_metrics(self, response_time: float, success: bool):
        """Update performance metrics"""
        if success:
            self.request_metrics['successful_requests'] += 1
            # Update average response time
            current_avg = self.request_metrics['average_response_time']
            total_successful = self.request_metrics['successful_requests']
            new_avg = ((current_avg * (total_successful - 1)) + 
                      response_time) / total_successful
            self.request_metrics['average_response_time'] = new_avg
        else:
            self.request_metrics['failed_requests'] += 1
            
    # API ENDPOINT METHODS
    
    def health_check(self) -> APIResponse:
        """Comprehensive health check with detailed diagnostics"""
        return self._execute_request('GET', '/api/health')
        
    def get_system_metrics(self) -> APIResponse:
        """Retrieve comprehensive system performance metrics"""
        return self._execute_request('GET', '/api/system/stats')
        
    def get_voice_profiles(self) -> APIResponse:
        """Get enterprise voice profiles with detailed specifications"""
        response = self._execute_request('GET', '/api/voices')
        
        if response.success and 'voices' in response.data:
            # Transform to VoiceProfile objects
            profiles = []
            for voice_data in response.data['voices']:
                profile = VoiceProfile(
                    voice_id=voice_data.get('voice_id', 'unknown'),
                    display_name=voice_data.get('name', 'Unknown Voice'),
                    language=voice_data.get('language', 'en'),
                    gender=voice_data.get('gender', 'neutral'),
                    is_premium=voice_data.get('is_premium', False),
                    quality_rating=voice_data.get('quality_rating', 0.0),
                    latency_ms=voice_data.get('latency_ms', 0.0)
                )
                profiles.append(profile)
                
            response.data['voice_profiles'] = profiles
            
        return response
        
    def synthesize_speech(
        self, 
        text: str, 
        voice_id: str = "alloy",
        quality: str = "premium",
        speed: float = 1.0
    ) -> APIResponse:
        """
        Enterprise text-to-speech synthesis
        
        Args:
            text: Text content to synthesize
            voice_id: Voice profile identifier
            quality: Audio quality setting (standard|premium|ultra)
            speed: Speech rate (0.5-2.0)
            
        Returns:
            APIResponse with audio data and metadata
        """
        payload = {
            "text": text,
            "voice": voice_id,
            "quality": quality,
            "speed": speed,
            "response_format": "mp3",
            "enable_ssml": True,
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        return self._execute_request('POST', '/api/tts', json=payload)
        
    def transcribe_audio(
        self, 
        audio_file,
        model: str = "whisper-large-v3",
        language: str = "auto"
    ) -> APIResponse:
        """
        Enterprise speech-to-text transcription
        
        Args:
            audio_file: Audio file object
            model: Whisper model to use
            language: Target language (auto for detection)
            
        Returns:
            APIResponse with transcription and metadata
        """
        try:
            files = {"file": audio_file}
            data = {
                "model": model,
                "language": language,
                "task": "transcribe",
                "temperature": 0.0,
                "response_format": "verbose_json",
                "timestamp_granularities": ["word", "segment"]
            }
            
            return self._execute_request(
                'POST', 
                '/api/speech/transcribe', 
                files=files, 
                data=data
            )
            
        except Exception as e:
            self.logger.error(f"Transcription error: {e}")
            return APIResponse(
                success=False,
                data={},
                error_message=f"Transcription failed: {str(e)}"
            )
            
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get client-side performance metrics"""
        return {
            **self.request_metrics,
            'circuit_breaker_open': self.circuit_breaker_open,
            'circuit_breaker_failures': self.circuit_breaker_failures
        }
