"""
VORTA Ultra Audio Processing Backend
NO PyAudio - Using professional audio libraries instead!
"""

import io
import logging
from typing import Optional

import numpy as np

# Professional audio processing stack
try:
    import librosa
    import pydub
    import soundfile as sf
    from pydub import AudioSegment
except ImportError as e:
    logging.warning(f"Audio library not available: {e}")
    sf = librosa = pydub = AudioSegment = None

logger = logging.getLogger(__name__)

class VortaAudioProcessor:
    """
    Enterprise-grade audio processor for VORTA voice pipeline
    Uses soundfile + librosa instead of PyAudio for better quality
    """
    
    def __init__(self):
        self.sample_rate = 16000  # Whisper optimal rate
        self.chunk_duration = 0.1  # 100ms chunks
        self.chunk_size = int(self.sample_rate * self.chunk_duration)
        
    async def process_audio_chunk(self, audio_data: bytes) -> Optional[np.ndarray]:
        """
        Process incoming audio chunk from WebSocket
        Convert to numpy array for Whisper processing
        """
        try:
            # Convert WebM/Opus to WAV using pydub
            audio_segment = AudioSegment.from_file(
                io.BytesIO(audio_data), 
                format="webm"
            )
            
            # Convert to mono, 16kHz for Whisper
            audio_segment = audio_segment.set_channels(1)
            audio_segment = audio_segment.set_frame_rate(self.sample_rate)
            
            # Convert to numpy array
            audio_array = np.array(audio_segment.get_array_of_samples())
            audio_array = audio_array.astype(np.float32) / 32768.0  # Normalize
            
            return audio_array
            
        except Exception as e:
            logger.error(f"❌ Audio chunk processing failed: {e}")
            return None
    
    async def process_audio_file(self, file_path: str) -> Optional[np.ndarray]:
        """
        Process uploaded audio file using soundfile + librosa
        Superior quality compared to PyAudio
        """
        try:
            # Load audio file with librosa (handles all formats)
            audio_data, sr = librosa.load(
                file_path, 
                sr=self.sample_rate,  # Resample to 16kHz
                mono=True  # Convert to mono
            )
            
            # Audio enhancement using librosa
            audio_data = self.enhance_audio_quality(audio_data, sr)
            
            return audio_data
            
        except Exception as e:
            logger.error(f"❌ Audio file processing failed: {e}")
            return None
    
    def enhance_audio_quality(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """
        Professional audio enhancement using librosa
        Much better than PyAudio's basic processing
        """
        try:
            # Remove silence
            audio_trimmed, _ = librosa.effects.trim(audio, top_db=20)
            
            # Normalize audio levels
            audio_normalized = librosa.util.normalize(audio_trimmed)
            
            # Reduce noise (spectral subtraction)
            # Note: For production, consider using more advanced noise reduction
            stft = librosa.stft(audio_normalized)
            magnitude = np.abs(stft)
            phase = np.angle(stft)
            
            # Simple noise gate
            noise_threshold = np.percentile(magnitude, 30)
            magnitude = np.where(magnitude < noise_threshold, 
                               magnitude * 0.1, magnitude)
            
            # Reconstruct audio
            enhanced_stft = magnitude * np.exp(1j * phase)
            enhanced_audio = librosa.istft(enhanced_stft)
            
            return enhanced_audio
            
        except Exception as e:
            logger.warning(f"⚠️ Audio enhancement failed, using original: {e}")
            return audio
    
    async def save_audio_file(self, audio_data: np.ndarray, 
                            output_path: str, format: str = "wav") -> bool:
        """
        Save processed audio using soundfile
        Professional quality output
        """
        try:
            sf.write(output_path, audio_data, self.sample_rate, format=format)
            logger.info(f"✅ Audio saved: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Audio save failed: {e}")
            return False
    
    def get_audio_stats(self, audio_data: np.ndarray) -> dict:
        """
        Get professional audio statistics using librosa
        """
        try:
            stats = {
                "duration": len(audio_data) / self.sample_rate,
                "sample_rate": self.sample_rate,
                "channels": 1,  # Always mono for Whisper
                "rms_energy": float(np.sqrt(np.mean(audio_data**2))),
                "peak_amplitude": float(np.max(np.abs(audio_data))),
                "zero_crossing_rate": float(np.mean(librosa.feature.zero_crossing_rate(audio_data))),
                "spectral_centroid": float(np.mean(librosa.feature.spectral_centroid(
                    y=audio_data, sr=self.sample_rate
                )))
            }
            return stats
            
        except Exception as e:
            logger.error(f"❌ Audio stats calculation failed: {e}")
            return {}

# Global audio processor instance
audio_processor = VortaAudioProcessor()

# WebSocket audio streaming handler
async def handle_audio_stream(websocket, audio_chunk: bytes):
    """
    Handle incoming audio stream from WebSocket
    Process with professional audio stack (NO PyAudio!)
    """
    try:
        # Process audio chunk
        audio_array = await audio_processor.process_audio_chunk(audio_chunk)
        
        if audio_array is not None:
            # Get audio quality metrics
            stats = audio_processor.get_audio_stats(audio_array)
            
            # Log quality info
            logger.info(f"🎤 Audio chunk processed: "
                       f"duration={stats.get('duration', 0):.2f}s, "
                       f"rms={stats.get('rms_energy', 0):.3f}")
            
            return audio_array
        else:
            logger.warning("⚠️ Audio chunk processing returned None")
            return None
            
    except Exception as e:
        logger.error(f"❌ Audio stream handling failed: {e}")
        return None

# Export for use in dashboard server
__all__ = ['VortaAudioProcessor', 'audio_processor', 'handle_audio_stream']
