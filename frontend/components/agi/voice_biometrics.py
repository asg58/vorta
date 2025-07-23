# frontend/components/agi/voice_biometrics.py
"""
VORTA: Voice Biometrics Security Module

This module provides advanced voice biometric capabilities, including speaker
identification ('who is speaking?') and verification ('is this person who they
claim to be?'). It's a critical component for personalized and secure interactions.

Note: This is a high-level simulation. A real-world implementation would require
deep learning models (e.g., using PyTorch or TensorFlow) trained on large voice datasets.
"""

import logging
import asyncio
from typing import Dict, Optional, List
from dataclasses import dataclass, field
import numpy as np
import hashlib

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Optional dependencies for audio processing
try:
    import librosa
    _AUDIO_LIBS_AVAILABLE = True
except ImportError:
    _AUDIO_LIBS_AVAILABLE = False

# --- Data Structures for Voice Biometrics ---

@dataclass
class Voiceprint:
    """Represents a user's unique voice characteristics."""
    user_id: str
    feature_vector: np.ndarray # A numerical representation of the voice
    enrollment_date: str
    version: str = "1.0"

@dataclass
class VerificationResult:
    """The result of a voice verification attempt."""
    is_verified: bool
    confidence: float # A score from 0.0 to 1.0
    message: str

# --- Core Biometrics Engine ---

class VoiceBiometricsEngine:
    """
    Handles the creation, storage, and comparison of voiceprints.
    """
    def __init__(self, threshold: float = 0.85):
        # In a real system, this would be a secure, encrypted database.
        self.voiceprint_db: Dict[str, Voiceprint] = {}
        self.threshold = threshold # Cosine similarity threshold for verification
        self.lock = asyncio.Lock()
        logger.info(f"VoiceBiometricsEngine initialized with verification threshold {self.threshold}.")

    def _extract_features(self, audio_data: np.ndarray, sr: int) -> np.ndarray:
        """
        Extracts a feature vector from audio data.
        This is the most critical part. A real system would use a sophisticated
        model like x-vectors or d-vectors. Here, we simulate it with MFCCs.
        """
        if not _AUDIO_LIBS_AVAILABLE:
            logger.warning("Librosa not found. Using dummy feature extraction.")
            # Fallback: hash of the audio data to simulate a feature vector
            return np.array([int(c, 16) for c in hashlib.sha256(audio_data.tobytes()).hexdigest()[:16]])

        # Use Mel-Frequency Cepstral Coefficients (MFCCs) as a basic feature
        mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=20)
        # Aggregate features over time (e.g., by taking the mean)
        return np.mean(mfccs, axis=1)

    async def enroll_user(self, user_id: str, audio_samples: List[np.ndarray], sr: int) -> bool:
        """
        Creates a new voiceprint for a user from multiple audio samples.
        """
        logger.info(f"Starting enrollment for user '{user_id}' with {len(audio_samples)} samples.")
        if len(audio_samples) < 3:
            logger.warning("Enrollment requires at least 3 audio samples for robustness.")
            return False

        try:
            # Extract features from all samples and average them to create a robust voiceprint
            feature_vectors = [self._extract_features(sample, sr) for sample in audio_samples]
            avg_feature_vector = np.mean(feature_vectors, axis=0)
            
            # Normalize the vector
            avg_feature_vector /= np.linalg.norm(avg_feature_vector)

            new_voiceprint = Voiceprint(
                user_id=user_id,
                feature_vector=avg_feature_vector,
                enrollment_date=str(datetime.now())
            )

            async with self.lock:
                self.voiceprint_db[user_id] = new_voiceprint
            
            logger.info(f"Successfully enrolled user '{user_id}'.")
            return True
        except Exception as e:
            logger.error(f"Error during enrollment for user '{user_id}': {e}", exc_info=True)
            return False

    async def verify_user(self, user_id: str, audio_sample: np.ndarray, sr: int) -> VerificationResult:
        """
        Verifies if an audio sample matches a user's enrolled voiceprint.
        """
        logger.info(f"Attempting verification for user '{user_id}'.")
        async with self.lock:
            if user_id not in self.voiceprint_db:
                return VerificationResult(is_verified=False, confidence=0.0, message="User not enrolled.")

            enrolled_voiceprint = self.voiceprint_db[user_id]
        
        try:
            incoming_features = self._extract_features(audio_sample, sr)
            incoming_features /= np.linalg.norm(incoming_features)

            # Calculate cosine similarity between the enrolled and incoming voiceprints
            similarity = np.dot(enrolled_voiceprint.feature_vector, incoming_features)
            
            is_verified = similarity >= self.threshold
            message = "Verification successful." if is_verified else "Verification failed: voice does not match."
            
            logger.info(f"Verification for '{user_id}': similarity={similarity:.4f}, verified={is_verified}")
            
            return VerificationResult(is_verified=is_verified, confidence=float(similarity), message=message)
        except Exception as e:
            logger.error(f"Error during verification for user '{user_id}': {e}", exc_info=True)
            return VerificationResult(is_verified=False, confidence=0.0, message="An error occurred during processing.")

    async def identify_speaker(self, audio_sample: np.ndarray, sr: int) -> Optional[str]:
        """
        Identifies who is speaking from a list of enrolled users (1:N matching).
        """
        logger.info("Attempting to identify speaker from database.")
        if not self.voiceprint_db:
            logger.warning("No users enrolled in the voiceprint database.")
            return None

        incoming_features = self._extract_features(audio_sample, sr)
        incoming_features /= np.linalg.norm(incoming_features)

        best_match_user = None
        highest_similarity = -1.0

        async with self.lock:
            for user_id, voiceprint in self.voiceprint_db.items():
                similarity = np.dot(voiceprint.feature_vector, incoming_features)
                if similarity > highest_similarity:
                    highest_similarity = similarity
                    best_match_user = user_id
        
        if highest_similarity >= self.threshold:
            logger.info(f"Identified speaker as '{best_match_user}' with similarity {highest_similarity:.4f}.")
            return best_match_user
        else:
            logger.info(f"Could not identify speaker. Best match '{best_match_user}' was below threshold ({highest_similarity:.4f}).")
            return None

# --- Example Usage ---

async def main():
    """Demonstrates the functionality of the VoiceBiometricsEngine."""
    logger.info("--- VORTA Voice Biometrics Engine Demonstration ---")

    if not _AUDIO_LIBS_AVAILABLE:
        logger.error("Cannot run demonstration without 'librosa' and 'numpy'. Please install them.")
        return

    engine = VoiceBiometricsEngine(threshold=0.9)
    sr = 22050

    # 1. Enroll two users
    logger.info("\n--- Scenario 1: Enrolling User 'Alice' and 'Bob' ---")
    # Simulate audio samples. In a real scenario, these would be actual recordings.
    # We add small variations to simulate real-world conditions.
    alice_voice_base = np.random.randn(sr * 3)
    alice_samples = [alice_voice_base + np.random.randn(sr * 3) * 0.05 for _ in range(3)]
    
    bob_voice_base = np.random.randn(sr * 3)
    bob_samples = [bob_voice_base + np.random.randn(sr * 3) * 0.05 for _ in range(3)]

    await engine.enroll_user("alice", alice_samples, sr)
    await engine.enroll_user("bob", bob_samples, sr)

    # 2. Verify Alice with her own voice
    logger.info("\n--- Scenario 2: Verifying Alice with her own voice ---")
    alice_test_sample = alice_voice_base + np.random.randn(sr * 3) * 0.05
    verification_result = await engine.verify_user("alice", alice_test_sample, sr)
    print(f"Verification result for Alice: {verification_result.is_verified}, Confidence: {verification_result.confidence:.4f}")

    # 3. Try to verify Alice with Bob's voice
    logger.info("\n--- Scenario 3: Verifying Alice with Bob's voice (should fail) ---")
    bob_test_sample = bob_voice_base + np.random.randn(sr * 3) * 0.05
    verification_result_fail = await engine.verify_user("alice", bob_test_sample, sr)
    print(f"Verification result for Alice (with Bob's voice): {verification_result_fail.is_verified}, Confidence: {verification_result_fail.confidence:.4f}")

    # 4. Identify an unknown speaker (who is Alice)
    logger.info("\n--- Scenario 4: Identifying an unknown speaker (Alice) ---")
    identified_user = await engine.identify_speaker(alice_test_sample, sr)
    print(f"Identified speaker: {identified_user}")


if __name__ == "__main__":
    from datetime import datetime
    # To run this demonstration, you might need to install:
    # pip install numpy librosa
    if not _AUDIO_LIBS_AVAILABLE:
        logger.warning("="*50)
        logger.warning("Running in limited functionality mode.")
        logger.warning("Please run 'pip install numpy librosa' for full features.")
        logger.warning("="*50)
    
    asyncio.run(main())

# Alias for backward compatibility
VoiceBiometrics = VoiceBiometricsEngine
