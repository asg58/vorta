# frontend/components/agi/adaptive_learning_engine.py
"""
VORTA: Adaptive Learning Engine

This module enables the AGI to learn from user interactions, adapt its behavior,
and personalize the user experience over time. It handles feedback, preference 
learning, and model fine-tuning based on real-world conversations.
"""

import logging
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
import asyncio
import time
import aiofiles

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# --- Data Structures for Learning ---

@dataclass
class UserFeedback:
    """Represents explicit or implicit feedback from a user."""
    conversation_id: str
    turn_id: int
    feedback_type: str  # e.g., "explicit_rating", "correction", "implicit_engagement"
    rating: Optional[float] = None # e.g., 1-5 stars
    correction: Optional[str] = None # e.g., user rephrasing a query
    engagement_metrics: Dict[str, Any] = field(default_factory=dict) # e.g., response_time, follow_up
    timestamp: float = field(default_factory=time.time)

@dataclass
class UserProfile:
    """Stores learned preferences and characteristics of a user."""
    user_id: str
    preferences: Dict[str, Any] = field(default_factory=dict)
    behavioral_patterns: Dict[str, Any] = field(default_factory=dict)
    last_updated: float = field(default_factory=time.time)

    def update_preference(self, key: str, value: Any, weight: float = 0.1):
        """Updates a preference using an exponential moving average."""
        current_value = self.preferences.get(key)
        if isinstance(current_value, (int, float)) and isinstance(value, (int, float)):
            self.preferences[key] = (1 - weight) * current_value + weight * value
        else:
            # For categorical or complex types, just update
            self.preferences[key] = value
        self.last_updated = time.time()
        logger.info(f"Updated preference '{key}' to '{value}' for user {self.user_id}")

# --- Learning Engine Components ---

class ProfileManager:
    """Manages the storage and retrieval of user profiles."""
    def __init__(self, storage_path: str = "./user_profiles"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)
        self.profiles: Dict[str, UserProfile] = {}
        self.lock = asyncio.Lock()
        logger.info(f"ProfileManager initialized. Storage path: {self.storage_path.resolve()}")

    async def get_profile(self, user_id: str) -> UserProfile:
        """Retrieves a user profile, loading from disk if necessary."""
        async with self.lock:
            if user_id in self.profiles:
                return self.profiles[user_id]
            
            profile_file = self.storage_path / f"{user_id}.json"
            if profile_file.exists():
                try:
                    async with aiofiles.open(profile_file, 'r') as f:
                        data = json.loads(await f.read())
                        profile = UserProfile(**data)
                        self.profiles[user_id] = profile
                        logger.info(f"Loaded profile for user {user_id} from disk.")
                        return profile
                except (json.JSONDecodeError, TypeError) as e:
                    logger.error(f"Error loading profile for user {user_id}: {e}. Creating new profile.")
            
            # Create a new profile if none exists
            logger.info(f"No profile found for user {user_id}. Creating a new one.")
            new_profile = UserProfile(user_id=user_id)
            self.profiles[user_id] = new_profile
            return new_profile

    async def save_profile(self, profile: UserProfile):
        """Saves a user profile to disk."""
        async with self.lock:
            profile_file = self.storage_path / f"{profile.user_id}.json"
            try:
                async with aiofiles.open(profile_file, 'w') as f:
                    await f.write(json.dumps(profile.__dict__, indent=4))
                logger.info(f"Successfully saved profile for user {profile.user_id}.")
            except (IOError, TypeError) as e:
                logger.error(f"Failed to save profile for user {profile.user_id}: {e}")

class FeedbackProcessor:
    """Processes user feedback to update user profiles and learning models."""
    def __init__(self, profile_manager: ProfileManager):
        self.profile_manager = profile_manager

    async def process_feedback(self, user_id: str, feedback: UserFeedback):
        """
        Analyzes feedback and triggers learning updates.
        This is the core logic where learning happens.
        """
        logger.info(f"Processing feedback of type '{feedback.feedback_type}' for user {user_id}.")
        profile = await self.profile_manager.get_profile(user_id)

        if feedback.feedback_type == "explicit_rating":
            if feedback.rating is not None:
                # Update a general satisfaction score
                profile.update_preference("satisfaction_score", feedback.rating, weight=0.2)

        elif feedback.feedback_type == "correction":
            if feedback.correction:
                # This is a strong signal for learning.
                # A real system would use this to fine-tune NLP models.
                logger.info(f"Correction received for user {user_id}: '{feedback.correction}'. Logging for model fine-tuning.")
                profile.update_preference("common_corrections", feedback.correction)

        elif feedback.feedback_type == "implicit_engagement":
            # Learn from how the user interacts
            if "response_time_seconds" in feedback.engagement_metrics:
                # Shorter response time might indicate higher engagement
                avg_response_time = feedback.engagement_metrics["response_time_seconds"]
                profile.update_preference("avg_user_response_time", avg_response_time)
            
            if "followed_suggestion" in feedback.engagement_metrics:
                # User followed a proactive suggestion
                new_rate = 1.0 if feedback.engagement_metrics["followed_suggestion"] else 0.0
                profile.update_preference("suggestion_success_rate", new_rate)

        # Save the updated profile
        await self.profile_manager.save_profile(profile)

# --- Main Adaptive Learning Engine ---

class AdaptiveLearningEngine:
    """Orchestrates the adaptive learning process for the AGI."""
    def __init__(self):
        self.profile_manager = ProfileManager()
        self.feedback_processor = FeedbackProcessor(self.profile_manager)
        self.feedback_queue = asyncio.Queue()
        self.worker_task = asyncio.create_task(self._feedback_worker())
        logger.info("AdaptiveLearningEngine initialized and worker started.")

    async def _feedback_worker(self):
        """A background worker that processes feedback from a queue."""
        logger.info("Feedback worker started. Waiting for feedback...")
        while True:
            try:
                user_id, feedback = await self.feedback_queue.get()
                await self.feedback_processor.process_feedback(user_id, feedback)
                self.feedback_queue.task_done()
            except Exception as e:
                logger.error(f"Error in feedback worker: {e}", exc_info=True)

    async def log_feedback(self, user_id: str, feedback: UserFeedback):
        """Public method to add feedback to the processing queue."""
        await self.feedback_queue.put((user_id, feedback))
        logger.info(f"Queued feedback of type '{feedback.feedback_type}' for user {user_id}.")

    async def get_user_profile(self, user_id: str) -> UserProfile:
        """Retrieves the most up-to-date profile for a user."""
        return await self.profile_manager.get_profile(user_id)

    async def stop(self):
        """Gracefully stops the feedback worker."""
        logger.info("Stopping AdaptiveLearningEngine...")
        await self.feedback_queue.join() # Wait for all items to be processed
        self.worker_task.cancel()
        try:
            await self.worker_task
        except asyncio.CancelledError:
            logger.info("Feedback worker is shutting down.")
            raise

# --- Example Usage ---

async def main():
    """Demonstrates the functionality of the AdaptiveLearningEngine."""
    logger.info("--- VORTA Adaptive Learning Engine Demonstration ---")
    
    engine = AdaptiveLearningEngine()
    user_id = "user_1234"

    # 1. Simulate some interactions and log feedback
    logger.info("\n--- Scenario 1: Logging various types of feedback ---")
    
    # Explicit positive feedback
    feedback1 = UserFeedback(conversation_id="conv_abc", turn_id=3, feedback_type="explicit_rating", rating=5.0)
    await engine.log_feedback(user_id, feedback1)

    # A user correction
    feedback2 = UserFeedback(conversation_id="conv_abc", turn_id=5, feedback_type="correction", correction="I meant 'weather in London', not 'weather in Luton'")
    await engine.log_feedback(user_id, feedback2)

    # Implicit feedback from engagement
    feedback3 = UserFeedback(
        conversation_id="conv_xyz", turn_id=2, feedback_type="implicit_engagement",
        engagement_metrics={"response_time_seconds": 1.2, "followed_suggestion": True}
    )
    await engine.log_feedback(user_id, feedback3)

    # Wait for the queue to be processed
    await asyncio.sleep(0.1)

    # 2. Retrieve the updated profile to see the learned preferences
    logger.info("\n--- Scenario 2: Retrieving the adapted user profile ---")
    profile = await engine.get_user_profile(user_id)
    
    print(f"\nUpdated Profile for User: {profile.user_id}")
    print(f"  - Last Updated: {time.ctime(profile.last_updated)}")
    print("  - Learned Preferences:")
    for key, value in profile.preferences.items():
        if isinstance(value, float):
            print(f"    - {key}: {value:.2f}")
        else:
            print(f"    - {key}: {value}")

    # 3. Stop the engine
    await engine.stop()
    logger.info("\nDemonstration complete.")


if __name__ == "__main__":
    asyncio.run(main())
