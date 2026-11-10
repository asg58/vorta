"""
Conversation engine for the live voice pipeline.

Wraps the shared Claude brain in the interface the pipeline expects:
`initialize()`, `is_initialized` and `process_conversation(...)`.
"""

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from shared.ai.claude_brain import ClaudeBrain
except ImportError:
    # Running from a source checkout: `shared/` lives at the repository root.
    sys.path.append(str(Path(__file__).resolve().parents[5]))
    from shared.ai.claude_brain import ClaudeBrain

logger = logging.getLogger(__name__)


class VortaInferenceEngine:
    """Generates the assistant's reply for each voice turn."""

    def __init__(self, settings: Any = None, brain: Optional[ClaudeBrain] = None):
        self.settings = settings
        self.brain = brain or ClaudeBrain()
        self.is_initialized = False

    async def initialize(self):
        self.is_initialized = True
        logger.info(f"Conversation engine ready (model: {self.brain.model})")

    async def process_conversation(
        self,
        messages: List[Dict[str, Any]],
        user_id: Optional[str] = None,
        conversation_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Return {"response": <text to speak>, "metadata": {...}}."""
        reply = await self.brain.reply(messages)
        return {"response": reply.text, "metadata": reply.metadata()}
