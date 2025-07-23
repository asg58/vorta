# frontend/components/agi/proactive_assistant.py
"""
VORTA: Proactive Assistant Engine

This module empowers the AGI to be more than just a reactive agent. It analyzes
context and conversation flow to proactively offer suggestions, retrieve relevant
information, and anticipate user needs before they are explicitly stated.
"""

import logging
import asyncio
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
import random
from datetime import datetime

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# --- Data Structures for Proactive Assistance ---

@dataclass
class ProactiveSuggestion:
    """Represents a suggestion offered to the user."""
    suggestion_id: str
    text: str # The text to present to the user, e.g., "Shall I book a table?"
    action: Dict[str, Any] # The action to take if accepted, e.g., {"type": "book_table", "details": {...}}
    confidence: float
    trigger: str # The reason for the suggestion, e.g., "intent_pattern", "context_keyword"

@dataclass
class AssistantContext:
    """The context provided to the assistant to make decisions."""
    conversation_history: List[str] = field(default_factory=list)
    current_intent: Optional[str] = None
    detected_entities: Dict[str, Any] = field(default_factory=dict)
    user_profile: Dict[str, Any] = field(default_factory=dict) # Learned preferences

# --- Suggestion Trigger Rules ---

class BaseTriggerRule:
    """Abstract base class for a rule that triggers a proactive suggestion."""
    async def evaluate(self, context: AssistantContext) -> Optional[ProactiveSuggestion]:
        raise NotImplementedError

class IntentPatternRule(BaseTriggerRule):
    """Triggers suggestions based on a sequence of intents."""
    def __init__(self):
        # Pattern: If user asks for a restaurant, then asks for its location, suggest booking.
        self.pattern = ["query_restaurant", "query_location"]
    
    async def evaluate(self, context: AssistantContext) -> Optional[ProactiveSuggestion]:
        # This is a simplified check. A real system would track intent history properly.
        if "restaurant" in context.detected_entities and "location" in context.detected_entities:
            restaurant_name = context.detected_entities["restaurant"]
            return ProactiveSuggestion(
                suggestion_id=f"book_{restaurant_name}",
                text=f"You're discussing {restaurant_name}. Shall I try to book a table for you?",
                action={"type": "book_table", "entity": restaurant_name},
                confidence=0.8,
                trigger="intent_pattern: query_restaurant -> query_location"
            )
        return None

class KeywordTriggerRule(BaseTriggerRule):
    """Triggers suggestions based on keywords in the conversation."""
    def __init__(self):
        self.triggers = {
            "tomorrow": "Shall I show you your calendar for tomorrow?",
            "bored": "Feeling bored? I can tell you a joke or suggest a movie.",
        }
        self.actions = {
            "tomorrow": {"type": "show_calendar", "date": "tomorrow"},
            "bored": {"type": "suggest_entertainment"},
        }

    async def evaluate(self, context: AssistantContext) -> Optional[ProactiveSuggestion]:
        last_utterance = context.conversation_history[-1].lower() if context.conversation_history else ""
        for keyword, suggestion_text in self.triggers.items():
            if keyword in last_utterance:
                return ProactiveSuggestion(
                    suggestion_id=f"keyword_{keyword}",
                    text=suggestion_text,
                    action=self.actions[keyword],
                    confidence=0.7,
                    trigger=f"keyword: '{keyword}'"
                )
        return None

class UserPreferenceRule(BaseTriggerRule):
    """Triggers suggestions based on learned user preferences."""
    async def evaluate(self, context: AssistantContext) -> Optional[ProactiveSuggestion]:
        # Example: If user frequently asks for news updates in the morning.
        is_morning = 6 <= datetime.now().hour < 12
        prefers_news = context.user_profile.get("prefers_morning_news", False)

        if is_morning and prefers_news:
            return ProactiveSuggestion(
                suggestion_id="morning_news_briefing",
                text="Good morning! Would you like your daily news briefing?",
                action={"type": "fetch_news"},
                confidence=0.9,
                trigger="user_preference: morning_news"
            )
        return None

# --- Proactive Assistant Engine ---

class ProactiveAssistantEngine:
    """
    Orchestrates the rules and logic for generating proactive suggestions.
    """
    def __init__(self):
        self.rules: List[BaseTriggerRule] = [
            IntentPatternRule(),
            KeywordTriggerRule(),
            UserPreferenceRule(),
        ]
        logger.info("ProactiveAssistantEngine initialized with default rules.")

    def register_rule(self, rule: BaseTriggerRule):
        """Adds a new suggestion rule to the engine."""
        self.rules.append(rule)
        logger.info(f"Registered new rule: {rule.__class__.__name__}")

    async def generate_suggestions(self, context: AssistantContext) -> List[ProactiveSuggestion]:
        """
        Evaluates all rules against the current context and returns the best suggestions.
        """
        logger.info("Generating proactive suggestions...")
        
        tasks = [rule.evaluate(context) for rule in self.rules]
        results = await asyncio.gather(*tasks)
        
        # Filter out None results and sort by confidence
        suggestions = [res for res in results if res is not None]
        ranked_suggestions = sorted(suggestions, key=lambda s: s.confidence, reverse=True)
        
        logger.info(f"Generated {len(ranked_suggestions)} potential suggestions.")
        return ranked_suggestions[:3] # Return top 3

# --- Example Usage ---

async def main():
    """Demonstrates the functionality of the ProactiveAssistantEngine."""
    logger.info("--- VORTA Proactive Assistant Engine Demonstration ---")
    
    engine = ProactiveAssistantEngine()

    # Scenario 1: Triggering a suggestion based on keywords
    logger.info("\n--- Scenario 1: Keyword-based suggestion ---")
    context1 = AssistantContext(conversation_history=["I'm so bored today."])
    suggestions1 = await engine.generate_suggestions(context1)
    print("\nSuggestions for 'I'm so bored today.':")
    for sug in suggestions1:
        print(f"  - Suggestion: '{sug.text}' (Confidence: {sug.confidence:.2f})")

    # Scenario 2: Triggering based on intent patterns and entities
    logger.info("\n--- Scenario 2: Intent-pattern-based suggestion ---")
    context2 = AssistantContext(
        conversation_history=["Find a good Italian restaurant nearby.", "What's the address of 'La Trattoria'?"],
        detected_entities={"restaurant": "La Trattoria", "location": "nearby"}
    )
    suggestions2 = await engine.generate_suggestions(context2)
    print("\nSuggestions after discussing a restaurant's location:")
    for sug in suggestions2:
        print(f"  - Suggestion: '{sug.text}' (Action: {sug.action})")

    # Scenario 3: Triggering based on user profile (simulated)
    logger.info("\n--- Scenario 3: User-preference-based suggestion ---")
    # We need to patch datetime for this to be deterministic
    from datetime import datetime as dt
    
    class MockDateTime(dt):
        @classmethod
        def now(cls):
            return dt(2025, 7, 22, 8, 0, 0) # Simulate 8 AM
            
    # Monkey-patching for demonstration purposes
    import datetime
    datetime.datetime = MockDateTime

    context3 = AssistantContext(
        conversation_history=["Good morning VORTA"],
        user_profile={"prefers_morning_news": True}
    )
    suggestions3 = await engine.generate_suggestions(context3)
    print("\nSuggestions for a user who likes morning news:")
    for sug in suggestions3:
        print(f"  - Suggestion: '{sug.text}' (Trigger: {sug.trigger})")


if __name__ == "__main__":
    asyncio.run(main())

# Alias for backward compatibility
ProactiveAssistant = ProactiveAssistantEngine
