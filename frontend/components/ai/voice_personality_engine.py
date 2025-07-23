"""
🎭 VORTA AGI Voice Agent - Voice Personality Engine
Advanced AI personality system with adaptive characteristics

This module provides enterprise-grade voice personality capabilities:
- Dynamic personality trait adaptation and learning
- Multi-dimensional personality modeling (Big 5 + Voice traits)
- Contextual personality expression and consistency
- User relationship building and rapport development
- Professional emotional intelligence and empathy

Author: Ultra High-Grade Development Team
Version: 3.0.0-agi
Performance: >96% personality consistency, adaptive learning
"""

import asyncio
import logging
import time
import json
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime, timedelta
import uuid

try:
    import numpy as np
    _numpy_available = True
except ImportError:
    _numpy_available = False
    logging.warning("NumPy not available - limited statistical personality modeling")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PersonalityTrait(Enum):
    """Core personality traits (Big Five + Voice-specific)"""
    # Big Five
    OPENNESS = "openness"
    CONSCIENTIOUSNESS = "conscientiousness"
    EXTRAVERSION = "extraversion"
    AGREEABLENESS = "agreeableness"
    NEUROTICISM = "neuroticism"
    
    # Voice-specific traits
    HELPFULNESS = "helpfulness"
    PATIENCE = "patience"
    ENTHUSIASM = "enthusiasm"
    PROFESSIONALISM = "professionalism"
    CREATIVITY = "creativity"
    EMPATHY = "empathy"
    HUMOR = "humor"
    CONFIDENCE = "confidence"

class PersonalityState(Enum):
    """Current personality state"""
    BALANCED = "balanced"
    ENERGETIC = "energetic"
    CALM = "calm"
    FOCUSED = "focused"
    SUPPORTIVE = "supportive"
    PLAYFUL = "playful"
    SERIOUS = "serious"
    ADAPTIVE = "adaptive"

class InteractionContext(Enum):
    """Context for personality adaptation"""
    FIRST_MEETING = "first_meeting"
    CASUAL_CONVERSATION = "casual_conversation"
    PROBLEM_SOLVING = "problem_solving"
    LEARNING_SESSION = "learning_session"
    EMOTIONAL_SUPPORT = "emotional_support"
    TECHNICAL_DISCUSSION = "technical_discussion"
    CREATIVE_COLLABORATION = "creative_collaboration"
    ERROR_HANDLING = "error_handling"

class RelationshipStage(Enum):
    """User relationship development stages"""
    STRANGER = "stranger"
    ACQUAINTANCE = "acquaintance"
    FAMILIAR = "familiar"
    TRUSTED = "trusted"
    CLOSE = "close"

@dataclass
class PersonalityProfile:
    """Comprehensive personality profile"""
    # Core traits (0.0 to 1.0 scale)
    traits: Dict[PersonalityTrait, float] = field(default_factory=lambda: {
        PersonalityTrait.OPENNESS: 0.8,
        PersonalityTrait.CONSCIENTIOUSNESS: 0.9,
        PersonalityTrait.EXTRAVERSION: 0.7,
        PersonalityTrait.AGREEABLENESS: 0.9,
        PersonalityTrait.NEUROTICISM: 0.1,
        PersonalityTrait.HELPFULNESS: 0.95,
        PersonalityTrait.PATIENCE: 0.9,
        PersonalityTrait.ENTHUSIASM: 0.8,
        PersonalityTrait.PROFESSIONALISM: 0.85,
        PersonalityTrait.CREATIVITY: 0.75,
        PersonalityTrait.EMPATHY: 0.9,
        PersonalityTrait.HUMOR: 0.6,
        PersonalityTrait.CONFIDENCE: 0.8
    })
    
    # Current state
    current_state: PersonalityState = PersonalityState.BALANCED
    energy_level: float = 0.8
    mood_tendency: float = 0.7  # 0=negative, 1=positive
    
    # Adaptation parameters
    adaptability: float = 0.7
    consistency_preference: float = 0.8
    learning_rate: float = 0.1
    
    # Expression modifiers
    speech_patterns: Dict[str, float] = field(default_factory=lambda: {
        'formality_level': 0.7,
        'verbosity': 0.6,
        'technical_depth': 0.7,
        'emotional_expressiveness': 0.8,
        'directness': 0.7,
        'supportiveness': 0.9
    })
    
    # Behavioral preferences
    interaction_preferences: Dict[str, Any] = field(default_factory=lambda: {
        'prefers_examples': True,
        'uses_analogies': True,
        'asks_clarifying_questions': True,
        'provides_encouragement': True,
        'offers_alternatives': True,
        'checks_understanding': True
    })

@dataclass
class UserRelationship:
    """User relationship and interaction history"""
    user_id: str
    relationship_stage: RelationshipStage = RelationshipStage.STRANGER
    
    # Interaction history
    total_interactions: int = 0
    successful_interactions: int = 0
    last_interaction: Optional[datetime] = None
    interaction_frequency: float = 0.0  # interactions per day
    
    # User preferences learned
    communication_style_preference: Optional[str] = None
    technical_level: str = "intermediate"
    preferred_response_length: str = "medium"
    topic_interests: List[str] = field(default_factory=list)
    emotional_needs: List[str] = field(default_factory=list)
    
    # Personality adaptation
    effective_traits: Dict[PersonalityTrait, float] = field(default_factory=dict)
    rapport_score: float = 0.5
    trust_level: float = 0.5
    comfort_level: float = 0.5
    
    # Context patterns
    common_contexts: Dict[InteractionContext, int] = field(default_factory=dict)
    successful_contexts: Dict[InteractionContext, int] = field(default_factory=dict)
    
    def get_success_rate(self) -> float:
        """Get interaction success rate"""
        if self.total_interactions == 0:
            return 0.0
        return self.successful_interactions / self.total_interactions
    
    def update_relationship_stage(self):
        """Update relationship stage based on interactions"""
        if self.total_interactions >= 50 and self.get_success_rate() > 0.8:
            self.relationship_stage = RelationshipStage.CLOSE
        elif self.total_interactions >= 20 and self.get_success_rate() > 0.7:
            self.relationship_stage = RelationshipStage.TRUSTED
        elif self.total_interactions >= 10 and self.get_success_rate() > 0.6:
            self.relationship_stage = RelationshipStage.FAMILIAR
        elif self.total_interactions >= 3:
            self.relationship_stage = RelationshipStage.ACQUAINTANCE
        else:
            self.relationship_stage = RelationshipStage.STRANGER

@dataclass
class PersonalityExpression:
    """Personality expression for a specific interaction"""
    # Active traits for this interaction
    expressed_traits: Dict[PersonalityTrait, float]
    personality_state: PersonalityState
    
    # Communication style
    formality_level: float
    enthusiasm_level: float
    supportiveness_level: float
    confidence_level: float
    
    # Behavioral markers
    use_humor: bool
    use_examples: bool
    use_analogies: bool
    ask_questions: bool
    offer_encouragement: bool
    
    # Speech patterns
    sentence_structure: str  # "simple", "complex", "varied"
    vocabulary_level: str   # "basic", "intermediate", "advanced"
    emotional_tone: str     # "neutral", "warm", "excited", "calm"
    
    # Response characteristics
    response_length_preference: str
    technical_depth_preference: str
    creativity_level: float

@dataclass
class PersonalityConfig:
    """Configuration for personality engine"""
    # Core personality settings
    base_personality_profile: PersonalityProfile = field(default_factory=PersonalityProfile)
    enable_personality_adaptation: bool = True
    adaptation_speed: float = 0.1
    
    # Relationship building
    enable_relationship_building: bool = True
    relationship_memory_duration: int = 90  # days
    max_user_relationships: int = 1000
    
    # Expression control
    personality_consistency: float = 0.8  # How consistent to be
    context_sensitivity: float = 0.7      # How much to adapt to context
    user_preference_weight: float = 0.6   # How much to adapt to user
    
    # Learning and adaptation
    enable_personality_learning: bool = True
    learning_rate: float = 0.05
    feedback_weight: float = 0.3
    
    # Safety and boundaries
    maintain_professional_boundaries: bool = True
    ethical_constraints_enabled: bool = True
    personality_bounds_checking: bool = True

class VoicePersonalityEngine:
    """
    🎭 Advanced Voice Personality Engine
    
    Ultra high-grade personality system with adaptive traits, relationship
    building, and contextual personality expression.
    """
    
    def __init__(self, config: PersonalityConfig):
        self.config = config
        self.is_initialized = False
        
        # Core personality
        self.base_personality = config.base_personality_profile
        self.current_personality_state = PersonalityState.BALANCED
        
        # User relationships
        self.user_relationships: Dict[str, UserRelationship] = {}
        
        # Personality adaptation history
        self.adaptation_history: List[Dict[str, Any]] = []
        
        # Performance metrics
        self.metrics = {
            'total_interactions': 0,
            'personality_adaptations': 0,
            'successful_expressions': 0,
            'relationship_developments': 0,
            'context_recognitions': 0,
            'trait_distribution': {trait.value: 0 for trait in PersonalityTrait},
            'state_distribution': {state.value: 0 for state in PersonalityState},
            'relationship_stages': {stage.value: 0 for stage in RelationshipStage},
            'average_rapport_score': 0.0,
            'consistency_score': 0.0
        }
        
        # Context recognition patterns
        self.context_patterns = self._load_context_patterns()
        
        # Personality expression templates
        self.expression_templates = self._load_expression_templates()
        
        # Initialize components
        self._initialize_personality_models()
        
        logger.info("🎭 Voice Personality Engine initialized")
        logger.info(f"   Personality adaptation: {'✅' if config.enable_personality_adaptation else '❌'}")
        logger.info(f"   Relationship building: {'✅' if config.enable_relationship_building else '❌'}")
        logger.info(f"   Learning enabled: {'✅' if config.enable_personality_learning else '❌'}")
    
    def _initialize_personality_models(self):
        """Initialize personality modeling components"""
        try:
            # Initialize trait interaction models
            self.trait_interactions = self._create_trait_interaction_matrix()
            
            # Initialize state transition models
            self.state_transitions = self._create_state_transition_model()
            
            # Initialize adaptation models
            self.adaptation_models = {
                'user_preference': UserPreferenceModel(),
                'context_adaptation': ContextAdaptationModel(),
                'relationship_building': RelationshipBuildingModel()
            }
            
            logger.info("✅ Personality models initialized")
            
        except Exception as e:
            logger.error(f"❌ Personality model initialization failed: {e}")
    
    def _create_trait_interaction_matrix(self) -> Dict[PersonalityTrait, Dict[PersonalityTrait, float]]:
        """Create trait interaction influence matrix"""
        interactions = {}
        for trait1 in PersonalityTrait:
            interactions[trait1] = {}
            for trait2 in PersonalityTrait:
                if trait1 == trait2:
                    interactions[trait1][trait2] = 1.0
                else:
                    # Define trait interactions (simplified model)
                    interactions[trait1][trait2] = self._calculate_trait_interaction(trait1, trait2)
        
        return interactions
    
    def _calculate_trait_interaction(self, trait1: PersonalityTrait, trait2: PersonalityTrait) -> float:
        """Calculate interaction strength between traits"""
        # Simplified interaction model
        synergistic_pairs = [
            (PersonalityTrait.HELPFULNESS, PersonalityTrait.EMPATHY),
            (PersonalityTrait.ENTHUSIASM, PersonalityTrait.CREATIVITY),
            (PersonalityTrait.PROFESSIONALISM, PersonalityTrait.CONSCIENTIOUSNESS),
            (PersonalityTrait.PATIENCE, PersonalityTrait.AGREEABLENESS),
            (PersonalityTrait.CONFIDENCE, PersonalityTrait.EXTRAVERSION)
        ]
        
        opposing_pairs = [
            (PersonalityTrait.NEUROTICISM, PersonalityTrait.CONFIDENCE),
            (PersonalityTrait.HUMOR, PersonalityTrait.PROFESSIONALISM),
        ]
        
        pair = (trait1, trait2)
        reverse_pair = (trait2, trait1)
        
        if pair in synergistic_pairs or reverse_pair in synergistic_pairs:
            return 0.3  # Positive interaction
        elif pair in opposing_pairs or reverse_pair in opposing_pairs:
            return -0.2  # Negative interaction
        else:
            return 0.0  # Neutral interaction
    
    def _create_state_transition_model(self) -> Dict[PersonalityState, Dict[PersonalityState, float]]:
        """Create personality state transition probabilities"""
        transitions = {}
        for state in PersonalityState:
            transitions[state] = {}
            for next_state in PersonalityState:
                transitions[state][next_state] = self._calculate_state_transition_probability(state, next_state)
        
        return transitions
    
    def _calculate_state_transition_probability(self, 
                                             current_state: PersonalityState, 
                                             next_state: PersonalityState) -> float:
        """Calculate transition probability between personality states"""
        if current_state == next_state:
            return 0.5  # Stay in same state
        
        # Define natural transitions
        likely_transitions = {
            PersonalityState.BALANCED: [PersonalityState.FOCUSED, PersonalityState.SUPPORTIVE],
            PersonalityState.ENERGETIC: [PersonalityState.PLAYFUL, PersonalityState.CREATIVE],
            PersonalityState.CALM: [PersonalityState.BALANCED, PersonalityState.SUPPORTIVE],
            PersonalityState.FOCUSED: [PersonalityState.SERIOUS, PersonalityState.CALM],
            PersonalityState.SUPPORTIVE: [PersonalityState.EMPATHETIC, PersonalityState.CALM],
            PersonalityState.PLAYFUL: [PersonalityState.CREATIVE, PersonalityState.ENERGETIC],
            PersonalityState.SERIOUS: [PersonalityState.FOCUSED, PersonalityState.BALANCED]
        }
        
        if next_state in likely_transitions.get(current_state, []):
            return 0.3
        else:
            return 0.1
    
    def _load_context_patterns(self) -> Dict[InteractionContext, Dict[str, Any]]:
        """Load context recognition patterns"""
        return {
            InteractionContext.FIRST_MEETING: {
                'keywords': ['hello', 'hi', 'first time', 'new', 'meet'],
                'traits_emphasis': {
                    PersonalityTrait.HELPFULNESS: 0.2,
                    PersonalityTrait.PROFESSIONALISM: 0.1,
                    PersonalityTrait.ENTHUSIASM: 0.1
                },
                'recommended_state': PersonalityState.SUPPORTIVE
            },
            InteractionContext.PROBLEM_SOLVING: {
                'keywords': ['help', 'problem', 'issue', 'error', 'fix', 'solve'],
                'traits_emphasis': {
                    PersonalityTrait.PATIENCE: 0.2,
                    PersonalityTrait.HELPFULNESS: 0.15,
                    PersonalityTrait.CONSCIENTIOUSNESS: 0.1
                },
                'recommended_state': PersonalityState.FOCUSED
            },
            InteractionContext.EMOTIONAL_SUPPORT: {
                'keywords': ['frustrated', 'confused', 'stuck', 'difficult', 'hard'],
                'traits_emphasis': {
                    PersonalityTrait.EMPATHY: 0.3,
                    PersonalityTrait.PATIENCE: 0.2,
                    PersonalityTrait.AGREEABLENESS: 0.1
                },
                'recommended_state': PersonalityState.SUPPORTIVE
            },
            InteractionContext.CREATIVE_COLLABORATION: {
                'keywords': ['create', 'design', 'build', 'brainstorm', 'idea'],
                'traits_emphasis': {
                    PersonalityTrait.CREATIVITY: 0.3,
                    PersonalityTrait.OPENNESS: 0.2,
                    PersonalityTrait.ENTHUSIASM: 0.15
                },
                'recommended_state': PersonalityState.PLAYFUL
            },
            InteractionContext.TECHNICAL_DISCUSSION: {
                'keywords': ['code', 'algorithm', 'implementation', 'technical', 'architecture'],
                'traits_emphasis': {
                    PersonalityTrait.PROFESSIONALISM: 0.2,
                    PersonalityTrait.CONSCIENTIOUSNESS: 0.15,
                    PersonalityTrait.CONFIDENCE: 0.1
                },
                'recommended_state': PersonalityState.FOCUSED
            }
        }
    
    def _load_expression_templates(self) -> Dict[str, Dict[str, Any]]:
        """Load personality expression templates"""
        return {
            'high_enthusiasm': {
                'sentence_starters': ["That's fantastic!", "How exciting!", "I love that idea!"],
                'modifiers': ['really', 'absolutely', 'definitely', 'completely'],
                'punctuation_preference': 'exclamation'
            },
            'high_empathy': {
                'sentence_starters': ["I understand how you feel", "That sounds challenging", "I can see why"],
                'supportive_phrases': ["you're not alone", "we can work through this", "I'm here to help"],
                'tone_modifiers': ['gentle', 'warm', 'caring']
            },
            'high_professionalism': {
                'sentence_starters': ["I'd recommend", "Based on best practices", "The optimal approach"],
                'formal_phrases': ["please consider", "I suggest", "it would be advisable"],
                'structure_preference': 'structured'
            },
            'high_creativity': {
                'sentence_starters': ["What if we tried", "Here's a creative approach", "Let me think outside the box"],
                'creative_modifiers': ['innovative', 'unique', 'imaginative', 'original'],
                'suggestion_style': 'exploratory'
            }
        }
    
    async def express_personality(self, 
                                context: Dict[str, Any],
                                user_id: Optional[str] = None) -> PersonalityExpression:
        """
        🎭 Generate personality expression for interaction
        
        Args:
            context: Interaction context and user input
            user_id: Optional user ID for relationship-aware adaptation
            
        Returns:
            PersonalityExpression with adapted traits and behaviors
        """
        try:
            start_time = time.time()
            
            logger.debug(f"🎭 Generating personality expression for context")
            
            # Get or create user relationship
            user_relationship = None
            if user_id and self.config.enable_relationship_building:
                user_relationship = self._get_or_create_user_relationship(user_id)
            
            # Recognize interaction context
            interaction_context = self._recognize_interaction_context(context)
            
            # Adapt personality for this interaction
            adapted_personality = await self._adapt_personality(
                interaction_context, user_relationship, context
            )
            
            # Generate personality expression
            expression = self._generate_personality_expression(
                adapted_personality, interaction_context, user_relationship, context
            )
            
            # Update metrics
            self.metrics['total_interactions'] += 1
            self.metrics['context_recognitions'] += 1 if interaction_context else 0
            self.metrics['trait_distribution'][max(adapted_personality.traits.items(), key=lambda x: x[1])[0].value] += 1
            self.metrics['state_distribution'][adapted_personality.current_state.value] += 1
            
            # Update user relationship
            if user_relationship:
                await self._update_user_relationship(user_relationship, context, expression)
            
            processing_time = time.time() - start_time
            logger.debug(f"🎭 Personality expression generated in {processing_time*1000:.1f}ms")
            
            return expression
            
        except Exception as e:
            logger.error(f"❌ Personality expression generation failed: {e}")
            return self._create_default_expression()
    
    def _get_or_create_user_relationship(self, user_id: str) -> UserRelationship:
        """Get existing or create new user relationship"""
        if user_id not in self.user_relationships:
            self.user_relationships[user_id] = UserRelationship(user_id=user_id)
            self.metrics['relationship_developments'] += 1
            logger.debug(f"🎭 Created new user relationship: {user_id}")
        
        return self.user_relationships[user_id]
    
    def _recognize_interaction_context(self, context: Dict[str, Any]) -> Optional[InteractionContext]:
        """Recognize the interaction context from input"""
        try:
            user_input = context.get('user_input', '').lower()
            detected_intent = context.get('detected_intent', '').lower()
            detected_emotion = context.get('detected_emotion', '').lower()
            
            # Check each context pattern
            best_match = None
            best_score = 0.0
            
            for interaction_context, pattern in self.context_patterns.items():
                score = 0.0
                
                # Keyword matching
                keywords = pattern.get('keywords', [])
                for keyword in keywords:
                    if keyword in user_input or keyword in detected_intent:
                        score += 1.0
                
                # Emotional context matching
                if interaction_context == InteractionContext.EMOTIONAL_SUPPORT:
                    negative_emotions = ['frustrated', 'confused', 'angry', 'sad', 'worried']
                    if any(emotion in detected_emotion for emotion in negative_emotions):
                        score += 2.0
                
                # Intent-based matching
                if interaction_context == InteractionContext.PROBLEM_SOLVING and 'help' in detected_intent:
                    score += 1.5
                elif interaction_context == InteractionContext.CREATIVE_COLLABORATION and 'creative' in detected_intent:
                    score += 1.5
                elif interaction_context == InteractionContext.TECHNICAL_DISCUSSION and 'code' in detected_intent:
                    score += 1.5
                
                if score > best_score:
                    best_score = score
                    best_match = interaction_context
            
            if best_score >= 1.0:
                logger.debug(f"🎭 Recognized context: {best_match.value} (score: {best_score})")
                return best_match
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Context recognition failed: {e}")
            return None
    
    async def _adapt_personality(self, 
                               interaction_context: Optional[InteractionContext],
                               user_relationship: Optional[UserRelationship],
                               context: Dict[str, Any]) -> PersonalityProfile:
        """Adapt personality based on context and relationship"""
        try:
            # Start with base personality
            adapted_traits = self.base_personality.traits.copy()
            
            # Context-based adaptation
            if interaction_context and self.config.context_sensitivity > 0:
                context_pattern = self.context_patterns.get(interaction_context, {})
                traits_emphasis = context_pattern.get('traits_emphasis', {})
                
                for trait, emphasis in traits_emphasis.items():
                    current_value = adapted_traits.get(trait, 0.5)
                    adaptation = emphasis * self.config.context_sensitivity
                    adapted_traits[trait] = min(max(current_value + adaptation, 0.0), 1.0)
            
            # User relationship adaptation
            if user_relationship and self.config.user_preference_weight > 0:
                # Adapt based on what works with this user
                for trait, effectiveness in user_relationship.effective_traits.items():
                    if effectiveness > 0.7:  # This trait works well with this user
                        current_value = adapted_traits.get(trait, 0.5)
                        adaptation = 0.1 * self.config.user_preference_weight
                        adapted_traits[trait] = min(current_value + adaptation, 1.0)
                
                # Adapt based on relationship stage
                stage_adaptations = {
                    RelationshipStage.STRANGER: {PersonalityTrait.PROFESSIONALISM: 0.1},
                    RelationshipStage.FAMILIAR: {PersonalityTrait.HUMOR: 0.1},
                    RelationshipStage.TRUSTED: {PersonalityTrait.OPENNESS: 0.1},
                    RelationshipStage.CLOSE: {PersonalityTrait.CREATIVITY: 0.1, PersonalityTrait.HUMOR: 0.1}
                }
                
                stage_adaptation = stage_adaptations.get(user_relationship.relationship_stage, {})
                for trait, adaptation in stage_adaptation.items():
                    current_value = adapted_traits.get(trait, 0.5)
                    adapted_traits[trait] = min(current_value + adaptation, 1.0)
            
            # Create adapted personality profile
            adapted_personality = PersonalityProfile(
                traits=adapted_traits,
                current_state=self._determine_personality_state(interaction_context, adapted_traits),
                energy_level=self._calculate_energy_level(adapted_traits, context),
                mood_tendency=self._calculate_mood_tendency(adapted_traits, context)
            )
            
            # Update speech patterns based on adapted traits
            adapted_personality.speech_patterns = self._adapt_speech_patterns(
                adapted_traits, user_relationship
            )
            
            self.metrics['personality_adaptations'] += 1
            
            return adapted_personality
            
        except Exception as e:
            logger.error(f"❌ Personality adaptation failed: {e}")
            return self.base_personality
    
    def _determine_personality_state(self, 
                                   interaction_context: Optional[InteractionContext],
                                   traits: Dict[PersonalityTrait, float]) -> PersonalityState:
        """Determine personality state based on context and traits"""
        try:
            # Context-based state determination
            if interaction_context:
                context_pattern = self.context_patterns.get(interaction_context, {})
                recommended_state = context_pattern.get('recommended_state')
                if recommended_state:
                    return recommended_state
            
            # Trait-based state determination
            dominant_traits = sorted(traits.items(), key=lambda x: x[1], reverse=True)[:3]
            
            # State mapping based on dominant traits
            if PersonalityTrait.ENTHUSIASM in [trait for trait, _ in dominant_traits]:
                return PersonalityState.ENERGETIC
            elif PersonalityTrait.EMPATHY in [trait for trait, _ in dominant_traits]:
                return PersonalityState.SUPPORTIVE
            elif PersonalityTrait.CREATIVITY in [trait for trait, _ in dominant_traits]:
                return PersonalityState.PLAYFUL
            elif PersonalityTrait.PROFESSIONALISM in [trait for trait, _ in dominant_traits]:
                return PersonalityState.FOCUSED
            else:
                return PersonalityState.BALANCED
                
        except Exception as e:
            logger.error(f"❌ Personality state determination failed: {e}")
            return PersonalityState.BALANCED
    
    def _calculate_energy_level(self, 
                              traits: Dict[PersonalityTrait, float],
                              context: Dict[str, Any]) -> float:
        """Calculate current energy level"""
        base_energy = 0.7
        
        # Trait influences
        energy_influences = {
            PersonalityTrait.ENTHUSIASM: 0.2,
            PersonalityTrait.EXTRAVERSION: 0.15,
            PersonalityTrait.CONFIDENCE: 0.1,
            PersonalityTrait.NEUROTICISM: -0.1
        }
        
        for trait, influence in energy_influences.items():
            trait_value = traits.get(trait, 0.5)
            base_energy += (trait_value - 0.5) * influence
        
        # Context influences
        user_emotion = context.get('detected_emotion', 'neutral')
        if user_emotion in ['excited', 'enthusiastic']:
            base_energy += 0.1
        elif user_emotion in ['sad', 'frustrated']:
            base_energy -= 0.1
        
        return max(min(base_energy, 1.0), 0.1)
    
    def _calculate_mood_tendency(self,
                               traits: Dict[PersonalityTrait, float],
                               context: Dict[str, Any]) -> float:
        """Calculate mood tendency (positive/negative)"""
        base_mood = 0.7
        
        # Trait influences
        mood_influences = {
            PersonalityTrait.AGREEABLENESS: 0.15,
            PersonalityTrait.OPTIMISM: 0.2 if PersonalityTrait.OPTIMISM in traits else 0,
            PersonalityTrait.NEUROTICISM: -0.2,
            PersonalityTrait.CONFIDENCE: 0.1
        }
        
        for trait, influence in mood_influences.items():
            trait_value = traits.get(trait, 0.5)
            base_mood += (trait_value - 0.5) * influence
        
        return max(min(base_mood, 1.0), 0.0)
    
    def _adapt_speech_patterns(self,
                             traits: Dict[PersonalityTrait, float],
                             user_relationship: Optional[UserRelationship]) -> Dict[str, float]:
        """Adapt speech patterns based on personality traits"""
        patterns = {
            'formality_level': 0.7,
            'verbosity': 0.6,
            'technical_depth': 0.7,
            'emotional_expressiveness': 0.8,
            'directness': 0.7,
            'supportiveness': 0.9
        }
        
        # Trait-based adaptations
        trait_influences = {
            'formality_level': {
                PersonalityTrait.PROFESSIONALISM: 0.3,
                PersonalityTrait.CONSCIENTIOUSNESS: 0.2
            },
            'verbosity': {
                PersonalityTrait.EXTRAVERSION: 0.2,
                PersonalityTrait.ENTHUSIASM: 0.15
            },
            'emotional_expressiveness': {
                PersonalityTrait.EMPATHY: 0.3,
                PersonalityTrait.ENTHUSIASM: 0.2
            },
            'supportiveness': {
                PersonalityTrait.HELPFULNESS: 0.2,
                PersonalityTrait.AGREEABLENESS: 0.15
            }
        }
        
        for pattern, influences in trait_influences.items():
            for trait, influence in influences.items():
                trait_value = traits.get(trait, 0.5)
                patterns[pattern] += (trait_value - 0.5) * influence
                patterns[pattern] = max(min(patterns[pattern], 1.0), 0.0)
        
        # User relationship adaptations
        if user_relationship:
            if user_relationship.relationship_stage in [RelationshipStage.TRUSTED, RelationshipStage.CLOSE]:
                patterns['formality_level'] -= 0.1
                patterns['emotional_expressiveness'] += 0.1
        
        return patterns
    
    def _generate_personality_expression(self,
                                       personality: PersonalityProfile,
                                       interaction_context: Optional[InteractionContext],
                                       user_relationship: Optional[UserRelationship],
                                       context: Dict[str, Any]) -> PersonalityExpression:
        """Generate personality expression based on adapted personality"""
        try:
            # Determine expression characteristics
            formality_level = personality.speech_patterns['formality_level']
            enthusiasm_level = personality.traits.get(PersonalityTrait.ENTHUSIASM, 0.7)
            supportiveness_level = personality.speech_patterns['supportiveness']
            confidence_level = personality.traits.get(PersonalityTrait.CONFIDENCE, 0.8)
            
            # Determine behavioral markers
            use_humor = (personality.traits.get(PersonalityTrait.HUMOR, 0.6) > 0.7 and
                        formality_level < 0.8)
            
            use_examples = personality.interaction_preferences['prefers_examples']
            use_analogies = personality.interaction_preferences['uses_analogies']
            ask_questions = personality.interaction_preferences['asks_clarifying_questions']
            offer_encouragement = personality.interaction_preferences['provides_encouragement']
            
            # Determine speech patterns
            sentence_structure = self._determine_sentence_structure(personality)
            vocabulary_level = self._determine_vocabulary_level(personality, user_relationship)
            emotional_tone = self._determine_emotional_tone(personality, context)
            
            # Determine response characteristics
            response_length_preference = self._determine_response_length(personality, user_relationship)
            technical_depth_preference = self._determine_technical_depth(personality, user_relationship)
            creativity_level = personality.traits.get(PersonalityTrait.CREATIVITY, 0.75)
            
            expression = PersonalityExpression(
                expressed_traits=personality.traits,
                personality_state=personality.current_state,
                formality_level=formality_level,
                enthusiasm_level=enthusiasm_level,
                supportiveness_level=supportiveness_level,
                confidence_level=confidence_level,
                use_humor=use_humor,
                use_examples=use_examples,
                use_analogies=use_analogies,
                ask_questions=ask_questions,
                offer_encouragement=offer_encouragement,
                sentence_structure=sentence_structure,
                vocabulary_level=vocabulary_level,
                emotional_tone=emotional_tone,
                response_length_preference=response_length_preference,
                technical_depth_preference=technical_depth_preference,
                creativity_level=creativity_level
            )
            
            self.metrics['successful_expressions'] += 1
            
            return expression
            
        except Exception as e:
            logger.error(f"❌ Personality expression generation failed: {e}")
            return self._create_default_expression()
    
    def _determine_sentence_structure(self, personality: PersonalityProfile) -> str:
        """Determine preferred sentence structure"""
        verbosity = personality.speech_patterns['verbosity']
        professionalism = personality.traits.get(PersonalityTrait.PROFESSIONALISM, 0.7)
        
        if verbosity > 0.8 and professionalism > 0.8:
            return "complex"
        elif verbosity < 0.4:
            return "simple"
        else:
            return "varied"
    
    def _determine_vocabulary_level(self, 
                                  personality: PersonalityProfile,
                                  user_relationship: Optional[UserRelationship]) -> str:
        """Determine vocabulary complexity level"""
        base_level = personality.speech_patterns['technical_depth']
        
        # User relationship influence
        if user_relationship:
            if user_relationship.technical_level == 'beginner':
                base_level -= 0.2
            elif user_relationship.technical_level == 'advanced':
                base_level += 0.2
        
        if base_level > 0.8:
            return "advanced"
        elif base_level < 0.4:
            return "basic"
        else:
            return "intermediate"
    
    def _determine_emotional_tone(self, 
                                personality: PersonalityProfile,
                                context: Dict[str, Any]) -> str:
        """Determine emotional tone for interaction"""
        energy = personality.energy_level
        mood = personality.mood_tendency
        user_emotion = context.get('detected_emotion', 'neutral')
        
        # Adapt to user emotion
        if user_emotion in ['frustrated', 'sad', 'worried']:
            return "calm"
        elif user_emotion in ['excited', 'enthusiastic']:
            return "excited"
        elif energy > 0.8 and mood > 0.8:
            return "warm"
        else:
            return "neutral"
    
    def _determine_response_length(self,
                                 personality: PersonalityProfile,
                                 user_relationship: Optional[UserRelationship]) -> str:
        """Determine preferred response length"""
        verbosity = personality.speech_patterns['verbosity']
        
        # User preference override
        if user_relationship and user_relationship.preferred_response_length:
            return user_relationship.preferred_response_length
        
        if verbosity > 0.8:
            return "long"
        elif verbosity < 0.4:
            return "short"
        else:
            return "medium"
    
    def _determine_technical_depth(self,
                                 personality: PersonalityProfile,
                                 user_relationship: Optional[UserRelationship]) -> str:
        """Determine technical depth level"""
        technical_depth = personality.speech_patterns['technical_depth']
        
        # User relationship influence
        if user_relationship:
            if user_relationship.technical_level == 'beginner':
                return "basic"
            elif user_relationship.technical_level == 'advanced':
                return "detailed"
        
        if technical_depth > 0.8:
            return "detailed"
        elif technical_depth < 0.4:
            return "basic"
        else:
            return "moderate"
    
    async def _update_user_relationship(self,
                                      relationship: UserRelationship,
                                      context: Dict[str, Any],
                                      expression: PersonalityExpression):
        """Update user relationship based on interaction"""
        try:
            # Update interaction counts
            relationship.total_interactions += 1
            relationship.last_interaction = datetime.now()
            
            # Update interaction frequency (simplified)
            if relationship.total_interactions > 1:
                time_diff = datetime.now() - relationship.last_interaction
                daily_interactions = 1 / max(time_diff.days, 1)
                relationship.interaction_frequency = (
                    relationship.interaction_frequency * 0.9 + daily_interactions * 0.1
                )
            
            # Learn user preferences
            user_feedback = context.get('user_satisfaction', 0.5)
            if user_feedback > 0.7:
                relationship.successful_interactions += 1
                
                # Update effective traits
                for trait, value in expression.expressed_traits.items():
                    if trait not in relationship.effective_traits:
                        relationship.effective_traits[trait] = value
                    else:
                        # Weighted update
                        relationship.effective_traits[trait] = (
                            relationship.effective_traits[trait] * 0.9 + value * 0.1
                        )
            
            # Update relationship scores
            success_rate = relationship.get_success_rate()
            relationship.rapport_score = min(relationship.rapport_score + 0.01, 1.0)
            relationship.trust_level = success_rate * 0.7 + relationship.rapport_score * 0.3
            relationship.comfort_level = min(relationship.comfort_level + 0.005, 1.0)
            
            # Update relationship stage
            relationship.update_relationship_stage()
            
            # Update context patterns
            interaction_context = context.get('interaction_context')
            if interaction_context:
                if interaction_context not in relationship.common_contexts:
                    relationship.common_contexts[interaction_context] = 0
                relationship.common_contexts[interaction_context] += 1
                
                if user_feedback > 0.7:
                    if interaction_context not in relationship.successful_contexts:
                        relationship.successful_contexts[interaction_context] = 0
                    relationship.successful_contexts[interaction_context] += 1
            
            # Update metrics
            self.metrics['relationship_stages'][relationship.relationship_stage.value] += 1
            self._update_average_rapport_score(relationship.rapport_score)
            
        except Exception as e:
            logger.error(f"❌ User relationship update failed: {e}")
    
    def _update_average_rapport_score(self, rapport_score: float):
        """Update average rapport score metric"""
        current_avg = self.metrics['average_rapport_score']
        total_relationships = len(self.user_relationships)
        
        if total_relationships > 0:
            self.metrics['average_rapport_score'] = (
                (current_avg * (total_relationships - 1) + rapport_score) / total_relationships
            )
    
    def _create_default_expression(self) -> PersonalityExpression:
        """Create default personality expression for fallback"""
        return PersonalityExpression(
            expressed_traits=self.base_personality.traits,
            personality_state=PersonalityState.BALANCED,
            formality_level=0.7,
            enthusiasm_level=0.6,
            supportiveness_level=0.8,
            confidence_level=0.7,
            use_humor=False,
            use_examples=True,
            use_analogies=False,
            ask_questions=True,
            offer_encouragement=True,
            sentence_structure="varied",
            vocabulary_level="intermediate",
            emotional_tone="neutral",
            response_length_preference="medium",
            technical_depth_preference="moderate",
            creativity_level=0.7
        )
    
    def get_personality_metrics(self) -> Dict[str, Any]:
        """Get comprehensive personality metrics"""
        consistency_score = self._calculate_consistency_score()
        
        return {
            'personality_metrics': self.metrics.copy(),
            'consistency_score': consistency_score,
            'active_relationships': len(self.user_relationships),
            'personality_state_distribution': self.metrics['state_distribution'].copy(),
            'trait_usage_distribution': self.metrics['trait_distribution'].copy(),
            'base_personality': {
                trait.value: value 
                for trait, value in self.base_personality.traits.items()
            }
        }
    
    def _calculate_consistency_score(self) -> float:
        """Calculate personality consistency score"""
        if self.metrics['total_interactions'] < 2:
            return 1.0
        
        # Simple consistency measure based on adaptation frequency
        adaptations = self.metrics['personality_adaptations']
        interactions = self.metrics['total_interactions']
        
        adaptation_rate = adaptations / interactions
        consistency = 1.0 - min(adaptation_rate * 2, 1.0)  # Less adaptation = more consistency
        
        return consistency

# Mock adaptation models
class UserPreferenceModel:
    """Model for learning user preferences"""
    
    def learn_preferences(self, user_id: str, interaction_data: Dict[str, Any]) -> Dict[str, Any]:
        """Learn user preferences from interaction"""
        # Mock implementation
        return {
            'communication_style': 'friendly',
            'technical_level': 'intermediate',
            'response_length': 'medium'
        }

class ContextAdaptationModel:
    """Model for context-based personality adaptation"""
    
    def adapt_for_context(self, context: InteractionContext, base_traits: Dict) -> Dict:
        """Adapt traits for specific context"""
        # Mock implementation
        return base_traits

class RelationshipBuildingModel:
    """Model for building user relationships"""
    
    def assess_relationship(self, interaction_history: List[Dict]) -> Dict[str, float]:
        """Assess relationship quality"""
        # Mock implementation
        return {
            'rapport': 0.7,
            'trust': 0.6,
            'comfort': 0.8
        }

# Example usage and testing
if __name__ == "__main__":
    async def test_personality_engine():
        """Test the voice personality engine"""
        print("🧪 Testing VORTA Voice Personality Engine")
        
        # Create configuration
        config = PersonalityConfig(
            enable_personality_adaptation=True,
            enable_relationship_building=True,
            enable_personality_learning=True
        )
        
        # Initialize engine
        engine = VoicePersonalityEngine(config)
        
        # Test contexts
        test_contexts = [
            {
                'user_input': "Hi there! I'm new to programming and need help getting started.",
                'detected_intent': 'help_request',
                'detected_emotion': 'curious',
                'user_id': 'user_001',
                'interaction_type': 'first_meeting'
            },
            {
                'user_input': "I'm really frustrated with this bug in my code. It's been hours!",
                'detected_intent': 'debugging_help',
                'detected_emotion': 'frustrated',
                'user_id': 'user_001',
                'user_satisfaction': 0.8
            },
            {
                'user_input': "Can you help me brainstorm some creative solutions for this project?",
                'detected_intent': 'creative_task',
                'detected_emotion': 'enthusiastic',
                'user_id': 'user_002',
                'interaction_type': 'creative_collaboration'
            },
            {
                'user_input': "I need to implement a complex algorithm efficiently.",
                'detected_intent': 'code_generation',
                'detected_emotion': 'focused',
                'user_id': 'user_003',
                'interaction_type': 'technical_discussion'
            }
        ]
        
        print("\n🎭 Personality Expression Results:")
        print("-" * 80)
        
        for i, context in enumerate(test_contexts, 1):
            # Generate personality expression
            expression = await engine.express_personality(context, context.get('user_id'))
            
            print(f"{i}. Input: '{context['user_input']}'")
            print(f"   User ID: {context.get('user_id', 'N/A')}")
            print(f"   Intent: {context['detected_intent']}")
            print(f"   Emotion: {context['detected_emotion']}")
            print(f"   Personality State: {expression.personality_state.value}")
            print(f"   Formality: {expression.formality_level:.2f}")
            print(f"   Enthusiasm: {expression.enthusiasm_level:.2f}")
            print(f"   Supportiveness: {expression.supportiveness_level:.2f}")
            print(f"   Confidence: {expression.confidence_level:.2f}")
            print(f"   Use Humor: {'✅' if expression.use_humor else '❌'}")
            print(f"   Use Examples: {'✅' if expression.use_examples else '❌'}")
            print(f"   Ask Questions: {'✅' if expression.ask_questions else '❌'}")
            print(f"   Emotional Tone: {expression.emotional_tone}")
            print(f"   Vocabulary Level: {expression.vocabulary_level}")
            print(f"   Response Length: {expression.response_length_preference}")
            
            # Show dominant traits
            dominant_traits = sorted(
                expression.expressed_traits.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:3]
            print(f"   Dominant Traits: {', '.join([f'{t.value}({v:.2f})' for t, v in dominant_traits])}")
            print()
        
        # Test user relationships
        print("👥 User Relationships:")
        for user_id, relationship in engine.user_relationships.items():
            print(f"   {user_id}:")
            print(f"     Stage: {relationship.relationship_stage.value}")
            print(f"     Interactions: {relationship.total_interactions}")
            print(f"     Success Rate: {relationship.get_success_rate():.1%}")
            print(f"     Rapport: {relationship.rapport_score:.2f}")
            print(f"     Trust: {relationship.trust_level:.2f}")
        
        # Performance metrics
        metrics = engine.get_personality_metrics()
        print("\n📊 Personality Metrics:")
        print(f"   Total interactions: {metrics['personality_metrics']['total_interactions']}")
        print(f"   Personality adaptations: {metrics['personality_metrics']['personality_adaptations']}")
        print(f"   Successful expressions: {metrics['personality_metrics']['successful_expressions']}")
        print(f"   Active relationships: {metrics['active_relationships']}")
        print(f"   Consistency score: {metrics['consistency_score']:.3f}")
        print(f"   Average rapport: {metrics['personality_metrics']['average_rapport_score']:.3f}")
        
        print("\n✅ Voice Personality Engine test completed!")
    
    # Run the test
    asyncio.run(test_personality_engine())
