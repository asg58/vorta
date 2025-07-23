# services/vector_store/__init__.py

"""
Vector Intelligence Service for VORTA AGI

This package provides the necessary components for managing long-term memory
and performing semantic searches using a vector database.
"""

from .vector_db_client import VectorDBClient
from .semantic_search import SemanticSearch
from .knowledge_base import KnowledgeBase

__all__ = [
    "VectorDBClient",
    "SemanticSearch",
    "KnowledgeBase",
]

print("VORTA Vector Store package loaded.")
