# services/vector_store/vector_db_client.py
"""
VORTA: Vector Database Client

This module provides a client for interacting with a vector database, such as
Pinecone, Weaviate, or a self-hosted solution like FAISS. It abstracts the
specific implementation details of the vector DB, allowing the rest of the
application to interact with it through a simple, unified interface.

Key Functions:
- Upserting (inserting or updating) vectors.
- Querying for similar vectors.
- Managing indexes and collections.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np

# Configure logging
logger = logging.getLogger(__name__)

# --- Optional Dependencies for Vector DBs ---
# This allows the system to work with whichever library is installed.
try:
    import pinecone
    _PINECONE_AVAILABLE = True
except ImportError:
    pinecone = None
    _PINECONE_AVAILABLE = False

try:
    import weaviate
    _WEAVIATE_AVAILABLE = True
except ImportError:
    weaviate = None
    _WEAVIATE_AVAILABLE = False

# --- Data Structures ---

@dataclass
class Vector:
    """Represents a single vector entry."""
    id: str
    values: List[float]
    metadata: Dict[str, Any] = None

@dataclass
class QueryResult:
    """Represents a single result from a similarity search."""
    id: str
    score: float
    metadata: Optional[Dict[str, Any]] = None

# --- Abstract Base Client ---

class BaseVectorClient:
    """Abstract base class for a vector database client."""
    async def connect(self):
        raise NotImplementedError

    async def disconnect(self):
        raise NotImplementedError

    async def upsert(self, index_name: str, vectors: List[Vector]) -> bool:
        raise NotImplementedError

    async def query(self, index_name: str, vector: List[float], top_k: int = 5) -> List[QueryResult]:
        raise NotImplementedError

    async def create_index(self, index_name: str, dimension: int):
        raise NotImplementedError

# --- Pinecone Client Implementation ---

class PineconeClient(BaseVectorClient):
    """A client for interacting with Pinecone."""
    def __init__(self, api_key: str, environment: str):
        if not _PINECONE_AVAILABLE:
            raise ImportError("Pinecone client is not available. Please install with 'pip install pinecone-client'.")
        self.api_key = api_key
        self.environment = environment
        self.pinecone = pinecone

    async def connect(self):
        logger.info("Connecting to Pinecone...")
        self.pinecone.init(api_key=self.api_key, environment=self.environment)
        logger.info("Successfully connected to Pinecone.")

    async def disconnect(self):
        # Pinecone SDK doesn't have an explicit disconnect method
        logger.info("Pinecone client does not require explicit disconnection.")
        pass

    async def create_index(self, index_name: str, dimension: int):
        if index_name not in self.pinecone.list_indexes():
            logger.info(f"Creating Pinecone index '{index_name}' with dimension {dimension}.")
            self.pinecone.create_index(name=index_name, dimension=dimension, metric='cosine')
        else:
            logger.info(f"Pinecone index '{index_name}' already exists.")

    async def upsert(self, index_name: str, vectors: List[Vector]) -> bool:
        index = self.pinecone.Index(index_name)
        vectors_to_upsert = [(v.id, v.values, v.metadata) for v in vectors]
        try:
            index.upsert(vectors=vectors_to_upsert)
            logger.info(f"Successfully upserted {len(vectors)} vectors to index '{index_name}'.")
            return True
        except Exception as e:
            logger.error(f"Failed to upsert to Pinecone: {e}", exc_info=True)
            return False

    async def query(self, index_name: str, vector: List[float], top_k: int = 5) -> List[QueryResult]:
        index = self.pinecone.Index(index_name)
        try:
            query_response = index.query(vector=vector, top_k=top_k, include_metadata=True)
            results = [
                QueryResult(id=m.id, score=m.score, metadata=m.metadata)
                for m in query_response.matches
            ]
            return results
        except Exception as e:
            logger.error(f"Failed to query Pinecone: {e}", exc_info=True)
            return []

# --- In-Memory/FAISS Client (Simulation) ---

class InMemoryVectorClient(BaseVectorClient):
    """
    A simple in-memory vector client for local development and testing.
    Uses numpy for calculations. A more advanced version would use FAISS.
    """
    def __init__(self):
        self.indexes: Dict[str, Dict[str, Vector]] = {}
        logger.info("Initialized InMemoryVectorClient (for local testing).")

    async def connect(self):
        logger.info("In-memory client connected (no-op).")
        pass

    async def disconnect(self):
        logger.info("In-memory client disconnected (no-op).")
        pass

    async def create_index(self, index_name: str, dimension: int):
        if index_name not in self.indexes:
            self.indexes[index_name] = {}
            logger.info(f"Created in-memory index '{index_name}' (dimension {dimension} is noted).")
        else:
            logger.info(f"In-memory index '{index_name}' already exists.")

    async def upsert(self, index_name: str, vectors: List[Vector]) -> bool:
        if index_name not in self.indexes:
            logger.error(f"Index '{index_name}' does not exist.")
            return False
        for v in vectors:
            self.indexes[index_name][v.id] = v
        logger.info(f"Upserted {len(vectors)} vectors to in-memory index '{index_name}'.")
        return True

    async def query(self, index_name: str, vector: List[float], top_k: int = 5) -> List[QueryResult]:
        if index_name not in self.indexes:
            return []
        
        query_vec = np.array(vector)
        all_vectors = list(self.indexes[index_name].values())
        
        # Calculate cosine similarity
        similarities = []
        for v in all_vectors:
            v_vec = np.array(v.values)
            # Cosine similarity = A . B / (||A|| * ||B||)
            sim = np.dot(query_vec, v_vec) / (np.linalg.norm(query_vec) * np.linalg.norm(v_vec))
            similarities.append(QueryResult(id=v.id, score=sim, metadata=v.metadata))
            
        # Sort by score and return top_k
        similarities.sort(key=lambda x: x.score, reverse=True)
        return similarities[:top_k]

# --- Client Factory ---

def get_vector_db_client(config: Dict[str, Any]) -> BaseVectorClient:
    """
    Factory function to get the appropriate vector DB client based on config.
    """
    db_type = config.get("type", "in_memory")
    if db_type == "pinecone" and _PINECONE_AVAILABLE:
        logger.info("Creating Pinecone vector DB client.")
        return PineconeClient(api_key=config["api_key"], environment=config["environment"])
    # Add other clients like Weaviate here
    # elif db_type == "weaviate" and _WEAVIATE_AVAILABLE:
    #     ...
    else:
        logger.info("Creating in-memory vector DB client for local testing.")
        return InMemoryVectorClient()
