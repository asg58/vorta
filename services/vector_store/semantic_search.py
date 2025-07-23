# services/vector_store/semantic_search.py
"""
VORTA: Semantic Search and RAG Service

This module provides the core logic for performing semantic search and
implementing Retrieval-Augmented Generation (RAG). It uses the vector database
client to find relevant documents and then combines them with a user's query
to provide more context-aware and accurate responses from a language model.
"""

import logging
from typing import List, Dict, Any
from .vector_db_client import BaseVectorClient

# Optional dependency for text processing
try:
    from sentence_transformers import SentenceTransformer
    _TRANSFORMERS_AVAILABLE = True
except ImportError:
    SentenceTransformer = None
    _TRANSFORMERS_AVAILABLE = False

logger = logging.getLogger(__name__)

class SemanticSearcher:
    """
    Handles embedding text and searching for semantically similar content.
    """
    def __init__(self, vector_db_client: BaseVectorClient, embedding_model_name: str = 'all-MiniLM-L6-v2'):
        self.db_client = vector_db_client
        self.model_name = embedding_model_name
        self.embedding_model = None

        if not _TRANSFORMERS_AVAILABLE:
            logger.warning("SentenceTransformers library not found. Semantic search will not be functional.")
            logger.warning("Please install with 'pip install sentence-transformers'.")
        else:
            try:
                logger.info(f"Loading embedding model: {self.model_name}")
                self.embedding_model = SentenceTransformer(self.model_name)
                logger.info("Embedding model loaded successfully.")
            except Exception as e:
                logger.error(f"Failed to load embedding model '{self.model_name}': {e}", exc_info=True)
                self.embedding_model = None

    def _get_embedding(self, text: str) -> List[float]:
        """Generates a vector embedding for a given text."""
        if not self.embedding_model:
            raise RuntimeError("Embedding model is not available.")
        
        # The model.encode method returns a numpy array, which we convert to a list
        embedding = self.embedding_model.encode(text, convert_to_tensor=False)
        return embedding.tolist()

    async def search(self, index_name: str, query_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Performs a semantic search for a given query.
        """
        if not self.embedding_model:
            logger.error("Cannot perform search because embedding model is not loaded.")
            return []

        logger.info(f"Performing semantic search in index '{index_name}' for query: '{query_text[:50]}...'")
        
        # 1. Convert the query text to a vector embedding
        query_vector = self._get_embedding(query_text)
        
        # 2. Query the vector database for similar vectors
        search_results = await self.db_client.query(index_name, query_vector, top_k=top_k)
        
        # 3. Format and return the results
        formatted_results = [
            {
                "id": res.id,
                "score": res.score,
                "text": res.metadata.get("text", ""), # Assuming text is stored in metadata
                "metadata": res.metadata
            }
            for res in search_results
        ]
        
        logger.info(f"Found {len(formatted_results)} relevant documents.")
        return formatted_results

class RAGProcessor:
    """
    Implements the Retrieval-Augmented Generation (RAG) pattern.
    """
    def __init__(self, searcher: SemanticSearcher, llm_client: Any):
        # llm_client would be a client for an LLM like OpenAI's API
        self.searcher = searcher
        self.llm_client = llm_client
        logger.info("RAGProcessor initialized.")

    async def generate_response(self, index_name: str, user_query: str) -> str:
        """
        Generates a response by first retrieving relevant context and then
        querying a language model.
        """
        logger.info(f"Generating RAG response for query: '{user_query[:50]}...'")
        
        # 1. Retrieve relevant documents from the vector store
        retrieved_docs = await self.searcher.search(index_name, user_query, top_k=3)
        
        if not retrieved_docs:
            logger.warning("No relevant documents found for RAG. Falling back to standard LLM response.")
            context_str = "No specific context found."
        else:
            # 2. Format the retrieved documents into a context string
            context_items = [f"- {doc['text']}" for doc in retrieved_docs]
            context_str = "\n".join(context_items)

        # 3. Construct a prompt for the language model
        prompt = f"""
        You are an intelligent assistant. Please answer the user's question based on the following context.
        If the context does not contain the answer, say that you don't have enough information.

        Context:
        {context_str}

        User's Question:
        {user_query}

        Answer:
        """
        
        logger.info("Sending prompt to LLM for final response generation.")
        
        # 4. Call the language model to get the final answer
        # This is a placeholder for a real LLM API call
        # response = await self.llm_client.create_completion(prompt=prompt)
        # return response.text
        
        # Simulated response for demonstration
        simulated_response = f"Based on the retrieved context, the answer regarding '{user_query}' is likely related to the documents found. The top result had a similarity score of {retrieved_docs[0]['score']:.4f}." if retrieved_docs else f"I could not find specific information about '{user_query}' in my knowledge base."
        
        return simulated_response

# --- Example Usage ---

async def main():
    """Demonstrates the SemanticSearcher and RAGProcessor."""
    from .vector_db_client import InMemoryVectorClient, Vector
    
    logger.info("--- Semantic Search & RAG Demonstration ---")

    if not _TRANSFORMERS_AVAILABLE:
        logger.error("Cannot run demonstration without 'sentence-transformers'. Please install it.")
        return

    # 1. Setup a dummy vector DB client
    db_client = InMemoryVectorClient()
    await db_client.create_index("documents", dimension=384) # all-MiniLM-L6-v2 has 384 dimensions

    # 2. Setup the searcher
    searcher = SemanticSearcher(db_client)
    
    # 3. Add some documents to our in-memory DB
    docs_to_add = [
        "The VORTA project is an advanced AGI voice agent.",
        "VORTA uses a microservices architecture with a central orchestrator.",
        "Key technologies include Python, FastAPI, Docker, and Kubernetes.",
        "The sky is blue because of Rayleigh scattering.",
        "Apples are a type of fruit that grow on trees."
    ]
    
    embeddings = [searcher._get_embedding(doc) for doc in docs_to_add]
    vectors_to_upsert = [
        Vector(id=str(i), values=emb, metadata={"text": doc})
        for i, (doc, emb) in enumerate(zip(docs_to_add, embeddings))
    ]
    await db_client.upsert("documents", vectors_to_upsert)

    # 4. Perform a semantic search
    query = "What is VORTA?"
    print(f"\n--- Performing semantic search for: '{query}' ---")
    results = await searcher.search("documents", query)
    for res in results:
        print(f"  - ID: {res['id']}, Score: {res['score']:.4f}, Text: '{res['text']}'")

    # 5. Use the RAG processor to generate a response
    # We'll use a dummy LLM client for this example
    dummy_llm_client = "DUMMY_LLM" 
    rag_processor = RAGProcessor(searcher, dummy_llm_client)
    
    print(f"\n--- Generating RAG response for: '{query}' ---")
    response = await rag_processor.generate_response("documents", query)
    print(f"RAG Response:\n{response}")
    
    print("\nDemonstration complete.")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
