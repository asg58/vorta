# services/vector_store/knowledge_base.py

import asyncio
from datetime import datetime
from typing import Any, Dict, List, Optional

from .semantic_search import SemanticSearch

class KnowledgeBase:
    """
    Manages the long-term knowledge of the AGI, using the semantic
    search service to store and retrieve information.
    """

    def __init__(self, semantic_search: SemanticSearch, index_name: str = "agi-knowledge-base"):
        """
        Initializes the KnowledgeBase.

        Args:
            semantic_search (SemanticSearch): The semantic search service instance.
            index_name (str): The default index name for the knowledge base.
        """
        self.search = semantic_search
        self.index_name = index_name
        print(f"KnowledgeBase initialized for index '{self.index_name}'.")

    async def add_entry(
        self,
        entry_id: str,
        text: str,
        source: str,
        metadata: Optional[Dict[str, Any]] = None,
        namespace: Optional[str] = "global",
    ) -> Dict[str, Any]:
        """
        Adds a new piece of knowledge to the knowledge base.

        Args:
            entry_id (str): A unique ID for the knowledge entry.
            text (str): The textual content of the knowledge.
            source (str): The source of the information (e.g., 'user_conversation', 'document').
            metadata (Optional[Dict[str, Any]]): Additional metadata.
            namespace (Optional[str]): The knowledge namespace (e.g., a user ID for personalization).

        Returns:
            Dict[str, Any]: The response from the underlying search service.
        """
        print(f"Adding knowledge entry '{entry_id}' from source '{source}'.")
        
        full_metadata = metadata or {}
        full_metadata.update({
            "source": source,
            "created_at": datetime.utcnow().isoformat(),
        })

        return await self.search.add_document(
            index_name=self.index_name,
            document_id=entry_id,
            text=text,
            metadata=full_metadata,
            namespace=namespace,
        )

    async def retrieve_relevant_knowledge(
        self,
        query: str,
        top_k: int = 3,
        namespace: Optional[str] = "global",
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieves the most relevant knowledge entries for a given query.

        Args:
            query (str): The query to find relevant knowledge for.
            top_k (int): The number of entries to retrieve.
            namespace (Optional[str]): The namespace to search within.
            filter (Optional[Dict[str, Any]]): Metadata filter to apply.

        Returns:
            List[Dict[str, Any]]: A list of relevant knowledge entries.
        """
        print(f"Retrieving top {top_k} knowledge entries for query: '{query}'")
        
        results = await self.search.search(
            index_name=self.index_name,
            query_text=query,
            top_k=top_k,
            namespace=namespace,
            filter=filter,
        )
        return results

    async def delete_entry(
        self, entry_id: str, namespace: Optional[str] = "global"
    ) -> Dict[str, Any]:
        """
        Deletes a knowledge entry by its ID.

        Args:
            entry_id (str): The ID of the entry to delete.
            namespace (Optional[str]): The namespace from which to delete.

        Returns:
            Dict[str, Any]: The response from the search service.
        """
        print(f"Deleting knowledge entry '{entry_id}'.")
        return await self.search.remove_document(
            index_name=self.index_name,
            document_id=entry_id,
            namespace=namespace,
        )

    async def get_summary(self, namespace: Optional[str] = "global") -> Dict[str, Any]:
        """
        Provides a summary of the knowledge base statistics.
        Note: This relies on the `describe_index_stats` method of the VectorDBClient.

        Args:
            namespace (Optional[str]): The namespace to get a summary for.

        Returns:
            Dict[str, Any]: A summary of the knowledge base.
        """
        print(f"Getting summary for knowledge base index '{self.index_name}'...")
        try:
            stats = await self.search.db_client.describe_index_stats(self.index_name)
            # This is a hypothetical structure of the stats response
            if namespace and "namespaces" in stats:
                return {
                    "index_name": self.index_name,
                    "namespace_stats": stats["namespaces"].get(namespace, {}),
                    "total_vector_count": stats.get("totalVectorCount", 0),
                }
            return stats
        except Exception as e:
            print(f"Could not retrieve stats for index '{self.index_name}': {e}")
            return {"error": str(e)}

# Example usage
async def main():
    import os
    from .vector_db_client import VectorDBClient

    # Setup mock clients
    api_key = os.environ.get("VECTOR_DB_API_KEY", "fake-api-key")
    host = os.environ.get("VECTOR_DB_HOST", "http://localhost:8001")
    
    db_client = VectorDBClient(api_key=api_key, host=host)
    search_service = SemanticSearch(db_client)
    kb = KnowledgeBase(search_service)

    try:
        # The following calls will fail without a running mock server.
        # They demonstrate the intended usage.
        
        # 1. Add knowledge
        # await kb.add_entry(
        #     entry_id="vorta-info-001",
        #     text="VORTA is an AGI voice agent designed for enterprise use.",
        #     source="manual_input"
        # )

        # 2. Retrieve knowledge
        # relevant_docs = await kb.retrieve_relevant_knowledge("What is VORTA?")
        # print("Retrieved documents:", relevant_docs)
        pass
    except Exception as e:
        print(f"An error occurred in the KnowledgeBase example: {e}")
    finally:
        await db_client.close()

if __name__ == "__main__":
    # asyncio.run(main())
    print("KnowledgeBase class defined. Run with a mock server to test functionality.")
