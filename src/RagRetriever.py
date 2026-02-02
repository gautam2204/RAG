# Rag retriever module
from typing import Any, Dict


class RagRetriever:
    def __init__(self, vector_store_manager, embddings_manager):
        self.vector_store_manager = vector_store_manager
        self.embddings_manager = embddings_manager
        
    def retrieve(self, query, top_k=5)->list[Dict[str, Any]]:
        """Retrieve top_k relevant documents for the given query."""
        # Generate embedding for the query
        query_embedding = self.embddings_manager.generate_embeddings(query)
        
        # Perform similarity search in the vector store
        results = self.vector_store_manager.get_vector_store().query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        
        return results
    
    
if __name__ == "__main__":
    from VectorStoreManager import VectorStoreManager
    from EmbeddingManager import EmbeddingManager

    # Initialize managers
    vector_store_manager = VectorStoreManager(persist_directory="./store/")
    embeddings_manager = EmbeddingManager()

    # Initialize RAG Retriever
    rag_retriever = RagRetriever(vector_store_manager, embeddings_manager)

    # Example query
    query = "Experience"
    results = rag_retriever.retrieve(query, top_k=3)
    print("+++++++++++ \n",results)
    print("Retrieved Documents:")
    for doc in results['documents']:
        print(doc)
