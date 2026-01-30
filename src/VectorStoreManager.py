import os
import uuid
from typing import List

import chromadb

from DocsChunking import Chunking
from EmbeddingManager import EmbeddingManager
from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer

from LoadData import DataIngestion

class VectorStoreManager:
    def __init__(self, collectiona_name="resumes_document",embeddings_list=None,persist_directory="../store/chroma_db"):
        
        # Create persistent directory if it doesn't exist
        if not os.path.exists(persist_directory):
            os.makedirs(persist_directory)
        
        self.collectiona_name = collectiona_name
        self.embeddings_list = embeddings_list
        self.persist_directory = persist_directory
        self.client = None
        self.collection = None
        self.collection = self.__initialize_store()
    
    def __initialize_store(self):
        import chromadb

        self.client = chromadb.PersistentClient(
                path=self.persist_directory
            )
        self.collection = self.client.get_or_create_collection(
            name=self.collectiona_name,
            metadata={"description": "Resume Embeddings for RAG"},
            )
        
        return self.collection

    def add_documents(self, documents: List[Document]):
        
        
        """
        Add LangChain Document objects to the vector store.
        Each document gets a UUID as its ID.
        """
        if len(documents) != len(self.embeddings_list):
            raise ValueError("Length of documents and embeddings must be equal.")
        
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        ids = [str(uuid.uuid4()) for _ in documents]

        # Add to Chroma vector store
        self.collection.add(
            documents=texts,
            metadatas=metadatas,
            ids=ids,
            embeddings=self.embeddings_list
        )

        print(f"✅ Added {len(documents)} documents with UUIDs to vector store.")

    def get_vector_store(self):
        """Return the Chroma vector store instance."""
        return self.collection

if __name__ == "__main__":
    # Example usage
    DataIngestionObj = DataIngestion(directory_path="./data")
    documents = DataIngestionObj.get_documents()
    chunked_documents = Chunking.split_documents(documents)
    print(f"Number of chunked documents: {len(chunked_documents)}")
    print(len(documents), len(chunked_documents))
    
    chunk_texts = [doc.page_content for doc in chunked_documents]
    
    manager = EmbeddingManager() 
    embeddings = manager.generate_embeddings(chunk_texts)
    print(f"Generated {len(embeddings)} embeddings.")
    print("Shape of first embedding:", embeddings[0].shape)

    manager = VectorStoreManager(embeddings_list = embeddings,persist_directory="./store/")
    manager.add_documents(chunked_documents)
    print("Documents added to vector store.")
    