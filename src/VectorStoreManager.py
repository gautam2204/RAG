import os
import uuid
from typing import List

import chromadb

from EmbeddingManager import EmbeddingManager
from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer


class EmbeddingManager:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize SentenceTransformer model."""
        self.model = SentenceTransformer(model_name)

    def generate_embeddings(self, texts: List[str]):
        """Generate embeddings for a list of texts."""
        return self.model.encode(texts, convert_to_numpy=True)


class VectorStoreManager:
    def __init__(self, persist_directory: str = "./chroma_db"):
        """
        Initialize Chroma vector store with persistent storage.
        """
        self.persist_directory = persist_directory
        os.makedirs(self.persist_directory, exist_ok=True)

        # Use custom SentenceTransformer embeddings
        self.embedding_function = EmbeddingManager()

        # Initialize Chroma
        self.vector_store = chromadb.PersistentClient(
            persist_directory=self.persist_directory,
            embedding_function=self.embedding_function.model.encode  # pass the encode function
        )

    def add_documents(self, documents: List[Document]):
        """
        Add LangChain Document objects to the vector store.
        Each document gets a UUID as its ID.
        """
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        ids = [str(uuid.uuid4()) for _ in documents]

        # Generate embeddings
        embeddings_list = self.embedding_function.generate_embeddings(texts)

        # Add to Chroma
        self.vector_store.add_texts(
            texts=texts,
            metadatas=metadatas,
            ids=ids,
            embeddings=embeddings_list
        )

        print(f"✅ Added {len(documents)} documents with UUIDs to vector store.")
        return ids

    def get_vector_store(self):
        """Return the Chroma vector store instance."""
        return self.vector_store


if __name__ == "__main__":
    # Example usage
    sample_doc = Document(
        page_content="Skills WORK EXPERIENCE TIAA GBS 2021-Current Associate Specialist",
        metadata={
            "source": "./data/Gautam_Rawat_resume_07_25.pdf",
            "page": 0,
            "producer": "www.smallpdf.com"
        }
    )

    manager = VectorStoreManager(persist_directory="./chroma_db")
    ids = manager.add_documents([sample_doc])
    print("Generated UUIDs:", ids)

    # Test retrieval
    results = manager.get_vector_store().similarity_search("work experience", k=2)
    for r in results:
        print(r.page_content, r.metadata)
