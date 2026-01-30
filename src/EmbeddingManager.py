from typing import List
from sentence_transformers import SentenceTransformer
import os
from DocsChunking import Chunking
from LoadData import DataIngestion
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'

class EmbeddingManager:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the embedding manager with a sentence transformer model.
        
        Args:
            model_name (str): Name of the sentence transformer model to load.
                              Defaults to 'all-MiniLM-L6-v2'.
        """
        self.model = SentenceTransformer(model_name)

    def generate_embeddings(self, texts: List[str]):
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts (List[str]): List of input strings.
        
        Returns:
            List[List[float]]: List of embeddings (each embedding is a vector).
        """
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings


if __name__ == "__main__":
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
