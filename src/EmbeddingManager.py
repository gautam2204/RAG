from typing import List
from sentence_transformers import SentenceTransformer

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
    manager = EmbeddingManager()
    sample_texts = [
        "Artificial Intelligence is transforming the world.",
        "Sentence transformers are great for embeddings."
    ]
    embeddings = manager.generate_embeddings(sample_texts)
    print(f"Generated {len(embeddings)} embeddings.")
    print("Shape of first embedding:", embeddings[0].shape)
