"""
This module contains the SemanticSearch class that uses the Sentence Transformers library to perform semantic search on a given query.
"""

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


class SemanticSearch:
    def __init__(self, model_name: str, chroma: Chroma) -> None:
        """
        Initialize the SemanticSearch object.

        Args:
            model_name (str): The name of the Hugging Face embeddings model.
            chroma_path (str): The path to the Chroma database.
        """
        self.model = HuggingFaceEmbeddings(model_name=model_name)
        self.chroma_db = chroma

    def search(
        self, query: str, top_k: int = 5, min_confidence: float = 0.6
    ) -> list[tuple[str, float]]:
        """
        Search for similar documents given a query.

        Args:
            query (str): The query string.
            top_k (int): The number of similar documents to return.
            min_confidence (float): The minimum confidence threshold.
        """

        results = self.chroma_db.similarity_search_with_score(query, k=top_k)
        results = sorted(results, key=lambda x: x[1], reverse=True)

        return [(doc, score) for doc, score in results if score >= min_confidence]
