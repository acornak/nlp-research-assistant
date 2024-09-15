"""
"""

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import pipeline, AutoModelForQuestionAnswering, AutoTokenizer

from serving.semantic_search import SemanticSearch


class QuestionAnswering:
    def __init__(
        self, embeddings_model: str, chroma: Chroma, qa_model_name: str
    ) -> None:
        """
        Initialize the QuestionAnsweringSystem object.

        Args:
            model_name (str): The name of the Hugging Face embeddings model.
            chroma_path (str): The path to the Chroma database.
            qa_model_name (str): The Hugging Face Question-Answering model.
        """
        self.embeddings_model = embeddings_model
        self.model = HuggingFaceEmbeddings(model_name=embeddings_model)
        self.chroma_db = chroma

        self.qa_model = AutoModelForQuestionAnswering.from_pretrained(qa_model_name)
        self.qa_tokenizer = AutoTokenizer.from_pretrained(qa_model_name)

        self.qa_pipeline = pipeline(
            "question-answering", model=self.qa_model, tokenizer=self.qa_tokenizer
        )

        self.min_confidence = 0.6

    def answer_question(self, question: str) -> str:
        """
        Answer a question given a query.

        Args:
            question (str): The question string.
        """
        search_engine = SemanticSearch(
            model_name=self.embeddings_model, chroma=self.chroma_db
        )

        results = search_engine.search(
            question, top_k=2, min_confidence=self.min_confidence
        )

        best_answer = None
        best_score = 0

        for result in results:
            context = result[0].page_content
            qa_result = self.qa_pipeline(question=question, context=context)

            if qa_result["score"] > best_score:
                best_score = qa_result["score"]
                best_answer = qa_result["answer"]

        return best_answer or "No confident answer found."
