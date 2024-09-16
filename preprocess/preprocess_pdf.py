"""
This module contains the Preprocess class, which is used to preprocess the data.
"""

import os
import logging


from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_chroma import Chroma
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings


logging.basicConfig(level=logging.INFO)


class Preprocess:
    def __init__(
        self,
        data_path: str,
        chroma_path: str,
        embeddings_model: str,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
    ) -> None:
        """
        Initialize the Preprocess object.

        Args:
            data_path (str): The directory where the data is stored.
            chroma_path (str): The directory where the ChromaDB is stored.
            embeddings_model (str): The name of the Hugging Face embeddings model.
            chunk_size (int): The size of each chunk.
            chunk_overlap (int): The overlap between chunks.
        """
        self.logger = logging.getLogger(__name__)

        self.data_path = data_path
        self.chroma_path = chroma_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.embeddings = HuggingFaceEmbeddings(model_name=embeddings_model)
        self.chunks: list[Document] = []

    def _load_and_create_chunks(self) -> None:
        """
        Load PDF files from the data directory.
        """
        try:

            self.logger.info("Loading PDFs...")

            for filename in os.listdir(self.data_path):
                if filename.endswith(".pdf"):
                    pdf_path = os.path.join(self.data_path, filename)
                    loader = UnstructuredPDFLoader(pdf_path)
                    chunks = loader.load_and_split(
                        RecursiveCharacterTextSplitter(
                            chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
                        )
                    )
                    self.chunks.extend(chunks)

            self.logger.info(f"Created {len(self.chunks)} chunks.")

        except Exception as e:
            self.logger.error(f"Error loading PDFs: {e}")
            exit(1)

    def _store_chunks(self) -> Chroma:
        """
        Store the chunks in ChromaDB.
        """
        db = Chroma.from_documents(
            self.chunks,
            self.embeddings,
            persist_directory=self.chroma_path,
        )

        return db

    def prepare_data(self) -> Chroma:
        """
        Prepare the data for training.
        """
        if os.path.exists(self.chroma_path):
            db = Chroma(
                persist_directory=self.chroma_path,
                embedding_function=self.embeddings,
            )
            return db

        self._load_and_create_chunks()
        db = self._store_chunks()

        return db
