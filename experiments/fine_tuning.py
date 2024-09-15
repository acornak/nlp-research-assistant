"""

"""

import os
import logging
import nltk

from langchain_community.document_loaders import UnstructuredPDFLoader
from sentence_transformers import SentenceTransformer, InputExample, losses

import mlflow
from sklearn.metrics.pairwise import cosine_similarity

import time
import numpy as np

from torch.utils.data import DataLoader

from datasets import Dataset  # noqa: F401

logging.basicConfig(level=logging.INFO)


class SentenceTransformerFineTuning:
    def __init__(self, data_path: str, embeddings_model: str) -> None:
        """
        Initialize the SentenceTransformerFineTuning object.

        Args:
            data_path (str): The directory where the data is stored.
            embeddings_model (str): The name of the Hugging Face embeddings model.
        """
        self.logger = logging.getLogger(__name__)

        self.data_path = data_path

        self.model = SentenceTransformer(embeddings_model)
        self.documents = []
        self.sentences = []
        self.sentence_pairs = []

    def _load_data(self) -> None:
        """
        Load the pdf data.
        """
        try:
            self.logger.info("Loading PDFs...")
            for filename in os.listdir(self.data_path):
                if filename.endswith(".pdf"):
                    pdf_path = os.path.join(self.data_path, filename)
                    loader = UnstructuredPDFLoader(pdf_path)
                    self.documents.extend(loader.load())

        except Exception as e:
            self.logger.error(f"Error loading PDFs: {e}")

    def _create_sentence_pairs(self) -> None:
        """
        Create sentence pairs for training.
        """
        self.logger.info("Creating sentence pairs...")

        for document in self.documents:
            sentences = nltk.sent_tokenize(document.page_content)

            for i in range(0, len(sentences) - 1, 2):
                sentence_pair = (sentences[i], sentences[i + 1])
                self.sentence_pairs.append(sentence_pair)

        self.logger.info(f"Created {len(self.sentence_pairs)} sentence pairs.")

    def _prepare_training_data(self) -> list[InputExample]:
        """
        Prepare the training data.
        """
        train_examples = []
        for sentence1, sentence2 in self.sentence_pairs:
            train_examples.append(InputExample(texts=[sentence1, sentence2]))

        return train_examples

    def _calculate_metrics(self, embeddings: np.array) -> float:
        """
        Calculate the metrics for the embeddings.

        Args:
            embeddings (np.array): The embeddings.
        """
        cosine_sim = cosine_similarity(embeddings)
        avg_cosine_sim = np.mean(cosine_sim)

        return avg_cosine_sim

    def fine_tune(self, output_path: str, epochs: int, batch_size: int) -> None:
        """
        Fine-tune the Sentence Transformer model.

        Args:
            output_path (str): The path to save the fine-tuned model.
            epochs (int): The number of epochs to train.
            batch_size (int): The batch size for training.
        """
        with mlflow.start_run():
            mlflow.log_param("epochs", epochs)
            mlflow.log_param("batch_size", batch_size)

            self._load_data()
            self._create_sentence_pairs()
            train_examples = self._prepare_training_data()

            train_dataloader = DataLoader(
                train_examples, shuffle=True, batch_size=batch_size
            )

            train_loss = losses.MultipleNegativesRankingLoss(
                self.model
            )  # unsupervised loss function

            self.model.fit(
                train_objectives=[(train_dataloader, train_loss)],
                epochs=epochs,
                output_path=output_path,
            )

            embeddings = self.model.encode(
                [
                    f"{sentence1} {sentence2}"
                    for sentence1, sentence2 in self.sentence_pairs
                ]
            )

            avg_cosine_sim = self._calculate_metrics(embeddings)

            mlflow.log_metric("avg_cosine_sim", avg_cosine_sim)

            mlflow.log_artifact(output_path)

            mlflow.pytorch.log_model(self.model, "model")

            start_time = time.time()
            _ = self.model.encode(self.sentences)
            inference_time = time.time() - start_time
            mlflow.log_metric("inference_time", inference_time)


if __name__ == "__main__":
    try:
        data_path = os.environ["DATA_PATH"]
        embeddings_model = os.environ["EMBEDDINGS_MODEL"]
    except KeyError as e:
        logging.error(f"Missing environment variable: {str(e)}")
        exit(1)

    fixture_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(fixture_dir, "fine-tuned-model")

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    fine_tuner = SentenceTransformerFineTuning(data_path, embeddings_model)
    fine_tuner.fine_tune(output_path=output_path, epochs=3, batch_size=16)
