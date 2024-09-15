"""
Main script to run the application. It checks if the necessary data is available and starts the server.
"""

import os
import logging
import uvicorn

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from experiments.fine_tuning import SentenceTransformerFineTuning
from preprocess.preprocess_pdf import Preprocess
from experiments.visualize import VisualizeEmbeddings

logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    try:
        data_path = os.environ["DATA_PATH"]
        chroma_path = os.environ["CHROMA_PATH"]
        chroma_fine_tuned_path = os.environ["CHROMA_FINE_TUNED_PATH"]
        embeddings_model = os.environ["EMBEDDINGS_MODEL"]
        embeddings_model_fine_tuned = os.environ["EMBEDDINGS_MODEL_FINE_TUNED"]
    except KeyError as e:
        logging.error(f"Missing environment variable: {str(e)}")
        exit(1)

    if not os.path.exists(data_path):
        logging.error("Data path does not exist.")
        exit(1)

    if not os.path.exists(embeddings_model_fine_tuned):
        logging.info(
            "Fine-tuned model not found. Starting training. This may take a while."
        )

        fine_tuner = SentenceTransformerFineTuning(data_path, embeddings_model)
        fine_tuner.fine_tune(
            output_path=embeddings_model_fine_tuned, epochs=3, batch_size=16
        )

    if not os.path.exists(chroma_path):
        logging.info("Chroma db not found. Starting preprocessing data.")

        preprocess = Preprocess(data_path, chroma_path, embeddings_model)
        preprocess.prepare_data()

    if not os.path.exists(chroma_fine_tuned_path):
        logging.info("Chroma fine-tuned db not found. Starting preprocessing data.")

        preprocess_fine_tuned = Preprocess(
            data_path, chroma_fine_tuned_path, embeddings_model_fine_tuned
        )
        preprocess_fine_tuned.prepare_data()

    if not os.path.exists(os.path.join("experiments", "plots")):
        base_path = os.path.join("experiments", "plots", "base")
        fine_tuned_path = os.path.join("experiments", "plots", "fine-tuned")

        os.makedirs(os.path.join(base_path, "wordcloud"))
        os.makedirs(os.path.join(fine_tuned_path, "wordcloud"))

        chroma_base = Chroma(
            persist_directory=chroma_path,
            embedding_function=HuggingFaceEmbeddings(model_name=embeddings_model),
        )

        visualizer_base = VisualizeEmbeddings(
            db=chroma_base,
        )
        visualizer_base.visualize(base_path)

        chroma_fine_tuned = Chroma(
            persist_directory=chroma_fine_tuned_path,
            embedding_function=HuggingFaceEmbeddings(
                model_name=embeddings_model_fine_tuned
            ),
        )

        visualizer_fine_tuned = VisualizeEmbeddings(
            db=chroma_fine_tuned,
        )
        visualizer_fine_tuned.visualize(fine_tuned_path)

    logging.info("All data prepared. Starting the server.")
    uvicorn.run("app.api:app", host="0.0.0.0", port=8000)
