"""
Application layer for the NLP Research Assistant application.
"""

import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from pydantic import BaseModel
from contextlib import asynccontextmanager

from serving.semantic_search import SemanticSearch
from serving.qa_model import QuestionAnswering


search_engine = None
fine_tuned_search_engine = None
qa_model = None
fine_tuned_qa_model = None
chroma_base = None
chroma_fine_tuned = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan event handler to initialize and cleanup resources.
    """
    global search_engine, fine_tuned_search_engine, qa_model, fine_tuned_qa_model

    # Load the embeddings models and paths
    embeddings_model = os.environ["EMBEDDINGS_MODEL"]
    chroma_path = os.environ["CHROMA_PATH"]
    embeddings_model_fine_tuned = os.environ["EMBEDDINGS_MODEL_FINE_TUNED"]
    chroma_fine_tuned_path = os.environ["CHROMA_FINE_TUNED_PATH"]

    chroma_base = Chroma(
        persist_directory=chroma_path,
        embedding_function=HuggingFaceEmbeddings(model_name=embeddings_model),
    )

    chroma_fine_tuned = Chroma(
        persist_directory=chroma_fine_tuned_path,
        embedding_function=HuggingFaceEmbeddings(
            model_name=embeddings_model_fine_tuned
        ),
    )

    # Initialize the SemanticSearch engines
    search_engine = SemanticSearch(
        model_name=embeddings_model,
        chroma=chroma_base,
    )

    fine_tuned_search_engine = SemanticSearch(
        model_name=embeddings_model_fine_tuned,
        chroma=chroma_fine_tuned,
    )

    # Initialize the QuestionAnswering models
    qa_model = QuestionAnswering(
        embeddings_model=embeddings_model,
        chroma=chroma_base,
        qa_model_name="distilbert-base-uncased-distilled-squad",
    )

    fine_tuned_qa_model = QuestionAnswering(
        embeddings_model=embeddings_model_fine_tuned,
        chroma=chroma_fine_tuned,
        qa_model_name="distilbert-base-uncased-distilled-squad",
    )
    print("Models loaded successfully.")

    yield


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    query: str


class QueryResponseItem(BaseModel):
    page_content: str
    similarity: float
    source: str


class QueryResponse(BaseModel):
    responses: list[QueryResponseItem]


@app.post("/search")
async def search(request: QueryRequest) -> QueryResponse:
    """
    Ask the model a question.
    """
    results = search_engine.search(request.query)

    response = QueryResponse(
        responses=[
            QueryResponseItem(
                page_content=result[0].page_content,
                similarity=result[1],
                source=result[0].metadata.get("source", None),
            )
            for result in results
        ]
    )

    return response


@app.post("/search-fine-tuned")
async def search_fine_tuned(request: QueryRequest) -> QueryResponse:
    """
    Ask the fine-tuned model a question.
    """
    results = fine_tuned_search_engine.search(request.query)

    response = QueryResponse(
        responses=[
            QueryResponseItem(
                page_content=result[0].page_content,
                similarity=result[1],
                source=result[0].metadata.get("source", None),
            )
            for result in results
        ]
    )

    return response


class ChatResponse(BaseModel):
    answer: str


@app.post("/chat")
async def chat(request: QueryRequest) -> ChatResponse:
    """
    Ask the model a question.
    """
    response = qa_model.answer_question(request.query)

    return ChatResponse(answer=response)


@app.post("/chat-fine-tuned")
async def chat_fine_tuned(request: QueryRequest) -> ChatResponse:
    """
    Ask the model a question.
    """
    response = fine_tuned_qa_model.answer_question(request.query)

    return ChatResponse(answer=response)


@app.get("/get-image/{image_path:path}")
async def get_image(image_path: str):
    """
    Serve an image file from the given file path.
    """
    if os.path.exists(image_path):
        return FileResponse(image_path)

    return JSONResponse(content={"error": "Image not found."}, status_code=404)


class PlotResponse(BaseModel):
    images: list[str]


@app.get("/base-plots")
async def get_base_plots() -> PlotResponse:
    """
    Get the base plots.
    """
    base_dir = os.path.join("experiments", "plots", "base")
    wordcloud_dir = os.path.join(base_dir, "wordcloud")

    image_paths = []

    if os.path.exists(base_dir):
        for file_name in os.listdir(base_dir):
            if file_name.endswith(".png"):
                image_paths.append(os.path.join(base_dir, file_name))

    if os.path.exists(wordcloud_dir):
        for file_name in os.listdir(wordcloud_dir):
            if file_name.endswith(".png"):
                image_paths.append(os.path.join(wordcloud_dir, file_name))

    if not image_paths:
        return {"message": "No images found."}

    return {"images": image_paths}


@app.get("/fine-tuned-plots")
async def get_fine_tuned_plots() -> PlotResponse:
    """
    Get the fine-tuned plots.
    """
    base_dir = os.path.join("experiments", "plots", "fine-tuned")
    wordcloud_dir = os.path.join(base_dir, "wordcloud")

    image_paths = []

    if os.path.exists(base_dir):
        for file_name in os.listdir(base_dir):
            if file_name.endswith(".png"):
                image_paths.append(os.path.join(base_dir, file_name))

    if os.path.exists(wordcloud_dir):
        for file_name in os.listdir(wordcloud_dir):
            if file_name.endswith(".png"):
                image_paths.append(os.path.join(wordcloud_dir, file_name))

    if not image_paths:
        return {"message": "No images found."}

    return {"images": image_paths}
