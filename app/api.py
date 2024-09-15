"""
Application layer for the NLP Research Assistant application.
"""

from fastapi import FastAPI

from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (adjust as needed)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/ask")
async def ask():
    return {"message": "Hello World"}


@app.post("/ask-fine-tuned")
async def experiment():
    return {"message": "Hello World"}
