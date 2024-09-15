# Research Assistant Project using NLP

## Introduction
This is a demo project using NLP to create a research assistant that can help in finding relevant research papers based on a query. The project usess sentence transformers to create embeddings of the research papers and the query. The embeddings are then used to find the most similar research papers to the query. Data are not included in the repository, but model was trained on a small amount of LLM-related research papers in PDF format. 

Alternatively, it could be trained on the [CORD-19 dataset](https://www.semanticscholar.org/cord19).


## How to run:
- Install the required packages using the following command - ```poetry install --no-root```

## Sources
- [Fine-tuning Sentence Transformer (before v3.0)](https://huggingface.co/blog/how-to-train-sentence-transformers)
- [Fine-tuning Sentence Transformer (v3.0)](https://huggingface.co/blog/train-sentence-transformers)
- [Understanding Cosine Similarity and Word Embeddings](https://spencerporter2.medium.com/understanding-cosine-similarity-and-word-embeddings-dbf19362a3c)
