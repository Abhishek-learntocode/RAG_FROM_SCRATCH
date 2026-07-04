# Retrieval-Augmented Question Answering System

A lightweight, local-first Retrieval-Augmented Generation (RAG) pipeline designed for low-latency document retrieval and context-aware response generation. This system is built entirely without heavy web frameworks, utilizing a pure command-line interface for efficient, fast-execution querying.

## System Architecture

This project emphasizes a strict separation of concerns, ensuring high modularity and maintainability:

*   **Storage & Metadata (SQLite):** A lightweight relational database (`DatabaseManager`) maps document metadata and raw text chunks to vector indices. This decoupling allows the vector store to remain strictly numerical, improving scalability.
*   **Vector Indexing (FAISS):** High-performance semantic similarity search is handled by Facebook AI Similarity Search (FAISS), optimized for L2 distance calculations on dense vectors.
*   **Embedding Pipeline (PyTorch):** Utilizes `sentence-transformers` for generating robust dense vector representations of document chunks.
*   **Generation Pipeline:** A Hugging Face inference pipeline dynamically integrates top-K retrieved contexts into a generator model (`google/flan-t5-small` by default) to synthesize factual answers.

## Tech Stack

*   **Language:** Python 3.x
*   **Machine Learning / AI:** PyTorch, Transformers, SentenceTransformers
*   **Vector Database:** FAISS
*   **Relational Database:** SQLite
*   **Document Processing:** PyPDF2

## Installation

1. Clone the repository:
   ```bash
   git clone [https://github.com/your-username/mbsqi-repo.git](https://github.com/your-username/mbsqi-repo.git)
   cd mbsqi-repo
   
