# Retrieval-Augmented Question Answering System

A production-ready, local-first Retrieval-Augmented Generation (RAG) pipeline. This system is engineered for low-latency document retrieval and context-aware response generation without relying on external APIs or heavy web frameworks. 

By running entirely via a command-line interface (CLI) and utilizing local storage and compute, this project delivers **real-life impact** through absolute data privacy, zero recurring API costs, and offline availability.

---

## 🏗️ High-Level Design (HLD)

The architecture is strictly modularized to separate state management, mathematical vector operations, and language generation:

1.  **Ingestion Pipeline:** Reads raw documents (PDF, TXT), sanitizes the text, and applies a sliding-window chunking algorithm to preserve semantic boundaries.
2.  **Storage Layer (SQLite):** Acts as the source of truth for all structured data. It maintains a relational schema mapping documents to their exact textual chunks.
3.  **Vector Layer (FAISS):** Manages the dense vector index. Instead of storing massive payloads, FAISS strictly handles L2 distance calculations, utilizing `IndexIDMap` to link vector clusters directly back to SQLite primary keys.
4.  **Generation Pipeline (PyTorch/Transformers):** Embeds user queries, retrieves the top-K vectors, fetches the raw text from SQLite via the returned IDs, and injects the context into a locally hosted language model for factual synthesis.

---

## 📐 Low-Level Design (LLD) & Engineering Principles

The codebase is structured around core software engineering paradigms, specifically adhering to **SOLID principles**:

* **Single Responsibility Principle (SRP):** * `DatabaseManager` exclusively handles SQLite connections, schema creation, and CRUD operations.
    * `DocumentIndexer` solely manages embedding generation and FAISS index mutations.
    * `RAGPipeline` orchestrates the retrieval and prompt generation.
* **Dependency Injection:** The `DatabaseManager` instance is injected into both the `DocumentIndexer` and `RAGPipeline` upon initialization. This prevents hardcoded database dependencies and makes unit testing significantly easier.
* **First-Principles Data Mapping:** Dense vectors in FAISS do not support payload storage by default. The system bridges this by converting the auto-incrementing `id` from the SQLite `chunks` table into a NumPy array (`int64`), inserting it into FAISS alongside the tensor array. Upon query retrieval, FAISS returns the integer ID, ensuring an O(1) primary key lookup in SQLite.

---

## 🛠️ Tech Stack & Constraints

* **Core Logic:** Python 3.x
* **Machine Learning:** PyTorch, Hugging Face `transformers`, `sentence-transformers`
* **Vector Search:** Facebook AI Similarity Search (FAISS) - configured for CPU/GPU fallback.
* **Relational Database:** SQLite3
* **Hardware Fallback:** The system dynamically evaluates `torch.cuda.is_available()` at runtime. It defaults to CUDA for tensor multiplications and model inference, but safely degrades to CPU execution if hardware constraints exist.

---

## 🚀 Installation & Setup

### 1. Prerequisites
Ensure you have Python 3.8+ installed. If you intend to use GPU acceleration, ensure your NVIDIA drivers and CUDA toolkit are configured.

### 2. Clone and Initialize
```bash
git clone [https://github.com/your-username/mbsqi-repo.git](https://github.com/your-username/mbsqi-repo.git)
cd mbsqi-repo

# Create and activate an isolated environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
