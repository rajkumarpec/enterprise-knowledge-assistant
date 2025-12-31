## 📚 Enterprise Knowledge Assistant
Local Multi-Document RAG System with Evaluation & Citations

🔹 Overview

This project implements a fully local Retrieval-Augmented Generation (RAG) system that answers user questions from multiple PDF documents.
It combines semantic search (FAISS), local LLMs via Ollama, citation-aware answers, and evaluation metrics to ensure reliability and trust.

The system follows production-style modular architecture and supports incremental document updates without rebuilding the entire vector store.

## 🔹 Key Capabilities

✅ Multi-PDF ingestion with document & page metadata

✅ Robust text chunking with overlap

✅ Sentence-Transformer embeddings

✅ FAISS vector store (initial + incremental indexing)

✅ Citation-aware RAG responses

✅ Faithfulness & retrieval relevance evaluation

✅ Local LLM inference using Ollama

✅ Streamlit UI with model switching

✅ Fully offline & privacy-preserving

## 🔹 Architecture

PDF Documents
     ↓
Ingestion (metadata enrichment)
     ↓
Chunking
     ↓
Embeddings (Sentence Transformers)
     ↓
FAISS Vector Store
     ↓
Retriever
     ↓
LLM (Ollama)
     ↓
Answer + Citations
     ↓
Evaluation (Faithfulness & Relevance)


## 🔹 Tech Stack

Language: Python

Embeddings: Sentence-Transformers (all-MiniLM-L6-v2)

Vector DB: FAISS

LLMs: LLaMA 3.2 / Phi-3 Mini (via Ollama)

Frameworks: LangChain

UI: Streamlit

## 🔹 Project Structure

src/
 ├── ingestion/        # PDF loading & metadata enrichment
 ├── chunking/         # Text chunking logic
 ├── embeddings/       # Embedding generation
 ├── vector_store/     # FAISS build, save, load, update
 ├── rag/              # Retrieval, prompt, citations
data/
 ├── documents/        # Input PDFs
 ├── processed/        # Saved vector store
notebooks/
 ├── ingestion
 ├── chunking
 ├── embeddings
 ├── faiss
 ├── evaluation
app.py                 # Streamlit UI
config.py              # Central configuration


## 🔹 How to Run
### 1️⃣ Install dependencies

pip install -r requirements.txt

### 2️⃣ Run Ollama (local LLM)

ollama run llama3.2:3b

### 3️⃣ Build vector store (initial)

Run notebooks in order:

Ingestion

Chunking

Embeddings

FAISS initial build

### 4️⃣ Launch UI
streamlit run app.py


## 🔹 Incremental Document Update

New PDFs can be added without rebuilding the entire index:

Add PDF to data/documents/

Run incremental FAISS update notebook

Restart Streamlit

## 🔹 Evaluation & Trust

The system includes an evaluation layer to reduce hallucinations:

Citation coverage – verifies source grounding

Faithfulness score – semantic similarity between answer and retrieved context

Retrieval relevance – query–chunk similarity scores

## 🔹 Privacy & Security

Fully local execution

No cloud APIs

Documents never leave the machine

Suitable for confidential or institutional documents.

## 🔹 Status

✔ Complete
✔ Resume-ready
✔ Interview-ready

🔹 Author

Raj Kumar
Assistant Professor, G D Goenka University
Data Science | Machine Learning | GenAI