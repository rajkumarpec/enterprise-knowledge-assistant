# 🔍 Enterprise Knowledge Assistant (RAG + FAISS)

This project implements a **Retrieval-Augmented Generation (RAG)** system using:
- Local sentence-transformer embeddings
- FAISS vector search
- Modular ingestion and chunking
- Pure Python (no LangChain abstractions)

## 📂 Project Structure

rag-project/
├── notebooks/
├── src/
├── data/
└── requirements.txt


## ⚙️ Tech Stack
- Python
- Sentence-Transformers
- FAISS
- Scikit-learn
- LangChain (Documents only)

## 🚀 How to Run
1. Install dependencies  
   `pip install -r requirements.txt`

2. Run notebooks in order:
   - `01_ingestion_chunking.ipynb`
   - `02_embeddings.ipynb`
   - `03_vector_store_faiss.ipynb`

## 🧠 Key Concepts
- Semantic embeddings
- Vector similarity search
- Retrieval-Augmented Generation (RAG)

## 📌 Status
- RAG retrieval pipeline completed
- LLM integration in progress
