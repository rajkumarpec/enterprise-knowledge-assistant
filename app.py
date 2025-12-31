import sys
import os
from pathlib import Path
import subprocess

import streamlit as st
from sentence_transformers import SentenceTransformer

import config
from src.vector_store.faiss_store import load_vector_store
from src.rag.citation import (
    retrieve_with_citations,
    build_rag_prompt_with_citations,
    format_citation_sources
)

# -------------------------------
# App Setup
# -------------------------------
st.set_page_config(
    page_title="Enterprise Knowledge Assistant",
    page_icon="📚",
    layout="wide"
)

st.title("📚 Enterprise Knowledge Assistant")
st.caption("Local RAG with FAISS + Ollama LLMs + Citations")

# -------------------------------
# Load Models (Cached)
# -------------------------------
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(config.EMBEDDING_MODEL_NAME)

@st.cache_resource
def load_vector_db():
    return load_vector_store(config.VECTOR_STORE_PATH)

embedding_model = load_embedding_model()
faiss_index, documents = load_vector_db()

# -------------------------------
# Ollama Call (MODEL-AGNOSTIC)
# -------------------------------
def call_llm(prompt: str, model_name: str) -> str:
    result = subprocess.run(
        ["ollama", "run", model_name],
        input=prompt.encode("utf-8"),
        capture_output=True
    )
    return result.stdout.decode("utf-8", errors="ignore").strip()

# -------------------------------
# Sidebar Controls
# -------------------------------
st.sidebar.header("⚙️ Settings")

# 🔹 Model selector
selected_model = st.sidebar.selectbox(
    "Select LLM Model",
    options=config.AVAILABLE_MODELS,
    index=config.AVAILABLE_MODELS.index(config.DEFAULT_UI_MODEL)
)

# 🔹 Retrieval depth
top_k = st.sidebar.slider(
    "Top-K Retrieved Chunks",
    min_value=1,
    max_value=10,
    value=config.TOP_K
)

st.sidebar.markdown("---")
st.sidebar.caption(
    "💡 Tip:\n"
    "- Use **phi-3:mini** for fast UI\n"
    "- Use **llama3.2:3b** for deeper reasoning"
)

# -------------------------------
# Main UI
# -------------------------------
query = st.text_input(
    "Ask a question from your documents:",
    placeholder="e.g. How does surface roughness affect heat transfer?"
)

if st.button("🔍 Ask") and query.strip():

    with st.spinner("🔎 Retrieving relevant context..."):
        retrieved_chunks = retrieve_with_citations(
            query=query,
            model=embedding_model,
            index=faiss_index,
            documents=documents,
            top_k=top_k
        )

    with st.spinner(f"🧠 Generating answer using `{selected_model}` ..."):
        prompt = build_rag_prompt_with_citations(query, retrieved_chunks)
        answer = call_llm(prompt, selected_model)

    # -------------------------------
    # Display Answer
    # -------------------------------
    st.subheader("🧠 Answer")
    st.write(answer)

    # -------------------------------
    # Display Sources
    # -------------------------------
    st.subheader("📚 Sources")
    st.text(format_citation_sources(retrieved_chunks))

    # -------------------------------
    # Debug (Optional)
    # -------------------------------
    with st.expander("🔎 View Prompt (Debug)"):
        st.code(prompt)
