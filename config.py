# config.py
from pathlib import Path

# =========================================================
# Project Paths
# =========================================================
PROJECT_ROOT = Path(__file__).resolve().parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw_docs"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

VECTOR_STORE_PATH = PROCESSED_DATA_DIR / "vector_store.pkl"


# =========================================================
# Embedding Model (DO NOT CHANGE frequently)
# =========================================================
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"


# =========================================================
# LLM Models (Generation Only)
# =========================================================
# Available local Ollama models
AVAILABLE_MODELS = [
    "phi-3:mini",     # Fast, low-latency (UI)
    "llama3.2:3b"     # Better reasoning (analysis)
]

# Default model for Streamlit UI
DEFAULT_UI_MODEL = "phi-3:mini"

# Optional: default model for offline analysis / notebooks
DEFAULT_ANALYSIS_MODEL = "llama3.2:3b"


# =========================================================
# Retrieval Settings
# =========================================================
TOP_K = 4



