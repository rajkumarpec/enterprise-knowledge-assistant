from typing import List, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain_core.documents import Document


def load_embedding_model(
    model_name: str = "all-MiniLM-L6-v2"
) -> SentenceTransformer:
    """
    Load a SentenceTransformer embedding model.

    NOTE:
    - Embedding model should remain FIXED across the project.
    - Changing this requires a full vector store rebuild.
    """
    return SentenceTransformer(model_name)


def generate_embeddings(
    documents: List[Document],
    model: SentenceTransformer,
    show_progress: bool = True
) -> np.ndarray:
    """
    Generate embeddings for LangChain Document chunks.

    Parameters
    ----------
    documents : List[Document]
        Chunked LangChain documents
    model : SentenceTransformer
        Loaded embedding model
    show_progress : bool
        Show progress bar during encoding

    Returns
    -------
    embeddings : np.ndarray
        Embedding matrix of shape (n_chunks, embedding_dim)
    """

    if not documents:
        raise ValueError("❌ No documents provided for embedding")

    # Safety check
    if not isinstance(documents[0], Document):
        raise TypeError(
            "❌ generate_embeddings expects List[Document]"
        )

    texts = [doc.page_content for doc in documents]

    embeddings = model.encode(
        texts,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
        normalize_embeddings=True  # 🔑 IMPORTANT for cosine similarity & FAISS
    )

    return embeddings
