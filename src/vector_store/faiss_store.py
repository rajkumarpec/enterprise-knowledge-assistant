from pathlib import Path
from typing import List, Tuple
import pickle
import numpy as np
import faiss
from langchain_core.documents import Document


# =====================================================
# Build FAISS Index (Initial)
# =====================================================
def build_faiss_index(
    embeddings: np.ndarray
) -> faiss.IndexFlatL2:
    """
    Build a FAISS index using L2 distance.

    Parameters
    ----------
    embeddings : np.ndarray
        Embedding matrix (num_vectors × dim)

    Returns
    -------
    index : faiss.IndexFlatL2
    """

    if embeddings is None or len(embeddings) == 0:
        raise ValueError("❌ Cannot build FAISS index with empty embeddings")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    print(f"✅ FAISS index built with {index.ntotal} vectors")
    return index


# =====================================================
# Save Vector Store
# =====================================================
def save_vector_store(
    index: faiss.IndexFlatL2,
    documents: List[Document],
    save_path: Path
):
    """
    Save FAISS index and associated documents.

    Parameters
    ----------
    index : faiss.IndexFlatL2
        FAISS index
    documents : List[Document]
        Chunked LangChain documents (with metadata)
    save_path : Path
        Path to save vector store (.pkl)
    """

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    with open(save_path, "wb") as f:
        pickle.dump(
            {
                "faiss_index": index,
                "documents": documents
            },
            f
        )

    print(f"💾 Vector store saved at:\n{save_path}")


# =====================================================
# Load Vector Store
# =====================================================
def load_vector_store(
    save_path: Path
) -> Tuple[faiss.IndexFlatL2, List[Document]]:
    """
    Load FAISS index and documents from disk.
    """

    save_path = Path(save_path)

    if not save_path.exists():
        raise FileNotFoundError(f"❌ Vector store not found at: {save_path}")

    with open(save_path, "rb") as f:
        store = pickle.load(f)

    print(f"✅ Vector store loaded from:\n{save_path}")
    return store["faiss_index"], store["documents"]


# =====================================================
# Incremental Update (ADD NEW VECTORS)
# =====================================================
def add_to_faiss_index(
    index: faiss.IndexFlatL2,
    documents: List[Document],
    new_embeddings: np.ndarray,
    new_documents: List[Document]
):
    """
    Incrementally add new embeddings + documents to FAISS index.

    Parameters
    ----------
    index : faiss.IndexFlatL2
        Existing FAISS index
    documents : List[Document]
        Existing documents list
    new_embeddings : np.ndarray
        Embeddings of new chunks
    new_documents : List[Document]
        Corresponding new Document objects
    """

    if len(new_embeddings) != len(new_documents):
        raise ValueError("❌ Embeddings and documents count mismatch")

    index.add(new_embeddings)
    documents.extend(new_documents)

    print(f"➕ Added {len(new_documents)} new vectors to FAISS index")
    print(f"📊 Total vectors now: {index.ntotal}")
