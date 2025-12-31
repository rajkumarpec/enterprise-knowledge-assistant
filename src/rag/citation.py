"""
citation.py
-------------
Utilities for citation-aware retrieval and prompt construction
for RAG pipelines.
"""

from typing import List, Dict
import faiss
from sentence_transformers import SentenceTransformer
from langchain_core.documents import Document


def retrieve_with_citations(
    query: str,
    model: SentenceTransformer,
    index: faiss.IndexFlatL2,
    documents: List[Document],
    top_k: int = 4
) -> List[Dict]:
    """
    Retrieve top-k relevant document chunks with citation IDs.

    Returns:
        List of dicts containing:
        - citation_id
        - content
        - distance
        - metadata
    """

    query_embedding = model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, top_k)

    results = []
    for rank, (idx, dist) in enumerate(zip(indices[0], distances[0]), start=1):
        results.append({
            "citation_id": f"[{rank}]",
            "content": documents[idx].page_content,
            "distance": float(dist),
            "metadata": documents[idx].metadata
        })

    return results


def build_rag_prompt_with_citations(
    query: str,
    retrieved_chunks: List[Dict]
) -> str:
    """
    Build a RAG prompt that enforces citation-based answers.
    """

    context_blocks = []
    for chunk in retrieved_chunks:
        context_blocks.append(
            f"{chunk['citation_id']} {chunk['content']}"
        )

    context_text = "\n\n".join(context_blocks)

    prompt = f"""
You are a knowledgeable assistant.
Answer the question using ONLY the context below.
Each factual statement MUST include citation IDs like [1], [2].
If the answer is not present in the context, say "I don't know".

Context:
{context_text}

Question:
{query}

Answer (with citations):
"""
    return prompt.strip()


def format_citation_sources(
    retrieved_chunks: List[Dict],
    max_chars: int = 400
) -> str:
    """
    Nicely format citation sources for display/logging.
    """

    output = []
    for chunk in retrieved_chunks:
        text = chunk["content"][:max_chars]
        output.append(
            f"{chunk['citation_id']} (distance={chunk['distance']:.4f})\n{text}"
        )

    return "\n\n" + "-" * 80 + "\n\n".join(output)
