from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


def chunk_documents(
    documents,
    chunk_size=1000,
    chunk_overlap=200,
    preview_chunks=3,
    verbose=True
):
    """
    STEP-2: Chunk documents for RAG (Multi-PDF Safe)

    Parameters
    ----------
    documents : list[Document]
        List of LangChain Document objects
    chunk_size : int
        Size of each chunk
    chunk_overlap : int
        Overlap between chunks
    preview_chunks : int
        Number of chunks to preview
    verbose : bool
        Print logs and previews

    Returns
    -------
    chunks : list[Document]
        Chunked LangChain Document objects with preserved metadata
    """

    # 🔒 HARD SAFETY CHECKS (NO SILENT FAILURES)
    if not isinstance(documents, list):
        raise TypeError("❌ documents must be a list of LangChain Document objects")

    # Flatten nested lists if any
    if documents and isinstance(documents[0], list):
        documents = [doc for sublist in documents for doc in sublist]

    # Validate document type
    if documents and not isinstance(documents[0], Document):
        raise TypeError(
            "❌ chunk_documents received invalid input.\n"
            "Expected: List[Document]\n"
            f"Received: List[{type(documents[0])}]"
        )

    if verbose:
        print("✂️ Chunking started")
        print(f"Chunk size     : {chunk_size}")
        print(f"Chunk overlap  : {chunk_overlap}")
        print(f"Total docs in  : {len(documents)}")

    # --- Text splitter ---
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )

    # --- Split documents ---
    chunks = splitter.split_documents(documents)

    # 🔑 ENSURE METADATA IS PRESERVED & NORMALIZED
    for chunk in chunks:
        chunk.metadata = chunk.metadata.copy()

        # Safety defaults (for multi-PDF usage)
        chunk.metadata.setdefault("source", "unknown")
        chunk.metadata.setdefault("page", None)
        chunk.metadata.setdefault("file_path", None)

    if verbose:
        print(f"📦 Total chunks created: {len(chunks)}")

        for i in range(min(preview_chunks, len(chunks))):
            print(f"\n🔹 CHUNK {i + 1}")
            print("Metadata:", chunks[i].metadata)
            print(chunks[i].page_content[:400])

    return chunks
