import os
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader


def ingest_pdf(
    pdf_path,
    preview_chars=500,
    verbose=True,
    show_preview=True
):
    """
    STEP-1: Document Ingestion for RAG (Multi-PDF Ready)

    Parameters
    ----------
    pdf_path : str or Path
        Path to PDF file
    preview_chars : int
        Number of characters to preview
    verbose : bool
        Print logs
    show_preview : bool
        Show metadata and text preview

    Returns
    -------
    documents : list
        List of LangChain Document objects with enriched metadata
    """

    pdf_path = Path(pdf_path)

    # --- File check ---
    if not pdf_path.exists():
        raise FileNotFoundError(f"❌ PDF not found at: {pdf_path}")

    if verbose:
        print(f"✅ Loading PDF: {pdf_path.name}")

    # --- Load PDF ---
    loader = PyPDFLoader(str(pdf_path))
    documents = loader.load()

    if verbose:
        print(f"📄 Total pages loaded: {len(documents)}")

    # --- 🔑 Enrich metadata for multi-PDF support ---
    for doc in documents:
        # Standardize metadata keys
        doc.metadata["source"] = pdf_path.name          # PDF filename
        doc.metadata["file_path"] = str(pdf_path)       # Full path
        doc.metadata["page"] = doc.metadata.get("page", None)

    # --- Optional inspection ---
    if show_preview and documents:
        first_doc = documents[0]

        print("\n🧾 Metadata (Page 1):")
        print(first_doc.metadata)

        print("\n📃 Page Content Preview:\n")
        print(first_doc.page_content[:preview_chars])

        full_text = "\n".join(doc.page_content for doc in documents)
        print(f"\n📊 Total characters in document: {len(full_text)}")

    # ✅ RETURN DOCUMENTS WITH METADATA
    return documents
