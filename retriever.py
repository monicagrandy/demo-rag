"""
retriever.py — Hybrid retrieval layer for Demo RAG.

Provides:
  1. Hybrid retriever (BM25 keyword + ChromaDB semantic) via EnsembleRetriever
  2. Date-based full-notes retrieval for the "Browse by Date" mode
"""

import pickle
from pathlib import Path

from config import is_excluded_relative_path
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

APP_DIR = Path(__file__).resolve().parent
CHROMA_DIR = APP_DIR / "chroma_db"
BM25_PATH = APP_DIR / "bm25_corpus.pkl"

EMBEDDING_MODEL = "text-embedding-3-small"


def _document_is_excluded(doc: Document) -> bool:
    source_file = doc.metadata.get("source_file", "")
    return bool(source_file and is_excluded_relative_path(source_file))


def _filter_documents(docs: list[Document]) -> list[Document]:
    return [doc for doc in docs if not _document_is_excluded(doc)]


class FilteredRetriever(BaseRetriever):
    """Wrap another retriever and drop excluded source files from results."""

    base_retriever: BaseRetriever
    limit: int = 5

    def _get_relevant_documents(self, query: str, *, run_manager) -> list[Document]:
        docs = self.base_retriever.invoke(query)
        return _filter_documents(docs)[: self.limit]


# ---------------------------------------------------------------------------
# Internal loaders
# ---------------------------------------------------------------------------


def _load_bm25_documents() -> list[Document]:
    """Deserialize the BM25 corpus into LangChain Documents."""
    with open(BM25_PATH, "rb") as f:
        corpus_data = pickle.load(f)

    docs = [
        Document(page_content=item["page_content"], metadata=item["metadata"])
        for item in corpus_data
    ]
    return _filter_documents(docs)


def _get_vectorstore(api_key: str) -> Chroma:
    """Load the persisted ChromaDB vector store."""
    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        openai_api_key=api_key,
    )
    return Chroma(
        persist_directory=str(CHROMA_DIR),
        embedding_function=embeddings,
        collection_name="class_notes",
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_hybrid_retriever(
    api_key: str,
    k: int = 5,
    bm25_weight: float = 0.4,
    semantic_weight: float = 0.6,
) -> BaseRetriever:
    """
    Build a hybrid retriever combining BM25 (keyword) and ChromaDB (semantic).

    Args:
        api_key: OpenAI API key for the embedding model.
        k: Number of results to return.
        bm25_weight: Weight for keyword results in the ensemble.
        semantic_weight: Weight for semantic results in the ensemble.

    Returns:
        Filtered retriever ready for use in the app and chains.
    """
    search_k = max(k * 3, 10)

    # BM25 keyword retriever
    bm25_docs = _load_bm25_documents()
    bm25_retriever = BM25Retriever.from_documents(bm25_docs, k=search_k)

    # ChromaDB semantic retriever
    vectorstore = _get_vectorstore(api_key)
    semantic_retriever = vectorstore.as_retriever(
        search_type="mmr",  # Maximal Marginal Relevance for diversity
        search_kwargs={"k": search_k},
    )

    # Combine via EnsembleRetriever
    hybrid = EnsembleRetriever(
        retrievers=[bm25_retriever, semantic_retriever],
        weights=[bm25_weight, semantic_weight],
    )

    return FilteredRetriever(base_retriever=hybrid, limit=k)


def get_notes_by_date(api_key: str, date_str: str) -> list[Document]:
    """
    Retrieve all chunks for a given class date, sorted by chunk_index.

    Args:
        api_key: OpenAI API key.
        date_str: Date string, e.g. "2026-04-09".

    Returns:
        List of Documents sorted by (source_file, chunk_index).
    """
    vectorstore = _get_vectorstore(api_key)

    # Use metadata filter to get all chunks for this date
    results = vectorstore.get(
        where={"class_date": date_str},
        include=["documents", "metadatas"],
    )

    if not results["documents"]:
        return []

    docs = []
    for content, metadata in zip(results["documents"], results["metadatas"]):
        doc = Document(page_content=content, metadata=metadata)
        if _document_is_excluded(doc):
            continue
        docs.append(doc)

    # Sort by source file, then chunk index for proper reconstruction
    docs.sort(key=lambda d: (d.metadata.get("source_file", ""), d.metadata.get("chunk_index", 0)))

    return docs


def get_available_dates(api_key: str) -> list[str]:
    """
    Get all unique class dates available in the vector store.

    Returns:
        Sorted list of date strings (excluding 'reference').
    """
    vectorstore = _get_vectorstore(api_key)

    # Get all metadata
    results = vectorstore.get(include=["metadatas"])
    dates = set()
    for meta in results["metadatas"]:
        if is_excluded_relative_path(meta.get("source_file", "")):
            continue
        d = meta.get("class_date", "reference")
        if d != "reference":
            dates.add(d)

    return sorted(dates)


def reconstruct_notes(docs: list[Document]) -> dict[str, str]:
    """
    Reconstruct full notes from chunks, grouped by source document title.

    Args:
        docs: Sorted list of Documents (from get_notes_by_date).

    Returns:
        Dict mapping title → full reconstructed text.
    """
    grouped: dict[str, list[Document]] = {}
    for doc in docs:
        title = doc.metadata.get("title", "Untitled")
        if title not in grouped:
            grouped[title] = []
        grouped[title].append(doc)

    result = {}
    for title, chunks in grouped.items():
        # Chunks are already sorted; join with a small separator
        full_text = "\n\n".join(chunk.page_content for chunk in chunks)
        result[title] = full_text

    return result
