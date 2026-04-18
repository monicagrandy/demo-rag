"""Offline indexing pipeline for the standalone Demo RAG app."""

import os
import pickle
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

from config import get_notes_root, get_source_specs

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CHROMA_DIR = Path(__file__).resolve().parent / "chroma_db"
BM25_PATH = Path(__file__).resolve().parent / "bm25_corpus.pkl"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

EMBEDDING_MODEL = "text-embedding-3-small"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_documents() -> list[Document]:
    """Load all source markdown files into LangChain Documents with metadata."""
    documents = []
    for src in get_source_specs():
        filepath = src["path"]
        notes_root = src["notes_root"]
        if not filepath.exists():
            print(f"  ⚠  Skipping missing file: {filepath}")
            continue

        text = filepath.read_text(encoding="utf-8").strip()
        if not text:
            print(f"  ⚠  Skipping empty file: {filepath.name}")
            continue

        doc = Document(
            page_content=text,
            metadata={
                "source_file": str(filepath.relative_to(notes_root)),
                "collection": src["collection"],
                "title": src["title"],
                "class_date": src["class_date"] or "reference",
            },
        )
        documents.append(doc)
        print(f"  ✓  Loaded {filepath.name} ({len(text):,} chars)")

    return documents


def chunk_documents(documents: list[Document]) -> list[Document]:
    """Split documents into chunks using RecursiveCharacterTextSplitter."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n## ", "\n### ", "\n#### ", "\n\n", "\n", ". ", " ", ""],
        keep_separator=True,
    )

    all_chunks = []
    for doc in documents:
        chunks = splitter.split_documents([doc])
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = i
            chunk.metadata["total_chunks"] = len(chunks)
        all_chunks.extend(chunks)

    return all_chunks


def build_vector_store(chunks: list[Document], api_key: str) -> Chroma:
    """Build and persist ChromaDB vector store."""
    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        openai_api_key=api_key,
    )

    # Remove existing DB to rebuild cleanly
    if CHROMA_DIR.exists():
        import shutil
        shutil.rmtree(CHROMA_DIR)

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(CHROMA_DIR),
        collection_name="class_notes",
    )

    return vectorstore


def build_bm25_corpus(chunks: list[Document]):
    """Serialize chunk data for BM25 retrieval at query time."""
    corpus_data = []
    for chunk in chunks:
        corpus_data.append({
            "page_content": chunk.page_content,
            "metadata": chunk.metadata,
        })

    with open(BM25_PATH, "wb") as f:
        pickle.dump(corpus_data, f)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    # Read API key from .streamlit/secrets.toml or environment
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        secrets_path = Path(__file__).resolve().parent / ".streamlit" / "secrets.toml"
        if secrets_path.exists():
            import tomllib
            try:
                with open(secrets_path, "rb") as f:
                    secrets = tomllib.load(f)
            except tomllib.TOMLDecodeError:
                print("❌ ERROR: Invalid .streamlit/secrets.toml format.")
                print('   Expected: OPENAI_API_KEY = "sk-..."')
                return
            api_key = secrets.get("OPENAI_API_KEY")

    if not api_key or api_key == "your-openai-api-key-here":
        print("❌ ERROR: Copy .streamlit/secrets.example.toml to .streamlit/secrets.toml and set OPENAI_API_KEY, or use an environment variable.")
        return

    print("=" * 60)
    notes_root = get_notes_root()

    print("📚 Demo RAG — Ingestion Pipeline")
    print("=" * 60)
    print(f"   App root: {Path(__file__).resolve().parent}")
    print(f"   Notes root: {notes_root}")
    print("   Source discovery: **/*.md under the configured notes root")

    # 1. Load
    print("\n📂 Loading documents...")
    documents = load_documents()
    print(f"\n   Loaded {len(documents)} documents")

    # 2. Chunk
    print("\n✂️  Chunking documents...")
    chunks = chunk_documents(documents)
    print(f"   Created {len(chunks)} chunks (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")

    # 3. Embed & store in ChromaDB
    print("\n🔢 Building vector store (ChromaDB)...")
    vectorstore = build_vector_store(chunks, api_key)
    print(f"   Stored {vectorstore._collection.count()} vectors in {CHROMA_DIR}")

    # 4. Build BM25 keyword corpus
    print("\n🔤 Building BM25 keyword index...")
    build_bm25_corpus(chunks)
    print(f"   Saved BM25 corpus to {BM25_PATH}")

    # Summary
    print("\n" + "=" * 60)
    print("✅ Ingestion complete!")
    print(f"   • {len(documents)} source documents")
    print(f"   • {len(chunks)} chunks indexed")
    print(f"   • Vector DB: {CHROMA_DIR}")
    print(f"   • BM25 index: {BM25_PATH}")

    # Show date-tagged content
    dates = sorted(set(
        d.metadata["class_date"]
        for d in documents
        if d.metadata["class_date"] != "reference"
    ))
    print(f"   • Dated notes available: {', '.join(dates) if dates else 'none'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
