"""Offline indexing pipeline for the standalone Demo RAG app."""

import pickle
import shutil
import tempfile
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import AuthenticationError

from config import get_notes_root, get_source_specs
from privacy import pii_redaction_enabled, redact_text
from runtime import resolve_openai_api_key

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

        redacted_body = redact_text(text)
        redacted_title = redact_text(src["title"])
        entity_types = sorted(set(redacted_body.entity_types + redacted_title.entity_types))
        entity_count = redacted_body.entity_count + redacted_title.entity_count

        doc = Document(
            page_content=redacted_body.text,
            metadata={
                "source_file": str(filepath.relative_to(notes_root)),
                "collection": src["collection"],
                "title": redacted_title.text,
                "class_date": src["class_date"] or "reference",
                "pii_redaction_applied": pii_redaction_enabled(),
                "pii_redaction_count": entity_count,
                "pii_redaction_entities": ",".join(entity_types),
            },
        )
        documents.append(doc)

        if entity_count:
            print(
                f"  🔐 Loaded {filepath.name} with {entity_count} redactions "
                f"({', '.join(entity_types)})"
            )
        else:
            print(f"  ✓  Loaded {filepath.name} ({len(redacted_body.text):,} chars)")

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


def replace_directory(source_dir: Path, target_dir: Path) -> None:
    """Swap a fully-built directory into place without deleting the current index first."""
    backup_dir = target_dir.with_name(f"{target_dir.name}_backup")
    if backup_dir.exists():
        shutil.rmtree(backup_dir)

    if not target_dir.exists():
        source_dir.replace(target_dir)
        return

    try:
        target_dir.replace(backup_dir)
        source_dir.replace(target_dir)
    except Exception:
        if backup_dir.exists() and not target_dir.exists():
            backup_dir.replace(target_dir)
        raise
    else:
        shutil.rmtree(backup_dir, ignore_errors=True)


def build_vector_store(chunks: list[Document], api_key: str) -> Chroma:
    """Build and persist ChromaDB vector store."""
    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        openai_api_key=api_key,
    )
    build_dir = Path(tempfile.mkdtemp(prefix="chroma-build-", dir=CHROMA_DIR.parent))

    try:
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=str(build_dir),
            collection_name="class_notes",
        )
        replace_directory(build_dir, CHROMA_DIR)
    except Exception:
        shutil.rmtree(build_dir, ignore_errors=True)
        raise

    return Chroma(
        persist_directory=str(CHROMA_DIR),
        embedding_function=embeddings,
        collection_name="class_notes",
    )


def build_bm25_corpus(chunks: list[Document]):
    """Serialize chunk data for BM25 retrieval at query time."""
    corpus_data = []
    for chunk in chunks:
        corpus_data.append(
            {
                "page_content": chunk.page_content,
                "metadata": chunk.metadata,
            }
        )

    with open(BM25_PATH, "wb") as handle:
        pickle.dump(corpus_data, handle)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    api_key = resolve_openai_api_key()
    if not api_key:
        print(
            "❌ ERROR: Copy .streamlit/secrets.example.toml to .streamlit/secrets.toml "
            "and set OPENAI_API_KEY, or use an environment variable."
        )
        return

    print("=" * 60)
    notes_root = get_notes_root()

    print("📚 Demo RAG — Ingestion Pipeline")
    print("=" * 60)
    print(f"   App root: {Path(__file__).resolve().parent}")
    print(f"   Notes root: {notes_root}")
    print("   Source discovery: **/*.md under the configured notes root")
    print(f"   PII redaction: {'enabled' if pii_redaction_enabled() else 'disabled'}")

    print("\n📂 Loading documents...")
    try:
        documents = load_documents()
    except RuntimeError as exc:
        print(f"❌ ERROR: {exc}")
        return
    print(f"\n   Loaded {len(documents)} documents")

    print("\n✂️  Chunking documents...")
    chunks = chunk_documents(documents)
    print(f"   Created {len(chunks)} chunks (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")

    print("\n🔢 Building vector store (ChromaDB)...")
    try:
        vectorstore = build_vector_store(chunks, api_key)
    except AuthenticationError as exc:
        print("❌ ERROR: OpenAI authentication failed. The existing Chroma index was left unchanged.")
        if "not_authorized_invalid_project" in str(exc) or "archived" in str(exc):
            print("   The API key or OpenAI project points to an archived project.")
            print("   Use a key from an active project, or unset OPENAI_PROJECT_ID and try again.")
        else:
            print("   Check OPENAI_API_KEY plus any OPENAI_PROJECT_ID or OPENAI_ORG_ID settings.")
        print(f"   API response: {exc}")
        return
    print(f"   Stored {vectorstore._collection.count()} vectors in {CHROMA_DIR}")

    print("\n🔤 Building BM25 keyword index...")
    build_bm25_corpus(chunks)
    print(f"   Saved BM25 corpus to {BM25_PATH}")

    print("\n" + "=" * 60)
    print("✅ Ingestion complete!")
    print(f"   • {len(documents)} source documents")
    print(f"   • {len(chunks)} chunks indexed")
    print(f"   • Vector DB: {CHROMA_DIR}")
    print(f"   • BM25 index: {BM25_PATH}")

    dates = sorted(
        set(
            d.metadata["class_date"]
            for d in documents
            if d.metadata["class_date"] != "reference"
        )
    )
    print(f"   • Dated notes available: {', '.join(dates) if dates else 'none'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
