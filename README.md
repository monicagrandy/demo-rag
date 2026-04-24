# Demo RAG

Standalone Streamlit repo for indexing markdown notes and querying them with hybrid retrieval.

This repository is intentionally decoupled from any one course repo. It works in two modes:

- self-contained demo mode using the synthetic sample notes checked into this repo
- external-notes mode by pointing the app at any private notes directory on disk

That split is the point of the repo:
- the app can be public
- your real notes can stay private
- the same code can index notes from different courses or projects

## Core Idea

`demo-rag` is the reusable app layer only.

It does not assume:
- a specific course structure
- a `Week_*` folder layout
- an `agentic-ai-study-guide` checkout

If your notes do follow a course structure, you can still target only the subset you want with `CLASS_NOTES_GLOB`.

## What It Does

- hybrid retrieval with BM25 + vector search
- direct question answering over indexed markdown notes
- browse-by-date for notes that contain or encode a date
- optional agentic mode using a ReAct loop
- groundedness scoring for direct-answer mode

## Repo Layout

- `app.py`: Streamlit UI
- `ingest.py`: builds the Chroma and BM25 indices
- `retriever.py`: retrieval helpers
- `chain.py`: direct RAG answer chain
- `agent.py`: agentic tool-routing mode
- `evaluator.py`: groundedness scoring
- `notes/`: synthetic sample notes for local testing

## Setup

From the repo root:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Preferred auth setup:

```bash
export OPENAI_API_KEY="sk-..."
```

Optional Streamlit fallback:

```bash
cp .streamlit/secrets.example.toml .streamlit/secrets.toml
```

Then paste your key into `.streamlit/secrets.toml`.

## Notes Source

By default the app indexes markdown files under:

```text
notes/
```

That is enough to test the app immediately after cloning.

The checked-in notes can include:

- `notes/sample/` for synthetic demo content
- `notes/agentic-ai/` for shareable class notes mirrored from a separate study workspace

To point the app at your real private notes instead:

```bash
export CLASS_NOTES_DIR="/absolute/path/to/your/notes"
```

The app will recursively index `*.md` files under that directory.

If your notes live inside a larger private repo and you only want a subset, also set a glob filter:

```bash
export CLASS_NOTES_DIR="/absolute/path/to/private/repo"
export CLASS_NOTES_GLOB="Week_*/class_notes/*.md"
```

Examples of useful glob filters:

```bash
export CLASS_NOTES_GLOB="**/*.md"
export CLASS_NOTES_GLOB="Week_*/class_notes/*.md"
export CLASS_NOTES_GLOB="lectures/*.md,review/*.md"
```

## Date Handling

Browse-by-date uses whichever date signal it can find first:

1. a line inside the note body like `Thu, 09 Apr 26`
2. an ISO-style filename prefix like `2026-04-15-topic.md`
3. a short filename like `04-09-26.md`

Undated notes still work for search and Q&A, but they will not appear in the date dropdown.

## First Run

Build the local index:

```bash
.venv/bin/python ingest.py
```

Start the app:

```bash
.venv/bin/streamlit run app.py
```

Generated local artifacts:

- `chroma_db/`
- `bm25_corpus.pkl`

Those files are gitignored.

## How To Use It

### Direct Q&A

Use direct mode when you want a concise answer grounded in retrieved notes.

Sample questions:

- `How are embeddings used to retrieve relevant documents?`
- `What evaluation metrics are used for RAG?`
- `How does the ReAct loop work in agentic RAG?`

### Browse by Date

Use date mode when you want the full note text for a dated class or lecture session.

### Agentic Mode

Use agentic mode when a question may benefit from several retrieval steps or tool decisions.

## Updating Notes

When you add or update notes:

1. Put the markdown files in your chosen notes directory.
2. Rerun:

```bash
.venv/bin/python ingest.py
```

3. Restart Streamlit if it is already running.

## Public / Private Split

If you want this repo to stay public while your notes stay private:

- keep your real notes outside this repo
- set `CLASS_NOTES_DIR` to that private notes path
- optionally narrow the indexed files with `CLASS_NOTES_GLOB`
- do not copy private notes into `notes/`

The checked-in `notes/` folder may contain synthetic demo notes and any shareable public notes you intentionally choose to publish. Keep private notes outside this repo.
