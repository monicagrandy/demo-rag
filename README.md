# Class Notes RAG

Standalone Streamlit app for searching and chatting with markdown class notes.

This repo is public-safe by design:
- the app code lives here
- the checked-in `notes/` folder contains only synthetic sample notes
- your real notes can stay anywhere else on disk and be indexed by setting `CLASS_NOTES_DIR`

## What It Does

- hybrid retrieval with BM25 + vector search
- direct question answering over indexed notes
- browse full notes by date for dated note files
- optional agentic mode using a ReAct loop
- groundedness scoring for direct-answer mode

## Repo Layout

- `app.py`: Streamlit UI
- `ingest.py`: builds the Chroma and BM25 indices
- `retriever.py`: retrieval helpers
- `chain.py`: direct RAG answer chain
- `agent.py`: agentic tool-routing mode
- `evaluator.py`: groundedness scoring
- `notes/`: synthetic sample notes

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

To point the app at your real private notes instead, set:

```bash
export CLASS_NOTES_DIR="/absolute/path/to/your/notes"
```

The app will recursively index `*.md` files under that directory.

If your private notes live inside a larger course repo and you only want to index a subset such as `class_notes`, also set a glob filter:

```bash
export CLASS_NOTES_DIR="/absolute/path/to/private/course/repo"
export CLASS_NOTES_GLOB="Week_*/class_notes/*.md"
```

Date support:
- if a note contains a line like `Thu, 09 Apr 26`, that date is used
- otherwise the app falls back to filenames like `04-09-26.md`
- undated notes still work for Q&A, but they will not appear in browse-by-date mode

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

Use direct mode when you want a concise answer grounded in the indexed notes.

Sample questions:
- `How are embeddings used to retrieve relevant documents?`
- `What evaluation metrics are used for RAG?`
- `How does the ReAct loop work in agentic RAG?`

### Browse by Date

Use date mode when you want the full note text for a specific class or lecture date.

### Agentic Mode

Use agentic mode when a question may need several retrieval steps or tool decisions.

## Updating Notes

When you add or update notes:

1. Put the markdown files in your chosen notes directory.
2. Rerun:

```bash
.venv/bin/python ingest.py
```

3. Restart Streamlit if it is already running.

## Privacy Model

If you want this repo to stay public while your notes stay private:

- keep your real notes outside this repo
- set `CLASS_NOTES_DIR` to that private notes path
- do not copy private notes into `notes/`

The sample notes in this repo are only there so a new user can try the app immediately.
