# Demo RAG

Standalone Streamlit repo for indexing markdown files and querying them with hybrid retrieval.

## What It Does

- hybrid retrieval with BM25 + vector search
- direct question answering over indexed markdown notes
- browse-by-date for notes that contain or encode a date
- optional agentic mode using a ReAct loop
- groundedness scoring for direct-answer mode
- Presidio-based PII redaction during ingestion and on generated answers
- benchmark runner for retrieval, groundedness, and PII redaction checks

## Repo Layout

- `app.py`: Streamlit UI
- `ingest.py`: builds the Chroma and BM25 indices
- `retriever.py`: retrieval helpers
- `chain.py`: direct RAG answer chain
- `agent.py`: agentic tool-routing mode
- `evaluator.py`: groundedness plus retrieval metric helpers
- `benchmarks/`: benchmark runner plus sample benchmark fixtures
- `notes/`: synthetic sample notes for local testing

## Setup

From the repo root:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
python -m spacy download en_core_web_sm
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

## PII Redaction

PII redaction is enabled by default during ingestion and on generated answers. The app uses Microsoft Presidio to replace detected entities with placeholders such as `<EMAIL_ADDRESS>`, `<PHONE_NUMBER>`, and `<PERSON>`.

The default setup is intentionally conservative for note corpora:

- structured PII stays enabled
- checked-in known names are matched through a custom Presidio recognizer
- date-like note filenames such as `04-09-26.md` and `04-26-2026` are preserved instead of being misread as phone numbers

Useful environment variables:

```bash
export PII_REDACTION_ENABLED=true
export PII_REDACTION_ENTITY_TYPES="KNOWN_PERSON,PHONE_NUMBER,EMAIL_ADDRESS,CREDIT_CARD,IP_ADDRESS,US_SSN"
export PII_REDACTION_SPACY_MODEL="en_core_web_sm"
export PII_REDACTION_SCORE_THRESHOLD="0.35"
```

Redaction happens before notes are embedded and again before direct or agentic answers are shown in the UI.

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
export CLASS_NOTES_GLOB="Week_*/class_notes/*.md,Week_*/class_notes.md/*.md,week_*/class_notes/*.md"
```

Examples of useful glob filters:

The app accepts comma-separated glob patterns, which is useful when a course mixes folder names such as `class_notes/` and `class_notes.md/`.

```bash
export CLASS_NOTES_GLOB="**/*.md"
export CLASS_NOTES_GLOB="Week_*/class_notes/*.md,Week_*/class_notes.md/*.md,week_*/class_notes/*.md"
export CLASS_NOTES_GLOB="lectures/*.md,review/*.md"
```

## Date Handling

Browse-by-date uses whichever date signal it can find first:

1. a line inside the note body like `Thu, 09 Apr 26`
2. an ISO-style filename prefix like `2026-04-15-topic.md`
3. a short filename like `04-09-26.md` or `05_17_26_notes.md`

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


## Benchmarks

Default benchmark fixtures live in `benchmarks/*.json`. Run the full suite after indexing notes:

```bash
.venv/bin/python benchmarks/run_benchmarks.py
```

This runs:

- retrieval precision and recall against labeled source files
- groundedness checks for direct-answer responses
- PII redaction checks for the Presidio pipeline

Optional output:

```bash
.venv/bin/python benchmarks/run_benchmarks.py --json-output benchmark-report.json
```

The benchmark runner uses the existing Chroma/BM25 artifacts, so rerun `ingest.py` before benchmarking newly added notes.

## Updating Notes

When you add or update notes:

1. Put the markdown files in your chosen notes directory.
2. If you are publishing checked-in notes from this repo, rerun:

```bash
.venv/bin/python scripts/sanitize_notes.py
```

3. Rebuild the local index:

```bash
.venv/bin/python ingest.py
```

4. Restart Streamlit if it is already running.

## Public / Private Split

If you want this repo to stay public while your notes stay private:

- keep your real notes outside this repo
- set `CLASS_NOTES_DIR` to that private notes path
- optionally narrow the indexed files with `CLASS_NOTES_GLOB`
- do not copy private notes into `notes/`

The checked-in `notes/` folder may contain synthetic demo notes and any shareable public notes you intentionally choose to publish. Keep private notes outside this repo.
