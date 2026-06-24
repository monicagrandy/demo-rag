import os
from typing import Iterable

from langchain_core.documents import Document
from openai import OpenAI


def extract_unique_source_files(docs: Iterable[Document]) -> list[str]:
    """Deduplicate retrieved chunks down to stable source-file identifiers."""
    seen: set[str] = set()
    source_files: list[str] = []
    for doc in docs:
        source_file = doc.metadata.get("source_file", "")
        if not source_file or source_file in seen:
            continue
        seen.add(source_file)
        source_files.append(source_file)
    return source_files


def score_retrieval_precision_recall(
    retrieved_source_files: list[str],
    relevant_source_files: list[str],
) -> dict:
    """Compute source-level precision and recall for a retrieval result set."""
    retrieved = list(dict.fromkeys(item for item in retrieved_source_files if item))
    relevant = list(dict.fromkeys(item for item in relevant_source_files if item))
    relevant_set = set(relevant)

    matched = [source for source in retrieved if source in relevant_set]
    false_positives = [source for source in retrieved if source not in relevant_set]
    missing = [source for source in relevant if source not in set(matched)]

    precision = len(matched) / len(retrieved) if retrieved else 0.0
    recall = len(matched) / len(relevant) if relevant else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "matched_source_files": matched,
        "false_positive_source_files": false_positives,
        "missing_source_files": missing,
    }


def score_groundedness(question: str, context: str, answer: str, api_key: str) -> dict:
    """
    Groundedness (Faithfulness): Is the answer supported by the retrieved context?
    Uses LLM-as-Judge with temperature=0 for reproducible scores.

    Returns: {"score": float 0.0-1.0, "rationale": str}
    """
    client = OpenAI(api_key=api_key, timeout=30.0, max_retries=2)
    chat_model = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")

    prompt = f"""Evaluate if the ANSWER is fully supported by the CONTEXT. Check for hallucinations or unsupported claims.

QUESTION:
{question}

CONTEXT:
{context}

ANSWER:
{answer}

Scoring rubric:
- 1.0 = every claim is directly supported by the context
- 0.5 = most claims are supported, minor details are inferred
- 0.0 = one or more claims contradict or are absent from the context

Respond in exactly this format:
Score: <number between 0.0 and 1.0>
Reasoning: <one sentence explaining your score>"""

    try:
        response = client.chat.completions.create(
            model=chat_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            timeout=30,
        )
        output = response.choices[0].message.content

        score = 0.5
        rationale = "Could not parse rationale."
        for line in output.strip().splitlines():
            if line.startswith("Score:"):
                try:
                    score = float(line.replace("Score:", "").strip())
                except ValueError:
                    score = 0.5
            elif line.startswith("Reasoning:"):
                rationale = line.replace("Reasoning:", "").strip()

        return {"score": score, "rationale": rationale}

    except Exception as exc:
        print(f"⚠ Groundedness evaluation error: {str(exc)[:80]}")
        return {"score": 0.5, "rationale": f"Evaluation error: {str(exc)[:80]}"}
