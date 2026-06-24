"""Run retrieval, groundedness, and PII-redaction benchmarks."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chain import build_rag_chain, format_docs
from evaluator import (
    extract_unique_source_files,
    score_groundedness,
    score_retrieval_precision_recall,
)
from privacy import pii_redaction_enabled, redact_text
from retriever import get_hybrid_retriever
from runtime import resolve_openai_api_key

DEFAULT_RETRIEVAL_CASES = REPO_ROOT / "benchmarks" / "retrieval_cases.json"
DEFAULT_QA_CASES = REPO_ROOT / "benchmarks" / "qa_cases.json"
DEFAULT_PII_CASES = REPO_ROOT / "benchmarks" / "pii_cases.json"


def load_cases(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def mean(values: list[float]) -> float:
    return statistics.mean(values) if values else 0.0


def run_retrieval_benchmarks(api_key: str, cases: list[dict], k: int) -> dict:
    retriever = get_hybrid_retriever(api_key, k=k)
    results = []
    for case in cases:
        docs = retriever.invoke(case["question"])
        retrieved_source_files = extract_unique_source_files(docs)
        metrics = score_retrieval_precision_recall(
            retrieved_source_files,
            case["relevant_source_files"],
        )
        passed = True
        if "min_precision" in case:
            passed = passed and metrics["precision"] >= case["min_precision"]
        if "min_recall" in case:
            passed = passed and metrics["recall"] >= case["min_recall"]

        results.append(
            {
                "id": case["id"],
                "question": case["question"],
                "retrieved_source_files": retrieved_source_files,
                **metrics,
                "passed": passed,
            }
        )

    return {
        "cases": results,
        "avg_precision": mean([case["precision"] for case in results]),
        "avg_recall": mean([case["recall"] for case in results]),
        "all_passed": all(case["passed"] for case in results),
    }


def run_qa_benchmarks(api_key: str, cases: list[dict], k: int) -> dict:
    retriever = get_hybrid_retriever(api_key, k=k)
    chain = build_rag_chain(retriever, api_key)
    results = []
    for case in cases:
        docs = retriever.invoke(case["question"])
        context = format_docs(docs)
        answer = redact_text(chain.invoke(case["question"])).text
        groundedness = score_groundedness(case["question"], context, answer, api_key)
        passed = groundedness["score"] >= case.get("min_groundedness", 0.0)

        results.append(
            {
                "id": case["id"],
                "question": case["question"],
                "answer": answer,
                "groundedness": groundedness,
                "passed": passed,
            }
        )

    return {
        "cases": results,
        "avg_groundedness": mean(
            [case["groundedness"]["score"] for case in results]
        ),
        "all_passed": all(case["passed"] for case in results),
    }


def run_pii_benchmarks(cases: list[dict]) -> dict:
    results = []
    for case in cases:
        redaction = redact_text(case["text"])
        forbidden_hits = [
            value for value in case.get("forbidden_substrings", []) if value in redaction.text
        ]
        missing_expected = [
            value for value in case.get("expected_substrings", []) if value not in redaction.text
        ]
        passed = not forbidden_hits and not missing_expected
        results.append(
            {
                "id": case["id"],
                "original_text": case["text"],
                "redacted_text": redaction.text,
                "entity_count": redaction.entity_count,
                "entity_types": list(redaction.entity_types),
                "forbidden_hits": forbidden_hits,
                "missing_expected": missing_expected,
                "passed": passed,
            }
        )

    return {
        "cases": results,
        "redaction_enabled": pii_redaction_enabled(),
        "all_passed": all(case["passed"] for case in results),
    }


def print_section(title: str, payload: dict) -> None:
    print(f"\n{title}")
    print("-" * len(title))
    for case in payload["cases"]:
        print(f"[{'PASS' if case['passed'] else 'FAIL'}] {case['id']}")
        if "precision" in case:
            print(
                f"  precision={case['precision']:.2f} recall={case['recall']:.2f} "
                f"retrieved={case['retrieved_source_files']}"
            )
        elif "groundedness" in case:
            print(
                f"  groundedness={case['groundedness']['score']:.2f} "
                f"answer={case['answer'][:140]}"
            )
        else:
            print(f"  redacted={case['redacted_text']}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, default=5, help="Top-K retrieval depth")
    parser.add_argument(
        "--retrieval-cases",
        type=Path,
        default=DEFAULT_RETRIEVAL_CASES,
        help="JSON file containing retrieval benchmark cases",
    )
    parser.add_argument(
        "--qa-cases",
        type=Path,
        default=DEFAULT_QA_CASES,
        help="JSON file containing groundedness benchmark cases",
    )
    parser.add_argument(
        "--pii-cases",
        type=Path,
        default=DEFAULT_PII_CASES,
        help="JSON file containing PII redaction benchmark cases",
    )
    parser.add_argument("--skip-retrieval", action="store_true")
    parser.add_argument("--skip-qa", action="store_true")
    parser.add_argument("--skip-pii", action="store_true")
    parser.add_argument(
        "--json-output",
        type=Path,
        help="Optional path to write the full benchmark report as JSON",
    )
    args = parser.parse_args()

    report: dict[str, dict] = {}

    needs_openai = not args.skip_retrieval or not args.skip_qa
    api_key = None
    if needs_openai:
        api_key = resolve_openai_api_key()
        if not api_key:
            print("❌ OPENAI_API_KEY is required for retrieval and groundedness benchmarks.")
            return 1

    if not args.skip_retrieval:
        retrieval_cases = load_cases(args.retrieval_cases)
        report["retrieval"] = run_retrieval_benchmarks(api_key, retrieval_cases, args.k)

    if not args.skip_qa:
        qa_cases = load_cases(args.qa_cases)
        report["qa"] = run_qa_benchmarks(api_key, qa_cases, args.k)

    if not args.skip_pii:
        pii_cases = load_cases(args.pii_cases)
        report["pii"] = run_pii_benchmarks(pii_cases)

    if "retrieval" in report:
        print_section("Retrieval Benchmarks", report["retrieval"])
        print(
            f"Average precision={report['retrieval']['avg_precision']:.2f} "
            f"average recall={report['retrieval']['avg_recall']:.2f}"
        )

    if "qa" in report:
        print_section("Groundedness Benchmarks", report["qa"])
        print(f"Average groundedness={report['qa']['avg_groundedness']:.2f}")

    if "pii" in report:
        print_section("PII Redaction Benchmarks", report["pii"])
        print(f"PII redaction enabled={report['pii']['redaction_enabled']}")

    if args.json_output:
        args.json_output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nWrote JSON report to {args.json_output}")

    overall_pass = all(section["all_passed"] for section in report.values()) if report else True
    print(f"\nOverall result: {'PASS' if overall_pass else 'FAIL'}")
    return 0 if overall_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
