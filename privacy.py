"""Presidio-backed PII redaction helpers for ingestion and LLM output."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from functools import lru_cache

os.environ.setdefault("TLDEXTRACT_CACHE", "/tmp/tldextract")

KNOWN_PERSON_ENTITY = "KNOWN_PERSON"
DEFAULT_ENTITY_TYPES = (
    KNOWN_PERSON_ENTITY,
    "PHONE_NUMBER",
    "EMAIL_ADDRESS",
    "CREDIT_CARD",
    "CRYPTO",
    "IBAN_CODE",
    "IP_ADDRESS",
    "US_BANK_NUMBER",
    "US_DRIVER_LICENSE",
    "US_PASSPORT",
    "US_SSN",
)
DEFAULT_LANGUAGE = "en"
DEFAULT_SCORE_THRESHOLD = 0.35
DEFAULT_SPACY_MODEL = "en_core_web_sm"
SAFE_SPAN_PATTERNS = (
    re.compile(r"\b\d{2}[-_]\d{2}[-_]\d{2}(?:\.md)?\b"),
    re.compile(r"\b\d{2}[-/]\d{2}[-/]\d{2}(?:\.md)?\b"),
    re.compile(r"\b\d{2}[-_]\d{2}[-_]\d{4}(?:\.md)?\b"),
    re.compile(r"\b\d{2}[-/]\d{2}[-/]\d{4}(?:\.md)?\b"),
    re.compile(r"\b\d{4}-\d{2}-\d{2}\b"),
)


@dataclass(frozen=True)
class RedactionResult:
    text: str
    entity_types: tuple[str, ...]
    entity_count: int


def pii_redaction_enabled() -> bool:
    value = os.getenv("PII_REDACTION_ENABLED", "true").strip().lower()
    return value not in {"0", "false", "no", "off"}


def _parse_entity_types() -> tuple[str, ...]:
    value = os.getenv("PII_REDACTION_ENTITY_TYPES", "")
    if not value:
        return DEFAULT_ENTITY_TYPES
    entity_types = tuple(item.strip() for item in value.split(",") if item.strip())
    return entity_types or DEFAULT_ENTITY_TYPES


def _parse_known_names() -> tuple[str, ...]:
    value = os.getenv("PII_REDACTION_KNOWN_NAMES", "")
    if not value:
        return ()
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _protect_safe_spans(text: str) -> tuple[str, dict[str, str]]:
    placeholders: dict[str, str] = {}

    def replace(match: re.Match[str]) -> str:
        token = f"__SAFE_TOKEN_{len(placeholders)}__"
        placeholders[token] = match.group(0)
        return token

    protected = text
    for pattern in SAFE_SPAN_PATTERNS:
        protected = pattern.sub(replace, protected)
    return protected, placeholders


def _restore_safe_spans(text: str, placeholders: dict[str, str]) -> str:
    restored = text
    for token, value in placeholders.items():
        restored = restored.replace(token, value)
    return restored


class PiiRedactor:
    """Redact common PII entities with Microsoft Presidio."""

    def __init__(self) -> None:
        self.language = os.getenv("PII_REDACTION_LANGUAGE", DEFAULT_LANGUAGE)
        self.known_names = _parse_known_names()
        self.entity_types = tuple(
            entity_type
            for entity_type in _parse_entity_types()
            if entity_type != KNOWN_PERSON_ENTITY or self.known_names
        )
        self.score_threshold = float(
            os.getenv("PII_REDACTION_SCORE_THRESHOLD", str(DEFAULT_SCORE_THRESHOLD))
        )
        self.model_name = os.getenv("PII_REDACTION_SPACY_MODEL", DEFAULT_SPACY_MODEL)
        self._analyzer, self._anonymizer, self._operator_config_cls = self._build_engines()

    def _build_engines(self):
        try:
            import spacy
            from presidio_analyzer import AnalyzerEngine, Pattern, PatternRecognizer
            from presidio_analyzer.nlp_engine import NlpEngineProvider
            from presidio_anonymizer import AnonymizerEngine
            from presidio_anonymizer.entities import OperatorConfig
        except ImportError as exc:
            raise RuntimeError(
                "PII redaction requires `presidio-analyzer`, `presidio-anonymizer`, and `spacy`."
            ) from exc

        try:
            spacy.load(self.model_name)
        except OSError as exc:
            raise RuntimeError(
                f"PII redaction requires the spaCy model `{self.model_name}`. "
                f"Install it with `python -m spacy download {self.model_name}`."
            ) from exc

        provider = NlpEngineProvider(
            nlp_configuration={
                "nlp_engine_name": "spacy",
                "models": [
                    {
                        "lang_code": self.language,
                        "model_name": self.model_name,
                    }
                ],
            }
        )
        nlp_engine = provider.create_engine()
        analyzer = AnalyzerEngine(
            nlp_engine=nlp_engine,
            supported_languages=[self.language],
        )
        analyzer.registry.remove_recognizer("UrlRecognizer")
        analyzer.registry.add_recognizer(
            PatternRecognizer(
                supported_entity="US_SSN",
                patterns=[
                    Pattern(
                        name="us_ssn_strict",
                        regex=r"\b\d{3}-\d{2}-\d{4}\b",
                        score=0.95,
                    )
                ],
                supported_language=self.language,
            )
        )
        if self.known_names:
            analyzer.registry.add_recognizer(
                PatternRecognizer(
                    supported_entity=KNOWN_PERSON_ENTITY,
                    deny_list=list(self.known_names),
                    supported_language=self.language,
                )
            )
        anonymizer = AnonymizerEngine()
        return analyzer, anonymizer, OperatorConfig

    def redact_text(self, text: str) -> RedactionResult:
        if not text:
            return RedactionResult(text=text, entity_types=(), entity_count=0)

        protected_text, placeholders = _protect_safe_spans(text)
        results = self._analyzer.analyze(
            text=protected_text,
            language=self.language,
            entities=list(self.entity_types),
            score_threshold=self.score_threshold,
        )
        if not results:
            return RedactionResult(text=text, entity_types=(), entity_count=0)

        entity_types = tuple(sorted({result.entity_type for result in results}))
        operators = {
            entity_type: self._operator_config_cls(
                "replace",
                {
                    "new_value": (
                        "<PERSON>"
                        if entity_type == KNOWN_PERSON_ENTITY
                        else f"<{entity_type}>"
                    )
                },
            )
            for entity_type in entity_types
        }
        redacted_text = self._anonymizer.anonymize(
            text=protected_text,
            analyzer_results=results,
            operators=operators,
        ).text
        redacted_text = _restore_safe_spans(redacted_text, placeholders)
        return RedactionResult(
            text=redacted_text,
            entity_types=entity_types,
            entity_count=len(results),
        )


@lru_cache(maxsize=1)
def get_pii_redactor() -> PiiRedactor:
    return PiiRedactor()


def redact_text(text: str) -> RedactionResult:
    if not pii_redaction_enabled():
        return RedactionResult(text=text, entity_types=(), entity_count=0)
    return get_pii_redactor().redact_text(text)
