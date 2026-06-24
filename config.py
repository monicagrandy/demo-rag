"""Configuration helpers for the standalone class notes RAG app."""

from __future__ import annotations

import os
import re
from pathlib import Path, PurePosixPath

APP_DIR = Path(__file__).resolve().parent
REPO_ROOT = APP_DIR
NOTES_ROOT_ENV_VARS = ("CLASS_NOTES_DIR", "RAG_NOTES_DIR")
DEFAULT_NOTES_ROOT = REPO_ROOT / "notes"
DEFAULT_NOTES_GLOB = "**/*.md"
DEFAULT_NOTES_EXCLUDE_GLOBS = (
    "**/*transcript*.md",
    "**/*-raw.md",
)

def _split_glob_patterns(value: str) -> list[str]:
    patterns = [pattern.strip() for pattern in value.split(",")]
    return [pattern for pattern in patterns if pattern]


MONTH_LOOKUP = {
    "jan": "01",
    "feb": "02",
    "mar": "03",
    "apr": "04",
    "may": "05",
    "jun": "06",
    "jul": "07",
    "aug": "08",
    "sep": "09",
    "oct": "10",
    "nov": "11",
    "dec": "12",
}


def get_notes_root() -> Path:
    """Resolve the notes directory to index."""
    for env_var in NOTES_ROOT_ENV_VARS:
        value = os.environ.get(env_var)
        if value:
            return Path(value).expanduser().resolve()
    return DEFAULT_NOTES_ROOT.resolve()


def get_notes_globs() -> list[str]:
    value = os.environ.get("CLASS_NOTES_GLOB", DEFAULT_NOTES_GLOB)
    return _split_glob_patterns(value)


def get_notes_exclude_globs() -> list[str]:
    value = os.environ.get("CLASS_NOTES_EXCLUDE_GLOB", "")
    return [*DEFAULT_NOTES_EXCLUDE_GLOBS, *_split_glob_patterns(value)]


def is_excluded_relative_path(relative_path: str) -> bool:
    normalized = relative_path.replace("\\", "/").lstrip("./")
    if not normalized:
        return False
    candidate = PurePosixPath(normalized)
    return any(candidate.match(pattern) for pattern in get_notes_exclude_globs())


def is_excluded_note_path(path: Path, notes_root: Path | None = None) -> bool:
    root = (notes_root or get_notes_root()).resolve()
    resolved_path = path.resolve()
    try:
        relative = resolved_path.relative_to(root)
    except ValueError:
        relative = Path(path.name)

    if any(part.startswith(".") for part in relative.parts):
        return True

    return is_excluded_relative_path(relative.as_posix())


def _normalize_year(year: str | None) -> str | None:
    if not year:
        return None
    return year if len(year) == 4 else f"20{year}"


def parse_class_date(filename_stem: str) -> str | None:
    iso_match = re.match(r"(\d{4})[-_](\d{2})[-_](\d{2})(?:$|[-_])", filename_stem)
    if iso_match:
        year, month, day = iso_match.groups()
        return f"{year}-{month}-{day}"

    match = re.match(r"(\d{2})[-_](\d{2})[-_](\d{2,4})(?:$|[-_])", filename_stem)
    if not match:
        return None

    month, day, year = match.groups()
    full_year = _normalize_year(year)
    if not full_year:
        return None
    return f"{full_year}-{month}-{day}"


def parse_class_date_from_text(path: Path) -> str | None:
    stem_date = parse_class_date(path.stem)
    fallback_year = stem_date[:4] if stem_date else None
    weekday_date_pattern = re.compile(
        r"^(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun),?\s+(\d{1,2})\s+([A-Za-z]{3,9})\s+(\d{2,4})\s*$"
    )
    meeting_date_pattern = re.compile(
        r"^Date:\s+([A-Za-z]{3,9})\s+(\d{1,2})(?:,?\s+(\d{2,4}))?\s*$"
    )

    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        stripped = line.strip()

        match = weekday_date_pattern.match(stripped)
        if match:
            day, month_name, year = match.groups()
            month = MONTH_LOOKUP.get(month_name[:3].lower())
            full_year = _normalize_year(year)
            if month and full_year:
                return f"{full_year}-{month}-{int(day):02d}"

        match = meeting_date_pattern.match(stripped)
        if match:
            month_name, day, year = match.groups()
            month = MONTH_LOOKUP.get(month_name[:3].lower())
            full_year = _normalize_year(year) or fallback_year
            if month and full_year:
                return f"{full_year}-{month}-{int(day):02d}"

    return None


def extract_markdown_title(path: Path) -> str:
    heading_pattern = re.compile(r"^#{1,6}\s+(.+?)\s*$")
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        stripped = line.strip()
        if stripped.startswith("Meeting Title:"):
            title = stripped.split(":", 1)[1].strip()
            if title:
                return title
        match = heading_pattern.match(stripped)
        if match:
            title = match.group(1).strip()
            if title:
                return title
    return re.sub(r"[_-]+", " ", path.stem).strip() or path.stem


def format_collection_label(path: Path, notes_root: Path) -> str:
    relative = path.relative_to(notes_root)
    if len(relative.parts) <= 1:
        return "Notes"
    first_part = relative.parts[0]
    return re.sub(r"[_-]+", " ", first_part).strip().title() or "Notes"


def get_source_specs() -> list[dict]:
    """Discover markdown note sources from the configured notes root."""
    notes_root = get_notes_root()
    if not notes_root.exists():
        return []

    discovered_paths: set[Path] = set()
    for pattern in get_notes_globs():
        discovered_paths.update(path.resolve() for path in notes_root.glob(pattern))

    specs = []
    for path in sorted(discovered_paths):
        if not path.is_file() or path.suffix.lower() != ".md":
            continue
        if is_excluded_note_path(path, notes_root):
            continue
        specs.append(
            {
                "path": path,
                "notes_root": notes_root,
                "collection": format_collection_label(path, notes_root),
                "title": extract_markdown_title(path),
                "class_date": parse_class_date_from_text(path) or parse_class_date(path.stem),
            }
        )
    return specs
