"""Configuration helpers for the standalone class notes RAG app."""

from __future__ import annotations

import os
import re
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
REPO_ROOT = APP_DIR
NOTES_ROOT_ENV_VARS = ("CLASS_NOTES_DIR", "RAG_NOTES_DIR")
DEFAULT_NOTES_ROOT = REPO_ROOT / "notes"
DEFAULT_NOTES_GLOB = "**/*.md"

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
    patterns = [pattern.strip() for pattern in value.split(",")]
    return [pattern for pattern in patterns if pattern]


def parse_class_date(filename_stem: str) -> str | None:
    match = re.fullmatch(r"(\d{2})-(\d{2})-(\d{2})", filename_stem)
    if not match:
        return None
    month, day, year = match.groups()
    return f"20{year}-{month}-{day}"


def parse_class_date_from_text(path: Path) -> str | None:
    date_pattern = re.compile(
        r"^(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun),?\s+(\d{1,2})\s+([A-Za-z]{3})\s+(\d{2,4})\s*$"
    )
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        match = date_pattern.match(line.strip())
        if not match:
            continue
        day, month_name, year = match.groups()
        month = MONTH_LOOKUP.get(month_name.lower())
        if not month:
            return None
        full_year = year if len(year) == 4 else f"20{year}"
        return f"{full_year}-{month}-{int(day):02d}"
    return None


def extract_markdown_title(path: Path) -> str:
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("# "):
            return line[2:].strip() or path.stem
    return path.stem.replace("-", " ")


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
        if any(part.startswith(".") for part in path.parts):
            continue
        if not path.is_file() or path.suffix.lower() != ".md":
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
