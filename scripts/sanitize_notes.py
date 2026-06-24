"""Sanitize checked-in note markdown files before committing or pushing."""

from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from privacy import redact_text

NOTES_ROOT = REPO_ROOT / "notes"
DROP_LINE_PREFIXES = (
    "Meeting participants:",
)
TRANSCRIPT_STUB = (
    "This public mirror omits raw session transcript content to remove names, "
    "personal details, and session logistics that are not part of the study material."
)


def build_transcript_stub(path: Path) -> str:
    note_name: str | None = None
    if path.name.endswith("_transcript.md"):
        note_name = path.name.replace("_transcript.md", ".md")
    elif path.name.endswith("-raw.md"):
        note_name = path.name.replace("-raw.md", ".md")

    lines = ["# Redacted Session Artifact", "", TRANSCRIPT_STUB]
    if note_name and (path.parent / note_name).exists():
        lines.extend([
            f"See `{note_name}` in this folder for the material-focused notes from this session.",
        ])
    return "\n".join(lines) + "\n"


def sanitize_markdown(text: str) -> str:
    kept_lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if any(stripped.startswith(prefix) for prefix in DROP_LINE_PREFIXES):
            continue
        kept_lines.append(line)
    sanitized = redact_text("\n".join(kept_lines)).text
    if text.endswith("\n"):
        return sanitized + "\n"
    return sanitized


def sanitize_path(path: Path) -> str:
    if path.name.endswith("_transcript.md") or path.name.endswith("-raw.md"):
        return build_transcript_stub(path)
    return sanitize_markdown(path.read_text(encoding="utf-8"))


def main() -> int:
    changed = 0
    for path in sorted(NOTES_ROOT.rglob("*.md")):
        original = path.read_text(encoding="utf-8")
        sanitized = sanitize_path(path)
        if sanitized == original:
            continue
        path.write_text(sanitized, encoding="utf-8")
        changed += 1
        print(f"Sanitized {path.relative_to(REPO_ROOT)}")
    print(f"Updated {changed} markdown files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
