"""Shared runtime helpers for local scripts and the Streamlit app."""

from __future__ import annotations

import os
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent


def resolve_openai_api_key() -> str | None:
    """Resolve the OpenAI API key from the environment or local secrets file."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key:
        return api_key

    secrets_path = APP_DIR / ".streamlit" / "secrets.toml"
    if not secrets_path.exists():
        return None

    import tomllib

    try:
        with open(secrets_path, "rb") as handle:
            secrets = tomllib.load(handle)
    except tomllib.TOMLDecodeError:
        return None

    api_key = secrets.get("OPENAI_API_KEY")
    if not api_key or api_key == "your-openai-api-key-here":
        return None
    return api_key
