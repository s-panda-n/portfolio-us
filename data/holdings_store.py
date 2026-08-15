"""
data.holdings_store — Persist holdings to a local JSON file.

Saved to data/.holdings.json (gitignored — personal data).
Gracefully returns an empty list if the file doesn't exist or can't be read.
"""

from __future__ import annotations

import json
from pathlib import Path

_FILE = Path(__file__).parent / ".holdings.json"


def load() -> list[dict]:
    """Load holdings from disk. Returns [] on any failure."""
    try:
        if _FILE.exists():
            return json.loads(_FILE.read_text())
    except Exception:
        pass
    return []


def save(holdings: list[dict]) -> None:
    """Write holdings to disk. Silently ignores write errors (e.g. read-only FS)."""
    try:
        _FILE.write_text(json.dumps(holdings, indent=2))
    except Exception:
        pass
