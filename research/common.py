"""Shared helpers for the research scripts (run them as `python -m research.<script>`)."""

from __future__ import annotations

import json
from pathlib import Path

from bhasharag.config import PROJECT_ROOT, get_settings
from bhasharag.db import MetadataDB

RESEARCH_DIR = PROJECT_ROOT / "research"
DATA_DIR = RESEARCH_DIR / "data"
RESULTS_DIR = RESEARCH_DIR / "results"


def load_kb_chunks() -> list[dict]:
    """All chunks currently in the app's knowledge base, with their document info."""
    settings = get_settings()
    if not settings.db_path.exists():
        raise SystemExit("No knowledge base found. Start the app and upload some documents first.")
    db = MetadataDB(settings.db_path)
    rows = db.conn.execute(
        """SELECT c.id, c.doc_id, c.page, c.language, c.token_count, c.text, d.filename
           FROM chunks c JOIN documents d ON d.id = c.doc_id ORDER BY c.id"""
    ).fetchall()
    if not rows:
        raise SystemExit("The knowledge base is empty. Upload some documents first.")
    return [dict(r) for r in rows]


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def markdown_table(headers: list[str], rows: list[list]) -> str:
    def fmt(v):
        return f"{v:.3f}" if isinstance(v, float) else str(v)

    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines += ["| " + " | ".join(fmt(v) for v in row) + " |" for row in rows]
    return "\n".join(lines)
