"""SQLite metadata store: documents, their chunks, and index bookkeeping."""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id          TEXT PRIMARY KEY,
    filename    TEXT NOT NULL,
    sha256      TEXT NOT NULL UNIQUE,
    stored_path TEXT NOT NULL,
    language    TEXT NOT NULL,
    num_pages   INTEGER NOT NULL,
    num_chunks  INTEGER NOT NULL,
    size_bytes  INTEGER NOT NULL,
    warnings    TEXT NOT NULL DEFAULT '',
    created_at  TEXT NOT NULL DEFAULT (datetime('now'))
);
-- AUTOINCREMENT guarantees ids are never reused, so a stale vector can never point
-- at a different chunk after deletions.
CREATE TABLE IF NOT EXISTS chunks (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id      TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    page        INTEGER NOT NULL,
    language    TEXT NOT NULL,
    token_count INTEGER NOT NULL,
    text        TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id);
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


class MetadataDB:
    def __init__(self, path: Path | str):
        self.conn = sqlite3.connect(str(path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.execute("PRAGMA journal_mode = WAL")
        self.conn.executescript(SCHEMA)

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        with self.conn:  # commits on success, rolls back on exception
            yield self.conn

    # documents -----------------------------------------------------------------
    def find_by_hash(self, sha256: str) -> sqlite3.Row | None:
        return self.conn.execute("SELECT * FROM documents WHERE sha256 = ?", (sha256,)).fetchone()

    def get_document(self, doc_id: str) -> sqlite3.Row | None:
        return self.conn.execute("SELECT * FROM documents WHERE id = ?", (doc_id,)).fetchone()

    def list_documents(self) -> list[sqlite3.Row]:
        return self.conn.execute("SELECT * FROM documents ORDER BY created_at DESC, filename").fetchall()

    def insert_document(self, conn: sqlite3.Connection, **fields) -> None:
        cols = ", ".join(fields)
        marks = ", ".join("?" for _ in fields)
        conn.execute(f"INSERT INTO documents ({cols}) VALUES ({marks})", tuple(fields.values()))

    def insert_chunk(self, conn: sqlite3.Connection, doc_id: str, chunk_index: int, page: int,
                     language: str, token_count: int, text: str) -> int:
        cur = conn.execute(
            "INSERT INTO chunks (doc_id, chunk_index, page, language, token_count, text) VALUES (?, ?, ?, ?, ?, ?)",
            (doc_id, chunk_index, page, language, token_count, text),
        )
        return int(cur.lastrowid)

    def delete_document(self, conn: sqlite3.Connection, doc_id: str) -> None:
        conn.execute("DELETE FROM documents WHERE id = ?", (doc_id,))

    # chunks --------------------------------------------------------------------
    def chunk_ids_for_document(self, doc_id: str) -> list[int]:
        rows = self.conn.execute("SELECT id FROM chunks WHERE doc_id = ?", (doc_id,)).fetchall()
        return [r["id"] for r in rows]

    def chunks_for_document(self, doc_id: str) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM chunks WHERE doc_id = ? ORDER BY chunk_index", (doc_id,)
        ).fetchall()

    def all_chunks(self) -> list[sqlite3.Row]:
        return self.conn.execute("SELECT id, doc_id, text FROM chunks ORDER BY id").fetchall()

    def get_chunks(self, ids: list[int]) -> dict[int, sqlite3.Row]:
        if not ids:
            return {}
        marks = ", ".join("?" for _ in ids)
        rows = self.conn.execute(
            f"""SELECT c.*, d.filename FROM chunks c JOIN documents d ON d.id = c.doc_id
                WHERE c.id IN ({marks})""",
            tuple(ids),
        ).fetchall()
        return {r["id"]: r for r in rows}

    def count_chunks(self) -> int:
        return int(self.conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0])

    # meta ----------------------------------------------------------------------
    def get_meta(self, key: str) -> str | None:
        row = self.conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return row["value"] if row else None

    def set_meta(self, key: str, value: str) -> None:
        with self.conn:
            self.conn.execute(
                "INSERT INTO meta (key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (key, value),
            )
