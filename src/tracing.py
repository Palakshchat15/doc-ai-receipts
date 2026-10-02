"""SQLite tracing (data/traces.db, one row per extraction request) and the review queue (data/reviews.db)."""
from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

TRACE_SCHEMA = """
CREATE TABLE IF NOT EXISTS traces (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    source TEXT,            -- app | eval
    receipt TEXT,           -- file name or dataset id
    mode TEXT,              -- ocr_llm | vision | regex
    model TEXT,
    prompt_version TEXT,
    t_ocr REAL, t_llm REAL, t_validate REAL, t_total REAL,
    prompt_tokens INTEGER, completion_tokens INTEGER,
    n_boxes INTEGER,
    input TEXT,             -- OCR text given to the LLM
    output TEXT,            -- extracted JSON
    confidence REAL,
    decision TEXT,
    rules TEXT,             -- JSON
    error TEXT
);
"""

REVIEW_SCHEMA = """
CREATE TABLE IF NOT EXISTS reviews (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts_created REAL NOT NULL,
    ts_reviewed REAL,
    receipt TEXT,
    image_path TEXT,
    model TEXT,
    confidence REAL,
    reasons TEXT,           -- JSON list
    predicted TEXT,         -- JSON (model output)
    corrected TEXT,         -- JSON (after human edit)
    status TEXT NOT NULL    -- pending | approved | rejected
);
"""


class _DB:
    schema = ""

    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.executescript(self.schema)

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path, timeout=30)

    def _insert(self, table: str, row: dict) -> int:
        cols = ", ".join(row)
        with self._conn() as c:
            cur = c.execute(f"INSERT INTO {table} ({cols}) VALUES ({', '.join('?' * len(row))})",
                            list(row.values()))
            return int(cur.lastrowid)

    def df(self, q: str, params: tuple = ()):
        import pandas as pd

        with self._conn() as c:
            return pd.read_sql_query(q, c, params=params)


def _j(v):
    return v if (v is None or isinstance(v, str)) else json.dumps(v, ensure_ascii=False)


class Tracer(_DB):
    schema = TRACE_SCHEMA

    def log(self, **kw) -> int:
        row = {"ts": time.time(), **{k: _j(v) if k in ("output", "rules") else v for k, v in kw.items()}}
        return self._insert("traces", row)

    def read(self, limit: int = 500):
        return self.df(f"SELECT * FROM traces ORDER BY id DESC LIMIT {int(limit)}")


class ReviewStore(_DB):
    schema = REVIEW_SCHEMA

    def add(self, receipt: str, image_path: str, model: str, confidence: float, reasons: list,
            predicted: dict) -> int:
        return self._insert("reviews", {
            "ts_created": time.time(), "receipt": receipt, "image_path": image_path, "model": model,
            "confidence": confidence, "reasons": _j(reasons), "predicted": _j(predicted),
            "status": "pending"})

    def list(self, status: str | None = None):
        if status:
            return self.df("SELECT * FROM reviews WHERE status = ? ORDER BY id", (status,))
        return self.df("SELECT * FROM reviews ORDER BY id")

    def resolve(self, review_id: int, corrected: dict | None, status: str = "approved") -> None:
        with self._conn() as c:
            c.execute("UPDATE reviews SET corrected = ?, status = ?, ts_reviewed = ? WHERE id = ?",
                      (_j(corrected), status, time.time(), int(review_id)))

    def has_receipt(self, receipt: str) -> bool:
        with self._conn() as c:
            return c.execute("SELECT 1 FROM reviews WHERE receipt = ? LIMIT 1", (receipt,)).fetchone() is not None
