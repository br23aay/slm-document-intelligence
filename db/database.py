"""
database.py
-----------
SQLite pipeline for the SLM Document Intelligence Service.
Logs every query with the answer, retrieved context, and evaluation scores.
"""

import sqlite3
import os
from datetime import datetime
from typing import Optional

DB_PATH = os.environ.get("DB_PATH", "db/queries.db")


def get_connection() -> sqlite3.Connection:
    parent = os.path.dirname(DB_PATH)
    if parent:
        os.makedirs(parent, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Create tables if they do not exist."""
    conn = get_connection()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS queries (
            id                INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp         TEXT    NOT NULL,
            question          TEXT    NOT NULL,
            answer            TEXT    NOT NULL,
            context_used      TEXT,
            groundedness_score REAL,
            hallucination_flag INTEGER,
            safety_score      REAL,
            chunks_retrieved  INTEGER,
            processing_ms     INTEGER
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_timestamp ON queries(timestamp)
    """)
    conn.commit()
    conn.close()


def log_query(
    question: str,
    answer: str,
    context_used: Optional[str] = None,
    groundedness_score: Optional[float] = None,
    hallucination_flag: Optional[bool] = None,
    safety_score: Optional[float] = None,
    chunks_retrieved: Optional[int] = None,
    processing_ms: Optional[int] = None,
) -> int:
    """Log a query and its evaluation scores. Returns the new row ID."""
    conn = get_connection()
    cursor = conn.execute(
        """
        INSERT INTO queries
            (timestamp, question, answer, context_used, groundedness_score,
             hallucination_flag, safety_score, chunks_retrieved, processing_ms)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            datetime.utcnow().isoformat(),
            question,
            answer,
            context_used,
            groundedness_score,
            int(hallucination_flag) if hallucination_flag is not None else None,
            safety_score,
            chunks_retrieved,
            processing_ms,
        ),
    )
    conn.commit()
    row_id = cursor.lastrowid
    conn.close()
    return row_id


def get_recent_queries(limit: int = 50) -> list[dict]:
    """Retrieve the most recent logged queries."""
    limit = min(limit, 200)
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM queries ORDER BY id DESC LIMIT ?", (limit,)
    ).fetchall()
    conn.close()
    return [dict(row) for row in rows]


def get_stats() -> dict:
    """Aggregate statistics across all logged queries."""
    conn = get_connection()
    row = conn.execute("""
        SELECT
            COUNT(*)                                    AS total,
            ROUND(AVG(groundedness_score), 4)           AS avg_groundedness,
            ROUND(AVG(safety_score), 4)                 AS avg_safety,
            SUM(hallucination_flag)                     AS hallucination_count,
            ROUND(AVG(processing_ms), 1)                AS avg_processing_ms,
            ROUND(AVG(chunks_retrieved), 1)             AS avg_chunks_retrieved,
            MAX(timestamp)                              AS last_query_at
        FROM queries
    """).fetchone()
    conn.close()
    total = row["total"] or 0
    return {
        "total_queries": total,
        "avg_groundedness": row["avg_groundedness"],
        "avg_safety": row["avg_safety"],
        "hallucination_count": row["hallucination_count"] or 0,
        "hallucination_rate": round((row["hallucination_count"] or 0) / total, 4) if total else 0,
        "avg_processing_ms": row["avg_processing_ms"],
        "avg_chunks_retrieved": row["avg_chunks_retrieved"],
        "last_query_at": row["last_query_at"],
    }
    
