"""Synthetic document store backed by SQLite.

Uses SQLAlchemy Core when it is installed (the intended path) and falls back to
the standard-library ``sqlite3`` module when it is not, behind identical
function signatures. All queries are parameterised; user-derived strings never
enter SQL text directly. The data is entirely synthetic (Faker-generated, or a
deterministic generator if Faker is absent) and exists only to give the
access-control logic something to gate.
"""

from __future__ import annotations

import logging
import random
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Relative weighting of generated classifications (more low-sensitivity docs).
_CLASSIFICATION_WEIGHTS = {"TS": 0.05, "S": 0.15, "C": 0.30, "U": 0.50}

# Detect SQLAlchemy once.
try:
    from sqlalchemy import (
        Column, Integer, MetaData, String, Table, Text,
        create_engine, func, insert, inspect, or_, select,
    )
    from sqlalchemy.engine import Engine

    _HAVE_SQLALCHEMY = True
except Exception:  # pragma: no cover - exercised only when sqlalchemy absent
    _HAVE_SQLALCHEMY = False
    Engine = Any  # type: ignore


# A tiny engine shim so the stdlib path returns an object that knows its path.
class _SqliteEngine:
    """Minimal stand-in for a SQLAlchemy Engine over stdlib sqlite3."""

    def __init__(self, db_path: Path):
        self.db_path = str(db_path)

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn


def _fake_faker():
    """Return a Faker instance if available, else a deterministic stub."""
    try:
        import faker

        return faker.Faker()
    except Exception:  # pragma: no cover - exercised only when faker absent
        class _Stub:
            _words = ["strategy", "budget", "network", "security", "policy",
                      "logistics", "personnel", "research", "finance", "audit"]

            def bs(self):
                return " ".join(random.sample(self._words, 3))

            def catch_phrase(self):
                return " ".join(random.sample(self._words, 2)).title()

            def paragraphs(self, nb=2):
                return [" ".join(random.choices(self._words, k=20)) for _ in range(nb)]

        return _Stub()


def _fake_content(faker_instance, max_len: int = 600) -> str:
    paragraphs = faker_instance.paragraphs(nb=random.randint(1, 3))
    return "\n\n".join(paragraphs)[:max_len]


# ============================================================================
# Public API — dispatches to the SQLAlchemy or stdlib implementation.
# ============================================================================
def get_engine(db_path: Path):
    """Create (and smoke-test) an engine for ``db_path``."""
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if _HAVE_SQLALCHEMY:
        engine = create_engine(f"sqlite:///{db_path}", future=True)
        with engine.connect():
            pass
        logger.info("Database ready (SQLAlchemy): %s", db_path)
        return engine
    engine = _SqliteEngine(db_path)
    with engine.connect():
        pass
    logger.info("Database ready (stdlib sqlite3): %s", db_path)
    return engine


def setup_schema(engine, table_name: str) -> None:
    """Create the documents table if it does not already exist."""
    if _HAVE_SQLALCHEMY:
        metadata = MetaData()
        Table(
            table_name, metadata,
            Column("DocID", Integer, primary_key=True),
            Column("Title", String(255), nullable=False),
            Column("Content", Text),
            Column("Classification", String(10), nullable=False),
            Column("Subject", String(255)),
        )
        metadata.create_all(engine)
    else:
        with engine.connect() as conn:
            conn.execute(
                f"CREATE TABLE IF NOT EXISTS {table_name} ("
                "DocID INTEGER PRIMARY KEY AUTOINCREMENT, "
                "Title TEXT NOT NULL, Content TEXT, "
                "Classification TEXT NOT NULL, Subject TEXT)"
            )
            conn.commit()
    logger.info("Schema ready for table '%s'.", table_name)


def _count(engine, table_name: str) -> int:
    if _HAVE_SQLALCHEMY:
        docs = Table(table_name, MetaData(), autoload_with=engine)
        with engine.connect() as conn:
            return conn.execute(select(func.count(docs.c.DocID))).scalar_one_or_none() or 0
    with engine.connect() as conn:
        cur = conn.execute(f"SELECT COUNT(*) FROM {table_name}")
        return cur.fetchone()[0]


def populate(engine, table_name: str, target_count: int,
             classification_levels: Dict[str, int]) -> int:
    """Top up the table to ``target_count`` synthetic documents (idempotent)."""
    current = _count(engine, table_name)
    to_add = target_count - current
    if to_add <= 0:
        logger.info("Database already populated (%d docs).", current)
        return 0

    fake = _fake_faker()
    levels = sorted(classification_levels, key=lambda k: classification_levels[k])
    weights = [_CLASSIFICATION_WEIGHTS.get(lvl, 0.25) for lvl in levels]
    rows = [
        {
            "Title": fake.bs().title(),
            "Content": _fake_content(fake),
            "Classification": random.choices(levels, weights=weights, k=1)[0],
            "Subject": fake.catch_phrase(),
        }
        for _ in range(to_add)
    ]

    start = time.time()
    if _HAVE_SQLALCHEMY:
        docs = Table(table_name, MetaData(), autoload_with=engine)
        with engine.begin() as conn:
            conn.execute(insert(docs), rows)
    else:
        with engine.connect() as conn:
            conn.executemany(
                f"INSERT INTO {table_name} (Title, Content, Classification, Subject) "
                "VALUES (:Title, :Content, :Classification, :Subject)",
                rows,
            )
            conn.commit()
    logger.info("Inserted %d docs in %.2fs (total %d).",
                len(rows), time.time() - start, current + len(rows))
    return len(rows)


def search_documents(engine, table_name: str, search_terms: str,
                     user_clearance_level: int,
                     classification_levels: Dict[str, int],
                     limit: int = 25) -> List[Dict[str, Any]]:
    """Case-insensitive LIKE search across Title/Content/Subject.

    Filtered so only documents at or below the user's clearance are returned.
    Need-to-know is enforced on retrieval, not search (search returns metadata
    only, no content).
    """
    pattern = f"%{search_terms.lower()}%"
    if _HAVE_SQLALCHEMY:
        docs = Table(table_name, MetaData(), autoload_with=engine)
        stmt = (
            select(docs.c.DocID, docs.c.Title, docs.c.Classification, docs.c.Subject)
            .where(or_(
                func.lower(docs.c.Title).like(pattern),
                func.lower(docs.c.Content).like(pattern),
                func.lower(docs.c.Subject).like(pattern),
            ))
            .limit(limit)
        )
        with engine.connect() as conn:
            found = [dict(r) for r in conn.execute(stmt).mappings().all()]
    else:
        with engine.connect() as conn:
            cur = conn.execute(
                f"SELECT DocID, Title, Classification, Subject FROM {table_name} "
                "WHERE lower(Title) LIKE :p OR lower(Content) LIKE :p "
                "OR lower(Subject) LIKE :p LIMIT :lim",
                {"p": pattern, "lim": limit},
            )
            found = [dict(r) for r in cur.fetchall()]

    accessible = [
        row for row in found
        if user_clearance_level >= classification_levels.get(row.get("Classification"), 99)
    ]
    logger.info("Search '%s': %d matched, %d accessible.",
                search_terms, len(found), len(accessible))
    return accessible


def get_document(engine, table_name: str, doc_id: int) -> Optional[Dict[str, Any]]:
    """Fetch a single document row by ID, or ``None`` if absent."""
    if _HAVE_SQLALCHEMY:
        docs = Table(table_name, MetaData(), autoload_with=engine)
        stmt = select(docs).where(docs.c.DocID == doc_id)
        with engine.connect() as conn:
            row = conn.execute(stmt).mappings().fetchone()
        return dict(row) if row else None
    with engine.connect() as conn:
        cur = conn.execute(
            f"SELECT * FROM {table_name} WHERE DocID = :id", {"id": doc_id}
        )
        row = cur.fetchone()
    return dict(row) if row else None


def fetch_contents(engine, table_name: str, doc_ids: List[int]) -> Dict[int, Optional[str]]:
    """Bulk-fetch document contents for a list of IDs."""
    if not doc_ids:
        return {}
    if _HAVE_SQLALCHEMY:
        docs = Table(table_name, MetaData(), autoload_with=engine)
        stmt = select(docs.c.DocID, docs.c.Content).where(docs.c.DocID.in_(doc_ids))
        with engine.connect() as conn:
            return {row[0]: row[1] for row in conn.execute(stmt).all()}
    placeholders = ",".join("?" for _ in doc_ids)
    with engine.connect() as conn:
        cur = conn.execute(
            f"SELECT DocID, Content FROM {table_name} WHERE DocID IN ({placeholders})",
            list(doc_ids),
        )
        return {row[0]: row[1] for row in cur.fetchall()}


def preview(engine, table_name: str, limit: int = 20) -> List[Dict[str, Any]]:
    """Return a small metadata preview for display purposes."""
    limit = max(1, min(int(limit), 500))
    if _HAVE_SQLALCHEMY:
        docs = Table(table_name, MetaData(), autoload_with=engine)
        stmt = (select(docs.c.DocID, docs.c.Title, docs.c.Classification, docs.c.Subject)
                .order_by(docs.c.DocID).limit(limit))
        with engine.connect() as conn:
            return [dict(r) for r in conn.execute(stmt).mappings().all()]
    with engine.connect() as conn:
        cur = conn.execute(
            f"SELECT DocID, Title, Classification, Subject FROM {table_name} "
            "ORDER BY DocID LIMIT :lim", {"lim": limit}
        )
        return [dict(r) for r in cur.fetchall()]
