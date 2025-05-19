"""ORM model for persistent assistant memory snippets.

Each row represents a short textual piece of information the assistant wants
to remember across sessions.  We store *tags* as a JSON-encoded list of
strings to keep the schema minimal and future-proof (no separate join table
required).  In Postgres the column is mapped to ``JSONB`` for efficient
querying; with SQLite we fall back to a plain ``TEXT`` column that stores the
JSON representation – SQLite does not have a native JSON type until recent
versions but the behaviour is good enough for tests.
"""

from __future__ import annotations

import datetime as _dt
import json as _json
from typing import List

from sqlalchemy import Column, DateTime, Integer, String, Text, types
from sqlalchemy.dialects import postgresql
from sqlalchemy.ext.mutable import MutableDict

from ..base import Base


class _JSONEncodedDict(types.TypeDecorator):
    """Fallback JSON storage for SQLite when JSON/JSONB not available."""

    impl = Text
    cache_ok = True

    def process_bind_param(self, value, dialect):  # noqa: D401, N802
        if value is None:
            return None
        return _json.dumps(value)

    def process_result_value(self, value, dialect):  # noqa: D401, N802
        if value is None:
            return None
        return _json.loads(value)


def _json_column():
    """Return a JSON column compatible with both Postgres and SQLite."""

    try:
        from sqlalchemy import inspect

        import sqlalchemy  # noqa: F401  (only for version check)
    except ImportError as exc:  # pragma: no cover – impossible in test env
        raise RuntimeError("SQLAlchemy must be installed") from exc

    # Detect if we are on Postgres at runtime.
    return (
        MutableDict.as_mutable(postgresql.JSONB)
        if str(Base.metadata.bind.url).startswith("postgresql")
        else _JSONEncodedDict
    )


class MemoryEntry(Base):
    """SQLAlchemy ORM model."""

    __tablename__ = "memory_entries"

    id: int = Column(Integer, primary_key=True, autoincrement=True)
    text: str = Column(String(length=4096), nullable=False)
    tags: List[str] = Column(_json_column(), nullable=False, default=list)
    timestamp: _dt.datetime = Column(DateTime(timezone=True), default=_dt.datetime.utcnow, nullable=False)

    # ---------------------------------------------------------------------
    # Convenience helpers
    # ---------------------------------------------------------------------
    def as_dict(self):  # noqa: D401
        return {
            "id": self.id,
            "text": self.text,
            "tags": self.tags,
            "timestamp": self.timestamp.isoformat() + "Z",
        }
