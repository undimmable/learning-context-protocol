"""High-level wrapper around the ORM model to provide the `_MemoryStore` API.

The existing FastAPI routes in *mcp_server.py* currently expect a very simple
in-memory list-backed store with these methods:

    * ``append(entry_dict)``      – persist a new snippet
    * ``all() -> list[dict]``     – return **all** snippets as dictionaries

We re-implement the same surface while delegating persistence to Postgres (or
SQLite in dev/test).  This means the public behaviour as observed by the
tests remains unchanged while we migrate the underlying storage medium.
"""

from __future__ import annotations

import datetime as _dt
from typing import List

from sqlalchemy import select

from .base import init_db, session_scope
from .model.memory_entry import MemoryEntry


class MemoryStore:  # noqa: D101 – public API documented in class docstring
    def __init__(self) -> None:
        # Ensure tables exist – cheap no-op when already created.
        init_db()

    # ------------------------------------------------------------------
    # Public helpers mimicking the historical JSONL behaviour
    # ------------------------------------------------------------------
    def append(self, entry: dict) -> None:  # noqa: D401
        """Insert a new entry.

        The caller may omit the timestamp; we always override it with the
        current UTC time to guarantee consistency.
        """

        with session_scope() as session:
            # Use datetime.now(UTC) instead of utcnow() to avoid deprecation warning
            timestamp = _dt.datetime.now(_dt.UTC)
            
            row = MemoryEntry(
                text=entry["text"],
                tags=list(entry.get("tags", [])),
                timestamp=timestamp,
            )
            session.add(row)

    def all(self) -> List[dict]:  # noqa: D401
        """Return **all** snippets sorted chronologically (oldest first)."""

        with session_scope() as session:
            rows = session.execute(select(MemoryEntry).order_by(MemoryEntry.id)).scalars().all()
            return [r.as_dict() for r in rows]

    # ------------------------------------------------------------------
    # *Testing* helpers – not used by production code but extremely helpful
    # to keep unit-tests hermetic.
    # ------------------------------------------------------------------

    def clear(self) -> None:  # noqa: D401
        """Remove **all** rows from the table."""

        from sqlalchemy import delete

        with session_scope() as session:
            session.execute(delete(MemoryEntry))
