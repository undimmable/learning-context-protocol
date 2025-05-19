"""Database configuration and SQLAlchemy base setup.

This module centralises the SQLAlchemy engine, session maker and declarative
base used across the backend.  The connection URL is resolved in the following
order (first match wins):

1.   The environment variable ``DATABASE_URL`` – this is the canonical way to
     point the application at a production Postgres instance, e.g.::

         export DATABASE_URL=postgresql+psycopg2://user:pass@localhost:5432/app

2.   Fallback to a *file-backed* SQLite database in the current working
     directory (``sqlite:///./app.db``).  Using a persistent file rather than
     an in-memory database means developers still see data across process
     restarts while keeping the test environment self-contained.

The design deliberately keeps **no** hard dependency on Postgres at import
time so unit-tests can run without additional infrastructure.  When
``DATABASE_URL`` points at Postgres the exact same code paths are exercised –
SQLAlchemy makes the backend agnostic at runtime.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./app.db")

# ``echo=False`` so unit-tests stay quiet – developers can enable SQL echo by
# simply exporting ``SQLALCHEMY_ECHO=1`` without touching the code.
engine = create_engine(
    DATABASE_URL,
    echo=os.getenv("SQLALCHEMY_ECHO", "0") == "1",
    future=True,
)

SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)

Base = declarative_base()

# Bind metadata early so helpers that look at ``Base.metadata.bind`` (e.g.
# dynamic JSON column selection) can introspect the underlying dialect.
Base.metadata.bind = engine


@contextmanager
def session_scope() -> Generator["Session", None, None]:
    """Provide a transactional scope around a series of operations."""

    from sqlalchemy.orm import Session  # imported lazily to avoid circulars

    session: Session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:  # noqa: BLE001 – propagate original stacktrace
        session.rollback()
        raise
    finally:
        session.close()


def init_db() -> None:
    """Create all tables (no-op if they already exist).

    In production we rely on migrations managed by Alembic, but creating the
    tables programmatically is convenient for unit-tests and ad-hoc local
    runs where migrations might not have been executed yet.
    """

    import importlib
    import pkgutil

    # Import *all* modules inside ``src.db.model`` so the ORM knows
    # about every mapped class when ``Base.metadata.create_all`` is executed.
    from pathlib import Path

    models_pkg = "src.db.model"

    for _, modname, _ in pkgutil.walk_packages(
        [str(Path(__file__).parent / "model")], prefix=f"{models_pkg}."
    ):
        importlib.import_module(modname)

    Base.metadata.create_all(bind=engine)
