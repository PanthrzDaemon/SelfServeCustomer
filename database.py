"""Small SQLite helpers. All queries take parameters; never format values into SQL."""

import sqlite3
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from app.config import settings

SCHEMA_PATH = Path(__file__).with_name("schema.sql")

Params = Sequence[Any] | dict[str, Any]


def get_connection(db_path: str | Path | None = None) -> sqlite3.Connection:
    """Open a connection with dict-like rows and foreign keys enforced."""
    conn = sqlite3.connect(db_path or settings.database_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


@contextmanager
def connect(db_path: str | Path | None = None) -> Iterator[sqlite3.Connection]:
    """Connection that commits on success, rolls back on error and always closes."""
    conn = get_connection(db_path)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_schema(conn: sqlite3.Connection) -> None:
    """Create all tables. Safe to run on an existing database."""
    conn.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))


def fetch_one(conn: sqlite3.Connection, sql: str, params: Params = ()) -> sqlite3.Row | None:
    return conn.execute(sql, params).fetchone()


def fetch_all(conn: sqlite3.Connection, sql: str, params: Params = ()) -> list[sqlite3.Row]:
    return conn.execute(sql, params).fetchall()


def execute(conn: sqlite3.Connection, sql: str, params: Params = ()) -> int:
    """Run a write statement and return the number of affected rows."""
    return conn.execute(sql, params).rowcount


def table_names(conn: sqlite3.Connection) -> set[str]:
    rows = fetch_all(conn, "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'")
    return {row["name"] for row in rows}


def table_columns(conn: sqlite3.Connection, table: str) -> list[str]:
    # PRAGMA arguments can't be bound as parameters, so only allow known tables.
    if table not in table_names(conn):
        raise ValueError(f"unknown table: {table}")
    return [row["name"] for row in fetch_all(conn, f"PRAGMA table_info({table})")]
