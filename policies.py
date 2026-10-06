"""Look up the policies that currently govern a request (policy_registry)."""

from datetime import date
from pathlib import Path

from app.db.database import connect, fetch_all


def active_policy_sources(db_path: str | Path, policy_types: list[str], on: date | None = None) -> list[str]:
    """Source IDs of active, in-effect policies of the given types, in request order."""
    if not policy_types:
        return []
    today = (on or date.today()).isoformat()
    placeholders = ", ".join("?" for _ in policy_types)
    with connect(db_path) as conn:
        rows = fetch_all(conn, f"""
            SELECT policy_type, source_id FROM policy_registry
            WHERE active = 1 AND policy_type IN ({placeholders})
              AND effective_from <= ? AND (effective_to IS NULL OR effective_to > ?)
            ORDER BY version DESC
        """, (*policy_types, today, today))
    by_type: dict[str, list[str]] = {}
    for row in rows:
        by_type.setdefault(row["policy_type"], []).append(row["source_id"])
    return [sid for t in policy_types for sid in by_type.get(t, [])]
