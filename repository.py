"""Persistence for conversations, handoffs and audit records.

Callers pass already-redacted text. JSON columns are (de)serialized here.
"""

import json
import sqlite3
from datetime import datetime, timezone

from app.db.database import fetch_all, fetch_one

HANDOFF_JSON_FIELDS = ("evidence", "tools_used", "unresolved_questions")
AUDIT_JSON_FIELDS = ("security_flags", "retrieved_sources", "tools_used", "tool_summaries", "citations", "critic", "events")


class OwnershipError(PermissionError):
    """The record belongs to a different account."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _decode(row: sqlite3.Row | None, json_fields: tuple[str, ...]) -> dict | None:
    if row is None:
        return None
    record = dict(row)
    for key in json_fields:
        if key in record and isinstance(record[key], str):
            record[key] = json.loads(record[key])
    return record


# ------------------------------------------------------------- conversations

def ensure_conversation(conn: sqlite3.Connection, conversation_id: str, account_id: str) -> None:
    row = fetch_one(conn, "SELECT account_id FROM conversations WHERE conversation_id = ?", (conversation_id,))
    now = utc_now()
    if row is None:
        conn.execute("INSERT INTO conversations VALUES (?, ?, ?, ?)", (conversation_id, account_id, now, now))
    elif row["account_id"] != account_id:
        raise OwnershipError("conversation belongs to another account")
    else:
        conn.execute("UPDATE conversations SET updated_at = ? WHERE conversation_id = ?", (now, conversation_id))


def conversation_owner(conn: sqlite3.Connection, conversation_id: str) -> str | None:
    row = fetch_one(conn, "SELECT account_id FROM conversations WHERE conversation_id = ?", (conversation_id,))
    return row["account_id"] if row else None


def add_message(conn: sqlite3.Connection, conversation_id: str, role: str, content: str, trace_id: str | None) -> None:
    conn.execute(
        "INSERT INTO messages (conversation_id, role, content, trace_id, created_at) VALUES (?, ?, ?, ?, ?)",
        (conversation_id, role, content, trace_id, utc_now()),
    )


def get_conversation(conn: sqlite3.Connection, conversation_id: str) -> dict | None:
    convo = fetch_one(conn, "SELECT * FROM conversations WHERE conversation_id = ?", (conversation_id,))
    if convo is None:
        return None
    messages = fetch_all(conn, """
        SELECT role, content, trace_id, created_at FROM messages
        WHERE conversation_id = ? ORDER BY message_id
    """, (conversation_id,))
    return {**dict(convo), "messages": [dict(m) for m in messages]}


# ------------------------------------------------------------------ handoffs

def insert_handoff(conn: sqlite3.Connection, record: dict) -> None:
    values = {k: json.dumps(v) if k in HANDOFF_JSON_FIELDS else v for k, v in record.items()}
    columns = ", ".join(values)
    conn.execute(f"INSERT INTO handoffs ({columns}) VALUES ({', '.join('?' for _ in values)})", tuple(values.values()))


def get_handoff(conn: sqlite3.Connection, handoff_id: str) -> dict | None:
    row = fetch_one(conn, "SELECT * FROM handoffs WHERE handoff_id = ?", (handoff_id,))
    record = _decode(row, HANDOFF_JSON_FIELDS)
    if record is not None:
        record["pii_redacted"] = bool(record["pii_redacted"])
    return record


# --------------------------------------------------------------------- audit

def insert_audit(conn: sqlite3.Connection, record: dict) -> None:
    values = {k: json.dumps(v) if k in AUDIT_JSON_FIELDS else v for k, v in record.items()}
    columns = ", ".join(values)
    conn.execute(f"INSERT INTO audit_log ({columns}) VALUES ({', '.join('?' for _ in values)})", tuple(values.values()))


def get_audit(conn: sqlite3.Connection, trace_id: str) -> dict | None:
    record = _decode(fetch_one(conn, "SELECT * FROM audit_log WHERE trace_id = ?", (trace_id,)), AUDIT_JSON_FIELDS)
    if record is not None:
        record["escalated"] = bool(record["escalated"])
    return record
