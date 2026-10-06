"""Shared types for deterministic support tools."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


@dataclass(frozen=True)
class ToolContext:
    """Who is calling. account_id is the authenticated customer's account."""

    account_id: str
    db_path: str | Path


class ToolResult(BaseModel):
    tool: str
    ok: bool
    data: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None
    # One-line, non-sensitive description for the audit log and the UI.
    summary: str = ""


FORBIDDEN = "forbidden"
NOT_FOUND = "not_found"


def forbidden(tool: str) -> ToolResult:
    # Deliberately says nothing about whether the other account exists.
    return ToolResult(tool=tool, ok=False, error=FORBIDDEN,
                      summary="Access denied: tools only return data for the caller's own account.")


def authorize(ctx: ToolContext, target_account_id: str | None) -> bool:
    """A customer may only access their own account."""
    return target_account_id is None or target_account_id == ctx.account_id


def mask_email(email: str) -> str:
    local, _, domain = email.partition("@")
    return f"{local[:1]}***@{domain}" if domain else "***"
