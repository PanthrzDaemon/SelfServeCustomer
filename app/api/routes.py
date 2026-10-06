"""Support API.

Demo authentication: the caller's account is the `account_id` in POST /support
and the `X-Account-Id` header on read endpoints. Every read endpoint checks that
the record belongs to that account. A real deployment would derive the account
from a verified session or token instead.
"""

from functools import lru_cache
from typing import Literal

from fastapi import APIRouter, Depends, Header, HTTPException, Query
from pydantic import BaseModel, Field

from app.agents.graph import SupportAgent, SupportOutcome
from app.db import repository
from app.db.database import connect, fetch_all, fetch_one
from app.llm import ollama_available, ollama_models
from app.rag.embeddings import EmbeddingError
from app.rag.ingest import IngestStats, ingest
from app.services import Services, get_services
from app.sources import load_source_register

router = APIRouter()

ACCOUNT_ID_PATTERN = r"^ACC-\d{4}$"


class SupportRequest(BaseModel):
    conversation_id: str | None = Field(default=None, max_length=64, pattern=r"^[A-Za-z0-9_.:-]+$")
    account_id: str = Field(pattern=ACCOUNT_ID_PATTERN)
    message: str = Field(min_length=1, max_length=4000)
    product_version: Literal["4.2", "4.3"] | None = None


@lru_cache(maxsize=1)
def _default_agent() -> SupportAgent:
    return SupportAgent(get_services())


def get_agent() -> SupportAgent:
    return _default_agent()


def services_dep() -> Services:
    return get_services()


def _require_account(services: Services, account_id: str) -> None:
    with connect(services.db_path) as conn:
        if fetch_one(conn, "SELECT 1 FROM accounts WHERE account_id = ?", (account_id,)) is None:
            raise HTTPException(status_code=404, detail="Unknown account")


def caller_account(x_account_id: str = Header(..., pattern=ACCOUNT_ID_PATTERN)) -> str:
    return x_account_id


@router.post("/support", response_model=SupportOutcome)
def support(request: SupportRequest, agent: SupportAgent = Depends(get_agent)) -> SupportOutcome:
    _require_account(agent.s, request.account_id)
    try:
        return agent.run(
            account_id=request.account_id,
            message=request.message,
            conversation_id=request.conversation_id,
            product_version=request.product_version,
        )
    except repository.OwnershipError:
        raise HTTPException(status_code=403, detail="Conversation belongs to another account")


@router.get("/conversations/{conversation_id}")
def conversation(conversation_id: str, account_id: str = Depends(caller_account),
                 services: Services = Depends(services_dep)) -> dict:
    with connect(services.db_path) as conn:
        record = repository.get_conversation(conn, conversation_id)
    # Same response for "missing" and "someone else's", so IDs can't be probed.
    if record is None or record["account_id"] != account_id:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return record


@router.get("/handoffs/{handoff_id}")
def handoff(handoff_id: str, account_id: str = Depends(caller_account),
            services: Services = Depends(services_dep)) -> dict:
    with connect(services.db_path) as conn:
        record = repository.get_handoff(conn, handoff_id)
    if record is None or record["account_id"] != account_id:
        raise HTTPException(status_code=404, detail="Handoff not found")
    return record


@router.get("/audit/{trace_id}")
def audit(trace_id: str, account_id: str = Depends(caller_account),
          services: Services = Depends(services_dep)) -> dict:
    with connect(services.db_path) as conn:
        record = repository.get_audit(conn, trace_id)
    if record is None or record["account_id"] != account_id:
        raise HTTPException(status_code=404, detail="Audit record not found")
    return record


@router.get("/sources")
def sources(
    doc_type: str | None = Query(default=None),
    product_version: Literal["4.2", "4.3"] | None = Query(default=None),
) -> dict:
    """Knowledge-source metadata (from the source register). No document content."""
    records = load_source_register()
    if doc_type:
        records = [r for r in records if r["doc_type"] == doc_type]
    if product_version:
        records = [r for r in records if product_version in r["product_versions"]]
    return {"count": len(records), "sources": records}


@router.post("/ingest", response_model=IngestStats)
def run_ingest(services: Services = Depends(services_dep),
               x_admin_token: str | None = Header(default=None)) -> IngestStats:
    if services.cfg.admin_token and x_admin_token != services.cfg.admin_token:
        raise HTTPException(status_code=401, detail="Admin token required")
    try:
        stats = ingest(services.embedder, services.store)
    except EmbeddingError as exc:
        raise HTTPException(status_code=503, detail=f"Embedding provider unavailable: {exc}")
    services.retriever._vocabulary = None  # knowledge changed; rebuild coverage vocabulary
    return stats


@router.get("/health/dependencies")
def dependencies(services: Services = Depends(services_dep)) -> dict:
    cfg = services.cfg
    try:
        with connect(services.db_path) as conn:
            accounts = fetch_one(conn, "SELECT COUNT(*) AS n FROM accounts")["n"]
        database = {"ok": True, "accounts": accounts}
    except Exception as exc:  # noqa: BLE001 - report any DB problem as unhealthy
        database = {"ok": False, "error": type(exc).__name__}
    try:
        vector = {"ok": True, "chunks": services.store.count(), "embedding_model": services.store.embedding_model()}
    except Exception as exc:  # noqa: BLE001
        vector = {"ok": False, "error": type(exc).__name__}
    models = ollama_models(cfg)
    return {
        "database": database,
        "vector_store": vector,
        "ollama": {
            "reachable": ollama_available(cfg),
            "llm_model": cfg.ollama_model,
            "llm_model_available": any(m == cfg.ollama_model or m.split(":")[0] == cfg.ollama_model for m in models),
            "embedding_model": cfg.embedding_model,
            "embedding_model_available": any(m.split(":")[0] == cfg.embedding_model.split(":")[0] for m in models),
            "llm_enabled": cfg.llm_enabled,
        },
    }


@router.get("/demo/accounts")
def demo_accounts(services: Services = Depends(services_dep)) -> dict:
    """Synthetic demo accounts for the frontend's account picker (disabled in production)."""
    if services.cfg.environment == "production":
        raise HTTPException(status_code=404, detail="Not found")
    with connect(services.db_path) as conn:
        rows = fetch_all(conn, "SELECT account_id, company_name, plan, status, product_version FROM accounts ORDER BY account_id")
    return {"accounts": [dict(r) for r in rows]}
