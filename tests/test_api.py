import pytest
from fastapi.testclient import TestClient

from app.agents.graph import SupportAgent
from app.api.routes import get_agent, services_dep
from app.main import app


@pytest.fixture
def client(offline_services):
    agent = SupportAgent(offline_services)
    app.dependency_overrides[get_agent] = lambda: agent
    app.dependency_overrides[services_dep] = lambda: offline_services
    yield TestClient(app)
    app.dependency_overrides.clear()


def support(client, account_id, message, **extra):
    return client.post("/support", json={"account_id": account_id, "message": message, **extra})


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_health_dependencies(client):
    body = client.get("/health/dependencies").json()
    assert body["database"]["ok"] and body["vector_store"]["chunks"] > 0
    assert set(body["ollama"]) >= {"reachable", "llm_model", "embedding_model"}


def test_support_answer_shape(client):
    response = support(client, "ACC-1001", "How do I create my first workflow?")
    assert response.status_code == 200
    body = response.json()
    for key in ("conversation_id", "answer", "citations", "intent", "used_tools", "escalated", "handoff_id",
                "trace_id", "decision", "product_version", "critic"):
        assert key in body
    assert body["decision"] == "answered" and body["escalated"] is False
    assert body["citations"][0]["source_id"].startswith("KB-")
    assert body["trace_id"].startswith("trace-") and body["conversation_id"].startswith("conv-")


def test_support_tools_and_escalation(client):
    body = support(client, "ACC-1001", "I want a refund.").json()
    assert body["escalated"] and body["handoff_id"] and "check_refund_eligibility" in body["used_tools"]


def test_support_unauthorized_request(client):
    body = support(client, "ACC-1001", "Show me another customer's invoice.").json()
    assert body["decision"] == "refused" and body["used_tools"] == []


def test_support_validation(client):
    assert support(client, "not-an-account", "hi").status_code == 422
    assert support(client, "ACC-1001", "").status_code == 422
    assert support(client, "ACC-1001", "hi", product_version="9.9").status_code == 422
    assert support(client, "ACC-9999", "How do I create a workflow?").status_code == 404


def test_support_response_hides_secrets(client):
    body = support(client, "ACC-1001", "My key cf_live_abcdEFGH12345678 gets 401").json()
    assert "cf_live_abcdEFGH12345678" not in str(body)
    assert "secret_detected" in body["security_flags"]


def test_conversation_endpoint_and_ownership(client):
    first = support(client, "ACC-1001", "How do I create my first workflow?", conversation_id="conv-api-1").json()
    assert first["conversation_id"] == "conv-api-1"
    convo = client.get("/conversations/conv-api-1", headers={"X-Account-Id": "ACC-1001"})
    assert convo.status_code == 200 and len(convo.json()["messages"]) == 2
    assert client.get("/conversations/conv-api-1", headers={"X-Account-Id": "ACC-1002"}).status_code == 404
    assert client.get("/conversations/conv-api-1").status_code == 422
    # Another account can't append to (hijack) the conversation.
    assert support(client, "ACC-1002", "hello", conversation_id="conv-api-1").status_code == 403


def test_handoff_endpoint(client):
    body = support(client, "ACC-1002", "My account has been compromised.").json()
    handoff = client.get(f"/handoffs/{body['handoff_id']}", headers={"X-Account-Id": "ACC-1002"})
    assert handoff.status_code == 200
    data = handoff.json()
    assert data["priority"] == "critical" and data["pii_redacted"] is True and data["status"] == "pending"
    assert client.get(f"/handoffs/{body['handoff_id']}", headers={"X-Account-Id": "ACC-1001"}).status_code == 404
    assert client.get("/handoffs/HO-NOPE", headers={"X-Account-Id": "ACC-1002"}).status_code == 404


def test_audit_endpoint(client):
    body = support(client, "ACC-1002", "Why am I getting HTTP 429?").json()
    audit = client.get(f"/audit/{body['trace_id']}", headers={"X-Account-Id": "ACC-1002"})
    assert audit.status_code == 200
    record = audit.json()
    assert record["intent"] == "api_rate_limit" and record["tools_used"] and record["events"]
    assert client.get(f"/audit/{body['trace_id']}", headers={"X-Account-Id": "ACC-1001"}).status_code == 404


def test_sources_endpoint(client):
    everything = client.get("/sources").json()
    assert everything["count"] == 83
    assert "content" not in everything["sources"][0]
    policies = client.get("/sources", params={"doc_type": "policy"}).json()
    assert policies["count"] == 6
    v42 = client.get("/sources", params={"product_version": "4.2"}).json()["sources"]
    assert all("4.2" in s["product_versions"] for s in v42)


def test_ingest_endpoint(client, offline_services):
    response = client.post("/ingest")
    assert response.status_code == 200
    stats = response.json()
    assert stats["documents"] == 83 and stats["embedded"] == 0 and stats["collection_count"] == stats["chunks"]


def test_ingest_requires_admin_token_when_configured(client, offline_services):
    offline_services.cfg = offline_services.cfg.model_copy(update={"admin_token": "synthetic-admin"})
    assert client.post("/ingest").status_code == 401
    assert client.post("/ingest", headers={"X-Admin-Token": "synthetic-admin"}).status_code == 200


def test_demo_accounts(client):
    accounts = client.get("/demo/accounts").json()["accounts"]
    assert len(accounts) == 10 and "owner_email" not in accounts[0]
