"""Build human-handoff records (POLICY-ESCALATION-01).

Every free-text field is passed through redaction again before storage, and
pii_redacted is set only after that, so a handoff never carries secrets.
"""

import uuid

from app.db.repository import utc_now
from app.security.pii import redact

UNRESOLVED_QUESTIONS = {
    "security_incident": ["Confirm the scope of the suspected compromise and secure the workspace."],
    "customer_requested_human": ["The customer asked to speak with a person."],
    "personal_data_request": ["Verify the requester and process the personal-data request."],
    "refund_request": ["Approve or decline the refund request under POLICY-REFUND-01."],
    "identity_verification_required": ["Verify the requester's identity before changing account access or ownership."],
    "suspension_disputed": ["Review the customer's dispute of the workspace suspension."],
    "knowledge_gap": ["No reliable knowledge-base source answers this question."],
    "critic_failed": ["The assistant could not produce a grounded answer that passed review."],
}


def new_handoff_id() -> str:
    return "HO-" + uuid.uuid4().hex[:10].upper()


def _clean(text: str, limit: int) -> str:
    return redact(text).text[:limit]


def build_handoff(
    *,
    conversation_id: str,
    account: dict | None,
    account_id: str,
    reason: str,
    priority: str,
    intent: str | None,
    sentiment: str,
    question: str,
    product_version: str | None,
    evidence: list[str],
    tools_used: list[str],
    attempted_resolution: str,
) -> dict:
    who = f"{account_id}"
    if account:
        who += f" ({account['plan']} plan, status {account['status']}, CloudFlow {product_version or account['product_version']})"
    now = utc_now()
    return {
        "handoff_id": new_handoff_id(),
        "conversation_id": conversation_id,
        "account_id": account_id if account else None,
        "reason": reason,
        "priority": priority,
        "intent": intent,
        "sentiment": sentiment,
        "customer_summary": _clean(f"Customer {who} wrote: \"{question}\"", 1200),
        "evidence": [_clean(e, 300) for e in evidence][:12],
        "tools_used": tools_used,
        "attempted_resolution": _clean(attempted_resolution, 2000) or None,
        "unresolved_questions": UNRESOLVED_QUESTIONS.get(reason, ["Review this conversation."]),
        "pii_redacted": 1,
        "status": "pending",
        "created_at": now,
        "updated_at": now,
    }
