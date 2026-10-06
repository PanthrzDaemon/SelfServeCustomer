"""Decide what the agent does for a request: refuse, answer from knowledge
(RAG), use tools, or escalate. Escalation rules and priorities follow
POLICY-ESCALATION-01.
"""

import re
from dataclasses import dataclass, field

from app.agents.intents import disputes

PRIORITY_CRITICAL, PRIORITY_HIGH, PRIORITY_NORMAL, PRIORITY_LOW = "critical", "high", "normal", "low"

ACCOUNT_TOOLS_USAGE = ["lookup_account", "get_usage", "get_plan_limits"]

PLATFORM_HINT = re.compile(r"\b(?:cf-503|503|outage|down|degraded|slow|incident|unavailable)\b", re.IGNORECASE)
SUSPENSION_HINT = re.compile(r"\b(?:cf-403|403|suspend\w*|not running|stopped running)\b", re.IGNORECASE)
RESET_REQUEST = re.compile(r"\b(?:reset|forgot|forgotten|send|new password|change (?:my )?password)\b", re.IGNORECASE)


@dataclass
class Plan:
    route: str  # "proceed" | "refuse" | "out_of_scope"
    needs_rag: bool = True
    # Policy types whose active policies (from policy_registry) must be consulted.
    policy_types: list[str] = field(default_factory=list)
    tools: list[str] = field(default_factory=list)
    escalate: bool = False
    escalation_reason: str | None = None
    priority: str = PRIORITY_LOW
    refusal_reason: str | None = None
    # Extra terms appended to the retrieval query so short questions ("Why was I
    # charged?") reach the documents for their intent.
    retrieval_hint: str = ""

    @property
    def strategy(self) -> str:
        if self.route != "proceed":
            return self.route
        parts = []
        if self.needs_rag:
            parts.append("rag")
        if self.tools:
            parts.append("tools")
        label = "+".join(parts) or "direct"
        return f"{label}+escalate" if self.escalate else label


RETRIEVAL_HINTS = {
    "security_incident": "security incident compromised account exposed API key revoke",
    "human_request": "escalation human handoff support",
    "data_privacy": "personal data deletion privacy request",
    "refund": "refund policy eligibility",
    "account_access": "account access ownership transfer identity verification 2FA",
    "password_reset": "reset password",
    "api_rate_limit": "HTTP 429 API rate limit",
    "usage": "usage limits quota API rate limit",
    "billing": "invoice charges billing payment",
    "account_status": "account status suspended past_due failed payment reactivate",
    "platform_status": "platform status incident CF-503",
}


def plan_request(intent: str, message: str, account: dict | None) -> Plan:
    plan = _plan(intent, message, account)
    plan.retrieval_hint = RETRIEVAL_HINTS.get(intent, "")
    return plan


def _plan(intent: str, message: str, account: dict | None) -> Plan:
    if intent == "prompt_injection":
        return Plan(route="refuse", needs_rag=False, refusal_reason="prompt_injection")
    if intent == "unauthorized_request":
        return Plan(route="refuse", needs_rag=False, refusal_reason="unauthorized_account_access")
    if intent == "out_of_scope":
        return Plan(route="out_of_scope", needs_rag=False)

    if intent == "security_incident":
        return Plan(route="proceed", policy_types=["security_incident", "escalation"], tools=["lookup_account"],
                    escalate=True, escalation_reason="security_incident", priority=PRIORITY_CRITICAL)
    if intent == "human_request":
        return Plan(route="proceed", policy_types=["escalation"], escalate=True,
                    escalation_reason="customer_requested_human", priority=PRIORITY_LOW)
    if intent == "data_privacy":
        return Plan(route="proceed", policy_types=["data_privacy", "escalation"], escalate=True,
                    escalation_reason="personal_data_request", priority=PRIORITY_LOW)
    if intent == "refund":
        # POLICY-REFUND-01: every refund request is handed to the billing team.
        return Plan(route="proceed", policy_types=["refund"],
                    tools=["lookup_account", "get_invoices", "check_refund_eligibility"],
                    escalate=True, escalation_reason="refund_request", priority=PRIORITY_NORMAL)
    if intent == "account_access":
        return Plan(route="proceed", policy_types=["access_control"], tools=["lookup_account"], escalate=True,
                    escalation_reason="identity_verification_required", priority=PRIORITY_LOW)
    if intent == "password_reset":
        tools = ["send_password_reset"] if RESET_REQUEST.search(message) else []
        return Plan(route="proceed", policy_types=["account_security"], tools=tools)
    if intent in ("api_rate_limit", "usage"):
        return Plan(route="proceed", tools=list(ACCOUNT_TOOLS_USAGE))
    if intent == "billing":
        return Plan(route="proceed", tools=["lookup_account", "get_invoices"])
    if intent == "account_status":
        plan = Plan(route="proceed", tools=["lookup_account", "get_invoices"])
        if account and account.get("status") == "suspended" and disputes(message):
            plan.escalate, plan.escalation_reason, plan.priority = True, "suspension_disputed", PRIORITY_HIGH
        return plan
    if intent == "platform_status":
        return Plan(route="proceed", tools=["check_platform_status"])
    if intent == "troubleshooting":
        tools = []
        if PLATFORM_HINT.search(message):
            tools.append("check_platform_status")
        if SUSPENSION_HINT.search(message):
            tools += ["lookup_account", "get_invoices"]
        return Plan(route="proceed", tools=tools)
    # how_to, general_question
    return Plan(route="proceed")
