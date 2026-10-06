"""Self-critique of draft answers.

Deterministic checks always run. An optional LLM judge (settings.critic_llm_judge)
adds a groundedness opinion. Each check is reported in `checks` so failures are
explainable and testable.
"""

import json
import re

from pydantic import BaseModel, Field

from app.agents.composer import cited_ids
from app.llm import LLMError, LLMProvider
from app.security.authorization import other_account_ids
from app.security.pii import contains_sensitive

# IDs like ACC-1001, INV-2026-0001, KB-..., CF-503 are not free-standing numbers.
_ID_TOKEN = re.compile(r"\b[A-Za-z]{2,}[-_][\w.-]*\d[\w.-]*\b")
_NUMBER = re.compile(r"(?<![\w.])\d[\d,]*(?:\.\d+)?")

REFUND_PROMISE = re.compile(
    r"\b(?:i|we)(?:'ve| have| will|'ll)? (?:issued|processed|approved|refunded|credited|will refund|"
    r"will issue|have refunded)\b|\byou(?:'ll| will) (?:receive|get) (?:a |your |the )?(?:full )?refund\b|"
    r"\brefund (?:has been|is|was) (?:approved|processed|issued|granted)\b",
    re.IGNORECASE,
)
PASSWORD_DISCLOSURE = re.compile(r"\byour (?:new |current |temporary )?password (?:is|was|=|:)\s+\S", re.IGNORECASE)

# "Settings > Billing > Cancel plan": each "A > B" step must exist in the sources.
_NAV_STEP = re.compile(r"([A-Za-z][\w-]*)\s*>\s*([A-Za-z][\w-]*)")

BLOCKING = "blocking"
ADVISORY = "advisory"


class CriticIssue(BaseModel):
    code: str
    detail: str
    severity: str = BLOCKING


class CriticResult(BaseModel):
    passed: bool
    issues: list[CriticIssue] = Field(default_factory=list)
    should_escalate: bool = False
    checks: dict[str, bool] = Field(default_factory=dict)

    def feedback(self) -> list[str]:
        return [f"{i.code}: {i.detail}" for i in self.issues if i.severity == BLOCKING]


def _normalize_number(raw: str) -> str:
    value = raw.replace(",", "")
    if "." in value:
        value = value.rstrip("0").rstrip(".")
    return value


def numbers_in(text: str) -> set[str]:
    return {_normalize_number(n) for n in _NUMBER.findall(_ID_TOKEN.sub(" ", text))}


def ungrounded_numbers(answer: str, context: str) -> list[str]:
    """Numbers in the answer that appear nowhere in the context (sources, facts, question)."""
    allowed = numbers_in(context) | numbers_in(_ID_TOKEN.sub(lambda m: m.group(0).replace("-", " "), context))
    found = []
    for number in sorted(numbers_in(answer)):
        if "." not in number and number.isdigit() and int(number) <= 10:
            continue  # list numbering and small counts
        if number not in allowed:
            found.append(number)
    return found


def _plain(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[*`\"'“”‘’]", "", text))


def unsupported_navigation(answer: str, context: str) -> list[str]:
    """UI navigation steps in the answer that never appear in the sources (invented menus)."""
    known = {(a.lower(), b.lower()) for a, b in _NAV_STEP.findall(_plain(context))}
    return [f"{a} > {b}" for a, b in _NAV_STEP.findall(_plain(answer)) if (a.lower(), b.lower()) not in known]


JUDGE_SYSTEM = ("You check whether an answer is fully supported by the given sources and facts. "
                "Reply with JSON only: {\"grounded\": true|false, \"unsupported_claims\": [\"...\"]}.")


def llm_groundedness(llm: LLMProvider, answer: str, context: str) -> tuple[bool, list[str]] | None:
    try:
        raw = llm.generate(JUDGE_SYSTEM, f"SOURCES AND FACTS:\n{context[:6000]}\n\nANSWER:\n{answer}")
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        verdict = json.loads(match.group(0)) if match else None
    except (LLMError, ValueError):
        return None
    if not isinstance(verdict, dict) or "grounded" not in verdict:
        return None
    return bool(verdict["grounded"]), [str(c) for c in verdict.get("unsupported_claims", [])][:5]


def critique(
    answer: str,
    *,
    account_id: str,
    product_version: str | None,
    sources: list[dict],
    facts: list[str],
    question: str,
    requires_citation: bool,
    escalation_expected: bool,
    escalating: bool,
    judge: LLMProvider | None = None,
) -> CriticResult:
    issues: list[CriticIssue] = []
    checks: dict[str, bool] = {}
    by_id = {s["source_id"]: s for s in sources}
    citations = cited_ids(answer)

    checks["non_empty"] = bool(answer.strip())
    if not checks["non_empty"]:
        issues.append(CriticIssue(code="empty_answer", detail="The answer is empty."))

    invalid = [c for c in citations if c not in by_id]
    checks["citations_valid"] = not invalid
    if invalid:
        issues.append(CriticIssue(code="invented_citation",
                                  detail=f"Cited IDs that were not retrieved: {', '.join(invalid)}. Only cite allowed IDs."))

    valid = [c for c in citations if c in by_id]
    checks["citation_present"] = bool(valid) or not requires_citation
    if not checks["citation_present"]:
        issues.append(CriticIssue(code="missing_citation",
                                  detail="The answer uses knowledge-base content but cites no source. Cite source IDs in [brackets]."))

    wrong_version = [c for c in valid if product_version and product_version not in by_id[c]["product_versions"]]
    checks["version_correct"] = not wrong_version
    if wrong_version:
        issues.append(CriticIssue(code="version_mismatch",
                                  detail=f"Cited sources that don't apply to CloudFlow {product_version}: {', '.join(wrong_version)}."))

    authoritative_available = any(s["authority_level"] <= 2 for s in sources)
    only_tickets = bool(valid) and all(by_id[c]["authority_level"] >= 4 for c in valid)
    checks["authority_respected"] = not (only_tickets and authoritative_available)
    if not checks["authority_respected"]:
        issues.append(CriticIssue(code="low_authority_only",
                                  detail="Only historical tickets were cited although current documentation was retrieved. Base the answer on current documentation."))

    context = "\n".join([question, *facts, *(s["content"] + " " + s["title"] for s in sources)])
    unsupported = ungrounded_numbers(answer, context)
    checks["numbers_grounded"] = not unsupported
    if unsupported:
        issues.append(CriticIssue(code="ungrounded_number",
                                  detail=f"Numbers not found in sources or account facts: {', '.join(unsupported[:5])}."))

    invented_paths = unsupported_navigation(answer, context)
    checks["navigation_grounded"] = not invented_paths
    if invented_paths:
        issues.append(CriticIssue(code="unsupported_navigation",
                                  detail=f"Menu paths not found in sources: {', '.join(invented_paths[:3])}. "
                                         "Only describe navigation that the sources describe."))

    leaked = contains_sensitive(answer)
    checks["no_sensitive_data"] = not leaked
    if leaked:
        issues.append(CriticIssue(code="sensitive_data", detail=f"Answer contains sensitive data: {', '.join(leaked)}."))

    checks["no_password_disclosure"] = not PASSWORD_DISCLOSURE.search(answer)
    if not checks["no_password_disclosure"]:
        issues.append(CriticIssue(code="password_disclosure", detail="The answer appears to state a password."))

    checks["no_refund_promise"] = not REFUND_PROMISE.search(answer)
    if not checks["no_refund_promise"]:
        issues.append(CriticIssue(code="refund_promise",
                                  detail="The answer promises or confirms a refund; only the billing team can decide (POLICY-REFUND-01)."))

    others = other_account_ids(answer, account_id)
    checks["authorization_respected"] = not others
    if others:
        issues.append(CriticIssue(code="cross_account_data", detail="The answer mentions another customer's account."))

    checks["escalation_consistent"] = escalating or not escalation_expected
    if not checks["escalation_consistent"]:
        issues.append(CriticIssue(code="escalation_required", detail="Policy requires a human handoff for this request.",
                                  severity=ADVISORY))

    if judge is not None and answer.strip():
        verdict = llm_groundedness(judge, answer, context)
        if verdict is not None:
            grounded, claims = verdict
            checks["llm_grounded"] = grounded
            if not grounded:
                issues.append(CriticIssue(code="llm_ungrounded",
                                          detail="Unsupported claims: " + "; ".join(claims or ["unspecified"])))

    blocking = [i for i in issues if i.severity == BLOCKING]
    return CriticResult(
        passed=not blocking,
        issues=issues,
        should_escalate=not checks["escalation_consistent"],
        checks=checks,
    )
