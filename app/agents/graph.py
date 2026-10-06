"""The CloudFlow support agent as a LangGraph state machine.

    security_check -> authorize --(unauthorized/unknown)--> respond_direct -> finalize
                          |
                    classify_intent -> plan --(refuse/out_of_scope)--> respond_direct
                                         |
                    (rag?) retrieve -> (tools?) run_tools -> compose
                                                                |
                         (knowledge gap) -------------------> escalate -> finalize
                                                                |
                                                             critic --(fail, 1st)--> revise -> critic
                                                                |--(fail, 2nd)--> escalate
                                                                |--(pass, policy needs human)--> escalate
                                                                '--(pass)--> finalize

Every node records an event; finalize writes the redacted conversation and the
audit record for the trace.
"""

import operator
import time
import uuid
from collections.abc import Callable
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from app.agents import composer
from app.agents.critic import CriticResult, critique
from app.agents.escalation import build_handoff
from app.agents.intents import classify_intent, detect_sentiment, mentioned_version
from app.agents.planner import PRIORITY_LOW, PRIORITY_NORMAL, Plan, plan_request
from app.db import repository
from app.db.database import connect
from app.llm import LLMError
from app.rag.embeddings import EmbeddingError
from app.rag.retriever import RetrievalError
from app.security.authorization import requests_other_customer_data
from app.security.injection import detect_injection, is_suspicious
from app.security.pii import redact
from app.services import Services
from app.tools import ToolContext, assess_usage, run_tool
from app.tools.policies import active_policy_sources

MAX_REVISIONS = 1
TOP_K = 5


class SupportState(TypedDict, total=False):
    trace_id: str
    conversation_id: str
    account_id: str
    user_message: str
    redacted_message: str
    requested_version: str | None
    product_version: str | None
    version_source: str
    account: dict | None
    security_flags: list[str]
    sensitive_kinds: list[str]
    refusal_reason: str | None
    intent: str
    intent_matches: list[str]
    sentiment: str
    plan: dict
    retrieval: dict
    retrieved_context: list[dict]
    dropped_sources: list[dict]
    tool_results: dict[str, dict]
    usage_assessment: dict | None
    facts: list[str]
    draft_answer: str
    composer: str
    critic_result: dict | None
    critic_history: list[dict]
    revision_count: int
    escalation_required: bool
    escalation_reason: str | None
    priority: str | None
    handoff_id: str | None
    decision: str
    final_answer: str
    citations: list[dict]
    events: Annotated[list[dict], operator.add]
    started_at: str
    started_monotonic: float


class Citation(BaseModel):
    source_id: str
    title: str
    doc_type: str
    product_versions: list[str]
    authority_level: int


class SupportOutcome(BaseModel):
    conversation_id: str
    trace_id: str
    answer: str
    decision: str
    intent: str | None
    strategy: str
    product_version: str | None
    citations: list[Citation] = Field(default_factory=list)
    used_tools: list[str] = Field(default_factory=list)
    tool_summaries: list[str] = Field(default_factory=list)
    escalated: bool = False
    handoff_id: str | None = None
    escalation_reason: str | None = None
    priority: str | None = None
    critic: dict | None = None
    revision_count: int = 0
    security_flags: list[str] = Field(default_factory=list)
    composer: str | None = None


def _event(node: str, **detail: Any) -> dict:
    return {"node": node, "at": repository.utc_now(), **detail}


class SupportAgent:
    def __init__(self, services: Services):
        self.s = services
        self.graph = self._build()

    # ================================================================ nodes

    def security_check(self, state: SupportState) -> dict:
        message = state["user_message"]
        result = redact(message)
        flags = []
        if result.has_secrets:
            flags.append("secret_detected")
        if result.has_pii:
            flags.append("pii_detected")
        if detect_injection(message):
            flags.append("prompt_injection")
        return {
            "redacted_message": result.text,
            "security_flags": flags,
            "sensitive_kinds": result.kinds,
            "events": [_event("security_check", flags=flags, redacted_kinds=result.kinds)],
        }

    def authorize(self, state: SupportState) -> dict:
        ctx = ToolContext(state["account_id"], self.s.db_path)
        account = run_tool("lookup_account", ctx)
        if not account.ok:
            return {"account": None, "refusal_reason": "unknown_account",
                    "events": [_event("authorize", allowed=False, reason="unknown_account")]}

        message = state["redacted_message"]
        version = mentioned_version(message)
        source = "message"
        if version is None and state.get("requested_version"):
            version, source = state["requested_version"], "request"
        if version is None:
            version, source = account.data["product_version"], "account"

        cross = requests_other_customer_data(message, state["account_id"])
        update = {"account": account.data, "product_version": version, "version_source": source}
        if cross:
            flags = state.get("security_flags", []) + ["unauthorized_access_attempt"]
            return {**update, "refusal_reason": "unauthorized_account_access", "security_flags": flags,
                    "intent": "unauthorized_request",
                    "events": [_event("authorize", allowed=False, reason=cross)]}
        return {**update, "refusal_reason": None,
                "events": [_event("authorize", allowed=True, product_version=version, version_source=source)]}

    def classify(self, state: SupportState) -> dict:
        if "prompt_injection" in state.get("security_flags", []):
            intent, matched = "prompt_injection", []
        else:
            result = classify_intent(state["redacted_message"])
            intent, matched = result.intent, result.matched
        sentiment = detect_sentiment(state["redacted_message"])
        return {"intent": intent, "intent_matches": matched, "sentiment": sentiment,
                "events": [_event("classify_intent", intent=intent, sentiment=sentiment)]}

    def plan(self, state: SupportState) -> dict:
        plan = plan_request(state["intent"], state["redacted_message"], state.get("account"))
        update: dict = {"plan": plan.__dict__ | {"strategy": plan.strategy},
                        "escalation_required": plan.escalate,
                        "escalation_reason": plan.escalation_reason,
                        "priority": plan.priority if plan.escalate else None,
                        "events": [_event("plan", strategy=plan.strategy, tools=plan.tools,
                                          escalate=plan.escalate, reason=plan.escalation_reason)]}
        if plan.route == "refuse":
            update["refusal_reason"] = plan.refusal_reason
        return update

    def retrieve(self, state: SupportState) -> dict:
        plan = Plan(**{k: v for k, v in state["plan"].items() if k != "strategy"})
        version = state.get("product_version")
        query = f"{state['redacted_message']}\n{plan.retrieval_hint}".strip()
        try:
            result = self.s.retriever.retrieve(query, product_version=version, top_k=TOP_K)
            sources = [s.model_dump() for s in result.sources]
            if plan.policy_types:
                # Governing policies come from the operational registry, not from similarity.
                policy_ids = active_policy_sources(self.s.db_path, plan.policy_types)
                known = {s["source_id"] for s in sources}
                policies = [p.model_dump() for p in self.s.retriever.fetch_sources(policy_ids, query)
                            if p.source_id not in known and p.applies_to(version)]
                sources = policies + sources
            info = {"no_reliable_source": result.no_reliable_source and not sources, "message": result.message,
                    "excluded": [e.model_dump() for e in result.excluded if e.reason != "below_relevance_threshold"]}
        except (RetrievalError, EmbeddingError) as exc:
            sources, info = [], {"no_reliable_source": True, "message": "retrieval unavailable", "error": str(exc)}

        # Retrieved text is untrusted: drop sources that try to instruct the assistant.
        kept, dropped = [], []
        for source in sources:
            if is_suspicious(source["content"]):
                dropped.append({"source_id": source["source_id"], "reason": "untrusted_instructions"})
            else:
                kept.append(source)
        flags = state.get("security_flags", [])
        if dropped:
            flags = flags + ["untrusted_content_removed"]
        if not kept:
            info["no_reliable_source"] = True
        return {"retrieved_context": kept, "dropped_sources": dropped, "retrieval": info, "security_flags": flags,
                "events": [_event("retrieve", product_version=version,
                                  sources=[(s["source_id"], round(s["score"], 3)) for s in kept],
                                  dropped=dropped, no_reliable_source=info["no_reliable_source"])]}

    def run_tools(self, state: SupportState) -> dict:
        ctx = ToolContext(state["account_id"], self.s.db_path)
        results = {name: run_tool(name, ctx) for name in state["plan"]["tools"]}
        assessment = None
        if "get_usage" in results and "get_plan_limits" in results:
            assessment = assess_usage(results["get_usage"], results["get_plan_limits"])
        dumped = {name: r.model_dump() for name, r in results.items()}
        return {"tool_results": dumped, "usage_assessment": assessment,
                "facts": composer.tool_facts(dumped, assessment),
                "events": [_event("run_tools", results={n: (r.ok, r.summary) for n, r in results.items()},
                                  usage_assessment=assessment)]}

    def _generate(self, state: SupportState, feedback: list[str] | None = None) -> tuple[str, str]:
        sources, facts = state.get("retrieved_context", []), state.get("facts", [])
        if self.s.llm is not None:
            system, prompt = composer.build_prompt(
                state["redacted_message"], state.get("product_version"), sources, facts,
                escalating=bool(state.get("escalation_required")), feedback=feedback)
            try:
                return composer.clean_answer(self.s.llm.generate(system, prompt)), "llm"
            except LLMError:
                return composer.compose_extractive(state["redacted_message"], sources, facts), "extractive_fallback"
        return composer.compose_extractive(state["redacted_message"], sources, facts), "extractive"

    def compose(self, state: SupportState) -> dict:
        sources, facts = state.get("retrieved_context", []), state.get("facts", [])
        if not sources and not facts:
            if state.get("escalation_required"):
                # Already going to a human by policy; acknowledge rather than claim a knowledge gap.
                return {"draft_answer": composer.ESCALATION_ACK_TEXT, "composer": "canned", "critic_result": None,
                        "events": [_event("compose", composer="canned", reason="policy_escalation")]}
            # Policy R3: no authoritative source -> hand off instead of guessing.
            return {"draft_answer": composer.KNOWLEDGE_GAP_TEXT, "composer": "canned", "critic_result": None,
                    "escalation_required": True, "escalation_reason": "knowledge_gap", "priority": PRIORITY_LOW,
                    "events": [_event("compose", composer="canned", reason="knowledge_gap")]}
        draft, mode = self._generate(state)
        return {"draft_answer": draft, "composer": mode, "revision_count": 0,
                "events": [_event("compose", composer=mode, chars=len(draft))]}

    def critic(self, state: SupportState) -> dict:
        judge = self.s.llm if (self.s.cfg.critic_llm_judge and self.s.llm is not None) else None
        sources = state.get("retrieved_context", [])
        result = critique(
            state["draft_answer"],
            account_id=state["account_id"],
            product_version=state.get("product_version"),
            sources=sources,
            facts=state.get("facts", []),
            question=state["redacted_message"],
            requires_citation=bool(sources),
            escalation_expected=bool(state["plan"].get("escalate")),
            escalating=bool(state.get("escalation_required")),
            judge=judge,
        )
        dumped = result.model_dump()
        return {"critic_result": dumped, "critic_history": state.get("critic_history", []) + [dumped],
                "events": [_event("critic", passed=result.passed, issues=[i.code for i in result.issues],
                                  revision=state.get("revision_count", 0))]}

    def revise(self, state: SupportState) -> dict:
        feedback = CriticResult(**state["critic_result"]).feedback()
        draft, mode = self._generate(state, feedback=feedback)
        allowed = {s["source_id"] for s in state.get("retrieved_context", [])}
        draft = composer.remove_invalid_citations(draft, allowed)
        return {"draft_answer": draft, "composer": mode, "revision_count": state.get("revision_count", 0) + 1,
                "events": [_event("revise", composer=mode, feedback=feedback)]}

    def escalate(self, state: SupportState) -> dict:
        critic_result = state.get("critic_result")
        reason, priority = state.get("escalation_reason"), state.get("priority")
        draft = state["draft_answer"]
        if critic_result is not None and not critic_result["passed"]:
            # Never send an answer that failed review. Keep a policy reason if one applies.
            draft = composer.CRITIC_FAILED_TEXT
            if not state["plan"].get("escalate"):
                reason, priority = "critic_failed", priority or PRIORITY_LOW
        reason = reason or "policy_required"
        priority = priority or PRIORITY_NORMAL

        evidence = [f"[{s['source_id']}] {s['title']}" for s in state.get("retrieved_context", [])]
        evidence += [f"{name}: {r['summary']}" for name, r in state.get("tool_results", {}).items()]
        record = build_handoff(
            conversation_id=state["conversation_id"], account=state.get("account"), account_id=state["account_id"],
            reason=reason, priority=priority, intent=state.get("intent"), sentiment=state.get("sentiment", "neutral"),
            question=state["redacted_message"], product_version=state.get("product_version"),
            evidence=evidence, tools_used=list(state.get("tool_results", {})), attempted_resolution=draft,
        )
        with connect(self.s.db_path) as conn:
            repository.ensure_conversation(conn, state["conversation_id"], state["account_id"])
            repository.insert_handoff(conn, record)
        return {"handoff_id": record["handoff_id"], "escalation_required": True, "escalation_reason": reason,
                "priority": priority, "draft_answer": draft,
                "events": [_event("escalate", handoff_id=record["handoff_id"], reason=reason, priority=priority)]}

    def respond_direct(self, state: SupportState) -> dict:
        if state.get("intent") == "out_of_scope" and not state.get("refusal_reason"):
            return {"draft_answer": composer.OUT_OF_SCOPE_TEXT, "decision": "out_of_scope", "composer": "canned",
                    "events": [_event("respond_direct", decision="out_of_scope")]}
        reason = state.get("refusal_reason") or "unknown_account"
        return {"draft_answer": composer.REFUSAL_TEXTS[reason], "decision": "refused", "composer": "canned",
                "events": [_event("respond_direct", decision="refused", reason=reason)]}

    def finalize(self, state: SupportState) -> dict:
        decision = state.get("decision") or ("escalated" if state.get("handoff_id") else "answered")
        answer = state["draft_answer"].strip()
        if "secret_detected" in state.get("security_flags", []):
            answer = f"{answer}\n\n{composer.SECRET_NOTICE}"
        if state.get("handoff_id"):
            answer = f"{answer}\n\n{composer.handoff_notice(state['handoff_id'])}"
        answer = redact(answer).text  # last line of defence: nothing sensitive leaves the agent

        by_id = {s["source_id"]: s for s in state.get("retrieved_context", [])}
        citations = [
            {k: by_id[sid][k] for k in ("source_id", "title", "doc_type", "product_versions", "authority_level")}
            for sid in composer.cited_ids(answer) if sid in by_id
        ]
        tool_results = state.get("tool_results", {})
        completed = repository.utc_now()
        audit = {
            "trace_id": state["trace_id"],
            "conversation_id": state["conversation_id"],
            "account_id": state["account_id"],
            "request_redacted": state["redacted_message"],
            "product_version": state.get("product_version"),
            "intent": state.get("intent"),
            "decision": decision,
            "security_flags": state.get("security_flags", []),
            "retrieved_sources": [
                {"source_id": s["source_id"], "title": s["title"], "score": round(s["score"], 4),
                 "authority_level": s["authority_level"], "rank": s.get("rank")}
                for s in state.get("retrieved_context", [])
            ],
            "tools_used": list(tool_results),
            "tool_summaries": [f"{n}: {r['summary']}" for n, r in tool_results.items()],
            "answer": answer,
            "citations": [c["source_id"] for c in citations],
            "critic": {"final": state.get("critic_result"), "history": state.get("critic_history", [])},
            "revision_count": state.get("revision_count", 0),
            "escalated": int(bool(state.get("handoff_id"))),
            "handoff_id": state.get("handoff_id"),
            "composer": state.get("composer"),
            "events": state.get("events", []) + [_event("finalize", decision=decision)],
            "started_at": state["started_at"],
            "completed_at": completed,
            "duration_ms": int((time.monotonic() - state["started_monotonic"]) * 1000),
        }
        with connect(self.s.db_path) as conn:
            if state.get("account"):
                repository.ensure_conversation(conn, state["conversation_id"], state["account_id"])
                repository.add_message(conn, state["conversation_id"], "user", state["redacted_message"], state["trace_id"])
                repository.add_message(conn, state["conversation_id"], "assistant", answer, state["trace_id"])
                repository.insert_audit(conn, audit)
        return {"decision": decision, "final_answer": answer, "citations": citations,
                "events": [_event("finalize", decision=decision)]}

    # =============================================================== routing

    @staticmethod
    def _after_authorize(state: SupportState) -> str:
        return "respond_direct" if state.get("refusal_reason") else "classify_intent"

    @staticmethod
    def _after_plan(state: SupportState) -> str:
        plan = state["plan"]
        if plan["route"] != "proceed":
            return "respond_direct"
        if plan["needs_rag"]:
            return "retrieve"
        return "run_tools" if plan["tools"] else "compose"

    @staticmethod
    def _after_retrieve(state: SupportState) -> str:
        return "run_tools" if state["plan"]["tools"] else "compose"

    @staticmethod
    def _after_compose(state: SupportState) -> str:
        return "escalate" if state.get("composer") == "canned" else "critic"

    @staticmethod
    def _after_critic(state: SupportState) -> str:
        result = state["critic_result"]
        if not result["passed"]:
            return "revise" if state.get("revision_count", 0) < MAX_REVISIONS else "escalate"
        if state.get("escalation_required") or result["should_escalate"]:
            return "escalate"
        return "finalize"

    def _build(self):
        graph = StateGraph(SupportState)
        nodes: dict[str, Callable] = {
            "security_check": self.security_check,
            "authorize": self.authorize,
            "classify_intent": self.classify,
            "plan": self.plan,
            "retrieve": self.retrieve,
            "run_tools": self.run_tools,
            "compose": self.compose,
            "critic": self.critic,
            "revise": self.revise,
            "escalate": self.escalate,
            "respond_direct": self.respond_direct,
            "finalize": self.finalize,
        }
        for name, fn in nodes.items():
            graph.add_node(name, fn)
        graph.add_edge(START, "security_check")
        graph.add_edge("security_check", "authorize")
        graph.add_conditional_edges("authorize", self._after_authorize, ["respond_direct", "classify_intent"])
        graph.add_edge("classify_intent", "plan")
        graph.add_conditional_edges("plan", self._after_plan, ["respond_direct", "retrieve", "run_tools", "compose"])
        graph.add_conditional_edges("retrieve", self._after_retrieve, ["run_tools", "compose"])
        graph.add_edge("run_tools", "compose")
        graph.add_conditional_edges("compose", self._after_compose, ["escalate", "critic"])
        graph.add_conditional_edges("critic", self._after_critic, ["revise", "escalate", "finalize"])
        graph.add_edge("revise", "critic")
        graph.add_edge("escalate", "finalize")
        graph.add_edge("respond_direct", "finalize")
        graph.add_edge("finalize", END)
        return graph.compile()

    # ================================================================== API

    def run(self, *, account_id: str, message: str, conversation_id: str | None = None,
            product_version: str | None = None) -> SupportOutcome:
        conversation_id = conversation_id or f"conv-{uuid.uuid4().hex[:12]}"
        with connect(self.s.db_path) as conn:
            owner = repository.conversation_owner(conn, conversation_id)
        if owner is not None and owner != account_id:
            raise repository.OwnershipError("conversation belongs to another account")

        initial: SupportState = {
            "trace_id": f"trace-{uuid.uuid4().hex}",
            "conversation_id": conversation_id,
            "account_id": account_id,
            "user_message": message,
            "requested_version": product_version,
            "tool_results": {},
            "facts": [],
            "retrieved_context": [],
            "critic_history": [],
            "revision_count": 0,
            "events": [],
            "started_at": repository.utc_now(),
            "started_monotonic": time.monotonic(),
        }
        final = self.graph.invoke(initial)
        tool_results = final.get("tool_results", {})
        critic_result = final.get("critic_result")
        plan = final.get("plan") or {}
        return SupportOutcome(
            conversation_id=conversation_id,
            trace_id=final["trace_id"],
            answer=final["final_answer"],
            decision=final["decision"],
            intent=final.get("intent"),
            strategy=plan.get("strategy", "refuse" if final["decision"] == "refused" else final["decision"]),
            product_version=final.get("product_version"),
            citations=[Citation(**c) for c in final.get("citations", [])],
            used_tools=list(tool_results),
            tool_summaries=[f"{n}: {r['summary']}" for n, r in tool_results.items()],
            escalated=bool(final.get("handoff_id")),
            handoff_id=final.get("handoff_id"),
            escalation_reason=final.get("escalation_reason") if final.get("handoff_id") else None,
            priority=final.get("priority") if final.get("handoff_id") else None,
            critic={"passed": critic_result["passed"], "issues": [i["code"] for i in critic_result["issues"]]}
            if critic_result else None,
            revision_count=final.get("revision_count", 0),
            security_flags=final.get("security_flags", []),
            composer=final.get("composer"),
        )
