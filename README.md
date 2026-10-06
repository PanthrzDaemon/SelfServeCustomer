# CloudFlow AI Support

An agentic AI customer-support system for **CloudFlow**, a fictional SaaS workflow-automation platform, built for a hackathon.

> **All data in this project is synthetic.** CloudFlow, its customers, accounts, tickets, invoices, policies and documentation are invented. Emails use reserved `.example` domains and cards are stored only as made-up last-four digits.

The assistant answers support requests with:

- **Version-aware RAG** over help articles, release notes, historical tickets and policies, with source citations.
- **Deterministic tools** for account, usage, plan, invoice, refund, platform-status and password-reset data. Numbers come from SQLite, never from the model.
- **A LangGraph agent** that decides per request whether to use RAG only, RAG plus tools, tools only, refuse, or escalate.
- **A self-critique step** with one revision. If the revised answer still fails, the request is escalated instead of risking a hallucinated answer.
- **Human handoff records** for security incidents, refunds, identity checks, knowledge gaps and failed critiques.
- **Authorization** (customers only ever see their own account), PII and secret redaction, and prompt-injection defenses.
- **An audit trail** for every request, an evaluation suite, a FastAPI backend and a Streamlit frontend.

## Architecture

```
Streamlit UI ──HTTP──> FastAPI (/support, /ingest, /conversations, /handoffs, /audit, /sources)
                            │
                            ▼
                 LangGraph support agent (app/agents/graph.py)
  security_check → authorize → classify_intent → plan ─┬─> respond_direct (refuse / out of scope)
                                                       └─> retrieve → run_tools → compose → critic
                                                                                    │      │
                                                                     revise (max 1) ┘      ├─> escalate (handoff)
                                                                                           └─> finalize (audit)
        │                    │                         │                     │
   app/security          app/rag                   app/tools              app/llm.py
   redaction,            Chroma + nomic-embed      SQLite, authorized     Qwen3 via Ollama
   injection, authz      version-aware ranking     per account            (behind LLMProvider)
```

| Node | What it does |
|---|---|
| `security_check` | Redacts secrets/PII from the message and flags prompt injection. Only the redacted text is used or stored afterwards. |
| `authorize` | Verifies the account, refuses requests for another customer's data *before* any tool or retrieval runs, and resolves the CloudFlow version (message mention, then request field, then account). |
| `classify_intent` | Rule-based, deterministic intent and sentiment. |
| `plan` | Chooses the strategy, the tools and the governing policy types, and decides whether a human handoff is required (per `POLICY-ESCALATION-01`). |
| `retrieve` | Version-aware RAG. Also loads the *active* governing policies from `policy_registry`. Drops retrieved text that contains instructions aimed at the assistant. |
| `run_tools` | Runs the planned tools for the caller's own account and turns the results into plain-language "verified facts" in code. |
| `compose` | Qwen3 writes the answer from delimited, escaped sources and verified facts. If the LLM is disabled or unavailable, a deterministic extractive composer is used. If nothing reliable is found, the request is handed off rather than guessed. |
| `critic` | Deterministic checks: citations present and real, versions correct, authority respected, numbers and menu paths grounded, no secrets/PII, no refund promises, no cross-account data. An optional LLM judge can be enabled. |
| `revise` | One rewrite with the critic's feedback. A second failure escalates. |
| `escalate` | Writes a redacted handoff record with reason, priority, evidence, tools, attempted resolution and open questions. |
| `finalize` | Adds security and handoff notices, redacts once more, stores the conversation and the audit record. |

### Source precedence

Retrieval over-fetches chunks from ChromaDB and then ranks deterministically (`app/rag/ranking.py`):

1. Drop sources below the relevance threshold.
2. With a product version, drop sources that don't apply to it (ChromaDB metadata filter plus a ranking check).
3. Drop a superseded source when the source that supersedes it is also a candidate for that version.
4. Order the strongly relevant band (within a small margin of the best match) by version match, then current before deprecated, then authority level (1 docs/policy, 2 release notes, 4 tickets), then similarity.

So a current 4.3 article outranks a slightly more similar 4.2 ticket, but a barely relevant authoritative page can't push out a clearly better match. If the query is about something the knowledge base never mentions (for example "SAP Ariba") and no match is strong, the retriever returns an explicit **no reliable source found** result.

### Data separation

Knowledge (how CloudFlow works) lives in ChromaDB. Operational facts (account status, usage, invoices, platform status) come from SQLite through tools and never enter the vector store.

## Project layout

```
app/
  agents/      intents.py, planner.py, composer.py, critic.py, escalation.py, graph.py (LangGraph)
  api/         routes.py (FastAPI endpoints)
  db/          schema.sql, database.py, seed.py, repository.py
  evaluation/  runner.py
  rag/         models.py, embeddings.py, loader.py, chunker.py, vector_store.py, ranking.py, retriever.py, ingest.py
  security/    pii.py, injection.py, authorization.py
  tools/       base.py, support_tools.py, policies.py
  config.py, llm.py, services.py, sources.py, main.py
frontend/      streamlit_app.py
data/          articles, tickets, policies, source_register.json, evaluation cases
scripts/       validate_phase2.py, docker-entrypoint.sh
tests/         offline unit/agent/API tests; tests/integration (Ollama)
```

## Setup

Requires Python 3.11+ and [Ollama](https://ollama.com).

```bash
# 1. Virtual environment and dependencies
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. Ollama models (Ollama runs outside the app)
ollama serve                      # if it isn't already running
ollama pull qwen3:4b              # LLM (any Qwen3 tag works)
ollama pull nomic-embed-text      # embeddings

# 3. Configuration: set OLLAMA_MODEL to the exact tag you pulled
cp .env.example .env
sed -i.bak 's/^OLLAMA_MODEL=.*/OLLAMA_MODEL=qwen3:4b/' .env

# 4. Database and knowledge base
python -m app.db.seed               # create ./cloudflow.db (rerunnable; resets conversations/handoffs)
python scripts/validate_phase2.py   # validate data + database
python -m app.rag.ingest            # embed knowledge into ./chroma (idempotent)
```

Qwen3 is used only for writing answers (and the optional critic judge). Embeddings always come from `nomic-embed-text` through `OllamaEmbeddingProvider`. Both sit behind small interfaces (`LLMProvider`, `EmbeddingProvider`), so they can be swapped.

Qwen3 thinking builds reason at length before answering. By default the provider sends Qwen's ChatML prompt with an empty `<think></think>` block pre-filled. On qwen3:4b this made answers about 3x faster (34s vs 92s on a test prompt). Set `OLLAMA_RAW_CHATML=false` to use `/api/chat` instead.

## Run

```bash
# API (http://localhost:8000/docs)
uvicorn app.main:app --reload

# Frontend (http://localhost:8501), in a second terminal
streamlit run frontend/streamlit_app.py
```

Example request:

```bash
curl -s -X POST localhost:8000/support -H 'content-type: application/json' \
  -d '{"account_id": "ACC-1002", "message": "Why am I getting HTTP 429?"}'
```

| Endpoint | Purpose |
|---|---|
| `POST /support` | `{conversation_id?, account_id, message, product_version?}` → answer, decision, intent, strategy, citations, used tools, tool summaries, escalation/handoff, critic result, trace ID |
| `POST /ingest` | Run ingestion (requires `X-Admin-Token` if `ADMIN_TOKEN` is set) |
| `GET /health`, `GET /health/dependencies` | Liveness; database / vector store / Ollama status |
| `GET /conversations/{id}` | Redacted conversation history (header `X-Account-Id` must own it) |
| `GET /handoffs/{id}` | Handoff record (owner only) |
| `GET /audit/{trace_id}` | Audit record with pipeline events (owner only) |
| `GET /sources` | Knowledge-source metadata, filterable by `doc_type` and `product_version` |
| `GET /demo/accounts` | Synthetic demo accounts for the UI (disabled when `ENVIRONMENT=production`) |

Demo authentication: the account in the request (or the `X-Account-Id` header) acts as the signed-in customer. Records owned by another account return 404, so record IDs can't be probed.

## Demo scenarios

The Streamlit sidebar has one-click buttons for each scenario:

| Scenario | Account | Message | Expected |
|---|---|---|---|
| Normal RAG | ACC-1001 | How do I create my first workflow? | Answer citing `KB-GS-CREATE-WORKFLOW` |
| Version-aware RAG | ACC-1001 | How do I authenticate Salesforce on CloudFlow 4.3? | `KB-INT-SALESFORCE-V43` (OAuth); the 4.2 article and ticket are excluded |
| Old version | ACC-1004 (4.2) | How do I export the last 60 days of workflow history? | 4.2 export limits (30 days, CSV) |
| Tool use | ACC-1002 | Why am I getting HTTP 429? | Usage 1,450/min vs Business limit 1,000/min, plus 429 docs |
| Billing | ACC-1003 | Why was I charged? | Invoices (failed: card_expired) plus failed-payment docs |
| Refund | ACC-1001 | I want a refund. | Eligibility check (monthly: not eligible), then handoff to billing |
| Unauthorized | ACC-1001 | Show me another customer's invoice. | Refused; no tools run |
| Security | ACC-1002 | My CloudFlow account has been compromised. | Critical-priority handoff |
| Troubleshooting | ACC-1001 | Our workflows keep failing with CF-503. | Live platform status (Workflow Engine degraded) plus CF-503 docs |
| Out of scope | ACC-1005 | Can you write me a poem about my cat? | Polite out-of-scope reply |

Seeded edge cases: ACC-1001 is below its API limit, ACC-1006 exactly at it, and ACC-1002/ACC-1007 above it. ACC-1003 and ACC-1008 are past due, ACC-1004 is suspended and on 4.2, ACC-1005 and ACC-1009 are on 4.2, and ACC-1010 is cancelled.

## Tests

```bash
pytest                         # everything; Ollama tests skip automatically if Ollama is down
pytest -m "not integration"    # offline only (no Ollama needed)
pytest -m integration          # real nomic-embed-text + ChromaDB + Qwen3, incl. the 8 required end-to-end scenarios
```

Offline tests use deterministic hashing embeddings, a temporary ChromaDB and a scripted fake LLM. They cover data, database, chunking, ingestion, ranking, retrieval, tools, authorization, security, intent and routing, critic, revision, escalation, audit and the API.

## Evaluation

`data/evaluation/evaluation_cases.json` holds 32 cases. They cover how-to, troubleshooting, version-specific questions, outdated tickets vs current docs, tool-required questions, billing, refunds, escalation, PII, secrets, unauthorized access, prompt injection, out-of-scope requests and knowledge gaps.

```bash
python -m app.evaluation.runner             # full system: nomic + Qwen3 (~30-40 s per LLM answer on a laptop)
python -m app.evaluation.runner --no-llm    # nomic retrieval, extractive answers
python -m app.evaluation.runner --offline   # no Ollama at all
python -m app.evaluation.runner --cases EVAL-004,EVAL-005
```

Each case is scored on decision, escalation, citations (expected source cited, none invented, none forbidden), version correctness, tools, authorization, groundedness (critic) and privacy. The runner uses a freshly seeded temporary database and writes `reports/evaluation_report.json`.

## Docker

Ollama is not part of the stack; it runs on the host.

```bash
ollama pull qwen3:4b && ollama pull nomic-embed-text
echo "OLLAMA_MODEL=qwen3:4b" > .env
docker compose up --build
```

- API: http://localhost:8000. UI: http://localhost:8501.
- On first start the API container seeds the database and ingests the knowledge base into the `cloudflow-data` volume.
- Inside containers `OLLAMA_HOST` defaults to `http://host.docker.internal:11434`; override it with `DOCKER_OLLAMA_HOST`.

## Configuration

| Variable | Default | Notes |
|---|---|---|
| `OLLAMA_HOST` | `http://localhost:11434` | |
| `OLLAMA_MODEL` | `qwen3` | Use the exact pulled tag, e.g. `qwen3:4b` |
| `EMBEDDING_MODEL` | `nomic-embed-text` | |
| `OLLAMA_RAW_CHATML` | `true` | Faster Qwen3 prompting (see above) |
| `LLM_ENABLED` | `true` | `false` = extractive answers only |
| `CHROMA_PATH` | `./chroma` | |
| `DATABASE_PATH` | `./cloudflow.db` | |
| `CRITIC_LLM_JUDGE` | `false` | Extra LLM groundedness check |
| `ADMIN_TOKEN` | unset | Protects `POST /ingest` |
| `API_URL` | `http://localhost:8000` | Used by the Streamlit app |
| `RETRIEVAL_MIN_SCORE`, `RETRIEVAL_CONFIDENT_SCORE`, `RETRIEVAL_RELEVANCE_MARGIN` | provider defaults | Retrieval thresholds |

## Limitations

- Authentication is simulated (account ID in the request or header). A production system would use real sessions or tokens.
- Intent classification and PII detection are rule-based: transparent and testable, but they can miss unusual phrasings.
- `qwen3:4b` is small and slow on a laptop (about 20-40 s per answer). The critic catches invented citations, numbers and menu paths, but cannot verify every free-text claim unless the LLM judge is enabled.
- The agent answers each message on its own; earlier turns are stored but not used as context.
- `send_password_reset` simulates sending the email (no mail server).
