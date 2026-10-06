-- CloudFlow AI Support: operational database (SQLite).
-- All data is synthetic. Dates are ISO 8601 text; timestamps are UTC ("...Z").

CREATE TABLE IF NOT EXISTS plan_limits (
    plan                    TEXT PRIMARY KEY
                            CHECK (plan IN ('Free', 'Pro', 'Business', 'Enterprise')),
    api_rate_limit_per_min  INTEGER NOT NULL CHECK (api_rate_limit_per_min > 0),
    monthly_workflow_runs   INTEGER NOT NULL CHECK (monthly_workflow_runs > 0),
    seats                   INTEGER NOT NULL CHECK (seats > 0),
    support_tier            TEXT    NOT NULL,
    monthly_price           REAL    NOT NULL CHECK (monthly_price >= 0)
);

CREATE TABLE IF NOT EXISTS accounts (
    account_id       TEXT PRIMARY KEY CHECK (account_id GLOB 'ACC-[0-9][0-9][0-9][0-9]'),
    company_name     TEXT NOT NULL,
    owner_email      TEXT NOT NULL,
    plan             TEXT NOT NULL REFERENCES plan_limits (plan),
    status           TEXT NOT NULL
                     CHECK (status IN ('active', 'past_due', 'suspended', 'cancelled')),
    product_version  TEXT NOT NULL CHECK (product_version IN ('4.2', '4.3')),
    created_at       TEXT NOT NULL
);

-- One row per account per billing period (YYYY-MM).
-- api_requests_per_min is the period average; peak_requests_per_min is the
-- highest count in any 60-second window and is what is compared to the plan limit.
CREATE TABLE IF NOT EXISTS usage (
    account_id             TEXT    NOT NULL REFERENCES accounts (account_id),
    period                 TEXT    NOT NULL CHECK (period GLOB '[0-9][0-9][0-9][0-9]-[0-9][0-9]'),
    workflow_runs          INTEGER NOT NULL CHECK (workflow_runs >= 0),
    api_requests_per_min   INTEGER NOT NULL CHECK (api_requests_per_min >= 0),
    peak_requests_per_min  INTEGER NOT NULL CHECK (peak_requests_per_min >= api_requests_per_min),
    PRIMARY KEY (account_id, period)
);

CREATE TABLE IF NOT EXISTS invoices (
    invoice_id      TEXT PRIMARY KEY,
    account_id      TEXT NOT NULL REFERENCES accounts (account_id),
    amount          REAL NOT NULL CHECK (amount >= 0),
    currency        TEXT NOT NULL CHECK (length(currency) = 3),
    charged_on      TEXT NOT NULL,
    status          TEXT NOT NULL CHECK (status IN ('paid', 'failed', 'refunded', 'open')),
    failure_reason  TEXT,
    -- Only the last four digits are ever stored; never a full card number.
    card_last4      TEXT CHECK (card_last4 IS NULL OR card_last4 GLOB '[0-9][0-9][0-9][0-9]'),
    CHECK ((status = 'failed') = (failure_reason IS NOT NULL))
);

CREATE INDEX IF NOT EXISTS idx_invoices_account ON invoices (account_id);

CREATE TABLE IF NOT EXISTS platform_status (
    component   TEXT PRIMARY KEY,
    status      TEXT NOT NULL CHECK (status IN (
                    'operational', 'degraded', 'partial_outage', 'major_outage', 'maintenance')),
    message     TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);

-- Operational state for policies. The policy documents in data/policies/ remain
-- the source of truth for policy content; source_id points at that document.
CREATE TABLE IF NOT EXISTS policy_registry (
    policy_id       TEXT PRIMARY KEY,
    policy_name     TEXT    NOT NULL,
    description     TEXT    NOT NULL,
    source_id       TEXT    NOT NULL UNIQUE,
    policy_type     TEXT    NOT NULL,
    active          INTEGER NOT NULL CHECK (active IN (0, 1)),
    effective_from  TEXT    NOT NULL,
    effective_to    TEXT    CHECK (effective_to IS NULL OR effective_to >= effective_from),
    version         INTEGER NOT NULL CHECK (version >= 1),
    created_at      TEXT    NOT NULL,
    updated_at      TEXT    NOT NULL
);

-- Human handoff records (written by the escalation workflow in a later phase).
-- Rows hold redacted summaries only: never passwords, API keys, tokens or other
-- secrets. pii_redacted must be 1, so a record can only be written after redaction.
-- evidence, tools_used and unresolved_questions are JSON arrays.
CREATE TABLE IF NOT EXISTS handoffs (
    handoff_id            TEXT PRIMARY KEY,
    conversation_id       TEXT NOT NULL,
    account_id            TEXT REFERENCES accounts (account_id),
    reason                TEXT NOT NULL,
    priority              TEXT NOT NULL DEFAULT 'normal'
                          CHECK (priority IN ('low', 'normal', 'high', 'critical')),
    intent                TEXT,
    sentiment             TEXT CHECK (sentiment IS NULL OR sentiment IN ('positive', 'neutral', 'negative')),
    customer_summary      TEXT NOT NULL,
    evidence              TEXT NOT NULL DEFAULT '[]' CHECK (json_valid(evidence) AND json_type(evidence) = 'array'),
    tools_used            TEXT NOT NULL DEFAULT '[]' CHECK (json_valid(tools_used) AND json_type(tools_used) = 'array'),
    attempted_resolution  TEXT,
    unresolved_questions  TEXT NOT NULL DEFAULT '[]'
                          CHECK (json_valid(unresolved_questions) AND json_type(unresolved_questions) = 'array'),
    pii_redacted          INTEGER NOT NULL CHECK (pii_redacted = 1),
    status                TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'open', 'resolved')),
    created_at            TEXT NOT NULL,
    updated_at            TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_handoffs_status ON handoffs (status);
CREATE INDEX IF NOT EXISTS idx_handoffs_account ON handoffs (account_id);

-- Conversation history. Message content is stored redacted (no secrets or PII).
CREATE TABLE IF NOT EXISTS conversations (
    conversation_id  TEXT PRIMARY KEY,
    account_id       TEXT NOT NULL REFERENCES accounts (account_id),
    created_at       TEXT NOT NULL,
    updated_at       TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS messages (
    message_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id  TEXT NOT NULL REFERENCES conversations (conversation_id),
    role             TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    content          TEXT NOT NULL,
    trace_id         TEXT,
    created_at       TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_messages_conversation ON messages (conversation_id, message_id);

-- One audit record per support request (trace). Text fields are redacted;
-- tool results are stored as one-line summaries, not raw data.
CREATE TABLE IF NOT EXISTS audit_log (
    trace_id           TEXT PRIMARY KEY,
    conversation_id    TEXT NOT NULL,
    account_id         TEXT NOT NULL,
    request_redacted   TEXT NOT NULL,
    product_version    TEXT,
    intent             TEXT,
    decision           TEXT NOT NULL CHECK (decision IN ('answered', 'escalated', 'refused', 'out_of_scope')),
    security_flags     TEXT NOT NULL DEFAULT '[]' CHECK (json_valid(security_flags)),
    retrieved_sources  TEXT NOT NULL DEFAULT '[]' CHECK (json_valid(retrieved_sources)),
    tools_used         TEXT NOT NULL DEFAULT '[]' CHECK (json_valid(tools_used)),
    tool_summaries     TEXT NOT NULL DEFAULT '[]' CHECK (json_valid(tool_summaries)),
    answer             TEXT NOT NULL,
    citations          TEXT NOT NULL DEFAULT '[]' CHECK (json_valid(citations)),
    critic             TEXT NOT NULL DEFAULT '{}' CHECK (json_valid(critic)),
    revision_count     INTEGER NOT NULL DEFAULT 0,
    escalated          INTEGER NOT NULL DEFAULT 0 CHECK (escalated IN (0, 1)),
    handoff_id         TEXT,
    composer           TEXT,
    events             TEXT NOT NULL DEFAULT '[]' CHECK (json_valid(events)),
    started_at         TEXT NOT NULL,
    completed_at       TEXT NOT NULL,
    duration_ms        INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_audit_conversation ON audit_log (conversation_id);
