"""Deterministic tools backed by SQLite.

Every account-scoped tool checks authorization before touching the database, so
another customer's data is never read. Tools return structured ToolResults;
the LLM only ever sees these results, never raw database access.
"""

from collections.abc import Callable

from app.db.database import connect, fetch_all, fetch_one
from app.tools.base import NOT_FOUND, ToolContext, ToolResult, authorize, forbidden, mask_email

REFUND_POLICY_ID = "POLICY-REFUND-01"
ANNUAL_REFUND_WINDOW_DAYS = 14
ANNUAL_DISCOUNT = 0.85


def _account_row(conn, account_id: str):
    return fetch_one(conn, "SELECT * FROM accounts WHERE account_id = ?", (account_id,))


def lookup_account(ctx: ToolContext, account_id: str | None = None) -> ToolResult:
    name = "lookup_account"
    if not authorize(ctx, account_id):
        return forbidden(name)
    with connect(ctx.db_path) as conn:
        row = _account_row(conn, ctx.account_id)
    if row is None:
        return ToolResult(tool=name, ok=False, error=NOT_FOUND, summary="Account not found.")
    data = {
        "account_id": row["account_id"],
        "company_name": row["company_name"],
        "owner_email_masked": mask_email(row["owner_email"]),
        "plan": row["plan"],
        "status": row["status"],
        "product_version": row["product_version"],
        "created_at": row["created_at"],
    }
    return ToolResult(tool=name, ok=True, data=data,
                      summary=f"{row['account_id']}: {row['plan']} plan, status {row['status']}, "
                              f"CloudFlow {row['product_version']}")


def get_plan_limits(ctx: ToolContext, plan: str | None = None) -> ToolResult:
    """Plan limits are public; defaults to the caller's plan."""
    name = "get_plan_limits"
    with connect(ctx.db_path) as conn:
        if plan is None:
            account = _account_row(conn, ctx.account_id)
            if account is None:
                return ToolResult(tool=name, ok=False, error=NOT_FOUND, summary="Account not found.")
            plan = account["plan"]
        row = fetch_one(conn, "SELECT * FROM plan_limits WHERE plan = ?", (plan,))
    if row is None:
        return ToolResult(tool=name, ok=False, error=NOT_FOUND, summary=f"Unknown plan {plan!r}.")
    data = dict(row)
    return ToolResult(tool=name, ok=True, data=data,
                      summary=f"{plan}: {data['api_rate_limit_per_min']:,} API requests/min, "
                              f"{data['monthly_workflow_runs']:,} runs/month, {data['seats']} seats")


def get_usage(ctx: ToolContext, account_id: str | None = None, period: str | None = None) -> ToolResult:
    name = "get_usage"
    if not authorize(ctx, account_id):
        return forbidden(name)
    with connect(ctx.db_path) as conn:
        periods = [r["period"] for r in fetch_all(
            conn, "SELECT period FROM usage WHERE account_id = ? ORDER BY period DESC", (ctx.account_id,))]
        if not periods:
            return ToolResult(tool=name, ok=False, error=NOT_FOUND, summary="No usage recorded.")
        period = period or periods[0]
        row = fetch_one(conn, "SELECT * FROM usage WHERE account_id = ? AND period = ?", (ctx.account_id, period))
    if row is None:
        return ToolResult(tool=name, ok=False, error=NOT_FOUND, summary=f"No usage for period {period}.")
    data = {**dict(row), "periods_available": periods}
    return ToolResult(tool=name, ok=True, data=data,
                      summary=f"{period}: {row['workflow_runs']:,} runs, peak {row['peak_requests_per_min']:,} "
                              f"API requests/min (average {row['api_requests_per_min']:,})")


def assess_usage(usage: ToolResult, limits: ToolResult) -> dict | None:
    """Compare usage to plan limits. Pure function, so the numbers are never the LLM's."""
    if not (usage.ok and limits.ok):
        return None
    peak = usage.data["peak_requests_per_min"]
    limit = limits.data["api_rate_limit_per_min"]
    runs = usage.data["workflow_runs"]
    quota = limits.data["monthly_workflow_runs"]

    def status(value: int, cap: int) -> str:
        return "above_limit" if value > cap else "at_limit" if value == cap else "below_limit"

    return {
        "period": usage.data["period"],
        "plan": limits.data["plan"],
        "peak_requests_per_min": peak,
        "api_rate_limit_per_min": limit,
        "api_limit_status": status(peak, limit),
        "api_peak_percent_of_limit": round(100 * peak / limit, 1),
        "workflow_runs": runs,
        "monthly_workflow_runs": quota,
        "run_quota_status": status(runs, quota),
    }


def get_invoices(ctx: ToolContext, account_id: str | None = None, limit: int = 6) -> ToolResult:
    name = "get_invoices"
    if not authorize(ctx, account_id):
        return forbidden(name)
    with connect(ctx.db_path) as conn:
        rows = fetch_all(conn, """
            SELECT invoice_id, amount, currency, charged_on, status, failure_reason, card_last4
            FROM invoices WHERE account_id = ? ORDER BY charged_on DESC, invoice_id DESC LIMIT ?
        """, (ctx.account_id, max(1, min(limit, 24))))
    invoices = [dict(r) for r in rows]
    failed = [i for i in invoices if i["status"] == "failed"]
    summary = f"{len(invoices)} invoice(s)"
    if invoices:
        latest = invoices[0]
        summary += (f"; latest {latest['invoice_id']} {latest['amount']:.2f} {latest['currency']} "
                    f"on {latest['charged_on']} ({latest['status']})")
    if failed:
        summary += f"; {len(failed)} failed"
    return ToolResult(tool=name, ok=True, data={"invoices": invoices}, summary=summary)


def check_refund_eligibility(ctx: ToolContext, invoice_id: str | None = None) -> ToolResult:
    """Apply POLICY-REFUND-01 deterministically. Never issues a refund."""
    name = "check_refund_eligibility"
    with connect(ctx.db_path) as conn:
        account = _account_row(conn, ctx.account_id)
        if account is None:
            return ToolResult(tool=name, ok=False, error=NOT_FOUND, summary="Account not found.")
        if invoice_id:
            invoice = fetch_one(conn, "SELECT * FROM invoices WHERE invoice_id = ?", (invoice_id,))
            # Another account's invoice is reported exactly like a missing one.
            if invoice is None or invoice["account_id"] != ctx.account_id:
                return ToolResult(tool=name, ok=False, error=NOT_FOUND,
                                  summary="No invoice with that ID on this account.")
        else:
            invoice = fetch_one(conn, """
                SELECT * FROM invoices WHERE account_id = ? AND status = 'paid'
                ORDER BY charged_on DESC, invoice_id DESC LIMIT 1
            """, (ctx.account_id,))
        plan = fetch_one(conn, "SELECT monthly_price FROM plan_limits WHERE plan = ?", (account["plan"],))
        duplicates = 0
        if invoice is not None:
            duplicates = fetch_one(conn, """
                SELECT COUNT(*) AS n FROM invoices
                WHERE account_id = ? AND status = 'paid' AND substr(charged_on, 1, 7) = substr(?, 1, 7)
            """, (ctx.account_id, invoice["charged_on"]))["n"]

    base = {"policy_source": REFUND_POLICY_ID, "requires_human_approval": True}
    if invoice is None:
        data = {**base, "invoice_id": None, "eligible": False, "reason_code": "no_paid_invoice",
                "explanation": "There is no paid invoice on this account to refund."}
    else:
        annual_price = round(plan["monthly_price"] * 12 * ANNUAL_DISCOUNT, 2) if plan else None
        billing_interval = "annual" if annual_price and abs(invoice["amount"] - annual_price) < 0.01 else "monthly"
        data = {**base, "invoice_id": invoice["invoice_id"], "amount": invoice["amount"],
                "currency": invoice["currency"], "charged_on": invoice["charged_on"],
                "invoice_status": invoice["status"], "billing_interval": billing_interval}
        if invoice["status"] == "refunded":
            data.update(eligible=False, reason_code="already_refunded",
                        explanation="This invoice has already been refunded.")
        elif invoice["status"] != "paid":
            data.update(eligible=False, reason_code="not_paid",
                        explanation=f"This invoice was not paid (status: {invoice['status']}), so there is nothing to refund.")
        elif duplicates > 1:
            data.update(eligible=True, reason_code="billing_error_duplicate",
                        explanation="More than one paid invoice exists for the same period, which is a billing error eligible for a full refund.")
        elif billing_interval == "annual":
            data.update(eligible=True, reason_code="annual_within_window_check",
                        explanation=f"Annual plans can be fully refunded within {ANNUAL_REFUND_WINDOW_DAYS} days of the first annual charge; a billing specialist confirms the date.")
        else:
            data.update(eligible=False, reason_code="monthly_not_refundable",
                        explanation="Monthly plans are not refunded for partial or unused periods. Cancelling stops future charges.")

    return ToolResult(tool=name, ok=True, data=data,
                      summary=f"{data.get('invoice_id') or 'no invoice'}: "
                              f"{'eligible' if data['eligible'] else 'not eligible'} ({data['reason_code']}); "
                              "human approval required")


def check_platform_status(ctx: ToolContext, component: str | None = None) -> ToolResult:
    name = "check_platform_status"
    with connect(ctx.db_path) as conn:
        if component:
            rows = fetch_all(conn, "SELECT * FROM platform_status WHERE lower(component) = lower(?)", (component,))
        else:
            rows = fetch_all(conn, "SELECT * FROM platform_status ORDER BY component")
    components = [dict(r) for r in rows]
    if not components:
        return ToolResult(tool=name, ok=False, error=NOT_FOUND, summary=f"Unknown component {component!r}.")
    impaired = [c for c in components if c["status"] != "operational"]
    summary = ("All components operational" if not impaired else
               "; ".join(f"{c['component']}: {c['status']}" for c in impaired))
    return ToolResult(tool=name, ok=True, data={"components": components, "impaired": impaired}, summary=summary)


def send_password_reset(ctx: ToolContext, account_id: str | None = None) -> ToolResult:
    """Trigger the self-service reset email to the registered owner address.

    Returns only a masked address. Never returns, generates or accepts a password
    or a reset link (POLICY-PASSWORD-RESET-01).
    """
    name = "send_password_reset"
    if not authorize(ctx, account_id):
        return forbidden(name)
    with connect(ctx.db_path) as conn:
        account = _account_row(conn, ctx.account_id)
    if account is None:
        return ToolResult(tool=name, ok=False, error=NOT_FOUND, summary="Account not found.")
    if account["status"] in ("suspended", "cancelled"):
        return ToolResult(tool=name, ok=False, error="account_inactive",
                          summary=f"Reset not sent: workspace is {account['status']}.")
    masked = mask_email(account["owner_email"])
    data = {"status": "reset_email_sent", "sent_to": masked, "link_expires_minutes": 60, "single_use": True}
    return ToolResult(tool=name, ok=True, data=data, summary=f"Password reset email sent to {masked}")


TOOLS: dict[str, Callable[..., ToolResult]] = {
    "lookup_account": lookup_account,
    "get_usage": get_usage,
    "get_plan_limits": get_plan_limits,
    "get_invoices": get_invoices,
    "check_refund_eligibility": check_refund_eligibility,
    "check_platform_status": check_platform_status,
    "send_password_reset": send_password_reset,
}


def run_tool(name: str, ctx: ToolContext, **kwargs) -> ToolResult:
    tool = TOOLS.get(name)
    if tool is None:
        return ToolResult(tool=name, ok=False, error="unknown_tool", summary=f"Unknown tool {name!r}.")
    return tool(ctx, **kwargs)
