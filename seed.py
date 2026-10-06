"""Create and seed the synthetic CloudFlow database.

Usage: python -m app.db.seed [path/to/cloudflow.db]

The database is rebuilt from scratch in a temporary file and then swapped into
place, so the seed is deterministic and safe to rerun. Every value is synthetic:
company names are invented, emails use the reserved .example domain, and cards
are stored only as made-up last-four digits.
"""

import os
import sys
from pathlib import Path

from app.config import settings
from app.db.database import connect, init_schema
from app.sources import DATA_DIR, SOURCE_FILES, load_json

# plan, api_rate_limit_per_min, monthly_workflow_runs, seats, support_tier, monthly_price
PLANS = [
    ("Free", 60, 1_000, 2, "community", 0.00),
    ("Pro", 300, 25_000, 10, "standard", 49.00),
    ("Business", 1_000, 250_000, 50, "priority", 299.00),
    ("Enterprise", 5_000, 2_000_000, 500, "dedicated", 1_999.00),
]

# account_id, company_name, owner_email, plan, status, product_version, created_at
ACCOUNTS = [
    ("ACC-1001", "Brightpath Logistics", "owner@brightpath.example", "Business", "active", "4.3", "2024-03-14T10:00:00Z"),
    ("ACC-1002", "Lumen Analytics", "owner@lumen-analytics.example", "Business", "active", "4.3", "2024-06-02T09:30:00Z"),
    ("ACC-1003", "Copperleaf Studio", "owner@copperleaf.example", "Pro", "past_due", "4.3", "2025-01-20T15:45:00Z"),
    ("ACC-1004", "Halcyon Health Systems", "owner@halcyon-health.example", "Enterprise", "suspended", "4.2", "2023-11-08T12:00:00Z"),
    ("ACC-1005", "Pebble and Pine", "owner@pebbleandpine.example", "Free", "active", "4.2", "2025-08-30T18:20:00Z"),
    ("ACC-1006", "Quillstone Media", "owner@quillstone.example", "Pro", "active", "4.3", "2025-02-11T08:15:00Z"),
    ("ACC-1007", "Tidewater Labs", "owner@tidewater-labs.example", "Free", "active", "4.3", "2026-05-05T13:00:00Z"),
    ("ACC-1008", "Orchard Lane Retail", "owner@orchardlane.example", "Business", "past_due", "4.3", "2024-09-17T11:10:00Z"),
    ("ACC-1009", "Granite Peak Consulting", "owner@granitepeak.example", "Pro", "active", "4.2", "2024-12-01T16:40:00Z"),
    ("ACC-1010", "Fernwood Events", "owner@fernwood-events.example", "Pro", "cancelled", "4.3", "2025-04-22T10:05:00Z"),
]

# account_id, period, workflow_runs, api_requests_per_min, peak_requests_per_min
# 2026-10 is the current (month-to-date) period. Edge cases vs plan API limit:
#   ACC-1001 below (640 < 1,000), ACC-1006 exactly at (300 = 300),
#   ACC-1002 above (1,450 > 1,000), ACC-1007 above (95 > 60).
#   ACC-1005 is exactly at its monthly run quota (1,000).
USAGE = [
    ("ACC-1001", "2026-09", 182_400, 230, 710),
    ("ACC-1001", "2026-10", 48_200, 210, 640),
    ("ACC-1002", "2026-09", 241_000, 610, 1_380),
    ("ACC-1002", "2026-10", 131_900, 520, 1_450),
    ("ACC-1003", "2026-09", 21_300, 40, 150),
    ("ACC-1003", "2026-10", 9_800, 35, 120),
    ("ACC-1004", "2026-09", 410_000, 900, 2_300),
    ("ACC-1004", "2026-10", 0, 0, 0),
    ("ACC-1005", "2026-09", 870, 10, 38),
    ("ACC-1005", "2026-10", 1_000, 12, 42),
    ("ACC-1006", "2026-09", 19_600, 120, 280),
    ("ACC-1006", "2026-10", 14_300, 140, 300),
    ("ACC-1007", "2026-09", 410, 18, 55),
    ("ACC-1007", "2026-10", 640, 30, 95),
    ("ACC-1008", "2026-09", 152_000, 150, 430),
    ("ACC-1008", "2026-10", 61_000, 140, 410),
    ("ACC-1009", "2026-09", 23_900, 70, 210),
    ("ACC-1009", "2026-10", 22_750, 65, 180),
    ("ACC-1010", "2026-09", 3_100, 9, 55),
]

# invoice_id, account_id, amount, currency, charged_on, status, failure_reason, card_last4
INVOICES = [
    ("INV-2026-0001", "ACC-1001", 299.00, "USD", "2026-08-01", "paid", None, "4821"),
    ("INV-2026-0002", "ACC-1001", 299.00, "USD", "2026-09-01", "paid", None, "4821"),
    ("INV-2026-0003", "ACC-1001", 299.00, "USD", "2026-10-01", "paid", None, "4821"),
    ("INV-2026-0004", "ACC-1002", 299.00, "USD", "2026-09-03", "paid", None, "1881"),
    ("INV-2026-0005", "ACC-1002", 299.00, "USD", "2026-10-03", "paid", None, "1881"),
    ("INV-2026-0006", "ACC-1003", 49.00, "USD", "2026-08-05", "paid", None, "0341"),
    ("INV-2026-0007", "ACC-1003", 49.00, "USD", "2026-09-05", "paid", None, "0341"),
    ("INV-2026-0008", "ACC-1003", 49.00, "USD", "2026-10-05", "failed", "card_expired", "0341"),
    ("INV-2026-0009", "ACC-1004", 1_999.00, "USD", "2026-07-10", "paid", None, "7720"),
    ("INV-2026-0010", "ACC-1004", 1_999.00, "USD", "2026-08-10", "failed", "card_declined", "7720"),
    ("INV-2026-0011", "ACC-1004", 1_999.00, "USD", "2026-09-10", "failed", "card_declined", "7720"),
    ("INV-2026-0012", "ACC-1006", 49.00, "USD", "2026-09-12", "paid", None, "6610"),
    ("INV-2026-0013", "ACC-1008", 299.00, "EUR", "2026-09-02", "paid", None, "5508"),
    ("INV-2026-0014", "ACC-1008", 299.00, "EUR", "2026-10-02", "failed", "insufficient_funds", "5508"),
    ("INV-2026-0015", "ACC-1009", 49.00, "USD", "2026-08-20", "paid", None, "9034"),
    ("INV-2026-0016", "ACC-1009", 49.00, "USD", "2026-09-20", "paid", None, "9034"),
    ("INV-2026-0017", "ACC-1010", 49.00, "USD", "2026-08-03", "paid", None, "2267"),
    ("INV-2026-0018", "ACC-1010", 49.00, "USD", "2026-08-03", "refunded", None, "2267"),
    ("INV-2026-0019", "ACC-1010", 49.00, "USD", "2026-09-03", "paid", None, "2267"),
]

# component, status, message, updated_at
PLATFORM_STATUS = [
    ("API", "operational", "All API endpoints are operating normally.", "2026-10-06T08:30:00Z"),
    ("Workflow Engine", "degraded",
     "Elevated CF-503 error rates for a subset of workflow runs. Automatic retries are mitigating most failures.",
     "2026-10-06T08:30:00Z"),
    ("Dashboard", "operational", "The web dashboard is operating normally.", "2026-10-06T08:30:00Z"),
    ("Salesforce Integration", "operational", "Salesforce connections are operating normally.", "2026-10-06T08:30:00Z"),
    ("Slack Integration", "operational", "Slack connections are operating normally.", "2026-10-06T08:30:00Z"),
    ("Billing", "operational", "Payments and invoicing are operating normally.", "2026-10-06T08:30:00Z"),
]


def policy_registry_rows(policies: list[dict]) -> list[tuple]:
    """Derive registry rows from the policy documents (the source of truth)."""
    rows = []
    for p in policies:
        if p.get("doc_type") != "policy":
            raise ValueError(f"{p.get('source_id')}: not a policy document")
        rows.append((
            p["source_id"],
            p["title"],
            p["content"],
            p["source_id"],
            p["policy_type"],
            int(p.get("active", True) and p["deprecated_on"] is None),
            p["effective_from"],
            p["deprecated_on"],
            p["version"],
            f"{p['effective_from']}T00:00:00Z",
            f"{p['last_updated']}T00:00:00Z",
        ))
    return rows


def seed(db_path: str | Path | None = None, data_dir: Path = DATA_DIR) -> dict[str, int]:
    """Rebuild the database at db_path and return row counts per table."""
    target = Path(db_path or settings.database_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".tmp")
    tmp.unlink(missing_ok=True)

    policies = load_json(data_dir / SOURCE_FILES["policies"])

    try:
        with connect(tmp) as conn:
            init_schema(conn)
            conn.executemany("INSERT INTO plan_limits VALUES (?, ?, ?, ?, ?, ?)", PLANS)
            conn.executemany("INSERT INTO accounts VALUES (?, ?, ?, ?, ?, ?, ?)", ACCOUNTS)
            conn.executemany("INSERT INTO usage VALUES (?, ?, ?, ?, ?)", USAGE)
            conn.executemany("INSERT INTO invoices VALUES (?, ?, ?, ?, ?, ?, ?, ?)", INVOICES)
            conn.executemany("INSERT INTO platform_status VALUES (?, ?, ?, ?)", PLATFORM_STATUS)
            conn.executemany(
                "INSERT INTO policy_registry VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                policy_registry_rows(policies),
            )
            counts = {
                table: conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                for table in ("plan_limits", "accounts", "usage", "invoices",
                              "platform_status", "policy_registry", "handoffs")
            }
        os.replace(tmp, target)
    finally:
        tmp.unlink(missing_ok=True)

    return counts


def main() -> None:
    db_path = sys.argv[1] if len(sys.argv) > 1 else settings.database_path
    counts = seed(db_path)
    print(f"Seeded {db_path}")
    for table, count in counts.items():
        print(f"  {table:<16} {count} rows")


if __name__ == "__main__":
    main()
