import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "data" / "customers.db"


def get_connection():
    return sqlite3.connect(DB_PATH)


def create_tables():
    conn = get_connection()
    cursor = conn.cursor()

    # Customer account information
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS customers (
            account_id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            email TEXT NOT NULL,
            plan TEXT NOT NULL,
            status TEXT NOT NULL,
            storage_used_gb REAL,
            storage_limit_gb REAL
        )
    """)

    # Invoice information
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS invoices (
            invoice_id TEXT PRIMARY KEY,
            account_id TEXT NOT NULL,
            amount REAL NOT NULL,
            status TEXT NOT NULL,
            invoice_date TEXT NOT NULL,
            FOREIGN KEY (account_id) REFERENCES customers(account_id)
        )
    """)

    conn.commit()
    conn.close()


def insert_sample_data():
    conn = get_connection()
    cursor = conn.cursor()

    customers = [
        ("A1001", "Rahul", "rahul@example.com", "Pro",
         "Active", 65, 100),

        ("A1002", "Priya", "priya@example.com", "Free",
         "Active", 8, 10),

        ("A1003", "Aman", "aman@example.com", "Business",
         "Suspended", 180, 200),

        ("A1004", "Neha", "neha@example.com", "Enterprise",
         "Active", 420, 1000),

        ("A1005", "Rohit", "rohit@example.com", "Pro",
         "Cancelled", 0, 100),
    ]

    cursor.executemany("""
        INSERT OR IGNORE INTO customers
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, customers)

    invoices = [
        ("INV001", "A1001", 49.99, "Paid", "2026-09-01"),
        ("INV002", "A1002", 0.00, "Paid", "2026-09-01"),
        ("INV003", "A1003", 199.99, "Overdue", "2026-09-01"),
        ("INV004", "A1004", 499.99, "Paid", "2026-09-01"),
        ("INV005", "A1005", 49.99, "Refunded", "2026-08-01"),
    ]

    cursor.executemany("""
        INSERT OR IGNORE INTO invoices
        VALUES (?, ?, ?, ?, ?)
    """, invoices)

    conn.commit()
    conn.close()


if __name__ == "__main__":
    create_tables()
    insert_sample_data()

    print("Database created successfully!")
    print(f"Database location: {DB_PATH}")