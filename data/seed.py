"""The quirky database and the disjoint train / held-out task banks.

The database is deliberately under-documented. Its schema (see SCHEMA_DDL) is
what the agent is shown, and it reveals column names and types but never their
semantics. Five conventions are hidden and can only be learned by attempting a
task, getting it wrong, and reflecting:

1. status is integer-coded (see STATUS_CODES); there are no string statuses.
2. orders.is_deleted = 1 rows are soft-deleted and must be excluded unless the
   question explicitly asks about deleted orders.
3. price_usd / list_price_usd are stored in integer cents, despite the
   _usd suffix that suggests dollars.
4. signup_ts / placed_ts are unix epoch seconds (use
   date(col, 'unixepoch') / strftime('%Y', col, 'unixepoch')).
5. A product's categories are many-to-many via the product_categories bridge;
   there is no category column on products.

Data is generated deterministically from a fixed seed so the database - and every
task's ground truth - is byte-stable across machines.
"""

from __future__ import annotations

import datetime as _dt
import random
import sqlite3

# Fixed seed for the *data*, independent of experiment seeds, so the DB and all
# ground-truth result sets are identical on every clone.
DB_SEED = 20240607

# --- Hidden semantics (the "data dictionary" the agent must rediscover) ------ #
STATUS_CODES: dict[str, int] = {
    "pending": 0,
    "paid": 1,
    "shipped": 2,
    "delivered": 3,
    "cancelled": 4,
    "refunded": 5,
}

COUNTRIES = ["US", "UK", "DE", "FR", "IN", "BR", "JP", "CA"]
CATEGORY_NAMES = [
    "Electronics",
    "Books",
    "Home",
    "Toys",
    "Garden",
    "Sports",
    "Grocery",
    "Clothing",
]

# The schema the agent sees. Intentionally comment-free: types and names only.
SCHEMA_DDL = """\
CREATE TABLE customers (
    id         INTEGER PRIMARY KEY,
    name       TEXT NOT NULL,
    country    TEXT NOT NULL,
    signup_ts  INTEGER NOT NULL
);

CREATE TABLE products (
    id             INTEGER PRIMARY KEY,
    name           TEXT NOT NULL,
    list_price_usd INTEGER NOT NULL
);

CREATE TABLE categories (
    id   INTEGER PRIMARY KEY,
    name TEXT NOT NULL
);

CREATE TABLE product_categories (
    product_id  INTEGER NOT NULL,
    category_id INTEGER NOT NULL
);

CREATE TABLE orders (
    id          INTEGER PRIMARY KEY,
    customer_id INTEGER NOT NULL,
    status      INTEGER NOT NULL,
    price_usd   INTEGER NOT NULL,
    placed_ts   INTEGER NOT NULL,
    is_deleted  INTEGER NOT NULL
);

CREATE TABLE order_items (
    order_id   INTEGER NOT NULL,
    product_id INTEGER NOT NULL,
    qty        INTEGER NOT NULL
);
"""


def _epoch(year: int, rng: random.Random) -> int:
    """Return a unix timestamp uniformly within the given calendar year."""
    start = int(_dt.datetime(year, 1, 1, tzinfo=_dt.UTC).timestamp())
    end = int(_dt.datetime(year + 1, 1, 1, tzinfo=_dt.UTC).timestamp()) - 1
    return rng.randint(start, end)


def build_db(path: str = ":memory:") -> sqlite3.Connection:
    """Build and populate the quirky database deterministically.

    Args:
        path: SQLite path. Defaults to a private in-memory database.

    Returns:
        An open connection to the populated database.
    """
    rng = random.Random(DB_SEED)
    conn = sqlite3.connect(path)
    conn.executescript(SCHEMA_DDL)
    cur = conn.cursor()

    # Customers: signups spread across 2020-2024.
    customers = []
    for cid in range(1, 151):
        country = rng.choice(COUNTRIES)
        signup = _epoch(rng.choice([2020, 2021, 2022, 2023, 2024]), rng)
        customers.append((cid, f"Customer {cid}", country, signup))
    cur.executemany("INSERT INTO customers VALUES (?,?,?,?)", customers)

    # Categories.
    cats = [(i + 1, name) for i, name in enumerate(CATEGORY_NAMES)]
    cur.executemany("INSERT INTO categories VALUES (?,?)", cats)

    # Products: prices in CENTS ($5.00-$500.00). Each product gets 1-2 categories.
    products = []
    prod_cats = []
    for pid in range(1, 41):
        cents = rng.randint(500, 50000)
        products.append((pid, f"Product {pid}", cents))
        for cat_id in rng.sample(range(1, 9), rng.randint(1, 2)):
            prod_cats.append((pid, cat_id))
    cur.executemany("INSERT INTO products VALUES (?,?,?)", products)
    cur.executemany("INSERT INTO product_categories VALUES (?,?)", prod_cats)

    # Orders: status integer-coded (weighted), price in cents, epoch timestamps,
    # ~12% soft-deleted. Placed across 2021-2025.
    status_choices = [0, 1, 2, 3, 4, 5]
    status_weights = [10, 25, 20, 30, 8, 7]
    orders = []
    order_items = []
    for oid in range(1, 601):
        cust = rng.randint(1, 150)
        status = rng.choices(status_choices, weights=status_weights)[0]
        cents = rng.randint(1000, 200000)
        placed = _epoch(rng.choice([2021, 2022, 2023, 2024, 2025]), rng)
        deleted = 1 if rng.random() < 0.12 else 0
        orders.append((oid, cust, status, cents, placed, deleted))
        for _ in range(rng.randint(1, 4)):
            order_items.append((oid, rng.randint(1, 40), rng.randint(1, 3)))
    cur.executemany("INSERT INTO orders VALUES (?,?,?,?,?,?)", orders)
    cur.executemany("INSERT INTO order_items VALUES (?,?,?)", order_items)

    conn.commit()
    return conn
