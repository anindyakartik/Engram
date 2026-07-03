"""Text-to-SQL domain over the quirky database.

Ground truth for every task is computed by executing a known-correct reference
query (which uses the hidden conventions correctly) against the seeded database at
construction time. The verifier then runs the agent's SQL and compares result sets.
No LLM is anywhere in the measurement path.

Task construction is template-based: each template is parameterised, and the
parameter values are split so that the training and held-out pools ask different
questions that exercise the same hidden conventions. Lessons therefore transfer
without the eval set being memorisable.
"""

from __future__ import annotations

import sqlite3

from data.seed import STATUS_CODES, build_db
from engram.domains.base import Task

# Parameter splits: disjoint values -> disjoint questions, shared hidden rules.
# Difficulty is graded: vanilla tasks need no hidden convention (the base model
# already passes them and they set the floor); mid tasks need one convention; hard
# order-tasks stack several (status codes + soft-delete + cents/epoch/joins).
_STATUS_TRAIN = ["pending", "shipped"]
_STATUS_EVAL = ["paid", "delivered"]
_YEAR_TRAIN = [2021, 2023]
_YEAR_EVAL = [2022, 2024]
_COUNTRY_TRAIN = ["US", "DE", "IN", "JP"]
_COUNTRY_EVAL = ["UK", "FR", "BR", "CA"]
_CAT_TRAIN = ["Electronics", "Books", "Home", "Toys"]
_CAT_EVAL = ["Garden", "Sports", "Grocery", "Clothing"]
# Price thresholds in dollars (the cents convention must convert these).
_PRICE_TRAIN = [50, 150, 300]
_PRICE_EVAL = [100, 200, 400]
# Compound (status x year) combos, split disjointly.
_COMBO_TRAIN = [("shipped", 2023), ("pending", 2021)]
_COMBO_EVAL = [("paid", 2022), ("delivered", 2024)]


def _round2(x: float) -> float:
    return round(float(x), 2)


class TextToSQLDomain:
    """Concrete text-to-SQL domain with a deterministic result-set verifier."""

    def __init__(self) -> None:
        self._conn = build_db()
        self._train = self._build_pool("train")
        self._eval = self._build_pool("eval")
        # Precompute and cache ground-truth result sets.
        self._ground: dict[str, list[tuple]] = {}
        for t in self._train + self._eval:
            self._ground[t.id] = self._run(t.reference)

    # ---- Domain protocol --------------------------------------------------- #
    def describe(self) -> str:
        """Return the schema (DDL) shown to the agent - no semantics included."""
        from data.seed import SCHEMA_DDL

        return SCHEMA_DDL

    def train_pool(self) -> list[Task]:
        return list(self._train)

    def eval_pool(self) -> list[Task]:
        return list(self._eval)

    def verify(self, task: Task, answer: str) -> bool:
        """Run the agent's SQL and compare its result set to ground truth.

        Comparison is order-insensitive across rows and tolerant of int/float and
        float rounding, but otherwise exact. Non-SELECT or malformed SQL fails.
        """
        if not _is_readonly_select(answer):
            return False
        try:
            got = self._run(answer)
        except sqlite3.Error:
            return False
        return _normalise(got) == _normalise(self._ground[task.id])

    # ---- internals --------------------------------------------------------- #
    def _run(self, sql: str) -> list[tuple]:
        cur = self._conn.cursor()
        cur.execute(sql)
        return cur.fetchall()

    def _build_pool(self, pool: str) -> list[Task]:
        is_train = pool == "train"
        statuses = _STATUS_TRAIN if is_train else _STATUS_EVAL
        years = _YEAR_TRAIN if is_train else _YEAR_EVAL
        countries = _COUNTRY_TRAIN if is_train else _COUNTRY_EVAL
        cats = _CAT_TRAIN if is_train else _CAT_EVAL
        prices = _PRICE_TRAIN if is_train else _PRICE_EVAL
        combos = _COMBO_TRAIN if is_train else _COMBO_EVAL
        tasks: list[Task] = []

        def add(tid: str, q: str, ttype: str, ref: str) -> None:
            tasks.append(Task(id=f"{pool}:{tid}", question=q, task_type=ttype, pool=pool, reference=ref))

        # ------------------------------------------------------------------ #
        # VANILLA - no hidden convention. The base model already passes these;
        # they set the floor so the domain sits in the hard-but-learnable zone.
        # ------------------------------------------------------------------ #
        vanilla = (
            [
                ("v:customers", "How many customers are registered?", "SELECT COUNT(*) FROM customers"),
                ("v:products", "How many products are in the catalog?", "SELECT COUNT(*) FROM products"),
                (
                    "v:countries",
                    "How many distinct countries do customers come from?",
                    "SELECT COUNT(DISTINCT country) FROM customers",
                ),
            ]
            if is_train
            else [
                ("v:categories", "How many product categories are there?", "SELECT COUNT(*) FROM categories"),
                ("v:items", "How many order line-item rows are there?", "SELECT COUNT(*) FROM order_items"),
                (
                    "v:orders_all",
                    "How many order records exist in total, including deleted ones?",
                    "SELECT COUNT(*) FROM orders",
                ),
            ]
        )
        for tid, q, ref in vanilla:
            add(tid, q, "vanilla_count", ref)
        # vanilla by country (customers table has no soft-delete / conventions).
        for c in countries[:2]:
            add(
                f"v:cust_country:{c}",
                f"How many customers are from {c}?",
                "vanilla_country",
                f"SELECT COUNT(*) FROM customers WHERE country = '{c}'",
            )

        # ------------------------------------------------------------------ #
        # MID - a single hidden convention.
        # ------------------------------------------------------------------ #
        # products_in_category: the category bridge only (no orders, no soft-delete).
        for cat in cats[:2]:
            add(
                f"prod_cat:{cat}",
                f"How many products belong to the {cat} category?",
                "products_in_category",
                "SELECT COUNT(DISTINCT p.id) FROM products p "
                "JOIN product_categories pc ON pc.product_id = p.id "
                "JOIN categories c ON c.id = pc.category_id "
                f"WHERE c.name = '{cat}'",
            )
        # expensive_products: the cents convention only.
        for d in prices[:2]:
            add(
                f"expensive:{d}",
                f"How many products have a list price above ${d}?",
                "expensive_products",
                f"SELECT COUNT(*) FROM products WHERE list_price_usd > {d * 100}",
            )

        # ------------------------------------------------------------------ #
        # HARD - order tasks that stack the soft-delete convention with others.
        # ------------------------------------------------------------------ #
        # status_count (status codes + soft-delete)
        for s in statuses:
            code = STATUS_CODES[s]
            add(
                f"status_count:{s}",
                f"How many orders are {s}?",
                "status_count",
                f"SELECT COUNT(*) FROM orders WHERE status = {code} AND is_deleted = 0",
            )
        # status_revenue (status codes + soft-delete + cents)
        for s in statuses:
            code = STATUS_CODES[s]
            add(
                f"status_revenue:{s}",
                f"What is the total revenue in dollars from {s} orders? Round to 2 decimals.",
                "status_revenue",
                f"SELECT ROUND(SUM(price_usd)/100.0, 2) FROM orders WHERE status = {code} AND is_deleted = 0",
            )
        # year_count (epoch dates + soft-delete)
        for y in years:
            add(
                f"year_count:{y}",
                f"How many orders were placed in {y}?",
                "year_count",
                f"SELECT COUNT(*) FROM orders WHERE strftime('%Y', placed_ts, 'unixepoch') = '{y}' AND is_deleted = 0",
            )
        # country_orders (join + soft-delete)
        for c in countries[:2]:
            add(
                f"country_orders:{c}",
                f"How many orders were placed by customers from {c}?",
                "country_orders",
                "SELECT COUNT(*) FROM orders o JOIN customers cu ON o.customer_id = cu.id "
                f"WHERE cu.country = '{c}' AND o.is_deleted = 0",
            )
        # category_count (bridge join + soft-delete)
        for cat in cats[:2]:
            add(
                f"category_count:{cat}",
                f"How many distinct orders include at least one product in the {cat} category?",
                "category_count",
                "SELECT COUNT(DISTINCT o.id) FROM orders o "
                "JOIN order_items oi ON oi.order_id = o.id "
                "JOIN product_categories pc ON pc.product_id = oi.product_id "
                "JOIN categories c ON c.id = pc.category_id "
                f"WHERE c.name = '{cat}' AND o.is_deleted = 0",
            )
        # compound (status codes + epoch + soft-delete)
        for s, y in combos:
            code = STATUS_CODES[s]
            add(
                f"combo:{s}:{y}",
                f"How many {s} orders were placed in {y}?",
                "compound",
                f"SELECT COUNT(*) FROM orders WHERE status = {code} "
                f"AND strftime('%Y', placed_ts, 'unixepoch') = '{y}' AND is_deleted = 0",
            )
        # ranking (soft-delete + grouping)
        if is_train:
            add(
                "top:country_most_orders",
                "Which country has the most orders? Return just the country code.",
                "ranking",
                "SELECT cu.country FROM orders o JOIN customers cu ON o.customer_id = cu.id "
                "WHERE o.is_deleted = 0 GROUP BY cu.country ORDER BY COUNT(*) DESC LIMIT 1",
            )
        else:
            add(
                "top:country_most_revenue",
                "Which country has the highest total order revenue? Return just the country code.",
                "ranking",
                "SELECT cu.country FROM orders o JOIN customers cu ON o.customer_id = cu.id "
                "WHERE o.is_deleted = 0 GROUP BY cu.country ORDER BY SUM(o.price_usd) DESC LIMIT 1",
            )
        return tasks


# --------------------------------------------------------------------------- #
# Verifier helpers (pure functions, unit-tested directly).
# --------------------------------------------------------------------------- #
def _is_readonly_select(sql: str) -> bool:
    """Return True iff sql is a single read-only SELECT/WITH statement."""
    s = sql.strip().rstrip(";").strip()
    if ";" in s:  # reject multiple statements
        return False
    head = s.lstrip("(").lower()
    if not (head.startswith("select") or head.startswith("with")):
        return False
    forbidden = ("insert", "update", "delete", "drop", "alter", "create", "attach", "pragma")
    return not any(f" {w} " in f" {s.lower()} " for w in forbidden)


def _normalise(rows: list[tuple]) -> list[tuple]:
    """Canonicalise a result set for order-insensitive, type-tolerant comparison."""
    out = []
    for row in rows:
        cells = []
        for v in row:
            if isinstance(v, bool):
                cells.append(int(v))
            elif isinstance(v, (int, float)):
                cells.append(_round2(v))
            else:
                cells.append(v)
        out.append(tuple(cells))
    out.sort(key=repr)
    return out
