"""Verifier correctness - the foundation of every downstream number.

These tests prove: reference queries are correct, right answers pass, quirk-violating
answers fail (and genuinely differ from ground truth), and non-SELECT/malformed SQL
is rejected.
"""

from __future__ import annotations

import pytest

from engram.domains.text_to_sql import TextToSQLDomain, _is_readonly_select, _normalise


@pytest.fixture(scope="module")
def domain() -> TextToSQLDomain:
    return TextToSQLDomain()


def _task(domain: TextToSQLDomain, tid: str):
    for t in domain.train_pool() + domain.eval_pool():
        if t.id == tid:
            return t
    raise KeyError(tid)


def test_pools_disjoint_and_cover_types(domain: TextToSQLDomain) -> None:
    train_ids = {t.id for t in domain.train_pool()}
    eval_ids = {t.id for t in domain.eval_pool()}
    assert train_ids and eval_ids
    assert train_ids.isdisjoint(eval_ids)
    # Same questions never appear in both pools.
    train_q = {t.question for t in domain.train_pool()}
    eval_q = {t.question for t in domain.eval_pool()}
    assert train_q.isdisjoint(eval_q)
    # Both pools exercise every task type (same hidden rules, different questions).
    train_types = {t.task_type for t in domain.train_pool()}
    eval_types = {t.task_type for t in domain.eval_pool()}
    expected = {
        "vanilla_count",
        "vanilla_country",
        "products_in_category",
        "expensive_products",
        "status_count",
        "status_revenue",
        "year_count",
        "country_orders",
        "category_count",
        "compound",
        "ranking",
    }
    assert expected <= train_types
    assert expected <= eval_types


def test_reference_answers_pass(domain: TextToSQLDomain) -> None:
    """Every task's own reference SQL must verify as correct."""
    for t in domain.train_pool() + domain.eval_pool():
        assert domain.verify(t, t.reference), f"reference failed for {t.id}"


def test_ground_truth_regression_anchors(domain: TextToSQLDomain) -> None:
    """Pin a few known values so accidental data/logic drift is caught."""
    # pending orders, soft-deletes excluded (train pool).
    assert domain.verify(
        _task(domain, "train:status_count:pending"),
        "SELECT COUNT(*) FROM orders WHERE status = 0 AND is_deleted = 0",
    )
    assert not domain.verify(
        _task(domain, "train:status_count:pending"),
        "SELECT 63",  # the count if soft-deletes are wrongly included
    )
    assert domain.verify(
        _task(domain, "train:status_count:pending"), "SELECT 54"
    )


@pytest.mark.parametrize(
    ("tid", "wrong_sql", "why"),
    [
        # Forgets the soft-delete filter -> overcounts.
        (
            "train:status_count:pending",
            "SELECT COUNT(*) FROM orders WHERE status = 0",
            "ignored is_deleted",
        ),
        # Treats status as a string -> matches nothing.
        (
            "train:status_count:pending",
            "SELECT COUNT(*) FROM orders WHERE status = 'pending' AND is_deleted = 0",
            "string status",
        ),
        # Revenue left in cents instead of dollars.
        (
            "eval:status_revenue:paid",
            "SELECT SUM(price_usd) FROM orders WHERE status = 1 AND is_deleted = 0",
            "cents not dollars",
        ),
        # Year filter without unixepoch -> zero rows.
        (
            "train:year_count:2023",
            "SELECT COUNT(*) FROM orders WHERE strftime('%Y', placed_ts) = '2023' AND is_deleted = 0",
            "no unixepoch",
        ),
        # Assumes a category column exists on products (there is none).
        (
            "train:category_count:Electronics",
            "SELECT COUNT(*) FROM orders WHERE is_deleted = 0",
            "no bridge join",
        ),
    ],
)
def test_quirk_violations_fail(domain: TextToSQLDomain, tid: str, wrong_sql: str, why: str) -> None:
    task = _task(domain, tid)
    # Sanity: the wrong query must actually differ from ground truth here, so the
    # test is meaningful (not accidentally passing because the quirk didn't bite).
    correct = domain.verify(task, task.reference)
    assert correct, f"reference itself failed for {tid}"
    assert not domain.verify(task, wrong_sql), f"quirk violation wrongly passed ({why})"


def test_non_select_and_malformed_rejected(domain: TextToSQLDomain) -> None:
    task = _task(domain, "train:status_count:pending")
    assert not domain.verify(task, "DROP TABLE orders")
    assert not domain.verify(task, "UPDATE orders SET is_deleted = 0")
    assert not domain.verify(task, "SELECT COUNT(*) FROM orders; DELETE FROM orders")
    assert not domain.verify(task, "this is not sql")
    assert not domain.verify(task, "")


def test_is_readonly_select_unit() -> None:
    assert _is_readonly_select("SELECT 1")
    assert _is_readonly_select("  select * from orders ;  ")
    assert _is_readonly_select("WITH x AS (SELECT 1) SELECT * FROM x")
    assert not _is_readonly_select("DELETE FROM orders")
    assert not _is_readonly_select("SELECT 1; SELECT 2")
    assert not _is_readonly_select("UPDATE orders SET status = 1")
    assert not _is_readonly_select("")


def test_normalise_order_and_type_tolerance() -> None:
    assert _normalise([(2,), (1,)]) == _normalise([(1,), (2,)])
    assert _normalise([(5,)]) == _normalise([(5.0,)])
    assert _normalise([(1.234,)]) == [(1.23,)]
    assert _normalise([(True,)]) == [(1,)]
