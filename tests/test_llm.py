"""Record/replay, caching, and cost-accounting logic for the LLM client.

Network is never touched: the live methods are monkeypatched so the deterministic
cassette machinery can be tested offline.
"""

from __future__ import annotations

import pytest

from engram.core.llm import LLMClient, LLMResult, ReplayMiss


def _fake_result(text="SELECT 1") -> LLMResult:
    return LLMResult(
        text=text,
        tool_calls=[{"name": "submit_sql", "args": {"query": text}}],
        usage={"input": 10, "output": 5, "total": 15},
    )


def test_generate_records_then_replays(tmp_path, monkeypatch):
    client = LLMClient(mode="auto", cassette_dir=tmp_path)
    calls = {"n": 0}

    def fake_live(system, contents, tools, force_tool):
        calls["n"] += 1
        return _fake_result()

    monkeypatch.setattr(client, "_live_generate", fake_live)

    r1 = client.generate(system="s", contents="q", tools=[])
    assert calls["n"] == 1 and r1.cached is False
    assert r1.first_arg("query") == "SELECT 1"

    # Identical call: served from cassette, no new live call.
    r2 = client.generate(system="s", contents="q", tools=[])
    assert calls["n"] == 1 and r2.cached is True
    assert r2.tool_calls == r1.tool_calls

    # Different prompt -> new live call + new cassette.
    client.generate(system="s", contents="different", tools=[])
    assert calls["n"] == 2


def test_replay_miss_raises(tmp_path):
    client = LLMClient(mode="replay", cassette_dir=tmp_path)
    with pytest.raises(ReplayMiss):
        client.generate(system="s", contents="never recorded", tools=[])


def test_embed_caches_per_text(tmp_path, monkeypatch):
    client = LLMClient(mode="auto", cassette_dir=tmp_path)
    seen: list[list[str]] = []

    def fake_embed(texts):
        seen.append(list(texts))
        return [[float(len(t))] * 3 for t in texts]

    monkeypatch.setattr(client, "_live_embed", fake_embed)

    a = client.embed(["alpha", "beta"])
    assert seen == [["alpha", "beta"]]
    assert a[0] == [5.0, 5.0, 5.0]

    # "beta" is cached; only "gamma" hits the live path.
    client.embed(["beta", "gamma"])
    assert seen[-1] == ["gamma"]


def test_cost_accounting(tmp_path, monkeypatch):
    client = LLMClient(mode="auto", cassette_dir=tmp_path)
    monkeypatch.setattr(client, "_live_generate", lambda *a: _fake_result())
    client.generate(system="s", contents="q", tools=[])
    stats = client.stats()
    assert stats["generate_calls"] == 1
    assert stats["tokens_in"] == 10 and stats["tokens_out"] == 5
    assert stats["cost_usd"] >= 0.0


def test_tool_spec_converts_to_sdk_types(tmp_path):
    """The plain-dict tool spec must convert to google-genai types without error."""
    client = LLMClient(mode="replay", cassette_dir=tmp_path)
    tools = [
        {
            "name": "submit_sql",
            "description": "submit the query",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "a select"}},
                "required": ["query"],
            },
        }
    ]
    objs = client._tool_objects(tools)
    assert objs and objs[0].function_declarations[0].name == "submit_sql"
