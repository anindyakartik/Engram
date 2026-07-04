"""Gemini client with deterministic record/replay, rate limiting, and cost accounting.

Design goals:
  * Reproducible. Every call is keyed on a canonical hash of its inputs (model,
    system, contents, tool spec, temperature, seed). On first live run the response
    is written to a committed cassette; afterwards it replays byte-identically with
    no API key. config.LLM_MODE selects auto | replay | record | live.
  * Free-tier friendly. Live calls pass through a shared token-bucket rate
    limiter and retry on HTTP 429 with exponential backoff.
  * Decoupled from the SDK. Callers pass plain-dict tool specs and receive a
    plain LLMResult; the google-genai types never leak out of this module.

Only generate and embed make network calls, and only when a cassette is
missing (auto), or in record/live mode.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import config


@dataclass
class LLMResult:
    """Result of a generate call, independent of the SDK types."""

    text: str = ""
    tool_calls: list[dict] = field(default_factory=list)  # [{"name": str, "args": dict}]
    usage: dict = field(default_factory=dict)  # {"input", "output", "total"}
    cached: bool = False

    def first_arg(self, key: str, default: str = "") -> str:
        """Return key from the first tool call's args, or default."""
        if self.tool_calls:
            return str(self.tool_calls[0].get("args", {}).get(key, default))
        return default

    def first_tool_args(self) -> dict:
        """Return the first tool call's full args dict (empty if no tool call)."""
        return self.tool_calls[0].get("args", {}) if self.tool_calls else {}


class ReplayMiss(RuntimeError):
    """Raised in replay mode when no cassette exists for a requested call."""


def _sha(obj: Any) -> str:
    blob = json.dumps(obj, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def _jsonable(value: Any) -> Any:
    """Best-effort conversion of SDK arg values into JSON-native types."""
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


class LLMClient:
    """Rate-limited, cost-accounting, record/replay wrapper around google-genai."""

    def __init__(self, mode: str | None = None, cassette_dir: Path | None = None) -> None:
        self.mode = mode or config.LLM_MODE
        self.cassette_dir = cassette_dir or config.CASSETTE_DIR
        (self.cassette_dir / "generate").mkdir(parents=True, exist_ok=True)
        (self.cassette_dir / "embed").mkdir(parents=True, exist_ok=True)
        self._client = None  # lazily created; replay needs no key
        self._last_call = 0.0
        # Cost/usage accounting (live + replayed, so cost reflects a real run).
        self.n_generate = 0
        self.n_embed = 0
        self.n_live = 0
        self.tokens_in = 0
        self.tokens_out = 0
        self.tokens_embed = 0

    # ---- public API -------------------------------------------------------- #
    def generate(
        self,
        *,
        system: str,
        contents: str,
        tools: list[dict] | None = None,
        force_tool: bool = True,
    ) -> LLMResult:
        """Generate content, optionally forcing a tool call.

        Args:
            system: System instruction.
            contents: The user prompt.
            tools: Plain-dict tool specs (name/description/parameters).
            force_tool: If True and tools are given, require a function call.

        Returns:
            An LLMResult with text and/or tool calls and token usage.
        """
        tools = tools or []
        key = {
            "kind": "generate",
            "model": config.GEMINI_MODEL,
            "system": system,
            "contents": contents,
            "tools": tools,
            "force_tool": force_tool,
            "temperature": config.TEMPERATURE,
            "seed": config.LLM_SEED,
        }
        path = self.cassette_dir / "generate" / f"{_sha(key)}.json"

        cached = self._load(path)
        if cached is not None:
            self.n_generate += 1
            self._account(cached["usage"])
            return LLMResult(
                text=cached["text"],
                tool_calls=cached["tool_calls"],
                usage=cached["usage"],
                cached=True,
            )
        if self.mode == "replay":
            raise ReplayMiss(f"no cassette for generate call {path.name}")

        result = self._live_generate(system, contents, tools, force_tool)
        record = {
            "text": result.text,
            "tool_calls": result.tool_calls,
            "usage": result.usage,
            "_key": key,  # stored for human inspection/debuggability
        }
        self._save(path, record)
        self.n_generate += 1
        self._account(result.usage)
        return result

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return an embedding vector per input text (cached per individual text)."""
        results: dict[int, list[float]] = {}
        missing: list[tuple[int, str]] = []
        for i, text in enumerate(texts):
            key = {"kind": "embed", "model": config.EMBED_MODEL, "dim": config.EMBED_DIM, "text": text}
            path = self.cassette_dir / "embed" / f"{_sha(key)}.json"
            cached = self._load(path)
            if cached is not None:
                results[i] = cached["embedding"]
            else:
                missing.append((i, text))

        if missing:
            if self.mode == "replay":
                raise ReplayMiss(f"no cassette for {len(missing)} embed call(s)")
            vectors = self._live_embed([t for _, t in missing])
            for (i, text), vec in zip(missing, vectors, strict=True):
                key = {"kind": "embed", "model": config.EMBED_MODEL, "dim": config.EMBED_DIM, "text": text}
                path = self.cassette_dir / "embed" / f"{_sha(key)}.json"
                self._save(path, {"embedding": vec, "_text": text})
                results[i] = vec
                self.tokens_embed += max(1, len(text) // 4)

        self.n_embed += len(texts)
        return [results[i] for i in range(len(texts))]

    def cost_usd(self) -> float:
        """Return the accumulated (illustrative) cost of this run in USD."""
        return (
            self.tokens_in / 1e6 * config.PRICE_INPUT_PER_1M
            + self.tokens_out / 1e6 * config.PRICE_OUTPUT_PER_1M
            + self.tokens_embed / 1e6 * config.PRICE_EMBED_PER_1M
        )

    def stats(self) -> dict:
        """Return a summary of calls, tokens, live calls, and cost."""
        return {
            "generate_calls": self.n_generate,
            "embed_calls": self.n_embed,
            "live_calls": self.n_live,
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "tokens_embed": self.tokens_embed,
            "cost_usd": round(self.cost_usd(), 6),
        }

    # ---- internals --------------------------------------------------------- #
    def _account(self, usage: dict) -> None:
        self.tokens_in += usage.get("input", 0)
        self.tokens_out += usage.get("output", 0)

    def _load(self, path: Path) -> dict | None:
        if self.mode == "record" or self.mode == "live":
            # record: overwrite; live: never read cassettes
            if self.mode == "live":
                return None
            return None
        if path.exists():
            return json.loads(path.read_text())
        return None

    def _save(self, path: Path, record: dict) -> None:
        if self.mode == "live":
            return  # live mode does not persist cassettes
        path.write_text(json.dumps(record, indent=2, sort_keys=True))

    def _ensure_client(self):
        if self._client is None:
            from google import genai

            key = os.environ.get("GEMINI_API_KEY")
            if not key or key == "your-key-here":
                raise RuntimeError(
                    "A live LLM call is required but GEMINI_API_KEY is not set. "
                    "Use LLM_MODE=replay to run fully offline from committed cassettes."
                )
            self._client = genai.Client(api_key=key)
        return self._client

    def _throttle(self) -> None:
        min_interval = 60.0 / config.RATE_LIMIT_RPM
        elapsed = time.monotonic() - self._last_call
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)
        self._last_call = time.monotonic()

    def _with_backoff(self, fn):
        last_exc: Exception | None = None
        for delay in (0, *config.BACKOFF_SCHEDULE):
            if delay:
                time.sleep(delay)
            self._throttle()
            try:
                self.n_live += 1
                return fn()
            except Exception as e:  # noqa: BLE001 - retry only rate/transient errors
                msg = str(e)
                if "429" in msg or "RESOURCE_EXHAUSTED" in msg or "503" in msg:
                    last_exc = e
                    continue
                raise
        raise RuntimeError(f"exhausted retries after rate limiting: {last_exc}")

    def _tool_objects(self, tools: list[dict]):
        from google.genai import types

        _map = {
            "object": types.Type.OBJECT,
            "string": types.Type.STRING,
            "integer": types.Type.INTEGER,
            "number": types.Type.NUMBER,
            "boolean": types.Type.BOOLEAN,
            "array": types.Type.ARRAY,
        }

        def to_schema(spec: dict):
            props = spec.get("properties")
            items = spec.get("items")
            return types.Schema(
                type=_map[spec.get("type", "string")],
                description=spec.get("description"),
                properties=({k: to_schema(v) for k, v in props.items()} if props else None),
                items=(to_schema(items) if items else None),
                required=spec.get("required"),
            )

        decls = [
            types.FunctionDeclaration(
                name=t["name"],
                description=t.get("description", ""),
                parameters=to_schema(t["parameters"]),
            )
            for t in tools
        ]
        return [types.Tool(function_declarations=decls)]

    def _live_generate(self, system, contents, tools, force_tool) -> LLMResult:
        from google.genai import types

        client = self._ensure_client()
        cfg_kwargs: dict[str, Any] = {
            "system_instruction": system,
            "temperature": config.TEMPERATURE,
            "seed": config.LLM_SEED,
        }
        if tools:
            cfg_kwargs["tools"] = self._tool_objects(tools)
            if force_tool:
                cfg_kwargs["tool_config"] = types.ToolConfig(
                    function_calling_config=types.FunctionCallingConfig(mode="ANY")
                )

        resp = self._with_backoff(
            lambda: client.models.generate_content(
                model=config.GEMINI_MODEL,
                contents=contents,
                config=types.GenerateContentConfig(**cfg_kwargs),
            )
        )
        calls = [
            {"name": fc.name, "args": _jsonable(dict(fc.args or {}))}
            for fc in (resp.function_calls or [])
        ]
        # Only read .text when there is no tool call (reading it on a function-call
        # response emits a noisy SDK warning about non-text parts).
        text = "" if calls else (resp.text or "")
        um = resp.usage_metadata
        usage = {
            "input": getattr(um, "prompt_token_count", 0) or 0,
            "output": getattr(um, "candidates_token_count", 0) or 0,
            "total": getattr(um, "total_token_count", 0) or 0,
        }
        return LLMResult(text=text, tool_calls=calls, usage=usage)

    def _live_embed(self, texts: list[str]) -> list[list[float]]:
        from google.genai import types

        client = self._ensure_client()
        resp = self._with_backoff(
            lambda: client.models.embed_content(
                model=config.EMBED_MODEL,
                contents=texts,
                config=types.EmbedContentConfig(output_dimensionality=config.EMBED_DIM),
            )
        )
        return [list(e.values) for e in resp.embeddings]


_DEFAULT: LLMClient | None = None


def get_client() -> LLMClient:
    """Return the process-wide default client (created on first use)."""
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = LLMClient()
    return _DEFAULT


def set_client(client: LLMClient) -> None:
    """Override the process-wide default client (used by the harness per run)."""
    global _DEFAULT
    _DEFAULT = client
