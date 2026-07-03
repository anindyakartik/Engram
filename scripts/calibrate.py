"""Calibration gate (Phase 2): measure raw base-model success on held-out eval.

Runs the base model with NO memory, schema-only, one SQL submission per task, and
reports the held-out success rate. The domain is only useful if this is meaningfully
below ceiling (room to improve) and above floor (not hopeless). Uses the SDK directly
because the full record/replay client is built in Phase 3.
"""

from __future__ import annotations

import os
import sys
import time

from dotenv import load_dotenv

import config
from engram.domains.text_to_sql import TextToSQLDomain

load_dotenv()

SYSTEM = (
    "You are a careful data analyst working with a SQLite database. "
    "You are given ONLY the schema (no data, no documentation). "
    "Write exactly ONE SQLite SELECT query that answers the question, and submit it "
    "by calling the submit_sql tool. Do not explain."
)


def _build_client():
    from google import genai

    key = os.environ.get("GEMINI_API_KEY")
    if not key or key == "your-key-here":
        print("GEMINI_API_KEY not set - cannot run calibration (live call needed).")
        sys.exit(2)
    return genai.Client(api_key=key)


def _submit_tool(types):
    decl = types.FunctionDeclaration(
        name="submit_sql",
        description="Submit the final SQL query that answers the question.",
        parameters=types.Schema(
            type=types.Type.OBJECT,
            properties={"query": types.Schema(type=types.Type.STRING)},
            required=["query"],
        ),
    )
    return [types.Tool(function_declarations=[decl])]


def main() -> int:
    from google.genai import types

    client = _build_client()
    tools = _submit_tool(types)
    domain = TextToSQLDomain()
    schema = domain.describe()

    tasks = domain.eval_pool()
    n_ok = 0
    interval = 60.0 / config.RATE_LIMIT_RPM
    print(f"Calibrating base model on {len(tasks)} held-out tasks (no memory)...\n")

    for i, t in enumerate(tasks, 1):
        prompt = f"Schema:\n{schema}\n\nQuestion: {t.question}"
        query = ""
        for delay in (0, *config.BACKOFF_SCHEDULE):
            if delay:
                time.sleep(delay)
            try:
                resp = client.models.generate_content(
                    model=config.GEMINI_MODEL,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=SYSTEM,
                        tools=tools,
                        temperature=config.TEMPERATURE,
                        seed=config.LLM_SEED,
                        tool_config=types.ToolConfig(
                            function_calling_config=types.FunctionCallingConfig(mode="ANY")
                        ),
                    ),
                )
                calls = resp.function_calls or []
                query = dict(calls[0].args).get("query", "") if calls else ""
                break
            except Exception as e:  # noqa: BLE001 - retry transient/rate errors
                if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                    continue
                raise
        ok = domain.verify(t, query)
        n_ok += ok
        print(f"  [{i:2d}/{len(tasks)}] {'PASS' if ok else 'FAIL'}  {t.task_type:16s} {t.question[:52]}")
        time.sleep(interval)

    rate = n_ok / len(tasks)
    print(f"\nBaseline held-out success (no memory): {n_ok}/{len(tasks)} = {rate:.0%}")
    if rate >= 0.85:
        print("VERDICT: too easy (near ceiling) - quirks need to be harder.")
    elif rate <= 0.05:
        print("VERDICT: too hard (near floor) - may be unlearnable.")
    else:
        print("VERDICT: in the sweet spot - hard but learnable. Proceed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
