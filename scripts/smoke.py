"""Phase-1 live smoke test for the google-genai SDK.

Confirms the exact call shapes Engram depends on BEFORE anything is built on them:
  1. Function calling  -> the model returns a structured tool call we can read.
  2. Embeddings        -> we get fixed-dimensional float vectors back.

Run:  python scripts/smoke.py   (requires GEMINI_API_KEY in .env; live call)

This script is deliberately dependency-light and does NOT import Engram internals,
so it validates the raw SDK contract in isolation.
"""

from __future__ import annotations

import os
import sys

from dotenv import load_dotenv

load_dotenv()


def main() -> int:
    key = os.environ.get("GEMINI_API_KEY")
    if not key or key == "your-key-here":
        print(
            "GEMINI_API_KEY not set. Copy .env.example to .env and add your key.\n"
            "This is the only step in the whole project that requires a live key\n"
            "besides recording cassettes; replaying the published result needs none."
        )
        return 2

    from google import genai
    from google.genai import types

    client = genai.Client(api_key=key)

    agent_model = "gemini-flash-lite-latest"
    embed_model = "gemini-embedding-001"

    # --- 1. Function calling ------------------------------------------------ #
    print("== Function calling ==")
    run_sql = types.FunctionDeclaration(
        name="run_sql",
        description="Execute a read-only SQL query against the database.",
        parameters=types.Schema(
            type=types.Type.OBJECT,
            properties={
                "query": types.Schema(
                    type=types.Type.STRING,
                    description="A single SQL SELECT statement.",
                ),
            },
            required=["query"],
        ),
    )
    tools = [types.Tool(function_declarations=[run_sql])]

    resp = client.models.generate_content(
        model=agent_model,
        contents="List all customers. Call the run_sql tool with the query.",
        config=types.GenerateContentConfig(
            tools=tools,
            temperature=0.0,
            tool_config=types.ToolConfig(
                function_calling_config=types.FunctionCallingConfig(mode="ANY")
            ),
        ),
    )

    calls = resp.function_calls or []
    if not calls:
        print("  UNEXPECTED: no function_calls on response. Dumping candidate parts:")
        for part in resp.candidates[0].content.parts:
            print("   part:", part)
        return 1
    fc = calls[0]
    print(f"  response.function_calls[0].name = {fc.name!r}")
    print(f"  response.function_calls[0].args = {dict(fc.args)!r}")
    print(f"  usage: {resp.usage_metadata}")

    # --- 2. Embeddings ------------------------------------------------------ #
    print("\n== Embeddings ==")
    er = client.models.embed_content(
        model=embed_model,
        contents=["status is stored as integer codes", "the sky is blue"],
        config=types.EmbedContentConfig(output_dimensionality=768),
    )
    vecs = er.embeddings
    print(f"  embeddings returned: {len(vecs)}")
    print(f"  embeddings[0] dim   : {len(vecs[0].values)}")
    print(f"  embeddings[0][:4]   : {[round(v, 4) for v in vecs[0].values[:4]]}")

    print("\nSMOKE OK - SDK shapes confirmed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
