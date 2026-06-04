"""Command-line entry point: ``python -m ragulator``.

Runs the same pipeline as the notebook UI, but as an interactive REPL so the
project is fully usable outside Colab (e.g. in VS Code's integrated terminal).
"""

from __future__ import annotations

import argparse
import sys

from .app import build_app
from .banner import show_banner
from .config import Settings
from .pipeline import process_request


def _print_result(out: dict) -> None:
    print(f"\n[ Role: {out['user_role']} | Clearance: {out['user_clearance']} | "
          f"Intent: {out['intent']} | Purpose: {out['purpose']} | {out['duration']}s ]")
    if not out["results"]:
        print("  (no results)")
        return
    for item in out["results"]:
        if "DocID" in item and "error" not in item:
            print(f"  [{item.get('access')}] DocID {item.get('DocID')} "
                  f"({item.get('Classification')}) NKT={item.get('need_to_know')} :: {item.get('Title')}")
            if item.get("reason"):
                print(f"      reason: {item['reason']}")
        else:
            key = next((k for k in (
                "policy_info", "general_response", "info", "warning", "not_found",
                "clarification_needed", "error") if k in item), None)
            label = key or "result"
            value = item.get(key, item)
            extra = f" :: {item.get('reason')}" if key == "error" and item.get("reason") else ""
            print(f"  [{label}] {value}{extra}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="The RAGulator - it regulates. RAG document access-control PoC by Psypher Labs (CLI).")
    parser.add_argument("--role", default="analyst", help="User role to assume.")
    parser.add_argument("--query", help="Run a single query and exit.")
    parser.add_argument("--no-llm", action="store_true",
                        help="Skip Gemini init (data-layer exploration only; queries will fail).")
    parser.add_argument("--nkt-mode", choices=["strict", "moderate", "loose"], default=None)
    args = parser.parse_args(argv)

    show_banner()

    overrides = {}
    if args.nkt_mode:
        overrides["nkt_mode"] = args.nkt_mode
    # Prefer a local config.yaml if present; otherwise pure env/defaults.
    settings = Settings.from_yaml("config.yaml", **overrides)

    try:
        app = build_app(settings, require_llm=not args.no_llm)
    except Exception as exc:  # noqa: BLE001
        print(f"\nInitialisation failed: {exc}", file=sys.stderr)
        print("Hint: ensure GOOGLE_API_KEY is set (env var or .env), or use --no-llm.", file=sys.stderr)
        return 1

    def run(q: str) -> None:
        out = process_request(raw_query=q, user_role=args.role, settings=settings,
                              gemini=app.gemini, engine=app.engine, api_cache=app.api_cache)
        _print_result(out)

    if args.query:
        run(args.query)
        return 0

    print(f"\nInteractive mode (role={args.role}). Type a query, or 'quit' to exit.")
    try:
        while True:
            q = input("\n> ").strip()
            if q.lower() in {"quit", "exit", "q"}:
                break
            if q:
                run(q)
    except (EOFError, KeyboardInterrupt):
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
