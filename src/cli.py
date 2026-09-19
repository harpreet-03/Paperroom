"""
CLI entry point.

Usage:
  python -m src.cli --query "recent work on KV-cache compression for LLMs"
  python -m src.cli --query 2401.12345
  python -m src.cli --query https://arxiv.org/abs/2401.12345
  python -m src.cli --resume 2401.12345          # skip pipeline, reload a past session
  python -m src.cli --query "..." --no-repl       # print briefing and exit (scripting)
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict

from dotenv import load_dotenv

load_dotenv()  # reads .env in the project root, no manual `export` needed

from src.nodes import qa
from src.pipeline import load_session, run_pipeline, save_session


def print_briefing(state) -> None:
    b = state.briefing
    print("\n" + "=" * 78)
    print(f"{b.title}")
    print(f"{', '.join(b.authors)}")
    print(f"arXiv:{b.arxiv_id}  ({b.published[:10]})   {b.link}")
    print("=" * 78)
    print("\nWHY IT MATTERS\n" + b.summary)
    print("\nPROBLEM STATEMENT\n" + b.problem_statement)
    print("\nMETHOD")
    for item in b.method:
        print(f"  - {item}")
    print("\nKEY RESULTS")
    for item in b.key_results:
        print(f"  - {item}")
    print("\nLIMITATIONS")
    for item in b.limitations:
        print(f"  - {item}")
    print("\nSUGGESTED FOLLOW-UP QUESTIONS")
    for item in b.suggested_questions:
        print(f"  - {item}")
    if state.parsed and state.parsed.parse_warning:
        print(f"\n[note] {state.parsed.parse_warning}")
    print("=" * 78 + "\n")


def repl(state) -> None:
    print("Ask questions about this paper (blank line or 'exit' to quit).\n")
    while True:
        try:
            question = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not question or question.lower() in ("exit", "quit"):
            break
        answer = qa.answer(state, question)
        print(answer + "\n")
    save_session(state)


def main() -> int:
    parser = argparse.ArgumentParser(description="Autonomous arXiv Paper Digest & QA Agent")
    parser.add_argument("--query", help="A topic, an arXiv id, or an arXiv URL")
    parser.add_argument("--resume", help="Resume a past session by arXiv id (skips pipeline)")
    parser.add_argument("--no-repl", action="store_true", help="Print briefing and exit, no QA loop")
    parser.add_argument("--json", action="store_true", help="Print briefing as JSON instead of text")
    parser.add_argument("-v", "--verbose", action="store_true", help="Print each graph node as it runs")
    args = parser.parse_args()

    if not args.query and not args.resume:
        parser.error("provide --query or --resume")

    if args.resume:
        state = load_session(args.resume)
        if state is None:
            print(f"No saved session found for '{args.resume}'.", file=sys.stderr)
            return 1
    else:
        print(f"Running pipeline for: {args.query!r}\n")
        state = run_pipeline(args.query, verbose=args.verbose)

        if state.status == "no_candidates":
            print("arXiv returned no matching papers. Try a broader topic or check the id.")
            return 1
        if state.status == "error" or state.briefing is None:
            print("Pipeline failed:")
            for err in state.errors:
                print(f"  - {err}")
            return 1

        save_session(state)

    if args.json:
        print(json.dumps(asdict(state.briefing), indent=2))
    else:
        print_briefing(state)

    if not args.no_repl:
        repl(state)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
