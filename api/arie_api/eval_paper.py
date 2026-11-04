"""Evaluate answers on one ingested paper from the command line.

    uv run python -m arie_api.eval_paper <paper_id> [--regenerate] [--compare-reranker]
        [--mode strict|explain]

Uses arie.yaml, the library database and every configured model, exactly as chat does.
The first run writes questions from the paper's own sentences and saves them; later runs
reuse them, so results stay comparable when you change a model or setting. Prints one
JSON line per answer, then the summary; exits 1 if any scored case failed.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from arie_api.paper_eval import MODES, require_available_mode
from arie_api.runtime import ResearchRuntime

ROOT = Path(__file__).resolve().parents[2]


async def evaluate(
    paper_id: str, *, regenerate: bool, compare_reranker: bool, mode: str = "strict"
) -> bool:
    runtime = ResearchRuntime(ROOT)
    try:
        await runtime.initialize()
        # The first load may download models; answers are then timed as in production.
        await asyncio.wait_for(runtime.models.warm(), 600.0)
        run = await runtime.evaluations.run_to_completion(
            paper_id, regenerate=regenerate, compare_reranker=compare_reranker, mode=mode
        )
    finally:
        runtime.close()
    for result in run["results"]:
        print(json.dumps(result), flush=True)
    print(
        json.dumps(
            {
                "status": run["status"],
                "message": run["message"],
                "settings": run["settings"],
                "summary": run["summary"],
            }
        )
    )
    if run["status"] != "complete":
        return False
    return all(item.get("passed") is not False for item in run["results"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate answers on one ingested paper.")
    parser.add_argument("paper_id")
    parser.add_argument("--regenerate", action="store_true", help="write new questions")
    parser.add_argument(
        "--compare-reranker", action="store_true", help="also answer without the reranker"
    )
    parser.add_argument("--mode", choices=MODES, default="strict", help="answer path to score")
    args = parser.parse_args()
    try:
        # Fail before the models warm up (up to 600 s) for a mode that cannot run.
        require_available_mode(args.mode)
    except ValueError as exc:
        parser.exit(2, f"error: {exc}\n")
    ok = asyncio.run(
        evaluate(
            args.paper_id,
            regenerate=args.regenerate,
            compare_reranker=args.compare_reranker,
            mode=args.mode,
        )
    )
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
