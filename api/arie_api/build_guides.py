"""Build the missing paper guides, one paper at a time (S14): ``make guides``.

Papers indexed before guides existed, or queued when the server stopped, have no guide; the
server only builds guides for papers it indexes. Prints one JSON line per paper and exits 1
if any guide failed.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from arie_core.harness.guide import COMPLETE

from arie_api.runtime import ResearchRuntime

ROOT = Path(__file__).resolve().parents[2]


async def build(paper_ids: list[str]) -> bool:
    runtime = ResearchRuntime(ROOT)
    ok = True
    try:
        await runtime.initialize()
        for paper_id in paper_ids or await runtime.missing_guides():
            result = await runtime.build_guide(paper_id)
            line: dict[str, object] = {"paper_id": paper_id}
            if result is None:
                line.update(status="not_indexed")
                ok = False
            else:
                line.update(
                    status=result.status,
                    reason=result.fallback_reason,
                    seconds=result.seconds,
                    dropped=list(result.guide.dropped) if result.guide else [],
                    cache_hit=result.usage.cache_hit if result.usage else 0,
                )
                ok = ok and result.status == COMPLETE
            print(json.dumps(line), flush=True)
    finally:
        runtime.close()
    return ok


def main() -> None:
    parser = argparse.ArgumentParser(description="Build missing paper guides, one at a time.")
    parser.add_argument("paper_ids", nargs="*", help="rebuild these papers (default: missing)")
    args = parser.parse_args()
    raise SystemExit(0 if asyncio.run(build(list(args.paper_ids))) else 1)


if __name__ == "__main__":
    main()
