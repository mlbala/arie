from __future__ import annotations

import argparse
import json
from pathlib import Path

from arie_api import routes


def export_openapi(output_path: str | Path) -> dict[str, object]:
    schema = routes.app.openapi()
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(schema, indent=2, sort_keys=True), encoding="utf-8")
    return schema


def main() -> None:
    parser = argparse.ArgumentParser(description="Export the ARIE OpenAPI schema.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("openapi.json"),
        help="Destination JSON file for the generated schema.",
    )
    args = parser.parse_args()
    export_openapi(args.output)


if __name__ == "__main__":
    main()
