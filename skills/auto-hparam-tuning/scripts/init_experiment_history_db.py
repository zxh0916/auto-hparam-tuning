#!/usr/bin/env python3
"""Initialize the CSV-backed experiment history layout."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiment_history import default_history_root


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compatibility entry point for experiment-history initialization. "
            "The backend is CSV-backed AHT sessions under <project_root>/aht/."
        )
    )
    parser.add_argument("--project-root", required=True, help="Target project root.")
    args = parser.parse_args()

    history_root = default_history_root(args.project_root)
    history_root.mkdir(parents=True, exist_ok=True)
    print(
        json.dumps(
            {
                "backend": "csv-session-store",
                "history_root": str(history_root),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
