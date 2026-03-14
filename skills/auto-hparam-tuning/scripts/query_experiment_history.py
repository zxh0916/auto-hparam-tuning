#!/usr/bin/env python3
"""Query experiment history rows from the CSV-backed AHT session store."""

from __future__ import annotations

import argparse
import json

from experiment_history import load_history_rows


def print_table(rows: list[dict[str, object]]) -> None:
    if not rows:
        print("No experiment rows found.")
        return

    table_rows = [
        [
            str(row["id"]),
            str(row["timestamp"]),
            str(row["run_name"]),
            json.dumps(row["metrics"], ensure_ascii=False, sort_keys=True),
        ]
        for row in rows
    ]
    headers = ["id", "timestamp", "run_name", "metrics"]
    widths = [
        max(len(headers[index]), *(len(record[index]) for record in table_rows))
        for index in range(len(headers))
    ]
    print(" | ".join(header.ljust(widths[index]) for index, header in enumerate(headers)))
    print("-+-".join("-" * width for width in widths))
    for record in table_rows:
        print(" | ".join(value.ljust(widths[index]) for index, value in enumerate(record)))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Read experiment history rows from AHT session CSVs for agent consumption or quick inspection. "
            "Default output is JSON."
        )
    )
    parser.add_argument("--project-root", required=True, help="Target project root that owns the aht/ directory.")
    parser.add_argument("--session-dir", help="Optional exact session directory to scope the query.")
    parser.add_argument("--run-name", help="Optional exact run_name filter.")
    parser.add_argument("--limit", type=int, default=20, help="Maximum number of rows to return.")
    parser.add_argument(
        "--format",
        choices=("json", "table"),
        default="json",
        help="Output format. 'json' is recommended for agents.",
    )
    args = parser.parse_args()

    loaded_rows = load_history_rows(
        args.project_root,
        session_dir=args.session_dir,
        limit=args.limit,
        run_name=args.run_name,
    )
    rows = [
        {
            "id": row.id,
            "session_dir": row.session_dir,
            "run_id": row.run_id,
            "run_name": row.run_name,
            "status": row.status,
            "timestamp": row.timestamp,
            "hydra_config": row.hydra_config,
            "metrics": row.metrics,
            "primary_metric": row.primary_metric,
            "best_step": row.best_step,
            "notes": row.notes,
        }
        for row in loaded_rows
    ]
    if args.format == "table":
        print_table(rows)
        return
    print(json.dumps(rows, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
