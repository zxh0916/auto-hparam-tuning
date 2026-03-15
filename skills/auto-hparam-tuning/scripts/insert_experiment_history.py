#!/usr/bin/env python3
"""Insert one experiment row into the CSV-backed experiment history."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from experiment_history import latest_session_dir, load_structured_file
from session_manager import create_run, create_session, update_run_result, write_run_payloads


def load_structured_value(
    inline_value: str | None,
    file_path: str | None,
    *,
    label: str,
) -> Any:
    if bool(inline_value) == bool(file_path):
        raise ValueError(f"Provide exactly one of the inline or file arguments for {label}.")

    if inline_value is not None:
        try:
            return json.loads(inline_value)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Inline {label} must be valid JSON.") from exc

    return load_structured_file(Path(file_path).expanduser().resolve())


def default_timestamp() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Insert one CSV-backed experiment record. If no session is provided, the latest AHT session "
            "under the project root is reused, or a new one is created automatically."
        )
    )
    parser.add_argument("--project-root", required=True, help="Target project root that owns the AHT history.")
    parser.add_argument("--session-dir", help="Optional exact session directory to append into.")
    parser.add_argument("--run-name", required=True, help="Logical run label.")
    parser.add_argument("--config-json", help="Inline JSON string for the resolved Hydra config.")
    parser.add_argument("--config-file", help="Path to the resolved Hydra config file in JSON or YAML format.")
    parser.add_argument("--metrics-json", help="Inline JSON string for metrics.")
    parser.add_argument("--metrics-file", help="Path to metrics JSON or YAML file.")
    parser.add_argument("--primary-metric-name", default=None, help="Metric key to mirror into results.csv.")
    parser.add_argument("--status", default="finished")
    parser.add_argument("--best-step", default=None)
    parser.add_argument("--timestamp", default=default_timestamp())
    parser.add_argument("--notes", default=None)
    parser.add_argument("--base-command", default=None)
    parser.add_argument("--goal", default=None)
    args = parser.parse_args()

    hydra_config = load_structured_value(args.config_json, args.config_file, label="config")
    metrics = load_structured_value(args.metrics_json, args.metrics_file, label="metrics")
    if not isinstance(hydra_config, dict):
        raise ValueError("Resolved config must be a JSON/YAML object.")
    if not isinstance(metrics, dict):
        raise ValueError("Metrics must be a JSON/YAML object.")

    session_dir = args.session_dir or latest_session_dir(args.project_root)
    if session_dir is None:
        session_dir = create_session(
            project_root=args.project_root,
            base_command=args.base_command,
            primary_metric=args.primary_metric_name,
            goal=args.goal,
            notes="Auto-created by insert_experiment_history.py",
        )["session_dir"]

    run_info = create_run(str(session_dir))
    run_id = int(run_info["run_id"])
    primary_metric = None
    if args.primary_metric_name:
        raw_value = metrics.get(args.primary_metric_name)
        if isinstance(raw_value, (int, float)):
            primary_metric = raw_value

    write_run_payloads(
        str(session_dir),
        run_id,
        run_name=args.run_name,
        hydra_config=hydra_config,
        metrics=metrics,
    )
    update_run_result(
        str(session_dir),
        run_id,
        run_name=args.run_name,
        status=args.status,
        primary_metric=primary_metric,
        best_step=args.best_step,
        start_time=args.timestamp,
        end_time=args.timestamp,
        notes=args.notes,
    )

    print(
        json.dumps(
            {
                "backend": "csv-session-store",
                "session_dir": str(session_dir),
                "run_id": run_id,
                "run_name": args.run_name,
                "timestamp": args.timestamp,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
