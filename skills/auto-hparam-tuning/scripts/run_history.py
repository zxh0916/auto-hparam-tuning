"""Experiment history tracker for AHT.

Provides a JSONL-based experiment database that records every run's full
configuration, command, metrics, diagnosis, and metadata for reproducibility
and analysis.

Usage
-----
CLI — record a run::

    python run_history.py record \\
        --project /path/to/project \\
        --run-name lr_sweep_01 \\
        --config '{"optimizer.lr": 0.0003}' \\
        --metrics '{"val/accuracy": 0.87, "val/loss": 0.34}' \\
        --diagnosis plateau \\
        --status success

CLI — list runs::

    python run_history.py list --project /path/to/project
    python run_history.py list --project /path/to/project --sort-by "metrics.val/accuracy" --top 5

CLI — show best run::

    python run_history.py best --project /path/to/project --metric "val/accuracy" --mode max

Library::

    from run_history import RunHistory
    history = RunHistory("/path/to/project")
    history.record(run_name="trial_01", config={"lr": 0.001}, metrics={"val/acc": 0.9})
    runs = history.list_runs()
    best = history.best_run(metric="val/acc", mode="max")
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


HISTORY_FILENAME = "aht_history.jsonl"


def _get_git_info(project_dir: str) -> dict[str, Any]:
    """Collect git metadata from the project directory."""
    info: dict[str, Any] = {"available": False}
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_dir,
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            info["available"] = True
            info["commit_hash"] = result.stdout.strip()

            # Check dirty state
            diff_result = subprocess.run(
                ["git", "diff", "--stat"],
                cwd=project_dir,
                capture_output=True,
                text=True,
                timeout=10,
            )
            info["dirty"] = bool(diff_result.stdout.strip())

            # Get branch name
            branch_result = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                cwd=project_dir,
                capture_output=True,
                text=True,
                timeout=10,
            )
            if branch_result.returncode == 0:
                info["branch"] = branch_result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return info


def _get_system_info() -> dict[str, Any]:
    """Collect basic system information."""
    info: dict[str, Any] = {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
    }

    # Try to get GPU info
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            gpus = []
            for line in result.stdout.strip().split("\n"):
                if line.strip():
                    gpus.append(line.strip())
            info["gpus"] = gpus
    except (FileNotFoundError, subprocess.TimeoutExpired):
        info["gpus"] = []

    return info


def _generate_run_id(run_name: str, timestamp: str) -> str:
    """Generate a short deterministic run ID."""
    raw = f"{run_name}:{timestamp}"
    return hashlib.sha256(raw.encode()).hexdigest()[:12]


class RunHistory:
    """JSONL-backed experiment history for a project.

    Parameters
    ----------
    project_dir : str
        Path to the project root. The history file ``aht_history.jsonl``
        will be created/read from this directory.
    """

    def __init__(self, project_dir: str) -> None:
        self.project_dir = Path(project_dir).resolve()
        self.history_file = self.project_dir / HISTORY_FILENAME

    def _load_all(self) -> list[dict[str, Any]]:
        """Load all run records from the JSONL file."""
        if not self.history_file.exists():
            return []
        records: list[dict[str, Any]] = []
        with open(self.history_file, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    print(
                        f"Warning: malformed JSON on line {line_no} of {self.history_file}",
                        file=sys.stderr,
                    )
        return records

    def _append(self, record: dict[str, Any]) -> None:
        """Append one record to the JSONL file."""
        with open(self.history_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def record(
        self,
        run_name: str,
        config: dict[str, Any] | None = None,
        overrides: list[str] | None = None,
        command: str | None = None,
        metrics: dict[str, float] | None = None,
        diagnosis: str | None = None,
        diagnosis_details: dict[str, Any] | None = None,
        status: str = "unknown",
        duration_seconds: float | None = None,
        event_files: list[str] | None = None,
        notes: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Record a single experiment run.

        Parameters
        ----------
        run_name : str
            Human-friendly name for the run.
        config : dict or None
            The hyperparameter configuration used.
        overrides : list[str] or None
            Raw Hydra override strings.
        command : str or None
            The full command that was executed.
        metrics : dict or None
            Final metric values (e.g., ``{"val/accuracy": 0.87}``).
        diagnosis : str or None
            Primary diagnostic label (e.g., ``"overfitting"``).
        diagnosis_details : dict or None
            Full diagnosis output from ``detect_patterns``.
        status : str
            Run status: ``"success"``, ``"failed"``, ``"timeout"``, ``"oom"``.
        duration_seconds : float or None
            Wall-clock run duration.
        event_files : list[str] or None
            Paths to TensorBoard event files.
        notes : str or None
            Free-form notes.
        extra : dict or None
            Any additional metadata.

        Returns
        -------
        dict
            The complete run record that was saved.
        """
        timestamp = datetime.now(timezone.utc).isoformat()
        run_id = _generate_run_id(run_name, timestamp)

        record: dict[str, Any] = {
            "run_id": run_id,
            "run_name": run_name,
            "timestamp": timestamp,
            "status": status,
            "config": config or {},
            "overrides": overrides or [],
            "command": command,
            "metrics": metrics or {},
            "diagnosis": diagnosis,
            "diagnosis_details": diagnosis_details,
            "duration_seconds": duration_seconds,
            "event_files": event_files or [],
            "notes": notes,
            "git": _get_git_info(str(self.project_dir)),
            "system": _get_system_info(),
        }
        if extra:
            record["extra"] = extra

        self._append(record)
        return record

    def list_runs(
        self,
        status: str | None = None,
        sort_by: str | None = None,
        descending: bool = True,
        top: int | None = None,
    ) -> list[dict[str, Any]]:
        """List recorded runs with optional filtering and sorting.

        Parameters
        ----------
        status : str or None
            Filter by status (e.g., ``"success"``).
        sort_by : str or None
            Dot-separated key path to sort by (e.g., ``"metrics.val/accuracy"``).
        descending : bool
            Sort in descending order (default *True*).
        top : int or None
            Return only the top N runs.

        Returns
        -------
        list[dict]
            Filtered, sorted run records.
        """
        records = self._load_all()

        if status:
            records = [r for r in records if r.get("status") == status]

        if sort_by:
            def _get_nested(d: dict, path: str) -> Any:
                for part in path.split("."):
                    if isinstance(d, dict):
                        d = d.get(part, None)  # type: ignore
                    else:
                        return None
                return d

            records.sort(
                key=lambda r: (_get_nested(r, sort_by) or float("-inf")),
                reverse=descending,
            )

        if top:
            records = records[:top]

        return records

    def best_run(
        self,
        metric: str,
        mode: str = "max",
        status: str = "success",
    ) -> dict[str, Any] | None:
        """Find the best run according to a specific metric.

        Parameters
        ----------
        metric : str
            The metric key (e.g., ``"val/accuracy"``).
        mode : str
            ``"max"`` or ``"min"``.
        status : str
            Only consider runs with this status.

        Returns
        -------
        dict or None
            The best run record, or *None* if no matching runs exist.
        """
        records = self.list_runs(status=status)
        if not records:
            return None

        valid = [(r, r.get("metrics", {}).get(metric)) for r in records]
        valid = [(r, v) for r, v in valid if v is not None]

        if not valid:
            return None

        if mode.lower() in {"max", "higher", "maximize"}:
            best = max(valid, key=lambda x: x[1])
        else:
            best = min(valid, key=lambda x: x[1])

        return best[0]

    def summary_table(self, top: int = 10, metric: str | None = None, mode: str = "max") -> str:
        """Generate a Markdown summary table of runs.

        Parameters
        ----------
        top : int
            Maximum number of runs to show.
        metric : str or None
            If given, sort by this metric.
        mode : str
            Sort direction for the metric.

        Returns
        -------
        str
            Markdown-formatted table.
        """
        if metric:
            runs = self.list_runs(
                sort_by=f"metrics.{metric}",
                descending=(mode.lower() in {"max", "higher", "maximize"}),
                top=top,
                status="success",
            )
        else:
            runs = self.list_runs(top=top)

        if not runs:
            return "No runs recorded yet.\n"

        # Collect all metric keys across runs
        all_metrics: set[str] = set()
        for r in runs:
            all_metrics.update(r.get("metrics", {}).keys())
        metric_cols = sorted(all_metrics)

        # Build header
        header = "| # | Run Name | Status | " + " | ".join(metric_cols) + " | Diagnosis | Duration |"
        separator = "|---|---|---|" + "|".join(["---"] * len(metric_cols)) + "|---|---|"

        lines = [header, separator]
        for i, r in enumerate(runs, 1):
            metrics_vals = [
                f"{r.get('metrics', {}).get(m, 'N/A')}" if isinstance(r.get('metrics', {}).get(m), (int, float))
                else str(r.get("metrics", {}).get(m, "N/A"))
                for m in metric_cols
            ]
            duration = r.get("duration_seconds")
            dur_str = f"{duration:.0f}s" if duration else "N/A"
            row = (
                f"| {i} | {r.get('run_name', 'N/A')} | {r.get('status', 'N/A')} | "
                + " | ".join(metrics_vals)
                + f" | {r.get('diagnosis', 'N/A')} | {dur_str} |"
            )
            lines.append(row)

        return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="AHT experiment history manager.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Sub-command")

    # --- record ---
    record_parser = subparsers.add_parser("record", help="Record a run")
    record_parser.add_argument("--project", "-p", required=True, help="Project directory")
    record_parser.add_argument("--run-name", "-n", required=True, help="Run name")
    record_parser.add_argument("--config", default=None, help="JSON config dict")
    record_parser.add_argument("--overrides", nargs="*", default=None, help="Hydra overrides")
    record_parser.add_argument("--command", default=None, help="Full command string")
    record_parser.add_argument("--metrics", default=None, help="JSON metrics dict")
    record_parser.add_argument("--diagnosis", default=None, help="Diagnosis label")
    record_parser.add_argument("--status", default="success", help="Run status")
    record_parser.add_argument("--duration", type=float, default=None, help="Duration in seconds")
    record_parser.add_argument("--notes", default=None, help="Free-form notes")

    # --- list ---
    list_parser = subparsers.add_parser("list", help="List recorded runs")
    list_parser.add_argument("--project", "-p", required=True, help="Project directory")
    list_parser.add_argument("--status", default=None, help="Filter by status")
    list_parser.add_argument("--sort-by", default=None, help="Dot-path to sort by")
    list_parser.add_argument("--top", type=int, default=None, help="Show top N runs")
    list_parser.add_argument("--format", choices=["json", "table"], default="json", help="Output format")

    # --- best ---
    best_parser = subparsers.add_parser("best", help="Show the best run")
    best_parser.add_argument("--project", "-p", required=True, help="Project directory")
    best_parser.add_argument("--metric", required=True, help="Metric key to optimize")
    best_parser.add_argument("--mode", default="max", help="'max' or 'min'")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == "record":
        history = RunHistory(args.project)
        config = json.loads(args.config) if args.config else None
        metrics = json.loads(args.metrics) if args.metrics else None
        record = history.record(
            run_name=args.run_name,
            config=config,
            overrides=args.overrides,
            command=args.command,
            metrics=metrics,
            diagnosis=args.diagnosis,
            status=args.status,
            duration_seconds=args.duration,
            notes=args.notes,
        )
        print(json.dumps(record, indent=2, ensure_ascii=False))

    elif args.command == "list":
        history = RunHistory(args.project)
        if args.format == "table":
            print(history.summary_table(
                top=args.top or 20,
                metric=args.sort_by.replace("metrics.", "") if args.sort_by and args.sort_by.startswith("metrics.") else None,
            ))
        else:
            runs = history.list_runs(
                status=args.status,
                sort_by=args.sort_by,
                top=args.top,
            )
            print(json.dumps(runs, indent=2, ensure_ascii=False))

    elif args.command == "best":
        history = RunHistory(args.project)
        best = history.best_run(metric=args.metric, mode=args.mode)
        if best:
            print(json.dumps(best, indent=2, ensure_ascii=False))
        else:
            print("No matching runs found.", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()
