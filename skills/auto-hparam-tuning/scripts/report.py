"""Experiment report generator for AHT.

Reads the experiment history and produces a Markdown summary report including
top-k run rankings, parameter-result comparison tables, and iteration insights.

Usage
-----
CLI::

    python report.py --project /path/to/project
    python report.py --project /path/to/project --metric val/accuracy --mode max --top 10
    python report.py --project /path/to/project --output aht_report.md

Library::

    from report import generate_report
    md = generate_report("/path/to/project", metric="val/accuracy", mode="max")
    print(md)
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _load_history(project_dir: str) -> list[dict[str, Any]]:
    """Load run history from the project's aht_history.jsonl."""
    history_file = Path(project_dir) / "aht_history.jsonl"
    if not history_file.exists():
        return []
    records: list[dict[str, Any]] = []
    with open(history_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return records


def _format_value(v: Any) -> str:
    """Format a value for display in a Markdown table."""
    if v is None:
        return "N/A"
    if isinstance(v, float):
        if abs(v) < 0.001 and v != 0:
            return f"{v:.2e}"
        return f"{v:.4f}"
    return str(v)


def _collect_all_config_keys(records: list[dict[str, Any]]) -> list[str]:
    """Collect all unique config keys across all runs."""
    keys: set[str] = set()
    for r in records:
        keys.update(r.get("config", {}).keys())
    return sorted(keys)


def _collect_all_metric_keys(records: list[dict[str, Any]]) -> list[str]:
    """Collect all unique metric keys across all runs."""
    keys: set[str] = set()
    for r in records:
        keys.update(r.get("metrics", {}).keys())
    return sorted(keys)


def generate_report(
    project_dir: str,
    metric: str | None = None,
    mode: str = "max",
    top_k: int = 10,
) -> str:
    """Generate a Markdown experiment report.

    Parameters
    ----------
    project_dir : str
        Path to the project root containing ``aht_history.jsonl``.
    metric : str or None
        Primary metric to rank by. If *None*, uses the first metric found.
    mode : str
        ``"max"`` or ``"min"`` — whether higher or lower is better.
    top_k : int
        Number of top runs to show in the ranking.

    Returns
    -------
    str
        Complete Markdown report.
    """
    records = _load_history(project_dir)

    if not records:
        return (
            "# AHT Experiment Report\n\n"
            f"**Project**: `{project_dir}`\n\n"
            "No experiment runs recorded yet. Run experiments first.\n"
        )

    # Filter successful runs for ranking
    successful = [r for r in records if r.get("status") == "success"]
    failed = [r for r in records if r.get("status") != "success"]

    # Determine primary metric
    all_metrics = _collect_all_metric_keys(records)
    if metric is None and all_metrics:
        metric = all_metrics[0]

    descending = mode.lower() in {"max", "higher", "maximize"}

    # Sort successful runs by the primary metric
    if metric and successful:
        def _metric_val(r: dict) -> float:
            v = r.get("metrics", {}).get(metric)
            return float(v) if v is not None else float("-inf") if descending else float("inf")
        successful.sort(key=_metric_val, reverse=descending)

    # --- Build Report ---
    sections: list[str] = []
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    # Header
    sections.append(f"# AHT Experiment Report\n")
    sections.append(f"**Project**: `{project_dir}`  ")
    sections.append(f"**Generated**: {timestamp}  ")
    sections.append(f"**Total Runs**: {len(records)} ({len(successful)} successful, {len(failed)} failed)  ")
    if metric:
        direction = "higher is better" if descending else "lower is better"
        sections.append(f"**Primary Metric**: `{metric}` ({direction})  ")
    sections.append("")

    # --- Executive Summary ---
    sections.append("## Executive Summary\n")
    if successful and metric:
        best_run = successful[0]
        best_val = best_run.get("metrics", {}).get(metric)
        worst_successful = successful[-1] if len(successful) > 1 else None

        sections.append(f"- **Best run**: `{best_run.get('run_name', 'N/A')}` — "
                        f"`{metric}` = **{_format_value(best_val)}**")
        if worst_successful and len(successful) > 1:
            worst_val = worst_successful.get("metrics", {}).get(metric)
            sections.append(f"- **Worst successful run**: `{worst_successful.get('run_name', 'N/A')}` — "
                            f"`{metric}` = {_format_value(worst_val)}")

        # Compute improvement from first to best
        first_run = records[0]
        first_val = first_run.get("metrics", {}).get(metric)
        if first_val is not None and best_val is not None:
            improvement = best_val - first_val if descending else first_val - best_val
            sections.append(f"- **Improvement** from first run to best: "
                            f"{_format_value(improvement)} "
                            f"({'↑' if improvement > 0 else '↓'} {_format_value(abs(improvement))})")

    if failed:
        fail_types: dict[str, int] = {}
        for r in failed:
            status = r.get("status", "unknown")
            fail_types[status] = fail_types.get(status, 0) + 1
        fail_summary = ", ".join(f"{v} {k}" for k, v in sorted(fail_types.items()))
        sections.append(f"- **Failed runs**: {fail_summary}")

    sections.append("")

    # --- Top-K Ranking ---
    sections.append(f"## Top-{min(top_k, len(successful))} Runs\n")

    if successful:
        metric_cols = _collect_all_metric_keys(successful[:top_k])
        config_keys = _collect_all_config_keys(successful[:top_k])

        # Build ranking table
        header_parts = ["#", "Run Name"]
        for m in metric_cols:
            header_parts.append(f"`{m}`")
        header_parts.append("Diagnosis")
        header_parts.append("Duration")

        header = "| " + " | ".join(header_parts) + " |"
        sep = "| " + " | ".join(["---"] * len(header_parts)) + " |"
        sections.append(header)
        sections.append(sep)

        for i, r in enumerate(successful[:top_k], 1):
            row_parts: list[str] = [str(i), r.get("run_name", "N/A")]
            for m in metric_cols:
                val = r.get("metrics", {}).get(m)
                row_parts.append(_format_value(val))
            row_parts.append(r.get("diagnosis", "N/A"))
            dur = r.get("duration_seconds")
            row_parts.append(f"{dur:.0f}s" if dur else "N/A")
            sections.append("| " + " | ".join(row_parts) + " |")

        sections.append("")
    else:
        sections.append("No successful runs to rank.\n")

    # --- Parameter-Result Comparison ---
    if successful and len(successful) > 1:
        sections.append("## Parameter-Result Comparison\n")
        config_keys = _collect_all_config_keys(successful[:top_k])

        if config_keys and metric:
            header_parts = ["Run Name"]
            for k in config_keys:
                header_parts.append(f"`{k}`")
            header_parts.append(f"`{metric}`")

            header = "| " + " | ".join(header_parts) + " |"
            sep = "| " + " | ".join(["---"] * len(header_parts)) + " |"
            sections.append(header)
            sections.append(sep)

            for r in successful[:top_k]:
                row_parts: list[str] = [r.get("run_name", "N/A")]
                for k in config_keys:
                    val = r.get("config", {}).get(k)
                    row_parts.append(_format_value(val))
                row_parts.append(_format_value(r.get("metrics", {}).get(metric)))
                sections.append("| " + " | ".join(row_parts) + " |")

            sections.append("")

    # --- Diagnosis Distribution ---
    sections.append("## Diagnosis Distribution\n")
    diag_counts: dict[str, int] = {}
    for r in records:
        d = r.get("diagnosis", "N/A") or "N/A"
        diag_counts[d] = diag_counts.get(d, 0) + 1

    for d, c in sorted(diag_counts.items(), key=lambda x: -x[1]):
        bar = "█" * c
        sections.append(f"- **{d}**: {c} runs {bar}")
    sections.append("")

    # --- Failed Runs ---
    if failed:
        sections.append("## Failed Runs\n")
        for r in failed:
            sections.append(f"- `{r.get('run_name', 'N/A')}` — status: **{r.get('status', 'unknown')}**")
            if r.get("config"):
                sections.append(f"  - Config: `{json.dumps(r['config'])}`")
            stderr = r.get("stderr", "")
            if stderr:
                sections.append(f"  - Error (last 200 chars): `{stderr[-200:]}`")
        sections.append("")

    # --- Iteration Timeline ---
    sections.append("## Iteration Timeline\n")
    for i, r in enumerate(records, 1):
        metric_str = ""
        if metric and r.get("metrics", {}).get(metric) is not None:
            metric_str = f" → `{metric}`={_format_value(r['metrics'][metric])}"
        status_emoji = {"success": "✅", "failed": "❌", "timeout": "⏰", "oom": "💥"}.get(
            r.get("status", ""), "❓"
        )
        sections.append(
            f"{i}. {status_emoji} **{r.get('run_name', 'N/A')}** "
            f"[{r.get('status', 'N/A')}]{metric_str}"
        )
        if r.get("diagnosis"):
            sections.append(f"   - Diagnosis: {r['diagnosis']}")
    sections.append("")

    # --- Best Configuration ---
    if successful and metric:
        sections.append("## Best Configuration\n")
        best = successful[0]
        sections.append(f"**Run**: `{best.get('run_name', 'N/A')}`  ")
        sections.append(f"**{metric}**: {_format_value(best.get('metrics', {}).get(metric))}  ")
        sections.append("")
        if best.get("config"):
            sections.append("```yaml")
            for k, v in sorted(best["config"].items()):
                sections.append(f"{k}: {v}")
            sections.append("```")
        if best.get("command"):
            sections.append(f"\n**Command**: `{best['command']}`")
        sections.append("")

    # Footer
    sections.append("---")
    sections.append(f"*Report generated by AHT (Automatic Hyperparameter Tuning) at {timestamp}*\n")

    return "\n".join(sections)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a Markdown experiment report from the AHT history file.\n\n"
            "Reads aht_history.jsonl from the project directory and produces a "
            "formatted report with rankings, parameter comparisons, and insights."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--project", "-p", required=True, metavar="DIR",
        help="Project directory containing aht_history.jsonl.",
    )
    parser.add_argument(
        "--metric", "-m", default=None, metavar="KEY",
        help="Primary metric to rank by (e.g. val/accuracy). Auto-detected if omitted.",
    )
    parser.add_argument(
        "--mode", default="max", choices=["max", "min"],
        help="Optimization direction: 'max' or 'min'. Default: max.",
    )
    parser.add_argument(
        "--top-k", "-k", type=int, default=10, metavar="N",
        help="Number of top runs to show. Default: 10.",
    )
    parser.add_argument(
        "--output", "-o", default=None, metavar="FILE",
        help="Output file path. Default: print to stdout.",
    )

    args = parser.parse_args()

    report = generate_report(
        project_dir=args.project,
        metric=args.metric,
        mode=args.mode,
        top_k=args.top_k,
    )

    if args.output:
        Path(args.output).write_text(report, encoding="utf-8")
        print(f"Report written to {args.output}", file=sys.stderr)
    else:
        print(report)


if __name__ == "__main__":
    main()
