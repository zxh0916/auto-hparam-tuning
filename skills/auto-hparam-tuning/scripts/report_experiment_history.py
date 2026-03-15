#!/usr/bin/env python3
"""Summarize and render reports from the CSV-backed experiment history."""

from __future__ import annotations

import argparse
import html
from pathlib import Path
from string import Template

from experiment_history import (
    build_svg_polyline,
    config_diff,
    default_history_root,
    format_float,
    load_history_rows,
    pick_metric,
    sort_metric_values,
    table_lines,
)


def build_report_markdown(
    *,
    history_root: Path,
    rows,
    metric,
    top_rows,
    recent_rows,
) -> str:
    best_row, best_value = top_rows[0]
    metric_values_by_id = {row.id: value for row, value in metric.values}
    lines = [
        "# Experiment History Report",
        "",
        f"- History root: `{history_root}`",
        f"- Total runs: `{len(rows)}`",
        f"- Metric: `{metric.name}`",
        f"- Goal: `{metric.goal}`",
        f"- Metric coverage: `{metric.coverage}/{len(rows)}`",
        f"- Best run: `#{best_row.id} {best_row.run_name}` with `{format_float(best_value)}`",
        "",
        "## Top Runs",
        "",
    ]
    lines.extend(
        table_lines(
            ["rank", "id", "run_name", metric.name, "timestamp", "session"],
            [
                [
                    str(index),
                    str(row.id),
                    row.run_name,
                    format_float(value),
                    row.timestamp,
                    row.session_dir,
                ]
                for index, (row, value) in enumerate(top_rows, start=1)
            ],
        )
    )
    lines.extend(["", "## Recent Runs", ""])
    lines.extend(
        table_lines(
            ["id", "run_name", metric.name, "timestamp", "status"],
            [
                [
                    str(row.id),
                    row.run_name,
                    format_float(metric_values_by_id[row.id]) if row.id in metric_values_by_id else "n/a",
                    row.timestamp,
                    row.status,
                ]
                for row in recent_rows
            ],
        )
    )
    if len(top_rows) > 1:
        base_row = top_rows[0][0]
        lines.extend(["", "## Config Diffs Among Top Runs", ""])
        for row, value in top_rows[1:]:
            diffs = config_diff(base_row.hydra_config, row.hydra_config)
            lines.append(f"### #{row.id} {row.run_name} ({format_float(value)})")
            if diffs:
                lines.extend([f"- {item}" for item in diffs])
            else:
                lines.append("- No config diffs detected in stored config payloads.")
            lines.append("")
    return "\n".join(lines).strip() + "\n"


def render_html_report(
    *,
    template_path: Path,
    history_root: Path,
    rows,
    metric,
    top_rows,
    recent_rows,
) -> str:
    best_row, best_value = top_rows[0]
    metric_values_by_id = {row.id: value for row, value in metric.values}
    trend_points = [metric_values_by_id[row.id] for row in rows if row.id in metric_values_by_id]
    recent_runs_table = table_lines(
        ["id", "run_name", metric.name, "timestamp"],
        [
            [
                str(row.id),
                row.run_name,
                format_float(metric_values_by_id[row.id]) if row.id in metric_values_by_id else "n/a",
                row.timestamp,
            ]
            for row in recent_rows
        ],
    )
    top_runs_table = table_lines(
        ["rank", "id", "run_name", metric.name, "session"],
        [
            [
                str(index),
                str(row.id),
                row.run_name,
                format_float(value),
                row.session_dir,
            ]
            for index, (row, value) in enumerate(top_rows, start=1)
        ],
    )
    diff_sections = []
    if len(top_rows) > 1:
        base_row = top_rows[0][0]
        for row, value in top_rows[1:]:
            diffs = config_diff(base_row.hydra_config, row.hydra_config)
            if diffs:
                diff_list = "<ul>" + "".join(f"<li>{html.escape(item)}</li>" for item in diffs) + "</ul>"
            else:
                diff_list = "<p>No config diffs detected in stored config payloads.</p>"
            diff_sections.append(
                "<article class=\"card\">"
                f"<h2>Diff vs #{best_row.id} {html.escape(best_row.run_name)}</h2>"
                f"<p>Compared with #{row.id} {html.escape(row.run_name)} ({format_float(value)})</p>"
                f"{diff_list}"
                "</article>"
            )

    template = Template(template_path.read_text(encoding="utf-8"))
    return template.substitute(
        title="Experiment History Report",
        history_root=html.escape(str(history_root)),
        total_runs=str(len(rows)),
        metric_name=html.escape(metric.name),
        metric_goal=html.escape(metric.goal),
        metric_coverage=f"{metric.coverage}/{len(rows)}",
        best_value=html.escape(format_float(best_value)),
        best_run=html.escape(f"#{best_row.id} {best_row.run_name}"),
        trend_svg=build_svg_polyline(trend_points),
        recent_runs_table=_table_to_html(recent_runs_table),
        top_runs_table=_table_to_html(top_runs_table),
        diff_sections="".join(diff_sections),
    )


def _table_to_html(lines: list[str]) -> str:
    headers = [cell.strip() for cell in lines[0].strip("|").split("|")]
    body_lines = lines[2:]
    header_html = "".join(f"<th>{html.escape(cell)}</th>" for cell in headers)
    rows_html = []
    for line in body_lines:
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        rows_html.append("<tr>" + "".join(f"<td>{html.escape(cell)}</td>" for cell in cells) + "</tr>")
    return "<table><thead><tr>" + header_html + "</tr></thead><tbody>" + "".join(rows_html) + "</tbody></table>"


def print_summary(*, history_root: Path, rows, metric, top_rows, recent_rows) -> None:
    best_row, best_value = top_rows[0]
    latest_row = recent_rows[-1]
    metric_values_by_id = {row.id: value for row, value in metric.values}

    print(f"history_root: {history_root}")
    print(f"total_runs: {len(rows)}")
    print(f"metric: {metric.name}")
    print(f"goal: {metric.goal}")
    print(f"coverage: {metric.coverage}/{len(rows)}")
    print(f"best_run: #{best_row.id} {best_row.run_name} -> {format_float(best_value)}")
    latest_metric = metric_values_by_id.get(latest_row.id)
    if latest_metric is None:
        print(f"latest_run: #{latest_row.id} {latest_row.run_name} -> n/a")
    else:
        print(f"latest_run: #{latest_row.id} {latest_row.run_name} -> {format_float(latest_metric)}")
    print("")
    print("top_runs:")
    for index, (row, value) in enumerate(top_rows, start=1):
        print(f"  {index}. #{row.id} {row.run_name} {format_float(value)} [{row.status}]")
    print("recent_runs:")
    for row in recent_rows:
        metric_text = "n/a"
        if row.id in metric_values_by_id:
            metric_text = format_float(metric_values_by_id[row.id])
        print(f"  - #{row.id} {row.run_name} {metric_text} [{row.status}]")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize the CSV-backed AHT experiment history and render Markdown/HTML reports.",
    )
    parser.add_argument("--project-root", required=True, help="Target project root that owns the aht/ directory.")
    parser.add_argument("--session-dir", help="Optional exact session directory to scope the report.")
    parser.add_argument("--metric", default=None, help="Explicit metric to report.")
    parser.add_argument("--goal", choices=("auto", "min", "max"), default="auto")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--recent-k", type=int, default=8)
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_summary = subparsers.add_parser("summary", help="Print a concise CLI summary.")
    p_report = subparsers.add_parser("report", help="Render Markdown and/or HTML reports.")
    p_report.add_argument("--output-markdown", default=None)
    p_report.add_argument("--output-html", default=None)
    args = parser.parse_args()

    rows = load_history_rows(args.project_root, session_dir=args.session_dir)
    if not rows:
        raise SystemExit("No CSV-backed experiment history rows were found.")

    metric = pick_metric(rows, requested_metric=args.metric, goal_override=args.goal)
    top_rows = sort_metric_values(metric.values, goal=metric.goal)[: max(args.top_k, 1)]
    recent_rows = rows[-max(args.recent_k, 1) :]
    history_root = default_history_root(args.project_root)

    if args.command == "summary":
        print_summary(history_root=history_root, rows=rows, metric=metric, top_rows=top_rows, recent_rows=recent_rows)
        return

    if not args.output_markdown and not args.output_html:
        raise SystemExit("Provide --output-markdown and/or --output-html.")

    if args.output_markdown:
        markdown = build_report_markdown(
            history_root=history_root,
            rows=rows,
            metric=metric,
            top_rows=top_rows,
            recent_rows=recent_rows,
        )
        Path(args.output_markdown).write_text(markdown, encoding="utf-8")
    if args.output_html:
        template_path = Path(__file__).resolve().parent.parent / "assets" / "experiment_history_report_template.html"
        html_text = render_html_report(
            template_path=template_path,
            history_root=history_root,
            rows=rows,
            metric=metric,
            top_rows=top_rows,
            recent_rows=recent_rows,
        )
        Path(args.output_html).write_text(html_text, encoding="utf-8")


if __name__ == "__main__":
    main()
