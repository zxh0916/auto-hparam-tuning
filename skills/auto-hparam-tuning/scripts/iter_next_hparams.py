#!/usr/bin/env python3
"""Plan and execute one practical hyperparameter iteration backed by CSV session history."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from experiment_history import (
    canonical_json,
    get_by_dotted_path,
    latest_session_dir,
    load_history_rows,
    load_structured_file,
    set_by_dotted_path,
)
from session_manager import append_report, create_run, create_session, update_run_result, write_run_payloads


def dump_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def value_distance(left: Any, right: Any) -> float:
    if left == right:
        return 0.0
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return abs(float(left) - float(right))
    return 1.0


@dataclass(frozen=True)
class Candidate:
    values: dict[str, Any]
    source: str
    score: tuple[int, float, str]


def normalize_param_values(spec: dict[str, Any]) -> list[Any]:
    if "values" in spec:
        values = spec["values"]
    elif "choices" in spec:
        values = spec["choices"]
    elif all(key in spec for key in ("min", "max", "step")):
        start = spec["min"]
        stop = spec["max"]
        step = spec["step"]
        if step == 0:
            raise ValueError("Search-space step must be non-zero.")
        values = []
        current = start
        while current <= stop if step > 0 else current >= stop:
            if isinstance(start, int) and isinstance(stop, int) and isinstance(step, int):
                values.append(int(current))
            else:
                values.append(round(float(current), 12))
            current = current + step
            if len(values) > 10000:
                break
    else:
        raise ValueError("Each parameter spec must define either values/choices or min/max/step.")

    if not isinstance(values, list) or not values:
        raise ValueError("Each parameter spec must resolve to a non-empty list of candidate values.")
    return values


def parse_search_space(search_space: dict[str, Any]) -> tuple[dict[str, list[Any]], dict[str, Any]]:
    parameters = search_space.get("parameters")
    if not isinstance(parameters, dict) or not parameters:
        raise ValueError("Search space must contain a non-empty 'parameters' mapping.")

    normalized = {name: normalize_param_values(spec) for name, spec in parameters.items()}
    metric = search_space.get("metric") or {}
    metric_name = metric.get("name")
    goal = str(metric.get("goal", "max")).lower()
    if not metric_name:
        raise ValueError("Search space metric.name is required.")
    if goal not in {"max", "min"}:
        raise ValueError("Search space metric.goal must be 'max' or 'min'.")
    return normalized, {
        "name": metric_name,
        "goal": goal,
        "run_name_prefix": search_space.get("run_name_prefix", "aht"),
    }


def history_signature(run: Any, param_names: list[str]) -> tuple[str, ...]:
    return tuple(canonical_json(get_by_dotted_path(run.hydra_config, name)) for name in param_names)


def infer_base_values(base_config: dict[str, Any], param_names: list[str], candidates: dict[str, list[Any]]) -> dict[str, Any]:
    resolved: dict[str, Any] = {}
    for name in param_names:
        try:
            current_value = get_by_dotted_path(base_config, name)
        except KeyError:
            current_value = candidates[name][0]
        if current_value in candidates[name]:
            resolved[name] = current_value
            continue
        resolved[name] = min(candidates[name], key=lambda candidate: value_distance(candidate, current_value))
    return resolved


def best_run(rows: list[Any], metric_name: str, goal: str) -> Any | None:
    usable_rows = [row for row in rows if metric_name in row.metrics]
    if not usable_rows:
        return None
    reverse = goal == "max"
    return sorted(usable_rows, key=lambda row: (float(row.metrics[metric_name]), int(row.id)), reverse=reverse)[0]


def enumerate_neighbor_candidates(center: dict[str, Any], candidates: dict[str, list[Any]], tried_signatures: set[tuple[str, ...]], param_names: list[str]) -> list[Candidate]:
    scored: list[Candidate] = []
    for position, name in enumerate(param_names):
        values = candidates[name]
        current_value = center[name]
        if current_value not in values:
            continue
        current_index = values.index(current_value)
        for offset in (-1, 1):
            neighbor_index = current_index + offset
            if not 0 <= neighbor_index < len(values):
                continue
            proposal = dict(center)
            proposal[name] = values[neighbor_index]
            signature = tuple(canonical_json(proposal[param]) for param in param_names)
            if signature in tried_signatures:
                continue
            scored.append(Candidate(values=proposal, source=f"neighbor:{name}", score=(1, 1.0, f"{position}:{neighbor_index}")))
    return sorted(scored, key=lambda candidate: candidate.score)


def enumerate_fallback_candidates(base_values: dict[str, Any], candidates: dict[str, list[Any]], tried_signatures: set[tuple[str, ...]], param_names: list[str]) -> list[Candidate]:
    all_candidates: list[Candidate] = []

    def walk(index: int, current: dict[str, Any]) -> None:
        if index == len(param_names):
            signature = tuple(canonical_json(current[name]) for name in param_names)
            if signature in tried_signatures:
                return
            changed_count = sum(1 for name in param_names if current[name] != base_values[name])
            numeric_distance = sum(value_distance(current[name], base_values[name]) for name in param_names)
            tie_breaker = "|".join(canonical_json(current[name]) for name in param_names)
            all_candidates.append(Candidate(values=dict(current), source="fallback-grid", score=(changed_count, float(numeric_distance), tie_breaker)))
            return

        name = param_names[index]
        for value in candidates[name]:
            current[name] = value
            walk(index + 1, current)
        current.pop(name, None)

    walk(0, {})
    return sorted(all_candidates, key=lambda candidate: candidate.score)


def plan_next_values(*, base_config: dict[str, Any], candidate_values: dict[str, list[Any]], rows: list[Any], metric_name: str, goal: str) -> tuple[dict[str, Any], dict[str, Any]]:
    param_names = sorted(candidate_values)
    base_values = infer_base_values(base_config, param_names, candidate_values)
    tried_signatures = {history_signature(row, param_names) for row in rows if row.hydra_config}

    base_signature = tuple(canonical_json(base_values[name]) for name in param_names)
    if base_signature not in tried_signatures:
        return base_values, {"strategy": "baseline", "based_on_run_id": None}

    best = best_run(rows, metric_name=metric_name, goal=goal)
    if best is not None:
        center = {name: get_by_dotted_path(best.hydra_config, name) for name in param_names}
        neighbors = enumerate_neighbor_candidates(center, candidate_values, tried_signatures, param_names)
        if neighbors:
            first = neighbors[0]
            return first.values, {"strategy": first.source, "based_on_run_id": best.id}

    fallbacks = enumerate_fallback_candidates(base_values, candidate_values, tried_signatures, param_names)
    if fallbacks:
        first = fallbacks[0]
        return first.values, {"strategy": first.source, "based_on_run_id": best.id if best else None}

    raise RuntimeError("No untried hyperparameter proposal remains in the declared search space.")


def apply_overrides(base_config: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    result = json.loads(json.dumps(base_config))
    for key, value in overrides.items():
        set_by_dotted_path(result, key, value)
    return result


def hydra_override_tokens(overrides: dict[str, Any]) -> list[str]:
    return [f"{key}={canonical_json(value)}" for key, value in overrides.items()]


class LocalExecutor:
    def run(self, command: str, *, cwd: Path, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
        child_env = os.environ.copy()
        if env:
            child_env.update(env)
        return subprocess.run(command, cwd=str(cwd), env=child_env, shell=True, text=True, capture_output=True, check=False)


def load_metrics(path: Path) -> dict[str, Any]:
    value = load_structured_file(path)
    if not isinstance(value, dict):
        raise ValueError("Metrics file must contain a JSON/YAML object.")
    return value


def next_run_name(prefix: str, rows: list[Any]) -> str:
    max_id = max((int(row.id) for row in rows), default=0)
    return f"{prefix}-{max_id + 1:04d}"


def default_timestamp() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def build_command(template: str, *, run_name: str, config_path: Path, metrics_path: Path, work_dir: Path, artifact_dir: Path, overrides: dict[str, Any]) -> str:
    override_tokens = hydra_override_tokens(overrides)
    context = {
        "run_name": run_name,
        "config_path": str(config_path),
        "metrics_path": str(metrics_path),
        "work_dir": str(work_dir),
        "artifact_dir": str(artifact_dir),
        "output_dir": str(artifact_dir),
        "override_args": shlex.join(override_tokens),
        "override_json": canonical_json(overrides),
    }
    return template.format(**context)


def build_launch_env(*, run_name: str, artifact_dir: Path, config_path: Path, metrics_path: Path, work_dir: Path) -> dict[str, str]:
    artifact_dir_str = str(artifact_dir)
    return {
        "AHT_RUN_NAME": run_name,
        "AHT_ARTIFACT_DIR": artifact_dir_str,
        "AHT_OUTPUT_DIR": artifact_dir_str,
        "AHT_CONFIG_PATH": str(config_path),
        "AHT_METRICS_PATH": str(metrics_path),
        "AHT_WORK_DIR": str(work_dir),
    }


def resolve_active_session(project_root: Path, *, session_dir: str | None, base_command: str, primary_metric: str, goal: str) -> str:
    if session_dir:
        return str(Path(session_dir).expanduser().resolve())
    latest = latest_session_dir(project_root)
    if latest is not None:
        return str(latest)
    return create_session(
        project_root=str(project_root),
        base_command=base_command,
        primary_metric=primary_metric,
        goal=goal,
        notes="Auto-created by iter_next_hparams.py",
    )["session_dir"]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Perform one minimal hyperparameter iteration: inspect CSV-backed history, propose the next config, "
            "execute the training command, and append the result back into the active AHT session."
        )
    )
    parser.add_argument("--project-root", required=True, help="Target project root.")
    parser.add_argument("--session-dir", help="Optional exact session directory to append into.")
    parser.add_argument("--base-config-file", required=True, help="Resolved baseline config in JSON or YAML.")
    parser.add_argument("--search-space-file", required=True, help="JSON/YAML search-space declaration.")
    parser.add_argument(
        "--train-command-template",
        required=True,
        help=(
            "Shell command template for one run. Available placeholders: {run_name}, {config_path}, "
            "{metrics_path}, {work_dir}, {artifact_dir}, {output_dir}, {override_args}, {override_json}."
        ),
    )
    parser.add_argument("--history-limit", type=int, default=200)
    args = parser.parse_args()

    project_root = Path(args.project_root).expanduser().resolve()
    base_config = load_structured_file(Path(args.base_config_file).expanduser().resolve())
    search_space = load_structured_file(Path(args.search_space_file).expanduser().resolve())
    if not isinstance(base_config, dict):
        raise ValueError("Base config file must contain a JSON/YAML object.")
    if not isinstance(search_space, dict):
        raise ValueError("Search space file must contain a JSON/YAML object.")

    candidate_values, metadata = parse_search_space(search_space)
    history_rows = load_history_rows(project_root, limit=args.history_limit)
    planned_values, planning_meta = plan_next_values(
        base_config=base_config,
        candidate_values=candidate_values,
        rows=history_rows,
        metric_name=metadata["name"],
        goal=metadata["goal"],
    )
    proposed_config = apply_overrides(base_config, planned_values)
    run_name = next_run_name(metadata["run_name_prefix"], history_rows)
    session_dir = resolve_active_session(
        project_root,
        session_dir=args.session_dir,
        base_command=args.train_command_template,
        primary_metric=metadata["name"],
        goal=metadata["goal"],
    )

    run_info = create_run(session_dir)
    run_id = int(run_info["run_id"])
    run_dir = Path(run_info["run_dir"])
    config_path = run_dir / "resolved_config.json"
    metrics_path = run_dir / "metrics.json"
    stdout_path = Path(run_info["stdout_path"])
    stderr_path = Path(run_info["stderr_path"])
    try:
        dump_json(config_path, proposed_config)
        command = build_command(
            args.train_command_template,
            run_name=run_name,
            config_path=config_path,
            metrics_path=metrics_path,
            work_dir=run_dir,
            artifact_dir=run_dir,
            overrides=planned_values,
        )
        launch_env = build_launch_env(
            run_name=run_name,
            artifact_dir=run_dir,
            config_path=config_path,
            metrics_path=metrics_path,
            work_dir=run_dir,
        )

        started_at = default_timestamp()
        update_run_result(session_dir, run_id, run_name=run_name, status="running", start_time=started_at)

        executor = LocalExecutor()
        completed = executor.run(command, cwd=project_root, env=launch_env)
        stdout_path.write_text(completed.stdout, encoding="utf-8")
        stderr_path.write_text(completed.stderr, encoding="utf-8")

        if completed.returncode != 0:
            update_run_result(
                session_dir,
                run_id,
                run_name=run_name,
                status="failed",
                end_time=default_timestamp(),
                notes=f"Training command failed with exit code {completed.returncode}.",
            )
            raise RuntimeError(
                "Training command failed.\n"
                f"exit_code={completed.returncode}\n"
                f"stdout:\n{completed.stdout}\n"
                f"stderr:\n{completed.stderr}"
            )
        if not metrics_path.exists():
            update_run_result(
                session_dir,
                run_id,
                run_name=run_name,
                status="failed",
                end_time=default_timestamp(),
                notes=f"Training command succeeded but did not create metrics file: {metrics_path}",
            )
            raise FileNotFoundError(f"Training command finished successfully but did not create metrics file: {metrics_path}")

        metrics = load_metrics(metrics_path)
        write_run_payloads(
            session_dir,
            run_id,
            run_name=run_name,
            hydra_config=proposed_config,
            metrics=metrics,
            summary_content=(
                f"# Run {run_id}\n\n"
                f"Run name: `{run_name}`\n\n"
                f"Strategy: `{planning_meta['strategy']}`\n\n"
                f"Overrides: `{canonical_json(planned_values)}`\n"
            ),
        )
        primary_metric = metrics.get(metadata["name"])
        if not isinstance(primary_metric, (int, float)):
            primary_metric = None
        finished_at = metrics.get("timestamp") if isinstance(metrics.get("timestamp"), str) else default_timestamp()
        update_run_result(
            session_dir,
            run_id,
            run_name=run_name,
            status="finished",
            primary_metric=primary_metric,
            end_time=finished_at,
            notes=f"strategy={planning_meta['strategy']}",
        )
        append_report(
            session_dir,
            (
                f"## Run {run_id}: {run_name}\n"
                f"- Strategy: `{planning_meta['strategy']}`\n"
                f"- Based on run: `{planning_meta.get('based_on_run_id')}`\n"
                f"- Metric `{metadata['name']}`: `{primary_metric}`\n"
                f"- Overrides: `{canonical_json(planned_values)}`\n"
            ),
        )
    except Exception as exc:
        current_time = default_timestamp()
        update_run_result(
            session_dir,
            run_id,
            run_name=run_name,
            status="failed",
            end_time=current_time,
            notes=str(exc),
        )
        raise

    print(
        json.dumps(
            {
                "backend": "csv-session-store",
                "session_dir": session_dir,
                "run_id": run_id,
                "run_name": run_name,
                "history_rows_seen": len(history_rows),
                "planning": {
                    "metric_name": metadata["name"],
                    "goal": metadata["goal"],
                    **planning_meta,
                },
                "proposed_overrides": planned_values,
                "command": command,
                "metrics": metrics,
                "artifacts_dir": str(run_dir),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
