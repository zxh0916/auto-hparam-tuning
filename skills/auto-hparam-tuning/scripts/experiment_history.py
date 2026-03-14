from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:  # pragma: no cover - optional dependency fallback
    yaml = None


DEFAULT_AHT_DIRNAME = "aht"
DEFAULT_METRIC_CANDIDATES = (
    "val/loss",
    "validation/loss",
    "eval/loss",
    "loss",
    "val/acc",
    "validation/acc",
    "eval/acc",
    "accuracy",
    "score",
    "reward",
)

RESULTS_COLUMNS = [
    "run_id",
    "run_name",
    "status",
    "primary_metric",
    "best_step",
    "run_dir",
    "override_path",
    "config_path",
    "metrics_path",
    "summary_path",
    "start_time",
    "end_time",
    "notes",
]


@dataclass(frozen=True)
class SessionRecord:
    project_root: Path
    session_dir: Path
    meta: dict[str, Any]


@dataclass(frozen=True)
class ExperimentRow:
    id: int
    session_dir: str
    session_created_at: str
    run_id: int
    run_name: str
    status: str
    hydra_config: dict[str, Any]
    metrics: dict[str, Any]
    timestamp: str
    primary_metric: float | None
    best_step: int | None
    notes: str
    results_row: dict[str, str]


@dataclass
class MetricSummary:
    name: str
    goal: str
    coverage: int
    values: list[tuple[ExperimentRow, float]]


def load_structured_file(path: Path) -> Any:
    text = path.read_text(encoding="utf-8")
    suffix = path.suffix.lower()
    if suffix == ".json":
        return json.loads(text)
    if suffix in {".yaml", ".yml"}:
        if yaml is None:
            raise RuntimeError("PyYAML is required to read YAML files.")
        return yaml.safe_load(text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        if yaml is None:
            raise ValueError(f"Unsupported file format for {path}. Use JSON or YAML.")
        return yaml.safe_load(text)


def dump_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def get_by_dotted_path(data: Any, dotted_path: str) -> Any:
    current = data
    for part in dotted_path.split("."):
        if not isinstance(current, dict) or part not in current:
            raise KeyError(dotted_path)
        current = current[part]
    return current


def set_by_dotted_path(data: dict[str, Any], dotted_path: str, value: Any) -> None:
    current: dict[str, Any] = data
    parts = dotted_path.split(".")
    for part in parts[:-1]:
        next_value = current.get(part)
        if not isinstance(next_value, dict):
            next_value = {}
            current[part] = next_value
        current = next_value
    current[parts[-1]] = value


def default_history_root(project_root: str | Path) -> Path:
    return Path(project_root).expanduser().resolve() / DEFAULT_AHT_DIRNAME


def _load_yaml_meta(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    if yaml is not None:
        value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if isinstance(value, dict):
            return value
        return {}

    result: dict[str, Any] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if ":" not in line:
            continue
        key, raw = line.split(":", 1)
        text = raw.strip().strip('"')
        result[key.strip()] = text
    return result


def _session_sort_key(session_dir: Path) -> tuple[str, str]:
    return (session_dir.parent.name, session_dir.name)


def find_sessions(project_root: str | Path) -> list[SessionRecord]:
    root = default_history_root(project_root)
    if not root.exists():
        return []

    sessions: list[SessionRecord] = []
    project_path = Path(project_root).expanduser().resolve()
    for date_dir in sorted((path for path in root.iterdir() if path.is_dir()), key=lambda path: path.name):
        for time_dir in sorted((path for path in date_dir.iterdir() if path.is_dir()), key=lambda path: path.name):
            meta = _load_yaml_meta(time_dir / "meta.yaml")
            sessions.append(SessionRecord(project_root=project_path, session_dir=time_dir, meta=meta))
    return sessions


def latest_session_dir(project_root: str | Path) -> Path | None:
    sessions = find_sessions(project_root)
    if not sessions:
        return None
    return sorted((record.session_dir for record in sessions), key=_session_sort_key)[-1]


def _read_results_rows(results_path: Path) -> list[dict[str, str]]:
    if not results_path.exists():
        return []
    text = results_path.read_text(encoding="utf-8").strip()
    if not text:
        return []

    rows: list[dict[str, str]] = []
    reader = csv.DictReader(text.splitlines())
    for raw_row in reader:
        row = {column: str(raw_row.get(column, "") or "") for column in RESULTS_COLUMNS}
        rows.append(row)
    return rows


def _safe_int(value: Any) -> int | None:
    text = str(value).strip()
    if not text:
        return None
    try:
        return int(float(text))
    except ValueError:
        return None


def _safe_float(value: Any) -> float | None:
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _resolve_payload(base_dir: Path, relpath: str, default_name: str) -> Path:
    text = relpath.strip()
    if text:
        return base_dir / text
    return base_dir / default_name


def _load_json_mapping(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    value = load_structured_file(path)
    if isinstance(value, dict):
        return value
    return {}


def load_history_rows(
    project_root: str | Path,
    *,
    session_dir: str | Path | None = None,
    limit: int | None = None,
    run_name: str | None = None,
) -> list[ExperimentRow]:
    sessions = find_sessions(project_root)
    scoped_dir = Path(session_dir).expanduser().resolve() if session_dir else None

    rows: list[ExperimentRow] = []
    for session in sessions:
        if scoped_dir is not None and session.session_dir != scoped_dir:
            continue
        results_rows = _read_results_rows(session.session_dir / "results.csv")
        for results_row in results_rows:
            run_id = _safe_int(results_row.get("run_id"))
            if run_id is None:
                continue
            normalized_run_name = results_row.get("run_name") or f"run-{run_id:04d}"
            if run_name and normalized_run_name != run_name:
                continue

            config_path = _resolve_payload(session.session_dir, results_row.get("config_path", ""), f"runs/{run_id}/resolved_config.json")
            metrics_path = _resolve_payload(session.session_dir, results_row.get("metrics_path", ""), f"runs/{run_id}/metrics.json")
            hydra_config = _load_json_mapping(config_path)
            metrics = _load_json_mapping(metrics_path)

            rows.append(
                ExperimentRow(
                    id=0,
                    session_dir=str(session.session_dir),
                    session_created_at=str(session.meta.get("created_at") or session.session_dir.as_posix()),
                    run_id=run_id,
                    run_name=normalized_run_name,
                    status=results_row.get("status", ""),
                    hydra_config=hydra_config,
                    metrics=metrics,
                    timestamp=results_row.get("end_time") or results_row.get("start_time") or str(session.meta.get("created_at") or ""),
                    primary_metric=_safe_float(results_row.get("primary_metric")),
                    best_step=_safe_int(results_row.get("best_step")),
                    notes=results_row.get("notes", ""),
                    results_row=results_row,
                )
            )

    rows.sort(key=lambda row: (row.session_created_at, row.run_id))
    numbered = [
        ExperimentRow(
            id=index,
            session_dir=row.session_dir,
            session_created_at=row.session_created_at,
            run_id=row.run_id,
            run_name=row.run_name,
            status=row.status,
            hydra_config=row.hydra_config,
            metrics=row.metrics,
            timestamp=row.timestamp,
            primary_metric=row.primary_metric,
            best_step=row.best_step,
            notes=row.notes,
            results_row=row.results_row,
        )
        for index, row in enumerate(rows, start=1)
    ]

    if limit is not None:
        return numbered[-limit:]
    return numbered


def flatten_mapping(value: dict[str, Any], *, prefix: str = "") -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    for key, item in value.items():
        full_key = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(item, dict):
            flattened.update(flatten_mapping(item, prefix=full_key))
        else:
            flattened[full_key] = item
    return flattened


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def metric_goal(metric_name: str) -> str:
    name = metric_name.lower()
    if any(token in name for token in ("loss", "error", "wer", "cer", "perplexity", "latency", "time")):
        return "min"
    return "max"


def metric_priority(metric_name: str) -> int:
    lowered = metric_name.lower()
    for index, candidate in enumerate(DEFAULT_METRIC_CANDIDATES):
        if lowered == candidate.lower():
            return index
    if "loss" in lowered:
        return len(DEFAULT_METRIC_CANDIDATES)
    if any(token in lowered for token in ("acc", "accuracy", "f1", "auc", "score", "reward")):
        return len(DEFAULT_METRIC_CANDIDATES) + 1
    return len(DEFAULT_METRIC_CANDIDATES) + 2


def collect_numeric_metrics(rows: list[ExperimentRow]) -> dict[str, list[tuple[ExperimentRow, float]]]:
    metric_map: dict[str, list[tuple[ExperimentRow, float]]] = {}
    for row in rows:
        for key, value in flatten_mapping(row.metrics).items():
            if is_number(value):
                metric_map.setdefault(key, []).append((row, float(value)))
    return metric_map


def pick_metric(
    rows: list[ExperimentRow],
    *,
    requested_metric: str | None,
    goal_override: str,
) -> MetricSummary:
    metric_map = collect_numeric_metrics(rows)
    if not metric_map:
        raise ValueError("No numeric metrics were found in the CSV-backed experiment history.")

    if requested_metric:
        if requested_metric not in metric_map:
            available = ", ".join(sorted(metric_map))
            raise ValueError(f"Metric '{requested_metric}' was not found. Available numeric metrics: {available}")
        goal = metric_goal(requested_metric) if goal_override == "auto" else goal_override
        return MetricSummary(
            name=requested_metric,
            goal=goal,
            coverage=len(metric_map[requested_metric]),
            values=metric_map[requested_metric],
        )

    ranked = sorted(metric_map.items(), key=lambda item: (-len(item[1]), metric_priority(item[0]), item[0]))
    metric_name, values = ranked[0]
    goal = metric_goal(metric_name) if goal_override == "auto" else goal_override
    return MetricSummary(name=metric_name, goal=goal, coverage=len(values), values=values)


def sort_metric_values(values: list[tuple[ExperimentRow, float]], *, goal: str) -> list[tuple[ExperimentRow, float]]:
    reverse = goal == "max"
    return sorted(values, key=lambda item: (item[1], item[0].id), reverse=reverse)


def format_float(value: float) -> str:
    return f"{value:.6g}"


def config_diff(base: dict[str, Any], other: dict[str, Any], *, limit: int = 8) -> list[str]:
    base_flat = flatten_mapping(base)
    other_flat = flatten_mapping(other)
    diffs: list[str] = []
    for key in sorted(set(base_flat) | set(other_flat)):
        left = base_flat.get(key, "<missing>")
        right = other_flat.get(key, "<missing>")
        if left != right:
            diffs.append(f"{key}: {left!r} -> {right!r}")
        if len(diffs) >= limit:
            break
    return diffs


def build_svg_polyline(points: list[float], *, width: int = 640, height: int = 220) -> str:
    if not points:
        return ""
    min_value = min(points)
    max_value = max(points)
    span = max(max_value - min_value, 1e-12)
    step_x = width / max(len(points) - 1, 1)
    coords: list[str] = []
    for index, value in enumerate(points):
        x = index * step_x
        normalized = (value - min_value) / span
        y = height - (normalized * (height - 20)) - 10
        coords.append(f"{x:.2f},{y:.2f}")
    polyline = " ".join(coords)
    return (
        f'<svg viewBox="0 0 {width} {height}" class="trend-chart" role="img" '
        f'aria-label="Metric trend chart">'
        f'<polyline fill="none" stroke="currentColor" stroke-width="3" points="{polyline}" />'
        "</svg>"
    )


def table_lines(headers: list[str], rows: list[list[str]]) -> list[str]:
    header_line = "| " + " | ".join(headers) + " |"
    separator_line = "| " + " | ".join("---" for _ in headers) + " |"
    body = ["| " + " | ".join(row) + " |" for row in rows]
    return [header_line, separator_line, *body]
