from __future__ import annotations

import argparse
import csv
import json
import subprocess
from dataclasses import dataclass
from datetime import datetime
from io import StringIO
from pathlib import PurePosixPath
from typing import Any, Literal

from omegaconf import OmegaConf

SessionStatus = Literal["running", "completed", "stopped", "failed"]
RunStatus = Literal["created", "running", "finished", "failed", "killed", "inconclusive"]

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


REMOTE_HELPER = r'''
import json
import sys
from pathlib import Path


def load_payload() -> dict:
    return json.load(sys.stdin)


def dump(value):
    sys.stdout.write(json.dumps(value, ensure_ascii=False))


payload = load_payload()
action = payload["action"]

if action == "mkdir":
    path = Path(payload["path"])
    path.mkdir(parents=True, exist_ok=True)
    dump({"ok": True})
elif action == "write_text":
    path = Path(payload["path"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload["text"], encoding="utf-8")
    dump({"ok": True})
elif action == "append_text":
    path = Path(payload["path"])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(payload["text"])
    dump({"ok": True})
elif action == "read_text":
    path = Path(payload["path"])
    dump({"text": path.read_text(encoding="utf-8")})
elif action == "exists":
    path = Path(payload["path"])
    dump({"exists": path.exists()})
elif action == "list_dir_names":
    path = Path(payload["path"])
    if not path.exists():
        dump({"names": []})
    else:
        names = sorted([p.name for p in path.iterdir()])
        dump({"names": names})
else:
    raise ValueError(f"Unsupported remote action: {action}")
'''


class SessionManagerError(RuntimeError):
    pass


@dataclass(frozen=True)
class TargetSpec:
    project_root: str
    ssh_host: str | None = None

    @property
    def is_remote(self) -> bool:
        return self.ssh_host is not None

    @property
    def label(self) -> str:
        return f"{self.ssh_host}:{self.project_root}" if self.is_remote else self.project_root


class Storage:
    def mkdir(self, path: str) -> None:
        raise NotImplementedError

    def write_text(self, path: str, text: str) -> None:
        raise NotImplementedError

    def append_text(self, path: str, text: str) -> None:
        raise NotImplementedError

    def read_text(self, path: str) -> str:
        raise NotImplementedError

    def exists(self, path: str) -> bool:
        raise NotImplementedError

    def list_dir_names(self, path: str) -> list[str]:
        raise NotImplementedError


class LocalStorage(Storage):
    def mkdir(self, path: str) -> None:
        from pathlib import Path

        Path(path).mkdir(parents=True, exist_ok=True)

    def write_text(self, path: str, text: str) -> None:
        from pathlib import Path

        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")

    def append_text(self, path: str, text: str) -> None:
        from pathlib import Path

        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a", encoding="utf-8") as f:
            f.write(text)

    def read_text(self, path: str) -> str:
        from pathlib import Path

        return Path(path).read_text(encoding="utf-8")

    def exists(self, path: str) -> bool:
        from pathlib import Path

        return Path(path).exists()

    def list_dir_names(self, path: str) -> list[str]:
        from pathlib import Path

        p = Path(path)
        if not p.exists():
            return []
        return sorted(child.name for child in p.iterdir())


class SSHStorage(Storage):
    def __init__(self, host: str):
        self.host = host

    def _call(self, payload: dict[str, Any]) -> dict[str, Any]:
        command = ["ssh", self.host, "python3", "-c", REMOTE_HELPER]
        proc = subprocess.run(
            command,
            input=json.dumps(payload, ensure_ascii=False),
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            raise SessionManagerError(
                f"Remote command failed on {self.host}: {proc.stderr.strip() or proc.stdout.strip()}"
            )
        output = proc.stdout.strip()
        if not output:
            return {}
        try:
            return json.loads(output)
        except json.JSONDecodeError as exc:
            raise SessionManagerError(f"Failed to decode remote JSON response: {output!r}") from exc

    def mkdir(self, path: str) -> None:
        self._call({"action": "mkdir", "path": path})

    def write_text(self, path: str, text: str) -> None:
        self._call({"action": "write_text", "path": path, "text": text})

    def append_text(self, path: str, text: str) -> None:
        self._call({"action": "append_text", "path": path, "text": text})

    def read_text(self, path: str) -> str:
        return self._call({"action": "read_text", "path": path})["text"]

    def exists(self, path: str) -> bool:
        return bool(self._call({"action": "exists", "path": path})["exists"])

    def list_dir_names(self, path: str) -> list[str]:
        return list(self._call({"action": "list_dir_names", "path": path})["names"])

def _stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)

def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")

def _default_storage(target: TargetSpec) -> Storage:
    return SSHStorage(target.ssh_host) if target.is_remote else LocalStorage()


def _load_results_rows(storage: Storage, results_path: str, columns: list[str]) -> list[dict[str, str]]:
    if not storage.exists(results_path):
        return []
    text = storage.read_text(results_path)
    if not text.strip():
        return []

    reader = csv.DictReader(StringIO(text))
    rows: list[dict[str, str]] = []
    for raw_row in reader:
        rows.append({col: _stringify(raw_row.get(col, "")) for col in columns})
    return rows


def _write_results_rows(storage: Storage, results_path: str, rows: list[dict[str, str]], columns: list[str]) -> None:
    normalized = [{col: _stringify(row.get(col, "")) for col in columns} for row in rows]
    normalized.sort(
        key=lambda row: _safe_int(row.get("run_id")) if _safe_int(row.get("run_id")) is not None else 10**18
    )
    buffer = StringIO()
    writer = csv.DictWriter(buffer, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    writer.writerows(normalized)
    storage.write_text(results_path, buffer.getvalue())

        key=lambda row: _safe_int(row.get("run_id")) if _safe_int(row.get("run_id")) is not None else 10**18
    )
    buffer = StringIO()
    writer = csv.DictWriter(buffer, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    writer.writerows(normalized)
    storage.write_text(results_path, buffer.getvalue())

def _load_results_dataframe(storage: Storage, results_path: str, columns: list[str]) -> pd.DataFrame:
    if storage.exists(results_path):
        text = storage.read_text(results_path)
        if text.strip():
            df = pd.read_csv(StringIO(text), dtype=str, keep_default_na=False)
        else:
            df = pd.DataFrame(columns=columns)
    else:
        df = pd.DataFrame(columns=columns)

    for col in columns:
        if col not in df.columns:
            df[col] = ""
    return df[columns].fillna("")

def _write_results_dataframe(storage: Storage, results_path: str, df: pd.DataFrame, columns: list[str]) -> None:
    normalized = df.copy()
    for col in columns:
        if col not in normalized.columns:
            normalized[col] = ""
    normalized = normalized[columns].fillna("")
    if not normalized.empty:
        normalized["run_id"] = normalized["run_id"].astype(str)
        normalized = normalized.sort_values(
            by="run_id",
            key=lambda s: s.map(lambda x: int(str(x)) if str(x).strip() else 10**18),
        )
    buffer = StringIO()
    writer = csv.DictWriter(buffer, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    writer.writerows(normalized)
    storage.write_text(results_path, buffer.getvalue())


def _upsert_results_row(storage: Storage, results_path: str, columns: list[str], row: dict[str, Any]) -> dict[str, str]:
    normalized_row = {
        col: (None if col not in row or row.get(col) is None else _stringify(row.get(col)))
        for col in columns
    }
    rows = _load_results_rows(storage, results_path, columns)
    key = _stringify(row.get("run_id", ""))
    existing = next((current for current in rows if _stringify(current.get("run_id")) == key), None)
    if existing is None:
        inserted = {col: ("" if normalized_row[col] is None else normalized_row[col]) for col in columns}
        rows.append(inserted)
        merged = inserted
    else:
        merged = dict(existing)
        for col in columns:
            if normalized_row[col] is not None:
                merged[col] = normalized_row[col]
        for index, current in enumerate(rows):
            if _stringify(current.get("run_id")) == key:
                rows[index] = merged
                break
    _write_results_rows(storage, results_path, rows, columns)
    if merged is not None:
        return {col: _stringify(merged.get(col, "")) for col in columns}
    if key:
        for current in rows:
            if _stringify(current.get("run_id")) == key:
                return {col: _stringify(current.get(col, "")) for col in columns}
    return {col: ("" if normalized_row[col] is None else normalized_row[col]) for col in columns}


def _join(root: str, *parts: str) -> str:
    return str(PurePosixPath(root, *parts))


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


def summarize_results(
    session_dir: str,
    ssh_host: str | None = None,
    top_k: int = 3,
    recent_k: int = 5,
    goal: str | None = None,
) -> dict[str, Any]:
    storage = _default_storage(TargetSpec(project_root=session_dir, ssh_host=ssh_host))
    results_path = _join(session_dir, "results.csv")
    meta_path = _join(session_dir, "meta.yaml")
    rows = _load_results_rows(storage, results_path, RESULTS_COLUMNS)

    inferred_goal = goal
    meta_text = storage.read_text(meta_path) if storage.exists(meta_path) else ""
    if inferred_goal is None and meta_text:
        meta_cfg = OmegaConf.create(meta_text)
        inferred_goal = meta_cfg.get("goal") or None
    if inferred_goal is None:
        inferred_goal = "maximize"

    if not rows:
        return {
            "session_dir": session_dir,
            "results_csv": results_path,
            "goal": inferred_goal,
            "run_count": 0,
            "finished_run_count": 0,
            "failed_run_count": 0,
            "latest_run_id": None,
            "best_run_id": None,
            "best_primary_metric": None,
            "top_runs": [],
            "recent_runs": [],
            "trend_hint": "no_runs",
        }

    working = []
    for row in rows:
        enriched = dict(row)
        enriched["run_id_int"] = _safe_int(row.get("run_id"))
        enriched["primary_metric_float"] = _safe_float(row.get("primary_metric"))
        enriched["best_step_int"] = _safe_int(row.get("best_step"))
        working.append(enriched)
    working.sort(key=lambda row: row["run_id_int"] if row["run_id_int"] is not None else 10**18)

    run_count = len(working)
    finished_rows = [row for row in working if row["status"] == "finished"]
    failed_rows = [row for row in working if row["status"] == "failed"]
    latest_row = working[-1]

    metric_rows = [row for row in working if row["primary_metric_float"] is not None]
    best_row: dict[str, Any] | None = None
    top_runs: list[dict[str, Any]] = []
    if metric_rows:
        ascending = str(inferred_goal).lower() in {"min", "minimize", "lower", "low"}
        metric_rows = sorted(
            metric_rows,
            key=lambda row: (
                row["primary_metric_float"],
                row["run_id_int"] if row["run_id_int"] is not None else 10**18,
            ),
            reverse=not ascending,
        )
        best_row = metric_rows[0]
        for row in metric_rows[: max(top_k, 0)]:
            top_runs.append({
                "run_id": row["run_id_int"],
                "status": row["status"],
                "primary_metric": row["primary_metric_float"],
                "best_step": row["best_step_int"],
                "run_dir": row["run_dir"],
                "notes": row["notes"],
            })

    recent_runs: list[dict[str, Any]] = []
    for row in working[-max(recent_k, 0) :]:
        recent_runs.append({
            "run_id": row["run_id_int"],
            "status": row["status"],
            "primary_metric": row["primary_metric_float"],
            "best_step": row["best_step_int"],
            "run_dir": row["run_dir"],
            "notes": row["notes"],
        })

    trend_hint = "insufficient_metric_history"
    # Use the most recent metric-bearing runs in run-id order to compute trend,
    # rather than the metric-sorted `metric_rows`.
    recent_metric_rows: list[dict[str, Any]] = []
    for row in reversed(working):
        if row["primary_metric_float"] is None:
            continue
        recent_metric_rows.append(row)
        if len(recent_metric_rows) >= 3:
            break
    recent_metric_rows = list(reversed(recent_metric_rows))
    if len(recent_metric_rows) >= 2:
        values = [row["primary_metric_float"] for row in recent_metric_rows]
        deltas = [b - a for a, b in zip(values[:-1], values[1:])]
        if str(inferred_goal).lower() in {"min", "minimize", "lower", "low"}:
            deltas = [-x for x in deltas]
        pos = sum(delta > 0 for delta in deltas)
        neg = sum(delta < 0 for delta in deltas)
        if pos and not neg:
            trend_hint = "improving"
        elif neg and not pos:
            trend_hint = "degrading"
        elif not pos and not neg:
            trend_hint = "flat"
        else:
            trend_hint = "mixed"

    return {
        "session_dir": session_dir,
        "results_csv": results_path,
        "goal": inferred_goal,
        "run_count": run_count,
        "finished_run_count": len(finished_rows),
        "failed_run_count": len(failed_rows),
        "latest_run_id": _safe_int(latest_row["run_id"]),
        "latest_status": latest_row["status"],
        "latest_primary_metric": _safe_float(latest_row["primary_metric"]),
        "best_run_id": None if best_row is None else best_row["run_id_int"],
        "best_primary_metric": None if best_row is None else best_row["primary_metric_float"],
        "best_run_status": None if best_row is None else best_row["status"],
        "top_runs": top_runs,
        "recent_runs": recent_runs,
        "trend_hint": trend_hint,
    }


def create_session(
    project_root: str,
    base_command: str | None = None,
    primary_metric: str | None = None,
    goal: str | None = None,
    agent: str = "openclaw",
    skill: str = "auto-hparam-tuning",
    notes: str | None = None,
    ssh_host: str | None = None,
    timestamp: str | None = None,
) -> dict[str, Any]:
    target = TargetSpec(project_root=project_root, ssh_host=ssh_host)
    storage = _default_storage(target)

    now = datetime.now().astimezone() if timestamp is None else datetime.fromisoformat(timestamp)
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H-%M-%S")
    session_dir = _join(project_root, "aht", date_str, time_str)
    runs_dir = _join(session_dir, "runs")

    storage.mkdir(runs_dir)
    storage.write_text(_join(session_dir, "results.csv"), ",".join(RESULTS_COLUMNS) + "\n")
    storage.write_text(
        _join(session_dir, "report.md"),
        "# AHT Report\n\n## Session Summary\n"
        f"- Project: `{project_root}`\n"
        f"- Created at: `{now.isoformat(timespec='seconds')}`\n"
        f"- Base command: `{base_command or ''}`\n"
        f"- Primary metric: `{primary_metric or ''}`\n"
        f"- Goal: `{goal or ''}`\n\n",
    )
    meta = {
        "project_root": project_root,
        "session_dir": session_dir,
        "created_at": now.isoformat(timespec="seconds"),
        "agent": agent,
        "skill": skill,
        "base_command": base_command,
        "primary_metric": primary_metric,
        "goal": goal,
        "status": "running",
        "notes": notes,
        "storage": "ssh" if ssh_host else "local",
        "ssh_host": ssh_host,
    }
    storage.write_text(_join(session_dir, "meta.yaml"), OmegaConf.to_yaml(OmegaConf.create(meta)))

    return {
        "project_root": project_root,
        "session_dir": session_dir,
        "runs_dir": runs_dir,
        "results_csv": _join(session_dir, "results.csv"),
        "report_md": _join(session_dir, "report.md"),
        "meta_yaml": _join(session_dir, "meta.yaml"),
        "storage": meta["storage"],
        "ssh_host": ssh_host,
    }


def create_run(session_dir: str, ssh_host: str | None = None, notes: str | None = None) -> dict[str, Any]:
    storage = _default_storage(TargetSpec(project_root=session_dir, ssh_host=ssh_host))
    runs_dir = _join(session_dir, "runs")
    existing = [name for name in storage.list_dir_names(runs_dir) if name.isdigit()]
    next_id = 0 if not existing else min(set(range(len(existing) + 1)) - {int(x) for x in existing})
    run_dir = _join(runs_dir, str(next_id))
    copied_dir = _join(run_dir, "copied")
    storage.mkdir(copied_dir)
    storage.write_text(_join(run_dir, "override.yaml"), "# Fill in the run-specific Hydra override here.\n")
    storage.write_text(_join(run_dir, "command.sh"), "#!/usr/bin/env bash\nset -euo pipefail\n\n# Fill in the exact training command here.\n")
    storage.write_text(_join(run_dir, "stdout.log"), "")
    storage.write_text(_join(run_dir, "stderr.log"), "")
    storage.write_text(_join(run_dir, "metrics.json"), "{}\n")
    storage.write_text(
        _join(run_dir, "summary.md"),
        f"# Run {next_id}\n\n"
        "## Objective\n\n"
        "## Override\n\n"
        "## Result\n\n"
        "## Takeaway\n\n",
    )
    update_run_result(
        session_dir=session_dir,
        run_id=next_id,
        run_name=f"run-{next_id:04d}",
        status="created",
        run_dir=f"runs/{next_id}",
        override_path=f"runs/{next_id}/override.yaml",
        config_path=f"runs/{next_id}/resolved_config.json",
        metrics_path=f"runs/{next_id}/metrics.json",
        summary_path=f"runs/{next_id}/summary.md",
        start_time=None,
        end_time=None,
        primary_metric=None,
        best_step=None,
        notes=notes,
        ssh_host=ssh_host,
    )
    return {
        "run_id": next_id,
        "run_dir": run_dir,
        "override_path": _join(run_dir, "override.yaml"),
        "command_path": _join(run_dir, "command.sh"),
        "stdout_path": _join(run_dir, "stdout.log"),
        "stderr_path": _join(run_dir, "stderr.log"),
        "metrics_path": _join(run_dir, "metrics.json"),
        "summary_path": _join(run_dir, "summary.md"),
        "copied_dir": copied_dir,
    }


def update_run_result(
    session_dir: str,
    run_id: int,
    run_name: str | None = None,
    status: RunStatus | None = None,
    primary_metric: Any = None,
    best_step: Any = None,
    run_dir: str | None = None,
    override_path: str | None = None,
    config_path: str | None = None,
    metrics_path: str | None = None,
    summary_path: str | None = None,
    start_time: str | None = None,
    end_time: str | None = None,
    notes: str | None = None,
    ssh_host: str | None = None,
) -> dict[str, Any]:
    storage = _default_storage(TargetSpec(project_root=session_dir, ssh_host=ssh_host))
    results_path = _join(session_dir, "results.csv")
    row = {
        "run_id": run_id,
        "run_name": run_name or f"run-{run_id:04d}",
        "status": status or "",
        "primary_metric": primary_metric,
        "best_step": best_step,
        "run_dir": run_dir or f"runs/{run_id}",
        "override_path": override_path or f"runs/{run_id}/override.yaml",
        "config_path": config_path or f"runs/{run_id}/resolved_config.json",
        "metrics_path": metrics_path or f"runs/{run_id}/metrics.json",
        "summary_path": summary_path or f"runs/{run_id}/summary.md",
        "start_time": start_time,
        "end_time": end_time,
        "notes": notes,
    }
    normalized_row = _upsert_results_row(storage, results_path, RESULTS_COLUMNS, row)
    return {"results_csv": results_path, "row": normalized_row}


def write_run_payloads(
    session_dir: str,
    run_id: int,
    *,
    run_name: str | None = None,
    hydra_config: Any = None,
    metrics: Any = None,
    summary_content: str | None = None,
    ssh_host: str | None = None,
) -> dict[str, Any]:
    storage = _default_storage(TargetSpec(project_root=session_dir, ssh_host=ssh_host))
    config_relpath = f"runs/{run_id}/resolved_config.json"
    metrics_relpath = f"runs/{run_id}/metrics.json"
    summary_relpath = f"runs/{run_id}/summary.md"

    if hydra_config is not None:
        storage.write_text(
            _join(session_dir, config_relpath),
            json.dumps(hydra_config, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        )
    if metrics is not None:
        storage.write_text(
            _join(session_dir, metrics_relpath),
            json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        )
    if summary_content is not None:
        text = summary_content if summary_content.endswith("\n") else summary_content + "\n"
        storage.write_text(_join(session_dir, summary_relpath), text)

    row = update_run_result(
        session_dir=session_dir,
        run_id=run_id,
        run_name=run_name,
        config_path=config_relpath,
        metrics_path=metrics_relpath,
        summary_path=summary_relpath,
        ssh_host=ssh_host,
    )["row"]
    return {
        "run_dir": _join(session_dir, "runs", str(run_id)),
        "config_path": _join(session_dir, config_relpath),
        "metrics_path": _join(session_dir, metrics_relpath),
        "summary_path": _join(session_dir, summary_relpath),
        "row": row,
    }


def append_report(session_dir: str, content: str, ssh_host: str | None = None) -> dict[str, Any]:
    storage = _default_storage(TargetSpec(project_root=session_dir, ssh_host=ssh_host))
    report_path = _join(session_dir, "report.md")
    text = content if content.endswith("\n") else content + "\n"
    storage.append_text(report_path, text)
    return {"report_md": report_path, "appended_chars": len(text)}


def finalize_session(
    session_dir: str,
    status: SessionStatus,
    ssh_host: str | None = None,
    ended_at: str | None = None,
    notes: str | None = None,
) -> dict[str, Any]:
    storage = _default_storage(TargetSpec(project_root=session_dir, ssh_host=ssh_host))
    meta_path = _join(session_dir, "meta.yaml")
    cfg = OmegaConf.create(storage.read_text(meta_path))
    cfg.status = status
    if notes is not None:
        cfg.notes = notes
    cfg.ended_at = ended_at or _now_iso()
    storage.write_text(meta_path, OmegaConf.to_yaml(cfg))
    return {"meta_yaml": meta_path, "status": status}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create and maintain AHT tuning session directories locally or over SSH.",
    )
    parser.add_argument("--ssh-host", default=None, help="Optional SSH host. When set, operate on the remote filesystem.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_create = subparsers.add_parser("create-session", help="Create a new AHT session directory.")
    p_create.add_argument("project_root", help="Root of the target project. The session will be created under <project_root>/aht/.")
    p_create.add_argument("--base-command", default=None)
    p_create.add_argument("--primary-metric", default=None)
    p_create.add_argument("--goal", default=None)
    p_create.add_argument("--agent", default="openclaw")
    p_create.add_argument("--skill", default="auto-hparam-tuning")
    p_create.add_argument("--notes", default=None)
    p_create.add_argument("--timestamp", default=None, help="ISO timestamp override for reproducible tests.")

    p_run = subparsers.add_parser("create-run", help="Create the next run directory inside a session.")
    p_run.add_argument("session_dir")
    p_run.add_argument("--notes", default=None)

    p_update = subparsers.add_parser("update-run", help="Insert or update one run row in results.csv.")
    p_update.add_argument("session_dir")
    p_update.add_argument("run_id", type=int)
    p_update.add_argument("--run-name", default=None)
    p_update.add_argument("--status", default=None)
    p_update.add_argument("--primary-metric", default=None)
    p_update.add_argument("--best-step", default=None)
    p_update.add_argument("--run-dir", default=None)
    p_update.add_argument("--override-path", default=None)
    p_update.add_argument("--config-path", default=None)
    p_update.add_argument("--metrics-path", default=None)
    p_update.add_argument("--summary-path", default=None)
    p_update.add_argument("--start-time", default=None)
    p_update.add_argument("--end-time", default=None)
    p_update.add_argument("--notes", default=None)

    p_summarize = subparsers.add_parser("summarize-results", help="Summarize results.csv with pandas for agent decision-making.")
    p_summarize.add_argument("session_dir")
    p_summarize.add_argument("--top-k", type=int, default=3)
    p_summarize.add_argument("--recent-k", type=int, default=5)
    p_summarize.add_argument("--goal", default=None, help="Override optimization direction. Default: read from meta.yaml or assume maximize.")

    p_report = subparsers.add_parser("append-report", help="Append markdown content to report.md.")
    p_report.add_argument("session_dir")
    p_report.add_argument("content", help="Markdown text to append.")

    p_finalize = subparsers.add_parser("finalize-session", help="Mark a session as completed/stopped/failed.")
    p_finalize.add_argument("session_dir")
    p_finalize.add_argument("status", choices=["running", "completed", "stopped", "failed"])
    p_finalize.add_argument("--ended-at", default=None)
    p_finalize.add_argument("--notes", default=None)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "create-session":
        result = create_session(
            project_root=args.project_root,
            base_command=args.base_command,
            primary_metric=args.primary_metric,
            goal=args.goal,
            agent=args.agent,
            skill=args.skill,
            notes=args.notes,
            ssh_host=args.ssh_host,
            timestamp=args.timestamp,
        )
    elif args.command == "create-run":
        result = create_run(session_dir=args.session_dir, ssh_host=args.ssh_host, notes=args.notes)
    elif args.command == "update-run":
        result = update_run_result(
            session_dir=args.session_dir,
            run_id=args.run_id,
            run_name=args.run_name,
            status=args.status,
            primary_metric=args.primary_metric,
            best_step=args.best_step,
            run_dir=args.run_dir,
            override_path=args.override_path,
            config_path=args.config_path,
            metrics_path=args.metrics_path,
            summary_path=args.summary_path,
            start_time=args.start_time,
            end_time=args.end_time,
            notes=args.notes,
            ssh_host=args.ssh_host,
        )
    elif args.command == "summarize-results":
        result = summarize_results(
            session_dir=args.session_dir,
            ssh_host=args.ssh_host,
            top_k=args.top_k,
            recent_k=args.recent_k,
            goal=args.goal,
        )
    elif args.command == "append-report":
        result = append_report(session_dir=args.session_dir, content=args.content, ssh_host=args.ssh_host)
    elif args.command == "finalize-session":
        result = finalize_session(
            session_dir=args.session_dir,
            status=args.status,
            ssh_host=args.ssh_host,
            ended_at=args.ended_at,
            notes=args.notes,
        )
    else:
        raise SessionManagerError(f"Unsupported command: {args.command}")

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
