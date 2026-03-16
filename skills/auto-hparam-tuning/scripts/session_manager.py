from __future__ import annotations

import argparse
import csv
import json
import shlex
import subprocess
from datetime import datetime
from io import StringIO
from typing import Any, Literal
from pathlib import Path

from omegaconf import OmegaConf
import pandas as pd

from utils import (
    TargetSpec,
    Storage,
    LocalStorage,
    SSHStorage,
    default_storage as _default_storage,
    join as _join,
    get_sessions_spawn_command,
    ensure_override_in_defaults,
    load_results_dataframe,
    safe_int,
    safe_float,
    now_iso,
    upsert_results_row
)

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


class SessionManagerError(RuntimeError):
    pass


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
    df = load_results_dataframe(storage, results_path, RESULTS_COLUMNS)

    inferred_goal = goal
    meta_text = storage.read_text(meta_path) if storage.exists(meta_path) else ""
    if inferred_goal is None and meta_text:
        meta_cfg = OmegaConf.create(meta_text)
        inferred_goal = meta_cfg.get("goal") or None
    if inferred_goal is None:
        inferred_goal = "maximize"

    if df.empty:
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

    working = df.copy()
    working["run_id_int"] = working["run_id"].map(safe_int)
    working["primary_metric_float"] = working["primary_metric"].map(safe_float)
    working["best_step_int"] = working["best_step"].map(safe_int)
    working = working.sort_values(
        by="run_id_int",
        key=lambda s: s.map(lambda x: x if x is not None else 10**18),
    ).reset_index(drop=True)

    run_count = int(len(working))
    finished_df = working[working["status"] == "finished"].copy()
    failed_df = working[working["status"] == "failed"].copy()
    latest_row = working.iloc[-1]

    metric_df = working[working["primary_metric_float"].notna()].copy()
    best_row = None
    top_runs: list[dict[str, Any]] = []
    if not metric_df.empty:
        ascending = str(inferred_goal).lower() in {"min", "minimize", "lower", "low"}
        metric_df = metric_df.sort_values(by=["primary_metric_float", "run_id_int"], ascending=[ascending, True])
        best_row = metric_df.iloc[0]
        for _, row in metric_df.head(max(top_k, 0)).iterrows():
            top_runs.append({
                "run_id": row["run_id_int"],
                "status": row["status"],
                "primary_metric": row["primary_metric_float"],
                "best_step": row["best_step_int"],
                "run_dir": row["run_dir"],
                "notes": row["notes"],
            })

    recent_runs: list[dict[str, Any]] = []
    for _, row in working.tail(max(recent_k, 0)).iterrows():
        recent_runs.append({
            "run_id": row["run_id_int"],
            "status": row["status"],
            "primary_metric": row["primary_metric_float"],
            "best_step": row["best_step_int"],
            "run_dir": row["run_dir"],
            "notes": row["notes"],
        })

    trend_hint = "insufficient_metric_history"
    recent_metric_df = working[working["primary_metric_float"].notna()].tail(3).copy()
    if len(recent_metric_df) >= 2:
        values = recent_metric_df["primary_metric_float"].tolist()
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
        "finished_run_count": int(len(finished_df)),
        "failed_run_count": int(len(failed_df)),
        "latest_run_id": safe_int(latest_row["run_id"]),
        "latest_status": latest_row["status"],
        "latest_primary_metric": safe_float(latest_row["primary_metric"]),
        "best_run_id": None if best_row is None else best_row["run_id_int"],
        "best_primary_metric": None if best_row is None else best_row["primary_metric_float"],
        "best_run_status": None if best_row is None else best_row["status"],
        "top_runs": top_runs,
        "recent_runs": recent_runs,
        "trend_hint": trend_hint,
        "next_step":
            "The AHT loop just iterated once. Start another run with `python scripts/session_manager.py " + 
            f"--ssh-host \"{ssh_host}\" " if ssh_host is not None else "" +
            f"create-run {session_dir}` " +
            "or just stop at here if you believe the tuning process is completed."
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
    primary_config_path: str | None = None,
) -> dict[str, Any]:
    target = TargetSpec(project_root=project_root, ssh_host=ssh_host)
    storage = _default_storage(target)

    # Derive the override.yaml path that sits next to the primary Hydra config.
    override_yaml_path: str | None = None
    config_modified = False
    if primary_config_path is not None:
        parent = primary_config_path.rsplit("/", 1)[0] if "/" in primary_config_path else "."
        override_yaml_path = parent + "/override.yaml"
        config_modified = ensure_override_in_defaults(storage, primary_config_path)

    now = datetime.now().astimezone() if timestamp is None else datetime.fromisoformat(timestamp)
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H-%M-%S")
    session_dir = _join(project_root, "aht", date_str, time_str)
    runs_dir = _join(session_dir, "runs")

    storage.mkdir(runs_dir)
    storage.write_text(_join(session_dir, "results.csv"), ",".join(RESULTS_COLUMNS) + "\n")
    storage.write_text(_join(session_dir, "strategy.md"), "")
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
        "primary_config_path": primary_config_path,
        "override_yaml_path": override_yaml_path,
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
        "primary_config_path": primary_config_path,
        "override_yaml_path": override_yaml_path,
        "primary_config_defaults_modified": config_modified,
        "next_step":
            "Run `python scripts/session_manager.py " +
            f"--ssh-host \"{ssh_host}\" " if ssh_host is not None else "" +
            "append-report /path/to/project \"your understandings to the task\"` to write down your understandings to the task. "
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
        "next_step":
            f"Read and follow the instructions in {Path(__file__).parent.parent.joinpath("prompts", "plan_tuning_strategy.md")}." +
            "Then, specify a hparam combination with `python scripts/session_manager.py[ --ssh-host \"remotehost\"] write-override " +
            f"{session_dir} --run-id {next_id} " +
            "--override param0=value0 moduleA.param1=value1 moduleB.submodule0=value2 moduleC/submodule1=subconfig0"
    }


def write_override(
    session_dir: str,
    run_id: int,
    overrides: list[str] | None = None,
    yaml_content: str | None = None,
    ssh_host: str | None = None,
) -> dict[str, Any]:
    """Write Hydra overrides to ``<run_dir>/override.yaml``.

    Accepts either a list of dotted-path ``key=value`` strings (Hydra CLI style)
    or a raw YAML string.  Exactly one of *overrides* or *yaml_content* must be
    provided.

    Args:
        session_dir: AHT session directory.
        run_id: Integer run ID returned by ``create_run``.
        overrides: Hydra-style override list, e.g.
            ``["model.hidden_size=256", "optimizer.lr=0.001"]``.
            Parsed with ``OmegaConf.from_dotlist`` and serialized to YAML.
        yaml_content: Raw YAML string written verbatim to override.yaml.
        ssh_host: SSH host string. ``None`` means local.

    Returns:
        Dict with ``override_path`` and the written ``yaml_content``.
    """
    if (overrides is None) == (yaml_content is None):
        raise SessionManagerError("Exactly one of 'overrides' or 'yaml_content' must be provided.")

    if overrides is not None:
        cfg = OmegaConf.from_dotlist(overrides)
        text = OmegaConf.to_yaml(cfg)
    else:
        text = yaml_content if yaml_content.endswith("\n") else yaml_content + "\n"

    run_dir = _join(session_dir, "runs", str(run_id))
    override_path = _join(run_dir, "override.yaml")
    storage = _default_storage(TargetSpec(project_root=session_dir, ssh_host=ssh_host))
    storage.write_text(override_path, text)

    # Also copy to the override.yaml that lives next to the primary Hydra config,
    # whose path is recorded in meta.yaml by create_session().
    meta_path = _join(session_dir, "meta.yaml")
    override_yaml_path: str | None = None
    sync_warning: str | None = None
    try:
        if not storage.exists(meta_path):
            sync_warning = f"meta.yaml not found at {meta_path}; skipping config-dir sync."
        else:
            try:
                meta_cfg = OmegaConf.create(storage.read_text(meta_path))
            except Exception as exc:
                sync_warning = f"Failed to parse meta.yaml: {exc}; skipping config-dir sync."
                meta_cfg = None
            if meta_cfg is not None:
                override_yaml_path = meta_cfg.get("override_yaml_path") or None
                if not override_yaml_path:
                    sync_warning = "meta.yaml has no override_yaml_path (was create-session called with --primary-config-path?)."
                else:
                    try:
                        storage.write_text(override_yaml_path, text)
                    except Exception as exc:
                        sync_warning = f"Wrote run override but failed to sync to {override_yaml_path}: {exc}"
                        override_yaml_path = None
    except Exception as exc:
        sync_warning = f"Unexpected error during config-dir sync: {exc}"

    return {
        "run_id": run_id,
        "override_path": override_path,
        "override_yaml_path": override_yaml_path,
        "sync_warning": sync_warning,
        "yaml_content": text,
        "next_step":
            "Start a run with `python scripts/session_manager.py " +
            (f"--ssh-host \"{ssh_host}\" " if ssh_host is not None else "") +
            f"run-command {session_dir} --run-id {run_id} " +
            "--conda-env \"specified_conda_env\" --command-str \"specified_command\""
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
    normalized_row = upsert_results_row(storage, results_path, RESULTS_COLUMNS, row)
    return {
        "results_csv": results_path,
        "row": normalized_row,
        "next_step": 
            "Run `python scripts/session_manager.py " +
            f"--ssh-host \"{ssh_host}\" " if ssh_host is not None else "" +
            "append-report /path/to/project \"summary of this run\"`."
    }


def _tmux_session_name(session_dir: str, run_id: int) -> str:
    import hashlib
    h = hashlib.md5(session_dir.encode()).hexdigest()[:8]
    return f"aht-{h}-{run_id}"


def run_command(
    session_dir: str,
    run_id: int,
    command: str,
    conda_env: str | None = None,
    cwd: str | None = None,
    ssh_host: str | None = None,
    use_tmux: bool = True,
) -> dict[str, Any]:
    """Execute *command* for a run, redirecting stdout/stderr to the run's log files.

    When *use_tmux* is True (default), the command is launched in a detached
    tmux session and this function returns immediately with status ``"running"``.
    Call ``poll_run`` to check completion and update results.csv.

    When *use_tmux* is False, execution is synchronous (blocks until done).

    Args:
        session_dir: AHT session directory (absolute path on the target host).
        run_id: Integer run ID returned by ``create_run``.
        command: Shell command to execute.
        conda_env: Conda environment to activate via
            ``bash -i -c "conda activate <env> && <cmd>"``.
        cwd: Working directory on the target host. Defaults to the project root
            inferred from session_dir (three levels up).
        ssh_host: SSH host string. ``None`` means local execution.
        use_tmux: Launch asynchronously via a detached tmux session (default).
            Set to False for synchronous execution without tmux.

    Returns:
        Dict with run metadata. When async, ``"status"`` is ``"running"`` and
        ``"tmux_session"`` holds the session name for use with ``poll_run``.
        When sync, ``"returncode"`` and final ``"status"`` are included.
    """
    run_dir = _join(session_dir, "runs", str(run_id))
    stdout_path = _join(run_dir, "stdout.log")
    stderr_path = _join(run_dir, "stderr.log")
    effective_cwd = cwd or Path(session_dir).parent.parent.parent.resolve().as_posix()

    if conda_env:
        inner = "conda activate " + shlex.quote(conda_env) + " && " + command
        shell_cmd = "bash -i -c " + shlex.quote(inner)
    else:
        shell_cmd = command

    start_time = now_iso()
    update_run_result(session_dir, run_id, status="running", start_time=start_time, ssh_host=ssh_host)

    if use_tmux:
        tmux_name = _tmux_session_name(session_dir, run_id)
        returncode_path = _join(run_dir, "returncode.txt")
        full_cmd = (
            "cd " + shlex.quote(effective_cwd)
            + " && " + shell_cmd
            + " >" + shlex.quote(stdout_path)
            + " 2>" + shlex.quote(stderr_path)
            + "; echo $? >" + shlex.quote(returncode_path)
        )
        tmux_argv = ["tmux", "new-session", "-d", "-s", tmux_name, full_cmd]

        if ssh_host is None:
            proc = subprocess.run(tmux_argv, capture_output=True, text=True)
        else:
            print(" ".join(["ssh", ssh_host, " ".join(shlex.quote(a) for a in tmux_argv)]))
            proc = subprocess.run(
                ["ssh", ssh_host, " ".join(shlex.quote(a) for a in tmux_argv)],
                capture_output=True,
                text=True,
            )

        if proc.returncode != 0:
            update_run_result(session_dir, run_id, status="failed", end_time=now_iso(), ssh_host=ssh_host)
            raise SessionManagerError(f"tmux launch failed: {proc.stderr.strip() or proc.stdout.strip()}")

        storage = _default_storage(TargetSpec(project_root=session_dir, ssh_host=ssh_host))
        storage.write_text(_join(run_dir, "tmux_session.txt"), tmux_name)

        return {
            "run_id": run_id,
            "run_dir": run_dir,
            "stdout_path": stdout_path,
            "stderr_path": stderr_path,
            "tmux_session": tmux_name,
            "status": "running",
            "start_time": start_time,
            "cwd": effective_cwd,
            "next_step":
                "Query the run with " + 
                "`python scripts/session_manager.py " +
                f"--ssh-host \"{ssh_host}\" " if ssh_host is not None else "" +
                f"poll-run {session_dir} --run-id {run_id}`"
        }

    # ── Synchronous fallback ─────────────────────────────────────────────────
    try:
        if ssh_host is None:
            with open(stdout_path, "w", encoding="utf-8") as out_f, \
                    open(stderr_path, "w", encoding="utf-8") as err_f:
                proc = subprocess.run(
                    shell_cmd, shell=True, cwd=effective_cwd,
                    stdout=out_f, stderr=err_f,
                )
            returncode = proc.returncode
        else:
            remote_cmd = (
                "cd " + shlex.quote(effective_cwd)
                + " && " + shell_cmd
                + " >" + shlex.quote(stdout_path)
                + " 2>" + shlex.quote(stderr_path)
            )
            print(" ".join(["ssh", ssh_host, f"\"{remote_cmd}\""]))
            proc = subprocess.run(["ssh", ssh_host, f"\"{remote_cmd}\""], capture_output=True, text=True)
            returncode = proc.returncode
    except Exception as exc:
        end_time = now_iso()
        update_run_result(session_dir, run_id, status="failed", end_time=end_time, notes=str(exc), ssh_host=ssh_host)
        raise

    end_time = now_iso()
    status: RunStatus = "finished" if returncode == 0 else "failed"
    update_run_result(session_dir, run_id, status=status, end_time=end_time, ssh_host=ssh_host)
    
    next_step = []
    if status == "finished":
        if ssh_host is not None:
            next_step.append("Copy the event file and the full config file back to local disk.")
        next_step.append("Run `python scripts/analyze_event.py list-keys /path/to/the/local/event/file` to list scalar keys.")
        next_step.append("Then run `python scripts/analyze_event.py summarize /path/to/the/local/event/file key1 key2 keyn` to analyze the selected scalar events.")
    else:
        next_step.append("Run failed. Now try to figure out what's wrong according to the stdout.")

    return {
        "run_id": run_id,
        "run_dir": run_dir,
        "stdout_path": stdout_path,
        "stderr_path": stderr_path,
        "returncode": returncode,
        "status": status,
        "start_time": start_time,
        "end_time": end_time,
        "cwd": effective_cwd,
        "next_step": next_step
    }


def poll_run(
    session_dir: str,
    run_id: int,
    ssh_host: str | None = None,
    tail_lines: int = 50,
) -> dict[str, Any]:
    """Check whether a tmux-launched run has finished and update results.csv.

    Reads the tmux session name from ``<run_dir>/tmux_session.txt`` and queries
    ``tmux has-session``.  If the session is gone, reads the exit code from
    ``<run_dir>/returncode.txt``, updates results.csv, and returns the final status.

    Args:
        session_dir: AHT session directory.
        run_id: Integer run ID.
        ssh_host: SSH host string. ``None`` means local.
        tail_lines: Number of trailing lines from stdout.log to include in the
            response. Set to 0 to omit. Defaults to 50.

    Returns:
        Dict with ``run_id``, ``status``, ``returncode`` (None if still running),
        ``tmux_session``, and ``stdout_tail``.
    """
    run_dir = _join(session_dir, "runs", str(run_id))
    tmux_session_file = _join(run_dir, "tmux_session.txt")
    returncode_path = _join(run_dir, "returncode.txt")

    storage = _default_storage(TargetSpec(project_root=session_dir, ssh_host=ssh_host))
    if not storage.exists(tmux_session_file):
        raise SessionManagerError(
            f"No tmux_session.txt found for run {run_id}. "
            "Was run-command called with tmux enabled?"
        )

    tmux_name = storage.read_text(tmux_session_file).strip()
    check_argv = ["tmux", "has-session", "-t", tmux_name]

    if ssh_host is None:
        check = subprocess.run(check_argv, capture_output=True)
    else:
        check = subprocess.run(
            ["ssh", ssh_host, " ".join(shlex.quote(a) for a in check_argv)],
            capture_output=True,
        )

    stdout_path = _join(run_dir, "stdout.log")

    def _read_tail(path: str) -> str:
        if not storage.exists(path):
            return ""
        text = storage.read_text(path)
        if not tail_lines:
            return text
        lines = text.splitlines()
        return "\n".join(lines[-tail_lines:]) if len(lines) > tail_lines else text

    if check.returncode == 0:
        return {
            "run_id": run_id,
            "run_dir": run_dir,
            "tmux_session": tmux_name,
            "status": "running",
            "returncode": None,
            "stdout_tail": _read_tail(stdout_path),
            "next_step": "Extract the remaining time from the stdout, wait, and poll again."
        }

    # Session is gone → read exit code and finalize
    returncode: int | None = None
    if storage.exists(returncode_path):
        try:
            returncode = int(storage.read_text(returncode_path).strip())
        except ValueError:
            pass

    status: RunStatus = "finished" if returncode == 0 else "failed"
    end_time = now_iso()
    update_run_result(session_dir, run_id, status=status, end_time=end_time, ssh_host=ssh_host)
    
    next_step = []
    if status == "finished":
        if ssh_host is not None:
            next_step.append("Copy the event file and the full config file back to local disk.")
        next_step.append("Run `python scripts/analyze_event.py list-keys /path/to/the/local/event/file` to list scalar keys.")
        next_step.append("Then run `python scripts/analyze_event.py summarize /path/to/the/local/event/file key1 key2 keyn` to analyze the selected scalar events.")
    else:
        next_step.append("Run failed. Now try to figure out what's wrong according to the stdout.")
        

    return {
        "run_id": run_id,
        "run_dir": run_dir,
        "tmux_session": tmux_name,
        "status": status,
        "returncode": returncode,
        "end_time": end_time,
        "stdout_tail": _read_tail(stdout_path),
        "next_step": next_step
    }


def append_report(session_dir: str, content: str, ssh_host: str | None = None) -> dict[str, Any]:
    storage = _default_storage(TargetSpec(project_root=session_dir, ssh_host=ssh_host))
    report_path = _join(session_dir, "report.md")
    text = content if content.endswith("\n") else content + "\n"
    storage.append_text(report_path, text)
    if len(storage.list_dir_names(_join(session_dir, "runs"))) == 0:
        next_step = [
            "Now the session is initialized. Start the test run with `python scripts/session_manager.py " + 
            f"--ssh-host \"{ssh_host}\" " if ssh_host is not None else "" +
            f"create-run {session_dir}`"
        ]
    else:
        next_step = [
            "Run `python scripts/session_manager.py " +
            f"--ssh-host \"{ssh_host}\" " if ssh_host is not None else "" +
            f"summarize-results {session_dir}` to analyze the finished runs. "
        ]
    return {
        "report_md": report_path,
        "appended_chars": len(text),
        "next_step": next_step
    }


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
    cfg.ended_at = ended_at or now_iso()
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
    p_create.add_argument(
        "--primary-config-path",
        default=None,
        metavar="PATH",
        help=(
            "Absolute path to the primary Hydra config file (e.g. conf/config.yaml). "
            "When provided: (1) '- override' is inserted into its defaults list after '- _self_' "
            "if not already present; (2) the path to the sibling override.yaml is recorded in "
            "meta.yaml and used by write-override to sync overrides automatically."
        ),
    )

    p_run = subparsers.add_parser("create-run", help="Create the next run directory inside a session.")
    p_run.add_argument("session_dir")
    p_run.add_argument("--notes", default=None)

    p_update = subparsers.add_parser("update-run", help="Insert or update one run row in results.csv.")
    p_update.add_argument("session_dir")
    p_update.add_argument("--run-id", type=int)
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

    p_exec = subparsers.add_parser(
        "run-command",
        help="Launch a shell command for a run via tmux (async) or synchronously.",
    )
    p_exec.add_argument("session_dir")
    p_exec.add_argument("--run-id", type=int, required=True, help="Run ID returned by create-run.")
    p_exec.add_argument("--command-str", metavar="command", required=True, help="Shell command to execute.")
    p_exec.add_argument(
        "--conda-env",
        default=None,
        metavar="ENV",
        help="Conda environment to activate via 'bash -i -c \"conda activate ENV && cmd\"'.",
    )
    p_exec.add_argument(
        "--cwd",
        default=None,
        metavar="DIR",
        help="Working directory on the target host. Defaults to the project root.",
    )
    p_exec.add_argument(
        "--no-tmux",
        action="store_true",
        default=False,
        help="Disable tmux and run synchronously (blocks until the command finishes).",
    )

    p_override = subparsers.add_parser(
        "write-override",
        help="Write Hydra overrides to the override.yaml of a run.",
    )
    p_override.add_argument("session_dir")
    p_override.add_argument("--run-id", type=int, required=True, help="Run ID returned by create-run.")
    group = p_override.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--override",
        dest="overrides",
        action="append",
        metavar="KEY=VALUE",
        help=(
            "Hydra-style dotted-path override, e.g. --override model.hidden_size=256. "
            "Repeat to set multiple overrides. Converted to YAML via OmegaConf."
        ),
    )
    group.add_argument(
        "--yaml",
        dest="yaml_content",
        default=None,
        metavar="YAML",
        help="Raw YAML string written verbatim to override.yaml.",
    )

    p_poll = subparsers.add_parser(
        "poll-run",
        help="Check if a tmux-launched run has finished and update results.csv.",
    )
    p_poll.add_argument("session_dir")
    p_poll.add_argument("--run-id", type=int, required=True, help="Run ID to poll.")
    p_poll.add_argument(
        "--tail",
        type=int,
        default=50,
        metavar="N",
        help="Include the last N lines of stdout.log in the response. 0 = full log. (default: 50)",
    )

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
            primary_config_path=args.primary_config_path,
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
    elif args.command == "write-override":
        result = write_override(
            session_dir=args.session_dir,
            run_id=args.run_id,
            overrides=args.overrides,
            yaml_content=args.yaml_content,
            ssh_host=args.ssh_host,
        )
    elif args.command == "run-command":
        result = run_command(
            session_dir=args.session_dir,
            run_id=args.run_id,
            command=args.command_str,
            conda_env=args.conda_env,
            cwd=args.cwd,
            ssh_host=args.ssh_host,
            use_tmux=not args.no_tmux,
        )
    elif args.command == "poll-run":
        result = poll_run(
            session_dir=args.session_dir,
            run_id=args.run_id,
            ssh_host=args.ssh_host,
            tail_lines=args.tail,
        )
    else:
        raise SessionManagerError(f"Unsupported command: {args.command}")

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
