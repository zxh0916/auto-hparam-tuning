from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, Optional

from omegaconf import OmegaConf

from utils import (
    TargetSpec,
    Storage,
    default_storage as _default_storage,
    join as _join,
    ensure_override_in_defaults,
    load_results_dataframe,
    safe_int,
    safe_float,
    now_iso,
    upsert_results_row,
    next_step_postfix
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


def _json_safe_default(obj: Any) -> Any:
    if hasattr(obj, "item"):
        try:
            return obj.item()
        except Exception:
            pass
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")


class SessionManager:
    def __init__(self, session_dir: str, ssh_host: Optional[str] = None) -> None:
        self.session_dir = session_dir
        self.ssh_host = ssh_host
        self.storage: Storage = _default_storage(TargetSpec(project_root=session_dir, ssh_host=ssh_host))

    @property
    def results_path(self) -> str:
        return _join(self.session_dir, "results.csv")

    # ── Factory ───────────────────────────────────────────────────────────────

    @classmethod
    def create(
        cls,
        project_root: str,
        base_command: Optional[str] = None,
        primary_metric: Optional[str] = None,
        goal: Optional[str] = None,
        agent: str = "openclaw",
        skill: str = "auto-hparam-tuning",
        notes: Optional[str] = None,
        ssh_host: Optional[str] = None,
        timestamp: Optional[str] = None,
        primary_config_path: Optional[str] = None,
    ) -> tuple["SessionManager", dict[str, Any]]:
        """Create a new AHT session directory and return a ``(SessionManager, result)`` pair."""
        storage = _default_storage(TargetSpec(project_root=project_root, ssh_host=ssh_host))

        override_yaml_path: Optional[str] = None
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

        mgr = cls(session_dir, ssh_host)
        result = {
            "project_root": project_root,
            "session_dir": session_dir,
            "runs_dir": runs_dir,
            "results_csv": mgr.results_path,
            "report_md": _join(session_dir, "report.md"),
            "meta_yaml": _join(session_dir, "meta.yaml"),
            "storage": meta["storage"],
            "ssh_host": ssh_host,
            "primary_config_path": primary_config_path,
            "override_yaml_path": override_yaml_path,
            "primary_config_defaults_modified": config_modified,
            "next_step":
                "Run `python scripts/session_manager.py " +
                (f"--ssh-host \"{ssh_host}\" " if ssh_host is not None else "") +
                f"append-report {session_dir} \"your understandings to the task\"` " +
                "to write down your understandings to the task." +
                next_step_postfix()
        }
        return mgr, result

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _tmux_session_name(session_dir: str, run_id: int) -> str:
        h = hashlib.md5(session_dir.encode()).hexdigest()[:8]
        return f"aht-{h}-{run_id}"

    def _read_tail(self, path: str, tail_lines: int) -> str:
        if not self.storage.exists(path):
            return ""
        text = self.storage.read_text(path)
        if not tail_lines:
            return text
        lines = text.splitlines()
        return "\n".join(lines[-tail_lines:]) if len(lines) > tail_lines else text

    def _ssh_prefix(self) -> str:
        return f"--ssh-host \"{self.ssh_host}\" " if self.ssh_host is not None else ""

    # ── Core methods ──────────────────────────────────────────────────────────

    def create_run(self, notes: Optional[str] = None) -> dict[str, Any]:
        runs_dir = _join(self.session_dir, "runs")
        existing = [name for name in self.storage.list_dir_names(runs_dir) if name.isdigit()]
        next_id = 0 if not existing else min(set(range(len(existing) + 1)) - {int(x) for x in existing})
        run_dir = _join(runs_dir, str(next_id))
        copied_dir = _join(run_dir, "copied")
        self.storage.mkdir(copied_dir)
        self.storage.write_text(_join(run_dir, "override.yaml"), "# Fill in the run-specific Hydra override here.\n")
        self.storage.write_text(_join(run_dir, "command.sh"), "#!/usr/bin/env bash\nset -euo pipefail\n\n# Fill in the exact training command here.\n")
        self.storage.write_text(_join(run_dir, "stdout.log"), "")
        self.storage.write_text(_join(run_dir, "stderr.log"), "")
        self.storage.write_text(_join(run_dir, "metrics.json"), "{}\n")
        self.storage.write_text(
            _join(run_dir, "summary.md"),
            f"# Run {next_id}\n\n"
            "## Objective\n\n"
            "## Override\n\n"
            "## Result\n\n"
            "## Takeaway\n\n",
        )
        self.update_run_result(
            run_id=next_id,
            run_name=f"run-{next_id:04d}",
            status="created",
            run_dir=f"runs/{next_id}",
            override_path=f"runs/{next_id}/override.yaml",
            config_path=f"runs/{next_id}/resolved_config.json",
            metrics_path=f"runs/{next_id}/metrics.json",
            summary_path=f"runs/{next_id}/summary.md",
            notes=notes,
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
                f"Read and follow the instructions in {Path(__file__).parent.parent / 'prompts' / 'plan_tuning_strategy.md'}. "
                f"Then specify a hparam combination with `python scripts/session_manager.py {self._ssh_prefix()}"
                f"write-override {self.session_dir} --run-id {next_id} "
                "--override param0=value0 --override moduleA.param1=value1`" +
                next_step_postfix()
        }

    def write_override(
        self,
        run_id: int,
        overrides: Optional[list[str]] = None,
        yaml_content: Optional[str] = None,
    ) -> dict[str, Any]:
        """Write Hydra overrides to ``<run_dir>/override.yaml``.

        Accepts either a list of dotted-path ``key=value`` strings (Hydra CLI style)
        or a raw YAML string.  Exactly one of *overrides* or *yaml_content* must be
        provided.
        """
        if (overrides is None) == (yaml_content is None):
            raise SessionManagerError("Exactly one of 'overrides' or 'yaml_content' must be provided.")

        if overrides is not None:
            cfg = OmegaConf.from_dotlist(overrides)
            text = OmegaConf.to_yaml(cfg)
        else:
            text = yaml_content if yaml_content.endswith("\n") else yaml_content + "\n"

        run_dir = _join(self.session_dir, "runs", str(run_id))
        override_path = _join(run_dir, "override.yaml")
        self.storage.write_text(override_path, text)

        # Sync to the override.yaml that lives next to the primary Hydra config.
        meta_path = _join(self.session_dir, "meta.yaml")
        override_yaml_path: Optional[str] = None
        sync_warning: Optional[str] = None
        try:
            if not self.storage.exists(meta_path):
                sync_warning = f"meta.yaml not found at {meta_path}; skipping config-dir sync."
            else:
                try:
                    meta_cfg = OmegaConf.create(self.storage.read_text(meta_path))
                except Exception as exc:
                    sync_warning = f"Failed to parse meta.yaml: {exc}; skipping config-dir sync."
                    meta_cfg = None
                if meta_cfg is not None:
                    override_yaml_path = meta_cfg.get("override_yaml_path") or None
                    if not override_yaml_path:
                        sync_warning = "meta.yaml has no override_yaml_path (was create-session called with --primary-config-path?)."
                    else:
                        try:
                            self.storage.write_text(override_yaml_path, text)
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
                f"Start a run with `python scripts/session_manager.py {self._ssh_prefix()}"
                f"run-command {self.session_dir} --run-id {run_id} "
                "--conda-env \"specified_conda_env\" --command-str \"specified_command\"`" +
                next_step_postfix()
        }

    def update_run_result(
        self,
        run_id: int,
        run_name: Optional[str] = None,
        status: Optional[RunStatus] = None,
        primary_metric: Any = None,
        best_step: Any = None,
        run_dir: Optional[str] = None,
        override_path: Optional[str] = None,
        config_path: Optional[str] = None,
        metrics_path: Optional[str] = None,
        summary_path: Optional[str] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        notes: Optional[str] = None,
    ) -> dict[str, Any]:
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
        normalized_row = upsert_results_row(self.storage, self.results_path, RESULTS_COLUMNS, row)
        return {
            "results_csv": self.results_path,
            "row": normalized_row,
            "next_step":
                f"Run `python scripts/session_manager.py {self._ssh_prefix()}"
                f"append-report {self.session_dir} \"summary of this run\"`." +
                next_step_postfix()
        }

    def run_command(
        self,
        run_id: int,
        command: str,
        conda_env: Optional[str] = None,
        cwd: Optional[str] = None,
        use_tmux: bool = True,
    ) -> dict[str, Any]:
        """Execute *command* for a run, redirecting stdout/stderr to the run's log files.

        When *use_tmux* is True (default), the command is launched in a detached
        tmux session and this function returns immediately with status ``"running"``.
        Call ``poll_run`` to check completion and update results.csv.

        When *use_tmux* is False, execution is synchronous (blocks until done).
        """
        run_dir = _join(self.session_dir, "runs", str(run_id))
        stdout_path = _join(run_dir, "stdout.log")
        stderr_path = _join(run_dir, "stderr.log")
        effective_cwd = cwd or Path(self.session_dir).parent.parent.parent.resolve().as_posix()

        if conda_env:
            inner = "conda activate " + shlex.quote(conda_env) + " && " + command
            shell_cmd = "bash -i -c " + shlex.quote(inner)
        else:
            shell_cmd = command

        start_time = now_iso()
        self.update_run_result(run_id=run_id, status="running", start_time=start_time)

        if use_tmux:
            tmux_name = self._tmux_session_name(self.session_dir, run_id)
            returncode_path = _join(run_dir, "returncode.txt")
            full_cmd = (
                "cd " + shlex.quote(effective_cwd)
                + " && " + shell_cmd
                + " >" + shlex.quote(stdout_path)
                + " 2>" + shlex.quote(stderr_path)
                + "; echo $? >" + shlex.quote(returncode_path)
            )
            tmux_argv = ["tmux", "new-session", "-d", "-s", tmux_name, full_cmd]

            if self.ssh_host is None:
                proc = subprocess.run(tmux_argv, capture_output=True, text=True)
            else:
                print(" ".join(["ssh", self.ssh_host, " ".join(shlex.quote(a) for a in tmux_argv)]))
                proc = subprocess.run(
                    ["ssh", self.ssh_host, " ".join(shlex.quote(a) for a in tmux_argv)],
                    capture_output=True,
                    text=True,
                )

            if proc.returncode != 0:
                self.update_run_result(run_id=run_id, status="failed", end_time=now_iso())
                raise SessionManagerError(f"tmux launch failed: {proc.stderr.strip() or proc.stdout.strip()}")

            self.storage.write_text(_join(run_dir, "tmux_session.txt"), tmux_name)

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
                    f"A run has been started. Now, query its status with " +
                    "`python scripts/session_manager.py {self._ssh_prefix()} " +
                    f"poll-run {self.session_dir} --run-id {run_id}`" +
                    next_step_postfix()
            }

        # ── Synchronous fallback ─────────────────────────────────────────────
        try:
            if self.ssh_host is None:
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
                print(" ".join(["ssh", self.ssh_host, f"\"{remote_cmd}\""]))
                proc = subprocess.run(["ssh", self.ssh_host, f"\"{remote_cmd}\""], capture_output=True, text=True)
                returncode = proc.returncode
        except Exception as exc:
            end_time = now_iso()
            self.update_run_result(run_id=run_id, status="failed", end_time=end_time, notes=str(exc))
            raise

        end_time = now_iso()
        status: RunStatus = "finished" if returncode == 0 else "failed"
        self.update_run_result(run_id=run_id, status=status, end_time=end_time)

        next_step: list[str] = []
        if status == "finished":
            if self.ssh_host is not None:
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
            "next_step": next_step + [next_step_postfix()],
        }

    def poll_run(self, run_id: int, tail_lines: int = 50) -> dict[str, Any]:
        """Check whether a tmux-launched run has finished and update results.csv.

        Reads the tmux session name from ``<run_dir>/tmux_session.txt`` and queries
        ``tmux has-session``.  If the session is gone, reads the exit code from
        ``<run_dir>/returncode.txt``, updates results.csv, and returns the final status.
        """
        run_dir = _join(self.session_dir, "runs", str(run_id))
        tmux_session_file = _join(run_dir, "tmux_session.txt")
        returncode_path = _join(run_dir, "returncode.txt")
        stdout_path = _join(run_dir, "stdout.log")

        if not self.storage.exists(tmux_session_file):
            raise SessionManagerError(
                f"No tmux_session.txt found for run {run_id}. "
                "Was run-command called with tmux enabled?"
            )

        tmux_name = self.storage.read_text(tmux_session_file).strip()
        check_argv = ["tmux", "has-session", "-t", tmux_name]

        if self.ssh_host is None:
            check = subprocess.run(check_argv, capture_output=True)
        else:
            check = subprocess.run(
                ["ssh", self.ssh_host, " ".join(shlex.quote(a) for a in check_argv)],
                capture_output=True,
            )

        if check.returncode == 0:
            next_step = [
                "The run is not completed yet. Please estimate the remaining time from stdout_tail, " +
                "formalize the remaining time using `python scripts/eta.py`, " +
                "then call `cron.add` with following parameters to remind yourself when it completes:\n" +
                json.dumps({
                    "name": "AHT poll",
                    "schedule": {
                        "kind": "at",
                        "at": "<result of `python scripts/eta.py <remaining time>`>"
                    },
                    "sessionTarget": "main",
                    "wakeMode": "now",
                    "payload": {
                        "kind": "systemEvent",
                        "text": 
                            "According to the remaining time estimation, one AHT run should have finished. " +
                            f"Query the run with `python scripts/session_manager.py {self._ssh_prefix()}"
                            f"poll-run {self.session_dir} --run-id {run_id}`",
                    },
                    "deleteAfterRun": "true"
                }, ensure_ascii=False, indent=2) +
                next_step_postfix()
            ]
            return {
                "run_id": run_id,
                "run_dir": run_dir,
                "tmux_session": tmux_name,
                "status": "running",
                "returncode": None,
                "stdout_tail": self._read_tail(stdout_path, tail_lines),
                "next_step": next_step + [next_step_postfix()],
            }

        # Session is gone → read exit code and finalize
        returncode: Optional[int] = None
        if self.storage.exists(returncode_path):
            try:
                returncode = int(self.storage.read_text(returncode_path).strip())
            except ValueError:
                pass

        status: RunStatus = "finished" if returncode == 0 else "failed"
        end_time = now_iso()
        self.update_run_result(run_id=run_id, status=status, end_time=end_time)

        next_step: list[str] = []
        if status == "finished":
            if self.ssh_host is not None:
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
            "stdout_tail": self._read_tail(stdout_path, tail_lines),
            "next_step": next_step + [next_step_postfix()],
        }

    def append_report(self, content: str) -> dict[str, Any]:
        report_path = _join(self.session_dir, "report.md")
        text = content if content.endswith("\n") else content + "\n"
        self.storage.append_text(report_path, text)
        runs_dir = _join(self.session_dir, "runs")
        if len(self.storage.list_dir_names(runs_dir)) == 0:
            next_step = [
                f"Now the session is initialized. Start the test run with `python scripts/session_manager.py "
                f"{self._ssh_prefix()}create-run {self.session_dir}`"
            ]
        else:
            next_step = [
                f"Run `python scripts/session_manager.py {self._ssh_prefix()}"
                f"summarize-results {self.session_dir}` to analyze the finished runs."
            ]
        return {
            "report_md": report_path,
            "appended_chars": len(text),
            "next_step": next_step + [next_step_postfix()],
        }

    def finalize_session(
        self,
        status: SessionStatus,
        ended_at: Optional[str] = None,
        notes: Optional[str] = None,
    ) -> dict[str, Any]:
        meta_path = _join(self.session_dir, "meta.yaml")
        cfg = OmegaConf.create(self.storage.read_text(meta_path))
        cfg.status = status
        if notes is not None:
            cfg.notes = notes
        cfg.ended_at = ended_at or now_iso()
        self.storage.write_text(meta_path, OmegaConf.to_yaml(cfg))
        return {"meta_yaml": meta_path, "status": status}

    def summarize_results(
        self,
        top_k: int = 3,
        recent_k: int = 5,
        goal: Optional[str] = None,
    ) -> dict[str, Any]:
        meta_path = _join(self.session_dir, "meta.yaml")
        df = load_results_dataframe(self.storage, self.results_path, RESULTS_COLUMNS)

        inferred_goal = goal
        meta_text = self.storage.read_text(meta_path) if self.storage.exists(meta_path) else ""
        if inferred_goal is None and meta_text:
            meta_cfg = OmegaConf.create(meta_text)
            inferred_goal = meta_cfg.get("goal") or None
        if inferred_goal is None:
            inferred_goal = "maximize"

        if df.empty:
            return {
                "session_dir": self.session_dir,
                "results_csv": self.results_path,
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
            "session_dir": self.session_dir,
            "results_csv": self.results_path,
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
                f"The AHT loop just iterated once. Start another run with `python scripts/session_manager.py "
                f"{self._ssh_prefix()}create-run {self.session_dir}` "
                "or stop here if you believe the tuning process is completed." +
                next_step_postfix()
        }


# ── CLI ───────────────────────────────────────────────────────────────────────

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
        _, result = SessionManager.create(
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
    else:
        mgr = SessionManager(session_dir=args.session_dir, ssh_host=args.ssh_host)
        if args.command == "create-run":
            result = mgr.create_run(notes=args.notes)
        elif args.command == "update-run":
            result = mgr.update_run_result(
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
            )
        elif args.command == "summarize-results":
            result = mgr.summarize_results(top_k=args.top_k, recent_k=args.recent_k, goal=args.goal)
        elif args.command == "append-report":
            result = mgr.append_report(content=args.content)
        elif args.command == "finalize-session":
            result = mgr.finalize_session(status=args.status, ended_at=args.ended_at, notes=args.notes)
        elif args.command == "write-override":
            result = mgr.write_override(run_id=args.run_id, overrides=args.overrides, yaml_content=args.yaml_content)
        elif args.command == "run-command":
            result = mgr.run_command(
                run_id=args.run_id,
                command=args.command_str,
                conda_env=args.conda_env,
                cwd=args.cwd,
                use_tmux=not args.no_tmux,
            )
        elif args.command == "poll-run":
            result = mgr.poll_run(run_id=args.run_id, tail_lines=args.tail)
        else:
            raise SessionManagerError(f"Unsupported command: {args.command}")

    print(json.dumps(result, ensure_ascii=False, indent=2, default=_json_safe_default))


if __name__ == "__main__":
    main()
