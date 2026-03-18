from __future__ import annotations

import sys
import argparse
import hashlib
import json
import shlex
import subprocess
from datetime import datetime
from pathlib import PurePosixPath, Path
from typing import Any, Literal, Optional, Union

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
    system_prompt,
    get_sessions_spawn_command,
    get_cron_add_command
)
from analyze_event import summarize_scalar_curve, event2dataframe

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
        self.project_dir = PurePosixPath(self.session_dir).parents[2].as_posix()
        self.ssh_host = ssh_host
        self.storage: Storage = _default_storage(TargetSpec(project_root=session_dir, ssh_host=ssh_host))
        self.session_info = {
            "session_dir": self.session_dir,
            "local_python_executable": sys.executable,
            "storage": "ssh" if self.ssh_host else "local",
            "system_prompt": system_prompt()
        }
        if self.ssh_host:
            self.session_info["ssh_host"] = self.ssh_host
        self.strategy_path = _join(self.session_dir, "strategy.md")
        self.report_path = _join(session_dir, "report.md")
        self.script_path = Path(__file__).resolve()
        self.python_cmd = f"python {self.script_path.as_posix()} {self._ssh_prefix()}"
        self.eta_cmd = f"python {self.script_path.parent.joinpath('eta.py').as_posix()}"
        self.hparam_md_path = _join(self.project_dir, "HPARAM.md")
        assert self.storage.exists(self.hparam_md_path)
        assert self.storage.exists(_join(self.session_dir, "meta.yaml"))
        self.meta_cfg = OmegaConf.create(self.storage.read_text(_join(self.session_dir, "meta.yaml")))

    @property
    def results_path(self) -> str:
        return _join(self.session_dir, "results.csv")

    # ── Factory ───────────────────────────────────────────────────────────────

    @classmethod
    def create(
        cls,
        project_root: str,
        base_command: str,
        primary_metric: str,
        primary_config_path: str,
        goal: str,
        agent: str = "openclaw",
        skill: str = "auto-hparam-tuning",
        notes: Optional[str] = None,
        ssh_host: Optional[str] = None,
        timestamp: Optional[str] = None,
    ) -> tuple["SessionManager", dict[str, Any]]:
        """Create a new AHT session directory and return a ``(SessionManager, result)`` pair."""
        storage = _default_storage(TargetSpec(project_root=project_root, ssh_host=ssh_host))
        
        if not primary_config_path.startswith("/"):
            primary_config_path = _join(project_root, primary_config_path)
        assert storage.exists(primary_config_path), f"primary_config_path not exist: {primary_config_path}"
        override_yaml_path: Optional[str] = None
        parent = primary_config_path.rsplit("/", 1)[0] if "/" in primary_config_path else "."
        override_yaml_path = parent + "/override.yaml"
        ensure_override_in_defaults(storage, primary_config_path)
        storage.write_text(override_yaml_path, "")
        assert storage.exists(override_yaml_path), f"override_yaml_path not exist: {override_yaml_path}"

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
            f"- Base command: `{base_command}`\n"
            f"- Primary metric: `{primary_metric}`\n"
            f"- Goal: `{goal}`\n\n",
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
            **mgr.session_info,
            "project_root": project_root,
            "runs_dir": runs_dir,
            "results_csv": mgr.results_path,
            "report_md": _join(session_dir, "report.md"),
            "meta_yaml": _join(session_dir, "meta.yaml"),
            "storage": meta["storage"],
            "ssh_host": ssh_host,
            "primary_config_path": primary_config_path,
            "override_yaml_path": override_yaml_path,
            "next_step":
                "Run `python scripts/session_manager.py " +
                (f"--ssh-host \"{ssh_host}\" " if ssh_host is not None else "") +
                f"append-report {session_dir} \"your understandings to the task\"` " +
                "to write down your understandings to the task."
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
            **self.session_info,
            "run_id": next_id,
            "run_dir": run_dir,
            "override_path": _join(run_dir, "override.yaml"),
            "command_path": _join(run_dir, "command.sh"),
            "stdout_path": _join(run_dir, "stdout.log"),
            "stderr_path": _join(run_dir, "stderr.log"),
            "metrics_path": _join(run_dir, "metrics.json"),
            "summary_path": _join(run_dir, "summary.md"),
            "copied_dir": copied_dir,
            "next_step": (
                (self.generate_tuning_strategy() if next_id == 0 else "") +
                f"Run created. Now, start tuning by `{self.python_cmd} tune {self.session_dir} --run-id {next_id}` " +
                "and spawn a subagent following the instruction."
            )
        }
    
    def generate_tuning_strategy(self):
        self.storage.write_text(self.strategy_path, "")
        skill_dir = Path(__file__).resolve().parent.parent
        next_step = [
            "Spawn a subagent to walk through the project with following command: " + 
            get_sessions_spawn_command(
                label="tuning_strategy_generation",
                task=(
                    f"Read and follow the instruction in {str(skill_dir / 'prompts' / 'plan_tuning_strategy.md')} "+
                    f"and write your summary in {self.strategy_path} "+
                    (f" in remote host {self.ssh_host}." if self.ssh_host is not None else ".")
                )
            ) + 
            "After spawned, wait until the subagent returns. Then, follow the following instructions."
        ]
        
        return "\n".join(next_step)
    
    def tune(self, run_id: int) -> dict[str, Any]:
        """Spawn a subagent to decide the next hyperparameter override for *run_id*.

        Reads ``report.md``, ``strategy.md``, and the live results summary, then
        uses ``get_sessions_spawn_command`` to build a spawn instruction.  The
        subagent will call ``write-override`` autonomously; the caller should block
        until it returns before launching the run.
        """
        summary = self.summarize_results()

        task = (
            f"You are a hyperparameter tuning expert assisting an ongoing AHT session{(' on remote machine '+self.ssh_host) if self.ssh_host is not None else ''}.\n\n"
            f"## Context\n\n"
            f"### Session directory\n{self.session_dir}\n\n"
            f"### Hyperparameter structure document\n{self.hparam_md_path}\n\n"
            f"### Results summary\n```json\n{json.dumps(summary, indent=2, default=_json_safe_default)}\n```\n\n"
            f"### Session report path\n{self.report_path}\n\n"
            f"### Local python environemnt\n <the local conda or python environment>\n\n"
            f"## Your task\n\n"
            f"1. Carefully review the results summary, tuning strategy, and the run history in the report above.\n"
            f"2. Decide the best hyperparameter override to try next for the current run {run_id}, "
            f"following the guidance in `{self.strategy_path}`.\n"
            f"3. Start a run by running:\n"
            f"   ```bash\n"
            f"   {self.python_cmd} run {self.session_dir} --run-id {run_id} "
            f"--command-str \"specified_command\" --conda-env \"specified_conda_env\""
            f"--override key=value [--override key2=value2 ...]\n"
            f"   ```\n"
            f"4. Report what override you chose and the reasoning behind it.\n"
        )

        spawn_cmd = get_sessions_spawn_command(
            label=f"aht_tune_run{run_id}",
            task=task,
        )

        return {
            **self.session_info,
            "session_dir": self.session_dir,
            "run_id": run_id,
            "report_path": self.report_path,
            "strategy_path": self.strategy_path,
            "summarized_results": summary,
            "next_step": (
                f"Spawn a subagent to decide the next hyperparameter override: {spawn_cmd}" +
                "After spawning, DO NOTHING until the subagent returns." +
                f"After the subagent returns, poll the run with: " +
                f"`{self.python_cmd} poll-run {self.session_dir} --run-id {run_id}`"
            ),
        }

    def override_and_run(
        self,
        run_id: int,
        command: str,
        overrides: list[str] = [],
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
        
        cfg = OmegaConf.from_dotlist(overrides)
        text = OmegaConf.to_yaml(cfg)

        run_dir = _join(self.session_dir, "runs", str(run_id))
        override_path = _join(run_dir, "override.yaml")
        self.storage.write_text(override_path, text)

        # Sync to the override.yaml that lives next to the primary Hydra config.
        override_yaml_path: Optional[str] = None
        sync_warning: Optional[str] = None
        override_yaml_path = self.meta_cfg.get("override_yaml_path") or None
        if not override_yaml_path:
            sync_warning = "meta.yaml has no override_yaml_path (was create-session called with --primary-config-path?)."
        else:
            try:
                self.storage.write_text(override_yaml_path, text)
            except Exception as exc:
                sync_warning = f"Wrote run override but failed to sync to {override_yaml_path}: {exc}"
                override_yaml_path = None
        
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
                **self.session_info,
                "run_id": run_id,
                "run_dir": run_dir,
                "override_sync_warning": sync_warning,
                "stdout_path": stdout_path,
                "stderr_path": stderr_path,
                "tmux_session": tmux_name,
                "status": "running",
                "start_time": start_time,
                "cwd": effective_cwd,
                "next_step":
                    f"A run has been started. Now, return to the caller agent who called you and ask it to query the status with " +
                    f"`{self.python_cmd} poll-run {self.session_dir} --run-id {run_id}`"
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
            **self.session_info,
            "run_id": run_id,
            "run_dir": run_dir,
            "override_sync_warning": sync_warning,
            "stdout_path": stdout_path,
            "stderr_path": stderr_path,
            "returncode": returncode,
            "status": status,
            "start_time": start_time,
            "end_time": end_time,
            "cwd": effective_cwd,
            "next_step": next_step,
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
            **self.session_info,
            "results_csv": self.results_path,
            "row": normalized_row,
            "next_step":
                f"Run `python scripts/session_manager.py {self._ssh_prefix()}"
                f"append-report {self.session_dir} \"summary of this run\"`."
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
                "Was `python scripts/session_manager.py run` called with tmux enabled?"
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
                f"formalize the remaining time using `{self.eta_cmd}`, " +
                "then call `cron.add` tool in openclaw with following parameters to remind yourself when it completes:\n" +
                get_cron_add_command(
                    name=f"aht_poll_run{run_id}",
                    at="<result of eta.py>",
                    payload=
                        "According to the remaining time estimation, one AHT run should have finished. " +
                        f"Query the run with `{self.python_cmd} poll-run {self.session_dir} --run-id {run_id}`",
                ) +
                "\n Note: DO NOT USE SLEEP comamnd to wait. You MUST use `cron.add`."
            ]
            return {
                **self.session_info,
                "run_id": run_id,
                "run_dir": run_dir,
                "tmux_session": tmux_name,
                "status": "running",
                "returncode": None,
                "stdout_tail": self._read_tail(stdout_path, tail_lines),
                "next_step": next_step,
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

        if status == "finished":
            next_step = (
                "Run finished. Now:\n" +
                (f"- Copy back the tensorboard event file to the `/tmp/run{run_id}_tensorboard_event` using scp.\n") if self.ssh_host is not None else "\n" +
                f"- Then analyze the tensorboard events by `{self.python_cmd} analyze-event /path/to/event/file`."
            )
        else:
            next_step = "Run failed. Now try to figure out what's wrong according to the stdout."

        return {
            **self.session_info,
            "run_id": run_id,
            "run_dir": run_dir,
            "tmux_session": tmux_name,
            "status": status,
            "returncode": returncode,
            "end_time": end_time,
            "stdout_tail": self._read_tail(stdout_path, tail_lines),
            "next_step": next_step,
        }
    
    def analyze_event(
        self,
        run_id: int,
        event_path: str,
        smoothing: float = 0.0,
        quantile_low: float = 0.05,
        quantile_high: float = 0.95,
    ) -> dict[str, Any]:
        run_dir = _join(self.session_dir, "runs", str(run_id))
        df = event2dataframe(event_path)
        results = [
            summarize_scalar_curve(
                df=df,
                key=k,
                smoothing=smoothing,
                quantile_low=quantile_low,
                quantile_high=quantile_high,
                mode=self.meta_cfg.get("goal", "maximize"),
            )
            for k in list(df.columns)
        ]
        output_path = _join(self.session_dir, "runs", str(run_id), "event_analysis.json")
        self.storage.write_text(
            output_path,
            json.dumps(results, indent=2, default=_json_safe_default) + "\n",
        )
        task = (
            f"You are a hyperparameter tuning expert assisting an ongoing AHT session{(' on remote machine '+self.ssh_host) if self.ssh_host is not None else ''}.\n\n"
            f"Now a run has finished. Analyze the run and write a report.\n\n"
            f"## Context\n\n"
            f"### Session directory\n{self.session_dir}\n\n"
            f"### Hyperparameter structure document\n{self.hparam_md_path}\n\n"
            f"### Session report path\n{self.report_path}\n\n"
            f"### Run directory\n{run_dir}\n\n"
            f"### Override hyperparameters\n{_join(run_dir, 'override.yaml')}\n\n"
            f"### Local python environemnt\n <the local conda or python environment>\n\n"
            f"## Your task\n\n"
            f"- Find out where the tensorboard event file of the current run {run_id} is placed. "
            f"It should be noted in the hyperparameter structure document.\n"
            f"- Read the analysis in `event_analysis_path` and the report in `{self.report_path}`.\n"
            f"- Then update the report by `{self.python_cmd} append-report \"content\"`."
        )
        spawn_cmd = get_sessions_spawn_command(
            label=f"aht_analyze_run{run_id}",
            task=task,
        )
        next_step = f"Run finished. Spawn a subagent to analyze the run and update the report by {spawn_cmd}."
        return {
            **self.session_info,
            "run_id": run_id,
            "event_path": event_path,
            "event_analysis_path": output_path,
            "next_step": next_step
        }
    
    def append_report(self, content: str) -> dict[str, Any]:
        report_path = _join(self.session_dir, "report.md")
        text = content if content.endswith("\n") else content + "\n"
        self.storage.append_text(report_path, text)
        runs_dir = _join(self.session_dir, "runs")
        if len(self.storage.list_dir_names(runs_dir)) == 0:
            next_step = [
                f"Now the session is initialized. Start the test run with "
                f"`{self.python_cmd} create-run {self.session_dir}`."
            ]
        else:
            next_step = [
                f"Report appended. Run `{self.python_cmd} create-run {self.session_dir}` to create another run."
            ]
        return {
            **self.session_info,
            "report_md": report_path,
            "appended_chars": len(text),
            "next_step": next_step,
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
        recent_k: int = 5
    ) -> dict[str, Any]:
        meta_path = _join(self.session_dir, "meta.yaml")
        df = load_results_dataframe(self.storage, self.results_path, RESULTS_COLUMNS)

        meta_text = self.storage.read_text(meta_path) if self.storage.exists(meta_path) else ""
        if meta_text:
            meta_cfg = OmegaConf.create(meta_text)
            inferred_goal = meta_cfg.get("goal", "maximize")
        else:
            inferred_goal = "maximize"

        if df.empty:
            return {
                **self.session_info,
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
            **self.session_info,
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
                f"The AHT loop just iterated once. Start another run with `{self.python_cmd} create-run {self.session_dir}` "
                "or stop here if you believe the tuning process is completed." +
                "Check trend_hint: Stop if \"flat\" or \"degrading\" for 3+ consecutive runs,  budget exhausted, or objective already met." +
                "Otherwise go back to start another run."
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

    p_report = subparsers.add_parser("append-report", help="Append markdown content to report.md.")
    p_report.add_argument("session_dir")
    p_report.add_argument("content", help="Markdown text to append.")

    p_finalize = subparsers.add_parser("finalize-session", help="Mark a session as completed/stopped/failed.")
    p_finalize.add_argument("session_dir")
    p_finalize.add_argument("status", choices=["running", "completed", "stopped", "failed"])
    p_finalize.add_argument("--ended-at", default=None)
    p_finalize.add_argument("--notes", default=None)

    p_exec = subparsers.add_parser(
        "run",
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
    group = p_exec.add_mutually_exclusive_group(required=True)
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

    p_spawn = subparsers.add_parser(
        "tune",
        help=(
            "Spawn a subagent that reads the session context (report, strategy, results summary) "
            "and decides the next hyperparameter override then run with selected overrides."
        ),
    )
    p_spawn.add_argument("session_dir")
    p_spawn.add_argument("--run-id", type=int, required=True, help="Run ID the subagent should write the override for.")

    p_analyze = subparsers.add_parser(
        "analyze-event",
        help="Analyze all scalar curves in a TensorBoard event file and write event_analysis.json to the run directory.",
    )
    p_analyze.add_argument("session_dir")
    p_analyze.add_argument("--run-id", type=int, required=True, help="Run ID whose directory receives event_analysis.json.")
    p_analyze.add_argument("--event-path", required=True, metavar="PATH", help="Path to the TensorBoard event file.")
    p_analyze.add_argument("--smoothing", type=float, default=0.0, metavar="S", help="EMA smoothing factor (default: 0.0).")
    p_analyze.add_argument("--quantile-low", type=float, default=0.05, metavar="Q", help="Lower quantile for range stats (default: 0.05).")
    p_analyze.add_argument("--quantile-high", type=float, default=0.95, metavar="Q", help="Upper quantile for range stats (default: 0.95).")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.command == "create-session":
        _, result = SessionManager.create(
            project_root=args.project_root,
            base_command=args.base_command,
            primary_metric=args.primary_metric,
            primary_config_path=args.primary_config_path,
            goal=args.goal,
            agent=args.agent,
            skill=args.skill,
            notes=args.notes,
            ssh_host=args.ssh_host,
            timestamp=args.timestamp,
        )
    else:
        mgr = SessionManager(session_dir=args.session_dir, ssh_host=args.ssh_host)
        if args.command == "create-run":
            result = mgr.create_run(notes=args.notes)
        elif args.command == "append-report":
            result = mgr.append_report(content=args.content)
        elif args.command == "finalize-session":
            result = mgr.finalize_session(status=args.status, ended_at=args.ended_at, notes=args.notes)
        elif args.command == "run":
            result = mgr.override_and_run(
                run_id=args.run_id,
                command=args.command_str,
                overrides=args.overrides,
                conda_env=args.conda_env,
                cwd=args.cwd,
                use_tmux=not args.no_tmux,
            )
        elif args.command == "poll-run":
            result = mgr.poll_run(run_id=args.run_id, tail_lines=args.tail)
        elif args.command == "tune":
            result = mgr.tune(run_id=args.run_id)
        elif args.command == "analyze-event":
            result = mgr.analyze_event(
                run_id=args.run_id,
                event_path=args.event_path,
                smoothing=args.smoothing,
                quantile_low=args.quantile_low,
                quantile_high=args.quantile_high,
            )
        else:
            raise SessionManagerError(f"Unsupported command: {args.command}")

    print(json.dumps(result, ensure_ascii=False, indent=2, default=_json_safe_default))


if __name__ == "__main__":
    main()
