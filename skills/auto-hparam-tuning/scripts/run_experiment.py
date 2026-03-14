"""Unified experiment runner for AHT.

Generates Hydra override strings and launches training commands, capturing
stdout/stderr, enforcing timeouts, and collecting post-run metadata.

Usage
-----
CLI::

    python run_experiment.py \\
        --project /path/to/project \\
        --entry train.py \\
        --overrides "optimizer.lr=0.0003" "model.dropout=0.2" \\
        --run-name lr_sweep_01 \\
        --timeout 3600

    # Dry-run mode (print command without executing)
    python run_experiment.py \\
        --project /path/to/project \\
        --entry train.py \\
        --overrides "optimizer.lr=0.0003" \\
        --dry-run

Library::

    from run_experiment import run_experiment
    result = run_experiment(
        project_dir="/path/to/project",
        entry="train.py",
        overrides=["optimizer.lr=0.0003"],
        run_name="lr_sweep_01",
        timeout=3600,
    )
    print(result["exit_code"])
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _build_command(
    entry: str,
    overrides: list[str] | None = None,
    config_dir: str | None = None,
    config_name: str | None = None,
    extra_args: list[str] | None = None,
    python_executable: str = "python",
    multirun: bool = False,
) -> list[str]:
    """Build the training command as a list of tokens.

    Parameters
    ----------
    entry : str
        Training script path (e.g. ``train.py``).
    overrides : list[str] or None
        Hydra overrides, each as ``"key=value"``.
    config_dir : str or None
        Override for Hydra ``--config-dir``.
    config_name : str or None
        Override for Hydra ``--config-name``.
    extra_args : list[str] or None
        Additional raw CLI arguments.
    python_executable : str
        Python interpreter to use.
    multirun : bool
        If *True*, add ``--multirun`` / ``-m`` flag for Hydra sweep.

    Returns
    -------
    list[str]
        Command tokens ready for :func:`subprocess.run`.
    """
    cmd: list[str] = [python_executable, entry]

    if config_dir:
        cmd.extend(["--config-dir", config_dir])
    if config_name:
        cmd.extend(["--config-name", config_name])
    if multirun:
        cmd.append("--multirun")
    if overrides:
        cmd.extend(overrides)
    if extra_args:
        cmd.extend(extra_args)
    return cmd


def _detect_oom(stderr_text: str) -> bool:
    """Heuristic check for GPU OOM errors in stderr."""
    oom_patterns = [
        "CUDA out of memory",
        "OutOfMemoryError",
        "RuntimeError: CUDA error: out of memory",
        "torch.cuda.OutOfMemoryError",
        "CUBLAS_STATUS_ALLOC_FAILED",
        "cuDNN error: CUDNN_STATUS_ALLOC_FAILED",
    ]
    stderr_lower = stderr_text.lower()
    return any(p.lower() in stderr_lower for p in oom_patterns)


def _detect_nan(stderr_text: str, stdout_text: str) -> bool:
    """Heuristic check for NaN-related training failures."""
    combined = (stderr_text + stdout_text).lower()
    nan_patterns = [
        "loss is nan",
        "nan loss",
        "nan detected",
        "nan in gradient",
        "nan values",
    ]
    return any(p in combined for p in nan_patterns)


def _find_event_files(directory: Path) -> list[str]:
    """Search for TensorBoard event files created during the run."""
    events: list[str] = []
    if not directory.is_dir():
        return events
    for f in directory.rglob("events.out.tfevents.*"):
        events.append(str(f.resolve()))
    return sorted(events)


def run_experiment(
    project_dir: str,
    entry: str,
    overrides: list[str] | None = None,
    run_name: str | None = None,
    config_dir: str | None = None,
    config_name: str | None = None,
    timeout: int | None = None,
    dry_run: bool = False,
    env_vars: dict[str, str] | None = None,
    gpu_ids: list[int] | None = None,
    python_executable: str = "python",
    multirun: bool = False,
    extra_args: list[str] | None = None,
) -> dict[str, Any]:
    """Launch a training experiment and collect results.

    Parameters
    ----------
    project_dir : str
        Working directory for the training process.
    entry : str
        Training script (relative to *project_dir* or absolute).
    overrides : list[str] or None
        Hydra overrides as ``["key=value", ...]``.
    run_name : str or None
        Human-friendly name for this run.
    config_dir, config_name : str or None
        Hydra ``--config-dir`` and ``--config-name`` overrides.
    timeout : int or None
        Maximum wall-clock seconds before killing the process.
    dry_run : bool
        If *True*, build and return the command without executing.
    env_vars : dict or None
        Extra environment variables to set.
    gpu_ids : list[int] or None
        If given, set ``CUDA_VISIBLE_DEVICES`` accordingly.
    python_executable : str
        Python interpreter path or name.
    multirun : bool
        Enable Hydra multirun mode (``--multirun``).
    extra_args : list[str] or None
        Additional raw CLI arguments appended after overrides.

    Returns
    -------
    dict
        A run result dictionary with keys:

        - ``run_name``, ``command``, ``overrides``
        - ``start_time``, ``end_time``, ``duration_seconds``
        - ``exit_code``, ``timed_out``, ``oom_detected``, ``nan_detected``
        - ``stdout`` (last 5000 chars), ``stderr`` (last 5000 chars)
        - ``event_files``: list of event file paths found after the run
        - ``status``: ``"dry_run"`` | ``"success"`` | ``"failed"`` | ``"timeout"`` | ``"oom"``
    """
    cmd = _build_command(
        entry=entry,
        overrides=overrides,
        config_dir=config_dir,
        config_name=config_name,
        extra_args=extra_args,
        python_executable=python_executable,
        multirun=multirun,
    )
    cmd_str = " ".join(cmd)

    result: dict[str, Any] = {
        "run_name": run_name or "unnamed",
        "command": cmd_str,
        "command_tokens": cmd,
        "overrides": overrides or [],
        "project_dir": str(Path(project_dir).resolve()),
    }

    if dry_run:
        result["status"] = "dry_run"
        result["message"] = f"Dry run — would execute: {cmd_str}"
        return result

    # Prepare environment
    run_env = os.environ.copy()
    if gpu_ids is not None:
        run_env["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpu_ids)
    if env_vars:
        run_env.update(env_vars)

    # Launch
    start_time = time.time()
    start_dt = datetime.now(timezone.utc).isoformat()
    timed_out = False

    try:
        proc = subprocess.run(
            cmd,
            cwd=project_dir,
            env=run_env,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        exit_code = proc.returncode
        stdout_text = proc.stdout or ""
        stderr_text = proc.stderr or ""
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        exit_code = -1
        stdout_text = (exc.stdout or b"").decode("utf-8", errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        stderr_text = (exc.stderr or b"").decode("utf-8", errors="replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
    except FileNotFoundError:
        exit_code = -2
        stdout_text = ""
        stderr_text = f"Entry script not found: {entry}"
        timed_out = False

    end_time = time.time()
    end_dt = datetime.now(timezone.utc).isoformat()
    duration = end_time - start_time

    oom = _detect_oom(stderr_text)
    nan = _detect_nan(stderr_text, stdout_text)

    # Determine status
    if timed_out:
        status = "timeout"
    elif oom:
        status = "oom"
    elif exit_code != 0:
        status = "failed"
    else:
        status = "success"

    # Search for event files created during the run
    event_files = _find_event_files(Path(project_dir))

    result.update({
        "start_time": start_dt,
        "end_time": end_dt,
        "duration_seconds": round(duration, 2),
        "exit_code": exit_code,
        "timed_out": timed_out,
        "oom_detected": oom,
        "nan_detected": nan,
        "stdout": stdout_text[-5000:] if len(stdout_text) > 5000 else stdout_text,
        "stderr": stderr_text[-5000:] if len(stderr_text) > 5000 else stderr_text,
        "event_files": event_files,
        "status": status,
    })

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Launch a training experiment with Hydra overrides.\n\n"
            "Executes the specified training script in the project directory, "
            "applying the given Hydra overrides. Captures stdout/stderr, enforces "
            "timeout, and detects common failure modes (OOM, NaN)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--project", "-p", required=True, metavar="DIR",
        help="Project directory (used as working directory for the training process).",
    )
    parser.add_argument(
        "--entry", "-e", required=True, metavar="FILE",
        help="Training script path (relative to --project or absolute).",
    )
    parser.add_argument(
        "--overrides", "-O", nargs="*", default=None, metavar="K=V",
        help="Hydra overrides, e.g. optimizer.lr=0.0003 model.dropout=0.2",
    )
    parser.add_argument(
        "--run-name", "-n", default=None, metavar="NAME",
        help="Human-friendly name for this run.",
    )
    parser.add_argument(
        "--config-dir", default=None, metavar="DIR",
        help="Override Hydra --config-dir.",
    )
    parser.add_argument(
        "--config-name", default=None, metavar="NAME",
        help="Override Hydra --config-name.",
    )
    parser.add_argument(
        "--timeout", "-t", type=int, default=None, metavar="SEC",
        help="Maximum wall-clock seconds before killing the process.",
    )
    parser.add_argument(
        "--gpu", nargs="*", type=int, default=None, metavar="ID",
        help="GPU IDs to use (sets CUDA_VISIBLE_DEVICES).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print the command without executing.",
    )
    parser.add_argument(
        "--multirun", "-m", action="store_true",
        help="Enable Hydra multirun mode.",
    )
    parser.add_argument(
        "--python", default="python", metavar="EXE",
        help="Python executable to use. Default: python",
    )
    parser.add_argument(
        "--output", "-o", default=None, metavar="FILE",
        help="Write result JSON to this file instead of stdout.",
    )

    args = parser.parse_args()

    result = run_experiment(
        project_dir=args.project,
        entry=args.entry,
        overrides=args.overrides,
        run_name=args.run_name,
        config_dir=args.config_dir,
        config_name=args.config_name,
        timeout=args.timeout,
        dry_run=args.dry_run,
        gpu_ids=args.gpu,
        python_executable=args.python,
        multirun=args.multirun,
    )

    json_str = json.dumps(result, indent=2, ensure_ascii=False)

    if args.output:
        Path(args.output).write_text(json_str, encoding="utf-8")
        print(f"Result written to {args.output}", file=sys.stderr)
    else:
        print(json_str)

    # Exit with appropriate code
    if result.get("status") == "dry_run":
        sys.exit(0)
    sys.exit(0 if result.get("exit_code", -1) == 0 else 1)


if __name__ == "__main__":
    main()
