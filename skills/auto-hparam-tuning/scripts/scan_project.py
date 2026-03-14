"""Project structure scanner for AHT.

Scans a training project directory and identifies key components needed for
automated hyperparameter tuning: training entry points, Hydra config directories,
log directories, checkpoint paths, and TensorBoard event file locations.

Usage
-----
CLI::

    python scan_project.py --project /path/to/training/project
    python scan_project.py --project /path/to/project --output project_info.json

Library::

    from scan_project import scan_project
    info = scan_project("/path/to/project")
    print(info["entry_points"])
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Detection helpers
# ---------------------------------------------------------------------------

# Common names for training entry scripts
_ENTRY_PATTERNS = [
    "train.py", "main.py", "run.py", "run_training.py", "train_model.py",
    "train_net.py", "experiment.py", "launch.py",
]

# Common config directory names
_CONFIG_DIR_NAMES = [
    "conf", "configs", "config", "cfg", "hydra_configs", "hydra_conf",
]

# Common log directory names
_LOG_DIR_NAMES = [
    "logs", "log", "runs", "tb_logs", "tensorboard", "tb_events",
    "train_logs", "training_logs",
]

# Common checkpoint directory names
_CKPT_DIR_NAMES = [
    "checkpoints", "ckpt", "ckpts", "models", "saved_models",
    "model_checkpoints", "weights",
]

# Common output directory names
_OUTPUT_DIR_NAMES = [
    "outputs", "output", "results", "experiments", "exps",
]

# Regex for Hydra entry point detection
_HYDRA_MAIN_RE = re.compile(
    r"@hydra\.main\s*\(([^)]*)\)", re.DOTALL
)
_CONFIG_PATH_RE = re.compile(r'config_path\s*=\s*["\']([^"\']+)["\']')
_CONFIG_NAME_RE = re.compile(r'config_name\s*=\s*["\']([^"\']+)["\']')

# Regex for SummaryWriter detection
_SUMMARY_WRITER_RE = re.compile(
    r"SummaryWriter\s*\(([^)]*)\)", re.DOTALL
)
_LOG_DIR_ARG_RE = re.compile(r'log_dir\s*=\s*["\']?([^"\'),\s]+)')


def _find_files_by_name(root: Path, names: list[str], max_depth: int = 3) -> list[Path]:
    """Find files matching any of *names* under *root*, up to *max_depth*."""
    results: list[Path] = []
    for dirpath_str, dirnames, filenames in os.walk(root):
        dirpath = Path(dirpath_str)
        depth = len(dirpath.relative_to(root).parts)
        if depth > max_depth:
            dirnames.clear()
            continue
        # Skip hidden and common non-source directories
        dirnames[:] = [d for d in dirnames if not d.startswith(".") and d not in {
            "__pycache__", "node_modules", ".git", ".hg", "venv", "env",
            ".venv", ".env", ".tox", "dist", "build", "egg-info",
        }]
        for fname in filenames:
            if fname.lower() in [n.lower() for n in names]:
                results.append(dirpath / fname)
    return results


def _find_dirs_by_name(root: Path, names: list[str], max_depth: int = 3) -> list[Path]:
    """Find directories matching any of *names* under *root*."""
    results: list[Path] = []
    for dirpath_str, dirnames, _ in os.walk(root):
        dirpath = Path(dirpath_str)
        depth = len(dirpath.relative_to(root).parts)
        if depth > max_depth:
            dirnames.clear()
            continue
        dirnames[:] = [d for d in dirnames if not d.startswith(".") and d not in {
            "__pycache__", "node_modules", ".git",
        }]
        for dname in dirnames:
            if dname.lower() in [n.lower() for n in names]:
                results.append(dirpath / dname)
    return results


def _find_event_files(root: Path, max_depth: int = 5) -> list[Path]:
    """Find TensorBoard event files recursively."""
    results: list[Path] = []
    for dirpath_str, dirnames, filenames in os.walk(root):
        dirpath = Path(dirpath_str)
        depth = len(dirpath.relative_to(root).parts)
        if depth > max_depth:
            dirnames.clear()
            continue
        dirnames[:] = [d for d in dirnames if not d.startswith(".")]
        for fname in filenames:
            if fname.startswith("events.out.tfevents."):
                results.append(dirpath / fname)
    return results


def _find_python_files(root: Path, max_depth: int = 3) -> list[Path]:
    """Find all .py files under *root*."""
    results: list[Path] = []
    for dirpath_str, dirnames, filenames in os.walk(root):
        dirpath = Path(dirpath_str)
        depth = len(dirpath.relative_to(root).parts)
        if depth > max_depth:
            dirnames.clear()
            continue
        dirnames[:] = [d for d in dirnames if not d.startswith(".") and d not in {
            "__pycache__", "node_modules", ".git", "venv", "env", ".venv",
        }]
        for fname in filenames:
            if fname.endswith(".py"):
                results.append(dirpath / fname)
    return results


def _detect_hydra_entry(py_file: Path) -> dict[str, Any] | None:
    """Parse a Python file for @hydra.main() and extract config metadata."""
    try:
        content = py_file.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return None

    match = _HYDRA_MAIN_RE.search(content)
    if match is None:
        return None

    args_text = match.group(1)
    config_path_match = _CONFIG_PATH_RE.search(args_text)
    config_name_match = _CONFIG_NAME_RE.search(args_text)

    config_path = config_path_match.group(1) if config_path_match else None
    config_name = config_name_match.group(1) if config_name_match else None

    # Resolve config directory
    config_dir = None
    if config_path is not None:
        candidate = py_file.parent / config_path
        if candidate.is_dir():
            config_dir = str(candidate.resolve())
        else:
            config_dir = config_path  # relative, cannot resolve

    return {
        "file": str(py_file.resolve()),
        "config_path": config_path,
        "config_name": config_name,
        "config_dir_resolved": config_dir,
    }


def _detect_summary_writer(py_files: list[Path]) -> list[dict[str, Any]]:
    """Search Python files for TensorBoard SummaryWriter usage."""
    results: list[dict[str, Any]] = []
    for py_file in py_files:
        try:
            content = py_file.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        if "SummaryWriter" not in content:
            continue

        log_dirs: list[str] = []
        for match in _SUMMARY_WRITER_RE.finditer(content):
            args_text = match.group(1)
            ld_match = _LOG_DIR_ARG_RE.search(args_text)
            if ld_match:
                log_dirs.append(ld_match.group(1))

        results.append({
            "file": str(py_file.resolve()),
            "log_dir_references": log_dirs,
        })
    return results


def _detect_yaml_configs(config_dirs: list[Path]) -> list[dict[str, Any]]:
    """List YAML config files in the identified config directories."""
    configs: list[dict[str, Any]] = []
    for cdir in config_dirs:
        if not cdir.is_dir():
            continue
        for f in sorted(cdir.rglob("*.yaml")):
            rel = f.relative_to(cdir)
            configs.append({
                "path": str(f.resolve()),
                "relative": str(rel),
                "group": str(rel.parent) if len(rel.parts) > 1 else None,
            })
        for f in sorted(cdir.rglob("*.yml")):
            rel = f.relative_to(cdir)
            configs.append({
                "path": str(f.resolve()),
                "relative": str(rel),
                "group": str(rel.parent) if len(rel.parts) > 1 else None,
            })
    return configs


# ---------------------------------------------------------------------------
# Main scanner
# ---------------------------------------------------------------------------


def scan_project(project_dir: str) -> dict[str, Any]:
    """Scan a training project and return a structured ProjectInfo dict.

    Parameters
    ----------
    project_dir : str
        Path to the root of the training project.

    Returns
    -------
    dict
        A ``ProjectInfo`` dictionary with the following top-level keys:

        - ``project_dir``: absolute path to the scanned project
        - ``entry_points``: list of detected Hydra entry point files
        - ``hydra_entries``: list of ``@hydra.main()`` detections with config metadata
        - ``config_dirs``: list of detected Hydra config directories
        - ``config_files``: list of YAML config files found in config directories
        - ``log_dirs``: list of detected log directories
        - ``checkpoint_dirs``: list of detected checkpoint directories
        - ``output_dirs``: list of detected output directories
        - ``event_files``: list of TensorBoard event files
        - ``summary_writers``: list of SummaryWriter detections with log_dir references
    """
    root = Path(project_dir).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Project directory not found: {project_dir}")

    # Detect entry point scripts
    entry_candidates = _find_files_by_name(root, _ENTRY_PATTERNS)

    # Detect Hydra decorators in all Python files
    py_files = _find_python_files(root)
    hydra_entries: list[dict[str, Any]] = []
    for py_file in py_files:
        result = _detect_hydra_entry(py_file)
        if result is not None:
            hydra_entries.append(result)

    # Add Hydra-decorated files to entry point list if not already there
    hydra_files = {e["file"] for e in hydra_entries}
    for ep in entry_candidates:
        if str(ep.resolve()) not in hydra_files:
            pass  # keep as candidate
    entry_point_paths = list({str(p.resolve()) for p in entry_candidates} | hydra_files)

    # Detect config directories
    config_dirs = _find_dirs_by_name(root, _CONFIG_DIR_NAMES)
    # Also add directories referenced by Hydra entries
    for he in hydra_entries:
        cdir = he.get("config_dir_resolved")
        if cdir and Path(cdir).is_dir():
            cdir_path = Path(cdir)
            if cdir_path not in config_dirs:
                config_dirs.append(cdir_path)

    # Detect YAML configs
    config_files = _detect_yaml_configs(config_dirs)

    # Detect log directories
    log_dirs = _find_dirs_by_name(root, _LOG_DIR_NAMES)

    # Detect checkpoint directories
    ckpt_dirs = _find_dirs_by_name(root, _CKPT_DIR_NAMES)

    # Detect output directories
    output_dirs = _find_dirs_by_name(root, _OUTPUT_DIR_NAMES)

    # Detect TensorBoard event files
    event_files = _find_event_files(root)

    # Detect SummaryWriter usage
    sw_detections = _detect_summary_writer(py_files)

    return {
        "project_dir": str(root),
        "entry_points": sorted(entry_point_paths),
        "hydra_entries": hydra_entries,
        "config_dirs": sorted(str(d.resolve()) for d in config_dirs),
        "config_files": config_files,
        "log_dirs": sorted(str(d.resolve()) for d in log_dirs),
        "checkpoint_dirs": sorted(str(d.resolve()) for d in ckpt_dirs),
        "output_dirs": sorted(str(d.resolve()) for d in output_dirs),
        "event_files": sorted(str(f.resolve()) for f in event_files),
        "summary_writers": sw_detections,
        "summary": {
            "num_entry_points": len(entry_point_paths),
            "num_hydra_entries": len(hydra_entries),
            "num_config_dirs": len(config_dirs),
            "num_config_files": len(config_files),
            "num_event_files": len(event_files),
            "has_tensorboard": len(sw_detections) > 0 or len(event_files) > 0,
        },
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Scan a training project directory and output a structured JSON "
            "description of its components: entry points, config directories, "
            "log/checkpoint paths, and TensorBoard event files."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--project", "-p",
        required=True,
        metavar="DIR",
        help="Path to the root of the training project to scan.",
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        metavar="FILE",
        help="Optional output file path. If omitted, prints to stdout.",
    )
    args = parser.parse_args()

    try:
        info = scan_project(args.project)
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    json_str = json.dumps(info, indent=2, ensure_ascii=False)

    if args.output:
        Path(args.output).write_text(json_str, encoding="utf-8")
        print(f"Project info written to {args.output}", file=sys.stderr)
    else:
        print(json_str)


if __name__ == "__main__":
    main()
