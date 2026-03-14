from __future__ import annotations

import argparse
import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any


REMOTE_HELPER = r'''
import json
import sys
from pathlib import Path

payload = json.load(sys.stdin)
action = payload["action"]

if action == "exists":
    path = Path(payload["path"])
    sys.stdout.write(json.dumps({"exists": path.exists()}, ensure_ascii=False))
elif action == "glob":
    root = Path(payload["root"])
    patterns = payload["patterns"]
    matches = []
    for pattern in patterns:
        matches.extend(str(p) for p in root.glob(pattern))
    matches = sorted(dict.fromkeys(matches))
    sys.stdout.write(json.dumps({"matches": matches}, ensure_ascii=False))
elif action == "read_text":
    path = Path(payload["path"])
    sys.stdout.write(json.dumps({"text": path.read_text(encoding="utf-8")}, ensure_ascii=False))
else:
    raise ValueError(f"Unsupported action: {action}")
'''


class ProjectUnderstandingError(RuntimeError):
    pass


@dataclass(frozen=True)
class StorageTarget:
    project_root: str
    ssh_host: str | None = None

    @property
    def is_remote(self) -> bool:
        return self.ssh_host is not None


class Storage:
    def exists(self, path: str) -> bool:
        raise NotImplementedError

    def glob(self, root: str, patterns: list[str]) -> list[str]:
        raise NotImplementedError

    def read_text(self, path: str) -> str:
        raise NotImplementedError


class LocalStorage(Storage):
    def exists(self, path: str) -> bool:
        from pathlib import Path

        return Path(path).exists()

    def glob(self, root: str, patterns: list[str]) -> list[str]:
        from pathlib import Path

        base = Path(root)
        matches: list[str] = []
        for pattern in patterns:
            matches.extend(str(p) for p in base.glob(pattern))
        return sorted(dict.fromkeys(matches))

    def read_text(self, path: str) -> str:
        from pathlib import Path

        return Path(path).read_text(encoding="utf-8")


class SSHStorage(Storage):
    def __init__(self, host: str):
        self.host = host

    def _call(self, payload: dict[str, Any]) -> dict[str, Any]:
        proc = subprocess.run(
            ["ssh", self.host, "python3", "-c", REMOTE_HELPER],
            input=json.dumps(payload, ensure_ascii=False),
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            raise ProjectUnderstandingError(
                f"Remote command failed on {self.host}: {proc.stderr.strip() or proc.stdout.strip()}"
            )
        return json.loads(proc.stdout or "{}")

    def exists(self, path: str) -> bool:
        return bool(self._call({"action": "exists", "path": path})["exists"])

    def glob(self, root: str, patterns: list[str]) -> list[str]:
        return list(self._call({"action": "glob", "root": root, "patterns": patterns})["matches"])

    def read_text(self, path: str) -> str:
        return self._call({"action": "read_text", "path": path})["text"]


PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"


def _storage(target: StorageTarget) -> Storage:
    return SSHStorage(target.ssh_host) if target.is_remote else LocalStorage()


def _join(root: str, *parts: str) -> str:
    return str(PurePosixPath(root, *parts))


def inspect_project(project_root: str, ssh_host: str | None = None) -> dict[str, Any]:
    target = StorageTarget(project_root=project_root, ssh_host=ssh_host)
    storage = _storage(target)

    candidates = {
        "claude_md": _join(project_root, "CLAUDE.md"),
        "agents_md": _join(project_root, "AGENTS.md"),
        "agent_md": _join(project_root, "AGENT.md"),
        "project_md": _join(project_root, "aht", "PROJECT.md"),
        "hparam_md": _join(project_root, "aht", "HPARAM.md"),
    }
    docs = {
        name: {"path": path, "exists": storage.exists(path)}
        for name, path in candidates.items()
    }

    read_order = [key for key in ["claude_md", "agents_md", "agent_md", "project_md", "hparam_md"] if docs[key]["exists"]]
    need_generate_project_md = not any(docs[key]["exists"] for key in ["claude_md", "agents_md", "agent_md", "project_md"])
    need_generate_hparam_md = not docs["hparam_md"]["exists"]

    return {
        "project_root": project_root,
        "storage": "ssh" if ssh_host else "local",
        "ssh_host": ssh_host,
        "docs": docs,
        "read_order": read_order,
        "need_generate_project_md": need_generate_project_md,
        "need_generate_hparam_md": need_generate_hparam_md,
        "prompts": {
            "generate_project_md": str(PROMPTS_DIR / "generate_project_md.md"),
            "get_hparam_structure": str(PROMPTS_DIR / "get_hparam_structure.md"),
            "understand_run_command": str(PROMPTS_DIR / "understand_run_command.md"),
        },
        "next_steps": [
            "If need_generate_project_md is true, use sessions_spawn with prompts/generate_project_md.md to create PROJECT.md in the aht/ folder (create it if needed).",
            "If need_generate_hparam_md is true, use the hyperparameter-structure prompt to create HPARAM.md in the aht/ folder.",
            "Otherwise, read existing project-level docs in the order specified by read_order.",
            "Then run the command-specific understanding prompt for the current session.",
        ],
    }


def prepare_run_understanding(project_root: str, run_command: str, ssh_host: str | None = None) -> dict[str, Any]:
    inspection = inspect_project(project_root=project_root, ssh_host=ssh_host)
    available_context = [
        info["path"] for info in inspection["docs"].values() if info["exists"]
    ]
    return {
        **inspection,
        "run_command": run_command,
        "available_context_files": available_context,
        "required_prompt": inspection["prompts"]["understand_run_command"],
        "execution_outline": [
            "Ensure project-level docs and HPARAM.md exist in the aht/ directory or are generated first.",
            "Read the available context files listed above.",
            "Run prompts/understand_run_command.md with the given project root and run command.",
            "Save the resulting note into the active AHT session as a session-specific understanding artifact.",
        ],
    }


# ---------------------------------------------------------------------------
# detect_run_command  – heuristic auto-discovery of the training entry point
# ---------------------------------------------------------------------------

_ENTRY_NAME_PATTERNS = [
    r"^train\.py$",
    r"^main\.py$",
    r"^run\.py$",
    r"^run_training\.py$",
    r"^train_.*\.py$",
    r"^main_.*\.py$",
]

_HYDRA_IMPORT_RE = re.compile(r"hydra", re.IGNORECASE)
_HYDRA_DECORATOR_RE = re.compile(r"@hydra\.main|hydra\.initialize|hydra\.compose", re.IGNORECASE)
_CONFIG_NAME_RE = re.compile(r'config_name\s*=\s*["\']([^"\']+)["\']')
_CONFIG_PATH_RE = re.compile(r'config_path\s*=\s*["\']([^"\']+)["\']')


def _score_entry(rel_path: str, content: str) -> int:
    """Return a confidence score for this file being the main training entrypoint."""
    score = 0
    name = Path(rel_path).name.lower()
    for pat in _ENTRY_NAME_PATTERNS:
        if re.match(pat, name):
            score += 30
            break
    # Reward files closer to the project root (fewer path segments)
    depth = rel_path.replace("\\", "/").count("/")
    score += max(0, 20 - depth * 5)
    if _HYDRA_DECORATOR_RE.search(content):
        score += 50
    elif _HYDRA_IMPORT_RE.search(content):
        score += 20
    if "if __name__" in content:
        score += 10
    return score


def _suggest_command(project_root: str, rel_path: str, content: str) -> str:
    """Build a suggested run command from the entry script and its Hydra config hints."""
    # Normalise to forward-slashes for the shell command
    script = rel_path.replace("\\", "/")
    # Try to find logger config-group override hint
    config_name_m = _CONFIG_NAME_RE.search(content)
    config_name = config_name_m.group(1) if config_name_m else None

    cmd = f"python {script}"
    # Almost all hydra+tb projects want `logger=tensorboard` when logging
    cmd += " logger=tensorboard"
    return cmd


def detect_run_command(project_root: str, ssh_host: str | None = None) -> dict[str, Any]:
    """Scan a project directory to auto-detect the Hydra training entry point and
    suggest a BASE_COMMAND, so the user doesn't need to specify one manually.

    Returns a dict with:
      - candidates: list of {rel_path, score, suggested_command, config_name, config_path}
      - best_command: the top-ranked suggested command
      - best_script: relative path of the top-ranked entry script
      - confidence: "high" | "medium" | "low"
      - note: human-readable summary
    """
    storage = _storage(StorageTarget(project_root=project_root, ssh_host=ssh_host))
    # Glob for all .py files (depth-limited to avoid digging into venvs/node_modules)
    all_py = storage.glob(project_root, ["*.py", "*/*.py", "*/*/*.py"])

    candidates = []
    for abs_path in all_py:
        # Skip obvious non-entrypoints
        rel = abs_path[len(project_root):].lstrip("/\\") if abs_path.startswith(project_root) else abs_path
        lower = rel.lower().replace("\\", "/")
        if any(skip in lower for skip in ["test", "setup.py", "conf", "__init__", ".egg"]):
            continue
        try:
            content = storage.read_text(abs_path)
        except Exception:
            continue
        score = _score_entry(rel, content)
        if score < 20:
            continue
        config_name_m = _CONFIG_NAME_RE.search(content)
        config_path_m = _CONFIG_PATH_RE.search(content)
        candidates.append({
            "rel_path": rel,
            "score": score,
            "suggested_command": _suggest_command(project_root, rel, content),
            "config_name": config_name_m.group(1) if config_name_m else None,
            "config_path": config_path_m.group(1) if config_path_m else None,
        })

    candidates.sort(key=lambda c: c["score"], reverse=True)

    if not candidates:
        return {
            "project_root": project_root,
            "candidates": [],
            "best_command": None,
            "best_script": None,
            "confidence": "low",
            "note": "No Python entry point with Hydra usage detected. Please specify BASE_COMMAND manually.",
        }

    best = candidates[0]
    confidence = "high" if best["score"] >= 70 else ("medium" if best["score"] >= 40 else "low")
    return {
        "project_root": project_root,
        "candidates": candidates[:5],  # top-5
        "best_command": best["suggested_command"],
        "best_script": best["rel_path"],
        "confidence": confidence,
        "note": (
            f"Auto-detected entry point: '{best['rel_path']}' (score={best['score']}, confidence={confidence}). "
            f"Suggested command: {best['suggested_command']!r}. "
            "Review candidates list and override if needed."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare the code-enforced project-understanding flow for AHT.",
    )
    parser.add_argument("--ssh-host", default=None, help="Optional SSH host. When set, inspect the remote project filesystem.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_inspect = subparsers.add_parser("inspect-project", help="Check for project-level docs and decide what must be generated.")
    p_inspect.add_argument("project_root")

    p_prepare = subparsers.add_parser("prepare-run-understanding", help="Prepare the command-aware understanding workflow for a run command.")
    p_prepare.add_argument("project_root")
    p_prepare.add_argument("run_command")

    p_detect = subparsers.add_parser(
        "detect-run-command",
        help="Auto-detect the Hydra training entry point and suggest a BASE_COMMAND.",
    )
    p_detect.add_argument("project_root")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "inspect-project":
        result = inspect_project(project_root=args.project_root, ssh_host=args.ssh_host)
    elif args.command == "prepare-run-understanding":
        result = prepare_run_understanding(
            project_root=args.project_root,
            run_command=args.run_command,
            ssh_host=args.ssh_host,
        )
    elif args.command == "detect-run-command":
        result = detect_run_command(project_root=args.project_root, ssh_host=args.ssh_host)
    else:
        raise ProjectUnderstandingError(f"Unsupported command: {args.command}")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
