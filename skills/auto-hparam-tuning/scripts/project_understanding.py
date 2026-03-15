from __future__ import annotations

import argparse
import base64
import json
import shlex
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
        helper_b64 = base64.b64encode(REMOTE_HELPER.encode("utf-8")).decode("ascii")
        remote_python = f"import base64; exec(base64.b64decode({helper_b64!r}).decode('utf-8'))"
        proc = subprocess.run(
            ["ssh", self.host, f"python3 -c {shlex.quote(remote_python)}"],
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
        "project_md": _join(project_root, "PROJECT.md"),
        "hparam_md": _join(project_root, "HPARAM.md"),
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
            "Read existing project-level docs in read_order.",
            "If need_generate_project_md is true, use sessions_spawn with prompts/generate_project_md.md to create PROJECT.md at the target project root.",
            "If need_generate_hparam_md is true, use the hyperparameter-structure prompt to create HPARAM.md.",
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
            "Ensure project-level docs and HPARAM.md exist or are generated first.",
            "Read the available context files listed above.",
            "Run prompts/understand_run_command.md with the given project root and run command.",
            "Save the resulting note into report.md after session creation using",
            "`python scripts/session_manager.py --ssh-host user@remotehost <path to project root> append-report <your understanding>`.",
        ],
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
    else:
        raise ProjectUnderstandingError(f"Unsupported command: {args.command}")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
