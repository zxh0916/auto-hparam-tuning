from __future__ import annotations

import json
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any


# ---------------------------------------------------------------------------
# Remote helper script (sent verbatim to the remote Python interpreter).
# Handles all actions used by both session_manager and project_understanding.
# ---------------------------------------------------------------------------

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
elif action == "glob":
    root = Path(payload["root"])
    patterns = payload["patterns"]
    matches = []
    for pattern in patterns:
        matches.extend(str(p) for p in root.glob(pattern))
    matches = sorted(dict.fromkeys(matches))
    dump({"matches": matches})
else:
    raise ValueError(f"Unsupported remote action: {action}")
'''


# ---------------------------------------------------------------------------
# Shared error type
# ---------------------------------------------------------------------------

class StorageError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# Target specification
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Storage abstraction
# ---------------------------------------------------------------------------

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

    def glob(self, root: str, patterns: list[str]) -> list[str]:
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

    def glob(self, root: str, patterns: list[str]) -> list[str]:
        from pathlib import Path

        base = Path(root)
        matches: list[str] = []
        for pattern in patterns:
            matches.extend(str(p) for p in base.glob(pattern))
        return sorted(dict.fromkeys(matches))


class SSHStorage(Storage):
    def __init__(self, host: str):
        self.host = host

    def _call(self, payload: dict[str, Any]) -> dict[str, Any]:
        remote_cmd = f"python3 -c {shlex.quote(REMOTE_HELPER)}"
        command = ["ssh", self.host, remote_cmd]
        proc = subprocess.run(
            command,
            input=json.dumps(payload, ensure_ascii=False),
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            raise StorageError(
                f"Remote command failed on {self.host}: {proc.stderr.strip() or proc.stdout.strip()}"
            )
        output = proc.stdout.strip()
        if not output:
            return {}
        try:
            return json.loads(output)
        except json.JSONDecodeError:
            lines = [line.strip() for line in output.splitlines() if line.strip()]
            for line in reversed(lines):
                try:
                    return json.loads(line)
                except json.JSONDecodeError:
                    continue
            raise StorageError(f"Failed to decode remote JSON response: {output!r}")

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

    def glob(self, root: str, patterns: list[str]) -> list[str]:
        return list(self._call({"action": "glob", "root": root, "patterns": patterns})["matches"])


# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------

def default_storage(target: TargetSpec) -> Storage:
    return SSHStorage(target.ssh_host) if target.is_remote else LocalStorage()


def join(root: str, *parts: str) -> str:
    return str(PurePosixPath(root, *parts))

def get_sessions_spawn_command(label: str, task: str):
    args = {
        "task": task,
        "runtime": "subagent",
        "label": label,
        "agentId": "your agentId",
        "mode": "run",
        "cleanup": "delete",
        "thinking": "low"
    }
    command = "sessions_spawn("
    for i, (k, v) in enumerate(args.items()):
        command = command + f"{k}=\"{v}\""
        if i != len(args) - 1:
            command = command + ", "
    command = command + ")"
    return command