from __future__ import annotations

import json
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import PurePosixPath, Path
from typing import Any, Literal
import csv
from datetime import datetime
from io import StringIO

from omegaconf import OmegaConf
import pandas as pd

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


def ensure_override_in_defaults(storage: Storage, primary_config_path: str) -> bool:
    """Ensure ``- override`` appears in the ``defaults`` list after ``- _self_``.

    Three cases handled:
    - ``defaults`` block exists and already has ``- override``: no-op.
    - ``defaults`` block exists with ``- _self_``: inserts ``- override`` after it.
    - No ``defaults`` block at all: appends one at the end of the file with
      ``- _self_`` and ``- override``.

    The file is rewritten in-place with original formatting preserved.
    Returns True if the file was modified.
    """
    if not storage.exists(primary_config_path):
        return False

    text = storage.read_text(primary_config_path)
    lines = text.splitlines(keepends=True)

    in_defaults = False
    defaults_found = False
    defaults_indent: int | None = None
    self_line_idx: int | None = None
    override_found = False

    for i, line in enumerate(lines):
        stripped = line.strip()
        if not in_defaults:
            if stripped == "defaults:" or stripped.startswith("defaults:"):
                in_defaults = True
                defaults_found = True
            continue
        # Inside defaults block: zero-indent non-list line → left the block
        indent = len(line) - len(line.lstrip())
        if stripped and not stripped.startswith("-") and not stripped.startswith("#"):
            if indent == 0:
                break
        if stripped.startswith("-"):
            if defaults_indent is None:
                defaults_indent = indent
            item = stripped[1:].strip()
            if item == "_self_":
                self_line_idx = i
            elif item == "override":
                override_found = True
                break

    if override_found:
        return False

    if not defaults_found:
        # No defaults block at all — append one at the end
        suffix = "" if text.endswith("\n") else "\n"
        storage.write_text(primary_config_path, text + suffix + "defaults:\n  - _self_\n  - override\n")
        return True

    # defaults block exists but override is missing
    indent_str = " " * (defaults_indent if defaults_indent is not None else 2)
    if self_line_idx is not None:
        lines.insert(self_line_idx + 1, f"{indent_str}- override\n")
    else:
        # _self_ not found — insert override at end of defaults block
        # Find the last list entry in the block
        last_list_idx = None
        in_defaults2 = False
        for i, line in enumerate(lines):
            stripped = line.strip()
            if not in_defaults2:
                if stripped == "defaults:" or stripped.startswith("defaults:"):
                    in_defaults2 = True
                continue
            indent = len(line) - len(line.lstrip())
            if stripped and not stripped.startswith("-") and not stripped.startswith("#"):
                if indent == 0:
                    break
            if stripped.startswith("-"):
                last_list_idx = i
        insert_at = (last_list_idx + 1) if last_list_idx is not None else len(lines)
        lines.insert(insert_at, f"{indent_str}- override\n")

    storage.write_text(primary_config_path, "".join(lines))
    return True


def stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)

def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")

def load_results_dataframe(storage: Storage, results_path: str, columns: list[str]) -> pd.DataFrame:
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

def write_results_dataframe(storage: Storage, results_path: str, df: pd.DataFrame, columns: list[str]) -> None:
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
    writer.writerows(normalized.to_dict(orient="records"))
    storage.write_text(results_path, buffer.getvalue())

def upsert_results_row(storage: Storage, results_path: str, columns: list[str], row: dict[str, Any]) -> dict[str, str]:
    normalized_row = {col: stringify(row.get(col, "")) for col in columns}
    df = load_results_dataframe(storage, results_path, columns)
    key = normalized_row["run_id"]
    if df.empty:
        df = pd.DataFrame([normalized_row], columns=columns)
    else:
        mask = df["run_id"].astype(str) == key
        if mask.any():
            for col in columns:
                df.loc[mask, col] = normalized_row[col]
        else:
            df = pd.concat([df, pd.DataFrame([normalized_row], columns=columns)], ignore_index=True)
    write_results_dataframe(storage, results_path, df, columns)
    return normalized_row

def safe_int(value: Any) -> int | None:
    text = str(value).strip()
    if not text:
        return None
    try:
        return int(float(text))
    except ValueError:
        return None

def safe_float(value: Any) -> float | None:
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None

def next_step_postfix() -> str:
    return (
        " There is no need to ask the user for the next decision. Just follow the instruction in \"next_step\" " +
        "and continue the tuning process until there's an error. " +
        "If there's no error, please strictly follow the instructions in \"next_step\". " +
        "Do not reply with any text, just continue using tools and finish your job."
    )
