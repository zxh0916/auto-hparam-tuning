from typing import Any, Literal, Optional
import json

def spawn_subagent(
    agent: str,
    label: str,
    task: str,
    model: Optional[str] = None
):
    assert agent in ["codex", "claudecode", "openclaw"]
    if agent == "codex":
        return ""
    elif agent == "claudecode":
        return ""
    else:
        prefix = (
            "Spawn a subagent with `sessions_spawn` tool with following args:\n"
        )
        args = {
            "task": task,
            "runtime": "subagent",
            "label": label,
            "agentId": "your agentId (use `agents_list` tool to acquire)",
            "mode": "run",
            "cleanup": "delete",
            "thinking": "low"
        }
        if model is not None:
            args["model"] = str(model)
        return prefix + json.dumps(args, ensure_ascii=False)

def add_cron_job(
    agent: str,
    name: str,
    at: str,
    payload: str
):
    assert agent in ["codex", "claudecode", "openclaw"]
    if agent == "codex":
        return ""
    elif agent == "claudecode":
        return ""
    else:
        prefix = "Create a cron job with openclaw with following command: "
        args = {
            "name": name,
            "at": at,
            "session": "main",
            "session-key": "use \"session_status\" tool to acquire",
            "wake": "now",
            "system-event": payload
        }
        command = "`openclaw cron add "
        for k, v in args.items():
            if k in ["name", "at", "system-event"] :
                command = command + f" --{k} \"{v}\""
            else:
                command = command + f" --{k} {v}"
        command = prefix + command + " --delete-after-run`"
        return command
