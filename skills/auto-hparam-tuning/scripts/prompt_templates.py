from typing import Any, Literal, Optional
import json

def spawn_subagent(
    description: str,
    agent: str,
    label: str,
    task: str,
    type: str = "general-purpose",
    model: Optional[str] = None
):
    assert agent in ["codex", "claudecode", "openclaw"]
    if agent == "codex":
        return ""
    elif agent == "claudecode":
        if type == "Explore":
            task = task + (
                "You may not have the permission to edit file in disk. " +
                "Just return the content to be written to the caller agent and ask it to edit for you."
            )
        prefix = (
            f"Spawn a subagent with `Agent` tool with following args to {description}:\n"
        )
        args = {
            "description": description,
            "subagent_type": type,
            "prompt": task,
            "run_in_background": False
        }
        if model is not None:
            args["model"] = str(model)
        return prefix + json.dumps(args, ensure_ascii=False)
    else:
        prefix = (
            f"Spawn a subagent with `sessions_spawn` tool with following args to {description}:\n"
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
    payload: str,
    eta_command: str
):
    assert agent in ["codex", "claudecode", "openclaw"]
    if agent == "codex":
        return ""
    elif agent == "claudecode":
        prefix = "Create a cron job with `CronCreate` tool with following args:\n"
        args = {
            "cron": f"$({eta_command} --cron <duration>)",
            "prompt": payload,
            "recurring": False
        }
        return prefix + json.dumps(args, ensure_ascii=False)
    else:
        prefix = "Create a cron job with openclaw with following command: "
        args = {
            "name": name,
            "at": f"$({eta_command} --iso8601 <duration>)",
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
