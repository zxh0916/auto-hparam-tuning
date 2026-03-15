from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from utils import (
    TargetSpec,
    Storage,
    LocalStorage,
    SSHStorage,
    default_storage,
    join as _join,
    get_sessions_spawn_command
)



class ProjectUnderstandingError(RuntimeError):
    pass


PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"


class ProjectInspector:
    def __init__(self, project_root: str, ssh_host: str | None = None):
        self.project_root = project_root
        self.ssh_host = ssh_host
        self.target = TargetSpec(project_root=project_root, ssh_host=ssh_host)
        self.storage = default_storage(self.target)
        self.project_candidates = {
            "claude_md": _join(project_root, "CLAUDE.md"),
            "agents_md": _join(project_root, "AGENTS.md"),
            "agent_md": _join(project_root, "AGENT.md"),
            "project_md": _join(project_root, "PROJECT.md"),
        }
        self.hparam_candidates = {
            "hparam_md": _join(project_root, "HPARAM.md"),
        }
        self.docs = {
            name: {"path": path, "exists": self.storage.exists(path)}
            for name, path in {**self.project_candidates, **self.hparam_candidates}.items()
        }

        self.read_order = [key for key in ["claude_md", "agents_md", "agent_md", "project_md", "hparam_md"] if self.docs[key]["exists"]]
        self.need_generate_project_md = not any(self.docs[key]["exists"] for key in ["claude_md", "agents_md", "agent_md", "project_md"])
        self.need_generate_hparam_md = not self.docs["hparam_md"]["exists"]
    
    def inspect_project(self) -> dict[str, Any]:
        next_steps = []
        
        if self.need_generate_project_md:
            project_md_path = _join(self.project_root, "PROJECT.md")
            self.storage.write_text(project_md_path, "")
            next_steps.append(
                "Spawn a subagent to walk through the project with following command: " + 
                get_sessions_spawn_command(
                    label="project_walkthrough_summary_generation",
                    task=(
                        f"Read and follow the instruction in {str(PROMPTS_DIR / "generate_project_md.md")} "+
                        f"and write your summary in {project_md_path} "+
                        (f" in remote host {self.ssh_host}." if self.ssh_host is not None else ".")
                    )
                )
            )
        else:
            project_existance = [v for v in self.project_candidates.values() if self.storage.exists(v)]
            assert len(self.project_candidates) > 0
            project_md_path = project_existance[0]
        
        hparam_md_path = _join(self.project_root, "HPARAM.md")
        if self.need_generate_hparam_md:
            self.storage.write_text(hparam_md_path, "")
            next_steps.append(
                "Spawn a subagent to explore the hyperparameter structure of the project with following command: " + 
                get_sessions_spawn_command(
                    label="hyperparameter_sturcture_summary_generation",
                    task=(
                        f"Read and follow the instruction in {str(PROMPTS_DIR / "get_hparam_structure.md")} "+
                        f"and write your summary in {hparam_md_path} "+
                        (f" in remote host {self.ssh_host}." if self.ssh_host is not None else ".")
                    )
                )
            )
        
        if (not self.need_generate_project_md) and (not self.need_generate_hparam_md):
            next_steps.append(
                "Both project summary and hparam structure exist. "
                f"Now, read the project summary in {str(PROMPTS_DIR / "generate_project_md.md")} "
                f"and the hparam structure in {str(PROMPTS_DIR / "get_hparam_structure.md")} "
                "then run `python scripts/project_understanding.py prepare-run-understanding <command>` "
                "to understant the task to be tuned."
            )
        else:
            next_steps.append(
                "After spawned the subagent, DO NOTHING until the subagent returns. "
                "Then, proceed with `python scripts/project_understanding.py prepare-run-understanding <command>`."
            )
        
        return {
            "project_root": self.project_root,
            "storage": "ssh" if self.ssh_host else "local",
            "ssh_host": self.ssh_host,
            "docs": self.docs,
            "read_order": self.read_order,
            "need_generate_project_md": self.need_generate_project_md,
            "need_generate_hparam_md": self.need_generate_hparam_md,
            "next_steps": next_steps,
        }

    def prepare_run_understanding(self, run_command: str) -> dict[str, Any]:
        assert (not self.need_generate_project_md) and (not self.need_generate_hparam_md), \
            "One of project summary and hparam structure is not found. Please run `python scripts/project_understanding.py inspect-project` first."
        available_context = [
            info["path"] for info in self.docs.values() if info["exists"]
        ]
        return {
            "run_command": run_command,
            "available_context_files": available_context,
            "required_prompt": str(PROMPTS_DIR / "understand_run_command.md"),
            "next_steps": [
                "Read the available context files listed above.",
                "Run prompts/understand_run_command.md with the given project root and run command.",
                "Save the resulting note into report.md after session creation using ",
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
    pi = ProjectInspector(project_root=args.project_root, ssh_host=args.ssh_host)
    if args.command == "inspect-project":
        result = pi.inspect_project()
    elif args.command == "prepare-run-understanding":
        result = pi.prepare_run_understanding(
            run_command=args.run_command
        )
    else:
        raise ProjectUnderstandingError(f"Unsupported command: {args.command}")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
