Generate a file named PROJECT.md that serves as an agent onboarding guide for the target project repository, and place the PROJECT.md under the root of the project repo (copy to the remote if exploring a project on a remote machine). Use English.

Your goal is to produce a clear, concise, and high-value project guide for future coding agents. Prefer concrete repository-specific facts backed by files you actually inspected. Keep the document concise, but cover the parts that matter for understanding, running, and modifying the project.

Document Requirements

- Title the document "Project Overview".
- Use Markdown headings (#, ##, etc.) for structure.
- Keep the document concise. 300-700 words is usually enough.
- Keep explanations short, direct, and specific to this project.
- Provide examples where helpful (commands, directory paths, naming patterns).
- Maintain a professional, instructional tone.
- If the project contains a nontrivial Hydra/OmegaConf/config system, mention the key config roots and point future agents to inspect the hyperparameter/config prompt output as well.

Recommended Sections

## Project Structure & Module Organization
- Outline the project structure, including where the source code, scripts, configs, tests, and experiment assets are located.
- Mention the key files and directories that control training or evaluation.

## Build, Test, and Development Commands
- List the main commands for setup, training, evaluation, and testing.
- Briefly explain what each command does.
- If commands vary by environment (local vs remote), mention that.

## Runtime Entry Points
- Identify the main executable entrypoints (training, evaluation, data prep, launch scripts).
- Mention where configuration enters runtime.

## Data, Outputs, and Artifacts
- State where logs, checkpoints, TensorBoard files, and generated artifacts are written.
- Mention any cleanup or large-file considerations.

## Coding / Editing Guidance for Agents
- Note anything future agents should inspect before making changes.
- Highlight any configuration, launcher, experiment, or orchestration files that should be read before editing code.

## Commit / Workflow Notes
- Summarize any repo-specific workflow constraints if they are visible from docs or existing files.

When present, prioritize facts from project docs such as `CLAUDE.md`, `AGENTS.md`, `README*`, and visible config / launch files over generic guesses.

De-emphasize:
- exhaustive directory listings
- generic style advice not grounded in the repository
- boilerplate PR advice unless the repo explicitly uses it
