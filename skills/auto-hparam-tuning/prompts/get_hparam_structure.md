Generate a file named HPARAM.md that serves as an agent operating guide for the project repository, and place the HPARAM.md under the root of the project repo (copy to the remote if exploring project in a remote machine). Use English.

Primary goal:
Produce a concise but high-value guide that helps future coding agents quickly understand how this project repository is configured, run, and experimentally controlled. If the project repository uses Hydra, OmegaConf, or other layered config systems, prioritize documenting the hyperparameter/configuration system over generic project repository tour content.

Document requirements:
- Title the document "Project Hyperparameter Structure".
- Use Markdown headings.
- Target 500-900 words if the repo contains a nontrivial config system; otherwise stay concise.
- Prefer concrete, project repository-specific facts backed by files you actually inspected.
- Include short command examples and exact paths where helpful.
- Avoid generic engineering advice.

Required sections:
1. Runtime Entry Points
   - Identify the main training / evaluation / launch entrypoints (e.g. `scripts/train.py::main()`).
   - State where Hydra or other config frameworks are initialized (e.g. @hydra.main, compose()).

2. Configuration Topology
   - Describe where the primary config root lives (e.g. conf/, configs/).
   - Explain the main config groups and what they control (model, dataset, optimizer, trainer, env, logging, launch, sweep).
   - Describe how defaults are composed and which files act as the top-level composition roots.

3. TensorBoard Integration
   - Check if the project uses TensorBoard SummaryWriter to log the scalar metrics with `grep -RIn "SummaryWriter" <project path>`.
   - Find the directory that SummaryWriter uses to log event files (e.g. `SummaryWriter(log_dir=<the path>, run_name))`)

4. Override Semantics
   - Explain how common hyperparameters are overridden from the CLI.
   - Distinguish overriding config values from switching config-group choices.
   - Include 3-6 repo-specific override examples.

5. Sweep / Experiment Orchestration
   - If Hydra multirun, sweeper, launcher, Optuna, Ax, Ray, Submitit, Slurm, or similar are used, explain where they are configured and how search spaces are declared.
   - Show the canonical command for running a sweep.

6. Config Debugging and Inspection
   - Include commands to print or inspect the resolved config (for Hydra, prefer --cfg / --resolve / --info when applicable).
   - State where past run configs / override logs are stored.

7. Agent Workflow for Config-Related Tasks
   - Instruct future agents that when a task touches training behavior, experiments, tuning, or reproducibility, they must inspect the config entrypoint, top-level defaults, relevant config groups, and any sweep / launcher configs before editing code.

When Hydra is present, spend less space on code style and prioritize documenting the configuration and hyperparameter system over generic project repository structure:
- config roots
- defaults composition
- experiment aliases
- launch/sweep plugins
- resolved-config inspection
- prior-run recovery paths

You must explain:
- where configuration enters runtime,
- where is the tensorboard event file is placed,
- how top-level defaults / config groups compose the final config,
- how common overrides and sweeps are expressed,
- and how to inspect the resolved config and previous run overrides.

For config-related tasks, future agents must inspect the config composition chain before changing code.

Optional sections:
- Project Structure & Module Organization (only if it helps explain config ownership)
- Build / Test commands
- Architecture overview
- Safety / secrets notes

De-emphasize:
- exhaustive directory listings
- generic coding style advice
- generic PR guidance unless it materially affects experiment workflow