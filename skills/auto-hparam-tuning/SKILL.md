---
name: auto-hparam-tuning
description: Understand and automatically tune the hyperparameters of a project that uses hydra, with respect to the specified metric(s).
---

# Automatic Hyperparameter Tuning

Automatically tune the hyperparameteres of a learning process: fetch results from tensorboard, analyze with pandas and numpy, then tune with hydra.

Language policy:
- This skill is authored in English.
- All user-facing feedback and messages must use the same language as the user.
- Internal reasoning/thinking may use any language and is not restricted.

## Overview

This skill will automatically the hyperparameters managed by hydra config system to optimize a learning process. Given a project with hydra-based hyperparameter structure, this skill first walk through the project and detect the entry script and function along with the main config file. Then, it triggers a test run with command specified by the user to get the metric keys from the tensorboard event file. After the test run, with respect to a major metric specified by the user or automatically selected by the agent, the skill recognizes which subset of the hyperparameters are related to the trend of the major metric and analyze their specific influence by both reading the code and watch the pattern as the agent changes them. After each run, the agent should log the result, the analysis, and the applied changes in a report. The agent should repeat this [...->run->analyze->log->tune->run->analyze->log->tune->...] loop for a user-specified times and then report the final result.

## Workflow

**IMPORTANT**: Always use provided scripts in scripts/, DO NOT run commands manually.

0. Run the `aht-init` sub-skill first.
   - Collect and normalize the minimum required inputs: project path, conda env, reference training launch script/method, and optimization target.
   - Reuse anything already provided in the current conversation/context as provisional values.
   - Always send a user-facing confirmation message covering all four fields, even if they appear inferable.
   - Stop and wait for the user's confirmation or corrections.
   - Only continue into the workflow below after the initialization info has been explicitly confirmed.

1. Walkthrough the project on the remote machine and analyze the hparam structure.

2. Given a specified running command, identify the task and related hyperparameters.

3. Start a test run and identify the major metric and other auxiliary useful metrics from the event file.

4. Loop for a specified time:

    1. Launch a test run.

    2. After the run complete, copy back the event file and the config file.

    3. Analyze the event file using `scripts/analyze_event.py`.

    4. Record the result of the run in a table and a report and analyze the pattern that lies between the runs. Use `scripts/session_manager.py` to create and maintain the canonical `aht/yyyy-mm-dd/hh-mm-ss/` directory under the target project, not under the skill repo.

    5. Come up with tuning suggestions, review those suggestions, and pick a best one.

    6. Formalize the suggestion into a `override.yaml`, upload into the remote directory that contains the main config file, then back to 1. to launch another run.


## Understand the Project and Create the Session

Before tuning the hparam of the project, you should always make sure that the `aht-init` step has already produced and the user has explicitly confirmed these four fields: project path, conda env name, reference training launch method/script, and optimization target.

## Pipeline Algorithm

resolve SKILL_DIR = absolute path to this SKILL.md's parent directory
resolve SM = python {SKILL_DIR}/scripts/session_manager.py[ --ssh-host user@remotehost]

### 1. UNDERSTAND PROJECT:
    a. python {SKILL_DIR}/scripts/project_understanding.py[ --ssh-host user@remotehost] inspect-project {PROJECT_DIR}
        → tells you which docs exist, what needs to be generated, and which prompts to use
    b. Follow {SKILL_DIR}/prompts/generate_project_md.md if PROJECT.md is missing
        → creates `{PROJECT_DIR}/PROJECT.md` (general project onboarding guide)
    c. Follow {SKILL_DIR}/prompts/get_hparam_structure.md if HPARAM.md is missing
        → creates `{PROJECT_DIR}/HPARAM.md` (Hydra config and hparam guide)

### 2. UNDERSTAND RUN COMMAND:
    a. python {SKILL_DIR}/scripts/project_understanding.py[ --ssh-host user@remotehost] prepare-run-understanding
        {PROJECT_DIR} "{BASE_COMMAND}"
        → returns required_prompt and available_context_files
    b. Follow {SKILL_DIR}/prompts/understand_run_command.md for the given command
        → produces session-specific understanding: active config chain, relevant files,
          output paths, metric candidates, tuning knobs

### 3. CREATE SESSION:
    a. See `{PROJECT_DIR}/HPARAM.md` to find the PRIMARY_CONFIG_PATH.
    b. {SM} create-session {PROJECT_DIR} \
        --base-command "{BASE_COMMAND}" \
        --primary-metric {METRIC} \
        --goal {GOAL} \
        --primary-config-path {PRIMARY_CONFIG_PATH}
        → creates {SESSION_DIR} = {PROJECT_DIR}/aht/yyyy-mm-dd/hh-mm-ss/
        → auto-inserts `- override` into the primary config's defaults list
        → next_step tells you to call append-report
    c. {SM} append-report {SESSION_DIR} "your understanding to the task"
        → writes task understanding into {SESSION_DIR}/report.md
        → next_step tells you to call create-run

### 4. TUNING LOOP (baseline run 0 + up to BUDGET tuning iterations):

    The main agent drives the outer loop with three commands per iteration.
    Subagents handle tuning decisions and post-run analysis autonomously.
    The main agent MUST BLOCK after each spawn and do nothing until the
    subagent returns.

    a. CREATE RUN:
        {SM} create-run {SESSION_DIR}
        → creates {SESSION_DIR}/runs/{N}/ scaffold files
        → if N == 0 (first run): automatically spawns a subagent to generate
          strategy.md by following {SKILL_DIR}/prompts/plan_tuning_strategy.md.
          BLOCK until that subagent returns before proceeding.
        → next_step tells you to call `tune`

    b. SPAWN TUNING SUBAGENT:
        {SM} tune {SESSION_DIR} --run-id {N}
        → embeds HPARAM.md, results summary, report, and strategy as context
        → spawns a subagent that:
            1. Reads the context and decides the best override for run {N}
            2. Calls `{SM} run {SESSION_DIR} --run-id {N} \
                   --command-str "{BASE_COMMAND}" [--conda-env {CONDA_ENV}] \
                   --override key=value [--override key2=value2 ...]`
               which writes override.yaml AND launches the command in a detached tmux session
            3. Returns the chosen override and reasoning to the main agent
        → BLOCK until the subagent returns.
        → next_step tells you to call poll-run

    c. POLL until finished:
        {SM} poll-run {SESSION_DIR} --run-id {N} [--tail 50]
        → if still running: next_step tells you to estimate remaining time via
          eta.py, then add a cron reminder via `cron.add` and STOP.
          Wait for the cron to wake you; then poll again.
        → if finished: results.csv is updated automatically, then spawns a subagent
          that:
            1. Locates the TensorBoard event file (path recorded in HPARAM.md)
            2. (remote only) Copies it back with scp
            3. Calls `{SM} analyze-event {SESSION_DIR} --run-id {N} \
                   --event-path /path/to/event`
               which writes {SESSION_DIR}/runs/{N}/event_analysis.json
            4. Reads the analysis and the current report
            5. Calls `{SM} append-report {SESSION_DIR} "## Run {N} ..."` to log results
          BLOCK until that subagent returns.

    d. COPYBACK and ANALYZE:
        copy back the event file if using ssh remote machine
        {SM} analyze-event {EVENT_FILE_PATH}
        BLOCK until that subagent returns.

### 5. FINALIZE:
    {SM} finalize-session {SESSION_DIR} completed [--notes "summary"]
    → present final report and best config to user

## Capability: experiment-history visualization/reporting

Use this when the user wants to inspect the session-backed experiment history, compare recent or best runs, or export a lightweight report.

- CLI summary:
  `python3 skills/auto-hparam-tuning/scripts/report_experiment_history.py --project-root /path/to/project summary`
- Markdown report:
  `python3 skills/auto-hparam-tuning/scripts/report_experiment_history.py --project-root /path/to/project report --output-markdown /path/to/experiment_history_report.md`
- HTML report:
  `python3 skills/auto-hparam-tuning/scripts/report_experiment_history.py --project-root /path/to/project report --output-html /path/to/experiment_history_report.html`

Behavior details:
- The script reads the stored per-run config and metrics payloads referenced by session CSVs.
- If `--metric` is omitted, it auto-selects a numeric metric from the current history contents, preferring common names such as `val/loss` when present.
- Metric direction defaults to `min` for loss/error-like names and `max` otherwise; override with `--goal min|max` if needed.
- Reports stay practical: project summary, top runs table, recent runs table, compact metric trend, and config diffs among top runs.
