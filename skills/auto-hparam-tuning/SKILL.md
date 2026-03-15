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

0. Run the `aht-init` sub-skill first.
   - Collect and normalize the minimum required inputs: project path, conda env, reference launch script/command, and optimization target.
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


## 1. Understand the Project and Create the Session

Before tuning the hparam of the project, you should always make sure that the `aht-init` step has already produced and the user has explicitly confirmed these four fields: project path, conda env name, reference launch command/script, and optimization target.

Then make sure that you know the project enough by:

1. Understand the whole project:

    1. `python scripts/project_understanding.py inspect-project <path to project root>` for local project or 

    2. `python scripts/project_understanding.py inspect-project --ssh-host user@remotehost <path to project root>` for project on remote machine.

2. Understand the task to be tuned: `python scripts/project_understanding.py prepare-run-understanding <path to project root> <command to be tuned>`.

3. Create the Session: `python scripts/session_manager.py create-session <path to project root> --ssh-host user@remotehost --base-command <command> --primary-metric <primary metric> --goal <goal>`.

4. Write the Session Understanding into the report: `python scripts/session_manager.py --ssh-host user@remotehost <path to project root> append-report <your understanding>`.


## Pipeline Algorithm

resolve SKILL_DIR = absolute path to this SKILL.md's parent directory

### 1. UNDERSTAND PROJECT:
    a. python {SKILL_DIR}/scripts/project_understanding.py[ --ssh-host user@remotehost] inspect-project {PROJECT_DIR}
        → tells you which docs exist, what needs to be generated, and which prompts to use
    b. Follow {SKILL_DIR}/prompts/generate_project_md.md if PROJECT.md is missing in `aht/`
        → creates `{PROJECT_DIR}/PROJECT.md` (general project onboarding guide)
    c. Follow {SKILL_DIR}/prompts/get_hparam_structure.md if HPARAM.md is missing in `aht/`
        → creates `{PROJECT_DIR}/HPARAM.md` (Hydra config and hparam guide)

### 2. UNDERSTAND RUN COMMAND:
    a. python {SKILL_DIR}/scripts/project_understanding.py[ --ssh-host user@remotehost] prepare-run-understanding
        {PROJECT_DIR} "{BASE_COMMAND}"
        → returns required_prompt and available_context_files
    b. Follow {SKILL_DIR}/prompts/understand_run_command.md for the given command
        → produces session-specific understanding: active config chain, relevant files,
          output paths, metric candidates, tuning knobs

### 3. CREATE SESSION:
    a. python {SKILL_DIR}/scripts/session_manager.py[ --ssh-host user@remotehost] \
        create-session {PROJECT_DIR} \
        --base-command "{BASE_COMMAND}" \
        --primary-metric {METRIC} \
        --goal {GOAL}
        → creates {PROJECT_DIR}/aht/yyyy-mm-dd/hh-mm-ss/
    b. python {SKILL_DIR}/scripts/session_manager.py[ --ssh-host user@remotehost] append-report
        {PROJECT_DIR} "your understanding to the task"
        → appends in {PROJECT_DIR}/aht/yyyy-mm-dd/hh-mm-ss/report.md

### 4. MICRO-BASELINE RUN:
    a. CREATE RUN:    session_manager.py create-run {SESSION_DIR}
    b. fill override.yaml returned by the Session Manager with empty content and copy the file content to override.yaml that lies within the same directory as the primary config file.
    c. Launch BASE_COMMAND within the conda environment specified by the user.
    d. Estimate the duration of the test run.
    e. (remote only) Once the test run complete, copy back the event file and the hydra config back to the /tmp file from the remote.
    d. Analyze event file with {SKILL_DIR}/scripts/analyze_event.py
      → confirm metric keys, establish micro-baseline performance
    PLAN STRATEGY:
      Follow {SKILL_DIR}/prompts/plan_tuning_strategy.md to create `{SESSION_DIR}/strategy.md`

### 5. TUNING LOOP (up to BUDGET iterations):
    a. CREATE RUN:    session_manager.py create-run {SESSION_DIR} (repeat for parallel runs)
    b. TUNE:          Review strategy.md. Decide hparam changes, write override.yaml in run dir
    c. OVERRIDE:      Overwrite the content of the override.yaml that lies within the same directory as the primary config file with the content of run/<runId>/override.yaml
    d. EXECUTE:       Launch training with overrides with specified conda environment.
    e. COPYBACK:      (remote only) Once the test run complete, copy back the event file and the hydra config back to the /tmp file from the remote.
    f. ANALYZE:       Analyze_event.py on event file → curve statistics
    g. RECORD:        Session_manager.py update-run with metrics and status
    h. LOG:           Session_manager.py append-report with analysis and takeaway
    i. CHECK:         Session_manager.py summarize-results → trend_hint
    j. TERMINATE?:    Stop if trend_hint is flat/degrading × 3, budget exhausted, or objective met

### 6. FINALIZE:
    session_manager.py finalize-session {SESSION_DIR} completed
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
