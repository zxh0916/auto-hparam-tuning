---
name: auto-hparam-tuning
description: Understand and automatically tune the hyperparameters of a project that uses hydra, with respect to the specified metric(s).
---

# Automatic Hyperparameter Tuning

Automatically tune the hyperparameteres of a learning process: fetch results from tensorboard, analyze with pandas and numpy, then tune with hydra.

## Overview

This skill will automatically the hyperparameters managed by hydra config system to optimize a learning process. Given a project with hydra-based hyperparameter structure, this skill first walk through the project and detect the entry script and function along with the main config file. Then, it triggers a test run with command specified by the user to get the metric keys from the tensorboard event file. After the test run, with respect to a major metric specified by the user or automatically selected by the agent, the skill recognizes which subset of the hyperparameters are related to the trend of the major metric and analyze their specific influence by both reading the code and watch the pattern as the agent changes them. After each run, the agent should log the result, the analysis, and the applied changes in a report. The agent should repeat this [...->run->analyze->log->tune->run->analyze->log->tune->...] loop for a user-specified times and then report the final result.

## Workflow

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

## Experiment History

Use the CSV-backed session store under `aht/yyyy-mm-dd/hh-mm-ss/` as the canonical experiment backend.
Do not create or depend on SQLite state. The canonical structured index is `results.csv`; detailed per-run payloads
live beside it under `runs/<id>/resolved_config.json`, `runs/<id>/metrics.json`, and `runs/<id>/summary.md`.

- Initialize the backend layout if needed:
  `python3 skills/auto-hparam-tuning/scripts/init_experiment_history_db.py --project-root /path/to/project`
- Insert one run after resolving the Hydra config and collecting metrics:
  `python3 skills/auto-hparam-tuning/scripts/insert_experiment_history.py --project-root /path/to/project --run-name baseline --config-file resolved_config.yaml --metrics-json '{"val/loss":0.42,"val/acc":0.91}' --primary-metric-name val/loss`
- Query recent runs for agent consumption:
  `python3 skills/auto-hparam-tuning/scripts/query_experiment_history.py --project-root /path/to/project --limit 20 --format json`

Behavior details:
- History is aggregated from all session CSVs under the target project unless `--session-dir` is used.
- `results.csv` stores the stable run index and artifact paths; per-run JSON/YAML payloads are loaded on demand.
- If no session exists yet, compatibility scripts auto-create one instead of requiring a separate database bootstrap step.

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

## Capability: iter_next_hparams

Use `iter_next_hparams` for one practical tuning iteration that stays fully script-backed and CSV-native:

1. Read the aggregated session history if it already exists.
2. Pick the next proposal from a declared search space.
3. Execute one local training command.
4. Record the new config and metrics into the active AHT session.

Primary command:

`python3 skills/auto-hparam-tuning/scripts/iter_next_hparams.py --project-root /path/to/project --base-config-file resolved_config.yaml --search-space-file search_space.yaml --train-command-template 'python train.py --config {config_path} --metrics-out {metrics_path}'`

Minimal search-space file shape:

```yaml
metric:
  name: val/loss
  goal: min
run_name_prefix: tune
parameters:
  optimizer.lr:
    values: [0.0001, 0.0003, 0.001]
  trainer.batch_size:
    values: [32, 64, 128]
```

Behavior details:
- No history or no matching rows: propose the baseline config values projected into the discrete search space.
- Existing rows: score the best prior run by `metric.name`, explore one-step neighbors first, then fall back to the nearest untried grid point.
- The executed run is appended to the active session and updates `results.csv`, `resolved_config.json`, `metrics.json`, and `report.md`.
