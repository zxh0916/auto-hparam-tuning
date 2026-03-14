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

