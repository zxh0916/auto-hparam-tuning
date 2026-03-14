---
name: auto-hparam-tuning
description: Understand and automatically tune the hyperparameters of a project that uses hydra, with respect to the specified metric(s).
---

# Automatic Hyperparameter Tuning

Automatically tune the hyperparameters of a learning process: fetch results from tensorboard, analyze with pandas and numpy, then tune with hydra.

## Overview

You are the AHT (Automatic Hyperparameter Tuning) agent. Given a project with hydra-based hyperparameter structure, you **autonomously** execute a closed-loop tuning cycle:

1. Walk through the project and detect the entry script, main config file, and hparam structure.
2. Given a running command from the user, identify the task and related hyperparameters.
3. Trigger a test run to get metric keys from the tensorboard event file.
4. With respect to a major metric (user-specified or auto-selected), recognize which hparams influence the metric by reading code and observing patterns as you change them.
5. After each run, log the result, analysis, and applied changes in a report.
6. Repeat the `[run → analyze → log → tune]` loop for a user-specified number of iterations, then report the final result.

**You execute the entire pipeline yourself. The user only provides a project path, a running command, and optionally a target metric. You handle everything else — never ask the user to run commands.**

## When to Activate

Activate this skill when the user asks to:
- Tune or optimize hyperparameters for a training project
- Analyze training results and suggest improvements
- Set up automated hyperparameter search
- Diagnose training issues (divergence, overfitting, plateau, etc.)
- Understand the hyperparameter structure of a Hydra project

## Path Convention

All relative paths in this document refer to the **skill directory** — the directory containing this `SKILL.md` file. This is referred to as `{SKILL_DIR}` throughout.

- Scripts: `{SKILL_DIR}/scripts/`
- Prompts: `{SKILL_DIR}/prompts/`

When executing scripts or reading prompts, resolve `{SKILL_DIR}` to the actual absolute path on disk based on where you read this file from.

## Prerequisites

Ensure the following Python packages are installed on the workstation where training runs:
```
tensorboard, hydra-core>=1.3, pandas, tqdm
```
Install via: `pip install -r {SKILL_DIR}/../../requirements.txt`

If a required package is missing when you first run a script, install prerequisites automatically before retrying.

## Autonomous Orchestration Protocol

**This skill is fully autonomous.** When activated, execute the entire pipeline yourself without requiring the user to run any commands.

### Required Input

Extract from the user's message:
- **PROJECT_DIR** (required): Path to the target training project (local or remote via SSH)
- **BASE_COMMAND** (required): The training command (e.g., `python train.py task=foo`)
- **METRIC** (optional): Target metric name (e.g., `val/loss`, `val/accuracy`)
- **GOAL** (optional): Optimization direction — `maximize` or `minimize`
- **BUDGET** (optional): Maximum tuning iterations (default: 10)
- **SSH_HOST** (optional): SSH host for remote projects (e.g., `user@server`)

If `PROJECT_DIR` or `BASE_COMMAND` is not provided, ask the user — these are the **only** questions you should need to ask. All other parameters will be discovered automatically.

### Pipeline Algorithm

```
resolve SKILL_DIR = absolute path to this SKILL.md's parent directory

1. LEARN:     Read {SKILL_DIR}/prompts/teach_hydra.md (Hydra primer)
2. SCAN:      Follow {SKILL_DIR}/prompts/get_hparam_structure.md → HPARAM.md
              Run {SKILL_DIR}/scripts/scan_project.py (if available) for extra metadata
3. HPARAMS:   Follow {SKILL_DIR}/prompts/filter_hparams.md → identify tunable hparams
4. SESSION:   python {SKILL_DIR}/scripts/session_manager.py create-session
              → creates {PROJECT_DIR}/aht/yyyy-mm-dd/hh-mm-ss/ with meta, results, report
5. TEST RUN:  Launch base command → identify metric keys from event file
6. LOOP (up to BUDGET iterations):
   a. CREATE RUN:  session_manager.py create-run → allocate run directory + override.yaml
   b. TUNE:        Decide hparam changes, write override.yaml
   c. EXECUTE:     Launch training with overrides
   d. ANALYZE:     analyze_event.py on event file → curve statistics
   e. RECORD:      session_manager.py update-run with metrics
   f. DIAGNOSE:    Follow diagnose_curve.md to interpret results
   g. LOG:         session_manager.py append-report with analysis
   h. TERMINATE?:  session_manager.py summarize-results → check trend
                   Stop if objective met / budget exhausted / <1% gain × 3 rounds
7. FINALIZE:  session_manager.py finalize-session → present final report
```

### Operating Principles

1. **Run everything yourself** — use Shell tools to execute scripts, use Read to follow prompt instructions
2. **Never ask the user to run commands** — you are the executor of all scripts, analysis, and experiments
3. **Minimize interruptions** — only pause to ask the user if a critical decision truly cannot be inferred
4. **Report progress** — briefly inform the user at key milestones (scan complete, iteration N result, final report)
5. **Handle errors autonomously** — consult the Error Handling section and attempt recovery before reporting failures

## Core Workflow

### Step 1: Walk Through the Project

**Goal**: Understand the project's hparam structure, entry points, and config system.

**Actions**:
1. Read `{SKILL_DIR}/prompts/teach_hydra.md` to understand Hydra concepts.
2. Follow `{SKILL_DIR}/prompts/get_hparam_structure.md` on the target project:
   - Generates `HPARAM.md` at the project root describing entry points, config topology, TensorBoard integration, override semantics, and sweep orchestration.
   - If the project is remote, copy HPARAM.md back locally.
3. Optionally run `{SKILL_DIR}/scripts/scan_project.py --project {PROJECT_DIR}` for machine-readable metadata.

### Step 2: Identify Task and Hyperparameters

**Goal**: Given the user's running command, identify the task and related hparams.

**Actions**:
1. Read and follow `{SKILL_DIR}/prompts/filter_hparams.md` to:
   - Parse the project's Hydra config files and the user's CLI overrides
   - Classify parameters: core optimization, model architecture, training strategy, or non-tunable
   - Determine type, scale, search range, and tuning priority
2. Record the tunable hparams for use in subsequent iterations.

### Step 3: Create Session and Test Run

**Goal**: Initialize the AHT session and run a test to discover available metrics.

**Actions**:
1. Create an AHT session:
   ```bash
   python {SKILL_DIR}/scripts/session_manager.py \
     [--ssh-host {SSH_HOST}] \
     create-session {PROJECT_DIR} \
     --base-command "{BASE_COMMAND}" \
     --primary-metric {METRIC} \
     --goal {GOAL}
   ```
   This creates the canonical session layout under `{PROJECT_DIR}/aht/yyyy-mm-dd/hh-mm-ss/`.

2. Launch the base command as a test run to identify metric keys from the tensorboard event file.
3. Use `{SKILL_DIR}/scripts/analyze_event.py` to inspect the event file and discover available scalar tags.
4. If the user didn't specify METRIC, auto-select the most appropriate primary metric from the discovered tags.

### Step 4: Tuning Loop

**Goal**: Iteratively tune hyperparameters by running experiments and analyzing results.

For each iteration:

#### 4a. Create Run Directory
```bash
python {SKILL_DIR}/scripts/session_manager.py \
  [--ssh-host {SSH_HOST}] \
  create-run {SESSION_DIR}
```
This creates `runs/<N>/` with `override.yaml`, `command.sh`, `metrics.json`, `summary.md`.

#### 4b. Decide Tuning Changes

Read and follow `{SKILL_DIR}/prompts/suggest_next_params.md` (or use your own reasoning based on `{SKILL_DIR}/prompts/diagnose_curve.md`):
- Apply the rule-based layer for deterministic adjustments based on diagnosis
- Apply LLM reasoning for contextual prioritization
- Write the chosen overrides into `override.yaml` in the run directory
- Upload to the remote config directory if working over SSH

#### 4c. Execute Training

Launch the training command with overrides. Capture stdout/stderr.

#### 4d. Analyze Results

Copy back the event file and config file, then analyze:
```bash
python {SKILL_DIR}/scripts/analyze_event.py {EVENT_PATH} {METRIC} --mode {GOAL}
```

#### 4e. Record and Diagnose

1. Update the run result:
   ```bash
   python {SKILL_DIR}/scripts/session_manager.py \
     [--ssh-host {SSH_HOST}] \
     update-run {SESSION_DIR} {RUN_ID} \
     --status finished \
     --primary-metric {METRIC_VALUE} \
     --best-step {BEST_STEP}
   ```
2. Follow `{SKILL_DIR}/prompts/diagnose_curve.md` to interpret the training curve.
3. Append analysis to the session report:
   ```bash
   python {SKILL_DIR}/scripts/session_manager.py \
     [--ssh-host {SSH_HOST}] \
     append-report {SESSION_DIR} "## Run {RUN_ID}\n\n{ANALYSIS}"
   ```

#### 4f. Check Termination

Summarize accumulated results:
```bash
python {SKILL_DIR}/scripts/session_manager.py \
  [--ssh-host {SSH_HOST}] \
  summarize-results {SESSION_DIR}
```
**Stop** if:
- Objective met (target metric reached desired level)
- Budget exhausted (iteration count ≥ BUDGET)
- `trend_hint` is `"flat"` or `"degrading"` for 3+ consecutive rounds
- Less than 1% improvement for 3 consecutive iterations

Otherwise, return to **4a** for the next iteration.

### Step 5: Finalize

**Goal**: Close the session and present results.

**Actions**:
1. Finalize the session:
   ```bash
   python {SKILL_DIR}/scripts/session_manager.py \
     [--ssh-host {SSH_HOST}] \
     finalize-session {SESSION_DIR} completed
   ```
2. Present the final report to the user, including:
   - Best configuration found
   - Metric progression across iterations
   - Key insights and diagnosis history

## Error Handling

Handle these errors autonomously before escalating to the user:

- **Event file not found**: Run a baseline experiment with the base command first.
- **Metric not found**: List available scalar tags from the event file and select the most likely primary metric. Ask the user only if truly ambiguous.
- **Training crash (exit code ≠ 0)**: Record the failure via `update-run --status failed`, inspect stderr, apply fix (e.g., reduce batch size for OOM), and retry.
- **OOM error**: Automatically reduce batch size by half (or increase gradient accumulation) and retry.
- **Config invalid**: Validate Hydra overrides with `--cfg job` before launching. Fix syntax errors.
- **Timeout**: Record partial results if event file exists. Consider reducing epochs or enabling early stopping.
- **SSH failure**: Retry the remote operation. If persistent, report to user.

## Session Layout

Each tuning session creates a canonical directory structure:

```
{PROJECT_DIR}/aht/
└── yyyy-mm-dd/
    └── hh-mm-ss/
        ├── meta.yaml          # Session metadata (project, command, metric, goal, status)
        ├── results.csv        # Run results table (run_id, status, primary_metric, ...)
        ├── report.md          # Accumulated analysis report
        └── runs/
            ├── 0/
            │   ├── override.yaml   # Hydra overrides for this run
            │   ├── command.sh      # Exact training command
            │   ├── stdout.log      # Training stdout
            │   ├── stderr.log      # Training stderr
            │   ├── metrics.json    # Extracted metrics
            │   ├── summary.md      # Run summary and takeaway
            │   └── copied/         # Copied event files and configs
            ├── 1/
            │   └── ...
            └── ...
```

## File Conventions

- Skill prompts: `{SKILL_DIR}/prompts/` — instruction sets for the agent to follow internally
- Skill scripts: `{SKILL_DIR}/scripts/` — standalone Python CLI tools
  - `analyze_event.py` — TensorBoard event → curve statistics
  - `session_manager.py` — Session/run lifecycle management (local + SSH)
  - `scan_project.py` — Project structure scanner (if available)
  - `detect_patterns.py` — Curve diagnosis (if available)
- Session data: `{PROJECT_DIR}/aht/` — all session directories and run data
- Project understanding: `{PROJECT_DIR}/HPARAM.md`

## Quick Reference

```
User: "帮我调一下 /home/user/my_project 的超参数，
       运行命令是 python train.py task=cifar10，目标是最小化 val/loss"

Agent 自动执行:
1. 扫描项目 → 生成 HPARAM.md, 识别可调超参
2. 创建 session → /home/user/my_project/aht/2026-03-14/17-30-00/
3. 测试运行 → 发现 val/loss, val/acc, train/loss 等 metric
4. 分析 baseline: val/loss=0.45, plateau at step 8000
5. 创建 run 0 → override.yaml: lr=3e-4, 执行 → val/loss=0.38
6. 创建 run 1 → override.yaml: lr=2e-4, dropout=0.2 → val/loss=0.33
7. 创建 run 2 → override.yaml: lr=2e-4, dropout=0.2, wd=1e-4 → val/loss=0.31
8. summarize-results: trend=improving, best=run 2
9. 经过 5 轮: best val/loss=0.31, finalize session
10. 呈现最终报告给用户
```
