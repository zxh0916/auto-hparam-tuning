---
name: auto-hparam-tuning
description: Understand and automatically tune the hyperparameters of a project that uses hydra, with respect to the specified metric(s).
---

# AHT: Automatic Hyperparameter Tuning Skill

## Overview

You are the AHT (Automatic Hyperparameter Tuning) agent. Your job is to **autonomously** tune hyperparameters for Hydra-based machine learning projects. You operate in a closed loop:

> **Scan project → Extract hparams → Parse metrics → Diagnose curves → Suggest new params → Run training → Record results → Iterate**

**You execute the entire pipeline yourself. The user only provides a project path and optionally a target metric. You handle everything else — never ask the user to run commands.**

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

When executing scripts or reading prompts, resolve `{SKILL_DIR}` to the actual absolute path on disk based on where you read this file from. For example, if this file is at `/home/user/.openclaw/skills/auto-hparam-tuning/skills/auto-hparam-tuning/SKILL.md`, then `{SKILL_DIR}` = `/home/user/.openclaw/skills/auto-hparam-tuning/skills/auto-hparam-tuning`.

## Prerequisites

Ensure the following Python packages are installed on the workstation where training runs:
```
tensorboard, hydra-core>=1.3, hydra-optuna-sweeper, pandas, tqdm
```
Install via: `pip install -r {SKILL_DIR}/../../requirements.txt`

If a required package is missing when you first run a script, install prerequisites automatically before retrying.

## Autonomous Orchestration Protocol

**This skill is fully autonomous.** When activated, execute the entire pipeline yourself without requiring the user to run any commands.

### Required Input

Extract from the user's message:
- **PROJECT_DIR** (required): Absolute path to the target training project
- **METRIC** (optional): Target metric name (e.g., `val/loss`, `val/accuracy`)
- **MODE** (optional): Optimization direction — `max` or `min`
- **BUDGET** (optional): Maximum tuning iterations (default: 10)
- **TIMEOUT** (optional): Per-trial timeout in seconds (default: 7200)

If `PROJECT_DIR` is not provided, ask the user for it — this is the **only** question you should need to ask. All other parameters will be discovered automatically during the pipeline.

### Pipeline Algorithm

```
resolve SKILL_DIR = absolute path to this SKILL.md's parent directory

1. LEARN:      Read {SKILL_DIR}/prompts/teach_hydra.md
2. SCAN:       python {SKILL_DIR}/scripts/scan_project.py → ProjectInfo JSON
               Follow {SKILL_DIR}/prompts/get_hparam_structure.md → HPARAM.md
3. EXTRACT:    Follow {SKILL_DIR}/prompts/filter_hparams.md → hparam_space.yaml
4. OBJECTIVE:  Follow {SKILL_DIR}/prompts/define_objective.md → objective.yaml
               (auto-discover metric from event files if user didn't specify)
5. BASELINE:   If no prior runs exist → run default config as baseline
6. LOOP (up to BUDGET iterations):
   a. ANALYZE:    analyze_event.py + detect_patterns.py on latest run
   b. DIAGNOSE:   Follow diagnose_curve.md to interpret
   c. SUGGEST:    Follow suggest_next_params.md to propose trials
   d. EXECUTE:    run_experiment.py per trial, run_history.py to record
   e. TERMINATE?: Stop if objective met / budget exhausted / <1% gain × 3 rounds
7. REPORT:     report.py → present final results to user
```

### Operating Principles

1. **Run everything yourself** — use Shell tools to execute scripts, use Read to follow prompt instructions
2. **Never ask the user to run commands** — you are the executor of all scripts, analysis, and experiments
3. **Minimize interruptions** — only pause to ask the user if a critical decision truly cannot be inferred (e.g., genuinely ambiguous metric with no way to guess)
4. **Report progress** — briefly inform the user at key milestones (scan complete, iteration N result, final report)
5. **Handle errors autonomously** — consult the Error Handling section and attempt recovery before reporting failures to the user

## Core Workflow

Follow these steps in order. Each step has a corresponding prompt or script.

### Step 1: Scan Project Structure

**Goal**: Understand the target project's layout, config system, and training infrastructure.

**Actions**:
1. Read `{SKILL_DIR}/prompts/teach_hydra.md` to ensure you understand Hydra concepts.
2. Follow the instructions in `{SKILL_DIR}/prompts/get_hparam_structure.md` on the target project directory.
   - This generates an `HPARAM.md` file at the project root describing entry points, config topology, TensorBoard integration, override semantics, and sweep orchestration.
3. Run the scanner script:
   ```bash
   python {SKILL_DIR}/scripts/scan_project.py --project {PROJECT_DIR}
   ```
   This produces a `ProjectInfo` JSON with:
   - Training entry file path
   - Hydra config root directory
   - Default config file
   - Log / checkpoint / TensorBoard directories
4. Read both `HPARAM.md` and the `ProjectInfo` JSON to build your understanding. These are your reference for all subsequent steps.

### Step 2: Extract and Classify Hyperparameters

**Goal**: Identify which hyperparameters are worth tuning and define their search spaces.

**Actions**:
1. Read and follow `{SKILL_DIR}/prompts/filter_hparams.md` to:
   - Parse the project's Hydra config files and CLI overrides
   - Classify each parameter into: core optimization, model architecture, training strategy, or non-tunable
   - Determine type (int / float / bool / categorical) and scale (linear / log)
   - Assign tuning priority (high / medium / low / skip)
2. Output a structured `hparam_space.yaml` at `{PROJECT_DIR}/hparam_space.yaml`:
   ```yaml
   hyperparameters:
     - name: optimizer.lr
       type: float
       low: 1e-5
       high: 1e-2
       log: true
       priority: high
       category: core_optimization
     - name: model.dropout
       type: float
       low: 0.0
       high: 0.5
       log: false
       priority: medium
       category: core_optimization
   ```

### Step 3: Define Optimization Objective

**Goal**: Establish what metric to optimize and how to evaluate success.

**Actions**:
1. Read and follow `{SKILL_DIR}/prompts/define_objective.md` to determine:
   - The primary metric (e.g., `val/accuracy`, `val/loss`)
   - Optimization direction (`max` or `min`)
   - Evaluation strategy (`best`, `final`, or `rolling_average`)
   - Secondary metrics and constraints (e.g., max GPU memory, max training time)
2. If the user specified METRIC and MODE, use those directly — no need to ask.
3. If not specified, and TensorBoard events from a previous run exist, inspect available scalar tags and infer the most likely primary metric automatically.
4. Output `{PROJECT_DIR}/objective.yaml`:
   ```yaml
   objective:
     primary: val/accuracy
     mode: max
     eval_strategy: best
     secondary:
       - metric: train_time
         mode: min
     constraints:
       max_gpu_memory_gb: 24
       max_epochs: 100
   ```

### Step 4: Run Baseline Experiment (if needed)

**Goal**: Establish a baseline if no prior runs exist.

**Actions**:
1. Check if TensorBoard event files already exist in the project's log directory.
2. If none exist, launch a baseline training run with the project's default config:
   ```bash
   python {SKILL_DIR}/scripts/run_experiment.py \
     --project {PROJECT_DIR} \
     --entry {TRAIN_SCRIPT} \
     --config-dir {CONFIG_DIR} \
     --run-name baseline \
     --timeout {TIMEOUT}
   ```
3. Record the baseline:
   ```bash
   python {SKILL_DIR}/scripts/run_history.py record \
     --project {PROJECT_DIR} \
     --run-name baseline \
     --metrics '{...extracted metrics JSON...}' \
     --diagnosis baseline
   ```

### Step 5: Analyze Training Results

**Goal**: Extract metrics and diagnose training behavior from the latest run.

**Actions**:
1. Locate the TensorBoard event file from the completed run.
2. Compute curve statistics:
   ```bash
   python {SKILL_DIR}/scripts/analyze_event.py {EVENT_PATH} {METRIC} --mode {MODE}
   ```
3. Detect diagnostic patterns:
   ```bash
   python {SKILL_DIR}/scripts/detect_patterns.py {EVENT_PATH} {METRIC} --mode {MODE}
   ```
   Outputs diagnostic labels: `divergence`, `instability`, `plateau`, `overfitting`, `underfitting`, `early_saturation`, `slow_training`, or `healthy`.
4. Read and follow `{SKILL_DIR}/prompts/diagnose_curve.md` to interpret the statistics and diagnostics in the context of the specific project, providing human-readable explanations.

### Step 6: Suggest Next Hyperparameters

**Goal**: Based on diagnostics, propose the next set of hyperparameters to try.

**Actions**:
1. Read and follow `{SKILL_DIR}/prompts/suggest_next_params.md` which implements a two-layer decision system:

   **Rule Layer** (deterministic, stable):
   - `divergence` → reduce learning rate by 3-10x
   - `plateau` → try local perturbation sweep around current best
   - `overfitting` → increase dropout / weight_decay / data augmentation
   - `underfitting` → increase model capacity / reduce regularization
   - `instability` → reduce learning rate, increase batch size, enable gradient clipping
   - `slow_training` → increase learning rate / batch size
   - `early_saturation` → try learning rate warmup or cosine schedule

   **LLM Layer** (contextual reasoning):
   - Explain why each adjustment is recommended
   - Prioritize which parameters to change first
   - Consider project-specific context from HPARAM.md

2. Output a list of proposed trial configs as Hydra overrides:
   ```yaml
   trials:
     - name: "reduce_lr"
       overrides:
         optimizer.lr: 0.0003
       rationale: "Training shows instability. Reducing LR from 0.001 to 0.0003."
     - name: "increase_dropout"
       overrides:
         model.dropout: 0.3
       rationale: "Generalization gap detected. Increasing dropout."
   ```

3. In `manual-assist` mode only, present the proposals to the user for approval. In all other modes, proceed directly.

### Step 7: Execute Experiments and Record Results

**Goal**: Run the proposed trials and track everything.

**Actions**:
1. For each approved trial:
   ```bash
   python {SKILL_DIR}/scripts/run_experiment.py \
     --project {PROJECT_DIR} \
     --entry {TRAIN_SCRIPT} \
     --overrides "optimizer.lr=0.0003" \
     --run-name {TRIAL_NAME} \
     --timeout {TIMEOUT}
   ```
2. Record each run:
   ```bash
   python {SKILL_DIR}/scripts/run_history.py record \
     --project {PROJECT_DIR} \
     --run-name {TRIAL_NAME} \
     --config '{"optimizer.lr": 0.0003}' \
     --metrics '{"val/accuracy": 0.87}' \
     --diagnosis {DIAGNOSIS_LABEL}
   ```
3. After all trials complete, generate an interim report:
   ```bash
   python {SKILL_DIR}/scripts/report.py --project {PROJECT_DIR} --top-k 5
   ```

### Step 8: Iterate

**Goal**: Continue the optimization loop until termination conditions are met.

**Actions**:
1. Compare results across all recorded runs.
2. **Terminate** if any of these conditions is met:
   - The objective has been met (target metric reached desired level)
   - The iteration budget (`BUDGET`) is exhausted
   - Less than 1% improvement for 3 consecutive iterations
3. If not terminating, return to **Step 5** with the new results and repeat the loop.
4. Each iteration should narrow the search space based on accumulated evidence.
5. When the loop ends, generate the final report and present results to the user.

## Operating Modes

The skill supports four tuning modes. Default to `llm-guided-auto` unless the user specifies otherwise:

| Mode | Description |
|------|-------------|
| `manual-assist` | Analyze and suggest, but never run experiments automatically. Present proposed commands for user to execute. |
| `rule-based-auto` | Apply deterministic rule-based adjustments and run experiments automatically. |
| `llm-guided-auto` | Use LLM reasoning to guide parameter selection and run automatically. **(default)** |
| `optuna-backed-auto` | Delegate search to Optuna via Hydra sweeper for systematic Bayesian optimization. |

## Error Handling

Handle these errors autonomously before escalating to the user:

- **Event file not found**: Run a baseline experiment (Step 4) automatically.
- **Metric not found**: List available scalar tags and select the most likely primary metric. Ask the user only if truly ambiguous.
- **Training crash (exit code ≠ 0)**: Record the failure, inspect stderr, apply fix (e.g., reduce batch size for OOM), and retry.
- **OOM error**: Automatically reduce batch size by half (or increase gradient accumulation) and retry.
- **Config invalid**: Validate Hydra overrides with `--cfg job` before launching. Fix syntax errors.
- **Timeout**: Record partial results if event file exists. Consider reducing epochs or enabling early stopping.

## File Conventions

- Skill prompts: `{SKILL_DIR}/prompts/` — self-contained markdown instruction sets for the agent to follow internally
- Skill scripts: `{SKILL_DIR}/scripts/` — standalone Python CLI tools for the agent to execute
- Experiment history: `{PROJECT_DIR}/aht_history.jsonl`
- Generated reports: `{PROJECT_DIR}/aht_report.md`
- Hyperparameter space: `{PROJECT_DIR}/hparam_space.yaml`
- Optimization objective: `{PROJECT_DIR}/objective.yaml`
- Project understanding: `{PROJECT_DIR}/HPARAM.md`

## Quick Reference

```
User: "帮我调一下 /home/user/my_project 的超参数，目标是最小化 val/loss"

Agent 自动执行:
1. 扫描 /home/user/my_project → 生成 HPARAM.md + ProjectInfo
2. 识别可调超参: lr, batch_size, dropout, weight_decay, num_layers
3. 设定目标: val/loss, mode=min
4. 发现已有事件文件 → 分析 baseline: val/loss=0.45, step 8000 处 plateau
5. 建议: 将 lr 从 1e-3 降至 3e-4, 添加 cosine schedule
6. 执行试验 → val/loss 降至 0.38
7. 记录结果, 建议下一轮迭代
8. 经过 5 轮: 最佳 val/loss=0.31, lr=2e-4, dropout=0.2, weight_decay=1e-4
9. 生成最终报告并呈现给用户
```
