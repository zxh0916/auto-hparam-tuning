---
name: auto-hparam-tuning
description: Understand and automatically tune the hyperparameters of a project that uses hydra, with respect to the specified metric(s).
---

# Automatic Hyperparameter Tuning

Automatically tune the hyperparameters of a learning process: fetch results from tensorboard, analyze with pandas and numpy, then tune with hydra.

## Overview

You are the AHT (Automatic Hyperparameter Tuning) agent. Given a project with Hydra-based hyperparameter structure, you **autonomously** execute a closed-loop tuning cycle:

1. Walk through the project and detect the entry script, main config file, and hparam structure.
2. Given a running command from the user, identify the task and related hyperparameters.
3. Trigger a test run to get metric keys from the tensorboard event file.
4. With respect to a major metric (user-specified or auto-selected), recognize which hparams influence the metric by reading code and observing patterns as you change them.
5. After each run, log the result, analysis, and applied changes in a report.
6. Repeat the `[run → analyze → log → tune]` loop for a user-specified number of iterations, then report the final result.

**You execute the entire pipeline yourself. The user only provides a project path and a run command. You handle everything else — never ask the user to run commands.**

## When to Activate

Activate this skill when the user asks to:
- Tune or optimize hyperparameters for a training project
- Analyze training results and suggest improvements
- Set up automated hyperparameter search
- Diagnose training issues (divergence, overfitting, plateau, etc.)
- Understand the hyperparameter structure of a Hydra project

## Path Convention

All paths in this document use `{SKILL_DIR}` to refer to the directory containing this `SKILL.md` file. Resolve it to the actual absolute path when executing scripts or reading prompts.

- Scripts: `{SKILL_DIR}/scripts/`
- Prompts: `{SKILL_DIR}/prompts/`

## Prerequisites

```
tensorboard, pandas, tqdm
```
Install via: `pip install -r {SKILL_DIR}/../../requirements.txt`

If a required package is missing, install it automatically before retrying.

## Autonomous Orchestration Protocol

**This skill is fully autonomous.** When activated, execute the entire pipeline without requiring the user to run any commands.

### Required Input

Extract from the user's message:
- **PROJECT_DIR** (required): Path to the target training project (local or remote via SSH)
- **BASE_COMMAND** (optional): The training command (e.g., `python train.py task=foo`) — auto-detected if omitted
- **METRIC** (optional): Target metric name — auto-discovered from event file if not specified
- **GOAL** (optional): `maximize` or `minimize` — inferred from metric name if not specified
- **BUDGET** (optional): Maximum tuning iterations (default: 10)
- **SSH_HOST** (optional): SSH host for remote projects

If `PROJECT_DIR` is not provided, ask the user — it is the **only** question you may need to ask.
If `BASE_COMMAND` is not provided, run `detect-run-command` (Step 0 below) to auto-detect it.

### Pipeline Algorithm

```
resolve SKILL_DIR = absolute path to this SKILL.md's parent directory

0. AUTO-DETECT RUN COMMAND (skip if BASE_COMMAND already provided):
   python {SKILL_DIR}/scripts/project_understanding.py detect-run-command {PROJECT_DIR}
   → returns best_command, best_script, confidence, and top-5 candidates
   → Use best_command as BASE_COMMAND if confidence is "high" or "medium"
   → If confidence is "low", ask the user to confirm or provide BASE_COMMAND

1. UNDERSTAND PROJECT:
   a. python {SKILL_DIR}/scripts/project_understanding.py inspect-project {PROJECT_DIR}
      → tells you which docs exist, what needs to be generated, and which prompts to use
   b. Follow {SKILL_DIR}/prompts/generate_project_md.md if PROJECT.md is missing in `aht/`
      → creates `{PROJECT_DIR}/aht/PROJECT.md` (general project onboarding guide)
   c. Follow {SKILL_DIR}/prompts/get_hparam_structure.md if HPARAM.md is missing in `aht/`
      → creates `{PROJECT_DIR}/aht/HPARAM.md` (Hydra config and hparam guide)

2. UNDERSTAND RUN COMMAND:
   a. python {SKILL_DIR}/scripts/project_understanding.py prepare-run-understanding
      {PROJECT_DIR} "{BASE_COMMAND}"
      → returns required_prompt and available_context_files
   b. Follow {SKILL_DIR}/prompts/understand_run_command.md for the given command
      → produces session-specific understanding: active config chain, relevant files,
        output paths, metric candidates, tuning knobs

3. CREATE SESSION:
   python {SKILL_DIR}/scripts/session_manager.py \
     [--ssh-host {SSH_HOST}] \
     create-session {PROJECT_DIR} \
     --base-command "{BASE_COMMAND}" \
     --primary-metric {METRIC} \
     --goal {GOAL}
   → creates {PROJECT_DIR}/aht/yyyy-mm-dd/hh-mm-ss/

4. MICRO-BASELINE RUN:
   a. CREATE RUN:   session_manager.py create-run {SESSION_DIR}
   b. Launch BASE_COMMAND with `hydra.run.dir={RUN_DIR}/workspace` AND a runtime truncation knob (e.g., `trainer.max_epochs=2`) to save time.
   c. Analyze event file with {SKILL_DIR}/scripts/analyze_event.py
   → confirm metric keys, establish micro-baseline performance

4.5 PLAN STRATEGY:
   Follow {SKILL_DIR}/prompts/plan_tuning_strategy.md to create `{SESSION_DIR}/strategy.md`

5. TUNING LOOP (up to BUDGET iterations):
   a. CREATE RUN:   session_manager.py create-run {SESSION_DIR} (repeat for parallel runs)
   b. TUNE:         Review strategy.md. Decide hparam changes, write override.yaml in run dir
   c. EXECUTE:      Launch training with overrides (append `ckpt_path=...` if promoting). Launch in parallel (`&` or `Start-Job`) if compute allows. Capture stdout/stderr.
   d. ANALYZE:      analyze_event.py on event file → curve statistics
   e. RECORD:       session_manager.py update-run with metrics and status
   f. LOG:          session_manager.py append-report with analysis and takeaway
   g. CHECK:        session_manager.py summarize-results → trend_hint
   h. TERMINATE?:   Stop if trend_hint is flat/degrading × 3, budget exhausted, or objective met

6. FINALIZE:
   session_manager.py finalize-session {SESSION_DIR} completed
   → present final report and best config to user
```

### Operating Principles

- **Run everything yourself** — use Shell tools to execute scripts, Read to follow prompt instructions
- **Never ask the user to run commands** — you are the executor
- **Minimize interruptions** — only pause if a critical decision cannot be inferred
- **Autonomous but Docile**: The agent drives the loop, but logs every decision and result into the `aht/` folder.
- **Project Isolation**: Do not modify project source code unless fixing a bug that prevents training. All tuning happens via Hydra command-line overrides.
- **Workspace Hygiene**: All generated documentation (`PROJECT.md`, `HPARAM.md`), session metadata, and temporary analysis files (`.json`, `.txt`) MUST be stored inside the `{PROJECT_ROOT}/aht/` directory. Never pollute the project root with temporary artifacts.
- **Report progress** — briefly inform the user at key milestones
- **Handle errors autonomously** — attempt recovery before escalating

## Core Workflow

### Step 1: Understand the Project

**Goal**: Build understanding of the project structure, config system, and hparams.

**Actions**:
1. Run the inspection helper to find out what docs exist and what needs to be generated:
   ```bash
   python {SKILL_DIR}/scripts/project_understanding.py \
     [--ssh-host {SSH_HOST}] \
     inspect-project {PROJECT_DIR}
   ```
   Returns `need_generate_project_md`, `need_generate_hparam_md`, `read_order`, and prompt paths.

2. Read existing docs in the order specified by `read_order` (CLAUDE.md, AGENTS.md, `aht/PROJECT.md`, `aht/HPARAM.md`).

3. If `need_generate_project_md` is true, follow `{SKILL_DIR}/prompts/generate_project_md.md` to create `PROJECT.md` inside `{PROJECT_DIR}/aht/`. (Create the `aht/` folder first if it doesn't exist).

4. If `need_generate_hparam_md` is true, follow `{SKILL_DIR}/prompts/get_hparam_structure.md` to create `HPARAM.md` inside `{PROJECT_DIR}/aht/`.

### Step 2: Understand the Run Command

**Goal**: Determine exactly what the run command does — which config is active, which files are touched, where outputs go, what metrics to expect.

**Actions**:
1. Prepare the command understanding context:
   ```bash
   python {SKILL_DIR}/scripts/project_understanding.py \
     [--ssh-host {SSH_HOST}] \
     prepare-run-understanding {PROJECT_DIR} "{BASE_COMMAND}"
   ```

2. Read all `available_context_files` returned above.

3. Follow `{SKILL_DIR}/prompts/understand_run_command.md` for this specific command. Produce a structured note covering:
   - Runtime path and entrypoint
   - Active config chain and CLI overrides
   - Relevant source files
   - Output and TensorBoard paths
   - Primary metric recommendation
   - Tuning knobs for this session (grouped by category, flagged by risk)

4. Save this note into the session directory as `run_understanding.md` once the session is created.

### Step 3: Create Session and Micro-Baseline

**Goal**: Initialize the AHT session directory and establish a short initial baseline to save compute time.

**Actions**:
1. Create the session:
   ```bash
   python {SKILL_DIR}/scripts/session_manager.py \
     [--ssh-host {SSH_HOST}] \
     create-session {PROJECT_DIR} \
     --base-command "{BASE_COMMAND}" \
     --primary-metric {METRIC} \
     --goal {GOAL}
   ```

2. Launch the base command with a **runtime truncation knob** appended (e.g., `trainer.max_epochs=1` or `max_steps=500`). This is a "Micro-Baseline" that prevents wasting time on a full standard training run if you only need initial metrics.
   ```bash
   python {SKILL_DIR}/scripts/analyze_event.py {EVENT_PATH} --list-tags
   ```

3. Confirm METRIC and GOAL. Update session meta if they changed.

### Step 4: Plan Strategy

**Goal**: Analyze the baseline and generate a formal roadmap before running blind experiments.

Follow `{SKILL_DIR}/prompts/plan_tuning_strategy.md` to generate `{SESSION_DIR}/strategy.md`.

### Step 5: Tuning Loop

Repeat the following for each iteration until termination conditions are met.

#### 5a. Allocate a Run Directory
```bash
python {SKILL_DIR}/scripts/session_manager.py [--ssh-host {SSH_HOST}] create-run {SESSION_DIR}
```
Returns `run_dir`, `override_path`, `command_path`, `stdout_path`, `stderr_path`.

#### 5b. Decide Hyperparameter Changes

Based on the run understanding, `strategy.md` roadmap, and accumulated results:
- Review `summarize-results` output from the previous iteration.
- Follow the active phase in your `strategy.md` roadmap.
- If a phase's hypothesis is proven wrong (e.g., instability, flatline), update `strategy.md` with a pivot plan.
- Write the chosen Hydra overrides to `override.yaml` in the run directory
- If remote: upload `override.yaml` to the project config directory

#### 5c. Execute Training

Launch the training command with overrides. Redirect stdout/stderr to `stdout.log` and `stderr.log` in the run directory. Copy back the event file and Hydra-generated config to `copied/` if remote.
**Important**: On Windows, you MUST set the environment variable `PYTHONIOENCODING=utf-8` or `PYTHONUTF8=1` for the training process to prevent `UnicodeEncodeError: 'gbk' codec` crashes from tqdm/logging.

**Parallel Execution (Adaptive)**: If the system has sufficient compute (e.g., multiple GPUs, or the model is very small/simple like MNIST) and you have planned a "Direct Strategy", you can launch multiple runs concurrently as background processes (e.g. `Start-Process` in PowerShell, or `&` in bash). Wait for them all to finish before analyzing the batch.

**Checkpoint Resuming (Progressive Tuning)**: If you are executing a Phase 2 or Phase 3 run, you MUST resume from the best checkpoint of the corresponding Phase 1 run to save time. Find the checkpoint (e.g., in `runs/{PREV_ID}/workspace/checkpoints/`) and append the checkpoint override to your command (e.g., `ckpt_path=runs/0/workspace/checkpoints/epoch=1.ckpt`).

#### 5d. Analyze Results

Run `analyze_event.py` on the finished run's event file.
**Tip**: Always redirect temporary output files (e.g., `res.json`) to the run directory (e.g., `runs/{ID}/analysis.json`) or the `aht/` root to maintain workspace hygiene.

```bash
python {SKILL_DIR}/scripts/analyze_event.py {EVENT_PATH} {METRIC} --mode {GOAL}
```

Inspect the returned statistics: `best_value`, `best_step`, `end_step`, `improvement`, `normalized_oscillation`.

#### 5e. Record the Run

```bash
python {SKILL_DIR}/scripts/session_manager.py [--ssh-host {SSH_HOST}] \
  update-run {SESSION_DIR} {RUN_ID} \
  --status finished \
  --primary-metric {BEST_VALUE} \
  --best-step {BEST_STEP}
```

#### 5f. Log Analysis to Report

```bash
python {SKILL_DIR}/scripts/session_manager.py [--ssh-host {SSH_HOST}] \
  append-report {SESSION_DIR} \
  "## Run {RUN_ID}: {TRIAL_NAME}\n\n**Overrides**: ...\n**Result**: {BEST_VALUE}\n**Analysis**: ..."
```

#### 5g. Check Termination

```bash
python {SKILL_DIR}/scripts/session_manager.py [--ssh-host {SSH_HOST}] \
  summarize-results {SESSION_DIR}
```

**Stop** if:
- `trend_hint` is `"flat"` or `"degrading"` for 3+ consecutive rounds
- Budget (BUDGET iterations) exhausted
- Objective met

### Step 6: Finalize

```bash
python {SKILL_DIR}/scripts/session_manager.py [--ssh-host {SSH_HOST}] \
  finalize-session {SESSION_DIR} completed
```

Present the final report to the user: best configuration, metric progression, key insights.

## Error Handling

- **UnicodeEncodeError / GBK codec error**: This happens frequently on Windows due to progress bars. Relaunch the training or analysis command with the environment variable `PYTHONUTF8=1` or `PYTHONIOENCODING=utf-8`.
- **Event file not found**: Run a baseline with the original command. If still missing, check TensorBoard integration in `aht/HPARAM.md`.
- **Metric not found**: Use `analyze_event.py --list-tags` to list available tags. Select the best candidate.
- **Training crash**: Record failure via `update-run --status failed`. Inspect stderr. Apply fix (OOM → halve batch size; NaN → reduce LR + add gradient clipping). Retry.
- **Config invalid**: Validate overrides with `python entrypoint.py --cfg job` before launching.
- **SSH failure**: Retry the remote operation up to 3 times. Report to user only if persistent.

## Session Layout

```
{PROJECT_DIR}/aht/
└── yyyy-mm-dd/
    └── hh-mm-ss/
        ├── meta.yaml              # Session metadata
        ├── results.csv            # Run results table
        ├── report.md              # Accumulated analysis report
        └── runs/
            ├── 0/
            │   ├── override.yaml  # Hydra overrides for this run
            │   ├── command.sh     # Exact training command
            │   ├── stdout.log
            │   ├── stderr.log
            │   ├── metrics.json
            │   ├── summary.md
            │   └── copied/        # Copied event files and configs
            └── 1/ ...
```

## Quick Reference

```
User: "帮我调一下 /home/user/my_project 的超参数，
       运行命令是 python train.py task=cifar10，目标是最小化 val/loss"

Agent 自动执行:
1. inspect-project → need_generate_project_md=true, need_generate_hparam_md=true
2. 生成 AHT 专属项目文档（放在 `aht/` 目录下）：`aht/PROJECT.md` 和 `aht/HPARAM.md`
3. prepare-run-understanding → 分析 train.py + conf/config.yaml + task/cifar10.yaml
4. 确认 metric=val/loss, goal=minimize, 关键 hparams: lr, dropout, weight_decay
5. create-session → /home/user/my_project/aht/2026-03-14/17-30-00/
6. 测试运行 → 验证 event file 生成，baseline val/loss=0.45
7. Run 0: lr=3e-4 → val/loss=0.38 (plateau at step 8000, consider cosine schedule)
8. Run 1: lr=3e-4, scheduler=cosine → val/loss=0.33
9. Run 2: lr=2e-4, dropout=0.2 → val/loss=0.31
10. summarize-results: trend=improving → continue
11. Run 3-4: fine-tune around best config
12. finalize-session → 呈现最终报告: best val/loss=0.29, lr=2e-4, dropout=0.2, wd=1e-4
```
