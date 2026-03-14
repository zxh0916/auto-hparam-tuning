# AHT: Automatic Hyperparameter Tuning with Coding Agents using Hydra

AHT is an agentic experiment optimization framework for Hydra-based ML training projects. It operates in a closed loop:

> **Scan project → Extract hparams → Parse metrics → Diagnose curves → Suggest new params → Run training → Record results → Iterate**

## Installation

Clone the repo into your global skill directory:
```bash
cd ~/.openclaw/skills
git clone https://github.com/zxh0916/auto-hparam-tuning.git
```

Then modify your openclaw config:
```json
{
  "skills": {
    "load": {
      "extraDirs": [
        "~/.openclaw/skills/auto-hparam-tuning/skills"
      ]
    },
    "entries": {
      "auto-hparam-tuning": { "enabled": true }
    }
  }
}
```

On workstation:
```bash
git clone https://github.com/zxh0916/auto-hparam-tuning.git
pip install -r auto-hparam-tuning/requirements.txt
```

## Core Features

- Automatically explore the project and understand the hparams
- Extract tunable hparams from commands
- Analyze the critical metrics from the tensorboard event file
- Judge the run with the metrics
- Tune the hparam according to the result
- Start a sweep to search in a smaller hparam space
- Write tuning logs into disk
- Create stable AHT session directories under the tuned project, both locally and over SSH

## Session Manager

A lightweight session manager is available at:

```bash
python skills/auto-hparam-tuning/scripts/session_manager.py --help
```

It creates a canonical layout under the **target project**:

```text
<project_root>/aht/yyyy-mm-dd/hh-mm-ss/
```

Example for a local project:

```bash
python skills/auto-hparam-tuning/scripts/session_manager.py \
  create-session /path/to/project \
  --base-command "python train.py task=foo" \
  --primary-metric val/acc \
  --goal maximize
```

Example for a remote project over SSH:

```bash
python skills/auto-hparam-tuning/scripts/session_manager.py \
  --ssh-host user@server \
  create-session /remote/project/path \
  --base-command "python train.py task=foo" \
  --primary-metric val/acc \
  --goal maximize
```

Then allocate runs inside that session:

```bash
python skills/auto-hparam-tuning/scripts/session_manager.py create-run <session_dir>
python skills/auto-hparam-tuning/scripts/session_manager.py --ssh-host user@server create-run <remote_session_dir>
```

You can also summarize the accumulated run history with pandas before deciding the next tuning move:

```bash
python skills/auto-hparam-tuning/scripts/session_manager.py summarize-results <session_dir>
python skills/auto-hparam-tuning/scripts/session_manager.py --ssh-host user@server summarize-results <remote_session_dir>
```

### Advanced Features (Planned)

- Convert non-Hydra config systems into Hydra-based ones
- Multi-GPU parallel trial scheduling and resource management
- Optuna-backed Bayesian optimization via Hydra sweeper
- Human-in-the-loop approval mode
- Interactive dashboard for experiment visualization
- Parameter importance analysis and Pareto frontier

## Architecture

```
skills/auto-hparam-tuning/
├── SKILL.md                          # Agent skill manifest & workflow definition
├── prompts/
│   ├── get_hparam_structure.md       # Project exploration → HPARAM.md generation
│   ├── teach_hydra.md                # Hydra framework primer for agents
│   ├── filter_hparams.md             # Hyperparameter extraction & classification
│   ├── define_objective.md           # Optimization objective definition
│   ├── diagnose_curve.md             # Training curve diagnosis interpretation
│   └── suggest_next_params.md        # Next parameter suggestion engine
└── scripts/
    ├── analyze_event.py              # TensorBoard event → curve statistics
    ├── session_manager.py            # Session/run lifecycle management (local + SSH)
    ├── scan_project.py               # Project directory → ProjectInfo (JSON)
    ├── detect_patterns.py            # Curve statistics → diagnostic labels (JSON)
    ├── run_experiment.py             # Launch training with overrides & capture results
    ├── run_history.py                # JSONL experiment database (record/list/best)
    └── report.py                     # History → Markdown experiment report
```

## Usage

This is an **agent skill** — you don't run the scripts manually. Simply tell your coding agent:

> "帮我调一下 `/path/to/my/project` 的超参数，运行命令是 `python train.py task=foo`，目标是最小化 `val/loss`"

The agent will automatically:
1. Scan your project structure and understand its Hydra configuration
2. Identify and classify tunable hyperparameters
3. Create an AHT session and run baseline training
4. Analyze training curves and diagnose issues
5. Suggest and execute improved hyperparameter configurations
6. Iterate until the objective is met or the budget is exhausted
7. Generate a final report with the best configuration

### Minimal Trigger

```
Tune my project at /path/to/project, command: python train.py task=foo
```

The agent discovers the metric and everything else automatically.

### Full Trigger

```
Tune hyperparameters for /path/to/project.
Command: python train.py task=cifar10
Target metric: val/accuracy (maximize).
Budget: 15 iterations.
Remote: user@gpu-server
```

## Script Reference (for developers)

The scripts in `scripts/` are internal tools used by the agent. They can also be run standalone for debugging:

| Script | Purpose |
|--------|---------|
| `session_manager.py` | Session/run lifecycle: create-session, create-run, update-run, summarize-results, finalize-session |
| `analyze_event.py` | Parse TensorBoard event files into curve statistics |
| `scan_project.py` | Detect project structure (entry points, config dirs, log dirs) |
| `detect_patterns.py` | Diagnose training curves (divergence, plateau, overfitting, etc.) |
| `run_experiment.py` | Launch training with Hydra overrides and capture results |
| `run_history.py` | JSONL-based experiment database (record, list, best) |
| `report.py` | Generate Markdown experiment reports |

## Roadmap

### P0: MVP Closed Loop
- [x] Convert `pd.DataFrame` from TensorBoard event file
- [x] Write functions to detect patterns from curves
- [x] Write a prompt that teaches the agent using Hydra
- [x] Write a prompt guiding the agent to explore the project and understand hparams
- [x] Write a prompt that filters hparams according to command
- [x] Implement project structure scanner
- [x] Implement unified experiment runner
- [x] Implement run history (JSONL)
- [x] Implement experiment report generator
- [x] Complete SKILL.md with full agent workflow
- [x] Implement session manager with local + SSH support

### P1: Stability & Intelligence
- [ ] Implement multi-run orchestration with GPU scheduling
- [ ] Integrate Optuna via Hydra sweeper backend
- [ ] Add failure-aware tuning (auto-retry, OOM recovery)
- [ ] Implement reproducibility metadata tracking (package versions, CUDA/driver)
- [ ] Add early rejection for unpromising trials

### P2: Agentic Advantage
- [ ] LLM-based tuning explanation generation
- [ ] Human-in-the-loop approval workflow
- [ ] Automatic experiment summary reports with insights
- [ ] Warm-start from previous experiment history

### P3: Productization
- [ ] Interactive dashboard (CLI + HTML)
- [ ] Parameter importance analysis
- [ ] Multi-objective optimization
- [ ] Non-Hydra to Hydra config conversion assistant
- [ ] Benchmark suite for validation

## Citing

If you find this project useful in your research please cite Hydra and AHT using the following BibTeX entries:

```bibtex
@Misc{Zhang2026AHT,
  author =       {Xinhong Zhang, Weipu Zhang, Haolin Chen},
  title =        {AHT: Automatic Hyperparameter Tuning with Coding Agents using Hydra},
  howpublished = {Github},
  year =         {2026},
  url =          {https://github.com/zxh0916/auto-hparam-tuning}
}
```
```bibtex
@Misc{Yadan2019Hydra,
  author =       {Omry Yadan},
  title =        {Hydra - A framework for elegantly configuring complex applications},
  howpublished = {Github},
  year =         {2019},
  url =          {https://github.com/facebookresearch/hydra}
}
```
