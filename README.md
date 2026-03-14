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

- **Project Structure Scanning** — Automatically detect training entry points, Hydra config directories, TensorBoard event paths, and checkpoint locations (`scripts/scan_project.py`)
- **Hyperparameter Extraction & Classification** — Parse Hydra configs to identify tunable parameters, classify types (int/float/bool/categorical), infer search ranges and priorities (`prompts/filter_hparams.md`)
- **Metric Understanding & Objective Definition** — Determine primary optimization metric, direction (max/min), evaluation strategy, and constraints (`prompts/define_objective.md`)
- **TensorBoard Event Analysis** — Load event files into DataFrames and compute 25+ curve statistics including trend, oscillation, and robust range (`scripts/analyze_event.py`)
- **Training Curve Diagnosis** — Automatically detect divergence, instability, plateau, overfitting, underfitting, early saturation, and slow training (`scripts/detect_patterns.py`)
- **Two-Layer Tuning Policy** — Rule-based deterministic adjustments combined with LLM contextual reasoning for next-parameter suggestions (`prompts/suggest_next_params.md`)
- **Unified Experiment Runner** — Launch training with Hydra overrides, enforce timeouts, detect OOM/NaN failures (`scripts/run_experiment.py`)
- **Experiment History Tracking** — JSONL-based run database with git hash, system info, metrics, and diagnosis for full reproducibility (`scripts/run_history.py`)
- **Automated Report Generation** — Markdown reports with top-k rankings, parameter-result comparison, diagnosis distribution (`scripts/report.py`)

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
    ├── analyze_event.py              # TensorBoard event → curve statistics (JSON)
    ├── detect_patterns.py            # Curve statistics → diagnostic labels (JSON)
    ├── scan_project.py               # Project directory → ProjectInfo (JSON)
    ├── run_experiment.py             # Launch training with overrides & capture results
    ├── run_history.py                # JSONL experiment database (record/list/best)
    └── report.py                     # History → Markdown experiment report
```

## Usage

This is an **agent skill** — you don't run the scripts manually. Simply tell your coding agent:

> "帮我调一下 `/path/to/my/project` 的超参数，目标是最小化 `val/loss`"

The agent will automatically:
1. Scan your project structure and understand its Hydra configuration
2. Identify and classify tunable hyperparameters
3. Run baseline training if needed
4. Analyze training curves and diagnose issues
5. Suggest and execute improved hyperparameter configurations
6. Iterate until the objective is met or the budget is exhausted
7. Generate a final report with the best configuration

### Minimal Trigger

```
Tune my project at /path/to/project
```

The agent discovers the metric and everything else automatically.

### Full Trigger

```
Tune hyperparameters for /path/to/project.
Target metric: val/accuracy (maximize).
Budget: 15 iterations, 2h timeout per trial.
```

## Script Reference (for developers)

The scripts in `scripts/` are internal tools used by the agent. They can also be run standalone for debugging:

| Script | Purpose | Example |
|--------|---------|---------|
| `scan_project.py` | Detect project structure | `python scripts/scan_project.py --project /path` |
| `analyze_event.py` | Parse TensorBoard events | `python scripts/analyze_event.py event_file val/loss --mode min` |
| `detect_patterns.py` | Diagnose training curves | `python scripts/detect_patterns.py event_file val/loss --mode min` |
| `run_experiment.py` | Launch training with overrides | `python scripts/run_experiment.py --project /path --entry train.py --overrides "lr=0.001"` |
| `run_history.py` | Record/query experiment history | `python scripts/run_history.py record --project /path --run-name test --metrics '{}'` |
| `report.py` | Generate experiment report | `python scripts/report.py --project /path --metric val/accuracy --mode max` |

## Operating Modes

| Mode | Description |
|------|-------------|
| `manual-assist` | Analyze and suggest, user executes manually |
| `rule-based-auto` | Deterministic rule-based adjustments, auto-execute |
| `llm-guided-auto` | LLM reasoning guides parameter selection, auto-execute |
| `optuna-backed-auto` | Bayesian optimization via Hydra Optuna sweeper |

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
