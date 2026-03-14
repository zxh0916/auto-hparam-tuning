# AHT: Automatic Hyperparameter Tuning with Coding Agents using Hydra

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

## Features

- Automatically explore the project and understand the hparams
- Extract tunable hparams from commands
- Analyze the critical metrics from the tensorboard event file
- Judge the run with the metrics
- Tune the hparam according to the result
- Write tuning logs into disk
- Create stable AHT session directories under the tuned project, both locally and over SSH

## Usage

This is an **agent skill** — you don't run the scripts manually. Simply tell your coding agent:

> "帮我调一下 `/path/to/my/project` 的超参数，运行命令是 `python train.py task=foo`，目标是最小化 `val/loss`"

The agent will automatically:
1. Understand the project structure and Hydra configuration
2. Analyze the run command to identify relevant hparams and expected outputs
3. Create an AHT session and run baseline training
4. Analyze training curves and tune hyperparameters iteratively
5. Generate a final report with the best configuration

### Minimal Trigger

```
Tune my project at /path/to/project, command: python train.py task=foo
```

### Full Trigger

```
Tune hyperparameters for /path/to/project.
Command: python train.py task=cifar10
Target metric: val/accuracy (maximize).
Budget: 15 iterations.
Remote: user@gpu-server
```

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

A lightweight project-understanding flow helper is also available:

```bash
python skills/auto-hparam-tuning/scripts/project_understanding.py inspect-project /path/to/project
python skills/auto-hparam-tuning/scripts/project_understanding.py prepare-run-understanding /path/to/project "python train.py task=foo"
python skills/auto-hparam-tuning/scripts/project_understanding.py --ssh-host user@server inspect-project /remote/project/path
```

## Architecture

```
skills/auto-hparam-tuning/
├── SKILL.md                              # Agent skill manifest & workflow
├── prompts/
│   ├── generate_project_md.md            # Generate PROJECT.md for the target project
│   ├── get_hparam_structure.md           # Generate HPARAM.md for Hydra config systems
│   └── understand_run_command.md         # Session-specific command understanding
└── scripts/
    ├── analyze_event.py                  # TensorBoard event → curve statistics
    ├── project_understanding.py          # Project inspection & understanding flow
    └── session_manager.py                # Session/run lifecycle management (local + SSH)
```

## Roadmap

- [x] Convert `pd.DataFrame` from TensorBoard event file
- [x] Implement session manager with local + SSH support
- [x] Project understanding flow (PROJECT.md, HPARAM.md, run command analysis)
- [ ] Multi-GPU parallel trial scheduling
- [ ] Optuna-backed Bayesian optimization via Hydra sweeper
- [ ] Interactive dashboard

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
