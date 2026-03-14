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

## CSV Experiment History

The experiment backend now uses the session layout under `<project_root>/aht/` as the source of truth.
Each session keeps a canonical `results.csv` index plus per-run payloads such as `resolved_config.json`,
`metrics.json`, `summary.md`, and the session-level `report.md`.

Compatibility helpers that preserve the old experiment-history workflow now read and write this CSV-backed store:

```bash
python skills/auto-hparam-tuning/scripts/init_experiment_history_db.py --project-root /path/to/project
python skills/auto-hparam-tuning/scripts/query_experiment_history.py --project-root /path/to/project --limit 20 --format json
python skills/auto-hparam-tuning/scripts/report_experiment_history.py --project-root /path/to/project summary
python skills/auto-hparam-tuning/scripts/report_experiment_history.py --project-root /path/to/project report --output-markdown /tmp/report.md --output-html /tmp/report.html
python skills/auto-hparam-tuning/scripts/iter_next_hparams.py --project-root /path/to/project --base-config-file resolved_config.yaml --search-space-file search_space.yaml --train-command-template 'python train.py --config {config_path} --metrics-out {metrics_path}'
```

`iter_next_hparams.py` keeps the `exp-launch` planning behavior: it inspects prior runs across the project,
tries the baseline first, then one-step neighbors of the current best run, then the nearest untried grid point.
Instead of appending a SQLite row, it records the run into the active AHT session and updates `results.csv`,
`metrics.json`, `resolved_config.json`, and `report.md`.

### Additional Features

- Convert the non-hydra config system into hydra-based ones
- Try multiple config in parallel across multiple GPUs
- A more user-friendly dashboard compare to tensorboard

## Roadmap

- [x] convert pd.DataFrame from tensorboard event file
- [ ] write some functions to detect patterns from the curves
- [ ] write a prompt that teaches the agent using hydra
- [ ] write a prompt guiding the agent to explore the project and understand the hparams (refer to the prompts of `/init` command of coding agents like codex and claude code maybe)
- [ ] write a prompt that makes the agent filter the hparams according to the command

## Citing

If you find this project useful in your research please cite hydra and AHT using the following BibTeX entry:

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
