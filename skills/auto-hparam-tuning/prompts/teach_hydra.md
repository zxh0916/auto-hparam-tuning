# Hydra Framework Primer for Coding Agents

You are a coding agent working with a machine learning project that uses **Hydra** (by Facebook Research) for configuration management. This document teaches you the essential Hydra concepts you need to effectively navigate, modify, and tune hyperparameters in Hydra-based projects.

## What is Hydra?

Hydra is a framework for elegantly configuring complex applications. It allows ML projects to:
- Define hierarchical configurations in YAML files
- Override any config value from the command line
- Compose configs from multiple sources (config groups)
- Run parameter sweeps (multirun) with a single command
- Integrate with optimization frameworks like Optuna

## Core Concepts

### 1. Config Directory and Entry Point

Every Hydra project has:
- A **config directory** (typically `conf/`, `configs/`, or `config/`) containing YAML files
- A **Python entry point** decorated with `@hydra.main()`:

```python
import hydra
from omegaconf import DictConfig

@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    print(cfg.optimizer.lr)  # access config values with dot notation

if __name__ == "__main__":
    main()
```

Key parameters of `@hydra.main()`:
- `config_path`: relative path from the Python file to the config directory
- `config_name`: name of the root config file (without `.yaml` extension)

### 2. Config Structure and Composition

Configs are organized hierarchically:

```
conf/
├── config.yaml          # root config (composition root)
├── model/
│   ├── resnet.yaml
│   └── transformer.yaml
├── optimizer/
│   ├── adam.yaml
│   └── sgd.yaml
├── dataset/
│   ├── cifar10.yaml
│   └── imagenet.yaml
└── trainer/
    └── default.yaml
```

The root `config.yaml` uses a **defaults list** to compose from groups:

```yaml
defaults:
  - model: resnet
  - optimizer: adam
  - dataset: cifar10
  - trainer: default

# Additional values defined directly
seed: 42
debug: false
```

Each config group file (e.g., `optimizer/adam.yaml`) contains:
```yaml
_target_: torch.optim.Adam
lr: 0.001
weight_decay: 0.0001
betas: [0.9, 0.999]
```

### 3. CLI Overrides

Hydra allows overriding any config value from the command line:

```bash
# Override a single value
python train.py optimizer.lr=0.0003

# Override multiple values
python train.py optimizer.lr=0.0003 model.dropout=0.2 seed=123

# Switch config group choice
python train.py optimizer=sgd model=transformer

# Override a value within a switched group
python train.py optimizer=sgd optimizer.lr=0.01 optimizer.momentum=0.9

# Append a new key (use +)
python train.py +experiment_name=my_run

# Remove a key (use ~)
python train.py ~model.dropout
```

**Override syntax rules**:
- `key=value` — override an existing key
- `+key=value` — add a new key that does not exist in the config
- `~key` — remove a key from the config
- `group=choice` — switch a config group to a different file

### 4. Multirun (Sweep)

Hydra supports running multiple configurations in sequence or parallel:

```bash
# Sweep over learning rates (use -m or --multirun flag)
python train.py -m optimizer.lr=0.001,0.0003,0.0001

# Grid sweep over multiple parameters
python train.py -m optimizer.lr=0.001,0.0003 model.dropout=0.1,0.3,0.5

# Range sweep
python train.py -m 'optimizer.lr=range(0.0001,0.01,0.001)'

# Glob sweep over config groups
python train.py -m 'model=glob(*)'
```

### 5. Optuna Sweeper Integration

For intelligent Bayesian optimization, Hydra integrates with Optuna via `hydra-optuna-sweeper`:

```yaml
# In config.yaml or a sweep config
defaults:
  - override hydra/sweeper: optuna

hydra:
  sweeper:
    sampler:
      _target_: optuna.samplers.TPESampler
      seed: 42
    direction: minimize  # or maximize
    n_trials: 50
    n_jobs: 1
    params:
      optimizer.lr:
        type: float
        low: 1e-5
        high: 1e-2
        log: true
      model.dropout:
        type: float
        low: 0.0
        high: 0.5
      model.num_layers:
        type: int
        low: 2
        high: 8
```

Launch with: `python train.py -m`

### 6. Output Directory Structure

Hydra automatically creates an output directory for each run:

```
outputs/
└── 2026-03-15/
    └── 14-30-22/          # timestamp-based
        ├── .hydra/
        │   ├── config.yaml     # resolved config for this run
        │   ├── overrides.yaml  # CLI overrides used
        │   └── hydra.yaml      # Hydra's own config
        ├── main.log            # application log
        └── ...                 # your outputs (checkpoints, events, etc.)
```

For multirun:
```
multirun/
└── 2026-03-15/
    └── 14-30-22/
        ├── 0/    # first trial
        ├── 1/    # second trial
        └── 2/    # third trial
```

### 7. Config Inspection Commands

Before modifying anything, always inspect the resolved config:

```bash
# Print the resolved config (all compositions applied)
python train.py --cfg job

# Print Hydra's own config
python train.py --cfg hydra

# Print with overrides
python train.py optimizer.lr=0.0003 --cfg job

# Show detailed info about config composition
python train.py --info config

# Show all available config groups and choices
python train.py --info defaults

# Show search path
python train.py --info searchpath
```

## Agent Workflow Rules

When working with a Hydra project:

1. **Always inspect before modifying**: Run `--cfg job` to see the full resolved config before making changes. Never guess at config structure.

2. **Prefer CLI overrides over file edits**: Change hyperparameters via Hydra overrides (`key=value`) rather than editing YAML files directly. This preserves the original config and enables reproducibility.

3. **Understand the defaults list**: Before switching config groups, read the `defaults:` section in the root config to understand the composition chain.

4. **Check `.hydra/overrides.yaml`**: For previous runs, this file records exactly which overrides were used, enabling reproducibility.

5. **Use `--cfg job` to validate overrides**: After constructing override strings, verify them with `--cfg job` before launching expensive training runs.

6. **Know the output directory convention**: Look in `outputs/` (single run) or `multirun/` (sweep) for past results, configs, and logs.

7. **Respect the config hierarchy**: When suggesting hyperparameter changes, express them as dot-separated paths matching the config structure (e.g., `optimizer.lr`, not just `lr`).
