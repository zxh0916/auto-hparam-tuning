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