# AHT: Automatic Hyperparameter Tuning with 🦞OpenClaw

> Tell the agent what to optimize. It reads your project, plans a strategy, runs experiments, and learns from each result — when you are enjoying your coffee.

[中文版 Readme](README_zh.md)

**TL;DR: AHT** is an [OpenClaw](https://github.com/openclaw/openclaw) skill that turns a coding agent into an autonomous hyperparameter tuning researcher for any deep learning project built on [Hydra](https://hydra.cc/).

Hyperparameter tuning remains one of the most tedious bottlenecks in deep learning research. Traditional search methods — grid search, random search, and Bayesian optimizers like [Optuna](https://optuna.org/) — treat the hyperparameter space as a black box: they sample configurations, evaluate metrics, and repeat, without ever reading a line of code or understanding *why* a learning rate of 1e-3 works better than 1e-2. Researchers, on the other hand, bring intuition — they read the model, inspect loss curves, and reason about what to try next. But that intuition is expensive: it demands hours of manual intervention and context-switching between experiments.

AHT bridges this gap. It teaches a coding agent to tune hyperparameters the way a researcher would — by reading the project first, then reasoning about what to change — while inheriting the tirelessness of automated search: it runs overnight, manages its own experiment queue, and wakes itself up when a training job finishes.

## Overview

AHT takes a fundamentally different approach. Instead of blind search, it equips a coding agent with the tools to **understand** the project first, then **reason** about what to change next:

1. **Read** — The agent walks through the codebase, parses the Hydra config hierarchy, and produces structured documentation (`PROJECT.md`, `HPARAM.md`) that captures the model architecture, training pipeline, and tunable knobs.
2. **Plan** — Before any experiment runs, the agent drafts a tuning strategy: which hyperparameters to prioritize, what ranges make sense given the architecture, and what patterns to watch for.
3. **Run** — Training commands are launched asynchronously in detached tmux sessions (locally or over SSH). The agent polls for completion, estimates ETAs, and uses cron reminders to wake itself up — no human babysitting required.
4. **Analyze** — After each run, TensorBoard event files are parsed into structured scalar summaries. The agent detects divergence, plateaus, and overfitting, then logs its findings in a cumulative report.
5. **Learn** — Each subsequent tuning decision is informed by the full run history: past overrides, metric trends, and the agent's own analysis. This closed loop lets the agent refine its strategy over time rather than exploring blindly.

The result is an iterative, context-aware tuning process that combines the rigor of systematic experimentation with the intuition of an experienced researcher — running autonomously from the first experiment to the final report.

### Compare to autonomous research / tuning approaches

Compare to existing autoresearch-like approaches, AHT occupies a very specific point in the design space: **skill-form**, **Hydra-native**, **low-intrusion**, and **tuning-focused**:

| Repo | Scope | As a skill | Platform support | intrusiveness to existing workflows |
| --- | --- | --- | --- | --- |
| [uditgoenka/autoresearch](https://github.com/uditgoenka/autoresearch) | general optimization / autonomous iteration | ✅ | Claude Code | High |
| [ARIS ⚔️](https://github.com/wanshuiyin/Auto-claude-code-research-in-sleep) | ML research workflows | ✅ | Claude Code / Codex / OpenClaw / any LLM agent | Medium |
| [aiming-lab/AutoResearchClaw](https://github.com/aiming-lab/AutoResearchClaw) | full autonomous research (idea → paper) | ❌ | OpenClaw / Claude Code / CLI | High |
| [HKUDS/ClawTeam](https://github.com/HKUDS/ClawTeam) | multi-agent orchestration for autonomous experiments | ✅ | Claude Code / Codex / OpenClaw / nanobot / Cursor / custom CLI agents | Medium |
| [karpathy/autoresearch](https://github.com/karpathy/autoresearch) | autonomous ML experimentation on a small LLM training repo | ❌ | - | It's an independent project |
| [facebookresearch/how-to-autorl](https://github.com/facebookresearch/how-to-autorl) | RL hparam tuning | ❌ | Hydra | Low |
| **[AHT](https://github.com/zxh0916/auto-hparam-tuning)** | **hparam tuning for Hydra projects** | ✅ | 🦞 **OpenClaw** (more on the way) | **Low** |

## ✨ Features

### Project and config understanding

AHT walks through the target project to identify the entry script, Hydra config structure, and tunable hyperparameters, producing `PROJECT.md` and `HPARAM.md` as structured references for subsequent tuning decisions.

### TensorBoard event analysis

AHT exposes TensorBoard scalar data to the agent, allowing it to detect training patterns such as divergence, plateaus, and overfitting from logged metrics.

### Context-aware tuning with run histories

In each tuning iteration, AHT spawns a subagent with the project overview, historical overrides, tuning strategy, and accumulated results as context, allowing it to learn from past runs and make informed decisions for the next override.

### Async execution with tmux

Training runs are launched in detached tmux sessions (both locally and over SSH), enabling the agent to poll for completion, estimate ETAs, and set cron reminders instead of blocking.

### Experiment history and reporting

AHT maintains a structured session directory (`aht/yyyy-mm-dd/hh-mm-ss/`) with per-run configs, metrics, and analysis. A built-in reporting script can generate summary, Markdown, or HTML reports comparing runs.

## 🔄 Workflow

1. **Understand the project** — Inspect the project structure and Hydra config hierarchy; generate `PROJECT.md` and `HPARAM.md` if missing.
2. **Understand the run command** — Analyze the user-provided training command to identify active configs, output paths, metric candidates, and relevant hyperparameters.
3. **Create a session** — Initialize a tuning session with the base command, primary metric, and optimization goal; auto-insert `- override` into the Hydra defaults list.
4. **Tuning loop** (baseline run + up to *N* iterations):
   1. Spawn a subagent to decide the best override based on the strategy and run history.
   2. Launch the run in a detached tmux session.
   3. Poll the run status; set a cron reminder if still running.
   4. Once finished, spawn a subagent to analyze the TensorBoard event file and update the report.
5. **Finalize** — Present the final report and best configuration to the user.

## 🚀 Quick Start

1. Clone the repo into your global skill directory and install the dependencies:
```bash
cd ~/.openclaw/skills
git clone https://github.com/zxh0916/auto-hparam-tuning.git
pip install -r auto-hparam-tuning/requirements.txt
```

2. Modify your OpenClaw config:
```json
{
  "skills": {
    "load": {
      "extraDirs": [
        "~/.openclaw/skills/auto-hparam-tuning/skills"
      ]
    },
    "entries": {
      "auto-hparam-tuning": { "enabled": true },
      "aht-init": { "enabled": true }
    }
  }
}
```

### Usage

```
/skill auto-hparam-tuning Please tune the project "/path/to/project" in "some_remote_machine", use remote conda environment "some_remote_conda_env" and local conda environment "some_local_conda_env".
```

## 📝 TODO List

- [ ] Add support to Codex and ClaudeCode
- [ ] Add a helper to transfer an existing non-hydra project into hydra-based one
- [ ] Add support to specify model for tuning and analyzing subagents
- [ ] ...

## 🤗 Citing

If you find this project useful in your research, please cite Hydra and AHT using the following BibTeX entries:

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

If you have any question, feel free to create an issue or join the wechat group:

<img src="imgs/wechat_group_20260318.jpeg" style="zoom:25%;" />

## Star History

<a href="https://www.star-history.com/?repos=zxh0916%2Fauto-hparam-tuning&type=timeline&legend=top-left">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/image?repos=zxh0916/auto-hparam-tuning&type=timeline&theme=dark&legend=top-left" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/image?repos=zxh0916/auto-hparam-tuning&type=timeline&legend=top-left" />
   <img alt="Star History Chart" src="https://api.star-history.com/image?repos=zxh0916/auto-hparam-tuning&type=timeline&legend=top-left" />
 </picture>
</a>