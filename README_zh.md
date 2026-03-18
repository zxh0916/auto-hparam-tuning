# AHT：基于 🦞OpenClaw 的自动超参数调优

> 你只需要告诉 agent 优化什么，剩下的——读代码、定策略、跑实验、总结经验——它都能自己搞定。你喝咖啡就好。

[English README](README.md)

**一句话介绍：** AHT 是一个 [OpenClaw](https://github.com/openclaw/openclaw) 技能（skill），能让 coding agent 变成你的自动调参助手，适用于任何基于 [Hydra](https://hydra.cc/) 的深度学习项目。

## 为什么需要 AHT？

调参大概是深度学习研究里最磨人的环节了。传统方法——网格搜索、随机搜索、[Optuna](https://optuna.org/) 之类的贝叶斯优化——本质上都是把超参数空间当黑盒：采样、评估、再采样，全程不看一行代码，也不关心 *为什么* 学习率 1e-3 比 1e-2 好。研究者当然有直觉：看看模型结构、瞅瞅 loss 曲线，就能猜下一步该试什么。但直觉是有代价的——你得盯着实验，来回切换，一耗就是好几个小时。

AHT 要做的事很简单：让 agent 像研究者一样调参——先读懂项目，再想清楚改什么——同时保留自动化方法的优势：不用人盯、整夜运行、训练跑完自动唤醒继续干活。

## 概述

AHT 不做盲目搜索。它让 agent 先**理解**项目，再**推理**下一步该怎么改：

1. **Read（阅读）** —— Agent 遍历代码库，解析 Hydra 配置层级，生成结构化文档（`PROJECT.md`、`HPARAM.md`），把模型架构、训练流程和可调参数都梳理清楚。
2. **Plan（规划）** —— 在跑任何实验之前，先制定调参策略：优先动哪些超参、取值范围多大合理、过程中重点关注什么信号。
3. **Run（运行）** —— 训练命令以异步方式在 tmux session 中后台启动（本地或 SSH 远程均可）。Agent 自己轮询进度、估算剩余时间，到点了用 cron 把自己叫醒——不需要人盯。
4. **Analyze（分析）** —— 每次运行结束后，解析 TensorBoard event 文件，提取标量指标。Agent 会判断是否出现了发散、停滞或过拟合，并把分析写进累计报告。
5. **Learn（学习）** —— 后续每一轮决策都能看到完整的实验历史：之前用了什么 override、指标怎么变的、agent 自己怎么分析的。这个闭环让策略越调越准，而不是在参数空间里瞎撞。

最终效果：一个迭代式、有上下文的调参流程，既有系统化实验的严谨，又有老手调参的直觉，从第一个实验到最终报告全程无人值守。

### 与同类方案的对比

和现有的 autoresearch 类项目相比，AHT 的定位很明确：**技能形态**、**Hydra 原生**、**低侵入**、**专注调参**。

| 项目 | 定位 | 技能形态 | 支持平台 | 对现有工作流的侵入性 |
| --- | --- | --- | --- | --- |
| [uditgoenka/autoresearch](https://github.com/uditgoenka/autoresearch) | 通用优化 / 自主迭代 | ✅ | Claude Code | 高 |
| [ARIS ⚔️](https://github.com/wanshuiyin/Auto-claude-code-research-in-sleep) | ML 研究工作流 | ✅ | Claude Code / Codex / OpenClaw / 任意 LLM agent | 中 |
| [aiming-lab/AutoResearchClaw](https://github.com/aiming-lab/AutoResearchClaw) | 全流程自动研究（从 idea 到论文） | ❌ | OpenClaw / Claude Code / CLI | 高 |
| [HKUDS/ClawTeam](https://github.com/HKUDS/ClawTeam) | 多 agent 协作的自主实验 | ✅ | Claude Code / Codex / OpenClaw / nanobot / Cursor / 自定义 CLI agent | 中 |
| [karpathy/autoresearch](https://github.com/karpathy/autoresearch) | 小型 LLM 训练项目上的自主 ML 实验 | ❌ | - | 独立项目 |
| [facebookresearch/how-to-autorl](https://github.com/facebookresearch/how-to-autorl) | RL 超参调优 | ❌ | Hydra | 低 |
| **[AHT](https://github.com/zxh0916/auto-hparam-tuning)** | **Hydra 项目的超参调优** | ✅ | 🦞 **OpenClaw**（更多平台支持中） | **低** |

## ✨ 功能特性

### 项目与配置理解

AHT 会遍历目标项目，识别入口脚本、Hydra 配置结构和可调超参，生成 `PROJECT.md` 和 `HPARAM.md` 供后续调参参考。

### TensorBoard 事件分析

AHT 把 TensorBoard 的标量数据暴露给 agent，让它能从训练日志中发现发散、停滞、过拟合等问题。

### 基于历史记录的上下文感知调参

每一轮迭代，AHT 都会启动一个 subagent，把项目概览、历史 override、调参策略和已有结果都喂给它，让它在充分了解过去实验的基础上决定下一步怎么调。

### 基于 tmux 的异步执行

训练任务在后台 tmux session 中启动（本地和 SSH 远程都支持），agent 可以轮询状态、估算剩余时间，任务没跑完就设个 cron 提醒，不会傻等。

### 实验历史与报告

AHT 维护结构化的 session 目录（`aht/yyyy-mm-dd/hh-mm-ss/`），每次运行的配置、指标和分析都有记录。内置报告脚本可以生成纯文本、Markdown 或 HTML 格式的对比报告。

## 🔄 工作流程

1. **理解项目** —— 检查项目结构和 Hydra 配置层级；如果没有 `PROJECT.md` 和 `HPARAM.md` 就自动生成。
2. **理解运行命令** —— 分析用户给的训练命令，搞清楚用了哪些配置、输出目录在哪、该看什么指标、哪些超参值得调。
3. **创建 session** —— 用基础命令、主指标和优化目标初始化一个调参 session，自动把 `- override` 插到 Hydra defaults 列表里。
4. **调参循环**（先跑一个 baseline，再迭代最多 *N* 轮）：
   1. 启动 subagent，根据当前策略和历史记录决定最优 override。
   2. 在后台 tmux session 中启动训练。
   3. 轮询运行状态；还没跑完就设 cron 提醒。
   4. 跑完后启动 subagent 分析 TensorBoard event 文件，更新报告。
5. **收尾** —— 把最终报告和最佳配置呈现给用户。

## 🚀 快速开始

1. 把仓库克隆到全局 skill 目录，安装依赖：

```bash
cd ~/.openclaw/skills
git clone https://github.com/zxh0916/auto-hparam-tuning.git
pip install -r auto-hparam-tuning/requirements.txt
```

2. 修改 OpenClaw 配置：

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

### 用法

```
/skill auto-hparam-tuning Please tune the project "/path/to/project" in "some_remote_machine", use remote conda environment "some_remote_conda_env" and local conda environment "some_local_conda_env".
```

#### 为子智能体单独设置模型

AHT支持通过在`openclaw.json`中设置环境变量来为负责超参调试和结果分析的子智能体单独指定模型:
```json
{
  "env": {
    "AHT_TUNING_MODEL": "minimax/minimax-m2.5",
    "AHT_ANALYZE_MODEL": "moonshot/kimi-k2.5"
  }
}
```
不填写这些环境变量时OpenClaw会使用当前智能体的默认模型(`agents.list[].model.primary`)。


## 📝 TODO

- [ ] 支持 Codex 和 Claude Code
- [ ] 提供工具帮助非 Hydra 项目迁移到 Hydra
- [ ] 支持为 tuning 和 analyzing subagent 指定模型
- [ ] ...

## 🤗 引用

如果这个项目对你的研究有帮助，请引用 Hydra 和 AHT：

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

有任何问题或者想法，欢迎提Issue或者加群讨论：

<img src="imgs/wechat_group_20260318.jpeg" style="zoom:25%;" />

## Star History

<a href="https://www.star-history.com/?repos=zxh0916%2Fauto-hparam-tuning&type=timeline&legend=top-left">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/image?repos=zxh0916/auto-hparam-tuning&type=timeline&theme=dark&legend=top-left" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/image?repos=zxh0916/auto-hparam-tuning&type=timeline&legend=top-left" />
   <img alt="Star History Chart" src="https://api.star-history.com/image?repos=zxh0916/auto-hparam-tuning&type=timeline&legend=top-left" />
 </picture>
</a>
