---
name: aht-init
description: Collect and confirm the minimum required initialization info before starting any AHT run: project path, environment or conda env name, reference launch script/command, and optimization target. Always send a user-facing confirmation request first, even when the values seem inferable from context, and wait for the user to confirm or update them before continuing.
---

# AHT Init

Use this sub-skill as the **front door** of the AHT workflow.

Its job is to gather the minimum initialization information, present the concrete values back to the user, and obtain explicit confirmation **before** the main AHT workflow starts.

## Required Fields

Collect and confirm these four items:

1. **Project path**: the target project root path
2. **Environment**: the runtime environment or conda environment name used to run training
3. **Reference launch script/command**: the launch script or exact training command to tune
4. **Optimization target**: the metric/objective to optimize, ideally with direction (`maximize`/`minimize`)

## Core Rules

1. **Extract candidate values from context when possible.**
   - Reuse values already present in the current conversation, task context, pasted command, or provided script path.
   - If a value can be inferred directly from the supplied command or path, treat it as a candidate value.

2. **Always ask for confirmation before execution.**
   - Even if all four fields appear available, do **not** proceed directly.
   - Always send one user-facing confirmation message that shows the current values and asks the user to confirm or update them.
   - Treat unconfirmed inferred values as provisional.

3. **Block AHT until the user responds.**
   - After sending the confirmation request, stop and wait.
   - Continue to the main `auto-hparam-tuning` workflow only after the user explicitly confirms the values or provides corrections.

4. **Keep normalization minimal.**
   - Preserve the user's wording when possible.
   - Normalize only enough to make downstream AHT steps unambiguous.

## Output Contract

### Case A: first-pass collection or unconfirmed information

Produce one concise English confirmation message to the user in this style:

```text
Before I start AHT, please confirm or update the initialization info:
- project path: /path/to/project
- environment / conda env: my-env
- reference launch command: python train.py ...
- optimization target: val/acc (maximize)

Reply with “confirm” if this is correct, or send the corrected fields.
```

Requirements:

- Always include all four fields.
- If a field is unknown, write `TBD` and ask the user to fill it in.
- Keep the message concise and action-oriented.
- Do not start project inspection, session creation, or any AHT execution in the same turn.

### Case B: user has explicitly confirmed or updated the information

Produce a short readiness summary containing:

- `project_path`
- `conda_env` (or equivalent environment name)
- `reference_command`
- `optimization_target`
- `goal` (`maximize` or `minimize`, if derivable)

Then explicitly state that AHT can proceed to project/run understanding.

## Extraction Hints

- If the user gives `cd /path/to/proj && conda activate xxx && bash train.sh`, you can extract provisional values such as:
  - project path: `/path/to/proj`
  - environment / conda env: `xxx`
  - reference command/script: `bash train.sh`
- If the user says `optimize val/acc`, infer:
  - optimization target: `val/acc`
  - goal: `maximize`
- If the user says `minimize val/loss`, infer:
  - optimization target: `val/loss`
  - goal: `minimize`
- If the metric name is given without direction and direction is not obvious, keep the metric and leave the direction as part of the confirmation request.

## Handoff to Main AHT Skill

Only after the user confirms or corrects the initialization info should you continue with the main `auto-hparam-tuning` workflow:

1. inspect project docs / hparam docs
2. understand the reference run command
3. create the AHT session
4. enter baseline run + tuning loop

This sub-skill is intentionally small and reusable. It is only responsible for **initial information extraction + explicit user confirmation gate**.