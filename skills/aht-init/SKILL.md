---
name: aht-init
description: Collect and confirm the minimum required initialization info before starting any AHT run: project path, environment or conda env name, reference training launch script/method, and optimization target. Always send a user-facing confirmation request first, even when the values seem inferable from context, and wait for the user to confirm or update them before continuing.
---

# AHT Init

Use this sub-skill as the **front door** of the AHT workflow.

Its job is to gather the minimum initialization information, present the concrete values back to the user, and obtain explicit confirmation **before** the main AHT workflow starts.

## Required Fields

Collect and confirm these four user-provided items:

1. **Project path**: the target project root path
2. **Environment**: the runtime environment or conda environment name used to run training
3. **Reference training launch method**: the launch script or exact training invocation to tune
4. **Optimization target**: the metric/objective to optimize, ideally with direction (`maximize`/`minimize`)

Also identify one internal runtime field before handing off to the main AHT workflow:

5. **Calling agent**: detect whether the skill is currently being invoked by `codex`, `claudecode`, or `openclaw`

## Core Rules

1. **Extract candidate values from context when possible.**
   - Reuse values already present in the current conversation, task context, pasted command, or provided script path.
   - If a value can be inferred directly from the supplied command or path, treat it as a candidate value.

2. **Detect the calling agent before handoff.**
   - Determine whether the current caller is `codex`, `claudecode`, or `openclaw`.
   - Prefer explicit runtime/context signals over guesswork.
   - If multiple signals conflict, choose the most direct one from the current runtime/tooling context and note low confidence internally.
   - If no reliable signal is available, default to `openclaw`.

3. **Always ask for confirmation before execution.**
   - Even if all four user fields appear available, do **not** proceed directly.
   - Always send one user-facing confirmation message that shows the current values and asks the user to confirm or update them.
   - Treat unconfirmed inferred values as provisional.
   - The detected `calling_agent` is internal metadata; do not ask the user to confirm it unless they specifically ask.

4. **Block AHT until the user responds.**
   - After sending the confirmation request, stop and wait.
   - Continue to the main `auto-hparam-tuning` workflow only after the user explicitly confirms the values or provides corrections.

5. **Keep normalization minimal.**
   - Preserve the user's wording when possible.
   - Normalize only enough to make downstream AHT steps unambiguous.

## Output Contract

### Case A: first-pass collection or unconfirmed information

Produce one concise English confirmation message to the user in this style:

```text
Before I start AHT, please confirm or update the initialization info:
- project path: /path/to/project
- environment / conda env: my-env
- reference training launch method: python train.py ...
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
- `calling_agent` (`codex`, `claudecode`, or `openclaw`)

Then explicitly state that AHT can proceed to project/run understanding.

## Extraction Hints

- If the user gives `cd /path/to/proj && conda activate xxx && bash train.sh`, you can extract provisional values such as:
  - project path: `/path/to/proj`
  - environment / conda env: `xxx`
  - reference training launch method/script: `bash train.sh`
- If the user says `optimize val/acc`, infer:
  - optimization target: `val/acc`
  - goal: `maximize`
- If the user says `minimize val/loss`, infer:
  - optimization target: `val/loss`
  - goal: `minimize`
- If the metric name is given without direction and direction is not obvious, keep the metric and leave the direction as part of the confirmation request.
- For `calling_agent`, use runtime/context/tooling signals such as:
  - `codex` if the current runtime or harness is clearly Codex
  - `claudecode` if the caller is clearly Claude Code
  - `openclaw` if invoked directly in OpenClaw or if no stronger signal is available

## Handoff to Main AHT Skill

Only after the user confirms or corrects the initialization info should you continue with the main `auto-hparam-tuning` workflow:

1. Read the skill documentation of `auto-hparam-tuning`
2. Start from `1. UNDERSTAND PROJECT`...

This sub-skill is intentionally small and reusable. It is only responsible for **initial information extraction + explicit user confirmation gate**.