Given a target project root and a specific run command, analyze only the code, configs, scripts, and docs that are relevant to this tuning session. Use English.

Primary goal:
Produce a concise but high-value session-specific understanding of what this command actually runs, which config and code paths it activates, which hyperparameters are most relevant, and where outputs / TensorBoard metrics will likely appear.

Inputs you should assume are available:
- project root
- run command
- any existing project-level docs such as `CLAUDE.md`, `AGENTS.md`, or `PROJECT.md`
- any existing hyperparameter structure guide such as `HPARAM.md`

Scope rule:
Do not try to summarize the whole repository again. Start from the given run command and trace outward only as needed.

Required tasks:

1. Resolve the actual runtime path
- Identify the executable entrypoint script / module launched by the command.
- Identify the main function or launcher it reaches.
- State where configuration enters runtime.

2. Resolve the active configuration chain
- Parse the command and list explicit CLI overrides.
- Distinguish value overrides from config-group switches.
- Identify the top-level config root and the composition root(s) actually used by this command.
- Name the config groups and files that are most relevant to this session.

3. Identify session-relevant code
- Point out the most relevant source files for this run.
- Focus on files that control:
  - task setup
  - model construction
  - optimizer / scheduler
  - data pipeline
  - trainer / loop
  - logging / checkpointing
- Ignore unrelated modules.

4. Infer output and logging locations
- Determine where checkpoints, logs, and TensorBoard event files are likely written.
- If Hydra is used, explain the run dir / sweep dir pattern that applies to this command.
- Mention where the resolved config and override history will be stored after launch.

5. Identify the likely optimization target
- List the main metric candidates this run may produce.
- State which metric is the best primary tuning target for this session if it can be inferred.
- State whether it should be maximized or minimized.
- If uncertain, say what quick baseline run or artifact inspection would disambiguate it.

6. Extract tuning-relevant hyperparameters
- Identify the hyperparameters most likely to matter for this command.
- Group them into categories such as:
  - optimization
  - model capacity
  - data / batch / rollout
  - loss weights
  - regularization
  - logging / evaluation cadence
- Identify **Runtime Truncation Knobs**: how to artificially shorten a run to
  save time (e.g., `trainer.max_epochs=2`, `trainer.limit_train_batches=0.1`,
  `max_steps=5000`). Required for a Progressive tuning strategy.
- Separate:
  - good first-round tuning knobs
  - dangerous / expensive knobs that should not be changed early

7. Produce concrete command-aware guidance
- Give 3-8 repo-specific CLI override examples that are relevant to this command.
- Include at least one safe baseline command for verifying the run.
- If sweeps are supported, show the canonical sweep form for this command.

Output format:
Write a concise structured note into the report after sessions creation with these headings:

## Session Run Understanding
### Command Summary
### Runtime Path
### Active Config Chain
### Relevant Files
### Outputs and TensorBoard
### Primary Metric Recommendation
### Runtime Truncation Knobs
### Tuning Knobs for This Session
### Safe First Actions

Style requirements:
- Be specific and repository-grounded.
- Prefer exact paths and concrete command fragments.
- Call out uncertainty explicitly instead of guessing.
- Keep it concise but actionable.

Important behavior rule for future agents:
Before changing code for a tuning task, inspect the command-specific entrypoint, the active config composition chain, the logging/output path, and the most relevant task/model/optimizer files touched by this command.
