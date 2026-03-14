# AHT Session Layout Design

## Goal

Define a stable, code-enforced on-disk layout for automatic hyperparameter tuning sessions.

This document only specifies the **experiment bookkeeping layer**:

- where AHT writes outputs
- how a tuning session is organized
- how runs are numbered
- what artifacts each run should contain
- what gets written into `results.csv` and `report.md`

This document does **not** try to hard-code the tuning intelligence itself. Project exploration, metric selection, hypothesis generation, and tuning decisions should still be driven by the agent prompt and reasoning.

---

## Design Principles

### 1. Write into the target project, not into the skill repo

All AHT outputs should be written under the **project being tuned**.

AHT must never use the skill repository as the default location for experiment outputs.

### 2. Separate deterministic bookkeeping from open-ended reasoning

The following should be enforced by code:

- session directory creation
- run directory creation
- run numbering
- canonical filenames
- appending rows to `results.csv`
- appending sections to `report.md`

The following should remain prompt-driven:

- identifying the main metric
- deciding which hparams to tune
- proposing the next override
- interpreting curve shapes
- deciding whether a run is good, bad, inconclusive, or failed

### 3. Be resumable and inspectable

A human should be able to open the project folder and quickly inspect:

- what sessions have been run
- what each run changed
- what metrics were obtained
- why the next run was chosen

A future agent should also be able to resume from the directory structure without relying on hidden state.

### 4. Prefer simple files over opaque databases

The initial design uses plain files:

- markdown
- csv
- yaml
- json
- logs

This makes the system easier to debug and easier to version if needed.

---

## Root Directory Convention

For a target project rooted at:

```text
/project/root
```

AHT writes all outputs under:

```text
/project/root/aht/
```

This directory is owned by the target project.

---

## Session Directory Convention

Each tuning session creates a new directory with the format:

```text
aht/yyyy-mm-dd/hh-mm-ss/
```

Example:

```text
aht/2026-03-14/13-29-00/
```

### Rationale

- date groups sessions by day
- time distinguishes multiple sessions in one day
- path is human-readable
- easy to sort lexicographically

### Session creation rules

When a new session starts:

1. create `aht/` if it does not exist
2. create `aht/yyyy-mm-dd/` if it does not exist
3. create `aht/yyyy-mm-dd/hh-mm-ss/`
4. create required session files
5. create `runs/` directory inside the session

If the exact timestamp path already exists, the implementation may:

- wait one second and retry, or
- append a suffix such as `-01`

But the primary format should remain `yyyy-mm-dd/hh-mm-ss`.

---

## Session Layout

A complete session directory should look like this:

```text
aht/
  2026-03-14/
    13-29-00/
      meta.yaml
      results.csv
      report.md
      runs/
        0/
          override.yaml
          command.sh
          stdout.log
          stderr.log
          metrics.json
          summary.md
          copied/
            config.yaml
            event_file
        1/
          ...
```

---

## Session-Level Files

### 1. `meta.yaml`

Stores stable metadata for the session.

Suggested fields:

```yaml
project_root: /project/root
session_dir: /project/root/aht/2026-03-14/13-29-00
created_at: 2026-03-14T13:29:00+08:00
agent: openclaw
skill: auto-hparam-tuning
base_command: python train.py task=foo
primary_metric: val/acc
goal: maximize
status: running
notes: null
```

#### Purpose

- identify what project this session belongs to
- identify the base command being tuned
- record the optimization target if known
- support later resuming and auditing

#### Notes

- `primary_metric` may initially be null and be filled later
- `goal` may be `maximize`, `minimize`, or null initially
- `status` may be `running`, `completed`, `stopped`, or `failed`

---

### 2. `results.csv`

Tabular summary of all runs in the session.

Each row corresponds to one run.

#### Initial required columns

```csv
run_id,status,primary_metric,best_step,run_dir,override_path,start_time,end_time,notes
```

#### Column meanings

- `run_id`: integer run index, starting from 0
- `status`: `created`, `running`, `finished`, `failed`, `killed`, `inconclusive`
- `primary_metric`: scalar value for the chosen main metric, if available
- `best_step`: step/epoch corresponding to the selected metric value, if available
- `run_dir`: relative path like `runs/0`
- `override_path`: relative path like `runs/0/override.yaml`
- `start_time`: ISO timestamp
- `end_time`: ISO timestamp
- `notes`: short human-readable note

#### Extensibility

Additional columns may be added later, for example:

- `aux_metrics_json`
- `hostname`
- `device`
- `exit_code`
- `event_file`
- `hydra_output_dir`
- `git_commit`
- `wall_time_sec`

The initial implementation should not over-design this. A stable small core is enough.

---

### 3. `report.md`

Human-readable running report for the whole session.

This file should be append-only in normal use.

#### Suggested structure

```md
# AHT Report

## Session Summary
- Project:
- Created at:
- Base command:
- Primary metric:
- Goal:

## Run 0
### Hypothesis
...

### Override
...

### Result
...

### Analysis
...

### Next Decision
...
```

#### Purpose

- preserve the agent's reasoning in a readable form
- help the user understand why each run happened
- provide context for later resuming or summarizing

#### Important distinction

`results.csv` is for structured aggregation.

`report.md` is for narrative reasoning.

Both are needed.

---

## Run Directory Convention

Each session contains:

```text
runs/
```

Each run gets a numeric directory starting from `0`:

```text
runs/0/
runs/1/
runs/2/
```

### Run numbering rules

- numbering starts at `0`
- numbering is contiguous within a session
- each new run gets the smallest non-used non-negative integer
- in normal sequential usage this means incrementing by 1

### Why numeric IDs

- compact
- easy to sort
- easy to reference in CSV/report
- matches the mental model of iterative search

---

## Run-Level Files

Each run directory may contain the following canonical files.

### Required

#### `override.yaml`
The effective hyperparameter override proposed for this run.

This is the most important artifact for reproducing the run.

#### `command.sh`
The exact command used to launch the run.

Purpose:
- reproducibility
- debugging
- transparency for the user

#### `summary.md`
A concise run-local summary.

Suggested sections:
- objective of this run
- applied override
- observed result
- notable warnings/errors
- takeaway

### Strongly recommended

#### `stdout.log`
Captured standard output from the run.

#### `stderr.log`
Captured standard error from the run.

#### `metrics.json`
Machine-readable extracted metric summary.

Suggested content:

```json
{
  "primary_metric": "val/acc",
  "goal": "maximize",
  "best_value": 0.857,
  "best_step": 17500,
  "selected_rule": "max",
  "aux_metrics": {
    "train/loss": 0.42,
    "val/loss": 0.71
  }
}
```

### Optional

#### `copied/`
Artifacts copied back from the remote or generated output tree.

Examples:
- copied config file used in the run
- copied TensorBoard event file
- copied optimization result file
- copied checkpoint metadata

This folder is useful when the actual training happened remotely and outputs need to be preserved locally in a stable place.

---

## Lifecycle API Proposal

The bookkeeping layer should expose a minimal API. Names are illustrative.

### `create_session(project_root, base_command=None, ...) -> session_dir`

Responsibilities:
- resolve target project root
- create `aht/yyyy-mm-dd/hh-mm-ss/`
- initialize `meta.yaml`
- initialize `results.csv`
- initialize `report.md`
- create `runs/`

### `create_run(session_dir) -> run_id, run_dir`

Responsibilities:
- allocate next run id
- create `runs/<run_id>/`
- create empty or initial run artifacts as needed
- append an initial row to `results.csv` with status `created`

### `update_run_result(session_dir, run_id, fields)`

Responsibilities:
- update a row in `results.csv`
- record status, timing, metric summary, notes

### `append_report(session_dir, markdown)`

Responsibilities:
- append narrative content to `report.md`

### `finalize_session(session_dir, status)`

Responsibilities:
- update `meta.yaml`
- mark session `completed`, `failed`, or `stopped`

This API should stay deliberately small.

---

## What Should Be Code-Enforced

The following rules should be enforced by implementation, not merely suggested by prompt:

1. all outputs go under `<project_root>/aht/`
2. every tuning session creates a new timestamped session directory
3. every session contains `meta.yaml`, `results.csv`, `report.md`, and `runs/`
4. every run gets a numeric directory starting from 0
5. every run has a canonical location for overrides and logs
6. every run is represented in `results.csv`
7. the session report is appended to a stable `report.md`

These are structural guarantees and should not depend on model compliance.

---

## What Should Remain Prompt-Driven

The following should remain in the skill prompt and agent reasoning loop:

1. how to understand the target project's Hydra structure
2. how to find the true training entrypoint
3. how to choose the primary metric
4. how to determine maximize vs minimize
5. how to interpret training curves
6. how to propose the next override
7. whether to exploit locally, broaden search, or stop

These are inherently project-dependent and should not be prematurely rigid.

---

## Resume Semantics

The design should support resuming a partially completed session.

### Minimum resume behavior

Given a `session_dir`, AHT should be able to:

- read `meta.yaml`
- inspect existing `runs/`
- infer the next run id
- continue appending to `results.csv`
- continue appending to `report.md`

### Open question

Whether resume should default to the **latest session** under `aht/` or require an explicit `session_dir` can be decided later.

For the initial implementation, explicit resume is safer.

---

## Remote Execution Considerations

When training happens on a remote machine, the session layout should still live under the target project as the canonical record.

There are two reasonable modes:

### Mode A: canonical AHT directory is on the remote project

Use when the project itself lives remotely and all tuning work happens there.

### Mode B: canonical AHT directory is local, with copied remote artifacts

Use when the orchestrator is local and only training executes remotely.

For the current workflow, the intended interpretation appears to be:

- the AHT directory should live under the **project being tuned**
- if that project is remote, then AHT outputs belong there
- copied artifacts can still be mirrored locally when needed

This point should be made explicit in the eventual implementation.

---

## Failure Handling

The bookkeeping layer should tolerate failed runs.

If a run fails:

- keep the run directory
- keep the logs
- keep the attempted override
- mark the row in `results.csv` as `failed`
- append a short explanation to `report.md`

Failed runs are useful data and should not be discarded.

---

## Minimal First Implementation

The first implementation should be intentionally small.

### Phase 1

Implement only:

- session creation
- run creation
- `results.csv` initialization and row append/update
- `report.md` initialization and append
- canonical file locations

### Phase 2

Later add:

- resume support
- stricter schema validation
- richer `meta.yaml`
- convenience helpers for copying event files/configs
- automatic extraction of summary fields into `metrics.json`

### Phase 3

Potential future additions:

- multi-run dashboards
- aggregation across sessions
- visualization utilities
- automatic report summarization

---

## Suggested Prompt Integration

Once the bookkeeping layer exists, the skill prompt should instruct the agent along the following lines:

1. create a new AHT session under the target project's `aht/` directory
2. use the bookkeeping tool/API to allocate each run directory
3. save every applied override into the run directory
4. save command, logs, metric summary, and copied artifacts into the run directory
5. append structured results to `results.csv`
6. append narrative reasoning to `report.md`
7. never write tuning outputs into the skill repository

This way the prompt focuses on decision-making while the file layout remains consistent.

---

## Recommendation

Adopt the following division of labor:

- **Code**: enforce session layout, file naming, run numbering, and result bookkeeping
- **Prompt**: guide project understanding, metric interpretation, and tuning decisions

This hybrid design is the most robust path:

- less drift than pure prompt enforcement
- less brittleness than encoding tuning strategy as rigid rules
- easy to inspect, debug, and extend

---

## Open Questions

These do not block the first implementation:

1. Should `results.csv` be updated in place, or should there also be per-run JSON records?
2. Should run directories include a snapshot of the fully resolved Hydra config by default?
3. Should session creation allow a user-provided label in addition to timestamp?
4. Should multiple metrics be represented as extra CSV columns or nested JSON?
5. Should the canonical AHT directory always be local when orchestration is local, even if the project is remote?

My current recommendation is to keep the first version simple and postpone these until the basic workflow is actually used.
