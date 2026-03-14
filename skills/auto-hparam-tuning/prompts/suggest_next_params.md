# Next Hyperparameter Suggestion Engine

You are a coding agent tasked with proposing the next set of hyperparameters to try based on the diagnosis of previous training runs. You implement a two-layer decision system: a deterministic rule layer for stable, evidence-based adjustments, and an LLM reasoning layer for contextual interpretation and prioritization.

## Input

You should have:
1. The diagnosis report from `prompts/diagnose_curve.md` (diagnostic labels and evidence)
2. The `hparam_space.yaml` (available hyperparameters and their search spaces)
3. The `objective.yaml` (optimization objective and constraints)
4. Run history from `scripts/run_history.py` (all previous runs and their results)
5. The project's `HPARAM.md` (project context)

## Layer 1: Rule-Based Decisions

These rules encode well-established ML training heuristics. Apply them deterministically based on the diagnosis label.

### Rule Table

| Diagnosis | Parameter | Adjustment | Magnitude | Rationale |
|---|---|---|---|---|
| `divergence` | `*.lr` | decrease | ÷5 to ÷10 | Reduce gradient step size |
| `divergence` | `*.gradient_clip_val` | add/decrease | set to 1.0 or ÷2 | Prevent gradient explosion |
| `instability` | `*.lr` | decrease | ÷2 to ÷5 | Reduce gradient noise |
| `instability` | `*.batch_size` | increase | ×2 | Reduce gradient variance |
| `instability` | `*.gradient_clip_val` | add/decrease | set to 1.0 | Stabilize updates |
| `plateau` | `*.lr` | try schedule | cosine/step decay | Escape local minimum |
| `plateau` | `*.lr` | local sweep | ×0.3 to ×3 | Search nearby |
| `plateau` | `*.optimizer` | switch | try AdamW/SGD+momentum | Different optimization landscape |
| `overfitting` | `*.dropout` | increase | +0.1 to +0.2 | More regularization |
| `overfitting` | `*.weight_decay` | increase | ×3 to ×10 | L2 regularization |
| `overfitting` | `*.max_epochs` | decrease | reduce by 30-50% | Early stopping |
| `underfitting` | `*.hidden_size` / `*.num_layers` | increase | ×1.5 to ×2 | More model capacity |
| `underfitting` | `*.lr` | increase | ×2 to ×5 | Faster learning |
| `underfitting` | `*.dropout` | decrease | -0.1 to set 0.0 | Less regularization |
| `underfitting` | `*.weight_decay` | decrease | ÷3 to ÷10 | Less regularization |
| `early_saturation` | `*.lr` | add warmup | warmup_steps = 5-10% of total | Gradual start |
| `early_saturation` | `*.lr_scheduler` | set | cosine or linear | Decay after warmup |
| `slow_training` | `*.lr` | increase | ×2 to ×5 | Faster convergence |
| `slow_training` | `*.batch_size` | increase | ×2 | More efficient updates |
| `healthy` | all high-priority | local sweep | ±20-50% of current | Fine-tune around good config |

### Rule Application Logic

```
1. Read the diagnosis label
2. Look up applicable rules from the table above
3. Filter rules to only include parameters that exist in hparam_space.yaml
4. For each applicable rule:
   a. Compute the new parameter value
   b. Clamp to the search space bounds [low, high]
   c. Ensure the value respects the parameter type (int/float/bool/categorical)
5. Output the list of proposed parameter changes
```

### Multi-Diagnosis Handling

When multiple diagnosis labels are present (e.g., `instability` + `slow_training`), resolve conflicts:

1. **Safety first**: If `divergence` is present, prioritize stabilization (reduce lr, add clipping) before anything else.
2. **No contradictions**: Do not simultaneously increase and decrease the same parameter.
   - `instability` says decrease lr, `slow_training` says increase lr → choose to decrease lr (safety wins).
   - `overfitting` says increase dropout, `underfitting` says decrease dropout → diagnose more carefully or try the safer option.
3. **Prioritize the more severe diagnosis**: `divergence` > `instability` > `overfitting`/`underfitting` > `plateau` > `slow_training` > `early_saturation` > `healthy`.

## Layer 2: LLM Contextual Reasoning

After the rule layer proposes parameter changes, apply your reasoning to:

### 2a. Prioritize Changes

Not all changes should be tried simultaneously. Rank them by:
1. **Impact**: Which change is most likely to fix the primary issue?
2. **Safety**: Which change is least likely to cause new problems?
3. **Cost**: Which change is cheapest to try (in terms of training time)?

Recommend trying 1-3 changes per iteration, not all at once (to isolate effects).

### 2b. Avoid Repeating Failures

Check the run history. If a parameter value was already tried and produced poor results, do not suggest it again. Instead:
- Try a different value in the same direction
- Try adjusting a different parameter
- Consider that the issue may not be solvable by that parameter alone

### 2c. Explain Reasoning

For each proposed change, provide:
- What problem it addresses
- Why this specific adjustment was chosen
- What the expected outcome is
- What to watch for (potential side effects)

### 2d. Consider Interactions

Some parameter changes interact:
- **Batch size ↑ → LR should ↑** (linear scaling rule: when batch_size doubles, lr should roughly double)
- **Dropout ↑ → may need more epochs** (more regularization = slower convergence)
- **Model size ↑ → may need to decrease batch_size** (GPU memory constraint)

## Output Format

Generate a trial proposal file:

```yaml
# AHT Trial Proposals
# Based on: <run_name> diagnosis
# Iteration: <iteration_number>

diagnosis_summary: "Overfitting detected: train_acc=95% but val_acc=82% (generalization gap=13%)"

proposals:
  # Trial 1: Primary recommendation
  - name: "increase_regularization"
    priority: 1
    overrides:
      optimizer.weight_decay: 0.001
      model.dropout: 0.3
    rationale: |
      The 13% generalization gap strongly indicates overfitting.
      Increasing weight_decay from 1e-4 to 1e-3 (10x) and dropout from 0.1 to 0.3
      should significantly improve generalization.
    expected_outcome: "Reduced generalization gap, val_acc improvement of 2-5%"
    risks: "May slow convergence; if val_acc drops, reduce dropout to 0.2"

  # Trial 2: Alternative approach
  - name: "reduce_training_length"
    priority: 2
    overrides:
      trainer.max_epochs: 30
    rationale: |
      Val accuracy peaked at epoch 20 (step 5000). Training for 50 epochs
      is wasting compute. Reducing to 30 epochs also reduces overfitting.
    expected_outcome: "Similar val_acc with 40% less training time"
    risks: "None significant — baseline already showed peak before epoch 30"

  # Trial 3: More aggressive if trials 1-2 don't work
  - name: "smaller_model_more_reg"
    priority: 3
    overrides:
      model.num_layers: 3
      model.dropout: 0.2
      optimizer.weight_decay: 0.0005
    rationale: |
      If regularization alone doesn't fix overfitting, reducing model capacity
      (from 4 to 3 layers) while adding moderate regularization may help.
    expected_outcome: "Lower capacity reduces overfitting risk at potential accuracy cost"
    risks: "May underfit if model is already minimal. Monitor train_acc closely."

# Parameters NOT changed (with explanation)
unchanged:
  - param: optimizer.lr
    current: 0.001
    reason: "LR appears appropriate — training converges well, issue is generalization not optimization"
  - param: trainer.batch_size
    current: 32
    reason: "Batch size is not related to the overfitting issue"

# Iteration strategy
iteration_plan: |
  1. Run trial 1 (increase_regularization) first
  2. If val_acc improves by >2%, fine-tune around the new values
  3. If no improvement, try trial 2 (reduce_training_length)
  4. If still no improvement, try trial 3 (smaller model)
  5. If none work, consider data augmentation or different model architecture
```

## Important Guidelines

1. **Change few parameters per trial**: Ideally 1-2 parameters per trial to isolate effects. Only change 3+ when they are strongly coupled (e.g., batch_size + lr).

2. **Always clamp to search space**: Never suggest values outside the bounds defined in `hparam_space.yaml`.

3. **Respect constraints**: Check `objective.yaml` constraints (GPU memory, time) before suggesting changes that increase resource usage.

4. **Be specific about magnitudes**: Don't say "increase learning rate" — say "increase lr from 0.001 to 0.003 (3x)".

5. **Consider the iteration budget**: If only 5 trials remain, focus on the highest-impact changes. Don't waste trials on minor adjustments.

6. **Know when to stop**: If the last 3 iterations showed <1% improvement, suggest stopping the tuning loop or trying a fundamentally different approach.
