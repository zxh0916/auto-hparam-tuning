# Training Curve Diagnosis

You are a coding agent tasked with interpreting training curve statistics and diagnostic labels to provide actionable insights. This prompt guides you to combine the quantitative output from `scripts/analyze_event.py` and `scripts/detect_patterns.py` with project-specific context to produce a clear diagnosis.

## Input

You should have:
1. The JSON output from `scripts/analyze_event.py` (curve statistics)
2. The JSON output from `scripts/detect_patterns.py` (diagnostic labels)
3. The project's `HPARAM.md` (project context)
4. The `objective.yaml` (optimization objective)
5. Optionally, statistics for both training and validation metrics (to detect overfitting)

## Diagnostic Patterns Reference

### 1. Divergence
**Symptoms**: Loss explodes (goes to infinity or NaN), metric worsens continuously.
**Evidence from statistics**:
- `worsened: true` with large negative `improvement`
- `final_value` is much worse than `initial_value`
- `delta` has wrong sign (positive for loss, negative for accuracy)
- Very high `oscillation` score

**Common causes**:
- Learning rate too high
- Gradient explosion (missing gradient clipping)
- Numerical instability (wrong precision)
- Data preprocessing bug (NaN in inputs)

**Typical fix**: Reduce learning rate by 5-10x, add gradient clipping, check for NaN in data.

### 2. Instability / Oscillation
**Symptoms**: Metric fluctuates wildly without settling.
**Evidence from statistics**:
- `normalized_oscillation > 0.5`
- `sign_change_count` is high relative to total steps
- Large gap between `quantile_low_value` and `quantile_high_value`

**Common causes**:
- Learning rate too high (but not high enough to diverge)
- Batch size too small (high gradient variance)
- Missing or insufficient gradient clipping

**Typical fix**: Reduce learning rate by 2-5x, increase batch size, add gradient clipping.

### 3. Plateau
**Symptoms**: Metric stops improving well before training ends.
**Evidence from statistics**:
- `best_step` is significantly earlier than `end_step` (e.g., `best_step < 0.5 * end_step`)
- Small `improvement` despite many remaining steps
- Low `oscillation` in the latter portion of training

**Common causes**:
- Learning rate too low or not decaying
- Model capacity reached
- Insufficient data diversity
- Optimizer stuck in local minimum

**Typical fix**: Try learning rate warmup + cosine decay, increase model capacity, try different optimizer.

### 4. Overfitting
**Symptoms**: Training metric keeps improving but validation metric worsens or plateaus.
**Evidence** (requires both train and val curves):
- Train metric improving: `worsened: false` for train curve
- Val metric worsening or plateaued: `worsened: true` or `improvement ≈ 0` for val curve
- Growing gap between train and val metrics (generalization gap)
- Val `best_step` is much earlier than `end_step`

**Common causes**:
- Insufficient regularization
- Too much model capacity for dataset size
- Training too long without early stopping
- Insufficient data augmentation

**Typical fix**: Increase dropout/weight_decay, reduce model size, add early stopping, increase data augmentation.

### 5. Underfitting
**Symptoms**: Both training and validation metrics are poor.
**Evidence**:
- `best_value` is far from expected performance
- Both train and val metrics show limited improvement
- Low `oscillation` (model is stable but bad)
- No overfitting gap (train and val are similarly poor)

**Common causes**:
- Model too small
- Learning rate too low
- Too much regularization (dropout/weight_decay too high)
- Insufficient training time

**Typical fix**: Increase model capacity, increase learning rate, reduce regularization, train longer.

### 6. Early Saturation
**Symptoms**: Metric quickly reaches a reasonable value but stops improving very early.
**Evidence**:
- `best_step` is very early (e.g., `best_step < 0.2 * end_step`)
- `improvement > 0` but `best_value` is close to `initial_value + small delta`
- Low `oscillation` after the early phase

**Common causes**:
- Learning rate too high initially (quick convergence, then stuck)
- No learning rate schedule
- Model or task is too easy
- Improper weight initialization

**Typical fix**: Use learning rate warmup, cosine/step decay schedule, larger model.

### 7. Slow Training
**Symptoms**: Metric is improving but very slowly relative to compute spent.
**Evidence**:
- `improvement > 0` but small relative to total steps
- `delta / count` (improvement per step) is very low
- Training is still improving at `end_step` (not plateaued, just slow)

**Common causes**:
- Learning rate too low
- Batch size too small (slow convergence per step)
- Inefficient model architecture

**Typical fix**: Increase learning rate, increase batch size, consider learning rate warmup.

## Diagnosis Report Format

After analyzing the statistics and identifying patterns, produce a diagnosis report in this format:

```yaml
diagnosis:
  run_name: <run_name>
  primary_metric: val/accuracy
  mode: max

  # Overall assessment
  status: <divergence|instability|plateau|overfitting|underfitting|early_saturation|slow_training|healthy>
  confidence: <high|medium|low>  # how confident you are in the diagnosis
  severity: <critical|warning|info>  # how urgently this needs fixing

  # Quantitative evidence
  evidence:
    final_value: 0.82
    best_value: 0.85
    best_step: 5000
    end_step: 20000
    improvement: 0.05
    oscillation: 0.15
    normalized_oscillation: 0.42
    generalization_gap: 0.13  # train_best - val_best (if both available)

  # Human-readable explanation
  explanation: |
    The validation accuracy peaked at 85% at step 5000 but has not improved
    in the remaining 15000 steps. The training accuracy continued to 98%,
    creating a 13% generalization gap. This indicates overfitting.

  # Recommended actions (ordered by priority)
  recommendations:
    - action: "Increase weight_decay from 1e-4 to 1e-3"
      priority: high
      expected_effect: "Reduce overfitting by penalizing large weights"
    - action: "Add dropout=0.3 to model layers"
      priority: high
      expected_effect: "Improve generalization"
    - action: "Enable early stopping with patience=10"
      priority: medium
      expected_effect: "Save compute by stopping when val metric plateaus"
```

## Important Guidelines

1. **Always compare train and val metrics** when both are available. The relationship between them is the most informative signal.

2. **Consider the training stage**: A metric that looks bad at step 100 might be fine — training just started. Always consider the proportion of training completed.

3. **Be conservative in diagnosis**: If you're uncertain, say so. A wrong diagnosis leads to wrong parameter adjustments.

4. **Provide actionable recommendations**: Every diagnosis should come with specific, concrete parameter changes — not vague advice like "try tuning the learning rate".

5. **Account for noise**: Validation metrics are often noisier than training metrics (smaller eval set). Don't diagnose instability from normal validation noise.
