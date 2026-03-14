# Optimization Objective Definition

You are a coding agent tasked with defining the optimization objective for an automated hyperparameter tuning session. The objective specifies what metric to optimize, in which direction, and under what constraints.

## Input

You should already have:
1. The project's `HPARAM.md` (understanding of the project)
2. Access to TensorBoard event files from previous runs (if any)
3. The user's stated goal (if any)

## Procedure

### Step 1: Discover Available Metrics

If TensorBoard event files exist from previous runs:

1. Run `scripts/analyze_event.py` with any available event file to list scalar tags:
   ```bash
   python scripts/analyze_event.py <event_path> --list-tags
   ```
   If `--list-tags` is not available, load the event file and inspect DataFrame columns.

2. Categorize the discovered metrics:

   | Category | Common Patterns | Typical Direction |
   |---|---|---|
   | **Training Loss** | `train/loss`, `train_loss`, `loss/train` | minimize |
   | **Validation Loss** | `val/loss`, `eval/loss`, `val_loss` | minimize |
   | **Training Accuracy** | `train/acc`, `train/accuracy`, `train_acc` | maximize |
   | **Validation Accuracy** | `val/acc`, `val/accuracy`, `eval/accuracy` | maximize |
   | **Learning Rate** | `lr`, `learning_rate`, `optimizer/lr` | informational (not an objective) |
   | **Gradient Norm** | `grad_norm`, `gradient_norm` | informational |
   | **Epoch/Step** | `epoch`, `global_step` | informational |
   | **Custom Metrics** | `val/f1`, `val/mAP`, `val/bleu`, `val/perplexity` | depends on metric |
   | **Resource Metrics** | `gpu_memory`, `throughput`, `samples_per_sec` | informational or constraint |

### Step 2: Determine Primary Objective

The primary objective is the single most important metric to optimize. Follow this decision logic:

1. **If the user explicitly stated a metric**: Use that. Example: "I want to maximize val/accuracy" → primary = `val/accuracy`, mode = `max`.

2. **If the user said "tune for best performance" without specifying**: Look for validation metrics in this priority order:
   - `val/accuracy` or `val/acc` → mode = `max`
   - `val/loss` or `eval/loss` → mode = `min`
   - `val/f1` or `val/f1_score` → mode = `max`
   - `val/mAP` → mode = `max`
   - `val/perplexity` or `val/ppl` → mode = `min`
   - `val/bleu` → mode = `max`
   - If no validation metrics exist, fall back to training metrics (but warn the user about overfitting risk).

3. **If ambiguous**: Ask the user to choose from the available metrics.

### Step 3: Choose Evaluation Strategy

How should the metric be evaluated across training steps?

| Strategy | Description | When to Use |
|---|---|---|
| `best` | Take the best value achieved at any step | Default. Works for most metrics. |
| `final` | Take the value at the last step | When training is expected to converge fully. |
| `rolling_average` | Average of the last N steps | When metrics are noisy (e.g., RL environments). |

Default to `best` unless the user specifies otherwise or the metric is known to be noisy.

### Step 4: Identify Secondary Metrics and Constraints

Secondary metrics are tracked but not directly optimized. Constraints are hard limits.

**Common secondary metrics**:
- Training time (wall clock)
- GPU memory usage
- Throughput (samples/sec)
- Generalization gap (train_metric - val_metric)

**Common constraints**:
- `max_gpu_memory_gb`: Maximum GPU memory the model may use
- `max_training_time_hours`: Maximum wall clock time for one trial
- `max_epochs`: Maximum number of training epochs
- `max_trials`: Maximum number of tuning trials (budget)
- `min_metric_threshold`: Minimum acceptable metric value (for early rejection)

### Step 5: Output Format

Generate an `objective.yaml` file:

```yaml
# AHT Optimization Objective
# Project: <project_name>
# Generated: <timestamp>

objective:
  # Primary metric to optimize
  primary: val/accuracy
  mode: max          # "max" or "min"
  eval_strategy: best  # "best", "final", or "rolling_average"

  # Secondary metrics to track (not optimized, but reported)
  secondary:
    - metric: val/loss
      mode: min
    - metric: train/accuracy
      mode: max
    - metric: training_time
      mode: min

  # Hard constraints — trials violating these are marked as failed
  constraints:
    max_gpu_memory_gb: 24
    max_training_time_hours: 2
    max_epochs: 100
    max_trials: 30

  # Early rejection — stop a trial early if clearly bad
  early_rejection:
    enabled: true
    metric: val/loss
    mode: min
    patience_epochs: 10       # stop if no improvement for N epochs
    min_delta: 0.001          # minimum improvement to count as progress

  # Notes for the agent
  notes: |
    This is a classification task on CIFAR-10.
    Validation accuracy is the primary metric.
    We have 1 GPU (RTX 3090, 24GB) and 4 hours of compute budget.
```

## Important Guidelines

1. **Always prefer validation metrics over training metrics** for the primary objective. Training metrics can overfit and are not reliable indicators of model quality.

2. **Always ask the user if unsure** about which metric to optimize. A wrong objective wastes all tuning compute.

3. **Be explicit about direction**: Some metrics are ambiguous (e.g., "perplexity" should be minimized, "BLEU" should be maximized). State the direction clearly.

4. **Set reasonable constraints**: If the user doesn't specify constraints, infer from context:
   - Check GPU memory from `nvidia-smi` if accessible
   - Estimate training time from baseline run duration
   - Default to 20-50 trials if no budget is specified

5. **Consider early rejection**: For expensive training runs, early rejection saves significant compute by stopping clearly unpromising trials.
