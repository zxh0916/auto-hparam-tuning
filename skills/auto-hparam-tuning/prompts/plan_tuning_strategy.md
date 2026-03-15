Generate a tuning strategy document named `strategy.md` in the current AHT session directory (`aht/<date>/<time>/strategy.md`).

Your goal is to formulate a structured exploration plan before diving into iterative hyperparameter tuning. Rather than making greedy, isolated decisions in every single step, you must analyze the baseline result, list your hypotheses, and lay out a phased roadmap for the agent to follow.

Inputs you should assume are available:
- `HPARAM.md` (config structure and parameters)
- The current session's `run_understanding.md` (tuning knobs and metric goals)
- Baseline run metrics (or initial run results) from `report.md` / `summary.json`

Instructions:

1. Analyze the Baseline
State the baseline metric and analyze the training behavior (Did it converge? Overfit? Oscillate?). Identify the primary bottleneck preventing better performance based on the available data.

2. Formulate 2-3 Hypotheses
List concrete hypotheses for improving the primary metric. 
Examples:
- Hypothesis A (Capacity): The baseline model is underfitting due to low parameter count. Increasing network depth/width will improve validation accuracy.
- Hypothesis B (Optimization): The model learns too slowly within the epoch limit. Increasing initial learning rate from 1e-3 to 3e-3 or using a different optimizer will speed up convergence.
- Hypothesis C (Regularization): The train loss is near 0 but validation loss is increasing. Adding dropout or weight decay will bridge the generalization gap.

3. Determine Strategy Topology (Adaptive Complexity)
Assess the project's scale based on the Micro-Baseline and project structure:
- **Simple/Fast Project (e.g., MNIST, NLP tasks on subset)**: If full training runs take only a few minutes or do not strain resources, complex truncation and resuming is overkill. Use a **Direct Strategy** where you evaluate full-length hypotheses directly, ideally in parallel.
- **Complex/Slow Project (e.g., large vision models, full-scale language)**: If training is expensive, use a **Progressive Strategy** to save compute by early stopping.

4. Define the Tuning Roadmap
Design a structured sequence of runs.
**For a Progressive Strategy**:
- **Phase 1: Broad Exploration (Micro-Runs)**. Run tests for only a fraction of normal time (e.g., 10-20%) to quickly discard poorly-performing architecture changes or extreme learning rates.
- **Phase 2: Refinement (Medium-Runs)**. Increase the runtime limit (e.g., 50%) to test top configurations from Phase 1. **Requires resuming from Phase 1 checkpoints.**
- **Phase 3: Exploitation (Full-Runs)**. Run the single best configuration for 100% full duration. **Requires resuming from Phase 2 checkpoints.**

**For a Direct Strategy**:
- **Phase 1: Test Hypothesis A & B**. Run configurations for 100% standard duration. Utilize parallel concurrent execution if compute permits.
- **Phase 2: Fine-Tuning**. Take the top config from Phase 1 and grid-search secondary parameters (e.g., batch size, dropout).

Output format:
Write a crisp, actionable markdown document with the following sections (Do not surround the output in markdown code blocks, just output the raw markdown text):

# Hyperparameter Tuning Strategy

## 1. Baseline Analysis
[Your analysis...]

## 2. Hypotheses
- [Hypothesis 1]
- [Hypothesis 2]

## 3. Tuning Roadmap
*(Adapt the phase structure depending on whether you chose Direct or Progressive)*

**(Example for Progressive)**:
- **Phase 1: Broad Exploration (Micro-Runs)**
  - Run intent: [What to override and why]
  - Target Runtime Override: [e.g., `trainer.max_epochs=2`]
  - Expected Outcome: [What validates this hypothesis]
- **Phase 2: Refinement (Medium-Runs)**
  - Run intent: [What to override and why]
  - Target Runtime Override: [e.g., `trainer.max_epochs=5`]
  - Checkpoint to Resume: [e.g., `ckpt_path="runs/X/workspace/..."` (Leave as TBD if drafting prior to Phase 1)]
  - Expected Outcome: ...

**(Example for Direct)**:
- **Phase 1: Test Major Hypotheses (Parallel)**
  - Run intent (Run A): ...
  - Run intent (Run B): ...
- **Phase 2: Fine-Tuning**
  - Run intent...