---
phase: 5
title: Hyperparameter Optimization
triggers: kaggle_ml_toolkit.model_optimizer
---

# Hyperparameter Optimization

## Context
Optimization squeezes the most performance from our top model candidates. We tune hyperparameters systematically using grid or randomized search, balancing thoroughness against computational budget. This phase takes the 2-3 models identified in selection and finds their best configurations.

## Decision Points

### Models to Optimize
**Question:** Which models from the selection phase should we tune?
**Options:**
- Top 1 only: Optimize the single best performer. Fastest, but risks missing a model that improves more with tuning.
- Top 2-3: Optimize the top performers in parallel. Better coverage, moderate cost.
- All viable: Optimize everything that passed selection. Thorough but expensive.
**Recommendation:** Top 2-3 — tuning can dramatically change rankings, so optimizing only the leader is risky.

### Parameter Grid Scope
**Question:** How broad should the hyperparameter search space be?
**Options:**
- Narrow: Search around known-good defaults with small perturbations. Fast convergence.
- Standard: Cover common ranges from documentation and competition literature. Good balance.
- Wide: Explore unusual parameter combinations that might surface non-obvious optima. Slower.
- Adaptive: Start standard, then narrow around the best region for a second pass.
**Recommendation:** Standard for first pass. Adaptive if computational budget allows a second round.

### Search Method
**Question:** Which search strategy should we use?
**Options:**
- Grid search: Exhaustive evaluation of all combinations. Guarantees finding the best in the grid but scales poorly with dimensions.
- Randomized search: Sample N combinations randomly. Finds good solutions faster in high-dimensional spaces.
- Bayesian (future): Use prior evaluations to guide search. Best efficiency but more complex to set up.
**Recommendation:** Randomized search with 50-100 iterations — covers more of the space than grid with fewer evaluations.

### Iteration Budget
**Question:** How many parameter combinations should we evaluate?
**Options:**
- Quick (10-20 iterations): Get a reasonable improvement fast. Good for initial exploration.
- Standard (50-100 iterations): Thorough coverage of the parameter space. Recommended default.
- Deep (200+ iterations): Maximum coverage. For final push or when marginal gains matter.
**Recommendation:** Standard (50-100) — diminishing returns beyond this for most tabular problems.

### Early Stopping
**Question:** Should we use early stopping during optimization?
**Options:**
- No early stopping: Evaluate all iterations regardless. Simple, predictable runtime.
- Performance plateau: Stop if no improvement after N consecutive iterations.
- Time-based: Set a wall-clock time limit and stop when reached.
**Recommendation:** Performance plateau (stop after 20 iterations without improvement) — saves time when the optimum is found early.

## Actions
Based on decisions above, Kiro will:
1. Define parameter grids for each selected model based on chosen scope
2. Execute `ModelOptimizer.optimize()` with chosen method and iteration budget
3. Log all results to `Evaluator` for experiment tracking
4. Present best parameters, scores, and improvement over untuned baseline
5. Compare optimized models head-to-head
6. Ask user: "Are these results satisfactory, or should we do a second pass with refined grids?"
7. Record optimization decisions and best configurations in the design doc
