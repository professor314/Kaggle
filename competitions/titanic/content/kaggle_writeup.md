# Titanic — Solution Writeup

## Summary

Achieved ~0.82 accuracy using a research-first approach that prioritizes domain knowledge over model complexity. The pipeline combines feature engineering informed by historical research (title extraction, family size analysis, domain-informed age imputation) with an optimized Random Forest classifier. Improvement of +0.05 over the gender-only baseline (0.77).

## Approach

A structured pipeline guided by domain research conducted before any modeling:

1. **Domain research** — Studied Titanic evacuation protocols, ship layout, class-based access to lifeboats, and prior Kaggle solutions
2. **Exploratory data analysis** — Confirmed research hypotheses, identified missing data patterns (Age 20%, Cabin 77%)
3. **Feature engineering** — Created research-backed features: Title, FamilySize, IsAlone, HasCabin, AgeBin
4. **Missing data handling** — Title-group median imputation for Age (preserves child/adult distinction)
5. **Model selection** — Compared LR, RF, GBM, KNN, SVM via stratified 5-fold CV
6. **Hyperparameter optimization** — Randomized search (50 iterations) on Random Forest
7. **Final prediction** — Optimized RF on full training set

## What Worked

- **Title extraction from Name** — Single most impactful feature. Encodes gender, age group, and social status. "Master" reliably identifies male children when Age is missing.
- **FamilySize = SibSp + Parch + 1** — Captures non-linear survival pattern: solo travelers (low survival) vs. small families (high survival) vs. large families (low survival).
- **Age imputation by title group** — Using title-specific medians (Master ~5, Miss ~22, Mr ~30) instead of global median preserves the critical child/adult distinction.
- **Random Forest** — Best balance of performance and robustness on a small (891 rows) mixed-type dataset.
- **HasCabin binary flag** — Converts 77% missingness into a useful signal (cabin recorded → likely upper class).
- **Research-first methodology** — Every feature had a documented rationale before implementation.

## What Didn't Work

- **Deck extraction from Cabin** — 77% missing makes deck letters unreliable. The binary HasCabin flag was sufficient.
- **Complex feature interactions** — Age × Pclass, FarePerPerson added negligible value. Simple features already captured the signal.
- **SVM** — Underperformed tree-based methods despite feature scaling. Not well-suited to this mixed-type small dataset.
- **KNN** — Too sensitive to feature scaling and distance metric choice for marginal benefit.
- **Aggressive feature engineering** — More features ≠ better predictions. The research-backed minimal set outperformed the kitchen sink.

## Final Model Description

**Optimized Random Forest Classifier**

Key parameters (from randomized search, 50 iterations, 5-fold stratified CV):
- `n_estimators`: 200-300
- `max_depth`: 8-10 (moderate constraint prevents overfitting)
- `min_samples_leaf`: 2-4 (forces generalization)
- `max_features`: 'sqrt' (feature subsampling adds diversity)
- `bootstrap`: True

Feature set (10 features):
- Original: Pclass, Sex, Age (imputed), Fare, Embarked
- Engineered: Title, FamilySize, IsAlone, HasCabin, AgeBin

## Score

| Metric | Score |
|--------|-------|
| Final CV Accuracy | ~0.82 |
| Baseline (gender-only) | 0.77 |
| Improvement | +0.05 |

The CV estimate uses stratified 5-fold cross-validation to maintain the ~38% survival rate in each fold. The gender-only baseline (all females survive, all males die) represents the simplest meaningful prediction.

## Key Takeaways

1. **Research before modeling** — Understanding "women and children first" + class-based deck access explains most variance
2. **Title is king** — A single regex extraction that captures gender + age + status
3. **Imputation strategy matters** — Title-based vs. global median is the difference between preserving or destroying the child signal
4. **Regularize on small data** — Constraining RF depth and leaf size matters more than adding estimators
5. **CV over leaderboard** — Stratified 5-fold gives stable estimates; public LB on 50% of test data is noisy

## AI Disclosure

This solution was developed with **Kiro**, an AI-powered coding agent that assisted with research synthesis, pipeline implementation, model comparison, and content generation.

Kaggle explicitly allows AI-assisted development in competitions. The human-AI collaboration focused on:
- **Human**: Strategic decisions, hypothesis selection, result interpretation
- **AI (Kiro)**: Implementation, systematic comparison, documentation generation

Full transparency: AI tools were used throughout the workflow, from domain research through submission generation.
