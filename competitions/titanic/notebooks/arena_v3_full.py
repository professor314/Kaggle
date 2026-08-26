"""
Titanic Arena V3 — Full ArenaGenerator → Hyperparameter Tune → Submit Pipeline
"""

import sys
import os
import warnings
import subprocess
import time

warnings.filterwarnings("ignore")
sys.path.insert(0, r"c:\Users\profe\OneDrive\Desktop\Sean Obsidian\Kaggle")

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    GradientBoostingClassifier, RandomForestClassifier,
    HistGradientBoostingClassifier,
)
from sklearn.linear_model import LogisticRegression
from scipy.stats import uniform, randint

from kaggle_ml_toolkit import ArenaGenerator, ModelArena, ModelOptimizer

# ============================================================
# 1. LOAD AND ENGINEER FEATURES
# ============================================================
BASE = r"c:\Users\profe\OneDrive\Desktop\Sean Obsidian\Kaggle\competitions\titanic"
train = pd.read_csv(os.path.join(BASE, "data", "train.csv"))
test = pd.read_csv(os.path.join(BASE, "data", "test.csv"))

def engineer_features(df):
    """Feature engineering for Titanic."""
    df = df.copy()
    
    # Title extraction
    df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
    title_map = {
        'Mr': 0, 'Miss': 1, 'Mrs': 2, 'Master': 3,
        'Dr': 4, 'Rev': 4, 'Col': 4, 'Major': 4, 'Mlle': 1,
        'Countess': 4, 'Ms': 1, 'Lady': 4, 'Jonkheer': 4,
        'Don': 4, 'Dona': 4, 'Mme': 2, 'Capt': 4, 'Sir': 4
    }
    df['TitleCode'] = df['Title'].map(title_map).fillna(4).astype(int)
    
    # Age imputation using title medians
    title_age_medians = df.groupby('TitleCode')['Age'].median()
    for tc in df['TitleCode'].unique():
        mask = (df['TitleCode'] == tc) & df['Age'].isnull()
        df.loc[mask, 'Age'] = title_age_medians.get(tc, df['Age'].median())
    df['Age'].fillna(df['Age'].median(), inplace=True)
    
    # Fare
    df['Fare'].fillna(df['Fare'].median(), inplace=True)
    
    # Core features
    df['IsFemale'] = (df['Sex'] == 'female').astype(int)
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
    df['IsChild'] = (df['Age'] < 16).astype(int)
    
    # Extended features
    df['FarePerPerson'] = df['Fare'] / df['FamilySize'].clip(lower=1)
    df['SexPclass'] = df['IsFemale'] * 10 + df['Pclass']
    df['WomenChildren1st'] = ((df['IsFemale'] | df['IsChild']) & (df['Pclass'] <= 2)).astype(int)
    df['FamilyBucket'] = pd.cut(df['FamilySize'], bins=[0, 1, 2, 4, 11], labels=[0, 1, 2, 3]).astype(int)
    df['AgePclass'] = df['Age'] * df['Pclass']
    df['FareLog'] = np.log1p(df['Fare'])
    
    return df

train_eng = engineer_features(train)
test_eng = engineer_features(test)

y = train_eng['Survived']
test_ids = test_eng['PassengerId']

# Feature sets
minimal_cols = ['IsFemale', 'Pclass', 'Age', 'Fare', 'FamilySize', 'IsAlone', 'IsChild', 'TitleCode']
extended_cols = minimal_cols + ['FarePerPerson', 'SexPclass', 'WomenChildren1st', 'FamilyBucket', 'AgePclass', 'FareLog']

X = train_eng[extended_cols].copy()
X_test = test_eng[extended_cols].copy()

print(f"Training data: {X.shape}")
print(f"Test data: {X_test.shape}")
print(f"Feature sets: minimal={len(minimal_cols)}, extended={len(extended_cols)}")

# ============================================================
# 2. ARENA GENERATOR
# ============================================================
print("\n" + "="*60)
print("PHASE 1: ARENA GENERATOR")
print("="*60)

gen = ArenaGenerator(
    X=X,
    y=y,
    problem_type="classification",
    feature_sets={"minimal": minimal_cols, "extended": extended_cols},
    time_budget_minutes=10,
    prior_cv_lb_gap=0.06,
)

arena = gen.build_arena(verbose=True)

# Also add HGB models (not in default ArenaGenerator)
from sklearn.ensemble import HistGradientBoostingClassifier
arena.add("HGB_minimal_conservative", HistGradientBoostingClassifier(
    max_iter=100, max_depth=3, learning_rate=0.1, min_samples_leaf=15, random_state=42
), feature_cols=minimal_cols)
arena.add("HGB_minimal_strong", HistGradientBoostingClassifier(
    max_iter=200, max_depth=4, learning_rate=0.05, min_samples_leaf=10, random_state=42
), feature_cols=minimal_cols)
arena.add("HGB_extended_strong", HistGradientBoostingClassifier(
    max_iter=200, max_depth=4, learning_rate=0.05, min_samples_leaf=10, random_state=42
), feature_cols=extended_cols)

results = arena.run(verbose=True)

print("\n--- ARENA LEADERBOARD ---")
lb = arena.leaderboard()
print(lb.to_string(index=False))

# ============================================================
# 3. HYPERPARAMETER TUNING (Top 3)
# ============================================================
print("\n" + "="*60)
print("PHASE 2: HYPERPARAMETER TUNING (Top 3)")
print("="*60)

optimizer = ModelOptimizer()

# Get top 3 model types from results
top3 = results[:3]
tuned_results = []

for i, result in enumerate(top3):
    print(f"\n--- Tuning #{i+1}: {result.name} (CV: {result.mean_score:.4f}) ---")
    
    model_type = result.model_type
    feature_cols = result.feature_cols
    X_feat = X[feature_cols]
    
    # Define param grids based on model type
    if model_type == "GradientBoostingClassifier":
        base_model = GradientBoostingClassifier(random_state=42)
        param_grid = {
            "n_estimators": [30, 50, 80, 100, 150],
            "max_depth": [2, 3, 4],
            "learning_rate": [0.05, 0.08, 0.1, 0.15, 0.2],
            "min_samples_leaf": [8, 10, 15, 20, 25],
            "subsample": [0.7, 0.8, 0.9, 1.0],
        }
    elif model_type == "HistGradientBoostingClassifier":
        base_model = HistGradientBoostingClassifier(random_state=42)
        param_grid = {
            "max_iter": [50, 100, 150, 200, 300],
            "max_depth": [2, 3, 4, 5],
            "learning_rate": [0.03, 0.05, 0.08, 0.1, 0.15],
            "min_samples_leaf": [5, 10, 15, 20, 25],
            "l2_regularization": [0.0, 0.1, 0.5, 1.0],
        }
    elif model_type == "RandomForestClassifier":
        base_model = RandomForestClassifier(random_state=42)
        param_grid = {
            "n_estimators": [100, 150, 200, 300],
            "max_depth": [3, 4, 5, 6, 8],
            "min_samples_leaf": [5, 8, 10, 15, 20],
            "max_features": ["sqrt", "log2", 0.5, 0.7],
        }
    elif model_type == "LogisticRegression":
        base_model = LogisticRegression(max_iter=2000, random_state=42)
        param_grid = {
            "C": [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0],
            "solver": ["lbfgs", "liblinear"],
        }
    else:
        # Generic fallback for other model types
        print(f"  Skipping tuning for {model_type} (no param grid defined)")
        tuned_results.append({
            "name": result.name, "cv_score": result.mean_score,
            "model": None, "features": feature_cols
        })
        continue
    
    opt_result = optimizer.optimize(
        model=base_model,
        param_grid=param_grid,
        X=X_feat,
        y=y,
        method="randomized",
        scoring="accuracy",
        cv_folds=5,
        n_iter=50,
        random_state=42,
    )
    
    print(f"  Best params: {opt_result['best_params']}")
    print(f"  Best CV score: {opt_result['best_score']:.4f}")
    
    tuned_results.append({
        "name": result.name,
        "cv_score": opt_result["best_score"],
        "model": opt_result["best_model"],
        "features": feature_cols,
        "params": opt_result["best_params"],
    })

# ============================================================
# 4. SELECT BEST AND SUBMIT
# ============================================================
print("\n" + "="*60)
print("PHASE 3: SUBMISSION")
print("="*60)

# Pick the best tuned model
best_tuned = max(tuned_results, key=lambda x: x["cv_score"])
print(f"\nBest tuned model: {best_tuned['name']}")
print(f"Tuned CV score: {best_tuned['cv_score']:.4f}")

# Fit on full training data
best_model = best_tuned["model"]
best_features = best_tuned["features"]
best_model.fit(X[best_features], y)

# Predict
predictions = best_model.predict(X_test[best_features])
predictions = predictions.astype(int)

# Create submission
sub_dir = os.path.join(BASE, "submissions")
os.makedirs(sub_dir, exist_ok=True)
sub_path = os.path.join(sub_dir, "arena_v3_tuned.csv")

submission = pd.DataFrame({
    "PassengerId": test_ids,
    "Survived": predictions
})
submission.to_csv(sub_path, index=False)
print(f"\nSubmission saved: {sub_path}")
print(f"Predictions distribution: {pd.Series(predictions).value_counts().to_dict()}")

# Submit via CLI
print("\n--- Submitting to Kaggle ---")
COMP_NAME = "titanic"
message = f"Arena_v3_tuned_{best_tuned['name']}_CV{best_tuned['cv_score']:.4f}"

result = subprocess.run(
    ["kaggle", "competitions", "submit", "-c", COMP_NAME, "-f", sub_path, "-m", message],
    capture_output=True, text=True
)
print(result.stdout)
if result.stderr:
    print(result.stderr)

# Wait and check score
print("Waiting 15s for score...")
time.sleep(15)
result = subprocess.run(
    ["kaggle", "competitions", "submissions", "-c", COMP_NAME],
    capture_output=True, text=True
)
print(result.stdout)

# ============================================================
# 5. SUMMARY
# ============================================================
print("\n" + "="*60)
print("TITANIC ARENA V3 SUMMARY")
print("="*60)
print(f"Previous best LB: 0.77272")
print(f"Arena winner: {results[0].name} (CV: {results[0].mean_score:.4f})")
print(f"Best tuned: {best_tuned['name']} (CV: {best_tuned['cv_score']:.4f})")
print(f"Estimated LB: {best_tuned['cv_score'] - 0.06:.4f}")
