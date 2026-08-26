"""
Spaceship Titanic Arena V3 — Full ArenaGenerator → Hyperparameter Tune → Submit Pipeline
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

from kaggle_ml_toolkit import ArenaGenerator, ModelArena, ModelOptimizer

# ============================================================
# 1. LOAD AND ENGINEER FEATURES
# ============================================================
BASE = r"c:\Users\profe\OneDrive\Desktop\Sean Obsidian\Kaggle\competitions\spaceship-titanic"
train = pd.read_csv(os.path.join(BASE, "data", "train.csv"))
test = pd.read_csv(os.path.join(BASE, "data", "test.csv"))

print(f"Raw train: {train.shape}, test: {test.shape}")

def engineer_features(df):
    """Feature engineering for Spaceship Titanic."""
    df = df.copy()
    
    # Parse PassengerId → GroupId, PersonInGroup
    df['GroupId'] = df['PassengerId'].str.split('_').str[0].astype(int)
    df['PersonInGroup'] = df['PassengerId'].str.split('_').str[1].astype(int)
    
    # Group size
    group_sizes = df.groupby('GroupId')['PassengerId'].transform('count')
    df['GroupSize'] = group_sizes
    df['IsAlone'] = (df['GroupSize'] == 1).astype(int)
    
    # Parse Cabin → Deck, Num, Side
    df['CabinDeck'] = df['Cabin'].str.split('/').str[0]
    df['CabinNum'] = pd.to_numeric(df['Cabin'].str.split('/').str[1], errors='coerce')
    df['CabinSide'] = df['Cabin'].str.split('/').str[2]
    
    # Cabin region (binning CabinNum)
    df['CabinRegion'] = pd.cut(df['CabinNum'], bins=[0, 300, 600, 900, 1200, 1500, 2000], 
                                labels=[0, 1, 2, 3, 4, 5]).astype(float)
    
    # Encode Deck
    deck_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'T': 7}
    df['DeckNum'] = df['CabinDeck'].map(deck_map)
    
    # Encode Side
    df['SideNum'] = (df['CabinSide'] == 'S').astype(float)
    
    # CryoSleep encoding
    df['CryoSleep'] = df['CryoSleep'].astype(float)
    
    # VIP encoding
    df['VIP'] = df['VIP'].astype(float)
    
    # HomePlanet encoding
    planet_map = {'Earth': 0, 'Europa': 1, 'Mars': 2}
    df['HomePlanetNum'] = df['HomePlanet'].map(planet_map)
    
    # Destination encoding
    dest_map = {'TRAPPIST-1e': 0, '55 Cancri e': 1, 'PSO J318.5-22': 2}
    df['DestinationNum'] = df['Destination'].map(dest_map)
    
    # Spending features
    spend_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    for col in spend_cols:
        df[col] = df[col].fillna(0)
    
    df['TotalSpending'] = df[spend_cols].sum(axis=1)
    df['LogTotalSpending'] = np.log1p(df['TotalSpending'])
    df['NoSpending'] = (df['TotalSpending'] == 0).astype(int)
    
    # Per-amenity spending flags
    for col in spend_cols:
        df[f'Spent_{col}'] = (df[col] > 0).astype(int)
    df['NumServicesUsed'] = df[[f'Spent_{col}' for col in spend_cols]].sum(axis=1)
    
    # Spending ratios
    total_safe = df['TotalSpending'].replace(0, 1)
    for col in spend_cols:
        df[f'{col}_Ratio'] = df[col] / total_safe
    
    # Log individual spending
    for col in spend_cols:
        df[f'Log_{col}'] = np.log1p(df[col])
    
    # Age features
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['IsChild'] = (df['Age'] < 12).astype(int)
    df['IsTeenager'] = ((df['Age'] >= 12) & (df['Age'] < 18)).astype(int)
    df['IsYoungAdult'] = ((df['Age'] >= 18) & (df['Age'] < 25)).astype(int)
    df['IsAdult'] = ((df['Age'] >= 25) & (df['Age'] < 60)).astype(int)
    df['IsElder'] = (df['Age'] >= 60).astype(int)
    
    # Interaction features
    df['EarthCryo'] = ((df['HomePlanetNum'] == 0) & (df['CryoSleep'] == 1)).astype(float)
    df['EuropaCryo'] = ((df['HomePlanetNum'] == 1) & (df['CryoSleep'] == 1)).astype(float)
    df['MarsCryo'] = ((df['HomePlanetNum'] == 2) & (df['CryoSleep'] == 1)).astype(float)
    df['CryoNoSpend'] = ((df['CryoSleep'] == 1) & (df['NoSpending'] == 1)).astype(float)
    
    # Fill NaN
    df = df.fillna(0)
    
    return df

train_eng = engineer_features(train)
test_eng = engineer_features(test)

y = train_eng['Transported'].astype(int)
test_ids = test_eng['PassengerId']

# Feature sets
core_22 = [
    'CryoSleep', 'Age', 'TotalSpending', 'LogTotalSpending', 'NoSpending',
    'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck',
    'HomePlanetNum', 'DestinationNum', 'DeckNum', 'SideNum', 'CabinNum',
    'GroupSize', 'IsAlone', 'VIP', 'IsChild', 'IsTeenager', 'IsYoungAdult', 'NumServicesUsed'
]

extended_50 = core_22 + [
    'Spent_RoomService', 'Spent_FoodCourt', 'Spent_ShoppingMall', 'Spent_Spa', 'Spent_VRDeck',
    'RoomService_Ratio', 'FoodCourt_Ratio', 'ShoppingMall_Ratio', 'Spa_Ratio', 'VRDeck_Ratio',
    'Log_RoomService', 'Log_FoodCourt', 'Log_ShoppingMall', 'Log_Spa', 'Log_VRDeck',
    'EarthCryo', 'EuropaCryo', 'MarsCryo', 'CryoNoSpend',
    'CabinRegion', 'IsAdult', 'IsElder', 'PersonInGroup',
    'HomePlanetNum', 'DestinationNum',  # these may be duplicated but pandas handles it
]
# Remove duplicates
extended_50 = list(dict.fromkeys(extended_50))

# Verify all columns exist
core_22 = [c for c in core_22 if c in train_eng.columns]
extended_50 = [c for c in extended_50 if c in train_eng.columns]

X = train_eng[extended_50].copy()
X_test = test_eng[extended_50].copy()

print(f"Feature sets: core_22={len(core_22)}, extended_50={len(extended_50)}")

# ============================================================
# 2. ARENA GENERATOR
# ============================================================
print("\n" + "="*60)
print("PHASE 1: ARENA GENERATOR")
print("="*60)

gen = ArenaGenerator(
    X=train_eng[extended_50],
    y=y,
    problem_type="classification",
    feature_sets={"core_22": core_22, "extended_50": extended_50},
    time_budget_minutes=10,
    prior_cv_lb_gap=0.01,
)

arena = gen.build_arena(verbose=True)

# Add HGB models (primary model from research)
arena.add("HGB_ext_conservative", HistGradientBoostingClassifier(
    max_iter=200, max_depth=4, learning_rate=0.1, min_samples_leaf=15, random_state=42
), feature_cols=extended_50)
arena.add("HGB_ext_strong", HistGradientBoostingClassifier(
    max_iter=300, max_depth=5, learning_rate=0.05, min_samples_leaf=10, random_state=42
), feature_cols=extended_50)
arena.add("HGB_ext_very_strong", HistGradientBoostingClassifier(
    max_iter=500, max_depth=5, learning_rate=0.03, min_samples_leaf=5, random_state=42
), feature_cols=extended_50)
arena.add("HGB_core_strong", HistGradientBoostingClassifier(
    max_iter=300, max_depth=5, learning_rate=0.05, min_samples_leaf=10, random_state=42
), feature_cols=core_22)
arena.add("HGB_ext_aggressive", HistGradientBoostingClassifier(
    max_iter=500, max_depth=6, learning_rate=0.05, min_samples_leaf=5, random_state=42
), feature_cols=extended_50)

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
top3 = results[:3]
tuned_results = []

for i, result in enumerate(top3):
    print(f"\n--- Tuning #{i+1}: {result.name} (CV: {result.mean_score:.4f}) ---")
    
    model_type = result.model_type
    feature_cols = result.feature_cols
    X_feat = train_eng[feature_cols]
    
    if model_type == "HistGradientBoostingClassifier":
        base_model = HistGradientBoostingClassifier(random_state=42)
        param_grid = {
            "max_iter": [200, 300, 400, 500, 700],
            "max_depth": [4, 5, 6, 7, 8],
            "learning_rate": [0.01, 0.03, 0.05, 0.08, 0.1],
            "min_samples_leaf": [3, 5, 8, 10, 15, 20],
            "l2_regularization": [0.0, 0.01, 0.1, 0.5, 1.0],
            "max_bins": [128, 255],
        }
    elif model_type == "GradientBoostingClassifier":
        base_model = GradientBoostingClassifier(random_state=42)
        param_grid = {
            "n_estimators": [100, 200, 300, 500],
            "max_depth": [3, 4, 5, 6],
            "learning_rate": [0.03, 0.05, 0.08, 0.1],
            "min_samples_leaf": [5, 8, 10, 15],
            "subsample": [0.7, 0.8, 0.9, 1.0],
        }
    elif model_type == "RandomForestClassifier":
        base_model = RandomForestClassifier(random_state=42)
        param_grid = {
            "n_estimators": [100, 200, 300, 500],
            "max_depth": [5, 8, 10, 12, 15],
            "min_samples_leaf": [2, 3, 5, 8, 10],
            "max_features": ["sqrt", "log2", 0.3, 0.5],
        }
    elif model_type == "LogisticRegression":
        base_model = LogisticRegression(max_iter=2000, random_state=42)
        param_grid = {
            "C": [0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0],
            "solver": ["lbfgs", "liblinear"],
        }
    else:
        print(f"  Skipping tuning for {model_type}")
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

# Pick best tuned model
valid_tuned = [t for t in tuned_results if t["model"] is not None]
best_tuned = max(valid_tuned, key=lambda x: x["cv_score"])
print(f"\nBest tuned model: {best_tuned['name']}")
print(f"Tuned CV score: {best_tuned['cv_score']:.4f}")

# Fit on full training data
best_model = best_tuned["model"].__class__(**best_tuned["model"].get_params())
best_features = best_tuned["features"]
best_model.fit(train_eng[best_features], y)

# Predict
predictions = best_model.predict(test_eng[best_features])

# Convert to True/False strings for submission
predictions_bool = [str(bool(p)) for p in predictions]

# Create submission
sub_dir = os.path.join(BASE, "submissions")
os.makedirs(sub_dir, exist_ok=True)
sub_path = os.path.join(sub_dir, "arena_v3_tuned.csv")

submission = pd.DataFrame({
    "PassengerId": test_ids,
    "Transported": predictions_bool
})
submission.to_csv(sub_path, index=False)
print(f"\nSubmission saved: {sub_path}")
print(f"Predictions distribution: {pd.Series(predictions_bool).value_counts().to_dict()}")

# Submit via CLI
print("\n--- Submitting to Kaggle ---")
COMP_NAME = "spaceship-titanic"
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
print("SPACESHIP TITANIC ARENA V3 SUMMARY")
print("="*60)
print(f"Previous best LB: 0.80243")
print(f"Arena winner: {results[0].name} (CV: {results[0].mean_score:.4f})")
print(f"Best tuned: {best_tuned['name']} (CV: {best_tuned['cv_score']:.4f})")
print(f"Estimated LB: {best_tuned['cv_score'] - 0.01:.4f}")
