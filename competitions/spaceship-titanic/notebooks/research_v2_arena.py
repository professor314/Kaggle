"""Spaceship Titanic Research V2 — Advanced Feature Engineering (NO LEAKAGE).

Research-driven features:
1. Group SIZE features (not survival rate — that leaks!)
2. Per-amenity spending flags
3. Spending ratios
4. Cabin number proximity (binned cabin regions)
5. Granular age groups
6. HomePlanet × CryoSleep interaction
7. HistGradientBoosting for native NaN handling

LESSON: GroupSurvivalRate computed from training labels caused MASSIVE
target leakage (CV=95.2%, LB=0.493). Only use structural group features.

Previous best: 0.79939 LB (GBM moderate, 22 features)
Target: 0.81+ LB
"""

import sys
import subprocess
import time
import numpy as np
import pandas as pd
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from kaggle_ml_toolkit import ModelArena, CompetitionConfig, SubmissionGenerator

# ==============================================================================
# LOAD DATA
# ==============================================================================
data_dir = project_root / "competitions" / "spaceship-titanic" / "data"
train_df = pd.read_csv(data_dir / "train.csv")
test_df = pd.read_csv(data_dir / "test.csv")

print(f"Train: {train_df.shape}, Test: {test_df.shape}")

test_ids = test_df["PassengerId"]
y = train_df["Transported"].astype(int)

# Combine for consistent feature engineering
train_df["_is_train"] = True
test_df["_is_train"] = False
test_df["Transported"] = np.nan
combined = pd.concat([train_df, test_df], ignore_index=True)

# ==============================================================================
# FEATURE ENGINEERING (NO TARGET LEAKAGE)
# ==============================================================================

# --- Parse PassengerId into Group and Person ---
combined["Group"] = combined["PassengerId"].str.split("_").str[0]
combined["PersonInGroup"] = combined["PassengerId"].str.split("_").str[1].astype(int)
combined["GroupSize"] = combined.groupby("Group")["PassengerId"].transform("count")
combined["IsAlone"] = (combined["GroupSize"] == 1).astype(int)

# --- Parse Cabin: Deck/Num/Side ---
combined["CabinDeck"] = combined["Cabin"].str.split("/").str[0]
combined["CabinNum"] = combined["Cabin"].str.split("/").str[1]
combined["CabinSide"] = combined["Cabin"].str.split("/").str[2]

# Encode deck
deck_map = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5, "G": 6, "T": 7}
combined["DeckNum"] = combined["CabinDeck"].map(deck_map).fillna(-1).astype(int)

# Side encoding
combined["IsPort"] = (combined["CabinSide"] == "P").astype(int)
combined["IsStar"] = (combined["CabinSide"] == "S").astype(int)

# --- 4. Cabin number proximity (binned regions) ---
combined["CabinNumeric"] = pd.to_numeric(combined["CabinNum"], errors="coerce")
combined["CabinRegion"] = pd.cut(
    combined["CabinNumeric"],
    bins=[0, 300, 600, 900, 1200, 1500, 2000],
    labels=[0, 1, 2, 3, 4, 5],
    include_lowest=True
).astype(float).fillna(-1).astype(int)

# --- Spending features ---
spending_cols = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
for col in spending_cols:
    combined[col] = combined[col].fillna(0)

combined["TotalSpending"] = combined[spending_cols].sum(axis=1)
combined["LogSpending"] = np.log1p(combined["TotalSpending"])
combined["NoSpending"] = (combined["TotalSpending"] == 0).astype(int)

# --- 2. Per-amenity spending flags ---
combined["SpentOnRoomService"] = (combined["RoomService"] > 0).astype(int)
combined["SpentOnFoodCourt"] = (combined["FoodCourt"] > 0).astype(int)
combined["SpentOnShoppingMall"] = (combined["ShoppingMall"] > 0).astype(int)
combined["SpentOnSpa"] = (combined["Spa"] > 0).astype(int)
combined["SpentOnVRDeck"] = (combined["VRDeck"] > 0).astype(int)
combined["NumServicesUsed"] = (
    combined["SpentOnRoomService"] + combined["SpentOnFoodCourt"] +
    combined["SpentOnShoppingMall"] + combined["SpentOnSpa"] + combined["SpentOnVRDeck"]
)

# --- 3. Spending ratios ---
total_plus_one = combined["TotalSpending"] + 1
combined["RoomServiceRatio"] = combined["RoomService"] / total_plus_one
combined["FoodCourtRatio"] = combined["FoodCourt"] / total_plus_one
combined["ShoppingMallRatio"] = combined["ShoppingMall"] / total_plus_one
combined["SpaRatio"] = combined["Spa"] / total_plus_one
combined["VRDeckRatio"] = combined["VRDeck"] / total_plus_one

# Log transform individual spending
for col in spending_cols:
    combined[f"Log{col}"] = np.log1p(combined[col])

# --- CryoSleep and VIP ---
combined["CryoSleep"] = combined["CryoSleep"].fillna(False).astype(int)
combined["VIP"] = combined["VIP"].fillna(False).astype(int)

# --- Age features ---
combined["Age"] = combined["Age"].fillna(combined["Age"].median())
combined["IsChild"] = (combined["Age"] < 12).astype(int)
combined["IsElder"] = (combined["Age"] > 60).astype(int)

# --- 5. Granular age groups ---
age_bins = [0, 5, 12, 18, 25, 40, 60, 100]
age_labels = [0, 1, 2, 3, 4, 5, 6]
combined["AgeGroup"] = pd.cut(combined["Age"], bins=age_bins, labels=age_labels, include_lowest=True)
combined["AgeGroup"] = combined["AgeGroup"].astype(float).fillna(3).astype(int)

# --- HomePlanet encoding ---
combined["HomePlanet"] = combined["HomePlanet"].fillna("Unknown")
combined["FromEarth"] = (combined["HomePlanet"] == "Earth").astype(int)
combined["FromEuropa"] = (combined["HomePlanet"] == "Europa").astype(int)
combined["FromMars"] = (combined["HomePlanet"] == "Mars").astype(int)

# --- Destination encoding ---
combined["Destination"] = combined["Destination"].fillna("Unknown")
combined["ToTRAPPIST"] = (combined["Destination"] == "TRAPPIST-1e").astype(int)
combined["To55Cancri"] = (combined["Destination"] == "55 Cancri e").astype(int)
combined["ToPSO"] = (combined["Destination"] == "PSO J318.5-22").astype(int)

# --- 6. HomePlanet × CryoSleep interaction ---
combined["EarthCryo"] = combined["FromEarth"] * combined["CryoSleep"]
combined["EuropaCryo"] = combined["FromEuropa"] * combined["CryoSleep"]
combined["MarsCryo"] = combined["FromMars"] * combined["CryoSleep"]

# Planet × VIP
combined["EarthVIP"] = combined["FromEarth"] * combined["VIP"]
combined["EuropaVIP"] = combined["FromEuropa"] * combined["VIP"]

# Age × CryoSleep
combined["AgeCryo"] = combined["Age"] * combined["CryoSleep"]

# ==============================================================================
# DEFINE FEATURE SETS (NO GroupSurvivalRate!)
# ==============================================================================

research_v2_features = [
    # Core
    "CryoSleep", "Age", "IsChild", "IsElder", "AgeGroup", "VIP",
    # Spending
    "TotalSpending", "LogSpending", "NoSpending", "NumServicesUsed",
    "RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck",
    "LogRoomService", "LogFoodCourt", "LogShoppingMall", "LogSpa", "LogVRDeck",
    # Spending flags
    "SpentOnRoomService", "SpentOnFoodCourt", "SpentOnShoppingMall",
    "SpentOnSpa", "SpentOnVRDeck",
    # Spending ratios
    "RoomServiceRatio", "FoodCourtRatio", "ShoppingMallRatio",
    "SpaRatio", "VRDeckRatio",
    # Cabin
    "DeckNum", "IsPort", "IsStar", "CabinRegion", "CabinNumeric",
    # Group (structural only, NO survival rates)
    "GroupSize", "IsAlone", "PersonInGroup",
    # Location
    "FromEarth", "FromEuropa", "FromMars",
    "ToTRAPPIST", "To55Cancri", "ToPSO",
    # Interactions
    "EarthCryo", "EuropaCryo", "MarsCryo",
    "EarthVIP", "EuropaVIP", "AgeCryo",
]

# Fill remaining NaN
for col in research_v2_features:
    combined[col] = combined[col].fillna(0)

# ==============================================================================
# SPLIT BACK INTO TRAIN/TEST
# ==============================================================================
train_processed = combined[combined["_is_train"] == True].copy()
test_processed = combined[combined["_is_train"] == False].copy()

X_train = train_processed[research_v2_features].copy()
X_test = test_processed[research_v2_features].copy()

print(f"\nFeature count: {len(research_v2_features)}")
print(f"Train NaN check: {X_train.isna().sum().sum()}")
print(f"Test NaN check: {X_test.isna().sum().sum()}")

# ==============================================================================
# RUN ARENA
# ==============================================================================
print("\n" + "=" * 60)
print("RUNNING ARENA — Spaceship Titanic Research V2 (No Leakage)")
print("=" * 60)

from sklearn.ensemble import HistGradientBoostingClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

arena = ModelArena(
    X=X_train,
    y=y,
    cv_folds=5,
    metric="accuracy",
    random_state=42,
    prior_cv_lb_gap=0.01,
)

# HistGradientBoosting variants
arena.add("HGB_conservative", HistGradientBoostingClassifier(
    max_iter=150, max_depth=4, learning_rate=0.08, min_samples_leaf=20, random_state=42
), feature_cols=research_v2_features)

arena.add("HGB_moderate", HistGradientBoostingClassifier(
    max_iter=200, max_depth=4, learning_rate=0.05, min_samples_leaf=15, random_state=42
), feature_cols=research_v2_features)

arena.add("HGB_strong", HistGradientBoostingClassifier(
    max_iter=300, max_depth=5, learning_rate=0.05, min_samples_leaf=12, random_state=42
), feature_cols=research_v2_features)

arena.add("HGB_very_strong", HistGradientBoostingClassifier(
    max_iter=400, max_depth=5, learning_rate=0.04, min_samples_leaf=10, random_state=42
), feature_cols=research_v2_features)

arena.add("HGB_aggressive", HistGradientBoostingClassifier(
    max_iter=500, max_depth=6, learning_rate=0.03, min_samples_leaf=8, random_state=42
), feature_cols=research_v2_features)

# GBM variants
arena.add("GBM_moderate", GradientBoostingClassifier(
    n_estimators=200, max_depth=4, learning_rate=0.05,
    min_samples_leaf=8, subsample=0.8, random_state=42
), feature_cols=research_v2_features)

arena.add("GBM_strong", GradientBoostingClassifier(
    n_estimators=300, max_depth=5, learning_rate=0.05,
    min_samples_leaf=5, subsample=0.8, random_state=42
), feature_cols=research_v2_features)

# RF
arena.add("RF_moderate", RandomForestClassifier(
    n_estimators=200, max_depth=8, min_samples_leaf=5, random_state=42
), feature_cols=research_v2_features)

arena.add("RF_strong", RandomForestClassifier(
    n_estimators=300, max_depth=10, min_samples_leaf=3, random_state=42
), feature_cols=research_v2_features)

arena.add("RF_deep", RandomForestClassifier(
    n_estimators=400, max_depth=12, min_samples_leaf=2, random_state=42
), feature_cols=research_v2_features)

results = arena.run(verbose=True)

# Print leaderboard
print("\n" + "=" * 60)
print("LEADERBOARD")
print("=" * 60)
lb = arena.leaderboard()
print(lb.to_string(index=False))

# Print recommendation
print("\n" + "=" * 60)
print("RECOMMENDATION")
print("=" * 60)
rec = arena.recommend()
for k, v in rec.items():
    print(f"  {k}: {v}")

# ==============================================================================
# GENERATE SUBMISSION
# ==============================================================================
print("\n" + "=" * 60)
print("GENERATING SUBMISSION")
print("=" * 60)

config = CompetitionConfig(
    competition_name="spaceship-titanic",
    problem_type="classification",
    primary_metric="accuracy",
    id_column="PassengerId",
    target_column="Transported",
)

# Get best model and predict
model, feature_cols = arena.get_best_model()
predictions = model.predict(X_test[feature_cols])
predictions_bool = predictions.astype(bool)

# Save submission
submission_dir = project_root / "competitions" / "spaceship-titanic" / "submissions"
submission_dir.mkdir(parents=True, exist_ok=True)

sub_gen = SubmissionGenerator()
submission_path = sub_gen.generate(
    predictions=predictions_bool,
    test_ids=test_ids,
    config=config,
    model_name="research_v2b_arena",
    output_dir=str(submission_dir),
)
print(f"Submission saved: {submission_path}")

# Submit to Kaggle
print("\nSubmitting to Kaggle...")
result = subprocess.run(
    ["kaggle", "competitions", "submit", "-c", "spaceship-titanic", "-f", submission_path,
     "-m", "Research V2b: no-leakage, spending ratios, cabin regions, age groups, HomePlanet×CryoSleep, HGB, 48 features"],
    capture_output=True, text=True
)
print(result.stdout)
if result.stderr:
    print(f"stderr: {result.stderr}")

# Check score
print("\nWaiting for score...")
time.sleep(15)
result = subprocess.run(
    ["kaggle", "competitions", "submissions", "-c", "spaceship-titanic"],
    capture_output=True, text=True
)
print(result.stdout)

print("\n✅ Spaceship Titanic Research V2b complete!")
