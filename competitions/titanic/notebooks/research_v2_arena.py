"""Titanic Research V2 — Advanced Feature Engineering with Arena Evaluation.

Research-driven features (NO TARGET LEAKAGE):
1. Ticket-group SIZE (not survival — that leaks!)
2. Surname-based family SIZE
3. Better Age imputation (Title × Pclass median)
4. Fare per person
5. Cabin Deck ordinal
6. Quantile-transform Age and Fare
7. Feature interactions (Sex × Pclass, Age × Pclass)

LESSON FROM V2a: Computing survival rates from training data and using
them as features causes MASSIVE target leakage during CV (CV=98.8%, LB=0.727).
Instead, we use only structural group features (size, not survival rates).

Previous best: 0.77272 LB (GBM conservative, 8 features)
Target: 0.78+ LB
"""

import sys
import subprocess
import time
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import QuantileTransformer
from sklearn.ensemble import (
    GradientBoostingClassifier, RandomForestClassifier,
    HistGradientBoostingClassifier
)
from sklearn.linear_model import LogisticRegression

# Add project root to path
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from kaggle_ml_toolkit import ModelArena, CompetitionConfig, SubmissionGenerator

# ==============================================================================
# LOAD DATA
# ==============================================================================
data_dir = project_root / "competitions" / "titanic" / "data"
train_df = pd.read_csv(data_dir / "train.csv")
test_df = pd.read_csv(data_dir / "test.csv")

print(f"Train: {train_df.shape}, Test: {test_df.shape}")

test_ids = test_df["PassengerId"]
y = train_df["Survived"]

# Combine for consistent feature engineering
train_df["_is_train"] = True
test_df["_is_train"] = False
test_df["Survived"] = np.nan
combined = pd.concat([train_df, test_df], ignore_index=True)

# ==============================================================================
# FEATURE ENGINEERING (NO TARGET LEAKAGE)
# ==============================================================================

# --- Title extraction ---
combined["Title"] = combined["Name"].str.extract(r" ([A-Za-z]+)\.", expand=False)
title_map = {
    "Mr": "Mr", "Miss": "Miss", "Mrs": "Mrs", "Master": "Master",
    "Dr": "Rare", "Rev": "Rare", "Col": "Rare", "Major": "Rare",
    "Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs", "Lady": "Mrs",
    "Countess": "Mrs", "Sir": "Rare", "Don": "Rare", "Dona": "Mrs",
    "Jonkheer": "Rare", "Capt": "Rare",
}
combined["Title"] = combined["Title"].map(title_map).fillna("Rare")

# --- 3. Better Age imputation (Title × Pclass median) ---
age_medians = combined.groupby(["Title", "Pclass"])["Age"].transform("median")
combined["Age"] = combined["Age"].fillna(age_medians)
combined["Age"] = combined["Age"].fillna(combined["Age"].median())

# --- Basic features ---
combined["IsFemale"] = (combined["Sex"] == "female").astype(int)
combined["FamilySize"] = combined["SibSp"] + combined["Parch"] + 1
combined["IsAlone"] = (combined["FamilySize"] == 1).astype(int)
combined["IsChild"] = (combined["Age"] < 12).astype(int)

# --- Fare imputation ---
fare_median_by_class = combined.groupby("Pclass")["Fare"].transform("median")
combined["Fare"] = combined["Fare"].fillna(fare_median_by_class)
combined["Fare"] = combined["Fare"].fillna(combined["Fare"].median())

# --- 1. Ticket-group SIZE (structural, no leakage) ---
combined["TicketGroupSize"] = combined.groupby("Ticket")["PassengerId"].transform("count")
combined["IsTicketAlone"] = (combined["TicketGroupSize"] == 1).astype(int)

# --- 2. Surname-based family SIZE (structural, no leakage) ---
combined["Surname"] = combined["Name"].str.split(",").str[0].str.strip()
combined["SurnameGroupSize"] = combined.groupby("Surname")["PassengerId"].transform("count")

# --- 4. Fare per person ---
combined["FarePerPerson"] = combined["Fare"] / combined["TicketGroupSize"]

# --- 5. Cabin Deck (extract deck letter, ordinal) ---
combined["CabinDeck"] = combined["Cabin"].str[0]
deck_map = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7, "T": 8}
combined["DeckOrdinal"] = combined["CabinDeck"].map(deck_map).fillna(0).astype(int)
combined["HasCabin"] = (combined["Cabin"].notna()).astype(int)

# --- 6. Quantile-transform Age and Fare ---
qt_age = QuantileTransformer(output_distribution="normal", random_state=42)
combined["Age_qt"] = qt_age.fit_transform(combined[["Age"]])
qt_fare = QuantileTransformer(output_distribution="normal", random_state=42)
combined["Fare_qt"] = qt_fare.fit_transform(combined[["Fare"]])
combined["FarePerPerson_qt"] = qt_fare.fit_transform(combined[["FarePerPerson"]])

# --- 7. Feature interactions ---
# Sex × Pclass (6 levels)
combined["SexPclass"] = combined["IsFemale"] * 3 + (3 - combined["Pclass"])

# Age × Pclass (children in lower class still had priority)
combined["AgePclass"] = combined["Age"] * combined["Pclass"]

# Women & children first signal
combined["WomenChildren1st"] = ((combined["IsFemale"] == 1) | (combined["IsChild"] == 1)).astype(int) * (4 - combined["Pclass"])

# Embarked encoding
combined["Embarked"] = combined["Embarked"].fillna("S")
embarked_map = {"S": 0, "C": 1, "Q": 2}
combined["EmbarkedCode"] = combined["Embarked"].map(embarked_map).fillna(0).astype(int)

# Title encoding (ordinal by survival probability: Mrs > Miss > Master > Mr > Rare)
title_code_map = {"Mrs": 4, "Miss": 3, "Master": 2, "Rare": 1, "Mr": 0}
combined["TitleCode"] = combined["Title"].map(title_code_map).fillna(0).astype(int)

# Family size buckets (non-linear effect: 2-4 is best)
combined["FamilyBucket"] = 0  # alone
combined.loc[combined["FamilySize"].between(2, 4), "FamilyBucket"] = 1  # small
combined.loc[combined["FamilySize"] >= 5, "FamilyBucket"] = 2  # large

# ==============================================================================
# DEFINE FEATURE SETS
# ==============================================================================

research_v2_features = [
    "Pclass", "IsFemale", "Age_qt", "Fare_qt", "FarePerPerson_qt",
    "FamilySize", "IsAlone", "IsChild",
    "TicketGroupSize", "SurnameGroupSize",
    "DeckOrdinal", "HasCabin",
    "SexPclass", "AgePclass", "WomenChildren1st",
    "EmbarkedCode", "TitleCode", "FamilyBucket",
]

minimal_features = [
    "IsFemale", "Pclass", "Age_qt", "SexPclass",
    "WomenChildren1st", "FarePerPerson_qt", "IsChild", "FamilyBucket",
]

# ==============================================================================
# SPLIT BACK INTO TRAIN/TEST
# ==============================================================================
train_processed = combined[combined["_is_train"] == True].copy()
test_processed = combined[combined["_is_train"] == False].copy()

X_train = train_processed[research_v2_features].copy().fillna(0)
X_test = test_processed[research_v2_features].copy().fillna(0)

print(f"\nTrain NaN check: {X_train.isna().sum().sum()}")
print(f"Test NaN check: {X_test.isna().sum().sum()}")
print(f"Research V2 features: {len(research_v2_features)}")
print(f"Minimal features: {len(minimal_features)}")

# ==============================================================================
# RUN CUSTOM ARENA (focused on what works for Titanic)
# ==============================================================================
print("\n" + "=" * 60)
print("RUNNING ARENA — Titanic Research V2 (No Leakage)")
print("=" * 60)

arena = ModelArena(
    X=X_train,
    y=y,
    cv_folds=5,
    metric="accuracy",
    random_state=42,
    prior_cv_lb_gap=0.06,
)

# GBM variants (proven best for Titanic)
arena.add("GBM_ultra_conservative", GradientBoostingClassifier(
    n_estimators=30, max_depth=2, learning_rate=0.15,
    min_samples_leaf=20, subsample=0.7, random_state=42
), feature_cols=minimal_features)

arena.add("GBM_conservative_v2", GradientBoostingClassifier(
    n_estimators=50, max_depth=2, learning_rate=0.1,
    min_samples_leaf=15, subsample=0.8, random_state=42
), feature_cols=minimal_features)

arena.add("GBM_conservative_research", GradientBoostingClassifier(
    n_estimators=50, max_depth=2, learning_rate=0.1,
    min_samples_leaf=15, subsample=0.8, random_state=42
), feature_cols=research_v2_features)

arena.add("GBM_moderate_minimal", GradientBoostingClassifier(
    n_estimators=80, max_depth=3, learning_rate=0.1,
    min_samples_leaf=10, subsample=0.8, random_state=42
), feature_cols=minimal_features)

arena.add("GBM_moderate_research", GradientBoostingClassifier(
    n_estimators=80, max_depth=3, learning_rate=0.1,
    min_samples_leaf=10, subsample=0.8, random_state=42
), feature_cols=research_v2_features)

arena.add("GBM_sweetspot", GradientBoostingClassifier(
    n_estimators=40, max_depth=2, learning_rate=0.12,
    min_samples_leaf=15, subsample=0.8, random_state=42
), feature_cols=minimal_features)

arena.add("GBM_sweetspot_research", GradientBoostingClassifier(
    n_estimators=40, max_depth=2, learning_rate=0.12,
    min_samples_leaf=15, subsample=0.8, random_state=42
), feature_cols=research_v2_features)

# HistGradientBoosting (handles NaN, built-in regularization)
arena.add("HGB_conservative", HistGradientBoostingClassifier(
    max_iter=50, max_depth=3, learning_rate=0.1,
    min_samples_leaf=20, random_state=42
), feature_cols=minimal_features)

arena.add("HGB_moderate", HistGradientBoostingClassifier(
    max_iter=100, max_depth=3, learning_rate=0.1,
    min_samples_leaf=15, random_state=42
), feature_cols=research_v2_features)

arena.add("HGB_research_strong", HistGradientBoostingClassifier(
    max_iter=150, max_depth=4, learning_rate=0.08,
    min_samples_leaf=10, random_state=42
), feature_cols=research_v2_features)

# RF (diversity for comparison)
arena.add("RF_conservative", RandomForestClassifier(
    n_estimators=100, max_depth=4, min_samples_leaf=10, random_state=42
), feature_cols=minimal_features)

arena.add("RF_research", RandomForestClassifier(
    n_estimators=150, max_depth=5, min_samples_leaf=8, random_state=42
), feature_cols=research_v2_features)

# Logistic Regression (stable baseline)
arena.add("LR_minimal", LogisticRegression(
    C=1.0, max_iter=2000, random_state=42
), feature_cols=minimal_features)

arena.add("LR_research", LogisticRegression(
    C=0.5, max_iter=2000, random_state=42
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
    competition_name="titanic",
    problem_type="classification",
    primary_metric="accuracy",
    id_column="PassengerId",
    target_column="Survived",
)

# Get best model and predict
model, feature_cols = arena.get_best_model()
predictions = model.predict(X_test[feature_cols])
predictions = predictions.astype(int)

# Save submission
submission_dir = project_root / "competitions" / "titanic" / "submissions"
submission_dir.mkdir(parents=True, exist_ok=True)

sub_gen = SubmissionGenerator()
submission_path = sub_gen.generate(
    predictions=predictions,
    test_ids=test_ids,
    config=config,
    model_name="research_v2b_arena",
    output_dir=str(submission_dir),
)
print(f"Submission saved: {submission_path}")

# Submit to Kaggle
print("\nSubmitting to Kaggle...")
result = subprocess.run(
    ["kaggle", "competitions", "submit", "-c", "titanic", "-f", submission_path,
     "-m", "Research V2b: no-leakage features, SexPclass interaction, WomenChildren1st, FarePerPerson, better Age imputation, FamilyBucket"],
    capture_output=True, text=True
)
print(result.stdout)
if result.stderr:
    print(f"stderr: {result.stderr}")

# Check score after delay
print("\nWaiting for score...")
time.sleep(15)
result = subprocess.run(
    ["kaggle", "competitions", "submissions", "-c", "titanic"],
    capture_output=True, text=True
)
print(result.stdout)

print("\n✅ Titanic Research V2b complete!")
