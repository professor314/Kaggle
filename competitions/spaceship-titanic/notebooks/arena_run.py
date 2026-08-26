"""Spaceship Titanic — Classification arena run.

Predict which passengers were transported to an alternate dimension.
Binary classification, ~8700 training rows (larger than Titanic!).
Applying lessons from strategies.md: can afford more complexity here.
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import pandas as pd
import numpy as np
from kaggle_ml_toolkit import ModelArena, CompetitionConfig, SubmissionGenerator
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

# === LOAD DATA ===
data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
train_df = pd.read_csv(os.path.join(data_dir, "train.csv"))
test_df = pd.read_csv(os.path.join(data_dir, "test.csv"))
print(f"Train: {train_df.shape}, Test: {test_df.shape}")
print(f"Target distribution: {train_df['Transported'].value_counts().to_dict()}")
print(f"Columns: {list(train_df.columns)}")

# === EDA QUICK LOOK ===
print(f"\nMissing values (top 5):")
missing = train_df.isnull().sum().sort_values(ascending=False).head(10)
for col, count in missing.items():
    if count > 0:
        print(f"  {col}: {count} ({count/len(train_df)*100:.1f}%)")

# === FEATURE ENGINEERING ===
def prepare(df):
    df = df.copy()
    
    # Parse PassengerId: gggg_pp (group_number, person_in_group)
    df['GroupId'] = df['PassengerId'].str.split('_').str[0].astype(int)
    df['PersonInGroup'] = df['PassengerId'].str.split('_').str[1].astype(int)
    
    # Parse Cabin: deck/num/side
    df['Deck'] = df['Cabin'].str.split('/').str[0]
    df['CabinNum'] = pd.to_numeric(df['Cabin'].str.split('/').str[1], errors='coerce')
    df['Side'] = df['Cabin'].str.split('/').str[2]
    
    # Binary features
    df['IsPort'] = (df['Side'] == 'P').astype(int)
    df['CryoSleep'] = df['CryoSleep'].fillna(False).astype(int)
    df['VIP'] = df['VIP'].fillna(False).astype(int)
    
    # Total spending (luxury amenities)
    spending_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    for col in spending_cols:
        df[col] = df[col].fillna(0)
    df['TotalSpending'] = df[spending_cols].sum(axis=1)
    df['NoSpending'] = (df['TotalSpending'] == 0).astype(int)
    
    # Log spending (right-skewed)
    df['LogSpending'] = np.log1p(df['TotalSpending'])
    
    # Age
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['IsChild'] = (df['Age'] < 13).astype(int)
    df['IsElder'] = (df['Age'] > 60).astype(int)
    
    # Group size
    group_sizes = df.groupby('GroupId')['PassengerId'].transform('count')
    df['GroupSize'] = group_sizes
    df['IsAlone'] = (df['GroupSize'] == 1).astype(int)
    
    # HomePlanet encoding
    df['FromEarth'] = (df['HomePlanet'] == 'Earth').fillna(0).astype(int)
    df['FromEuropa'] = (df['HomePlanet'] == 'Europa').fillna(0).astype(int)
    df['FromMars'] = (df['HomePlanet'] == 'Mars').fillna(0).astype(int)
    
    # Destination encoding
    df['ToTRAPPIST'] = (df['Destination'] == 'TRAPPIST-1e').fillna(0).astype(int)
    df['To55Cancri'] = (df['Destination'] == '55 Cancri e').fillna(0).astype(int)
    
    # Deck encoding (ordinal-ish based on common patterns)
    deck_map = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7, 'T': 8}
    df['DeckNum'] = df['Deck'].map(deck_map).fillna(0).astype(int)
    
    return df

train_df = prepare(train_df)
test_df = prepare(test_df)

y = train_df['Transported'].astype(int)

# === FEATURE SETS ===
minimal = ['CryoSleep', 'TotalSpending', 'Age']
core = ['CryoSleep', 'TotalSpending', 'LogSpending', 'NoSpending', 'Age', 'IsChild',
        'VIP', 'GroupSize', 'IsAlone', 'DeckNum', 'IsPort']
extended = core + ['FromEarth', 'FromEuropa', 'FromMars', 'ToTRAPPIST', 'To55Cancri',
                   'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'CabinNum']

# Fill any remaining NaN
for col in extended:
    med = train_df[col].median() if train_df[col].dtype != 'object' else 0
    train_df[col] = train_df[col].fillna(med)
    test_df[col] = test_df[col].fillna(med)

print(f"\nFeature sets: minimal={len(minimal)}, core={len(core)}, extended={len(extended)}")

# === MODEL ARENA ===
# 8700 rows — from strategies.md: can use depth 5-8, 10-30 features
arena = ModelArena(
    X=train_df,
    y=y,
    cv_folds=5,
    metric='accuracy',
    random_state=42,
    prior_cv_lb_gap=0.02,  # expect smaller gap on larger dataset
)

# Logistic Regression
arena.add('LR_core', LogisticRegression(max_iter=1000, C=1.0, random_state=42), core)
arena.add('LR_extended', LogisticRegression(max_iter=1000, C=0.5, random_state=42), extended)

# Random Forest
arena.add('RF_moderate', RandomForestClassifier(n_estimators=200, max_depth=8, min_samples_leaf=5, random_state=42), core)
arena.add('RF_extended', RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_leaf=3, random_state=42), extended)
arena.add('RF_conservative', RandomForestClassifier(n_estimators=150, max_depth=6, min_samples_leaf=10, random_state=42), core)

# Gradient Boosting
arena.add('GBM_conservative', GradientBoostingClassifier(n_estimators=100, max_depth=4, learning_rate=0.1, min_samples_leaf=10, subsample=0.8, random_state=42), core)
arena.add('GBM_moderate', GradientBoostingClassifier(n_estimators=200, max_depth=4, learning_rate=0.05, min_samples_leaf=8, subsample=0.8, random_state=42), extended)
arena.add('GBM_strong', GradientBoostingClassifier(n_estimators=300, max_depth=5, learning_rate=0.05, min_samples_leaf=5, subsample=0.8, random_state=42), extended)
arena.add('GBM_core_strong', GradientBoostingClassifier(n_estimators=200, max_depth=5, learning_rate=0.05, min_samples_leaf=5, subsample=0.8, random_state=42), core)

# Run
print("\n")
results = arena.run()
print("\n")
print(arena.leaderboard().to_string(index=False))

rec = arena.recommend()
print(f"\nRECOMMENDATION: {rec['reasoning']}")
if rec['warnings']:
    for w in rec['warnings']:
        print(f"  ⚠️  {w}")

# === GENERATE SUBMISSION ===
config = CompetitionConfig(
    competition_name='spaceship-titanic',
    target_column='Transported',
    id_column='PassengerId',
    problem_type='classification',
    primary_metric='accuracy',
    random_seed=42,
)

best_model, best_features = arena.get_best_model()
preds = best_model.predict(test_df[best_features])
# Convert predictions to True/False strings for Kaggle
preds_bool = pd.Series(preds).map({1: True, 0: False}).values

# Manual submission generation (Kaggle expects True/False, not 0/1)
sub_dir = os.path.join(os.path.dirname(__file__), "..", "submissions")
os.makedirs(sub_dir, exist_ok=True)
from datetime import datetime
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
sub_path = os.path.join(sub_dir, f"arena_{rec['recommended_name']}_{ts}.csv")
sub_df = pd.DataFrame({'PassengerId': test_df['PassengerId'], 'Transported': preds_bool})
sub_df.to_csv(sub_path, index=False)
print(f"\nSubmission: {os.path.basename(sub_path)}")
print(f"Transported: {sum(preds)}, Not: {len(preds)-sum(preds)}")

# Save report
report = arena.summary_report()
report_path = os.path.join(os.path.dirname(__file__), "..", "content", "arena_results.md")
os.makedirs(os.path.dirname(report_path), exist_ok=True)
with open(report_path, 'w', encoding='utf-8') as f:
    f.write(f"# Spaceship Titanic — Arena Results\n\n{report}")
print(f"Report saved.")
