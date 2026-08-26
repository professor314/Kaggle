"""Titanic Model Arena — Big loop comparing many pipeline configurations."""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import pandas as pd
import numpy as np
from kaggle_ml_toolkit import ModelArena, CompetitionConfig, SubmissionGenerator
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

# Load and prepare data
train_df = pd.read_csv(os.path.join(os.path.dirname(__file__), "..", "data", "train.csv"))
test_df = pd.read_csv(os.path.join(os.path.dirname(__file__), "..", "data", "test.csv"))

def prepare(df):
    df = df.copy()
    df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
    df['Title'] = df['Title'].replace({
        'Mlle':'Miss','Ms':'Miss','Mme':'Mrs',
        'Lady':'Rare','Countess':'Rare','Capt':'Rare','Col':'Rare',
        'Don':'Rare','Dr':'Rare','Major':'Rare','Rev':'Rare',
        'Sir':'Rare','Jonkheer':'Rare','Dona':'Rare'
    })
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
    df['IsFemale'] = (df['Sex'] == 'female').astype(int)
    df['Pclass_1'] = (df['Pclass'] == 1).astype(int)
    df['Pclass_3'] = (df['Pclass'] == 3).astype(int)
    df['FemaleUpperClass'] = ((df['Sex'] == 'female') & (df['Pclass'] <= 2)).astype(int)
    df['HasCabin'] = df['Cabin'].notna().astype(int)
    title_map = {'Mr': 0, 'Miss': 1, 'Mrs': 2, 'Master': 3, 'Rare': 4}
    df['TitleCode'] = df['Title'].map(title_map).fillna(4).astype(int)
    return df

train_df = prepare(train_df)
test_df = prepare(test_df)

# Impute Age using title medians
for title in train_df['Title'].unique():
    med = train_df.loc[train_df['Title'] == title, 'Age'].median()
    train_df.loc[(train_df['Title'] == title) & train_df['Age'].isna(), 'Age'] = med
    test_df.loc[(test_df['Title'] == title) & test_df['Age'].isna(), 'Age'] = med

global_med = train_df['Age'].median()
train_df['Age'] = train_df['Age'].fillna(global_med)
test_df['Age'] = test_df['Age'].fillna(global_med)
train_df['Embarked'] = train_df['Embarked'].fillna('S')
test_df['Fare'] = test_df['Fare'].fillna(test_df['Fare'].median())

# IsChild (after imputation)
train_df['IsChild'] = (train_df['Age'] < 12).astype(int)
test_df['IsChild'] = (test_df['Age'] < 12).astype(int)

y = train_df['Survived']

# --- Feature sets ---
minimal = ['IsFemale', 'Pclass']
simple = ['IsFemale', 'Pclass', 'IsChild', 'FamilySize']
moderate = ['IsFemale', 'Pclass', 'Age', 'Fare', 'FamilySize', 'IsAlone', 'IsChild', 'TitleCode']
interaction = ['IsFemale', 'Pclass', 'Age', 'Fare', 'FamilySize', 'IsAlone', 'IsChild',
               'TitleCode', 'FemaleUpperClass', 'Pclass_1', 'Pclass_3', 'HasCabin']

# --- Create Arena ---
# prior_cv_lb_gap based on our 2 submissions: avg gap ~6.6%
arena = ModelArena(
    X=train_df,
    y=y,
    cv_folds=5,
    metric='accuracy',
    random_state=42,
    prior_cv_lb_gap=0.065,
)

# --- Add pipeline configurations ---

# Ultra-simple baselines
arena.add('LR_minimal', LogisticRegression(max_iter=1000, random_state=42), minimal)
arena.add('LR_simple', LogisticRegression(max_iter=1000, random_state=42), simple)
arena.add('LR_moderate', LogisticRegression(max_iter=1000, C=1.0, random_state=42), moderate)
arena.add('LR_regularized', LogisticRegression(max_iter=1000, C=0.1, random_state=42), moderate)
arena.add('LR_interaction', LogisticRegression(max_iter=1000, C=0.5, random_state=42), interaction)

# Decision Trees (shallow = less overfitting)
arena.add('DT_depth2', DecisionTreeClassifier(max_depth=2, random_state=42), simple)
arena.add('DT_depth3', DecisionTreeClassifier(max_depth=3, random_state=42), moderate)
arena.add('DT_depth4', DecisionTreeClassifier(max_depth=4, random_state=42), moderate)

# Random Forest variations
arena.add('RF_shallow', RandomForestClassifier(n_estimators=100, max_depth=3, min_samples_leaf=10, random_state=42), moderate)
arena.add('RF_moderate', RandomForestClassifier(n_estimators=200, max_depth=5, min_samples_leaf=5, random_state=42), moderate)
arena.add('RF_conservative', RandomForestClassifier(n_estimators=150, max_depth=4, min_samples_leaf=8, random_state=42), moderate)
arena.add('RF_interaction', RandomForestClassifier(n_estimators=150, max_depth=4, min_samples_leaf=8, random_state=42), interaction)
arena.add('RF_simple', RandomForestClassifier(n_estimators=100, max_depth=3, min_samples_leaf=15, random_state=42), simple)

# Gradient Boosting variations
arena.add('GBM_very_conservative', GradientBoostingClassifier(n_estimators=30, max_depth=2, learning_rate=0.1, min_samples_leaf=20, subsample=0.8, random_state=42), simple)
arena.add('GBM_conservative', GradientBoostingClassifier(n_estimators=50, max_depth=2, learning_rate=0.1, min_samples_leaf=15, subsample=0.8, random_state=42), moderate)
arena.add('GBM_moderate', GradientBoostingClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, min_samples_leaf=10, subsample=0.8, random_state=42), moderate)
arena.add('GBM_strong', GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=0.05, min_samples_leaf=8, subsample=0.8, random_state=42), moderate)
arena.add('GBM_interaction', GradientBoostingClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, min_samples_leaf=10, subsample=0.8, random_state=42), interaction)
arena.add('GBM_simple', GradientBoostingClassifier(n_estimators=50, max_depth=2, learning_rate=0.15, min_samples_leaf=20, random_state=42), simple)

# KNN
arena.add('KNN_5_moderate', KNeighborsClassifier(n_neighbors=5), moderate)
arena.add('KNN_9_moderate', KNeighborsClassifier(n_neighbors=9), moderate)

# --- Run the Arena ---
results = arena.run()

# --- Show Results ---
print("\n" + "=" * 80)
print("FULL LEADERBOARD")
print("=" * 80)
print(arena.leaderboard().to_string(index=False))

# --- Recommendation ---
print("\n" + "=" * 80)
rec = arena.recommend()
print(f"RECOMMENDATION: {rec['reasoning']}")
if rec['warnings']:
    for w in rec['warnings']:
        print(f"  ⚠️  {w}")
print("=" * 80)

# --- Generate submission from the winner ---
config = CompetitionConfig.from_yaml(os.path.join(os.path.dirname(__file__), "..", "competition_config.yaml"))
sub_dir = os.path.join(os.path.dirname(__file__), "..", "submissions")
submission_path = arena.generate_submission(test_df, test_df['PassengerId'], config, output_dir=sub_dir)
print(f"\nSubmission generated: {submission_path}")

# --- Save report ---
report = arena.summary_report()
report_path = os.path.join(os.path.dirname(__file__), "..", "content", "arena_results.md")
with open(report_path, 'w', encoding='utf-8') as f:
    f.write(report)
print(f"Report saved: {report_path}")
