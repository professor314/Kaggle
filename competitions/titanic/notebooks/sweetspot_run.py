"""Titanic sweet-spot submission — same features as best LB, even more conservative model."""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import pandas as pd
import numpy as np
from kaggle_ml_toolkit import CompetitionConfig, SubmissionGenerator
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score

train_df = pd.read_csv(os.path.join(os.path.dirname(__file__), "..", "data", "train.csv"))
test_df = pd.read_csv(os.path.join(os.path.dirname(__file__), "..", "data", "test.csv"))

def prepare(df):
    df = df.copy()
    df['IsFemale'] = (df['Sex'] == 'female').astype(int)
    df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
    df['Title'] = df['Title'].replace({'Mlle':'Miss','Ms':'Miss','Mme':'Mrs','Lady':'Rare','Countess':'Rare','Capt':'Rare','Col':'Rare','Don':'Rare','Dr':'Rare','Major':'Rare','Rev':'Rare','Sir':'Rare','Jonkheer':'Rare','Dona':'Rare'})
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
    title_map = {'Mr': 0, 'Miss': 1, 'Mrs': 2, 'Master': 3, 'Rare': 4}
    df['TitleCode'] = df['Title'].map(title_map).fillna(4).astype(int)
    return df

train_df = prepare(train_df)
test_df = prepare(test_df)

for title in train_df['Title'].unique():
    med = train_df.loc[train_df['Title'] == title, 'Age'].median()
    train_df.loc[(train_df['Title'] == title) & train_df['Age'].isna(), 'Age'] = med
    test_df.loc[(test_df['Title'] == title) & test_df['Age'].isna(), 'Age'] = med
train_df['Age'] = train_df['Age'].fillna(28)
test_df['Age'] = test_df['Age'].fillna(28)
test_df['Fare'] = test_df['Fare'].fillna(test_df['Fare'].median())

train_df['IsChild'] = (train_df['Age'] < 13).astype(int)
test_df['IsChild'] = (test_df['Age'] < 13).astype(int)

features = ['IsFemale', 'Pclass', 'Age', 'Fare', 'FamilySize', 'IsAlone', 'IsChild', 'TitleCode']
X = train_df[features]
y = train_df['Survived']
X_test = test_df[features]

# Even more conservative than our 0.773 submission
model = GradientBoostingClassifier(
    n_estimators=40, max_depth=2, learning_rate=0.12,
    min_samples_leaf=15, subsample=0.75, random_state=42
)

cv = cross_val_score(model, X, y, cv=5, scoring='accuracy')
print(f'CV Accuracy (5-fold): {cv.mean():.4f} +/- {cv.std():.4f}')

model.fit(X, y)
preds = model.predict(X_test)

config = CompetitionConfig.from_yaml(os.path.join(os.path.dirname(__file__), "..", "competition_config.yaml"))
gen = SubmissionGenerator()
path = gen.generate(preds, test_df['PassengerId'], config, model_name='gbm_sweetspot', output_dir=os.path.join(os.path.dirname(__file__), "..", "submissions"))
print(f'File: {os.path.basename(path)}')
print(f'Survived: {sum(preds)}, Died: {len(preds)-sum(preds)}')
