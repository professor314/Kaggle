"""
LLM Classification Finetuning - TF-IDF Baseline
A simple baseline using TF-IDF + LogisticRegression for text classification.
This serves as a non-LLM baseline to establish a floor score.
"""

import pandas as pd
import numpy as np
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

# Find competition data
input_dir = '/kaggle/input/llm-classification-finetuning/'
print(f"Input directory contents:")
for f in os.listdir(input_dir):
    print(f"  {f}")

# Load data - try common file names
train = None
test = None

# Try loading train data
for fname in ['train.csv', 'training.csv', 'train_data.csv']:
    fpath = os.path.join(input_dir, fname)
    if os.path.exists(fpath):
        train = pd.read_csv(fpath)
        print(f"Loaded train from: {fname} - shape: {train.shape}")
        break

# Try loading test data
for fname in ['test.csv', 'testing.csv', 'test_data.csv']:
    fpath = os.path.join(input_dir, fname)
    if os.path.exists(fpath):
        test = pd.read_csv(fpath)
        print(f"Loaded test from: {fname} - shape: {test.shape}")
        break

if train is None or test is None:
    # If standard names don't work, load all CSVs and guess
    csv_files = [f for f in os.listdir(input_dir) if f.endswith('.csv')]
    print(f"CSV files found: {csv_files}")
    if len(csv_files) >= 2:
        # Load all and pick largest as train, next as test
        dfs = {}
        for f in csv_files:
            dfs[f] = pd.read_csv(os.path.join(input_dir, f))
            print(f"  {f}: shape={dfs[f].shape}, columns={dfs[f].columns.tolist()}")
        
        sorted_files = sorted(dfs.keys(), key=lambda x: len(dfs[x]), reverse=True)
        if train is None:
            train = dfs[sorted_files[0]]
            print(f"Using {sorted_files[0]} as train")
        if test is None:
            test = dfs[sorted_files[1]]
            print(f"Using {sorted_files[1]} as test")
    elif len(csv_files) == 1:
        # Only one file - might need to split
        train = pd.read_csv(os.path.join(input_dir, csv_files[0]))
        print(f"Only one CSV found: {csv_files[0]}, shape: {train.shape}")

print(f"\nTrain columns: {train.columns.tolist()}")
print(f"Train dtypes:\n{train.dtypes}")
print(f"\nTest columns: {test.columns.tolist()}")
print(f"\nTrain head:\n{train.head()}")

# Identify text columns and target column
# Strategy: find string columns for features, find the likely target
text_cols = train.select_dtypes(include=['object']).columns.tolist()
numeric_cols = train.select_dtypes(include=['number']).columns.tolist()

print(f"\nText columns: {text_cols}")
print(f"Numeric columns: {numeric_cols}")

# Heuristic: the target is likely a column in train but not in test
# or a column named 'label', 'target', 'class', etc.
target_col = None
id_col = None

# Check for common target names
for col in ['label', 'target', 'class', 'category', 'sentiment', 'rating']:
    if col in train.columns:
        target_col = col
        break

# Check for common ID names
for col in ['id', 'ID', 'Id', 'index']:
    if col in test.columns:
        id_col = col
        break

# If no obvious target, find column in train but not in test
if target_col is None:
    train_only_cols = set(train.columns) - set(test.columns)
    print(f"Columns in train but not test: {train_only_cols}")
    if len(train_only_cols) == 1:
        target_col = list(train_only_cols)[0]
    elif len(train_only_cols) > 1:
        # Pick the one that looks most like a target
        for col in train_only_cols:
            if train[col].nunique() < 50:  # Likely classification target
                target_col = col
                break
        if target_col is None:
            target_col = list(train_only_cols)[0]

print(f"\nIdentified target column: {target_col}")
print(f"Identified ID column: {id_col}")

if target_col:
    print(f"Target distribution:\n{train[target_col].value_counts()}")

# Identify text features (exclude target and ID)
feature_text_cols = [c for c in text_cols if c != target_col and c != id_col]
print(f"Feature text columns: {feature_text_cols}")

# If no text columns found in features, use all non-target, non-id columns
if not feature_text_cols:
    feature_text_cols = [c for c in train.columns if c != target_col and c != id_col]

# Combine text features into single text field
def combine_text(df, cols):
    """Combine multiple columns into single text for TF-IDF."""
    combined = df[cols[0]].astype(str).copy()
    for col in cols[1:]:
        combined = combined + ' ' + df[col].astype(str)
    return combined

train_text = combine_text(train, feature_text_cols)
test_text = combine_text(test, feature_text_cols)

# Encode target
le = LabelEncoder()
y_train = le.fit_transform(train[target_col].astype(str))
n_classes = len(le.classes_)
print(f"\nNumber of classes: {n_classes}")
print(f"Classes: {le.classes_}")

# TF-IDF vectorization
print("\nFitting TF-IDF...")
tfidf = TfidfVectorizer(
    max_features=20000,
    ngram_range=(1, 2),
    sublinear_tf=True,
    strip_accents='unicode'
)
X_train = tfidf.fit_transform(train_text)
X_test = tfidf.transform(test_text)

print(f"TF-IDF feature shape: {X_train.shape}")

# Train model
print("Training Logistic Regression...")
model = LogisticRegression(
    C=1.0,
    max_iter=1000,
    solver='lbfgs',
    random_state=42
)
model.fit(X_train, y_train)

train_acc = model.score(X_train, y_train)
print(f"Training accuracy: {train_acc:.4f}")

# Predict
predictions = model.predict(X_test)
predicted_labels = le.inverse_transform(predictions)

print(f"Prediction distribution:\n{pd.Series(predicted_labels).value_counts()}")

# Generate submission
submission = pd.DataFrame()
if id_col and id_col in test.columns:
    submission['id'] = test[id_col]
else:
    submission['id'] = range(len(test))

submission[target_col] = predicted_labels

# Check if submission format expects different column names
# Look for sample_submission.csv
sample_sub_path = os.path.join(input_dir, 'sample_submission.csv')
if os.path.exists(sample_sub_path):
    sample_sub = pd.read_csv(sample_sub_path)
    print(f"\nSample submission columns: {sample_sub.columns.tolist()}")
    print(f"Sample submission head:\n{sample_sub.head()}")
    
    # Match submission format to sample
    submission = pd.DataFrame()
    for col in sample_sub.columns:
        if col in test.columns:
            submission[col] = test[col]
        elif col == target_col or col in [target_col]:
            submission[col] = predicted_labels
        elif col.lower() in ['id', 'index']:
            if id_col and id_col in test.columns:
                submission[col] = test[id_col]
            else:
                submission[col] = range(len(test))
        else:
            # This might be the prediction column
            submission[col] = predicted_labels
    
    print(f"\nFinal submission columns: {submission.columns.tolist()}")

submission.to_csv('/kaggle/working/submission.csv', index=False)
print(f"\nSubmission saved: {submission.shape[0]} rows")
print(submission.head())
