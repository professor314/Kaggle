"""
Contradictory My Dear Watson - Natural Language Inference
Competition: contradictory-my-dear-watson
Task: Multi-class classification (entailment/neutral/contradiction)
"""
import os
import sys
import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
SUB_DIR = os.path.join(BASE_DIR, "submissions")
os.makedirs(SUB_DIR, exist_ok=True)

COMPETITION_SLUG = "contradictory-my-dear-watson"

print("=" * 60)
print("COMPETITION 3: Contradictory My Dear Watson (NLI)")
print("=" * 60)

# Check if data exists
train_path = os.path.join(DATA_DIR, "train.csv")
test_path = os.path.join(DATA_DIR, "test.csv")

if not os.path.exists(train_path):
    print(f"\nSKIPPED: Need to join competition at kaggle.com/competitions/{COMPETITION_SLUG}/rules")
    print("Data files not found - download returned 403 (rules not accepted).")
    sys.exit(0)

print("\n[1/5] Loading data...")
train = pd.read_csv(train_path)
test = pd.read_csv(test_path)
print(f"  Train shape: {train.shape}")
print(f"  Test shape: {test.shape}")
print(f"  Columns: {train.columns.tolist()}")
print(f"  Label distribution:\n{train['label'].value_counts()}")
print(f"  Languages: {train['language'].nunique()} unique")

# Feature engineering
print("\n[2/5] Feature engineering...")

def engineer_nli_features(df):
    """Create comparison features for premise-hypothesis pairs."""
    features = pd.DataFrame()
    features['premise_length'] = df['premise'].str.len()
    features['hypothesis_length'] = df['hypothesis'].str.len()
    features['length_diff'] = features['premise_length'] - features['hypothesis_length']
    features['length_ratio'] = features['premise_length'] / features['hypothesis_length'].clip(lower=1)
    
    # Word overlap features
    def word_overlap_ratio(row):
        premise_words = set(str(row['premise']).lower().split())
        hypothesis_words = set(str(row['hypothesis']).lower().split())
        if not hypothesis_words:
            return 0.0
        return len(premise_words & hypothesis_words) / len(hypothesis_words)
    
    def jaccard_similarity(row):
        premise_words = set(str(row['premise']).lower().split())
        hypothesis_words = set(str(row['hypothesis']).lower().split())
        union = premise_words | hypothesis_words
        if not union:
            return 0.0
        return len(premise_words & hypothesis_words) / len(union)
    
    features['word_overlap_ratio'] = df.apply(word_overlap_ratio, axis=1)
    features['jaccard_similarity'] = df.apply(jaccard_similarity, axis=1)
    
    # Count features
    features['premise_word_count'] = df['premise'].str.split().str.len()
    features['hypothesis_word_count'] = df['hypothesis'].str.split().str.len()
    features['word_count_diff'] = features['premise_word_count'] - features['hypothesis_word_count']
    
    return features

train_feats = engineer_nli_features(train)
test_feats = engineer_nli_features(test)
print(f"  Engineered {train_feats.shape[1]} features")

# TF-IDF Vectorization on combined text
print("\n[3/5] TF-IDF vectorization on premise + hypothesis...")

# Combine premise and hypothesis with separator
train['combined_text'] = train['premise'].fillna('') + ' [SEP] ' + train['hypothesis'].fillna('')
test['combined_text'] = test['premise'].fillna('') + ' [SEP] ' + test['hypothesis'].fillna('')

tfidf = TfidfVectorizer(
    max_features=10000, 
    ngram_range=(1, 2), 
    min_df=2,
    sublinear_tf=True,
)
X_tfidf_train = tfidf.fit_transform(train['combined_text'])
X_tfidf_test = tfidf.transform(test['combined_text'])

# Combine TF-IDF with engineered features
from scipy.sparse import hstack
X_train = hstack([X_tfidf_train, train_feats.values])
X_test = hstack([X_tfidf_test, test_feats.values])
y_train = train['label'].values

print(f"  TF-IDF features: {X_tfidf_train.shape[1]}")
print(f"  Total features: {X_train.shape[1]}")

# Model comparison
print("\n[4/5] Training and comparing models...")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

models = {
    'LogisticRegression': LogisticRegression(max_iter=1000, C=1.0, random_state=42),
    'LinearSVC': LinearSVC(max_iter=2000, C=1.0, random_state=42),
    'RandomForest': RandomForestClassifier(n_estimators=200, max_depth=None, random_state=42, n_jobs=-1),
}

results = {}
for name, model in models.items():
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy', n_jobs=-1)
    results[name] = {'mean': scores.mean(), 'std': scores.std()}
    print(f"  {name}: Accuracy = {scores.mean():.4f} (+/- {scores.std():.4f})")

# Pick best model
best_name = max(results, key=lambda k: results[k]['mean'])
best_model = models[best_name]
print(f"\n  Best model: {best_name} (Accuracy = {results[best_name]['mean']:.4f})")

# Train on full data and predict
print("\n[5/5] Generating submission...")
best_model.fit(X_train, y_train)
predictions = best_model.predict(X_test)

submission = pd.DataFrame({
    'id': test['id'],
    'prediction': predictions.astype(int)
})
submission_path = os.path.join(SUB_DIR, "submission.csv")
submission.to_csv(submission_path, index=False)
print(f"  Submission saved to: {submission_path}")
print(f"  Shape: {submission.shape}")
print(f"  Prediction distribution:\n{submission['prediction'].value_counts()}")

# Submit
print("\n  Submitting to Kaggle...")
submit_msg = f"Auto-pipeline: {best_name}, Acc={results[best_name]['mean']:.4f}"
exit_code = os.system(
    f'kaggle competitions submit -c {COMPETITION_SLUG} '
    f'-f "{submission_path}" -m "{submit_msg}"'
)
if exit_code == 0:
    print("  ✓ Submission successful!")
else:
    print("  ✗ Submission failed (may need to accept competition rules)")

print("\n" + "=" * 60)
print("COMPETITION 3 COMPLETE")
print("=" * 60)
