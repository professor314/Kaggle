"""
NLP Getting Started - Disaster Tweets Binary Classification
Competition: nlp-getting-started
Task: Predict if a tweet is about a real disaster (1) or not (0)
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
from sklearn.metrics import f1_score

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
SUB_DIR = os.path.join(BASE_DIR, "submissions")
os.makedirs(SUB_DIR, exist_ok=True)

COMPETITION_SLUG = "nlp-getting-started"

print("=" * 60)
print("COMPETITION 1: NLP Getting Started (Disaster Tweets)")
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
print(f"  Target distribution:\n{train['target'].value_counts()}")

# Feature engineering
print("\n[2/5] Feature engineering...")

def engineer_text_features(df):
    """Create text-based features from the 'text' column."""
    features = pd.DataFrame()
    features['text_length'] = df['text'].str.len()
    features['word_count'] = df['text'].str.split().str.len()
    features['has_url'] = df['text'].str.contains(r'http[s]?://', regex=True).astype(int)
    features['has_hashtag'] = df['text'].str.contains(r'#\w+', regex=True).astype(int)
    features['has_mention'] = df['text'].str.contains(r'@\w+', regex=True).astype(int)
    features['exclamation_count'] = df['text'].str.count(r'!')
    features['uppercase_ratio'] = df['text'].apply(
        lambda x: sum(1 for c in x if c.isupper()) / max(len(x), 1)
    )
    # Keyword presence
    features['has_keyword'] = df['keyword'].notna().astype(int)
    return features

train_feats = engineer_text_features(train)
test_feats = engineer_text_features(test)
print(f"  Engineered {train_feats.shape[1]} features")

# TF-IDF Vectorization
print("\n[3/5] TF-IDF vectorization...")
# Clean text for TF-IDF
def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+', '', text)  # remove URLs
    text = re.sub(r'@\w+', '', text)     # remove mentions
    text = re.sub(r'#(\w+)', r'\1', text) # remove # but keep word
    text = re.sub(r'[^a-z\s]', '', text)  # remove special chars
    return text.strip()

train['clean_text'] = train['text'].apply(clean_text)
test['clean_text'] = test['text'].apply(clean_text)

tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=2)
X_tfidf_train = tfidf.fit_transform(train['clean_text'])
X_tfidf_test = tfidf.transform(test['clean_text'])

# Combine TF-IDF with engineered features
from scipy.sparse import hstack
X_train = hstack([X_tfidf_train, train_feats.values])
X_test = hstack([X_tfidf_test, test_feats.values])
y_train = train['target'].values

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
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='f1', n_jobs=-1)
    results[name] = {'mean': scores.mean(), 'std': scores.std()}
    print(f"  {name}: F1 = {scores.mean():.4f} (+/- {scores.std():.4f})")

# Pick best model
best_name = max(results, key=lambda k: results[k]['mean'])
best_model = models[best_name]
print(f"\n  Best model: {best_name} (F1 = {results[best_name]['mean']:.4f})")

# Train on full data and predict
print("\n[5/5] Generating submission...")
best_model.fit(X_train, y_train)

# For LinearSVC, use decision_function; others use predict
predictions = best_model.predict(X_test)

submission = pd.DataFrame({
    'id': test['id'],
    'target': predictions.astype(int)
})
submission_path = os.path.join(SUB_DIR, "submission.csv")
submission.to_csv(submission_path, index=False)
print(f"  Submission saved to: {submission_path}")
print(f"  Shape: {submission.shape}")
print(f"  Prediction distribution:\n{submission['target'].value_counts()}")

# Submit
print("\n  Submitting to Kaggle...")
submit_msg = f"Auto-pipeline: {best_name}, F1={results[best_name]['mean']:.4f}"
exit_code = os.system(
    f'kaggle competitions submit -c {COMPETITION_SLUG} '
    f'-f "{submission_path}" -m "{submit_msg}"'
)
if exit_code == 0:
    print("  ✓ Submission successful!")
else:
    print("  ✗ Submission failed (may need to accept competition rules)")

print("\n" + "=" * 60)
print("COMPETITION 1 COMPLETE")
print("=" * 60)
