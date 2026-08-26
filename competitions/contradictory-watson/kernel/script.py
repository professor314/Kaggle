"""
Contradictory, My Dear Watson - TF-IDF + Overlap Features
Natural Language Inference: Given a premise and hypothesis, classify as
entailment (0), neutral (1), or contradiction (2).

Approach: TF-IDF on premise and hypothesis separately, plus hand-crafted
overlap and length features that capture entailment/contradiction signals.
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from scipy.sparse import hstack, csr_matrix

# Load data from Kaggle paths
train = pd.read_csv('/kaggle/input/contradictory-my-dear-watson/train.csv')
test = pd.read_csv('/kaggle/input/contradictory-my-dear-watson/test.csv')

print(f"Train: {train.shape}, Test: {test.shape}")
print(f"Labels: {train['label'].value_counts().to_dict()}")
print(f"Languages: {train['language'].nunique()}")

# Fill NaN text
for col in ['premise', 'hypothesis']:
    train[col] = train[col].fillna('')
    test[col] = test[col].fillna('')

# Combined text (premise + hypothesis together)
train['combined'] = train['premise'] + ' [SEP] ' + train['hypothesis']
test['combined'] = test['premise'] + ' [SEP] ' + test['hypothesis']

# TF-IDF: premise separately, hypothesis separately, and combined
print("Vectorizing...")
tfidf_premise = TfidfVectorizer(max_features=15000, ngram_range=(1, 2), sublinear_tf=True)
tfidf_hyp = TfidfVectorizer(max_features=15000, ngram_range=(1, 2), sublinear_tf=True)
tfidf_combined = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), sublinear_tf=True, analyzer='char_wb')

# Fit on all text
all_premises = pd.concat([train['premise'], test['premise']])
all_hyps = pd.concat([train['hypothesis'], test['hypothesis']])
all_combined = pd.concat([train['combined'], test['combined']])

tfidf_premise.fit(all_premises)
tfidf_hyp.fit(all_hyps)
tfidf_combined.fit(all_combined)

X_train_p = tfidf_premise.transform(train['premise'])
X_train_h = tfidf_hyp.transform(train['hypothesis'])
X_train_c = tfidf_combined.transform(train['combined'])
X_test_p = tfidf_premise.transform(test['premise'])
X_test_h = tfidf_hyp.transform(test['hypothesis'])
X_test_c = tfidf_combined.transform(test['combined'])

# Overlap features
print("Computing overlap features...")

def compute_overlap(df):
    features = []
    for _, row in df.iterrows():
        p_words = set(str(row['premise']).lower().split())
        h_words = set(str(row['hypothesis']).lower().split())
        
        # Word overlap
        overlap = len(p_words & h_words)
        union = len(p_words | h_words)
        jaccard = overlap / max(union, 1)
        h_coverage = overlap / max(len(h_words), 1)
        p_coverage = overlap / max(len(p_words), 1)
        
        # Length features
        p_len = len(p_words)
        h_len = len(h_words)
        len_diff = p_len - h_len
        len_ratio = h_len / max(p_len, 1)
        
        # Negation signal (crude)
        neg_words = {'not', 'no', 'never', 'neither', 'nobody', 'nothing', 'nowhere', 'nor'}
        p_neg = len(p_words & neg_words)
        h_neg = len(h_words & neg_words)
        neg_diff = abs(p_neg - h_neg)  # different negation = likely contradiction
        
        features.append([jaccard, h_coverage, p_coverage, len_diff, len_ratio, overlap, p_len, h_len, neg_diff])
    
    return np.array(features, dtype=np.float32)

train_overlap = compute_overlap(train)
test_overlap = compute_overlap(test)

# Language as feature
lang_tfidf = TfidfVectorizer(analyzer='word')
lang_tfidf.fit(pd.concat([train['language'], test['language']]))
X_train_lang = lang_tfidf.transform(train['language'])
X_test_lang = lang_tfidf.transform(test['language'])

# Combine all features
X_train = hstack([X_train_p, X_train_h, X_train_c, csr_matrix(train_overlap), X_train_lang])
X_test = hstack([X_test_p, X_test_h, X_test_c, csr_matrix(test_overlap), X_test_lang])
y_train = train['label'].values

print(f"Total features: {X_train.shape[1]}")

# Train
print("Training LogisticRegression...")
model = LogisticRegression(C=0.5, max_iter=1000, solver='lbfgs', random_state=42)
model.fit(X_train, y_train)

train_acc = model.score(X_train, y_train)
print(f"Train accuracy: {train_acc:.4f}")

# Predict
predictions = model.predict(X_test)
print(f"Predictions: {pd.Series(predictions).value_counts().to_dict()}")

# Save submission
submission = pd.DataFrame({'id': test['id'], 'prediction': predictions})
submission.to_csv('/kaggle/working/submission.csv', index=False)
print(f"Submission saved: {submission.shape[0]} rows")
