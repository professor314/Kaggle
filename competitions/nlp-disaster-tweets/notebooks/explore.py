"""Quick look at NLP Disaster Tweets data."""
import pandas as pd

data_dir = r'c:\Users\profe\OneDrive\Desktop\Sean Obsidian\Kaggle\competitions\nlp-disaster-tweets\data'
train = pd.read_csv(f'{data_dir}/train.csv')
test = pd.read_csv(f'{data_dir}/test.csv')

print(f'Train: {train.shape}, Test: {test.shape}')
print(f'Columns: {list(train.columns)}')
print(f'Target distribution: {train["target"].value_counts().to_dict()}')

print(f'\nSample tweets (disaster=1):')
for t in train[train['target'] == 1]['text'].head(3):
    print(f'  {t[:120]}')

print(f'\nSample tweets (not disaster=0):')
for t in train[train['target'] == 0]['text'].head(3):
    print(f'  {t[:120]}')

print(f'\nKeyword: {train["keyword"].nunique()} unique, {train["keyword"].isna().sum()} missing')
print(f'Location: {train["location"].nunique()} unique, {train["location"].isna().sum()} missing')
print(f'\nText length stats:')
print(train['text'].str.len().describe())
