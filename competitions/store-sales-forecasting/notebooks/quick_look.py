"""Quick look at the Store Sales forecasting data."""
import pandas as pd

data_dir = r'c:\Users\profe\OneDrive\Desktop\Sean Obsidian\Kaggle\competitions\store-sales-forecasting\data'

# Train overview
train = pd.read_csv(f'{data_dir}/train.csv', nrows=20)
print('=== TRAIN ===')
print('Columns:', list(train.columns))
print(train.head(5).to_string())

# Full size
train_dates = pd.read_csv(f'{data_dir}/train.csv', usecols=['date'])
print(f'\nTotal train rows: {len(train_dates):,}')
print(f'Date range: {train_dates["date"].min()} to {train_dates["date"].max()}')

# Test
test = pd.read_csv(f'{data_dir}/test.csv', nrows=5)
print('\n=== TEST ===')
print('Columns:', list(test.columns))
print(test.head(3).to_string())
print(f'Test rows: {len(pd.read_csv(f"{data_dir}/test.csv")):,}')

# Sample submission
sub = pd.read_csv(f'{data_dir}/sample_submission.csv', nrows=5)
print('\n=== SUBMISSION ===')
print(sub.head())

# Stores
stores = pd.read_csv(f'{data_dir}/stores.csv')
print(f'\n=== STORES ({len(stores)} stores) ===')
print(stores.to_string())

# Oil
oil = pd.read_csv(f'{data_dir}/oil.csv')
print(f'\n=== OIL ({len(oil)} rows) ===')
print(oil.head())
print(f'Date range: {oil["date"].min()} to {oil["date"].max()}')

# Holidays
holidays = pd.read_csv(f'{data_dir}/holidays_events.csv')
print(f'\n=== HOLIDAYS ({len(holidays)} events) ===')
print(holidays.head(10).to_string())

# Transactions
trans = pd.read_csv(f'{data_dir}/transactions.csv')
print(f'\n=== TRANSACTIONS ({len(trans):,} rows) ===')
print(trans.head())
