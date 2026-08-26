"""Quick look at the Digit Recognizer data."""
import pandas as pd

train = pd.read_csv("competitions/digit-recognizer/data/train.csv")
test = pd.read_csv("competitions/digit-recognizer/data/test.csv")
print(f"Train: {train.shape}, Test: {test.shape}")
print(f"Target (label): values {sorted(train['label'].unique())}")
print(f"Features: {train.shape[1]-1} pixel columns (28x28 = 784 pixels)")
print(f"Pixel value range: [{train.iloc[:,1:].min().min()}, {train.iloc[:,1:].max().max()}]")
print(f"Class distribution:")
print(train["label"].value_counts().sort_index().to_string())
