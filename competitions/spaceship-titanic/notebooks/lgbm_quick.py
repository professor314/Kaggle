"""Spaceship Titanic — Quick LightGBM submission."""
import pandas as pd
import numpy as np
import lightgbm as lgb
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

DATA_DIR = Path(r"c:\Users\profe\OneDrive\Desktop\Sean Obsidian\Projects\Kaggle\competitions\spaceship-titanic\data")
SUB_DIR = Path(r"c:\Users\profe\OneDrive\Desktop\Sean Obsidian\Projects\Kaggle\competitions\spaceship-titanic\submissions")

train = pd.read_csv(DATA_DIR / "train.csv")
test = pd.read_csv(DATA_DIR / "test.csv")

# Feature engineering
def engineer_features(df):
    df = df.copy()
    # Parse Cabin
    cabin_split = df["Cabin"].str.split("/", expand=True)
    df["Deck"] = cabin_split[0] if cabin_split.shape[1] >= 1 else np.nan
    df["CabinNum"] = pd.to_numeric(cabin_split[1], errors="coerce") if cabin_split.shape[1] >= 2 else np.nan
    df["Side"] = cabin_split[2] if cabin_split.shape[1] >= 3 else np.nan

    # Parse PassengerId for group
    df["Group"] = df["PassengerId"].str.split("_").str[0]
    df["GroupSize"] = df.groupby("Group")["Group"].transform("count")
    df["IsAlone"] = (df["GroupSize"] == 1).astype(int)

    # Spending
    spend_cols = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
    df["TotalSpending"] = df[spend_cols].sum(axis=1)
    df["NoSpending"] = (df["TotalSpending"] == 0).astype(int)
    df["LogSpending"] = np.log1p(df["TotalSpending"])

    # Spending ratios
    for col in spend_cols:
        df[f"{col}_ratio"] = df[col] / (df["TotalSpending"] + 1)

    # Booleans
    df["CryoSleep"] = df["CryoSleep"].astype(float)
    df["VIP"] = df["VIP"].astype(float)

    # Age
    df["IsChild"] = (df["Age"] < 13).astype(int)

    # Encode categoricals
    for col in ["HomePlanet", "Destination", "Deck", "Side"]:
        df[col] = df[col].astype("category").cat.codes

    return df

train = engineer_features(train)
test = engineer_features(test)

feature_cols = [
    "CryoSleep", "Age", "VIP", "RoomService", "FoodCourt", "ShoppingMall",
    "Spa", "VRDeck", "TotalSpending", "NoSpending", "LogSpending",
    "HomePlanet", "Destination", "Deck", "CabinNum", "Side",
    "GroupSize", "IsAlone", "IsChild",
    "RoomService_ratio", "FoodCourt_ratio", "ShoppingMall_ratio", "Spa_ratio", "VRDeck_ratio",
]

X_train = train[feature_cols]
y = train["Transported"].astype(int)
X_test = test[feature_cols]

# 5-fold CV
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof = np.zeros(len(X_train))
test_preds = np.zeros(len(X_test))

params = {"objective": "binary", "metric": "binary_logloss", "learning_rate": 0.03,
          "num_leaves": 63, "max_depth": 7, "min_child_samples": 30,
          "feature_fraction": 0.75, "bagging_fraction": 0.75, "bagging_freq": 5,
          "reg_alpha": 0.1, "reg_lambda": 1.0, "verbosity": -1, "seed": 42}

for fold, (tr_idx, val_idx) in enumerate(cv.split(X_train, y)):
    dtrain = lgb.Dataset(X_train.iloc[tr_idx], label=y.iloc[tr_idx])
    dval = lgb.Dataset(X_train.iloc[val_idx], label=y.iloc[val_idx])
    model = lgb.train(params, dtrain, 2000, valid_sets=[dval],
                      callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)])
    oof[val_idx] = model.predict(X_train.iloc[val_idx])
    test_preds += model.predict(X_test) / 5
    acc = accuracy_score(y.iloc[val_idx], (oof[val_idx] > 0.5).astype(int))
    print(f"  Fold {fold+1}: acc={acc:.4f} iter={model.best_iteration}")

overall_acc = accuracy_score(y, (oof > 0.5).astype(int))
print(f"\nOOF Accuracy: {overall_acc:.4f}")

submission = pd.DataFrame({"PassengerId": test["PassengerId"], "Transported": test_preds > 0.5})
sub_file = SUB_DIR / "lgbm_quick.csv"
submission.to_csv(sub_file, index=False)
print(f"Saved: {sub_file}")
