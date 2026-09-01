"""Spaceship Titanic - Boosting ensemble (LightGBM + XGBoost + CatBoost + HistGB).

Reuses the proven 50 no-leakage features from research_v2_arena.py, then builds
an OOF-stacked ensemble of gradient-boosting libraries with a logistic-regression
meta-learner. Compares against the best individual base model.

Prior best: 0.80243 LB (HistGradientBoosting, 50 features).
Target: 0.81+ via library diversity + stacking.

Run with the DL environment:
    .venv-dl\\Scripts\\python.exe competitions\\spaceship-titanic\\notebooks\\boosting_ensemble.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_score

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

REPO = Path(__file__).resolve().parents[3]
COMP = REPO / "competitions" / "spaceship-titanic"
DATA = COMP / "data"
SUB = COMP / "submissions"
SEED = 42
N_SPLITS = 5


def engineer(train_df, test_df):
    """Recreate the 50 no-leakage features from research_v2_arena.py."""
    train_df = train_df.copy()
    test_df = test_df.copy()
    train_df["_is_train"] = True
    test_df["_is_train"] = False
    test_df["Transported"] = np.nan
    c = pd.concat([train_df, test_df], ignore_index=True)

    c["Group"] = c["PassengerId"].str.split("_").str[0]
    c["PersonInGroup"] = c["PassengerId"].str.split("_").str[1].astype(int)
    c["GroupSize"] = c.groupby("Group")["PassengerId"].transform("count")
    c["IsAlone"] = (c["GroupSize"] == 1).astype(int)

    c["CabinDeck"] = c["Cabin"].str.split("/").str[0]
    c["CabinNum"] = c["Cabin"].str.split("/").str[1]
    c["CabinSide"] = c["Cabin"].str.split("/").str[2]
    deck_map = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5, "G": 6, "T": 7}
    c["DeckNum"] = c["CabinDeck"].map(deck_map).fillna(-1).astype(int)
    c["IsPort"] = (c["CabinSide"] == "P").astype(int)
    c["IsStar"] = (c["CabinSide"] == "S").astype(int)
    c["CabinNumeric"] = pd.to_numeric(c["CabinNum"], errors="coerce")
    c["CabinRegion"] = pd.cut(c["CabinNumeric"], bins=[0, 300, 600, 900, 1200, 1500, 2000],
                              labels=[0, 1, 2, 3, 4, 5], include_lowest=True).astype(float).fillna(-1).astype(int)

    spend = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
    for col in spend:
        c[col] = c[col].fillna(0)
    c["TotalSpending"] = c[spend].sum(axis=1)
    c["LogSpending"] = np.log1p(c["TotalSpending"])
    c["NoSpending"] = (c["TotalSpending"] == 0).astype(int)
    for col in spend:
        c[f"SpentOn{col}"] = (c[col] > 0).astype(int)
        c[f"Log{col}"] = np.log1p(c[col])
    c["NumServicesUsed"] = sum(c[f"SpentOn{col}"] for col in spend)
    tp1 = c["TotalSpending"] + 1
    for col in spend:
        c[f"{col}Ratio"] = c[col] / tp1

    c["CryoSleep"] = c["CryoSleep"].fillna(False).astype(int)
    c["VIP"] = c["VIP"].fillna(False).astype(int)
    c["Age"] = c["Age"].fillna(c["Age"].median())
    c["IsChild"] = (c["Age"] < 12).astype(int)
    c["IsElder"] = (c["Age"] > 60).astype(int)
    c["AgeGroup"] = pd.cut(c["Age"], bins=[0, 5, 12, 18, 25, 40, 60, 100],
                           labels=[0, 1, 2, 3, 4, 5, 6], include_lowest=True).astype(float).fillna(3).astype(int)

    c["HomePlanet"] = c["HomePlanet"].fillna("Unknown")
    c["FromEarth"] = (c["HomePlanet"] == "Earth").astype(int)
    c["FromEuropa"] = (c["HomePlanet"] == "Europa").astype(int)
    c["FromMars"] = (c["HomePlanet"] == "Mars").astype(int)
    c["Destination"] = c["Destination"].fillna("Unknown")
    c["ToTRAPPIST"] = (c["Destination"] == "TRAPPIST-1e").astype(int)
    c["To55Cancri"] = (c["Destination"] == "55 Cancri e").astype(int)
    c["ToPSO"] = (c["Destination"] == "PSO J318.5-22").astype(int)
    c["EarthCryo"] = c["FromEarth"] * c["CryoSleep"]
    c["EuropaCryo"] = c["FromEuropa"] * c["CryoSleep"]
    c["MarsCryo"] = c["FromMars"] * c["CryoSleep"]
    c["EarthVIP"] = c["FromEarth"] * c["VIP"]
    c["EuropaVIP"] = c["FromEuropa"] * c["VIP"]
    c["AgeCryo"] = c["Age"] * c["CryoSleep"]

    feats = [
        "CryoSleep", "Age", "IsChild", "IsElder", "AgeGroup", "VIP",
        "TotalSpending", "LogSpending", "NoSpending", "NumServicesUsed",
        "RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck",
        "LogRoomService", "LogFoodCourt", "LogShoppingMall", "LogSpa", "LogVRDeck",
        "SpentOnRoomService", "SpentOnFoodCourt", "SpentOnShoppingMall", "SpentOnSpa", "SpentOnVRDeck",
        "RoomServiceRatio", "FoodCourtRatio", "ShoppingMallRatio", "SpaRatio", "VRDeckRatio",
        "DeckNum", "IsPort", "IsStar", "CabinRegion", "CabinNumeric",
        "GroupSize", "IsAlone", "PersonInGroup",
        "FromEarth", "FromEuropa", "FromMars", "ToTRAPPIST", "To55Cancri", "ToPSO",
        "EarthCryo", "EuropaCryo", "MarsCryo", "EarthVIP", "EuropaVIP", "AgeCryo",
    ]
    for col in feats:
        c[col] = c[col].fillna(0)
    tr = c[c["_is_train"]][feats].reset_index(drop=True)
    te = c[~c["_is_train"]][feats].reset_index(drop=True)
    return tr, te, feats


def base_models():
    return {
        "lgb": lgb.LGBMClassifier(n_estimators=400, num_leaves=31, learning_rate=0.03,
                                  subsample=0.8, colsample_bytree=0.8, min_child_samples=20,
                                  random_state=SEED, verbose=-1),
        "xgb": xgb.XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.03,
                                 subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
                                 random_state=SEED, verbosity=0),
        "cat": CatBoostClassifier(iterations=500, depth=5, learning_rate=0.03,
                                  random_seed=SEED, verbose=0),
        "hgb": HistGradientBoostingClassifier(max_iter=300, max_depth=5, learning_rate=0.05,
                                              min_samples_leaf=12, random_state=SEED),
    }


def main():
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    test_ids = test["PassengerId"]
    y = train["Transported"].astype(int).values

    Xtr, Xte, feats = engineer(train, test)
    print(f"Features: {len(feats)} | Train {Xtr.shape} | Test {Xte.shape}")

    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

    # Individual base-model CV scores.
    print("\nBase model CV accuracy:")
    for name, mdl in base_models().items():
        scores = cross_val_score(mdl, Xtr, y, cv=cv, scoring="accuracy")
        print(f"  {name}: {scores.mean():.4f} +/- {scores.std():.4f}")

    # OOF stacking.
    names = list(base_models().keys())
    oof = np.zeros((len(Xtr), len(names)))
    test_meta = np.zeros((len(Xte), len(names)))

    for j, name in enumerate(names):
        fold_test = np.zeros((len(Xte), N_SPLITS))
        for k, (tri, vai) in enumerate(cv.split(Xtr, y)):
            mdl = base_models()[name]
            mdl.fit(Xtr.iloc[tri], y[tri])
            oof[vai, j] = mdl.predict_proba(Xtr.iloc[vai])[:, 1]
            fold_test[:, k] = mdl.predict_proba(Xte)[:, 1]
        test_meta[:, j] = fold_test.mean(axis=1)

    meta = LogisticRegression(C=1.0, max_iter=1000, random_state=SEED)
    meta_scores = cross_val_score(meta, oof, y, cv=cv, scoring="accuracy")
    print(f"\nStacked ensemble OOF CV accuracy: {meta_scores.mean():.4f} +/- {meta_scores.std():.4f}")

    # Simple average blend for comparison.
    blend_oof = oof.mean(axis=1)
    blend_acc = accuracy_score(y, (blend_oof >= 0.5).astype(int))
    print(f"Simple-average blend OOF accuracy: {blend_acc:.4f}")

    # The boosting libraries are highly correlated, so stacking/averaging does not
    # beat the best individual model here. Pick whichever CV score is highest.
    lgb_oof = oof[:, names.index("lgb")]
    lgb_acc = accuracy_score(y, (lgb_oof >= 0.5).astype(int))
    candidates = {
        "lgb_only": (lgb_acc, (test_meta[:, names.index("lgb")] >= 0.5)),
        "stack": (meta_scores.mean(), None),
        "blend": (blend_acc, (test_meta.mean(axis=1) >= 0.5)),
    }
    best_name = max(candidates, key=lambda k: candidates[k][0])
    print(f"\nBest by CV: {best_name} ({candidates[best_name][0]:.4f})")

    if best_name == "stack":
        meta.fit(oof, y)
        test_pred = meta.predict(test_meta).astype(bool)
    else:
        test_pred = candidates[best_name][1].astype(bool)

    sub = pd.DataFrame({"PassengerId": test_ids, "Transported": test_pred})
    SUB.mkdir(parents=True, exist_ok=True)
    out = SUB / "boosting_best.csv"
    sub.to_csv(out, index=False)
    print(f"\nSaved: {out}  shape {sub.shape}")
    print(f"Pred balance: {sub['Transported'].value_counts().to_dict()}")
    print(f"\n=== DONE === best {best_name} CV {candidates[best_name][0]:.4f} (prior best LB 0.8024)")


if __name__ == "__main__":
    main()
