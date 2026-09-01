"""Titanic - leak-free family/group survival features (Phase-3 revisit).

The experiments log shows the ceiling is ~0.773 and that MORE model complexity
HURTS (overfitting). The one documented lever never done correctly is
FAMILY/GROUP SURVIVAL: a prior attempt computed it WITH leakage (CV 0.988 →
LB 0.727, disaster). Done leak-free with out-of-fold encoding, this group signal
is what pushes top public solutions to ~0.78-0.80.

This script implements the well-known "woman-child group survival" heuristic
(a public Titanic technique) computed WITHOUT leakage:
  - Group by Surname + Ticket. Within each group, the survival of *other*
    members is a strong signal (families lived/died together).
  - We only use group info that does NOT include the row itself, and only from
    training labels via out-of-fold folds — no target leakage.

Baseline to beat: 0.773 LB.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

REPO = Path(__file__).resolve().parents[3]
DATA = REPO / "competitions" / "titanic" / "data"
SUB = REPO / "competitions" / "titanic" / "submissions"
SEED = 42


def base_features(df):
    df = df.copy()
    df["Title"] = df["Name"].str.extract(r",\s*([^\.]+)\.")
    df["Surname"] = df["Name"].str.split(",").str[0].str.strip()
    df["IsFemale"] = (df["Sex"] == "female").astype(int)
    df["IsChild"] = (df["Age"].fillna(df["Age"].median()) < 16).astype(int)
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["Fare"] = df["Fare"].fillna(df["Fare"].median())
    return df


def main():
    train = base_features(pd.read_csv(DATA / "train.csv"))
    test = base_features(pd.read_csv(DATA / "test.csv"))

    # Public "woman-child-group" heuristic:
    # A group = same Surname (and shares a Ticket or family). Women & children in
    # a group tend to share a fate. We build a GROUP KEY and, out-of-fold, compute
    # the mean survival of the group's WOMEN+CHILDREN excluding the current row.
    for df in (train, test):
        df["GroupKey"] = df["Surname"] + "_" + df["Pclass"].astype(str)
        df["is_wc"] = ((df["IsFemale"] == 1) | (df["IsChild"] == 1)).astype(int)

    y = train["Survived"].values
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    # Out-of-fold group-survival for train (no leakage: fold-out rows never see
    # their own label or their own row's contribution).
    train["grp_surv"] = np.nan
    global_wc_rate = train.loc[train["is_wc"] == 1, "Survived"].mean()

    for tri, vai in cv.split(train, y):
        tr = train.iloc[tri]
        # group survival among women+children in the training fold only
        grp = (tr[tr["is_wc"] == 1].groupby("GroupKey")["Survived"].agg(["sum", "count"]))
        rate = ((grp["sum"]) / (grp["count"])).to_dict()
        cnt = grp["count"].to_dict()
        def lookup(k):
            if k in rate and cnt.get(k, 0) >= 1:
                return rate[k]
            return global_wc_rate
        train.loc[train.index[vai], "grp_surv"] = train.iloc[vai]["GroupKey"].map(lookup).values

    # For test: use the FULL training set's group survival.
    grp_full = train[train["is_wc"] == 1].groupby("GroupKey")["Survived"].agg(["sum", "count"])
    rate_full = (grp_full["sum"] / grp_full["count"]).to_dict()
    cnt_full = grp_full["count"].to_dict()
    test["grp_surv"] = test["GroupKey"].map(
        lambda k: rate_full[k] if (k in rate_full and cnt_full.get(k, 0) >= 1) else global_wc_rate)

    # ---- The heuristic prediction (public "gender + group" rule) ----
    # Default: females survive, males die (the strong gender prior).
    # Override with the group signal for women/children whose group clearly died
    # or men/boys whose group clearly survived.
    def predict(df, grp_col="grp_surv"):
        pred = df["IsFemale"].values.copy()  # 1 if female else 0
        g = df[grp_col].values
        wc = df["is_wc"].values
        # Women/children in a group that mostly DIED -> predict die
        die_override = (wc == 1) & (g <= 0.1)
        # Boys/men in a group that mostly SURVIVED -> predict survive
        live_override = (df["IsFemale"].values == 0) & (g >= 0.9)
        pred = pred.copy()
        pred[die_override] = 0
        pred[live_override] = 1
        return pred.astype(int)

    # Evaluate the heuristic OOF on train.
    oof_pred = predict(train)
    acc = (oof_pred == y).mean()
    print(f"OOF accuracy of gender+group heuristic: {acc:.4f}")
    base_acc = (train["IsFemale"].values == y).mean()
    print(f"  (gender-only baseline OOF: {base_acc:.4f})")

    test_pred = predict(test)
    sub = pd.DataFrame({"PassengerId": test["PassengerId"], "Survived": test_pred})
    SUB.mkdir(parents=True, exist_ok=True)
    out = SUB / "family_groups_oof.csv"
    sub.to_csv(out, index=False)
    print(f"\nSaved: {out}  survived rate {sub['Survived'].mean():.3f}")
    print("=== DONE ===")


if __name__ == "__main__":
    main()
