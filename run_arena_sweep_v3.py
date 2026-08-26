"""
ArenaGenerator V3 Sweep — All 3 Competitions
=============================================
Runs ArenaGenerator + ModelOptimizer + Kaggle submission for:
1. Titanic (classification)
2. House Prices (regression)
3. Spaceship Titanic (classification)

Each competition follows the same pipeline:
- Feature engineering from research_v2_findings
- ArenaGenerator with 2 feature sets, 10-min budget
- ModelOptimizer on the winner (randomized, 50 iterations)
- Generate submission + submit via Kaggle CLI
- Check LB score
"""

import sys
import time
import subprocess
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    HistGradientBoostingClassifier,
)
from sklearn.linear_model import Ridge, Lasso
from sklearn.base import clone

warnings.filterwarnings("ignore")

# Add project root to path
PROJECT_ROOT = Path(r"c:\Users\profe\OneDrive\Desktop\Sean Obsidian\Kaggle")
sys.path.insert(0, str(PROJECT_ROOT))

from kaggle_ml_toolkit import (
    ArenaGenerator,
    ModelOptimizer,
    CompetitionConfig,
    ModelArena,
)

# ============================================================================
# GLOBALS
# ============================================================================
RESULTS = {}  # Store final results for summary table


def submit_to_kaggle(file_path: str, competition: str, message: str) -> str:
    """Submit to Kaggle via CLI and return result."""
    cmd = [
        "kaggle", "competitions", "submit",
        "-c", competition,
        "-f", file_path,
        "-m", message,
    ]
    print(f"\n  Submitting: {Path(file_path).name}")
    print(f"  Competition: {competition}")
    print(f"  Message: {message[:80]}...")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode == 0:
            print(f"  ✅ Submission successful!")
            print(f"     {result.stdout.strip()}")
            return "success"
        else:
            error_msg = result.stderr.strip() or result.stdout.strip()
            print(f"  ❌ Submission failed: {error_msg}")
            if "accept" in error_msg.lower() or "rules" in error_msg.lower():
                print(f"  ⚠️  You need to accept the competition rules at:")
                print(f"     https://www.kaggle.com/c/{competition}/rules")
            return f"failed: {error_msg}"
    except subprocess.TimeoutExpired:
        print(f"  ❌ Submission timed out")
        return "timeout"
    except FileNotFoundError:
        print(f"  ❌ Kaggle CLI not found. Install with: pip install kaggle")
        return "cli_not_found"


def check_lb_score(competition: str) -> str:
    """Check latest leaderboard score."""
    print(f"\n  Waiting 10 seconds for LB processing...")
    time.sleep(10)
    
    cmd = ["kaggle", "competitions", "submissions", "-c", competition]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            lines = result.stdout.strip().split("\n")
            if len(lines) >= 2:
                # Parse the most recent submission
                print(f"  Latest submissions:")
                for line in lines[:4]:
                    print(f"    {line}")
                return lines[1] if len(lines) > 1 else "no score yet"
            return "no submissions found"
        else:
            return f"check failed: {result.stderr.strip()}"
    except Exception as e:
        return f"error: {e}"


# ============================================================================
# COMPETITION 1: TITANIC
# ============================================================================
def run_titanic():
    """Full ArenaGenerator sweep for Titanic."""
    print("\n" + "=" * 70)
    print("COMPETITION 1: TITANIC")
    print("=" * 70)
    
    # --- Load Data ---
    data_dir = PROJECT_ROOT / "competitions" / "titanic" / "data"
    train_df = pd.read_csv(data_dir / "train.csv")
    test_df = pd.read_csv(data_dir / "test.csv")
    
    print(f"\n  Train: {train_df.shape}, Test: {test_df.shape}")
    
    # --- Feature Engineering ---
    print("\n  [1/6] Feature Engineering...")
    
    def engineer_titanic(df):
        """Apply best non-leaking features from research V2."""
        df = df.copy()
        
        # Title extraction + grouping
        df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
        title_map = {
            'Mr': 'Mr', 'Miss': 'Miss', 'Mrs': 'Mrs', 'Master': 'Master',
            'Dr': 'Rare', 'Rev': 'Rare', 'Col': 'Rare', 'Major': 'Rare',
            'Mlle': 'Miss', 'Countess': 'Rare', 'Ms': 'Miss', 'Lady': 'Rare',
            'Jonkheer': 'Rare', 'Don': 'Rare', 'Dona': 'Rare', 'Mme': 'Mrs',
            'Capt': 'Rare', 'Sir': 'Rare'
        }
        df['Title'] = df['Title'].map(title_map).fillna('Rare')
        
        # TitleCode (ordinal)
        title_code = {'Mr': 0, 'Miss': 1, 'Mrs': 2, 'Master': 3, 'Rare': 4}
        df['TitleCode'] = df['Title'].map(title_code).fillna(4).astype(int)
        
        # Age imputation by (Title, Pclass) median
        for title in df['Title'].unique():
            for pclass in [1, 2, 3]:
                mask = (df['Title'] == title) & (df['Pclass'] == pclass) & df['Age'].isna()
                fill_val = df.loc[(df['Title'] == title) & (df['Pclass'] == pclass), 'Age'].median()
                if pd.isna(fill_val):
                    fill_val = df.loc[df['Title'] == title, 'Age'].median()
                if pd.isna(fill_val):
                    fill_val = df['Age'].median()
                df.loc[mask, 'Age'] = fill_val
        
        # Fill remaining
        df['Age'] = df['Age'].fillna(df['Age'].median())
        
        # FamilySize, IsAlone
        df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
        df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
        
        # FamilyBucket
        df['FamilyBucket'] = df['FamilySize'].apply(
            lambda x: 1 if x == 1 else (2 if x <= 4 else 3)
        )
        
        # IsFemale, IsChild
        df['IsFemale'] = (df['Sex'] == 'female').astype(int)
        df['IsChild'] = (df['Age'] < 16).astype(int)
        
        # TicketGroupSize for FarePerPerson
        ticket_counts = df.groupby('Ticket')['Ticket'].transform('count')
        df['TicketGroupSize'] = ticket_counts
        
        # Fare imputation
        df['Fare'] = df['Fare'].fillna(df['Fare'].median())
        
        # FarePerPerson
        df['FarePerPerson'] = df['Fare'] / df['TicketGroupSize'].clip(lower=1)
        
        # SexPclass interaction (encoded 0-5)
        df['SexPclass'] = df['IsFemale'] * 3 + (df['Pclass'] - 1)
        
        return df
    
    train_eng = engineer_titanic(train_df)
    test_eng = engineer_titanic(test_df)
    
    # For test TicketGroupSize, need combined data
    combined = pd.concat([train_df[['Ticket']], test_df[['Ticket']]], ignore_index=True)
    ticket_counts_all = combined.groupby('Ticket')['Ticket'].transform('count')
    # Re-engineer test with combined ticket counts
    test_eng['TicketGroupSize'] = combined.iloc[len(train_df):].groupby('Ticket')['Ticket'].transform('count').values
    test_eng['FarePerPerson'] = test_eng['Fare'] / test_eng['TicketGroupSize'].clip(lower=1)
    
    y = train_eng['Survived']
    
    # Define feature sets
    minimal_features = ['IsFemale', 'Pclass', 'Age', 'FarePerPerson', 'FamilyBucket', 'IsChild', 'TitleCode', 'SexPclass']
    extended_features = minimal_features + ['Fare', 'SibSp', 'Parch', 'FamilySize', 'IsAlone']
    
    feature_sets = {
        "minimal": minimal_features,
        "extended": extended_features,
    }
    
    print(f"  Feature sets: minimal={len(minimal_features)}, extended={len(extended_features)}")
    
    # --- ArenaGenerator ---
    print("\n  [2/6] Running ArenaGenerator (budget=10min)...")
    
    gen = ArenaGenerator(
        X=train_eng[extended_features],  # Use extended so both sets are subsets
        y=y,
        problem_type="classification",
        feature_sets=feature_sets,
        time_budget_minutes=10,
        metric="accuracy",
        random_state=42,
        prior_cv_lb_gap=0.06,
    )
    
    arena = gen.build_arena(verbose=True)
    results = arena.run(verbose=True)
    
    # Print leaderboard
    lb = arena.leaderboard()
    print("\n  Arena Leaderboard (top 10):")
    print(lb.head(10).to_string(index=False))
    
    # --- Get Winner ---
    winner = results[0]
    print(f"\n  [3/6] Arena Winner: {winner.name}")
    print(f"         CV: {winner.mean_score:.4f} ± {winner.std_score:.4f}")
    print(f"         Worthiness: {winner.submit_worthiness:.4f}")
    
    # --- ModelOptimizer on Winner ---
    print("\n  [4/6] Optimizing winner with ModelOptimizer (50 iterations)...")
    
    param_grid = {
        "n_estimators": [30, 50, 75, 100, 150],
        "max_depth": [2, 3, 4],
        "learning_rate": [0.05, 0.1, 0.15, 0.2],
        "min_samples_leaf": [10, 15, 20, 25, 30],
        "subsample": [0.7, 0.8, 0.9],
    }
    
    # Use GBM as the base model for optimization
    base_model = GradientBoostingClassifier(random_state=42)
    winner_features = winner.feature_cols
    
    optimizer = ModelOptimizer()
    opt_result = optimizer.optimize(
        model=base_model,
        param_grid=param_grid,
        X=train_eng[winner_features],
        y=y,
        method="randomized",
        scoring="accuracy",
        cv_folds=5,
        n_iter=50,
        random_state=42,
    )
    
    print(f"  Best params: {opt_result['best_params']}")
    print(f"  Best CV score: {opt_result['best_score']:.4f}")
    
    # --- Generate Submission ---
    print("\n  [5/6] Generating submission...")
    
    best_model = opt_result['best_model']
    # Refit on full training data
    best_model.fit(train_eng[winner_features], y)
    predictions = best_model.predict(test_eng[winner_features])
    
    # Create submission
    sub_dir = PROJECT_ROOT / "competitions" / "titanic" / "submissions"
    sub_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sub_path = sub_dir / f"arena_v3_optimized_{timestamp}.csv"
    
    sub_df = pd.DataFrame({
        'PassengerId': test_eng['PassengerId'],
        'Survived': predictions.astype(int),
    })
    sub_df.to_csv(sub_path, index=False)
    print(f"  Saved: {sub_path}")
    print(f"  Shape: {sub_df.shape}, Survived distribution: {sub_df['Survived'].value_counts().to_dict()}")
    
    # --- Submit ---
    print("\n  [6/6] Submitting to Kaggle...")
    model_details = f"GBM depth={opt_result['best_params'].get('max_depth')}, " \
                    f"trees={opt_result['best_params'].get('n_estimators')}, " \
                    f"lr={opt_result['best_params'].get('learning_rate')}, " \
                    f"CV={opt_result['best_score']:.4f}"
    
    message = f"ArenaGen V3 + optimized: {model_details}"
    submit_result = submit_to_kaggle(str(sub_path), "titanic", message)
    
    # Check LB
    lb_info = check_lb_score("titanic")
    
    RESULTS["titanic"] = {
        "previous_best": 0.77272,
        "cv_score": opt_result['best_score'],
        "params": opt_result['best_params'],
        "winner": winner.name,
        "submission_file": str(sub_path),
        "submit_status": submit_result,
        "lb_info": lb_info,
    }
    
    return opt_result


# ============================================================================
# COMPETITION 2: HOUSE PRICES
# ============================================================================
def run_house_prices():
    """Full ArenaGenerator sweep for House Prices."""
    print("\n" + "=" * 70)
    print("COMPETITION 2: HOUSE PRICES")
    print("=" * 70)
    
    # --- Load Data ---
    data_dir = PROJECT_ROOT / "competitions" / "house-prices" / "data"
    train_df = pd.read_csv(data_dir / "train.csv")
    test_df = pd.read_csv(data_dir / "test.csv")
    
    print(f"\n  Train: {train_df.shape}, Test: {test_df.shape}")
    
    # --- Feature Engineering ---
    print("\n  [1/7] Feature Engineering...")
    
    def engineer_house_prices(df, is_train=True, train_ref=None):
        """Apply research V2 feature engineering."""
        df = df.copy()
        
        # Quality ordinals
        quality_map = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'NA': 0}
        quality_cols = ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond',
                       'HeatingQC', 'KitchenQual', 'FireplaceQu', 'GarageQual', 'GarageCond']
        
        for col in quality_cols:
            if col in df.columns:
                df[col] = df[col].map(quality_map).fillna(0)
        
        # Derived features
        df['TotalSF'] = df.get('TotalBsmtSF', pd.Series(0, index=df.index)).fillna(0) + \
                        df.get('1stFlrSF', pd.Series(0, index=df.index)).fillna(0) + \
                        df.get('2ndFlrSF', pd.Series(0, index=df.index)).fillna(0)
        
        df['TotalBath'] = df.get('FullBath', pd.Series(0, index=df.index)).fillna(0) + \
                         0.5 * df.get('HalfBath', pd.Series(0, index=df.index)).fillna(0) + \
                         df.get('BsmtFullBath', pd.Series(0, index=df.index)).fillna(0) + \
                         0.5 * df.get('BsmtHalfBath', pd.Series(0, index=df.index)).fillna(0)
        
        df['HouseAge'] = df.get('YrSold', pd.Series(2010, index=df.index)).fillna(2010) - \
                        df.get('YearBuilt', pd.Series(1970, index=df.index)).fillna(1970)
        
        df['RemodAge'] = df.get('YrSold', pd.Series(2010, index=df.index)).fillna(2010) - \
                        df.get('YearRemodAdd', pd.Series(1970, index=df.index)).fillna(1970)
        
        df['GarageCars'] = df.get('GarageCars', pd.Series(0, index=df.index)).fillna(0)
        df['GarageArea'] = df.get('GarageArea', pd.Series(0, index=df.index)).fillna(0)
        
        # Polynomial features
        df['OverallQual_sq'] = df['OverallQual'] ** 2
        df['TotalSF_sq'] = df['TotalSF'] ** 2
        
        # Interaction
        df['Qual_x_GrLivArea'] = df['OverallQual'] * df['GrLivArea']
        
        # Label encode all remaining categoricals
        cat_cols = df.select_dtypes(include=['object']).columns
        for col in cat_cols:
            df[col] = df[col].fillna('None')
            # Use label encoding
            df[col] = df[col].astype('category').cat.codes
        
        # Fill remaining NaN with 0
        df = df.fillna(0)
        
        return df
    
    # Remove outliers from training
    outlier_mask = ~((train_df['GrLivArea'] > 4000) & (train_df['SalePrice'] < 300000))
    train_df = train_df[outlier_mask].reset_index(drop=True)
    print(f"  After outlier removal: {train_df.shape}")
    
    # Log-transform target
    y = np.log1p(train_df['SalePrice'])
    
    # Engineer features
    train_eng = engineer_house_prices(train_df, is_train=True)
    test_eng = engineer_house_prices(test_df, is_train=False, train_ref=train_df)
    
    # Log-transform skewed numerics
    numeric_cols = train_eng.select_dtypes(include=[np.number]).columns
    numeric_cols = [c for c in numeric_cols if c not in ['Id', 'SalePrice']]
    
    from scipy.stats import skew
    skewed = []
    for col in numeric_cols:
        if col in train_eng.columns:
            sk = skew(train_eng[col].dropna())
            if abs(sk) > 0.75:
                skewed.append(col)
                train_eng[col] = np.log1p(train_eng[col].clip(lower=0))
                if col in test_eng.columns:
                    test_eng[col] = np.log1p(test_eng[col].clip(lower=0))
    
    print(f"  Log-transformed {len(skewed)} skewed features")
    
    # Define feature sets
    # Remove ID and target
    drop_cols = ['Id', 'SalePrice']
    all_features = [c for c in train_eng.columns if c not in drop_cols]
    
    # Core numeric features (~20 most important)
    core_features = [
        'OverallQual', 'GrLivArea', 'TotalSF', 'TotalBath', 'GarageCars',
        'GarageArea', 'HouseAge', 'RemodAge', 'TotalBsmtSF', '1stFlrSF',
        'OverallCond', 'LotArea', 'BsmtQual', 'KitchenQual', 'ExterQual',
        'FireplaceQu', 'OverallQual_sq', 'TotalSF_sq', 'Qual_x_GrLivArea',
        'FullBath',
    ]
    # Filter to actually available columns
    core_features = [c for c in core_features if c in train_eng.columns]
    
    # Full feature set
    full_features = [c for c in all_features if c in train_eng.columns and c in test_eng.columns]
    
    feature_sets = {
        "core_numeric": core_features,
        "full": full_features,
    }
    
    print(f"  Feature sets: core_numeric={len(core_features)}, full={len(full_features)}")
    
    # --- ArenaGenerator ---
    print("\n  [2/7] Running ArenaGenerator (budget=10min)...")
    
    # Ensure test has all features
    for col in full_features:
        if col not in test_eng.columns:
            test_eng[col] = 0
    
    gen = ArenaGenerator(
        X=train_eng[full_features],
        y=y,
        problem_type="regression",
        feature_sets=feature_sets,
        time_budget_minutes=10,
        metric="rmse",
        random_state=42,
        prior_cv_lb_gap=0.01,
    )
    
    arena = gen.build_arena(verbose=True)
    results = arena.run(verbose=True)
    
    # Print leaderboard
    lb = arena.leaderboard()
    print("\n  Arena Leaderboard (top 10):")
    print(lb.head(10).to_string(index=False))
    
    # --- Get Winner ---
    winner = results[0]
    print(f"\n  [3/7] Arena Winner: {winner.name}")
    print(f"         CV RMSE: {winner.mean_score:.4f} ± {winner.std_score:.4f}")
    
    # --- ModelOptimizer on Winner ---
    print("\n  [4/7] Optimizing winner...")
    
    param_grid_hp = {
        "n_estimators": [100, 200, 300, 400, 500],
        "max_depth": [3, 4, 5, 6],
        "learning_rate": [0.01, 0.03, 0.05, 0.08, 0.1],
        "min_samples_leaf": [5, 8, 10, 15, 20],
        "subsample": [0.7, 0.8, 0.9, 1.0],
    }
    
    base_model_hp = GradientBoostingRegressor(random_state=42)
    winner_features = winner.feature_cols
    
    optimizer = ModelOptimizer()
    opt_result = optimizer.optimize(
        model=base_model_hp,
        param_grid=param_grid_hp,
        X=train_eng[winner_features],
        y=y,
        method="randomized",
        scoring="rmse",
        cv_folds=5,
        n_iter=50,
        random_state=42,
    )
    
    print(f"  Best params: {opt_result['best_params']}")
    print(f"  Best CV RMSE: {opt_result['best_score']:.5f}")
    
    # --- Stacking Submission ---
    print("\n  [5/7] Building stacking ensemble (Ridge + Lasso + GBM)...")
    
    from sklearn.model_selection import KFold
    
    # Get predictions from each model using cross-validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    # Fit final models on full training data
    ridge_model = Ridge(alpha=10)
    lasso_model = Lasso(alpha=0.0005, max_iter=5000)
    gbm_model = opt_result['best_model']
    
    # Fit all models
    ridge_model.fit(train_eng[winner_features], y)
    lasso_model.fit(train_eng[winner_features], y)
    gbm_model_full = clone(gbm_model)
    gbm_model_full.fit(train_eng[winner_features], y)
    
    # Predict on test
    pred_ridge = ridge_model.predict(test_eng[winner_features])
    pred_lasso = lasso_model.predict(test_eng[winner_features])
    pred_gbm = gbm_model_full.predict(test_eng[winner_features])
    
    # Average
    pred_stack = (pred_ridge + pred_lasso + pred_gbm) / 3.0
    
    # Inverse log transform
    pred_final = np.expm1(pred_stack)
    pred_final = np.clip(pred_final, 0, None)  # No negative prices
    
    print(f"  Stacked predictions — min: {pred_final.min():.0f}, max: {pred_final.max():.0f}, mean: {pred_final.mean():.0f}")
    
    # --- Generate Submission ---
    print("\n  [6/7] Generating submission...")
    
    sub_dir = PROJECT_ROOT / "competitions" / "house-prices" / "submissions"
    sub_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sub_path = sub_dir / f"arena_v3_stack_{timestamp}.csv"
    
    sub_df = pd.DataFrame({
        'Id': test_eng['Id'].astype(int),
        'SalePrice': pred_final,
    })
    sub_df.to_csv(sub_path, index=False)
    print(f"  Saved: {sub_path}")
    
    # --- Submit ---
    print("\n  [7/7] Submitting to Kaggle...")
    details = f"Ridge+Lasso+GBM stack, GBM depth={opt_result['best_params'].get('max_depth')}, " \
              f"trees={opt_result['best_params'].get('n_estimators')}, CV_RMSE={opt_result['best_score']:.5f}"
    message = f"ArenaGen V3 + stack: {details}"
    
    submit_result = submit_to_kaggle(
        str(sub_path),
        "house-prices-advanced-regression-techniques",
        message
    )
    
    lb_info = check_lb_score("house-prices-advanced-regression-techniques")
    
    RESULTS["house-prices"] = {
        "previous_best": 0.12728,
        "cv_score": opt_result['best_score'],
        "params": opt_result['best_params'],
        "winner": winner.name,
        "submission_file": str(sub_path),
        "submit_status": submit_result,
        "lb_info": lb_info,
    }
    
    return opt_result


# ============================================================================
# COMPETITION 3: SPACESHIP TITANIC
# ============================================================================
def run_spaceship_titanic():
    """Full ArenaGenerator sweep for Spaceship Titanic."""
    print("\n" + "=" * 70)
    print("COMPETITION 3: SPACESHIP TITANIC")
    print("=" * 70)
    
    # --- Load Data ---
    data_dir = PROJECT_ROOT / "competitions" / "spaceship-titanic" / "data"
    train_df = pd.read_csv(data_dir / "train.csv")
    test_df = pd.read_csv(data_dir / "test.csv")
    
    print(f"\n  Train: {train_df.shape}, Test: {test_df.shape}")
    
    # --- Feature Engineering ---
    print("\n  [1/6] Feature Engineering...")
    
    def engineer_spaceship(df):
        """Apply research V2 features for Spaceship Titanic."""
        df = df.copy()
        
        # Parse PassengerId for group info
        df['GroupId'] = df['PassengerId'].apply(lambda x: x.split('_')[0] if isinstance(x, str) else '0')
        group_sizes = df.groupby('GroupId')['GroupId'].transform('count')
        df['GroupSize'] = group_sizes
        df['IsAlone'] = (df['GroupSize'] == 1).astype(int)
        
        # CryoSleep and VIP (binary)
        df['CryoSleep'] = df['CryoSleep'].map({True: 1, False: 0, 'True': 1, 'False': 0}).fillna(0).astype(int)
        df['VIP'] = df['VIP'].map({True: 1, False: 0, 'True': 1, 'False': 0}).fillna(0).astype(int)
        
        # Spending columns
        spend_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
        for col in spend_cols:
            df[col] = df[col].fillna(0)
        
        df['TotalSpending'] = df[spend_cols].sum(axis=1)
        df['LogSpending'] = np.log1p(df['TotalSpending'])
        df['NoSpending'] = (df['TotalSpending'] == 0).astype(int)
        
        # Spending ratios
        for col in spend_cols:
            df[f'{col}Ratio'] = df[col] / df['TotalSpending'].clip(lower=1)
        
        # Per-amenity flags
        for col in spend_cols:
            df[f'SpentOn{col}'] = (df[col] > 0).astype(int)
        
        # NumServicesUsed
        df['NumServicesUsed'] = sum((df[col] > 0).astype(int) for col in spend_cols)
        
        # Cabin parsing
        df['Cabin_Deck'] = df['Cabin'].apply(lambda x: x.split('/')[0] if isinstance(x, str) else 'Unknown')
        df['Cabin_Num'] = df['Cabin'].apply(
            lambda x: int(x.split('/')[1]) if isinstance(x, str) and len(x.split('/')) >= 2 and x.split('/')[1].isdigit() else 0
        )
        df['Cabin_Side'] = df['Cabin'].apply(lambda x: x.split('/')[2] if isinstance(x, str) and len(x.split('/')) >= 3 else 'Unknown')
        
        # DeckNum encoding
        deck_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'T': 7, 'Unknown': 8}
        df['DeckNum'] = df['Cabin_Deck'].map(deck_map).fillna(8).astype(int)
        
        # IsPort (cabin side)
        df['IsPort'] = (df['Cabin_Side'] == 'P').astype(int)
        
        # CabinRegion (bin cabin numbers)
        df['CabinRegion'] = pd.cut(df['Cabin_Num'], bins=[0, 300, 600, 900, 1200, 1500, 2000], labels=[0, 1, 2, 3, 4, 5]).astype(float).fillna(3)
        
        # Age groups (7 bins)
        df['Age'] = df['Age'].fillna(df['Age'].median())
        df['AgeGroup'] = pd.cut(df['Age'], bins=[0, 5, 12, 18, 25, 40, 60, 100], labels=[0, 1, 2, 3, 4, 5, 6]).astype(float).fillna(3)
        
        # HomePlanet encoding
        planet_map = {'Earth': 0, 'Europa': 1, 'Mars': 2}
        df['HomePlanet_enc'] = df['HomePlanet'].map(planet_map).fillna(0).astype(int)
        
        # Destination encoding
        dest_map = {'TRAPPIST-1e': 0, '55 Cancri e': 1, 'PSO J318.5-22': 2}
        df['Destination_enc'] = df['Destination'].map(dest_map).fillna(0).astype(int)
        
        # HomePlanet × CryoSleep interactions
        df['EarthCryo'] = ((df['HomePlanet_enc'] == 0) & (df['CryoSleep'] == 1)).astype(int)
        df['EuropaCryo'] = ((df['HomePlanet_enc'] == 1) & (df['CryoSleep'] == 1)).astype(int)
        df['MarsCryo'] = ((df['HomePlanet_enc'] == 2) & (df['CryoSleep'] == 1)).astype(int)
        
        return df
    
    train_eng = engineer_spaceship(train_df)
    test_eng = engineer_spaceship(test_df)
    
    # Target
    y = train_eng['Transported'].map({True: 1, False: 0, 'True': 1, 'False': 0}).fillna(0).astype(int)
    
    # Define feature sets
    core_features = [
        'CryoSleep', 'VIP', 'Age', 'TotalSpending', 'LogSpending', 'NoSpending',
        'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck',
        'NumServicesUsed', 'GroupSize', 'IsAlone', 'DeckNum', 'IsPort',
        'CabinRegion', 'AgeGroup', 'HomePlanet_enc', 'Destination_enc',
        'EarthCryo', 'EuropaCryo', 'MarsCryo',
        'RoomServiceRatio', 'SpaRatio',
    ]
    
    full_features = core_features + [
        'FoodCourtRatio', 'ShoppingMallRatio', 'VRDeckRatio',
        'SpentOnRoomService', 'SpentOnFoodCourt', 'SpentOnShoppingMall',
        'SpentOnSpa', 'SpentOnVRDeck',
        'Cabin_Num', 'FamilyBucket' if 'FamilyBucket' in train_eng.columns else 'GroupSize',
    ]
    # Remove duplicates and ensure all exist
    full_features = list(dict.fromkeys(full_features))
    full_features = [c for c in full_features if c in train_eng.columns and c in test_eng.columns]
    core_features = [c for c in core_features if c in train_eng.columns and c in test_eng.columns]
    
    feature_sets = {
        "core": core_features,
        "full": full_features,
    }
    
    print(f"  Feature sets: core={len(core_features)}, full={len(full_features)}")
    
    # --- ArenaGenerator ---
    print("\n  [2/6] Running ArenaGenerator (budget=10min)...")
    
    gen = ArenaGenerator(
        X=train_eng[full_features],
        y=y,
        problem_type="classification",
        feature_sets=feature_sets,
        time_budget_minutes=10,
        metric="accuracy",
        random_state=42,
        prior_cv_lb_gap=0.01,
    )
    
    arena = gen.build_arena(verbose=True)
    results = arena.run(verbose=True)
    
    lb = arena.leaderboard()
    print("\n  Arena Leaderboard (top 10):")
    print(lb.head(10).to_string(index=False))
    
    # --- Get Winner ---
    winner = results[0]
    print(f"\n  [3/6] Arena Winner: {winner.name}")
    print(f"         CV: {winner.mean_score:.4f} ± {winner.std_score:.4f}")
    
    # --- ModelOptimizer with HistGradientBoosting ---
    print("\n  [4/6] Optimizing with HistGradientBoostingClassifier (50 iterations)...")
    
    param_grid_ss = {
        "max_iter": [200, 300, 400, 500],
        "max_depth": [4, 5, 6, 7],
        "learning_rate": [0.03, 0.05, 0.08, 0.1],
        "min_samples_leaf": [5, 10, 15, 20],
        "max_leaf_nodes": [31, 50, 80, None],
        "l2_regularization": [0, 0.1, 1.0],
    }
    
    base_model_ss = HistGradientBoostingClassifier(random_state=42)
    winner_features = winner.feature_cols
    
    optimizer = ModelOptimizer()
    opt_result = optimizer.optimize(
        model=base_model_ss,
        param_grid=param_grid_ss,
        X=train_eng[winner_features],
        y=y,
        method="randomized",
        scoring="accuracy",
        cv_folds=5,
        n_iter=50,
        random_state=42,
    )
    
    print(f"  Best params: {opt_result['best_params']}")
    print(f"  Best CV score: {opt_result['best_score']:.4f}")
    
    # --- Generate Submission ---
    print("\n  [5/6] Generating submission...")
    
    best_model = opt_result['best_model']
    best_model.fit(train_eng[winner_features], y)
    predictions = best_model.predict(test_eng[winner_features])
    
    # Spaceship Titanic expects True/False
    pred_bool = predictions.astype(bool)
    
    sub_dir = PROJECT_ROOT / "competitions" / "spaceship-titanic" / "submissions"
    sub_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sub_path = sub_dir / f"arena_v3_hgb_{timestamp}.csv"
    
    sub_df = pd.DataFrame({
        'PassengerId': test_eng['PassengerId'],
        'Transported': pred_bool,
    })
    sub_df.to_csv(sub_path, index=False)
    print(f"  Saved: {sub_path}")
    print(f"  Shape: {sub_df.shape}, Transported distribution: {sub_df['Transported'].value_counts().to_dict()}")
    
    # --- Submit ---
    print("\n  [6/6] Submitting to Kaggle...")
    details = f"HGB max_iter={opt_result['best_params'].get('max_iter')}, " \
              f"depth={opt_result['best_params'].get('max_depth')}, " \
              f"lr={opt_result['best_params'].get('learning_rate')}, " \
              f"CV={opt_result['best_score']:.4f}"
    message = f"ArenaGen V3 + HGB optimized: {details}"
    
    submit_result = submit_to_kaggle(str(sub_path), "spaceship-titanic", message)
    lb_info = check_lb_score("spaceship-titanic")
    
    RESULTS["spaceship-titanic"] = {
        "previous_best": 0.80243,
        "cv_score": opt_result['best_score'],
        "params": opt_result['best_params'],
        "winner": winner.name,
        "submission_file": str(sub_path),
        "submit_status": submit_result,
        "lb_info": lb_info,
    }
    
    return opt_result


# ============================================================================
# FINAL SUMMARY & LOG UPDATES
# ============================================================================
def print_final_summary():
    """Print the final results table."""
    print("\n\n" + "=" * 70)
    print("=== FINAL RESULTS ===")
    print("=" * 70)
    print(f"\n{'Competition':<25} {'Previous Best':<15} {'New CV Score':<15} {'Arena Winner':<30} {'Status'}")
    print("-" * 100)
    
    for comp, data in RESULTS.items():
        print(f"{comp:<25} {data['previous_best']:<15.5f} {data['cv_score']:<15.5f} {data['winner']:<30} {data['submit_status']}")
    
    print("\n" + "=" * 70)
    print("Detailed parameters:")
    for comp, data in RESULTS.items():
        print(f"\n  {comp}:")
        print(f"    Winner: {data['winner']}")
        print(f"    Best params: {data['params']}")
        print(f"    File: {data['submission_file']}")


def update_experiment_logs():
    """Update experiments_log.md for each competition."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    
    for comp, data in RESULTS.items():
        log_path = PROJECT_ROOT / "competitions" / comp / "experiments_log.md"
        
        entry = f"""
---

## ArenaGenerator V3 Sweep — {timestamp}

**Arena Winner:** {data['winner']}
**CV Score:** {data['cv_score']:.5f}
**Optimized Params:** {data['params']}
**Previous Best LB:** {data['previous_best']}
**Submission Status:** {data['submit_status']}
**File:** {Path(data['submission_file']).name}
**Method:** ArenaGenerator (10-min budget, 2 feature sets) → ModelOptimizer (randomized, 50 iter)
"""
        
        try:
            with open(log_path, 'a', encoding='utf-8') as f:
                f.write(entry)
            print(f"  Updated: {log_path}")
        except Exception as e:
            print(f"  Failed to update {log_path}: {e}")


# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("ArenaGenerator V3 — Full Sweep + Optimization + Submission")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    try:
        run_titanic()
    except Exception as e:
        print(f"\n  ❌ Titanic failed: {e}")
        import traceback
        traceback.print_exc()
        RESULTS["titanic"] = {"previous_best": 0.77272, "cv_score": 0, "params": {}, "winner": "FAILED", "submission_file": "", "submit_status": str(e), "lb_info": ""}
    
    try:
        run_house_prices()
    except Exception as e:
        print(f"\n  ❌ House Prices failed: {e}")
        import traceback
        traceback.print_exc()
        RESULTS["house-prices"] = {"previous_best": 0.12728, "cv_score": 0, "params": {}, "winner": "FAILED", "submission_file": "", "submit_status": str(e), "lb_info": ""}
    
    try:
        run_spaceship_titanic()
    except Exception as e:
        print(f"\n  ❌ Spaceship Titanic failed: {e}")
        import traceback
        traceback.print_exc()
        RESULTS["spaceship-titanic"] = {"previous_best": 0.80243, "cv_score": 0, "params": {}, "winner": "FAILED", "submission_file": "", "submit_status": str(e), "lb_info": ""}
    
    # Final summary
    print_final_summary()
    
    # Update logs
    print("\n\nUpdating experiment logs...")
    update_experiment_logs()
    
    print(f"\n\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
