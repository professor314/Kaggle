"""
Full Arena + Hyperparameter Tuning + Submission Pipeline
=========================================================
Runs ArenaGenerator → ModelOptimizer → Submission for all 3 competitions:
1. Titanic (classification, 891 rows)
2. House Prices (regression, 1460 rows)
3. Spaceship Titanic (classification, 8693 rows)

Estimated runtime: ~35 minutes total
"""

import sys
import os

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

import time
import subprocess
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings('ignore')

# Add parent to path for toolkit imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kaggle_ml_toolkit import ArenaGenerator, ModelOptimizer
from sklearn.ensemble import (
    GradientBoostingClassifier, GradientBoostingRegressor,
    RandomForestClassifier, RandomForestRegressor,
    HistGradientBoostingClassifier,
)
from sklearn.linear_model import LogisticRegression, Ridge, Lasso, ElasticNet
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
from scipy.stats import skew

# Base directory
BASE_DIR = Path(__file__).resolve().parent

# Store results for final summary
RESULTS = {}


def submit_to_kaggle(comp_name, filepath, message):
    """Submit a file to Kaggle and retrieve score."""
    print(f"\n  Submitting to '{comp_name}'...")
    result = subprocess.run(
        ["kaggle", "competitions", "submit", "-c", comp_name, "-f", str(filepath), "-m", message],
        capture_output=True, text=True
    )
    print(f"  {result.stdout.strip() or result.stderr.strip()}")
    
    print("  Waiting 12s for score processing...")
    time.sleep(12)
    
    score_result = subprocess.run(
        ["kaggle", "competitions", "submissions", "-c", comp_name],
        capture_output=True, text=True
    )
    print(f"  Recent submissions:\n{score_result.stdout[:600]}")
    return score_result.stdout


def build_param_grid_around_winner(winner_params, model_type):
    """Expand a parameter grid around the winner's parameters.
    
    Only tunes the key hyperparameters that matter for each model type.
    Avoids tuning parameters that have constraints (e.g., GBM alpha must be in (0,1)).
    """
    # Define which params are tunable per model type
    TUNABLE_PARAMS = {
        'GradientBoostingClassifier': ['n_estimators', 'max_depth', 'learning_rate', 'min_samples_leaf', 'subsample'],
        'GradientBoostingRegressor': ['n_estimators', 'max_depth', 'learning_rate', 'min_samples_leaf', 'subsample'],
        'RandomForestClassifier': ['n_estimators', 'max_depth', 'min_samples_leaf'],
        'RandomForestRegressor': ['n_estimators', 'max_depth', 'min_samples_leaf'],
        'LogisticRegression': ['C'],
        'Ridge': ['alpha'],
        'Lasso': ['alpha'],
        'ElasticNet': ['alpha', 'l1_ratio'],
        'HistGradientBoostingClassifier': ['max_iter', 'max_depth', 'learning_rate', 'min_samples_leaf'],
        'KNeighborsClassifier': ['n_neighbors'],
    }
    
    tunable = TUNABLE_PARAMS.get(model_type, list(winner_params.keys()))
    grid = {}
    
    for key, val in winner_params.items():
        if key not in tunable:
            continue
        if val is None:
            continue
        if not isinstance(val, (int, float)):
            continue
            
        if key == 'n_estimators':
            v = int(val)
            grid[key] = sorted(set([max(50, v - 50), v, v + 50, v + 100]))
        elif key == 'max_iter':
            v = int(val)
            grid[key] = sorted(set([max(100, v - 100), v, v + 100, v + 200]))
        elif key == 'max_depth':
            v = int(val)
            grid[key] = sorted(set([max(2, v - 1), v, v + 1, v + 2]))
        elif key == 'learning_rate':
            v = float(val)
            grid[key] = sorted(set([max(0.01, round(v * 0.6, 4)), round(v, 4),
                                     round(v * 1.5, 4), round(v * 2, 4)]))
        elif key == 'min_samples_leaf':
            v = int(val)
            grid[key] = sorted(set([max(1, v - 3), v, v + 2, v + 5]))
        elif key == 'subsample':
            v = float(val)
            grid[key] = [max(0.5, round(v - 0.1, 2)), round(v, 2), min(1.0, round(v + 0.1, 2))]
        elif key == 'C':
            v = float(val)
            grid[key] = [v / 10, v / 2, v, v * 2, v * 10]
        elif key == 'alpha' and model_type in ('Ridge', 'Lasso', 'ElasticNet'):
            # Only tune alpha for linear models (not GBM where alpha is quantile param)
            v = float(val)
            grid[key] = [max(0.0001, v / 10), v / 2, v, v * 2, v * 5]
        elif key == 'n_neighbors':
            v = int(val)
            grid[key] = sorted(set([max(1, v - 2), v, v + 2, v + 4]))
        elif key == 'l1_ratio':
            v = float(val)
            grid[key] = [max(0.1, v - 0.2), v, min(0.9, v + 0.2)]
    
    return grid


def instantiate_model_from_type(model_type_name, params, problem_type):
    """Recreate a model instance from its type name and params."""
    import inspect
    
    classifiers = {
        'GradientBoostingClassifier': GradientBoostingClassifier,
        'RandomForestClassifier': RandomForestClassifier,
        'LogisticRegression': LogisticRegression,
        'HistGradientBoostingClassifier': HistGradientBoostingClassifier,
    }
    regressors = {
        'GradientBoostingRegressor': GradientBoostingRegressor,
        'RandomForestRegressor': RandomForestRegressor,
        'Ridge': Ridge,
        'Lasso': Lasso,
        'ElasticNet': ElasticNet,
    }
    
    all_models = {**classifiers, **regressors}
    
    if model_type_name in all_models:
        cls = all_models[model_type_name]
        valid_params = inspect.signature(cls).parameters
        filtered = {k: v for k, v in params.items() if k in valid_params}
        return cls(**filtered)
    
    raise ValueError(f"Unknown model type: {model_type_name}")


# ============================================================================
# COMPETITION 1: TITANIC
# ============================================================================
def run_titanic():
    print("\n" + "=" * 70)
    print("COMPETITION 1: TITANIC (Classification, 891 rows)")
    print("=" * 70)
    
    # Load data
    train = pd.read_csv(BASE_DIR / "titanic" / "data" / "train.csv")
    test = pd.read_csv(BASE_DIR / "titanic" / "data" / "test.csv")
    
    print(f"  Train: {train.shape}, Test: {test.shape}")
    
    # Feature Engineering (from Research V2)
    def engineer_titanic(df):
        df = df.copy()
        
        # Title extraction
        df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
        title_map = {
            'Mr': 1, 'Miss': 2, 'Mrs': 3, 'Master': 4,
            'Dr': 5, 'Rev': 5, 'Col': 5, 'Major': 5, 'Capt': 5,
            'Mlle': 2, 'Ms': 2, 'Mme': 3, 'Lady': 3, 'Countess': 3,
            'Dona': 3, 'Don': 1, 'Sir': 1, 'Jonkheer': 1,
        }
        df['TitleCode'] = df['Title'].map(title_map).fillna(1).astype(int)
        
        # Age imputation (title-based)
        title_age_medians = df.groupby('Title')['Age'].median()
        for title, median_age in title_age_medians.items():
            mask = (df['Age'].isna()) & (df['Title'] == title)
            df.loc[mask, 'Age'] = median_age
        df['Age'] = df['Age'].fillna(df['Age'].median())
        
        # Core features
        df['IsFemale'] = (df['Sex'] == 'female').astype(int)
        df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
        df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
        df['IsChild'] = (df['Age'] < 16).astype(int)
        df['Fare'] = df['Fare'].fillna(df['Fare'].median())
        df['FarePerPerson'] = df['Fare'] / df['FamilySize'].clip(lower=1)
        
        # Interaction features
        df['SexPclass'] = df['IsFemale'] * 10 + df['Pclass']
        
        return df
    
    train = engineer_titanic(train)
    test = engineer_titanic(test)
    
    # Define feature sets
    minimal_features = ['IsFemale', 'Pclass', 'Age', 'Fare', 'FamilySize', 'IsAlone', 'IsChild', 'TitleCode']
    research_features = minimal_features + ['FarePerPerson', 'SexPclass', 'SibSp', 'Parch']
    
    # Prepare X, y
    y = train['Survived']
    
    feature_sets = {
        "minimal": minimal_features,
        "research": research_features,
    }
    
    print(f"  Feature sets: minimal={len(minimal_features)}, research={len(research_features)}")
    
    # Run ArenaGenerator
    print("\n  Running ArenaGenerator (time_budget=10 min)...")
    gen = ArenaGenerator(
        X=train[research_features],  # Use research features as the superset
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
    
    # Get winner info
    winner = results[0]
    arena_cv = winner.mean_score
    print(f"\n  Arena Winner: {winner.name}")
    print(f"  CV Score: {arena_cv:.4f} ± {winner.std_score:.4f}")
    print(f"  Model Type: {winner.model_type}")
    print(f"  Params: {winner.model_params}")
    
    # Hyperparameter tuning
    print("\n  Running ModelOptimizer (30 iterations)...")
    param_grid = build_param_grid_around_winner(winner.model_params, winner.model_type)
    
    if not param_grid:
        # Fallback grid for models without tunable params
        param_grid = {'C': [0.01, 0.1, 1.0, 10.0]} if winner.model_type == 'LogisticRegression' else {
            'n_estimators': [100, 150, 200], 'max_depth': [2, 3, 4, 5]
        }
    
    print(f"  Param grid: {param_grid}")
    
    base_model = instantiate_model_from_type(winner.model_type, winner.model_params, "classification")
    
    optimizer = ModelOptimizer()
    opt_result = optimizer.optimize(
        model=base_model,
        param_grid=param_grid,
        X=train[winner.feature_cols],
        y=y,
        method="randomized",
        scoring="accuracy",
        cv_folds=5,
        n_iter=30,
        random_state=42,
    )
    
    tuned_cv = opt_result['best_score']
    print(f"  Tuned CV: {tuned_cv:.4f}")
    print(f"  Best Params: {opt_result['best_params']}")
    
    # Generate submission with tuned model
    best_model = opt_result['best_model']
    best_model.fit(train[winner.feature_cols], y)
    predictions = best_model.predict(test[winner.feature_cols])
    
    submission = pd.DataFrame({
        'PassengerId': test['PassengerId'],
        'Survived': predictions.astype(int)
    })
    
    sub_path = BASE_DIR / "titanic" / "submissions" / "arena_tuned_submission.csv"
    sub_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(sub_path, index=False)
    print(f"  Submission saved: {sub_path}")
    print(f"  Predictions distribution: {pd.Series(predictions).value_counts().to_dict()}")
    
    # Submit
    lb_output = submit_to_kaggle("titanic", sub_path, 
        f"Arena+Tuned {winner.model_type} CV={tuned_cv:.4f}")
    
    RESULTS['titanic'] = {
        'arena_cv': arena_cv,
        'tuned_cv': tuned_cv,
        'winner': winner.name,
        'model_type': winner.model_type,
    }


# ============================================================================
# COMPETITION 2: HOUSE PRICES
# ============================================================================
def run_house_prices():
    print("\n" + "=" * 70)
    print("COMPETITION 2: HOUSE PRICES (Regression, 1460 rows)")
    print("=" * 70)
    
    # Load data
    train = pd.read_csv(BASE_DIR / "house-prices" / "data" / "train.csv")
    test = pd.read_csv(BASE_DIR / "house-prices" / "data" / "test.csv")
    
    print(f"  Train: {train.shape}, Test: {test.shape}")
    
    # Remove outliers
    train = train[~((train['GrLivArea'] > 4000) & (train['SalePrice'] < 200000))].reset_index(drop=True)
    print(f"  After outlier removal: {train.shape}")
    
    # Log transform target
    y = np.log1p(train['SalePrice'])
    
    # Feature Engineering (from Research V2)
    def engineer_house_prices(df, is_train=True):
        df = df.copy()
        
        # Total SF
        df['TotalSF'] = df.get('TotalBsmtSF', pd.Series(0, index=df.index)).fillna(0) + \
                        df.get('1stFlrSF', pd.Series(0, index=df.index)).fillna(0) + \
                        df.get('2ndFlrSF', pd.Series(0, index=df.index)).fillna(0)
        
        # Quality ordinal encoding
        quality_map = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'NA': 0}
        quality_cols = ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 
                       'HeatingQC', 'KitchenQual', 'FireplaceQu', 'GarageQual', 'GarageCond']
        for col in quality_cols:
            if col in df.columns:
                df[col] = df[col].map(quality_map).fillna(0).astype(int)
        
        # Other ordinal mappings
        if 'BsmtExposure' in df.columns:
            df['BsmtExposure'] = df['BsmtExposure'].map({'Gd': 4, 'Av': 3, 'Mn': 2, 'No': 1, 'NA': 0}).fillna(0)
        if 'BsmtFinType1' in df.columns:
            df['BsmtFinType1'] = df['BsmtFinType1'].map({'GLQ': 6, 'ALQ': 5, 'BLQ': 4, 'Rec': 3, 'LwQ': 2, 'Unf': 1, 'NA': 0}).fillna(0)
        if 'Functional' in df.columns:
            df['Functional'] = df['Functional'].map({'Typ': 8, 'Min1': 7, 'Min2': 6, 'Mod': 5, 'Maj1': 4, 'Maj2': 3, 'Sev': 2, 'Sal': 1}).fillna(8)
        if 'GarageFinish' in df.columns:
            df['GarageFinish'] = df['GarageFinish'].map({'Fin': 3, 'RFn': 2, 'Unf': 1, 'NA': 0}).fillna(0)
        if 'PavedDrive' in df.columns:
            df['PavedDrive'] = df['PavedDrive'].map({'Y': 3, 'P': 2, 'N': 1}).fillna(1)
        if 'CentralAir' in df.columns:
            df['CentralAir'] = df['CentralAir'].map({'Y': 1, 'N': 0}).fillna(0)
        
        # Label encode remaining categoricals
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = df[col].fillna('Missing')
            df[col] = df[col].astype('category').cat.codes
        
        # Fill remaining NaN with 0
        df = df.fillna(0)
        
        # Log transform skewed numeric features
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        skewed_cols = []
        for col in numeric_cols:
            if df[col].nunique() > 10:  # Only transform features with enough variation
                sk = skew(df[col].dropna())
                if abs(sk) > 0.75:
                    skewed_cols.append(col)
        
        for col in skewed_cols:
            df[col] = np.log1p(df[col].clip(lower=0))
        
        # Polynomial features
        if 'OverallQual' in df.columns:
            df['OverallQual_sq'] = df['OverallQual'] ** 2
        if 'TotalSF' in df.columns:
            df['TotalSF_sq'] = df['TotalSF'] ** 2
        
        # Interaction features
        if 'OverallQual' in df.columns and 'GrLivArea' in df.columns:
            df['Qual_x_Area'] = df['OverallQual'] * df['GrLivArea']
        
        return df
    
    train_eng = engineer_house_prices(train, is_train=True)
    test_eng = engineer_house_prices(test, is_train=False)
    
    # Drop non-feature columns
    drop_cols = ['Id', 'SalePrice'] if 'SalePrice' in train_eng.columns else ['Id']
    train_features = train_eng.drop(columns=[c for c in drop_cols if c in train_eng.columns])
    test_features = test_eng.drop(columns=[c for c in ['Id'] if c in test_eng.columns])
    
    # Align columns
    common_cols = list(set(train_features.columns) & set(test_features.columns))
    common_cols.sort()
    train_features = train_features[common_cols]
    test_features = test_features[common_cols]
    
    print(f"  Total features after engineering: {len(common_cols)}")
    
    # Define feature sets
    # Core numeric: top 15 important features
    core_numeric = [c for c in ['OverallQual', 'GrLivArea', 'TotalSF', 'GarageCars', 'GarageArea',
                                 'TotalBsmtSF', '1stFlrSF', 'FullBath', 'TotRmsAbvGrd',
                                 'YearBuilt', 'YearRemodAdd', 'Fireplaces', 'BsmtFinSF1',
                                 'ExterQual', 'KitchenQual'] if c in common_cols]
    
    # Full encoded: all features
    full_encoded = common_cols
    
    feature_sets = {
        "core_numeric": core_numeric,
        "full_encoded": full_encoded,
    }
    
    print(f"  Feature sets: core_numeric={len(core_numeric)}, full_encoded={len(full_encoded)}")
    
    # Run ArenaGenerator
    print("\n  Running ArenaGenerator (time_budget=10 min)...")
    gen = ArenaGenerator(
        X=train_features,
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
    
    # Get winner info
    winner = results[0]
    arena_cv = winner.mean_score
    print(f"\n  Arena Winner: {winner.name}")
    print(f"  CV RMSE: {arena_cv:.5f} ± {winner.std_score:.5f}")
    print(f"  Model Type: {winner.model_type}")
    
    # Hyperparameter tuning
    print("\n  Running ModelOptimizer (30 iterations)...")
    param_grid = build_param_grid_around_winner(winner.model_params, winner.model_type)
    
    if not param_grid:
        param_grid = {'alpha': [0.01, 0.1, 1.0, 5.0, 10.0, 50.0]}
    
    print(f"  Param grid: {param_grid}")
    
    base_model = instantiate_model_from_type(winner.model_type, winner.model_params, "regression")
    
    try:
        optimizer = ModelOptimizer()
        opt_result = optimizer.optimize(
            model=base_model,
            param_grid=param_grid,
            X=train_features[winner.feature_cols],
            y=y,
            method="randomized",
            scoring="rmse",
            cv_folds=5,
            n_iter=30,
            random_state=42,
        )
        tuned_cv = opt_result['best_score']
        tuned_model = opt_result['best_model']
        print(f"  Tuned CV RMSE: {tuned_cv:.5f}")
        print(f"  Best Params: {opt_result['best_params']}")
    except Exception as e:
        print(f"  Optimizer failed: {e}")
        print(f"  Falling back to arena winner.")
        tuned_cv = arena_cv
        tuned_model = instantiate_model_from_type(winner.model_type, winner.model_params, "regression")
    
    # Also try stacking: Ridge + Lasso + arena winner average
    print("\n  Building Stack (Ridge + Lasso + Arena Winner)...")
    
    # Find best Ridge alpha from arena results
    ridge_results = [r for r in results if 'Ridge' in r.model_type]
    lasso_results = [r for r in results if 'Lasso' in r.model_type]
    
    best_ridge_alpha = 10.0  # default
    best_lasso_alpha = 0.001  # default
    if ridge_results:
        best_ridge_alpha = ridge_results[0].model_params.get('alpha', 10.0)
    if lasso_results:
        best_lasso_alpha = lasso_results[0].model_params.get('alpha', 0.001)
    
    # Use full_encoded features for the stack
    stack_features = full_encoded
    
    ridge_model = Ridge(alpha=best_ridge_alpha)
    lasso_model = Lasso(alpha=best_lasso_alpha, max_iter=5000)
    # tuned_model already set above (either from optimizer or fallback)
    
    # Evaluate stack via CV
    from sklearn.model_selection import KFold as KF
    kf = KF(n_splits=5, shuffle=True, random_state=42)
    stack_scores = []
    
    for train_idx, val_idx in kf.split(train_features[stack_features]):
        X_tr = train_features[stack_features].iloc[train_idx]
        X_val = train_features[stack_features].iloc[val_idx]
        y_tr = y.iloc[train_idx]
        y_val = y.iloc[val_idx]
        
        from sklearn.base import clone
        r = clone(ridge_model).fit(X_tr, y_tr)
        l = clone(lasso_model).fit(X_tr, y_tr)
        g = clone(tuned_model).fit(X_tr[winner.feature_cols], y_tr)
        
        pred_r = r.predict(X_val)
        pred_l = l.predict(X_val)
        pred_g = g.predict(X_val[winner.feature_cols])
        
        # Simple average
        pred_stack = (pred_r + pred_l + pred_g) / 3.0
        
        rmse = np.sqrt(np.mean((pred_stack - y_val) ** 2))
        stack_scores.append(rmse)
    
    stack_cv = np.mean(stack_scores)
    print(f"  Stack CV RMSE: {stack_cv:.5f} ± {np.std(stack_scores):.5f}")
    
    # Use whichever is better: tuned model or stack
    use_stack = stack_cv < tuned_cv
    final_cv = min(stack_cv, tuned_cv)
    
    if use_stack:
        print(f"  → Using STACK (better by {tuned_cv - stack_cv:.5f})")
        # Fit on full training data
        ridge_model.fit(train_features[stack_features], y)
        lasso_model.fit(train_features[stack_features], y)
        tuned_model.fit(train_features[winner.feature_cols], y)
        
        pred_r = ridge_model.predict(test_features[stack_features])
        pred_l = lasso_model.predict(test_features[stack_features])
        pred_g = tuned_model.predict(test_features[winner.feature_cols])
        
        predictions_log = (pred_r + pred_l + pred_g) / 3.0
    else:
        print(f"  → Using TUNED model (better by {stack_cv - tuned_cv:.5f})")
        tuned_model.fit(train_features[winner.feature_cols], y)
        predictions_log = tuned_model.predict(test_features[winner.feature_cols])
    
    # Convert back from log scale
    predictions = np.expm1(predictions_log)
    predictions = np.clip(predictions, 0, None)  # No negative prices
    
    submission = pd.DataFrame({
        'Id': test['Id'],
        'SalePrice': predictions
    })
    
    sub_path = BASE_DIR / "house-prices" / "submissions" / "arena_tuned_submission.csv"
    sub_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(sub_path, index=False)
    print(f"  Submission saved: {sub_path}")
    print(f"  Price range: ${predictions.min():.0f} - ${predictions.max():.0f}")
    
    # Submit
    lb_output = submit_to_kaggle("house-prices-advanced-regression-techniques", sub_path,
        f"Arena+{'Stack' if use_stack else 'Tuned'} CV_RMSE={final_cv:.5f}")
    
    RESULTS['house_prices'] = {
        'arena_cv': arena_cv,
        'tuned_cv': tuned_cv,
        'stack_cv': stack_cv,
        'final_cv': final_cv,
        'used_stack': use_stack,
        'winner': winner.name,
        'model_type': winner.model_type,
    }


# ============================================================================
# COMPETITION 3: SPACESHIP TITANIC
# ============================================================================
def run_spaceship_titanic():
    print("\n" + "=" * 70)
    print("COMPETITION 3: SPACESHIP TITANIC (Classification, 8693 rows)")
    print("=" * 70)
    
    # Load data
    train = pd.read_csv(BASE_DIR / "spaceship-titanic" / "data" / "train.csv")
    test = pd.read_csv(BASE_DIR / "spaceship-titanic" / "data" / "test.csv")
    
    print(f"  Train: {train.shape}, Test: {test.shape}")
    
    # Feature Engineering (from Research V2)
    def engineer_spaceship(df):
        df = df.copy()
        
        # CryoSleep (dominant feature)
        df['CryoSleep'] = df['CryoSleep'].fillna(False).astype(int)
        df['VIP'] = df['VIP'].fillna(False).astype(int)
        
        # Spending features
        spending_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
        for col in spending_cols:
            df[col] = df[col].fillna(0)
        
        df['TotalSpending'] = df[spending_cols].sum(axis=1)
        df['LogSpending'] = np.log1p(df['TotalSpending'])
        df['NoSpending'] = (df['TotalSpending'] == 0).astype(int)
        
        # Per-amenity flags
        df['SpentOnRoomService'] = (df['RoomService'] > 0).astype(int)
        df['SpentOnFoodCourt'] = (df['FoodCourt'] > 0).astype(int)
        df['SpentOnShoppingMall'] = (df['ShoppingMall'] > 0).astype(int)
        df['SpentOnSpa'] = (df['Spa'] > 0).astype(int)
        df['SpentOnVRDeck'] = (df['VRDeck'] > 0).astype(int)
        df['NumServicesUsed'] = df[['SpentOnRoomService', 'SpentOnFoodCourt', 
                                     'SpentOnShoppingMall', 'SpentOnSpa', 'SpentOnVRDeck']].sum(axis=1)
        
        # Spending ratios
        total_safe = df['TotalSpending'].clip(lower=1)
        df['RoomServiceRatio'] = df['RoomService'] / total_safe
        df['FoodCourtRatio'] = df['FoodCourt'] / total_safe
        df['SpaRatio'] = df['Spa'] / total_safe
        df['VRDeckRatio'] = df['VRDeck'] / total_safe
        df['ShoppingMallRatio'] = df['ShoppingMall'] / total_safe
        
        # Cabin parsing
        df['Cabin_Deck'] = df['Cabin'].str.split('/').str[0]
        df['Cabin_Num'] = pd.to_numeric(df['Cabin'].str.split('/').str[1], errors='coerce')
        df['Cabin_Side'] = df['Cabin'].str.split('/').str[2]
        
        # Deck encoding
        deck_map = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7, 'T': 8}
        df['DeckNum'] = df['Cabin_Deck'].map(deck_map).fillna(0).astype(int)
        
        # Side encoding (P=Port, S=Starboard)
        df['SideNum'] = df['Cabin_Side'].map({'P': 0, 'S': 1}).fillna(0).astype(int)
        
        # Cabin Region (0-300, 300-600, etc.)
        df['CabinRegion'] = pd.cut(df['Cabin_Num'], bins=[-1, 300, 600, 900, 1200, 1500, 2000],
                                    labels=[0, 1, 2, 3, 4, 5]).astype(float).fillna(0).astype(int)
        
        # Group features from PassengerId
        df['GroupId'] = df['PassengerId'].str.split('_').str[0]
        group_sizes = df.groupby('GroupId')['PassengerId'].transform('count')
        df['GroupSize'] = group_sizes
        df['IsAlone'] = (df['GroupSize'] == 1).astype(int)
        
        # Age features
        df['Age'] = df['Age'].fillna(df['Age'].median())
        df['AgeGroup'] = pd.cut(df['Age'], bins=[0, 5, 12, 18, 25, 40, 60, 100],
                                labels=[0, 1, 2, 3, 4, 5, 6]).astype(float).fillna(3).astype(int)
        df['IsChild'] = (df['Age'] < 12).astype(int)
        df['IsTeenager'] = ((df['Age'] >= 12) & (df['Age'] < 18)).astype(int)
        
        # HomePlanet encoding
        df['HomePlanet'] = df['HomePlanet'].fillna('Unknown')
        planet_map = {'Earth': 0, 'Europa': 1, 'Mars': 2, 'Unknown': 3}
        df['HomePlanetNum'] = df['HomePlanet'].map(planet_map).fillna(3).astype(int)
        
        # Destination encoding
        df['Destination'] = df['Destination'].fillna('Unknown')
        dest_map = {'TRAPPIST-1e': 0, 'PSO J318.5-22': 1, '55 Cancri e': 2, 'Unknown': 3}
        df['DestinationNum'] = df['Destination'].map(dest_map).fillna(3).astype(int)
        
        # Interactions
        df['EarthCryo'] = ((df['HomePlanet'] == 'Earth') & (df['CryoSleep'] == 1)).astype(int)
        df['EuropaCryo'] = ((df['HomePlanet'] == 'Europa') & (df['CryoSleep'] == 1)).astype(int)
        df['MarsCryo'] = ((df['HomePlanet'] == 'Mars') & (df['CryoSleep'] == 1)).astype(int)
        
        return df
    
    train = engineer_spaceship(train)
    test = engineer_spaceship(test)
    
    y = train['Transported'].astype(int)
    
    # Define feature sets
    core_features = ['CryoSleep', 'TotalSpending', 'LogSpending', 'NoSpending',
                     'DeckNum', 'SideNum', 'GroupSize', 'Age', 'HomePlanetNum',
                     'DestinationNum', 'NumServicesUsed']
    
    full_v2_features = core_features + [
        'VIP', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck',
        'SpentOnRoomService', 'SpentOnFoodCourt', 'SpentOnShoppingMall', 'SpentOnSpa', 'SpentOnVRDeck',
        'RoomServiceRatio', 'FoodCourtRatio', 'SpaRatio', 'VRDeckRatio', 'ShoppingMallRatio',
        'CabinRegion', 'IsAlone', 'AgeGroup', 'IsChild', 'IsTeenager',
        'EarthCryo', 'EuropaCryo', 'MarsCryo',
        'Cabin_Num',
    ]
    
    # Filter to only columns that exist
    full_v2_features = [c for c in full_v2_features if c in train.columns]
    
    feature_sets = {
        "core": core_features,
        "full_v2": full_v2_features,
    }
    
    print(f"  Feature sets: core={len(core_features)}, full_v2={len(full_v2_features)}")
    
    # For ArenaGenerator, we need to use standard models (it doesn't support HGB natively)
    # We'll run the arena AND separately test HGB
    print("\n  Running ArenaGenerator (time_budget=10 min)...")
    gen = ArenaGenerator(
        X=train[full_v2_features],
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
    
    # Also test HGB directly (arena doesn't include it)
    print("\n  Testing HistGradientBoosting separately...")
    hgb_configs = [
        {'max_iter': 200, 'max_depth': 4, 'learning_rate': 0.1, 'min_samples_leaf': 20},
        {'max_iter': 300, 'max_depth': 5, 'learning_rate': 0.05, 'min_samples_leaf': 10},
        {'max_iter': 400, 'max_depth': 5, 'learning_rate': 0.05, 'min_samples_leaf': 15},
        {'max_iter': 300, 'max_depth': 6, 'learning_rate': 0.08, 'min_samples_leaf': 10},
    ]
    
    best_hgb_score = 0
    best_hgb_params = {}
    best_hgb_features = full_v2_features
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    for i, params in enumerate(hgb_configs):
        hgb = HistGradientBoostingClassifier(random_state=42, **params)
        scores = cross_val_score(hgb, train[full_v2_features], y, cv=skf, scoring='accuracy')
        mean_score = scores.mean()
        print(f"    HGB config {i+1}: {mean_score:.4f} ± {scores.std():.4f} | {params}")
        if mean_score > best_hgb_score:
            best_hgb_score = mean_score
            best_hgb_params = params
    
    # Compare arena winner vs HGB
    arena_winner = results[0]
    arena_cv = arena_winner.mean_score
    
    print(f"\n  Arena Winner: {arena_winner.name} CV={arena_cv:.4f}")
    print(f"  Best HGB: CV={best_hgb_score:.4f}")
    
    # Use the better one
    if best_hgb_score > arena_cv:
        print(f"  → HGB wins (by {best_hgb_score - arena_cv:.4f})")
        use_hgb = True
        winner_cv = best_hgb_score
        winner_features = full_v2_features
        winner_params = best_hgb_params
        winner_model_type = 'HistGradientBoostingClassifier'
    else:
        print(f"  → Arena winner wins (by {arena_cv - best_hgb_score:.4f})")
        use_hgb = False
        winner_cv = arena_cv
        winner_features = arena_winner.feature_cols
        winner_params = arena_winner.model_params
        winner_model_type = arena_winner.model_type
    
    # Hyperparameter tuning on the winner
    print("\n  Running ModelOptimizer (30 iterations)...")
    
    if use_hgb:
        # Build HGB-specific grid — stay close to the winning config
        param_grid = {
            'max_iter': [best_hgb_params['max_iter'] - 50,
                        best_hgb_params['max_iter'],
                        best_hgb_params['max_iter'] + 50,
                        best_hgb_params['max_iter'] + 100],
            'max_depth': [max(3, best_hgb_params['max_depth'] - 1),
                         best_hgb_params['max_depth'],
                         best_hgb_params['max_depth'] + 1],
            'learning_rate': [max(0.01, best_hgb_params['learning_rate'] * 0.7),
                            best_hgb_params['learning_rate'],
                            min(0.15, best_hgb_params['learning_rate'] * 1.4),
                            min(0.2, best_hgb_params['learning_rate'] * 2)],
            'min_samples_leaf': [max(5, best_hgb_params['min_samples_leaf'] - 5),
                                best_hgb_params['min_samples_leaf'],
                                best_hgb_params['min_samples_leaf'] + 5,
                                best_hgb_params['min_samples_leaf'] + 10],
        }
        base_model = HistGradientBoostingClassifier(random_state=42)
    else:
        param_grid = build_param_grid_around_winner(winner_params, winner_model_type)
        if not param_grid:
            param_grid = {'n_estimators': [100, 200, 300], 'max_depth': [3, 4, 5, 6]}
        base_model = instantiate_model_from_type(winner_model_type, winner_params, "classification")
    
    print(f"  Param grid: {param_grid}")
    
    optimizer = ModelOptimizer()
    opt_result = optimizer.optimize(
        model=base_model,
        param_grid=param_grid,
        X=train[winner_features],
        y=y,
        method="randomized",
        scoring="accuracy",
        cv_folds=5,
        n_iter=30,
        random_state=42,
    )
    
    tuned_cv = opt_result['best_score']
    print(f"  Tuned CV: {tuned_cv:.4f}")
    print(f"  Best Params: {opt_result['best_params']}")
    
    # If tuned is worse than pre-tune winner, use the pre-tune winner instead
    if tuned_cv < winner_cv:
        print(f"  WARNING: Tuned CV ({tuned_cv:.4f}) < Pre-tune ({winner_cv:.4f}). Using pre-tune model.")
        if use_hgb:
            best_model = HistGradientBoostingClassifier(random_state=42, **best_hgb_params)
        else:
            best_model = instantiate_model_from_type(winner_model_type, winner_params, "classification")
        tuned_cv = winner_cv
    else:
        best_model = opt_result['best_model']
    
    # Generate submission
    best_model.fit(train[winner_features], y)
    predictions = best_model.predict(test[winner_features])
    
    # IMPORTANT: Kaggle expects True/False in Transported column
    submission = pd.DataFrame({
        'PassengerId': test['PassengerId'],
        'Transported': predictions.astype(bool)
    })
    
    sub_path = BASE_DIR / "spaceship-titanic" / "submissions" / "arena_tuned_submission.csv"
    sub_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(sub_path, index=False)
    print(f"  Submission saved: {sub_path}")
    print(f"  Predictions distribution: {pd.Series(predictions).value_counts().to_dict()}")
    
    # Submit
    lb_output = submit_to_kaggle("spaceship-titanic", sub_path,
        f"Arena+Tuned {winner_model_type} CV={tuned_cv:.4f}")
    
    RESULTS['spaceship_titanic'] = {
        'arena_cv': winner_cv,
        'tuned_cv': tuned_cv,
        'winner': f"HGB({best_hgb_params})" if use_hgb else arena_winner.name,
        'model_type': winner_model_type,
    }


# ============================================================================
# MAIN EXECUTION
# ============================================================================
if __name__ == "__main__":
    start_time = time.time()
    
    print("=" * 70)
    print("FULL ARENA + HYPERPARAMETER TUNING + SUBMISSION PIPELINE")
    print("=" * 70)
    print(f"Start time: {time.strftime('%H:%M:%S')}")
    print(f"Competitions: Titanic, House Prices, Spaceship Titanic")
    print(f"Budget per competition: 10 min arena + ~5 min tuning")
    print()
    
    # Run all three competitions
    try:
        run_titanic()
    except Exception as e:
        print(f"\n  ERROR in Titanic: {e}")
        import traceback
        traceback.print_exc()
        RESULTS['titanic'] = {'arena_cv': 'ERROR', 'tuned_cv': 'ERROR', 'winner': str(e), 'model_type': ''}
    
    try:
        run_house_prices()
    except Exception as e:
        print(f"\n  ERROR in House Prices: {e}")
        import traceback
        traceback.print_exc()
        RESULTS['house_prices'] = {'arena_cv': 'ERROR', 'tuned_cv': 'ERROR', 'winner': str(e), 'model_type': ''}
    
    try:
        run_spaceship_titanic()
    except Exception as e:
        print(f"\n  ERROR in Spaceship Titanic: {e}")
        import traceback
        traceback.print_exc()
        RESULTS['spaceship_titanic'] = {'arena_cv': 'ERROR', 'tuned_cv': 'ERROR', 'winner': str(e), 'model_type': ''}
    
    # Final Summary
    elapsed = time.time() - start_time
    print("\n\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    print(f"Total runtime: {elapsed/60:.1f} minutes\n")
    
    print(f"{'Competition':<22} | {'Previous Best':<13} | {'Arena Winner CV':<15} | {'Tuned CV':<10} | {'Model Type'}")
    print("-" * 90)
    
    # Titanic
    t = RESULTS.get('titanic', {})
    t_arena = f"{t.get('arena_cv', '?'):.4f}" if isinstance(t.get('arena_cv'), float) else str(t.get('arena_cv', '?'))
    t_tuned = f"{t.get('tuned_cv', '?'):.4f}" if isinstance(t.get('tuned_cv'), float) else str(t.get('tuned_cv', '?'))
    print(f"{'Titanic':<22} | {'0.77272':<13} | {t_arena:<15} | {t_tuned:<10} | {t.get('model_type', '?')}")
    
    # House Prices
    h = RESULTS.get('house_prices', {})
    h_arena = f"{h.get('arena_cv', '?'):.5f}" if isinstance(h.get('arena_cv'), float) else str(h.get('arena_cv', '?'))
    h_tuned = f"{h.get('final_cv', h.get('tuned_cv', '?')):.5f}" if isinstance(h.get('final_cv', h.get('tuned_cv')), float) else str(h.get('tuned_cv', '?'))
    stack_note = " (stack)" if h.get('used_stack') else ""
    print(f"{'House Prices':<22} | {'0.12728':<13} | {h_arena:<15} | {h_tuned + stack_note:<10} | {h.get('model_type', '?')}")
    
    # Spaceship
    s = RESULTS.get('spaceship_titanic', {})
    s_arena = f"{s.get('arena_cv', '?'):.4f}" if isinstance(s.get('arena_cv'), float) else str(s.get('arena_cv', '?'))
    s_tuned = f"{s.get('tuned_cv', '?'):.4f}" if isinstance(s.get('tuned_cv'), float) else str(s.get('tuned_cv', '?'))
    print(f"{'Spaceship Titanic':<22} | {'0.80243':<13} | {s_arena:<15} | {s_tuned:<10} | {s.get('model_type', '?')}")
    
    print("\n" + "=" * 70)
    print("Pipeline complete!")
    print("=" * 70)
