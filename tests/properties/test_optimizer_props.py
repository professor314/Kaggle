"""Property-based tests for ModelOptimizer.

**Validates: Requirements 12.1, 12.3**
"""

from hypothesis import given, settings, strategies as st
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier

from kaggle_ml_toolkit.model_optimizer import ModelOptimizer


@st.composite
def simple_param_grids(draw):
    """Generate simple param grids with known value lists for RandomForestClassifier."""
    n_estimators_values = draw(
        st.lists(
            st.sampled_from([5, 10, 15, 20, 25]),
            min_size=1,
            max_size=3,
            unique=True,
        )
    )
    max_depth_values = draw(
        st.lists(
            st.sampled_from([2, 3, 4, 5]),
            min_size=1,
            max_size=2,
            unique=True,
        )
    )
    return {
        "n_estimators": n_estimators_values,
        "max_depth": max_depth_values,
    }


@settings(max_examples=10)
@given(param_grid=simple_param_grids())
def test_optimizer_best_params_from_grid(param_grid):
    """Property 24: Optimizer Best Params from Grid.

    For any successful optimization, best_params values must be from
    the original param_grid.

    **Validates: Requirements 12.1, 12.3**
    """
    X, y = make_classification(
        n_samples=50,
        n_features=5,
        n_informative=3,
        random_state=42,
    )

    optimizer = ModelOptimizer()
    result = optimizer.optimize(
        model=RandomForestClassifier(random_state=42),
        param_grid=param_grid,
        X=X,
        y=y,
        method="grid",
        scoring="accuracy",
        cv_folds=2,
        random_state=42,
    )

    best_params = result["best_params"]

    # Every returned best param value must exist in the original grid
    for param_name, param_value in best_params.items():
        assert param_name in param_grid, (
            f"best_params contains unknown param '{param_name}'"
        )
        assert param_value in param_grid[param_name], (
            f"best_params['{param_name}'] = {param_value} not in grid {param_grid[param_name]}"
        )
