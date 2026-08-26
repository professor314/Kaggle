# Contributing to Kaggle ML Toolkit

Welcome! We're glad you're interested in contributing to the Kaggle ML Toolkit. Whether you're fixing a bug, adding a feature, improving documentation, or writing educational content, your contributions help make this project better for the entire Kaggle community. Every contribution matters, and we appreciate your time.

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/<your-username>/kaggle-ml-toolkit.git
   cd kaggle-ml-toolkit
   ```
3. **Install in development mode** with dev dependencies:
   ```bash
   pip install -e ".[dev]"
   ```
4. **Run the test suite** to verify everything works:
   ```bash
   pytest
   ```

## Finding Work

- Check [ROADMAP.md](ROADMAP.md) for planned features and upcoming milestones
- Look for GitHub Issues tagged **"good first issue"** or **"help wanted"**
- Phase 2 and Phase 3 features are open for contribution — pick one that interests you
- If you have an idea not on the roadmap, open an issue to discuss it first

## Development Workflow

1. **Create a feature branch** from `main`:
   ```bash
   git checkout -b feat/your-feature-name main
   ```
2. **Write your code** following the project style (see Code Style below)
   - Use type hints on all public methods
   - Write docstrings in Google format
   - Keep DataFrames immutable — transformation methods return new DataFrames
3. **Write tests** for your changes:
   - Unit tests for all new code
   - Property-based tests using Hypothesis where applicable
4. **Ensure all tests pass**:
   ```bash
   pytest
   ```
5. **Submit a Pull Request** with a descriptive title and summary explaining what changed and why

## Code Style

- **Type hints** on all public methods and functions
- **Docstrings** in [Google format](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings)
- **Immutable DataFrames**: All transformation methods return new DataFrames (never mutate input)
- **Maximum line length**: 100 characters
- **String formatting**: Use f-strings for all string formatting

Example:

```python
def detect_outliers(
    df: pd.DataFrame,
    column: str,
    method: str = "iqr",
    threshold: float = 1.5,
) -> dict[str, Any]:
    """Detect outliers in a DataFrame column.

    Args:
        df: Input DataFrame to analyze.
        column: Name of the column to check for outliers.
        method: Detection method, either "iqr" or "zscore".
        threshold: Sensitivity threshold for detection.

    Returns:
        A dictionary containing outlier indices, count, and column name.

    Raises:
        ValueError: If the column contains only missing values.
    """
    ...
```

## Testing Requirements

- All new modules must have unit tests in `tests/unit/`
- Core modules should have property-based tests in `tests/properties/` using [Hypothesis](https://hypothesis.readthedocs.io/)
- Minimum **100 examples** per property test:
  ```python
  from hypothesis import given, settings
  import hypothesis.strategies as st

  @settings(max_examples=100)
  @given(st.floats(allow_nan=False, allow_infinity=False))
  def test_some_property(value: float) -> None:
      ...
  ```
- All tests must pass before a PR can be merged

## Commit Messages

Use the following format for commit messages:

```
type(scope): description
```

**Types:**
| Type       | Usage                                  |
|------------|----------------------------------------|
| `feat`     | A new feature                          |
| `fix`      | A bug fix                              |
| `docs`     | Documentation changes only             |
| `test`     | Adding or updating tests               |
| `refactor` | Code change that neither fixes nor adds|
| `chore`    | Maintenance tasks, dependency updates  |

**Examples:**
```
feat(cleaner): add z-score outlier detection
fix(loader): handle empty CSV files gracefully
docs(blog): add Titanic EDA walkthrough article
test(evaluator): add property tests for metric persistence
refactor(selector): extract ranking logic into helper
chore(deps): update hypothesis to 6.100
```

## Blog Content

Educational articles are a core part of this project. Contributions to documentation and blog content are highly valued.

- Articles are welcome in `docs/blog/`
- Follow the **dual-save pattern**: save competition-specific content in the competition's content directory as well as in `docs/blog/`
- Update `docs/blog/README.md` index when adding new articles
- Write for a broad audience — explain the "why" alongside the "how"
- Include visualizations, code snippets, and references to source data where relevant
