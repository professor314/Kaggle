# Kaggle ML Toolkit

**An AI-assisted research and learning platform for Kaggle competitions**

<!-- Badges placeholder -->
![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

---

The Kaggle ML Toolkit is a structured workflow where a human collaborator works with Kiro (AI agent) to make informed decisions at every stage of a machine learning competition — from domain research through modeling to educational content creation.

The platform is built on three core principles:

- **Research-first**: Domain knowledge collection and hypothesis formation precede any modeling. Every competition starts with a structured research document before a single model is trained.
- **Decision-driven**: Kiro presents distilled information and clear options; the user makes strategic choices. You focus on *what* to do, not *how* to implement it.
- **Educational output**: Every project produces publishable content — blog posts, writeups, and articles — explaining the full process so others can learn from your approach.

## Prerequisites

- **Python 3.10+**
- **pip** (package installer)
- **git** (version control)
- **Kaggle account** (required for CLI data downloads and submissions)

## Installation

Clone the repository and install in development mode:

```bash
git clone <repository-url>
cd Kaggle
```

Install the package (editable/development mode):

```bash
pip install -e .
```

Install with development dependencies (testing, linting):

```bash
pip install -e ".[dev]"
```

## Quick Start

```python
from kaggle_ml_toolkit import CompetitionConfig, DataCleaner, ModelSelector

# Load competition configuration
config = CompetitionConfig.from_yaml("competitions/titanic/competition_config.yaml")

# Clean and prepare data
cleaner = DataCleaner()
df_clean = cleaner.impute_numeric(df, columns=["Age", "Fare"], strategy="median")
df_encoded = cleaner.encode(df_clean, columns=["Sex", "Embarked"], method="onehot")

# Compare models automatically
selector = ModelSelector()
results = selector.compare(X, y, problem_type="classification", metric="accuracy")
print(results)  # Ranked DataFrame of model performances
```

## Project Structure

```
kaggle/
├── .kiro/
│   ├── steering/              # Kiro decision workflow steering files
│   │   ├── 01-research.md
│   │   ├── 02-eda.md
│   │   ├── 03-feature-engineering.md
│   │   ├── 04-model-selection.md
│   │   ├── 05-optimization.md
│   │   ├── 06-ensemble.md
│   │   ├── 07-submission.md
│   │   └── 08-content-creation.md
│   └── specs/
├── kaggle_ml_toolkit/         # Core Python package
│   ├── __init__.py
│   ├── loader.py              # Data loading utilities
│   ├── cleaner.py             # Data cleaning and preprocessing
│   ├── feature_engineer.py    # Feature creation and transformation
│   ├── feature_selector.py    # Domain-informed feature selection
│   ├── eda_engine.py          # Automated exploratory data analysis
│   ├── model_selector.py      # Multi-model comparison
│   ├── model_optimizer.py     # Hyperparameter tuning
│   ├── cross_validator.py     # Advanced CV strategies
│   ├── ensemble_builder.py    # Ensemble construction
│   ├── evaluator.py           # Metric computation and persistence
│   ├── interpreter.py         # Model interpretability (SHAP, PDP)
│   ├── augmenter.py           # Data augmentation
│   ├── submission_generator.py# Kaggle submission file generation
│   ├── content_generator.py   # Educational content generation
│   ├── research.py            # Research document management
│   ├── config.py              # Competition configuration
│   ├── pipeline.py            # Pipeline orchestration
│   └── utils.py               # Shared utilities
├── docs/
│   ├── kaggle_guide.md        # Kaggle platform reference
│   └── blog/                  # Generated educational articles
├── competitions/
│   └── titanic/               # Example: Titanic competition
│       ├── notebooks/
│       ├── data/
│       ├── submissions/
│       ├── research/
│       └── content/
├── tests/
│   ├── unit/
│   ├── properties/
│   ├── integration/
│   └── conftest.py
├── pyproject.toml
├── README.md
├── ROADMAP.md
├── CONTRIBUTING.md
└── .gitignore
```

## Kaggle CLI Setup

The toolkit integrates with the Kaggle CLI for downloading competition data and submitting predictions. For a full guide, see [`docs/kaggle_guide.md`](docs/kaggle_guide.md).

**Quick setup:**

1. Install the Kaggle CLI:
   ```bash
   pip install kaggle
   ```

2. Create an API token at [kaggle.com/settings](https://www.kaggle.com/settings) → "Create New Token"

3. Place the downloaded `kaggle.json` file at:
   - **Linux/macOS**: `~/.kaggle/kaggle.json`
   - **Windows**: `C:\Users\<username>\.kaggle\kaggle.json`

4. Set permissions (Linux/macOS):
   ```bash
   chmod 600 ~/.kaggle/kaggle.json
   ```

5. Verify installation:
   ```bash
   kaggle competitions list
   ```

## Compute Configuration (CPU cores / GPU)

The toolkit auto-detects your hardware and uses all CPU cores (and the GPU, if a
CUDA-enabled PyTorch is installed) for the gradient boosters and cross-validation.
You don't need to configure anything to get fast runs.

To tune it for your own machine, override without editing code, either way works
(environment variables win over the file):

**Environment variables**

```bash
KAGGLE_TOOLKIT_CORES=8     # cap worker threads (default: all logical cores)
KAGGLE_TOOLKIT_USE_GPU=0   # 0/false = force CPU, 1/true = use GPU, unset = auto
```

**Config file** — copy `compute.example.yaml` to `compute.yaml` (repo root or
`~/.kaggle_toolkit/compute.yaml`):

```yaml
cores: 8          # or null for all cores
use_gpu: false    # or null to auto-detect
```

Then in code:

```python
from kaggle_ml_toolkit import compute
import lightgbm as lgb

print(compute.summary())              # e.g. "cores=8 (capped) | off (config)"
model = lgb.LGBMClassifier(**compute.lgbm_params(), n_estimators=1500)
# compute.xgb_params() and compute.catboost_params() do the same for XGB/CatBoost
```

`compute.yaml` is gitignored (it's machine-specific); commit only the example.

## How to Contribute

We welcome contributions! See [`CONTRIBUTING.md`](CONTRIBUTING.md) for guidelines on:

- Picking up planned features from the roadmap
- Code style and testing requirements
- Submitting pull requests
- Commit message conventions

## Phased Development

The toolkit is built incrementally. See [`ROADMAP.md`](ROADMAP.md) for full details.

| Phase | Focus | Status |
|-------|-------|--------|
| **Phase 1** | Core pipeline (load, clean, feature eng, model select, evaluate, submit), domain research, EDA, educational content generation | 🔄 In Progress |
| **Phase 2** | Advanced ensembles, model interpretability (SHAP), data augmentation, advanced CV, Kaggle CLI automation, Code Competition support | 📋 Planned |
| **Phase 3** | Deep learning (PyTorch/TensorFlow), cloud scaling (AWS CLI), advanced experiment tracking | 📋 Planned |

## License

MIT License — see [LICENSE](LICENSE) for details.
