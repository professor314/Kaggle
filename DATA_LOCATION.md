# Where the big data lives (not in this repo)

The large, regenerable data for these competitions is **NOT** in git or Google
Drive. It lives on the local disk only:

**`C:\Users\profe\kaggle-data\`**

The folder structure there **mirrors this repo's layout**, so a path like
`competitions/rsna-knee/weights/` in the code maps to
`C:\Users\profe\kaggle-data\competitions\rsna-knee\weights\`.

## What's there (~3.6 GB)
- `competitions/*/data/` — downloaded datasets (also re-downloadable via `kaggle competitions download`)
- `competitions/*/weights/`, `nli_model/`, `_nli_ds/`, `contradictory-watson/artifacts/` — model weights / trained artifacts
- `competitions/kaggriculture/replays/` — downloaded top-player replays (~926 MB)
- `competitions/*/submissions/`, `artifacts*/`, `lb/`, `lb2/`, `evo/state_tourney/`

## When you return to a project
The code still uses the original relative paths (e.g. `competitions/rsna-knee/weights/`).
Point it at the data one of two ways:
1. Set a base-data-dir variable / env var to `C:\Users\profe\kaggle-data` and join
   the relative path, **or**
2. Make a directory symlink so the repo path resolves to the data folder, e.g.:
   `New-Item -ItemType Junction -Path competitions\rsna-knee\weights -Target C:\Users\profe\kaggle-data\competitions\rsna-knee\weights`

## The Python env
`.venv-dl/` was **discarded** (regenerable). Recreate with:
`python -m venv .venv-dl` then `pip install -r requirements.txt` (+ `kaggle-environments` for kaggriculture).
