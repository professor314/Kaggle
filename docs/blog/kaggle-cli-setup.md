# Kaggle CLI Setup Guide

**Published:** 2025-01-20

---

## Installation

Install the Kaggle CLI via pip:

```bash
pip install kaggle
```

This gives you the `kaggle` command for downloading data, submitting predictions, and browsing competitions from the terminal.

## API Credential Configuration

1. Log in to [kaggle.com/settings](https://www.kaggle.com/settings)
2. Scroll to the **API** section and click **Create New Token**
3. A `kaggle.json` file downloads — this contains your credentials
4. Move it to the expected location:
   - **Linux/macOS:** `~/.kaggle/kaggle.json`
   - **Windows:** `C:\Users\<username>\.kaggle\kaggle.json`
5. On Linux/macOS, restrict permissions: `chmod 600 ~/.kaggle/kaggle.json`

## Common Commands

| Command | Purpose |
|---------|---------|
| `kaggle competitions list` | Browse active competitions |
| `kaggle competitions download -c titanic` | Download competition data |
| `kaggle competitions submit -c titanic -f submission.csv -m "message"` | Submit predictions |
| `kaggle competitions submissions -c titanic` | View your submission history |
| `kaggle datasets list -s "keyword"` | Search public datasets |

## Troubleshooting

- **"Could not find kaggle.json"** — Ensure the file is in `~/.kaggle/` (or `%USERPROFILE%\.kaggle\` on Windows).
- **"403 Forbidden"** — You may need to accept the competition rules on the Kaggle website before downloading data.
- **"401 Unauthorized"** — Regenerate your API token; the old one may have expired.
- **Timeout errors** — Check your internet connection. The Kaggle API occasionally experiences slowdowns during peak hours.

Run `python scripts/verify_kaggle_cli.py` to automatically check your setup.
