#!/usr/bin/env python
"""Verify Kaggle CLI installation and connectivity.

Checks:
1. kaggle command is available on PATH
2. ~/.kaggle/kaggle.json credentials file exists
3. Kaggle API connectivity (list competitions)

Usage:
    python scripts/verify_kaggle_cli.py
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path


def check_kaggle_command() -> bool:
    """Check if the kaggle CLI command is available."""
    kaggle_path = shutil.which("kaggle")
    if kaggle_path:
        print(f"[PASS] kaggle CLI found at: {kaggle_path}")
        return True
    else:
        print("[FAIL] kaggle CLI not found on PATH")
        print("  Remediation: Install the Kaggle CLI with:")
        print("    pip install kaggle")
        return False


def check_credentials_file() -> bool:
    """Check if ~/.kaggle/kaggle.json exists."""
    # Support both Linux/macOS and Windows credential paths
    if sys.platform == "win32":
        kaggle_dir = Path(os.environ.get("USERPROFILE", "")) / ".kaggle"
    else:
        kaggle_dir = Path.home() / ".kaggle"

    creds_file = kaggle_dir / "kaggle.json"

    if creds_file.exists():
        print(f"[PASS] Credentials file found: {creds_file}")
        return True
    else:
        print(f"[FAIL] Credentials file not found: {creds_file}")
        print("  Remediation:")
        print("    1. Go to https://www.kaggle.com/settings")
        print("    2. Click 'Create New Token' under the API section")
        print(f"    3. Save the downloaded kaggle.json to: {kaggle_dir}/")
        if sys.platform != "win32":
            print("    4. Set permissions: chmod 600 ~/.kaggle/kaggle.json")
        return False


def check_connectivity() -> bool:
    """Test Kaggle API connectivity by listing competitions."""
    try:
        result = subprocess.run(
            ["kaggle", "competitions", "list", "--sort-by", "latestDeadline"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0:
            print("[PASS] Kaggle API connectivity verified (competitions list succeeded)")
            return True
        else:
            print("[FAIL] Kaggle API call failed")
            print(f"  Error: {result.stderr.strip()}")
            print("  Remediation:")
            print("    - Verify your kaggle.json contains valid credentials")
            print("    - Check your internet connection")
            print("    - Ensure your Kaggle API token has not expired")
            return False
    except FileNotFoundError:
        print("[FAIL] Could not run kaggle command (not installed)")
        print("  Remediation: pip install kaggle")
        return False
    except subprocess.TimeoutExpired:
        print("[FAIL] Kaggle API call timed out (>30s)")
        print("  Remediation: Check your internet connection")
        return False


def main():
    """Run all Kaggle CLI verification checks."""
    print("=" * 60)
    print("Kaggle CLI Verification")
    print("=" * 60)
    print()

    results = []

    print("1. Checking kaggle command availability...")
    results.append(check_kaggle_command())
    print()

    print("2. Checking credentials file...")
    results.append(check_credentials_file())
    print()

    print("3. Checking API connectivity...")
    # Only test connectivity if the command and credentials exist
    if all(results):
        results.append(check_connectivity())
    else:
        print("[SKIP] Skipping connectivity test (prerequisites not met)")
        results.append(False)
    print()

    # Summary
    print("=" * 60)
    passed = sum(results)
    total = len(results)
    if all(results):
        print(f"Result: ALL CHECKS PASSED ({passed}/{total})")
        print("Your Kaggle CLI is ready to use!")
    else:
        print(f"Result: {passed}/{total} checks passed")
        print("Please address the issues above before using Kaggle CLI features.")
    print("=" * 60)

    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
