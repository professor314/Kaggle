"""Kiro hook: keep Kaggle work tied to the toolkit, steering docs, git, and per-competition READMEs.

Kiro runs this script at two points (configured in ~/.kiro/agents/kaggle.json):

  userPromptSubmit  When Sean sends a message. If it looks like Kaggle work, we print a short
                    briefing. Kiro adds whatever we print to STDOUT into the assistant's context,
                    so this is how the workflow rules, the toolkit's module list and the repo's
                    git state reach the assistant every turn, even when the session was started
                    outside the Kaggle repo (where the repo's own steering docs don't auto-load).

  stop              When the assistant finishes a turn. If Kaggle work happened and the repo has
                    uncommitted/unpushed changes or a recently-touched competition has no README.md, we print a reminder
                    to STDERR and exit 1. Kiro shows a failing hook's STDERR to the user as a
                    warning, so neither of us can quietly skip the end-of-session checklist.

Kiro passes the event as JSON on STDIN, e.g.
    {"hook_event_name": "userPromptSubmit", "cwd": "...", "prompt": "..."}
    {"hook_event_name": "stop", "cwd": "...", "assistant_response": "..."}

Design choices:
  * Standard library only, so it runs under any Python without installing anything.
  * Never crash the chat: any unexpected error exits 0 (silent) instead of breaking a turn.
  * Fast: hooks have a timeout (we set 20 s), so we only run cheap git commands and read
    module docstrings with `ast` instead of importing the toolkit (which pulls in pandas,
    lightgbm, torch... far too slow for a hook).
"""
from __future__ import annotations

import ast
import datetime as dt
import json
import re
import subprocess
import sys
from pathlib import Path

# Where the platform lives. Hard-coded on purpose: the hook must find the repo even when the
# chat was started in some other folder (that was exactly the "vacuum" problem).
REPO = Path.home() / "My Drive" / "Programming" / "kaggle"
TOOLKIT = REPO / "kaggle_ml_toolkit"
STEERING = REPO / ".kiro" / "steering"
BLOG = REPO / "docs" / "blog"

# Words that mean "this turn is about Kaggle". Matched case-insensitively as whole words.
KAGGLE_WORDS = r"kaggle|competition|leaderboard|submission|submit|kernel|playground|toolkit|" \
               r"s6e\d+|lightgbm|xgboost|catboost|oof|cv score|public lb"


def is_kaggle(text: str, cwd: str) -> bool:
    """True if the text mentions Kaggle work or the chat is running inside the Kaggle repo."""
    in_repo = str(REPO).lower() in (cwd or "").lower()
    return in_repo or bool(re.search(rf"\b({KAGGLE_WORDS})\b", text or "", re.IGNORECASE))


def git(*args: str) -> str:
    """Run a git command in the repo and return its output ('' on any failure)."""
    try:
        out = subprocess.run(["git", "-C", str(REPO), *args], capture_output=True,
                             text=True, timeout=8)
        return out.stdout.strip()
    except Exception:
        return ""


def git_state() -> tuple[int, int]:
    """(number of changed/untracked paths, number of commits not yet pushed)."""
    dirty = len([l for l in git("status", "--porcelain").splitlines() if l.strip()])
    ahead = git("rev-list", "--count", "@{u}..HEAD")  # '' if no upstream is configured
    return dirty, int(ahead) if ahead.isdigit() else 0


def toolkit_index() -> list[str]:
    """One line per toolkit module: name + first line of its docstring.

    Parsed with `ast` (reads the file as text) rather than imported, so it is instant and
    cannot fail because a heavy dependency is missing.
    """
    lines = []
    for py in sorted(TOOLKIT.glob("*.py")):
        if py.name.startswith("_"):
            continue
        try:
            doc = ast.get_docstring(ast.parse(py.read_text(encoding="utf-8"))) or ""
        except Exception:
            doc = ""
        first = doc.strip().splitlines()[0] if doc.strip() else "(no docstring - add one)"
        lines.append(f"  - {py.stem}: {first[:110]}")
    return lines


def readmes_missing() -> list[str]:
    """Competitions touched in the last 2 days (experiments_log.md) that lack a README.md.

    The per-competition README replaces the old daily blog draft: it records the design
    decisions (see 00-kaggle-workflow.md, "README standard").
    """
    import time
    missing = []
    for log in (REPO / "competitions").glob("*/experiments_log.md"):
        recent = time.time() - log.stat().st_mtime < 2 * 86400
        if recent and not (log.parent / "README.md").exists():
            missing.append(log.parent.name)
    return missing


def on_prompt(event: dict) -> None:
    """Print the briefing that gets injected into the assistant's context."""
    if not is_kaggle(event.get("prompt", ""), event.get("cwd", "")):
        return  # not Kaggle work: stay silent so other projects aren't cluttered
    dirty, ahead = git_state()
    missing = readmes_missing()
    print("KAGGLE WORKFLOW HOOK (from ~/.kiro/agents/kaggle.json)")
    print(f"Repo: {REPO}")
    print(f"MUST READ before acting: {STEERING / '00-kaggle-workflow.md'} and the phase doc(s)")
    print("  for this step (01-research .. 08-content-creation), plus docs/LESSONS.md and the")
    print("  competition's experiments_log.md. Use/extend the toolkit before writing new code.")
    print(f"Git: {dirty} uncommitted path(s), {ahead} unpushed commit(s).")
    print("Competition README missing for: " + (", ".join(missing) if missing else "none") + "  (README standard in 00-kaggle-workflow.md)")
    print("Toolkit modules (kaggle_ml_toolkit):")
    print("\n".join(toolkit_index()))


def on_stop(event: dict) -> None:
    """Nag (via a warning) if Kaggle work ended without git + README housekeeping."""
    if not is_kaggle(event.get("assistant_response", ""), event.get("cwd", "")):
        return
    dirty, ahead = git_state()
    problems = []
    if dirty:
        problems.append(f"{dirty} uncommitted change(s) in the Kaggle repo")
    if ahead:
        problems.append(f"{ahead} commit(s) not pushed to GitHub")
    if readmes_missing():
        problems.append("README.md missing for: " + ", ".join(readmes_missing()))
    if problems:
        print("Kaggle checklist: " + "; ".join(problems) + ".", file=sys.stderr)
        sys.exit(1)  # non-zero = Kiro shows the STDERR text to Sean as a warning


def main() -> None:
    # Windows consoles default to cp1252, and toolkit docstrings contain characters like
    # arrows and dashes. Printing those would raise UnicodeEncodeError and (because of the
    # catch-all below) silently cut the briefing short. Force UTF-8 and never fail on a char.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    try:
        event = json.loads(sys.stdin.read() or "{}")
    except Exception:
        return
    name = (event.get("hook_event_name") or "").lower()
    if name == "userpromptsubmit":
        on_prompt(event)
    elif name == "stop":
        on_stop(event)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise  # keep on_stop's deliberate exit(1)
    except Exception:
        sys.exit(0)  # a broken hook must never break the chat
