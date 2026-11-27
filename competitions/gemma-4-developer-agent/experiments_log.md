# Gemma 4 Developer Agent (agent harness) - Experiment Log

**Kaggle ref:** gemma-4-developer-agent | **Deadline:** 2026-12-02

## 2026-11-12 - First pass (research only)
- Verified deadline, reward, team count, file list via kaggle CLI.
- Listed top public notebooks via kaggle kernels list (see design_doc.md section 2).
- Tried data download: not possible, rules not accepted (403 on DownloadDataFiles) - for S6E11 confirmed directly; the others assumed the same until the account joins.
- No submissions made. No experiments run.
- Priority: 2 (closes 12/2, big prize, strong starter)

## 2026-11-13 - Submission 1: official sample agent, 31B, no adapters
**Format (VERIFIED from HARNESS_README.md + Getting Started notebook):** submit `submission.zip` (declarative ADK agent: agent.yaml, prompts, sub_agents, optional LoRA adapters <3 GiB) as a notebook output. Graded by the swegemma harness on 4x L4 (vLLM, 32k context): agent patches 129 SWE-bench style tasks (fastapi etc.), score = resolution rate. One submission per day.
**Kernel:** seanconnolly/gemma-4-developer-agent-v1 (CPU-only, packaging only). Notebooks/kernel-v1/.
- Gotcha: must keep the pinned `docker_image` from the reference metadata. Without it Kaggle used Python 3.13 and the wheelhouse pip install failed.
- Gotcha: full competition zip is 10.8 GB. Don't download it locally (it sits under Google Drive). Read files in the Kaggle notebook instead.
**Submission:** 58331749, status PENDING at 2026-11-14 04:43 UTC.
**Next levers:** tune prompts/system.md (repro-first, run tests, minimal diff), eval_config budgets, sampling/thinking budget, try 12B for more turns per time budget; run the notebook's Phase 1/2 local eval on a few tasks (L4 GPU quota) before submitting changes.

## 2026-11-27 - Submission 2 (v2): raise budgets, new prompt
**Finding (VERIFIED from HARNESS_README.md section 7):** the sample eval_config.yaml gives each task 1 minute, 10 tool calls, 50 turns, 60 s command timeout, and its system prompt tells the agent to finish in 8-10 turns. Harness defaults are 60 min / 100 tool calls / 500 turns. Our v1 score 0.05 (leaders ~0.24) is probably budget starvation, not model weakness.
**Change:** eval_config 120 s command timeout, 50 tool calls, 8 min, 120 turns. New prompt: reproduce in /tmp, fix root cause with small edits, run targeted tests, check git diff, submit_patch last, never empty patch. Model unchanged (31B, no adapters).
**Risk:** total scoring wall-clock is not documented. 8 min x 129 tasks could hit a cap if tasks run sequentially. If v2 errors out, drop to 4 min.
**Submission:** 59023895, PENDING (kernel seanconnolly/gemma-4-developer-agent-v2, private: competition-linked notebooks cannot be public until it ends). 0 submissions left today.
**Next:** read the score; tomorrow try sampling (temperature, thinking_budget 8192) or the 12B model for more turns.
