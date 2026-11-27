# Gemma 4 Developer Agent

## 1. Problem and metric
Submit a declarative ADK agent (agent.yaml, prompts, sub-agents, optional LoRA under 3 GiB). The swegemma harness runs it on 129 SWE-bench-style tasks (4x L4, 32k context). Score = fraction of tasks whose hidden tests pass. One submission per day. Closes 2026-12-02.

## 2. Results
| # | Approach | Public LB | Status |
|---|---|---|---|
| 1 | Official sample agent, 31B, 1 min / 10 tool call budget | 0.05 | VERIFIED |
| 2 | Budgets 8 min / 50 tools / 120 turns + reproduce-verify prompt | pending | submitted 2026-11-27 |

## 3. Design decisions
- **Raise the budget first.** The sample agent is capped at 1 minute and 10 tool calls (harness default is 60 min, 100 calls). Evidence: README section 7.1. Alternative rejected: training a LoRA first, which costs days and the budget cap would still bind.
- **Prompt: reproduce, fix, verify, check diff, submit last.** The sample prompt pushes the agent to stop in 8-10 turns. Scratch files go in /tmp because stray files in /workspace land in the patch (README section 10). Empty patches always score zero.
- **Keep the 31B model, no adapters.** The dummy sample adapters do not fit 31B and are removed. 12B is the fallback if wall-clock is a problem.
- **Moderate budget, not the maximum.** Total scoring time is undocumented, so 8 min per task is a compromise.

## 4. Toolkit reuse
None. Packaging follows Ryan Holbrook's Getting Started notebook. Kernels are private while the competition runs.

## 5. Reproduce
`kaggle kernels push -p competitions/gemma-4-developer-agent/notebooks/kernel-v2` then `kaggle competitions submit gemma-4-developer-agent -k seanconnolly/gemma-4-developer-agent-v2 -v <n> -f submission.zip`.

## 6. What's next
1. Read the v2 score. If the run failed on time, cut to 4 min.
2. Try thinking_budget 8192, lower temperature, or the 12B model.
3. Add a sub-agent to isolate exploration context.