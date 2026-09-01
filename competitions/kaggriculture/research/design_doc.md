# Kaggriculture — Design Doc

**Competition:** Kaggriculture ($50K) | **Deadline:** 2026-09-30
**Type:** Code / agent Competition | **Status:** greedy agent submitted

## Domain research
Agent-based competition (submit an agent that acts in a simulated agriculture
environment, scored by outcome, not static predictions). Featured-prize comp
($50K) requiring identity/selfie verification to be eligible for the reward
(completed).

## Prior art
Agent competitions reward: a solid greedy/heuristic baseline first, then either
search (rollouts / MCTS) or a learned policy. Read the environment's AGENTS.md /
README (in `../data/`) for the exact action/observation API and scoring.

## Design decisions
| Decision | Choice | Rationale |
|---|---|---|
| Baseline | Greedy heuristic agent | Establish a scoring baseline fast |
| Eligibility | Selfie/identity verification | Required for the $50K prize (done) |
| Scope | Iterate if ROI justifies | $50K reward but agent design is heavy |

## Status / next
Greedy agent submitted. Improvement path: study the env API in `../data/`, add
lookahead/rollout planning or a learned policy. High reward but high effort;
prioritize against the tabular/DL roadmap and remaining deadline.
