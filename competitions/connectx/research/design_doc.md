# ConnectX — Design Doc

**Competition:** ConnectX (Connect Four, agent vs agent)
**Metric:** Skill rating from head-to-head games | **Status:** minimal (rank ~64, 1 submission)
**Type:** Simulation / agent competition

## Domain research
Connect Four on a configurable board. You submit an **agent** (a function that
returns a move given the board state), not predictions. Scored by playing
against other agents. This is outside the toolkit's tabular/DL core, so effort
has been intentionally minimal.

## Prior art
Strong agents use minimax / negamax with alpha-beta pruning and a heuristic
evaluation (center control, connected-N counts, threat detection). Top agents
add deeper search or learned value functions.

## Design decisions
| Decision | Choice | Rationale |
|---|---|---|
| Agent | Heuristic + shallow lookahead | Cheap baseline, beats random/one-step |
| Scope | Minimal | Not a fit for the ML toolkit; low priority |

## Status / next
Baseline agent submitted (rank ~64). Improvement path if revisited: negamax with
alpha-beta to depth 5-7 + a threat-aware evaluation. Low priority relative to
the tabular/DL roadmap.
