# GA Run Stats

A log of every genetic-algorithm training run: how much was played, how long it
took, and the best result. Append a new section per run. Totals are computed as
`generations × population × games_per_eval` (games) and `× 720` (turns).

## Run 1 — 2026-09-02 (vs `starter`)
| Metric | Value |
|---|---|
| Command | `evo/ga.py --pop 48 --gens 60 --games 6 -j 12 --seed 7` |
| Generations completed | 23 (of 60; stopped early — winrate had saturated) |
| Population | 48 |
| Games per genome eval | 6 (both seats) |
| **Total games played** | **6,624** |
| Turns per game | 720 |
| **Total game-turns** | **4,769,280** |
| **Total agent decisions** (both players) | **9,538,560** |
| Parallel workers | 12 |
| Wall-clock compute (sum of gen times) | ~20.1 min (~52s/gen) |
| Opponent | built-in `starter` |
| **Best genome** | fitness 4994 → **$3,994 money, 100% winrate**, found at gen 10 |
| Baseline before GA | greedy agent: 0% winrate, ~$1,168 (lost money) |
| Artifacts | `evo/state/best.json`, `history.csv`, `gen_000..022.json` |

### Per-generation best (money / winrate)
gen0 3253/33% · gen4 3484/67% · gen5 3810/83% · gen6 3663/100% · gen10 3994/100%
(peak) · gen17 3848/100% · gen21 4158/67% · gen22 3848/100%
