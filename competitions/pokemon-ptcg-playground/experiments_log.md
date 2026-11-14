# Pokemon TCG AI Battle Challenge (Playground) - Experiment Log

**Kaggle ref:** the-pokemon-company-ptcg-ai-battle-challenge-playground | **Deadline:** 2027-01-08

## 2026-11-12 - First pass (research only)
- Verified deadline, reward, team count, file list via kaggle CLI.
- Listed top public notebooks via kaggle kernels list (see design_doc.md section 2).
- Tried data download: not possible, rules not accepted (403 on DownloadDataFiles) - for S6E11 confirmed directly; the others assumed the same until the account joins.
- No submissions made. No experiments run.
- Priority: 3b (fun, no money)

## 2026-11-13 - Submission 1: official sample agent
**Format (VERIFIED from sample_submission/):** submission.tar.gz with main.py (agent(obs_dict) -> option indices; returns 60-card deck when obs.select is None), deck.csv (60 card IDs), cg/ SDK (includes cg.dll, so the simulator can run locally on Windows).
**Approach:** packaged the sample as is (random legal options, sample deck). Pure floor.
**Submissions:** 58332029 and 58332051. The same file went in twice because the first CLI call's output was swallowed. PENDING (ladder episodes).
**Next:** heuristic agent (attack when possible, attach energy, evolve), run local matches with cg/sim.py vs the random agent; study the 'Ultra 900+' PPO bundle (credited public checkpoint, notebooks/ref/).
