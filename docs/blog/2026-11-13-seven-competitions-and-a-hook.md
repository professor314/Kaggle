---
title: "Seven Kaggle Competitions in One Night, and Why I Built a Hook to Slow Down"
date: 2026-11-13T21:30:00-08:00
draft: true
categories: ["data-science"]
tags: ["kaggle", "python", "ai-assisted", "workflow"]
description: "First submissions to seven new Kaggle competitions, what each one actually asks for, and the workflow hook I added after realizing we had stopped using our own toolkit."
---

# Seven Kaggle Competitions in One Night, and Why I Built a Hook to Slow Down

*DRAFT. Notes for a post. Fill in the story and the final scores before publishing.*

## The goal

Get one real submission into every new Kaggle competition I hadn't entered yet. Not good
submissions. Just a working pipeline for each, so I know what every competition actually
wants before deciding where to spend time.

## What each competition turned out to be

| Competition | What you submit | First score |
|---|---|---|
| Playground S6E11 (credit card fraud) | probabilities CSV | 0.94673 AUC (leader 0.95465) |
| Gemma 4 Developer Agent | a YAML-defined coding agent, graded on 129 real bug fixes | pending |
| Name That LLM | which of 18 LLMs wrote a text | 0.315 accuracy (chance 0.056) |
| Squares in Squares | positions of unit squares packed into a box, n = 1 to 1000 | 28.94 (lower is better, best public ~22.29) |
| Pokemon TCG AI Battle | a game-playing agent plus a 60-card deck | 345.1 (random-move agent) |
| Enveda CASMI26 | 25 candidate molecules per mass spectrum | pending |
| Google Quantum Circuit Compilation | compiled circuit files checked by a verifier | pending |
| Physical Engineering's Last Exam | a GitHub repo with an agent harness | not yet, needs setup |

Only two of these are "normal" machine learning. The rest are agent design, geometry,
chemistry search, and circuit optimization. That was the most useful thing I learned all night.

## What worked

- **Copying the sample submission first.** For the strange formats (agents, circuits, card
  games), packaging the official sample is the fastest way to prove the pipeline works.
- **GroupKFold for Name That LLM.** Every model answered the same prompts, so a random split
  leaks prompt wording into validation. Grouping by prompt gave an honest 0.31, which matched
  the leaderboard (0.315).
- **Reading big data inside Kaggle.** Enveda's training file is 3 GB. Running the search in a
  Kaggle notebook meant I never downloaded it.

## What went wrong

- A cancelled download kept running in the background and quietly put a 10.8 GB zip in my
  Google Drive folder.
- One submission went in twice because the command's output got lost and I retried.
- Kaggle moved where competition data is mounted. Seven of my public notebooks broke because
  they hard-coded the old path.
- The big one: **I stopped using my own toolkit.** Under time pressure, the assistant wrote
  one-off scripts, skipped most of the steering docs, and left code in a temp folder.

## The fix: a hook

So I added a Kiro hook to the `kaggle` agent. Every time I send a Kaggle-related message, it
injects the workflow rules, the toolkit's module list, and the repo's git state into the
conversation. When a turn ends with uncommitted code or no blog draft, it warns me. The rules
themselves live in a steering doc: use or extend the toolkit first, write heavily commented code
meant for learning, add anything reusable to the toolkit with a test, log every experiment,
commit and push, and keep a blog draft like this one.

## Next

- Rebuild S6E11 and Name That LLM on the toolkit (ModelArena, EnsembleBuilder).
- Move the Kaggle path resolver into the toolkit, since seven notebooks needed it.
- TODO before publishing: final scores for the pending submissions, a screenshot of the hook
  output, and a short "what each competition taught me" section.
