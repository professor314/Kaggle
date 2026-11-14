---
phase: 1
title: Domain Research
triggers: kaggle_ml_toolkit.research
---

# Domain Research

## Context
The research phase is where we build domain expertise before touching any data or models. This is the foundation — strong research leads to better hypotheses, smarter feature engineering, and higher leaderboard scores. We consult academic papers, Kaggle discussion forums, prior competition solutions, and domain-specific knowledge bases.

## Required Outputs
Every competition's `research/design_doc.md` MUST include:
1. **Domain research** — what is this problem about in the real world?
2. **Prior art** — what did top Kaggle solutions do?
3. **Feature engineering plan** — what features and why?
4. **Design decisions** — every major architectural choice with its rationale (model choice, CV strategy, feature constraints, transforms, what was excluded and why)

## CRITICAL RULES
- **Never work on closed competitions** — always verify the competition deadline before investing time. Use `kaggle competitions list` to check active competitions.
- **Always check the Code and Discussion tabs** on Kaggle before building our own approach. Top-voted notebooks and discussion threads reveal what works, what doesn't, and competition-specific quirks.
- **Use public notebooks as research, not as copy-paste** — understand the ideas, then implement in our own toolkit style.
- **Record facts as VERIFIED or ASSUMED, never bare claims.** Any statement in a
  design_doc or experiments_log that will guide later work must be tagged either
  `VERIFIED` (with the command + evidence that proves it) or `ASSUMED` (needs
  checking). An ASSUMED item must not drive more than one action before it is
  verified. This prevents a single unchecked observation from sending us in
  circles. See `docs/COE-watson-submission-loop.md` and the anti-loop rules in
  `07-submission.md`.

## Decision Points

### Research Scope
**Question:** How deep should we go on domain research for this competition?
**Options:**
- Quick survey: Skim top 3 Kaggle discussions + competition overview. Best for Getting Started competitions where speed matters more than novelty.
- Standard research: Read top solutions from similar competitions, scan relevant papers, summarize domain knowledge. Good balance for most competitions.
- Deep dive: Full literature review, multiple domain sources, detailed hypothesis formation. Best for Featured competitions or unfamiliar domains.
**Recommendation:** Standard research for first attempt; upgrade to deep dive if initial models underperform expectations.

### Domain Sources
**Question:** Which domain knowledge sources should we consult?
**Options:**
- Kaggle Discussions only: Past winner writeups, competition forums, dataset descriptions
- Kaggle + academic: Add Google Scholar papers, domain textbooks, Wikipedia for background
- Kaggle + academic + industry: Add industry reports, practitioner blogs, specialized databases
**Recommendation:** Start with Kaggle + academic. Industry sources help for niche domains (finance, healthcare, NLP).

### Prior Work Search
**Question:** How should we search for prior work on this problem type?
**Options:**
- Competition-specific: Search for this exact competition's past solutions and discussions
- Problem-class: Search for the general problem type (e.g., survival prediction, image classification)
- Both: Competition-specific first, then broaden to problem class for transferable insights
**Recommendation:** Both — competition-specific solutions give direct tactics; problem-class research reveals techniques others may have missed.

### Hypothesis Formation
**Question:** How many initial hypotheses should we form before EDA?
**Options:**
- Focused (2-3 hypotheses): Pick the most promising angles from research. Faster iteration.
- Broad (5-7 hypotheses): Cover multiple angles. Better for complex or novel problems.
- Exhaustive (8+ hypotheses): Document everything plausible. Best when domain is unfamiliar.
**Recommendation:** Broad (5-7) — gives enough variety to guide EDA without becoming unwieldy.

### Model Architecture Considerations
**Question:** Which model families should we flag for later investigation?
**Options:**
- Classical ML only: Logistic regression, tree-based models, SVMs. Sufficient for tabular data.
- Classical + boosting: Add XGBoost, LightGBM, CatBoost. Standard for competitive tabular problems.
- Classical + boosting + neural: Add TabNet, neural embeddings. For large datasets or when boosting plateaus.
**Recommendation:** Classical + boosting for Phase 1 (tabular competitions). Neural options noted for Phase 3.

## Actions
Based on decisions above, Kiro will:
1. Execute `ResearchDocumentGenerator.create_research_document()` to scaffold the research doc
2. Populate sections based on chosen research scope and sources
3. Record hypotheses in the research document
4. Flag model architectures for the model selection phase
5. Execute `ResearchDocumentGenerator.create_design_doc()` with findings
6. Present a summary of findings and ask the user to confirm or refine hypotheses before moving to EDA
