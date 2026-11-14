# Requirements — RSNA Knee v2 (research-grounded, tiered)

## Introduction
Rebuild the RSNA Knee approach from what actually wins (see `research_notes.md`,
grounded in the public 0.903 / rank-18 blueprint). Two independent quality levers:
(A) **label extraction from reports** and (B) **the image model that learns from
those labels and scores the test set**. Label quality is the ceiling; the image
model converts it into a leaderboard score.

Our current honest numbers (gold-58 macro-AUC): rules 0.607, NLI 0.556, image ~0.52,
best LB 0.533. The blueprint reaches ~0.83 labels → 0.857 honest image CV → 0.903 LB.

## Tiered approaches (each a shippable milestone; do in order, stop when ROI drops)

### Tier 0 — Correctness fixes (cheap, unblocks everything)
Fix the known mistakes regardless of tier: unmentioned-finding = 0; per-series
percentile windowing + RescaleSlope/Intercept + MONOCHROME1 invert + InstanceNumber
sort; never horizontal-flip; GroupKFold excluding the 58 gold.

### Tier 1 — Better free labels + MIL image model (target: beat 0.533 LB, honest gold > 0.65)
- Labels: upgrade rules with the clinical definitions from the blueprint
  (negation, unmentioned=0, synovitis/effusion/contusion/fracture rules,
  weight the Impression section). Optionally fuse the public
  `pilkwang/rsna-knee-llm-labels` dataset as a second opinion.
- Image: **MIL model** (per-slice timm backbone → max-pool → 12 logits), series
  selection (≤3, fluid-sensitive, one/plane), EfficientNet-B3 @ 288, T=12 slices.
- Validate gold-58 (leak-free). Submit.

### Tier 2 — LLM labels (target: labels ~0.80+, LB > 0.85)
- Generate labels with an open LLM (offline). Two options:
  1. Local LLM on Sean's box (RTX 3090) or a paid API (~$1-20, one-time) over the
     4,407 reports with the clinical JSON prompt.
  2. Reuse the public LLM-label dataset directly.
- Fuse sources weighted by per-class gold accuracy → soft labels.
- Retrain the Tier-1 MIL model on the better labels. 5-fold ensemble.

### Tier 3 — Full competitive pipeline (target: ~0.88-0.90 LB)
- 5-fold GroupKFold, EfficientNet-B3, gated-attention vs max pool (search),
  medium aug, bf16, warmup→cosine, best-val checkpoints.
- Robust inference kernel: decode fallback chain, adaptive time budget, periodic
  submission.csv writes, 0.5 on decode failure, 5-fold sigmoid mean.

## Requirements

### Requirement 1: Correct, clinical label extraction
1. WHEN a finding is not mentioned in a report THE extractor SHALL label it 0.
2. WHEN a finding is explicitly negated THE extractor SHALL label it 0.
3. THE extractor SHALL apply the per-finding clinical rules (synovitis breadth,
   contusion traumatic-pattern, fracture avulsion/insufficiency, current-only).
4. THE extractor SHALL be validated against the 58 gold studies (per-class +
   macro), errors read, and rules revised at least once.
5. WHERE multiple label sources exist THE system SHALL fuse them weighted by
   per-class gold accuracy, emitting soft labels on disagreement.

### Requirement 2: MIL image model (not mean-pooled)
1. THE model SHALL embed each slice with a shared 2D backbone and pool per-slice
   embeddings (max-pool baseline), NOT average slices into one image.
2. THE model SHALL output 12 independent logits; study pred = mean of series logits.
3. THE model SHALL select ≤3 series preferring fluid-sensitive, one per plane.
4. THE preprocessing SHALL apply RescaleSlope/Intercept, invert MONOCHROME1, sort
   by InstanceNumber, per-series 1-99 percentile window, resize 288, grayscale.
5. THE augmentation SHALL NOT horizontally flip.

### Requirement 3: Honest validation
1. THE training SHALL use GroupKFold(5) by StudyInstanceUID.
2. THE 58 gold studies SHALL be EXCLUDED from all training folds; gold macro-AUC
   is measured only on checkpoints that never saw each gold study.
3. THE loss SHALL be BCEWithLogitsLoss on soft labels (no thresholding).
4. Constant label columns within a fold's val split SHALL be skipped for AUC.

### Requirement 4: Competition-safe inference
**User Story:** As the competitor, I want the scoring kernel to survive the code-
competition environment, so that a valid submission is always produced.

#### Acceptance Criteria
1. THE kernel SHALL run internet OFF with `machine_shape: NvidiaTeslaT4`, weights
   attached as a dataset.
2. THE kernel SHALL reproduce training preprocessing exactly.
3. THE kernel SHALL include a DICOM decode fallback chain and write a valid
   submission.csv periodically; decode failure → 0.5 for that study.
4. THE kernel SHALL average fold sigmoids and degrade gracefully under the time cap.

### Requirement 5: LLM label cost is estimated and approved before spend
**User Story:** As the owner, I want a token/cost estimate before any paid LLM
run, so I approve spend knowingly and pick the best model for the money.

#### Acceptance Criteria
1. BEFORE any paid LLM labeling run THE plan SHALL present an itemized estimate:
   input tokens, output tokens, chosen model, per-token rate, and total $.
2. THE estimate SHALL be based on the measured report corpus (median ~977 chars ≈
   ~250 tok, max ~4,743 ≈ ~1,200 tok; 4,407 reports; one call per report scoring
   all 12 findings in a single JSON response).
3. THE system SHALL prefer the FREE options first (local open LLM on the RTX 3090,
   or the public `pilkwang/rsna-knee-llm-labels` dataset) and only spend on an API
   model IF free options underperform the gold-58 target.
4. WHEN an API model is used THE choice SHALL favor the best accuracy-per-dollar
   model available at run time (a strong instruction-following model), and the run
   SHALL be a one-time offline pass saved to a labels file/dataset (never called
   from the internet-off kernel).
5. THE actual token counts and $ spent SHALL be recorded in experiments_log.

### Requirement 6: Public-asset license compliance is documented and verifiable
**User Story:** As the owner, I want every reused public dataset/model checked for
its license and recorded with links, so I can verify prize eligibility and we never
build on something we're not allowed to use.

#### Acceptance Criteria
1. BEFORE reusing any public dataset or model THE system SHALL record its source
   URL, author, stated license, and a prize-eligibility note in
   `competitions/rsna-knee/PUBLIC_ASSETS_LICENSES.md`.
2. WHERE an asset is unlicensed, non-commercial, or has an unclear/incompatible
   license THE system SHALL flag it and NOT use it in a prize submission until cleared.
3. THE license doc SHALL be kept current whenever a new public asset is added and
   SHALL be human-readable so the user can independently verify each link.
4. BEFORE a final/prize submission THE system SHALL re-confirm that every asset in
   the winning pipeline is license-cleared.

## Token / cost estimates (from the measured corpus)

Job: 1 call per report, all 12 findings in one JSON answer, over 4,407 reports.
- Input ≈ ~1,000 tok/report avg (prompt rules ~250 + report ~250-1,200) → **~4.4M input tokens**.
- Output ≈ ~60 tok/report (12 keys + brief) → **~0.26M output tokens**.

| Option | Model class | ~Rate (in / out per 1M) | Est. total | Notes |
|---|---|---|---|---|
| FREE-local | Open LLM on RTX 3090 (e.g. Qwen2.5-32B-Instruct 4-bit) | $0 | **$0** | Slow (hours), best free quality; blueprint used a 35B locally |
| FREE-dataset | Reuse `pilkwang/rsna-knee-llm-labels` | $0 | **$0** | Someone else's LLM labels; instant |
| CHEAP-API | small (e.g. GPT-4o-mini / Haiku class, ~$0.15/$0.60) | | **~$0.80** | Good baseline, likely ~0.75-0.80 |
| MID-API | mid (e.g. GPT-4o / Sonnet class, ~$3/$15) | | **~$17** | Blueprint-level ~0.83, best value pick |
| TOP-API | top (e.g. Opus / GPT-4-class, ~$15/$75) | | **~$85** | Diminishing returns for label extraction |

Rates are approximate and change; re-verify at run time. **Recommended spend
path:** free-local or free-dataset first → if under the gold target, one **MID-API**
pass (~$17) as the best accuracy-per-dollar. Reserve TOP-API only if MID stalls.

## Out of scope (for now)
- Beating rank 18. Goal is a correct, tiered pipeline that climbs from 0.533.
- Second independently-trained LLM cross-check extractor (Tier 4+ later).
