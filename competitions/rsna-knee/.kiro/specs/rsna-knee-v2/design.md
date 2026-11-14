# Design — RSNA Knee v2 (tiered, research-grounded)

## Overview
Two decoupled subsystems, each independently validated on the 58 gold studies:
- **Label pipeline** (`labels/`): reports → 12 soft labels for all 4,407 train
  studies. Runs OFFLINE (rules, local LLM, or a paid API one-time). Output is a
  labels CSV shipped to Kaggle as a dataset.
- **Image pipeline** (`image/`): MIL model trains on those labels, validates
  leak-free on gold, and scores the test images in the internet-off kernel.

Everything is grounded in `research_notes.md` (the 0.903 blueprint). Design goal:
convert label quality into a real leaderboard score, climbing 0.533 → tiered targets.

## Component 1: Label pipeline

### 1a. Clinical extractor (rules → LLM)
- Input: `train.csv` Report text (EN + ~12 languages).
- Rules encoded once, reused by every extractor variant:
  - unmentioned finding = 0; explicit negation = 0; weight Impression section;
    current-only; synovitis breadth (Hoffa/plica/friction); contusion needs
    traumatic pattern; fracture includes avulsion/insufficiency; ruptured graft
    = ACL positive, intact graft = 0.
- Variants (tiers):
  - T1: improved keyword/regex rules with the clinical logic above.
  - T2: LLM extractor, fixed JSON of 12 keys, temp 0, reasoning off, retry on
    malformed JSON. Model per the cost table (free-local / mid-API).
- Output per study: 12 probabilities in [0,1].

### 1b. Fusion
- Sources: T1 rules, LLM, public `pilkwang/rsna-knee-llm-labels`.
- Weight each source's vote by its measured PER-CLASS gold accuracy.
- Agreement → hard 0/1; disagreement → reliability-weighted soft label.
- Emit `train_labels.csv` (StudyInstanceUID + 12 soft labels).

### 1c. Extractor validation (gate)
- Score every extractor/fusion against the 58 gold: per-class + macro AUC and
  accuracy. Read errors, revise rules/prompt once. Record numbers. Do NOT proceed
  to a full image run until labels clear the tier target.

## Component 2: Image pipeline (MIL)

### 2a. Series selection (`pick_series`)
- From each study's series (train_series.csv metadata), pick ≤3: prefer
  fluid-sensitive; one per plane (sagittal/coronal/axial); fill remainder.

### 2b. DICOM ETL (train offline → cache; mirror in kernel)
- Per slice: RescaleSlope/Intercept; invert MONOCHROME1; read InstanceNumber.
- Sort series slices by InstanceNumber; per-series 1-99 percentile window→[0,1];
  center-crop depth to ~64; resize 288×288 INTER_AREA; store uint8.
- Cache as .npz offline for fast training; kernel reproduces the exact math with a
  pydicom→pylibjpeg→gdcm decode fallback.

### 2c. Model (`KneeMILModel`)
```
per slice  -> shared timm backbone (EfficientNet-B3, in_chans=1) -> embedding
per series -> pool(embeddings)  [max-pool baseline; try mean / gated-attention]
           -> linear head -> 12 logits
study      -> mean of series logits -> 12 probs
```
- Loss: BCEWithLogitsLoss on soft labels. Optim: AdamW, warmup→cosine, bf16.
- Aug: rotation, brightness/gamma, slight rescale. **No horizontal flip.**
- Slices: T=12 train (jittered positions), evenly spaced (deterministic) at eval.

### 2d. Validation (leak-free)
- GroupKFold(5) by StudyInstanceUID. EXCLUDE the 58 gold from the fold pool.
- Score each gold study only on the fold checkpoint that never trained on it.
- Skip constant label columns per fold. Report CV macro-AUC + honest gold-58.

### 2e. Inference kernel (code comp)
- internet OFF, `machine_shape: NvidiaTeslaT4`, weights + labels as datasets.
- Reproduce ETL exactly; decode fallback chain; select series deterministically.
- Average 5 fold sigmoids. Write submission.csv every 25 studies. Decode failure
  → 0.5. Adaptive budget: drop eval slices 32→16, then ensemble 5→3, to fit ~9h.

## Correctness properties
1. Submission shape matches sample; 12 probs in [0,1]; no NaN; row per test study.
2. No horizontal flip anywhere (laterality preserved).
3. Gold-58 is never in any training fold (leak-free metric).
4. Kernel ETL is bit-exact vs the cached training tensors (parity check).
5. Unmentioned/negated findings map to 0 in the extractor.

## Testing strategy
- Local unit tests: extractor rules on crafted reports (negation, unmentioned,
  each clinical rule); series-selection prefers fluid-sensitive/one-per-plane;
  ETL windowing shape/range; submission assembler; fold-split excludes gold.
- Extractor gate: gold-58 macro-AUC printed and logged before any image run.
- Image smoke: 1 fold, few epochs, tiny slice budget → prove it runs + logs a
  non-chance gold AUC, THEN scale to 5-fold/8-epoch.

## Optional edge tracks (layer onto the tiers when base pipeline works)

### Track P — MRNet-pretrained backbone (cheap, high ROI)
Pretrain (or fetch a public checkpoint of) the 2D backbone on Stanford MRNet
(1,370 knee MRIs, ACL/meniscus/abnormal). Load as the MIL backbone init instead
of ImageNet. Attach as a Kaggle dataset (internet-off safe). Expected bump on
ACL/meniscus. Legal, competition-relevant.

### Track S — Self-supervised pretraining on the 820k unlabeled slices
Contrastive/SSL (SimCLR-style, anatomy-aware) on our own unlabeled slice pool,
then fine-tune the MIL head on weak labels. Bigger effort; differentiated edge
(few teams will do it). Only after Tiers 1-2 prove out.

### Track M — Multi-plane fusion
Separate per-plane handling (sagittal/coronal/axial) fused at the study level,
rather than treating all series identically. Fits `pick_series`; classic MRNet win.

### Track E — Efficiency-prize model (separate award)
A compact, fast variant (EfficientNet-B0, few slices, single/low fold) tuned to
the efficiency-track rules. Can win money without topping accuracy. Build once the
accuracy pipeline exists; submit to the efficiency track.

## Deployment / sequencing
Tier 0 fixes → Tier 1 (better rules + MIL, submit) → Tier 2 (LLM labels, retrain)
→ Tier 3 (full 5-fold + robust kernel). Edge tracks P/M are cheap add-ons to
Tier 1-3; S and E are opportunistic. Cost gate before any paid LLM run
(Requirement 5). Blog post stays draft until a real score lands.
