# Implementation Plan — RSNA Knee v2

Grounded in `research_notes.md` + `design.md`. Each tier is a shippable milestone;
stop/reassess when ROI drops. Prove-it-runs-small before scaling every kernel.
All work runs on Kaggle (data mounted); local box only edits, pushes, and does the
one-time offline label generation. ~7 weeks to the 2026-10-22 deadline.

## Tier 0 — Correctness fixes + local test harness
- [x] 0.1 `labels/rules.py`: clinical report rules (unmentioned=0, negation=0,
      Impression-weighted, synovitis breadth, contusion traumatic, fracture
      avulsion/insufficiency, strong-negation override). DONE. Requirements 1.1-1.3
- [x] 0.2 `image/etl.py`: pure windowing pipeline (RescaleSlope/Intercept,
      MONOCHROME1 invert, per-series 1-99 percentile window, resize 288, uint8) +
      `read_dicom_slice` wrapper. DONE. Requirement 2.4
- [x] 0.3 `image/series.py`: `pick_series` (≤3, fluid-sensitive, one/plane). DONE. Req 2.3
- [x] 0.4 `image/folds.py`: GroupKFold(5) by StudyID EXCLUDING the 58 gold (fold=-1). DONE. Req 3.1-3.2
- [x] 0.5 pytest `test_tier0.py`: 18 tests (negation/unmentioned/each rule family;
      ETL shape+range+invert; series fluid/plane; folds exclude gold + group by
      study; submission format). All 18 pass locally.

## Tier A — Assemble from public parts FIRST (fastest path to a real score)
Reuse published assets instead of rebuilding (see research_notes "PUBLIC ASSETS").
**Verify each dataset's/model's license (task A.0) before ANY reuse, and re-confirm
before a final/prize submission.**
- [x] A.0 **License audit + link doc.** DONE 2026-09-02. All key assets verified via
      Kaggle API: raptor-knee-widedense CC0 (0.924 LB single model), stevenleehans
      labels CC0 (0.8927 gold), pilkwang/lixin73/yunusgmsoy/rayanbabur labels CC0,
      alenic images MIT; only barun2104 3D volumes = "other" (flagged). Recorded in
      `competitions/rsna-knee/PUBLIC_ASSETS_LICENSES.md`. Requirements 6.x.
- [x] A.1 **EARLY SUBMISSION.** DONE 2026-09-02, scored 2026-09-03: **LB 0.924**
      (vs 0.533 from-scratch, 0.500 smoke; leaders ~0.95). Reproduced the CC0 raptor
      CoAtNet (`dreaddevelopment/raptor-knee-widedense`) as our own internet-off T4
      kernel (`kernel_a1/`), `seanconnolly/rsna-knee-a1`. This is the baseline A.2–A.6 must beat.
- [ ] A.2 Evaluate public label sets on the 58 gold; pick the best/merged one
      (`stevenleehans/...`, `yunusgmsoy/...4-source-merged`,
      `rayanbabur/...calibrated-soft-targets`). Record gold macro-AUC of each.
- [ ] A.3 Use a preprocessed image dataset (`alenic/...jpeg-224x224` or
      `barun2104/...3d-volumes`) to skip DICOM ETL for training.
- [ ] A.4 Init backbone from a public medical checkpoint (RadImageNet heads /
      DINOv2-knee / MRNet) rather than ImageNet.
- [ ] A.5 Train our MIL head on (best public labels + preprocessed images +
      medical backbone); leak-free gold-58 validation.
- [ ] A.6 Inference kernel (internet off, T4); **CHECKPOINT: submit + log LB + gold.**
      This is our new baseline to beat, likely well above 0.533.

## Tier 1 — Better free labels + MIL image model (target: beat 0.533 LB; gold > 0.65)
- [ ] 1.1 Run improved rules over 4,407 reports; validate gold-58 (per-class+macro),
      read errors, revise once. Log numbers. Requirement 1.4
- [ ] 1.2 (fusion) optionally merge public `pilkwang/rsna-knee-llm-labels` weighted
      by per-class gold accuracy → soft `train_labels.csv`. Requirement 1.5
- [ ] 1.3 `image/model.py` `KneeMILModel`: per-slice timm backbone (in_chans=1) →
      max-pool → 12 logits; study = mean of series logits. Requirements 2.1-2.2
- [ ] 1.4 `image/train.py`: BCEWithLogits on soft labels, AdamW, warmup→cosine,
      bf16, aug WITHOUT horizontal flip; best-val checkpoint per fold. Reqs 2.5, 3.3-3.4
- [ ] 1.5 Image smoke run on Kaggle: 1 fold, few epochs, small T → prove it runs,
      logs a non-chance gold AUC. Then scale.
- [ ] 1.6 `image/infer.py` kernel: internet off, machine_shape T4, decode fallback
      chain, deterministic series pick, fold-sigmoid mean, periodic submission.csv,
      0.5 on decode fail, adaptive time budget. Requirement 4
- [ ] 1.7 **CHECKPOINT: submit** the Tier-1 MIL model; record LB + honest gold-58 in
      experiments_log. Should beat the Tier-A reuse baseline.

## Track P — MRNet-pretrained backbone (do alongside Tier 1)
- [ ] P.1 Get Stanford MRNet weights or pretrain EfficientNet on MRNet (ACL/
      meniscus/abnormal); save + attach as a Kaggle dataset (offline-safe)
- [ ] P.2 Init `KneeMILModel` backbone from MRNet instead of ImageNet; retrain a
      fold; compare gold-58 vs ImageNet init. Keep if better.

## Track M — Multi-plane fusion (cheap add-on to Tier 1)
- [ ] M.1 Handle sagittal/coronal/axial series and fuse at study level; compare
      gold-58 vs plane-agnostic. Keep if better.

## Tier 2 — LLM labels (target: labels ~0.80+, LB > 0.85)
- [ ] 2.1 Present the token/cost estimate (Requirement 5) and get spend approval.
- [ ] 2.2 Generate labels offline: FREE-local (Qwen2.5-32B on RTX 3090) or, if
      approved, one MID-API pass (~$17). Fixed 12-key JSON, temp 0, retry on bad JSON.
- [ ] 2.3 Validate the LLM labels gold-58; revise the prompt once from the errors.
- [ ] 2.4 Fuse rules + LLM + public dataset (per-class-weighted) → soft labels.
- [ ] 2.5 Retrain the Tier-1 MIL model on the better labels (5-fold).
      **CHECKPOINT: submit + log LB + gold.**

## Tier 3 — Full competitive pipeline (target: ~0.88-0.90 LB)
- [ ] 3.1 5-fold GroupKFold, EfficientNet-B3 @ 288, T=12; pooling search (max vs
      gated-attention); medium aug; 8 epochs; bf16.
- [ ] 3.2 Honest cross-fitted gold-58 (each gold scored only by the fold that held
      it out). Record CV + gold.
- [ ] 3.3 Robust 5-fold ensemble inference kernel; verify ETL parity vs cache;
      adaptive budget. **CHECKPOINT: submit + log LB + gold** (our best full-pipeline run).

## Track S — Self-supervised pretraining (opportunistic, after Tier 2)
- [ ] S.1 SimCLR-style SSL on a sample of the 820k unlabeled slices → backbone init.
- [ ] S.2 Fine-tune MIL head on weak labels; compare gold-58. Keep if better.

## Track E — Efficiency-prize model (separate award)
- [ ] E.1 Compact fast variant (EfficientNet-B0, few slices, 1-3 folds) tuned to
      the efficiency-track rules; measure runtime + accuracy.
- [ ] E.2 Submit to the efficiency track; log.

## Cross-cutting
- [ ] X.0 Maintain `competitions/rsna-knee/PUBLIC_ASSETS_LICENSES.md` (see A.0): keep
      every reused dataset/model's URL + license + eligibility note current. This is
      the user-verifiable record; update it whenever a new public asset is added.
- [ ] X.1 Keep research/design_doc.md + experiments_log.md current (VERIFIED/ASSUMED tags).
- [ ] X.2 Commit + push to professor314/Kaggle after each tier (exclude data/weights/artifacts/.kiro).
- [ ] X.3 **Keep the project story current.** Add a dated chapter/scoreboard row to
      `STORY.md` at every breakthrough or milestone (this is the in-project
      narrative). The draft blog post
      (`imadestuff-blog/.../rsna-knee-mri-baseline/index.md`) is drawn FROM
      `STORY.md`; keep it `draft: true` and publish only once the model is real.

## Submission checkpoints (submit early + often as complexity grows)
Each is a real LB score, logged in experiments_log.md, building on the last:
1. **A.1** — public model reuse (fastest real score, do ASAP).
2. **A.6** — our MIL head on public labels+images+backbone.
3. **1.7** — Tier-1 MIL on our improved free labels.
4. **2.5** — retrained on LLM/fused labels (cost-gated).
5. **3.3** — full 5-fold ensemble (best effort).
6. **E.2** — efficiency-track submission (separate prize).

## Task Dependency Graph
```
Tier0(0.1-0.5) ─┬─> Tier1(1.1..1.7) ─┬─> Tier2(2.1..2.5) ─> Tier3(3.1..3.3)
                │                     ├─> TrackP(P.1,P.2)
                │                     └─> TrackM(M.1)
                └────────────────────────> (folds/etl feed everything)
Tier2 ─> TrackS(S.1,S.2)
Tier1 ─> TrackE(E.1,E.2)
X.1/X.2 run continuously; X.3 after first real LB score.
```

## Notes
- Never horizontal-flip (5 laterality labels). Exclude gold-58 from training folds.
- Metric is ranking-only → no calibration; mean-ensembling is free/safe.
- Public LB != private (prevalence differs); trust honest gold + CV, don't overfit LB.
- Cost gate before any paid LLM (Req 5). Prefer free label sources first.
