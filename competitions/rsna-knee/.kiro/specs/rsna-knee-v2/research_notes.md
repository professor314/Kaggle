# RSNA Knee — Research Notes (domain grounding for the v2 spec)

Sources: the public 18th-place writeup (0.903 LB, Yash Bishnoi, HuggingFace), the
JunhaoLiXD V03 repo, MRNet paper lineage, and our own experiments. All facts
below are used to design the tiered plan.

## What actually wins this competition (0.903 LB blueprint)
1. **Label quality is the whole ceiling.** Every image-side lever tops out at the
   quality of the report-derived labels. Winner: LLM extractor (Qwen3-35B via
   Ollama, temp 0, fixed JSON of 12 keys) → **0.833 mean per-class accuracy** vs
   the 58 gold. Rules/NLI (us) top out ~0.6.
2. **Labeling rules that matter (clinical, not keyword):**
   - Unmentioned finding = **0** (radiologists report positives, stay silent on
     negatives). [We wrongly treated report-blanks as "unlabeled".]
   - Explicit negation ("no effusion", "sin rotura") = 0.
   - Impression/Conclusion section weighted highest.
   - Only current findings (not chronic/post-op). Ruptured graft = positive ACL;
     intact graft = 0.
   - Synovitis: widen to include Hoffa impingement, plica, friction syndromes.
   - Effusion: "trace/some" is a judgment call (annotator noise here too).
   - Contusion: require a traumatic pattern (not degenerative marrow edema).
   - Fracture: include avulsion + insufficiency fractures.
   - Validate the prompt against the 58 gold, read the errors, revise.
3. **Fuse multiple label sources** weighted by per-class gold accuracy; soft
   labels when sources disagree. A public LLM-label dataset exists as a 3rd opinion:
   `pilkwang/rsna-knee-llm-labels`.
4. **Image model = MIL (MRNet-style), NOT mean-pool-into-one-image.**
   - Shared 2D `timm` backbone (grayscale, 1 channel) applied per slice.
   - Pool per-slice embeddings → per-series embedding → linear head → 12 logits.
   - Study prediction = mean of series-level logits.
   - Pooling: mean / max / gated-attention. Their HP search picked **max-pool**.
   - Winning config: **EfficientNet-B3**, K=3 series, T=12 slices, medium aug.
5. **Series selection:** pick up to 3 series, prefer **fluid-sensitive**, one per
   plane (sagittal/coronal/axial), fill remaining from the rest.
6. **DICOM preprocessing:** apply RescaleSlope/Intercept; invert MONOCHROME1;
   sort slices by InstanceNumber; per-series 1-99 percentile window → [0,1];
   center-crop depth to ~64; resize 288x288 INTER_AREA; cache uint8.
7. **NEVER horizontal-flip** (5 of 12 labels are laterality-specific: medial vs
   lateral meniscus, medial vs lateral OA, MCL).
8. **Validation:** GroupKFold(5) by StudyID; BCEWithLogitsLoss on SOFT labels
   (no thresholding); AdamW + warmup→cosine; bf16. **EXCLUDE the 58 gold from
   training folds** or the gold metric leaks (their 0.9357 was leaked; honest
   0.857). Skip constant label columns in a fold's val (AUC undefined).
9. **Inference (code comp, internet OFF, ~9h, ~1300 studies):** reproduce ETL
   bit-exactly; decode fallback chain pydicom→pylibjpeg→gdcm; average 5 fold
   sigmoids; write submission.csv every N studies; failed decode → 0.5; adaptive
   time budget (drop slices, then drop ensemble size).
10. **Honest ceilings:** report-vs-image disagreement caps text labels ~0.88-0.92.
    Metric is ranking-only (ROC AUC) → no calibration needed; mean-ensemble is safe.

## Why OUR attempts stalled (mapped to the above)
- Image ~0.52: we mean-pooled K slices into ONE blurry image (destroys the local
  lesion signal) — needed MIL per-slice + max-pool. Also B0@192 vs B3@288, and we
  blended all series instead of selecting 3 informative ones.
- Labels ~0.55-0.61: keyword rules / zero-shot NLI can't encode the clinical
  definitions; and we treated unmentioned as "unlabeled" instead of 0.
- Both are fixable and both are known-good in the blueprint.

## Strategic context (deadline, field, prizes)
- **Deadline: 2026-10-22.** Winners announced Nov, recognized at RSNA 2026
  (Nov 29-Dec 3, Chicago). Today ~2026-09-01 → **~7 weeks left.**
- **$77,000 total prizes**, INCLUDING a first-time **"most efficient model" award**
  (separate track — small/fast models can win money without topping accuracy).
- Field: 792+ teams and growing fast (430 on day 2). "All are welcome" — mix of
  individuals, radiology-AI researchers, and institutions. Official RSNA challenge
  → expect serious medical-imaging labs + strong Kaggle GMs at the top (~0.95).
- Public vs private: RSNA warns finding prevalence differs across train/public/
  private, so public rank ≠ final rank. Don't overfit the public LB.

## Additional edges beyond the 0.903 blueprint (from papers)
1. **Pretrain/transfer on Stanford MRNet** (1,370 knee MRIs w/ ACL/meniscus/abnormal
   labels, public). A knee-MRI-pretrained backbone should beat ImageNet init on
   our 12 findings. MRNet studies hit ~0.93-0.96 AUC on ACL. Strong, legal,
   competition-relevant pretraining source. [arxiv 2010.01947, ModAn-MulSupCon
   0.964 ACL].
2. **Multi-plane fusion done right** (separate sagittal/coronal/axial subnets then
   fuse) beat naive multi-plane; classic MRNet result. Aligns with our per-plane
   series selection.
3. **Supervised contrastive / SSL pretraining on the 820k unlabeled slices** (we
   have a huge unlabeled image pool) before fine-tuning on weak labels — anatomy/
   modality-aware contrastive pretraining set SOTA on MRNet-ACL (0.964).
4. **CNN transfer still beats ViT** at this data scale (multiple 2025 papers) →
   EfficientNet/ResNet is the right family; don't waste time on big transformers.
5. **Efficiency prize angle:** a compact model (EfficientNet-B0/B3, few slices)
   tuned for speed could target the separate efficiency award — a realistic
   secondary goal for us vs. the accuracy leaders.

## PUBLIC ASSETS we can reuse (don't redo common work) — verified via `kaggle datasets list`
The community has already published most building blocks. Reuse instead of rebuild:

**Ready-made labels (skip our extractor; all beat our ~0.6 rules):**
- `stevenleehans/rsna-knee-llm-report-labels` — 39 votes, usability 0.71 (most trusted)
- `yunusgmsoy/rsna-knee-llm-labels-4-source-merged` — fusion of 4 label sources
- `rayanbabur/rsna-knee-2026-calibrated-soft-targets` — calibrated soft labels
- `pilkwang/rsna-knee-llm-labels`, `lixin73/...GPT-5.6-Sol`, `laymond/...Qwen3-8B`

**Preprocessed images (skip the 100GB DICOM ETL):**
- `alenic/rsna-knee-abnormality-detection-jpeg-224x224` (11GB, usability 0.88)
- `barun2104/rsna-knee-mri-processed-3d-volumes` (17GB 3D volumes)

**Pretrained backbones / heads / full models (skip most training):**
- `dreaddevelopment/raptor-knee-widedense` — trained 12-finding model, usability 0.94
- `tonylica/rsna-knee-bend-dinov3-0917-repro-assets` — assets to reproduce a 0.917 solution
- `pilkwang/rsna-knee-weights`, `antoinegg1/...RadImageNet-heads-v15`,
  `girishbose/dinov2-vits14-rsna-knee`, `mattiaangeli/...RadImageNet-folds`

**Implication:** the "assemble from public parts" path likely gets us to a strong
LB fast: (best public labels) + (preprocessed images) + (RadImageNet/DINO backbone)
+ our own MIL head/training. Legal on Kaggle if the datasets are public. Verify
licenses before using in a final/prize submission.

## Potential breakthroughs (highest expected value for us)
- **Label quality is still the #1 lever** (blueprint's own conclusion: ceiling
  ~0.83, gap to ~0.90 image-vs-report). A better multilingual LLM extractor +
  multi-source fusion is the single biggest gain.
- **MRNet-pretrained backbone** = cheap, legal, likely a real bump on ACL/meniscus.
- **SSL on our 820k slices** = bigger effort, potential real edge since few teams
  will bother.
- **Ensemble diversity** across planes/backbones/label-sources (metric is
  ranking-only → mean-ensembling is free and safe).

## Our constraints (verified)
- Test has NO reports (labels are train-only; scored model is image-based). ✅
- Kernel internet OFF; `machine_shape: NvidiaTeslaT4`. ✅ (winner used a local LLM
  offline for labels — we can pre-generate labels offline instead.)
- GPU session limit 2; long runs need the adaptive-budget pattern.
- Terminal interrupts long foreground commands → background + poll.
