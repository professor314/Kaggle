# Public Assets — License Audit (RSNA Knee)

Task A.0 / Requirement 6. Human-verifiable record of every public Kaggle dataset
or model we reuse. **Licenses below were read from the live Kaggle API metadata
(`kaggle datasets metadata <slug>`) on 2026-09-02.** Re-confirm before the final
prize submission.

License safety: `CC0-1.0` = public domain, no attribution needed (safest) ·
`MIT` = permissive, keep the notice · `CC BY 4.0` = safe with attribution ·
`CC BY-NC*` = NON-commercial, NOT prize-safe · `other`/`Unknown` = must read the
page's own terms before prize use.

How to verify yourself: open the URL → right sidebar **License** field. Or run
`kaggle datasets metadata <owner>/<slug> -p <tmp>` and read `info.licenses`.

Status: ✅ CLEARED (prize-safe) · ⚠️ REVIEW (terms need reading) · ⛔ BLOCKED.

## Competition data (baseline terms)
| Asset | URL | License / Terms | Status |
|---|---|---|---|
| RSNA Knee competition data | https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/data | Competition rules (RSNA), research use | ✅ (competition use) |

## Ready-made labels (report-derived) — verified 2026-09-02
| Asset | URL | Author | License | Gold macro-AUC | Status |
|---|---|---|---|---|---|
| rsna-knee-llm-report-labels **(best; use `llm_labels_v4_blend.csv`)** | https://www.kaggle.com/datasets/stevenleehans/rsna-knee-llm-report-labels | stevenleehans | **CC0-1.0** | **0.8927** | ✅ CLEARED |
| rsna-knee-llm-labels-4-source-merged | https://www.kaggle.com/datasets/yunusgmsoy/rsna-knee-llm-labels-4-source-merged | yunusgmsoy | **CC0-1.0** | (merge) | ✅ CLEARED |
| rsna-knee-2026-calibrated-soft-targets | https://www.kaggle.com/datasets/rayanbabur/rsna-knee-2026-calibrated-soft-targets | rayanbabur | **CC0-1.0** | (soft) | ✅ CLEARED |
| rsna-knee-llm-labels | https://www.kaggle.com/datasets/pilkwang/rsna-knee-llm-labels | pilkwang | **CC0-1.0** | 0.8700 | ✅ CLEARED |
| rsna-knee-llm-report-labels-sol56 | https://www.kaggle.com/datasets/lixin73/rsna-knee-llm-report-labels-sol56 | lixin73 | **CC0-1.0** | 0.8347 (decorrelated, good blend) | ✅ CLEARED |

## Preprocessed images — verified 2026-09-02
| Asset | URL | Author | License | Status |
|---|---|---|---|---|
| rsna-knee-abnormality-detection-jpeg-224x224 | https://www.kaggle.com/datasets/alenic/rsna-knee-abnormality-detection-jpeg-224x224 | alenic | **MIT** | ✅ CLEARED (keep notice) |
| rsna-knee-mri-processed-3d-volumes | https://www.kaggle.com/datasets/barun2104/rsna-knee-mri-processed-3d-volumes | barun2104 | **other** | ⚠️ REVIEW (read page terms before prize use) |

## Pretrained backbones / heads / full models — verified 2026-09-02
| Asset | URL | Author | License | Notes | Status |
|---|---|---|---|---|---|
| **raptor-knee-widedense** (trained 12-finding CoAtNet) | https://www.kaggle.com/datasets/dreaddevelopment/raptor-knee-widedense | dreaddevelopment | **CC0-1.0** | **0.924 public LB / 0.9167 gold single model.** Companion notebook: `dreaddevelopment/knee-mri-twelve-findings-from-a-single-model`. arch `coatnet_rmlp_2_rw_384`, res 384, 3-slice windows, attention pooling head, 64-slice stack | ✅ CLEARED |
| rsna-knee-bend-dinov3-0917-repro-assets | https://www.kaggle.com/datasets/tonylica/rsna-knee-bend-dinov3-0917-repro-assets | tonylica | ⚠️ verify if used | not yet needed | ⚠️ REVIEW |
| dinov2-vits14-rsna-knee | https://www.kaggle.com/datasets/girishbose/dinov2-vits14-rsna-knee | girishbose | ⚠️ verify if used | not yet needed | ⚠️ REVIEW |

## Our own assets (created by us)
| Asset | URL | License | Status |
|---|---|---|---|
| seanconnolly/timm-efficientnet-b0-weights | https://www.kaggle.com/datasets/seanconnolly/timm-efficientnet-b0-weights | timm Apache-2.0 upstream | ✅ |
| seanconnolly/mdeberta-v3-base-mnli-xnli | https://www.kaggle.com/datasets/seanconnolly/mdeberta-v3-base-mnli-xnli | MIT upstream | ✅ |
| seanconnolly/rsna-knee-nli-weaklabels | https://www.kaggle.com/datasets/seanconnolly/rsna-knee-nli-weaklabels | our derivation (comp terms) | ✅ |

## Conclusions
- **A.1 early submission** → `raptor-knee-widedense` is **CC0** and prize-safe.
  It is also stronger than expected (0.924 single-model), so it's our fast real score.
- **Labels** → `stevenleehans` v4 blend (CC0, 0.8927 gold) is the best label source;
  `pilkwang` + `lixin73` (both CC0) are good decorrelated blend partners.
- **Images** → `alenic` JPEG-224 (MIT) is prize-safe; `barun2104` 3D volumes is
  "other" → only use after reading its page terms.
- Re-confirm all of the above before the 2026-10-22 final submission.
