# LLM Classification Finetuning — Design Doc

**Competition:** LLM Classification Finetuning
**Metric:** (log loss / accuracy — confirm from overview) | **Status:** kernel v3, minimal
**Type:** Code Competition (notebook submission)

## Domain research
Fine-tune / prompt an LLM to classify text (e.g., which of two chatbot responses
a human prefers). Phase 3 target that leans on transformer fine-tuning and,
where it fits, parameter-efficient methods (LoRA).

## Prior art
Standard route: fine-tune a mid-size pretrained transformer as a classifier, or
LoRA-adapt a larger base to fit GPU memory. Careful prompt/pair formatting and
class-balanced training matter.

## Design decisions
| Decision | Choice | Rationale |
|---|---|---|
| Model | Pretrained transformer classifier; LoRA if base too large | Fits Kaggle GPU memory budget |
| Compute | Kaggle T4 kernel | P100 breaks shipped PyTorch |
| Submission | Notebook/code submission | Code competition |

## Status / next
Kernel v3 drafted, not scored. Next: confirm the exact task format + metric from
the overview, decide full fine-tune vs LoRA based on base size, run on the GPU
kernel, submit. Lower priority than Watson (to 0.85+) and the open Playground
episode.
