# Cross-competition lessons

Short, VERIFIED gotchas that apply beyond one competition. Add to this whenever something bites
us. Each line: the lesson, then where it came from.

## Kaggle platform
- Competition files mount at `/kaggle/input/competitions/<slug>` now, not `/kaggle/input/<slug>`.
  Hard-coded paths broke 7 notebooks. Resolve the path, don't hard-code it. (portfolio fix, 2026-11-12)
- Pushing a kernel re-runs it. Changing title or visibility via CLI = a new run (GPU quota for GPU
  notebooks). Retitling also changes the URL slug, and punctuation matters ("TF-IDF" -> `tf-idf`). (2026-11-12/13)
- Keep the reference notebook's pinned `docker_image` in kernel-metadata.json. Without it the
  Gemma wheelhouse failed to install on Python 3.13. (gemma, 2026-11-13)
- GPU model comes from `machine_shape` (`NvidiaTeslaT4`, `NvidiaL4`), not `accelerator`. (watson, 2026-08-31)
- Batch limits: 5 concurrent CPU sessions, 2 GPU sessions. Pushes beyond that fail; retry later. (2026-11-12)
- Kaggle only accepts tags from its fixed list; custom keywords are dropped silently-ish. (2026-11-12)
- Code competitions: submit from the kernel output (`kaggle competitions submit <c> -k <kernel> -v <n> -f <file>`).

## Windows / local environment
- PowerShell 5 `Set-Content -Encoding utf8` writes a BOM; the kaggle CLI rejects JSON with a BOM.
  Write with `[IO.File]::WriteAllText(path, text, UTF8Encoding($false))`. (2026-11-12)
- Never download big competition zips into `My Drive` (it syncs to Google Drive). Gemma's was 10.8 GB.
  Check `kaggle competitions files` sizes first and read big data inside a Kaggle kernel. (2026-11-13)
- A cancelled `kaggle competitions download` can keep running in the background. Check for a
  stray `kaggle` process before retrying. (2026-11-13)
- The kaggle CLI's output is easy to lose in PowerShell; a "lost" submit may have succeeded.
  Check `kaggle competitions submissions <c>` before resubmitting. (pokemon double submit, 2026-11-13)

## Modeling
- Correlated gradient boosters don't blend well; diversity (linear on target encodings, different
  feature sets) matters more. (S6E9, spaceship, house-prices)
- OOF target encoding broke the House Prices plateau. (house-prices, 2026-10-24)
- When every class answers the same prompts, use GroupKFold on the prompt id or CV leaks. (name-that-llm, 2026-11-13)
- Playground public LB has tracked OOF AUC within ~0.001 so far. Trust OOF. (S6E9, S6E11)
