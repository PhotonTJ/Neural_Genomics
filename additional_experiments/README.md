# nDNA — ICLR experiment suite

Fourteen experiments backing the ICLR submission, each run on **two base models**
(28 runs total). Everything writes `.npz` plus a JSON provenance manifest into
`results/`, so any number in the paper traces back to the run that produced it.

The suite is self-contained: it does not import from the parent repo, and it recomputes
rather than re-reads, because the parent repo's stored artefacts are min–max normalised
per plot and absolute κ / ℒ / ‖v‖ are not recoverable from them.

---

## Layout

```
ndna_iclr/
├── config.py              models, probes, sweeps, triad hyper-parameters
├── core/
│   ├── triad.py           κ, ℒ, ‖v‖ on the Fisher–Rao sphere  (the paper's §3)
│   ├── dtw.py             banded DTW, joint normalisation, acceptance threshold
│   ├── models.py          load / quantize / prune / gain-trade / LoRA / merge
│   ├── behavior.py        perplexity, ROUGE-L, BLEU, EM, F1, accuracy
│   ├── probes.py          the ten FLAN tasks + probe corpora
│   ├── io.py              NPZ + manifest (git hash, config, timestamp)
│   └── harness.py         shared CLI and run loop
├── experiments/           exp01 … exp14, one file each
├── results/               *.npz + *.json          (git-ignored except .gitkeep)
├── figures/  tables/      generated artefacts
├── scripts/               run_all.sh, smoke.py, make_tables.py
└── tests/test_core.py     model-free tests; run these first
```

## Install

```bash
pip install -r requirements.txt
python tests/test_core.py          # 8 tests, no GPU, no torch needed
python scripts/smoke.py            # one tiny model end-to-end, needs torch
```

## Models

Two recent, small-enough-to-iterate bases, set in `config.py`:

| key | id |
|---|---|
| `qwen3_4b` | `Qwen/Qwen3-4B` |
| `gemma3_1b` | `google/gemma-3-1b-pt` |

Override with `NDNA_MODELS=qwen3_4b` or by editing `config.MODELS`. Aligned siblings
used by exp03/04 are in `config.SIBLINGS`.

---

## The fourteen experiments

Ordered so that stopping early still leaves a coherent paper.

### Tier A — integrity

| | experiment | fills | cost |
|---|---|---|---|
| 01 | `dose_response` — quantization 5 doses × pruning 8 doses, triad + perplexity at each. Tests **monotonicity**, not just endpoints. | dose–response table | ~1 day, unattended |
| 02 | `operation_classification` — recover which operation was applied, family-held-out CV, with single-quantity ablations | recovery table | ~half day |

### Tier B — makes existing claims operational

| | experiment | fills | cost |
|---|---|---|---|
| 03 | `behavior_vs_geometry` — behaviour on x, DTW on y. **The only place BLEU/ROUGE appear.** | the money figure | ~half day |
| 04 | `task_stability` — ten FLAN tasks → θ, *and* the joint normalisation that lets θ apply to §5 | stability table, θ | ~half day |
| 08 | `absolute_triad` — raw per-layer arrays under one stated convention + commitment depth ℓ*(ε) | absolute table | hours |

### Tier C — kills specific reviewer objections

| | experiment | objection killed | cost |
|---|---|---|---|
| 05 | `invariance` — RMSNorm gain trade; triad flat in *c*, raw hidden length scales with *c* | "why not hidden states?" | hours |
| 06 | `tuned_lens` — recompute under a trained per-layer translator | "logit lens is unreliable early" | ~1 day |
| 07 | `patching` — correlate triad against activation-patching effect | "correlational, not causal" | ~1 day |
| 09 | `stats_control` — paired Wilcoxon across matched folds + matched-*k* PCA | "is the gap significant / just dimensionality?" | minutes |

### Tier D — use cases (what the measurement *enables*)

| | experiment | claim | cost |
|---|---|---|---|
| 10 | `provenance` — which base is this checkpoint built on? Reports **margins**, includes hard negatives and an undeclared-distillation case | licensing / merge disclosure | ~half day |
| 11 | `merge_screening` — does pre-merge parent distance predict merged quality? | screen merges before running them | ~1 day |
| 12 | `collapse_early_warning` — forecast fitted on generations 1–5 only, report **lead time** vs a perplexity baseline | stop a self-training loop early | 1–2 days |
| 13 | `compression_knee` — geometric knee predicts behavioural knee; transfers across models | pick a compression budget without eval | minutes (rides on exp01) |
| 14 | `alignment_stage` — SFT-only or SFT-then-DPO? | forensics beyond "which base" | ~half day |

---

## Running

```bash
# one experiment, both models
python experiments/exp01_dose_response.py

# subset, dry run first
python experiments/exp04_task_stability.py --models qwen3_4b --dry-run

# everything, in dependency order
bash scripts/run_all.sh

# aggregation-only steps (no model loaded)
python experiments/exp02_operation_classification.py --aggregate
python experiments/exp09_stats_control.py
python experiments/exp13_compression_knee.py

# emit LaTeX tables from whatever results exist
python scripts/make_tables.py
```

Every driver accepts `--models --seeds --probe --n-prompts --device --batch-size
--keep-last-k --curvature --dry-run`. Start with `--dry-run` to see the plan.

**Dependencies between experiments.** 02 → 09, 01 → 13. Everything else is standalone.

---

## Conventions that matter

**Joint normalisation.** DTW costs are comparable only within a set normalised together.
`dtw.pairwise_matrix` therefore takes the whole set at once and normalises across it;
never normalise per curve. This is why exp04 profiles the task prototypes *and* the
lifecycle variants in one call — it is the fix for the scope limitation recorded in the
paper's Appendix A.8.

**Same text, same order.** A base model and its modified counterpart are always profiled
on identical prompts. `probes.probe_hash` is stored so this is checkable after the fact.

**Curvature estimator.** `--curvature turn` (default) is the intrinsic turning-angle form
normalised by Fisher–Rao arc length; `chord` is the extrinsic estimator of Eq. 6.
exp08 computes both and reports their correlation.

**Degenerate triangles are excluded, not zeroed.** Counting them as zero curvature biases
κ downward exactly where the trajectory stalls.

**BLEU / ROUGE / F1 are an axis, not a result.** They appear only in exp03, as the
behavioural x-axis against which geometric change is plotted. They are not evidence of
model quality and should not be reported as such.

---

## Honest limitations of this suite

- `models.quantize` is round-to-nearest and `models.prune` is magnitude-based. Both are
  transparent rather than state of the art. **Name the method in every caption**; if you
  need competitive 2-bit or 70% numbers, swap in GPTQ/AWQ/AQLM or SparseGPT/Wanda.
- `models.rescale_residual_gains` applies one half of the gain trade. exp05 therefore
  *measures* the logit deviation instead of assuming the edit is function-preserving —
  read `max_logit_deviation` before interpreting that run.
- exp14's second stage is preference-shaped continued fine-tuning, not a full DPO
  objective. It tests whether a second stage is geometrically detectable, not which loss
  produced it.
- exp12 is the only experiment needing a fresh training loop; it is the first to cut if
  time is short.
- Absolute κ values depend on τ, `keep_last_k` and the probe. Do not compare absolute
  numbers across runs with different settings — the manifest records all three.
