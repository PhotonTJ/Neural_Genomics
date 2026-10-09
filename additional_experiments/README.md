# Additional experiments — nDNA

Fourteen experiments supporting the nDNA paper. **All fourteen are implemented; four have
been run and only three produced usable results.** This README records what survived, what
did not, and why — so that nothing in `results/` needs a caveat that isn't written down
here.

Everything writes JSON plus a provenance manifest (git hash, config, array shapes) into
`results/`, so any number in the paper traces back to the run that produced it.

---

## Status of every experiment

| | Experiment | Status | Models run |
|---|---|---|---|
| 01 | `dose_response` | ❌ **results deleted** — see below | — |
| 02 | `operation_classification` | ❌ **results deleted** — null | — |
| 03 | `behavior_vs_geometry` | ⚠️ **partial** — DTW good, ΔPPL unusable on compressed variants | 4 |
| 04 | `task_stability` | ✅ **good** — clean and consistent 4/4 | 4 |
| 05 | `invariance` | ❌ **results deleted** — test was invalid; needs a code fix | — |
| 06 | `tuned_lens` | ⬜ not run | — |
| 07 | `patching` | ⚠️ **raw data good, derived stats confounded** | 4 |
| 08 | `absolute_triad` | ✅ **good** — raw per-layer profiles under one convention | 4 |
| 09 | `stats_control` | ❌ **results deleted** — null, underpowered | — |
| 10 | `provenance` | ⬜ not run | — |
| 11 | `merge_screening` | ⬜ not run | — |
| 12 | `collapse_early_warning` | ⬜ not run | — |
| 13 | `compression_knee` | ❌ **results deleted** — depends on exp01 | — |
| 14 | `alignment_stage` | ⬜ not run | — |

Models: `gemma3_1b`, `llama31_8b`, `qwen3_4b`, `qwen3_8b`. Smoke runs on
`pythia-160m` and `tiny-gpt2` were deleted — they were sanity checks, not results.

---

## What survived, and what it says

### exp04 — task stability ✅

The strongest result in the suite. Ten FLAN tasks, 90 pairwise DTW distances per model,
acceptance threshold θ fixed at the 90th percentile of task-induced variation, then
lifecycle variants compared on the same joint normalisation.

| model | θ (q90) | quant-4bit | prune-40% | instruct sibling |
|---|---|---|---|---|
| gemma3-1b | 0.0534 | 0.0064 ✗ | **0.2540 ✓** | 0.0262 ✗ |
| llama31-8b | 0.0162 | 0.0010 ✗ | **0.1602 ✓** | 0.0012 ✗ |
| qwen3-4b | 0.0535 | 0.0120 ✗ | **0.1702 ✓** | 0.0098 ✗ |
| qwen3-8b | 0.0422 | 0.0086 ✗ | **0.2524 ✓** | 0.0106 ✗ |

✓ = exceeds θ. Structured pruning clears the threshold by 4–10× on every model;
4-bit quantization and instruction-tuned siblings do not.

> **⚠️ This contradicts the alignment claim.** Instruction-tuned siblings fall *below* the
> task-variation threshold on all four models. On llama31-8b the sibling sits at 0.00122
> against a task-pair minimum of 0.00125 — quieter than the closest pair of tasks. By the
> suite's own pre-registered rule, **instruction tuning is not a detectable change.**
> Earlier numbers suggesting an ordered SFT/DPO relationship came from the parent repo's
> per-block-normalised similarity reports; exp04 is the correctly-normalised version and
> disagrees. Before treating this as final, verify the sibling checkpoint actually loaded
> rather than the base being profiled twice.

### exp08 — absolute triad ✅

Raw per-layer κ, ℒ and ‖v‖ for four models under a single stated convention
(τ = 1, `keep_last_k`, probe hash recorded in the manifest), plus both curvature
estimators and the commitment-depth curve. This is what makes absolute values
reproducible — the parent repo ships only plotted, per-figure-normalised profiles, which
is why every earlier table quoting absolute κ inherited a scale ambiguity.

### exp03 — behaviour vs geometry ⚠️

**The DTW column is sound.** Siblings sit at 0.0013–0.0265 while pruning reaches
0.18–0.51, consistent with exp04.

**The ΔPPL column is only usable for `quant_4bit` and the siblings** (+1.2 to +15.1).
The `quant_2bit`, `prune_30` and `prune_60` rows show perplexity deltas of 10⁵–10¹²,
because they use the same crude compression as exp01. Do not plot those rows against
behaviour until exp01 is fixed.

### exp07 — patching ⚠️

**Keep the raw arrays** (`patch_effect`, `length`, `belief`, `tail_length`): the patching
measurement is real and expensive to reproduce.

**Do not use the derived statistics as they stand.** Two problems:

1. ρ(tail, patch) = −0.98 to −0.99 across all four models is almost certainly a **depth
   confound** — `tail_length` decreases monotonically with depth and patch effect
   increases monotonically with depth, so they are anti-correlated by construction.
   Recompute as a partial correlation controlling for layer index.
2. `exclusion_precision = 0.000` everywhere is a **mis-specified test, not a refutation of
   the commitment bound.** The mask tests layer ℓ itself rather than the layers after it;
   and more fundamentally, the bound constrains the model's *own* trajectory, while
   patching injects a state the model would never produce. The bound says nothing about
   that. This test needs redesigning or dropping.

---

## Why the deleted results were deleted

**exp01 — dose–response.** Perplexity reached 10⁶–10¹² at the higher doses
(gemma at 2-bit: 2.75 × 10¹²; llama at 10% structured pruning: 4.3 × 10⁵). The models were
destroyed, so the geometry measured rubble. Spearman ρ came out inconsistent and often
*positive* — the opposite of the predicted decrease.

*Cause:* `models.quantize` is round-to-nearest and `models.prune` is structured magnitude
pruning. Both are transparent but far too crude at these doses. *Fix:* swap in GPTQ/AWQ
and SparseGPT/Wanda, and cap the sweep where perplexity stays within roughly 2× of
baseline. Name the method in every caption.

**exp13 — compression knee.** Reads exp01's output. Invalid for the same reason.

**exp05 — invariance.** `max|Δlogit|` was 27–1969, so the edit was **not
function-preserving** and the test could not measure invariance. The script's built-in
guard caught this rather than letting a false result through.

*Fix, and it is clean:* RMSNorm is scale-invariant, so **do not touch the norm gains at
all.** Instead multiply the embedding output and every block's output projection by `c`.
The residual stream becomes `c·x` throughout, every RMSNorm output is unchanged, and the
function is exactly preserved. `core/models.rescale_residual_gains` needs rewriting on
those lines.

**exp02 / exp09 — operation classification and paired stats.** Null. Triad accuracy
66.3 ± 11.0 against κ-only 69.4 ± 7.0 — the triad did **not** beat its own single
components, and κ alone was nominally best. Every Wilcoxon came back p > 0.31 with only
1–3 effective paired folds. The perplexity, KL and CKA baselines did not make it into the
output at all. Nothing here supports a claim, so nothing here should appear in a table.

---

## Layout

```
additional_experiments/
├── README.md              this file
├── config.py              models, probes, sweeps, triad hyper-parameters
├── core/
│   ├── triad.py           κ, ℒ, ‖v‖ on the Fisher–Rao sphere
│   ├── dtw.py             banded DTW, joint normalisation, acceptance threshold
│   ├── models.py          load / quantize / prune / gain-trade / LoRA / merge
│   ├── behavior.py        perplexity, ROUGE-L, BLEU, EM, F1, accuracy
│   ├── probes.py          the ten FLAN tasks + probe corpora
│   ├── io.py              JSON + manifest (git hash, config, timestamp)
│   └── harness.py         shared CLI and run loop
├── experiments/           exp01 … exp14
├── results/               surviving runs only
├── scripts/               run_all.sh, smoke.py, make_tables.py, npz_to_json.py
└── tests/test_core.py     model-free tests; run these first
```

## Running

```bash
pip install -r requirements.txt
python tests/test_core.py          # 8 tests, no GPU, no torch needed
python scripts/smoke.py            # tiny model end-to-end, needs torch

python experiments/exp04_task_stability.py --models qwen3_4b --dry-run
bash scripts/run_all.sh
```

Every driver accepts `--models --seeds --probe --n-prompts --device --batch-size
--keep-last-k --curvature --dry-run`. Start with `--dry-run`, which works without torch.

**Dependencies:** 02 → 09, 01 → 13. Everything else is standalone.

---

## Conventions that matter

**Joint normalisation.** DTW costs are comparable only within a set normalised together.
`dtw.pairwise_matrix` takes the whole set at once; never normalise per curve. A test
(`test_joint_normalisation_is_joint`) fails if that is reintroduced. exp04 profiles the
task prototypes *and* the lifecycle variants in one call for exactly this reason.

**Same text, same order.** A base model and its modified counterpart are always profiled
on identical prompts. `probes.probe_hash` is stored so this is checkable afterwards.

**Curvature estimator.** `--curvature turn` (default) is the intrinsic turning-angle form
normalised by Fisher–Rao arc length; `chord` is the extrinsic estimator. exp08 computes
both and reports their correlation.

**Degenerate triangles are excluded, not zeroed.** Counting them as zero curvature biases
κ downward exactly where the trajectory stalls.

**BLEU / ROUGE / F1 are an axis, not a result.** They appear only in exp03, as the
behavioural x-axis. They are not evidence of model quality.

**Absolute κ depends on τ, `keep_last_k` and the probe.** Never compare absolute values
across runs with different settings — the manifest records all three.

---

## Next steps, in priority order

1. **Fix `rescale_residual_gains`** (scale embeddings and block output projections; leave
   norm gains alone) and rerun exp05. Cheap, and it is the experiment that justifies
   working on predictions rather than hidden states.
2. **Replace the compression backends** and rerun exp01, then exp13.
3. **Recompute exp07's correlations** as partial correlations controlling for depth, and
   redesign or drop the exclusion test.
4. **Verify exp04's sibling checkpoints loaded correctly**, then decide whether the
   alignment claim changes.
5. Run exp06, 10, 11, 12, 14 — none has been attempted yet.

## Exp 15/16: matched logit-trajectory comparison and field direction

Pre-registration: `experiments/exp15_prereg.json` (checkpoints, prompt sets, channel sets, metrics).
Do not edit it after the first run; the evaluation refuses profiles made under a different file.

```
huggingface-cli login                                   # Llama-3.1 and Gemma-3 are gated
python experiments/exp15_launch.py --smoke              # 2 min pipeline check on tiny models
python experiments/exp15_launch.py                      # 6 checkpoints, one per GPU, then exp16
```

Outputs: `results/exp15/<family>__<role>__<set>.npz` + `.json` manifest per prompt set,
`results/exp15/prompt_sets.json` (the exact prompts), `results/exp16/summary.json` and
`results/exp16/table_matched.tex`. Logs: `logs/exp15/`. Re-running resumes from finished sets.
