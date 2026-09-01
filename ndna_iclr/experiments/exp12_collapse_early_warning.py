"""Exp 12 -- how many generations before a self-training loop degenerates?

The scientifically strongest result available, and the only one here that needs a fresh
training loop rather than a re-aggregation. A model is repeatedly fine-tuned on its own
generations; we record the triad and an output-quality measure at every generation.

The claim under test is a *forecast*, not a description: fit the triad trend on the first
`fit_generations` only, extrapolate to the generation at which quality crosses a
threshold, and compare the lead time against the same forecast made from perplexity.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C                                            # noqa: E402
from core import harness, models, probes, triad, behavior      # noqa: E402


def _distinct_n(texts, n=3):
    grams, tot = set(), 0
    for t in texts:
        w = t.split()
        for i in range(max(0, len(w) - n + 1)):
            grams.add(tuple(w[i:i + n])); tot += 1
    return len(grams) / max(tot, 1)


def _forecast_cross(series, threshold, fit_upto):
    """Linear extrapolation of series[:fit_upto] to the first crossing of threshold."""
    y = np.asarray(series[:fit_upto], float)
    if len(y) < 2:
        return np.nan
    x = np.arange(len(y), dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    if abs(slope) < 1e-12:
        return np.nan
    g = (threshold - intercept) / slope
    return float(g) if g > 0 else np.nan


def run(model_key, model_id, seed, args):
    from peft import PeftModel
    probe, _ = probes.load_probe(args.probe, n=args.n_prompts, seed=seed, cache_dir=C.CACHE)
    seeds_txt, _ = probes.load_probe(args.probe, n=args.n_train, seed=seed + 1,
                                     cache_dir=C.CACHE)
    tcfg = harness.triad_cfg(args)
    work = os.path.join(C.CACHE, f"collapse_{model_key}_seed{seed}")
    os.makedirs(work, exist_ok=True)

    kap, ln, bel, ppl, div = [], [], [], [], []
    corpus = list(seeds_txt)
    adapter = None

    for gen in range(args.generations + 1):
        base, tok = models.load(model_id, device=args.device, cache_dir=C.CACHE)
        m = PeftModel.from_pretrained(base, adapter).eval() if adapter else base

        pr = triad.profile_model(m, tok, probe, tcfg, device=args.device)
        kap.append(float(np.mean(pr["kappa"])))
        ln.append(float(pr["length_total"]))
        bel.append(float(np.mean(pr["belief"])))
        ppl.append(behavior.perplexity(m, tok, probe, device=args.device,
                                       batch_size=args.batch_size))
        gen_txt = behavior.generate(m, tok, probe[:args.n_gen], device=args.device,
                                    max_new_tokens=48, batch_size=args.batch_size)
        div.append(_distinct_n(gen_txt))
        print(f"     gen {gen}: kappa={kap[-1]:.5f} L={ln[-1]:.3f} ppl={ppl[-1]:.2f} "
              f"distinct3={div[-1]:.3f}", flush=True)
        models.free(m if adapter else None, base)

        if gen == args.generations:
            break
        corpus = gen_txt if len(gen_txt) >= 32 else corpus
        adapter = os.path.join(work, f"gen{gen+1}")
        models.lora_finetune(model_id, corpus, adapter, steps=args.steps,
                             device=args.device, cache_dir=C.CACHE, seed=seed + gen)

    div = np.array(div); kap = np.array(kap); ln = np.array(ln); ppl = np.array(ppl)
    thr = args.quality_drop * div[0]
    actual = np.nonzero(div <= thr)[0]
    actual = int(actual[0]) if actual.size else -1

    # forecast from geometry vs from perplexity, both using only the first k generations
    scale = lambda v: (v - v[0]) / (abs(v[0]) + 1e-12)
    g_geo = _forecast_cross(scale(ln), (thr - div[0]) / (abs(div[0]) + 1e-12), args.fit_generations)
    g_ppl = _forecast_cross(scale(ppl), (thr - div[0]) / (abs(div[0]) + 1e-12), args.fit_generations)

    arrays = {
        "generation": np.arange(len(div)),
        "kappa_mean": kap, "length_total": ln, "belief_mean": bel,
        "perplexity": ppl, "distinct3": div,
        "quality_threshold": thr,
        "actual_collapse_generation": actual,
        "forecast_from_geometry": g_geo,
        "forecast_from_perplexity": g_ppl,
        "fit_generations": args.fit_generations,
        "lead_time_geometry": (actual - args.fit_generations) if actual >= 0 else np.nan,
    }
    return arrays, "recursive self-training; forecast fitted on early generations only"


if __name__ == "__main__":
    p = harness.base_parser("exp12_collapse_early_warning", __doc__)
    p.add_argument("--generations", type=int, default=C.COLLAPSE_GENERATIONS)
    p.add_argument("--fit-generations", type=int, default=5)
    p.add_argument("--steps", type=int, default=120)
    p.add_argument("--n-train", type=int, default=256)
    p.add_argument("--n-gen", type=int, default=64)
    p.add_argument("--quality-drop", type=float, default=0.6,
                   help="collapse = distinct-3 falls to this fraction of generation 0")
    a = p.parse_args()
    harness.run_over_models("exp12_collapse_early_warning", a, run)
