"""Exp 11 -- can we predict whether a merge is worth doing, before doing it?

Merging is cheap to attempt and expensive to evaluate. If the pre-merge geometric
distance between two parents predicts merged-model quality, the triad becomes a screening
tool: two forward passes instead of a merge plus an evaluation.

Protocol: LoRA-adapt one base model on several disjoint corpora to create parents,
measure all parent-parent DTW distances, merge every pair, evaluate each merge, and
regress quality on the pre-merge distance. Also records the asymmetry of each merge --
distance to parent A versus parent B -- which is the quantitative form of "merging does
not interpolate".
"""
import itertools
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C                                            # noqa: E402
from core import harness, models, probes, triad, dtw, behavior  # noqa: E402

PARENT_CORPORA = ["cnn_dailymail", "wmt16", "imdb", "common_gen"]


def _spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return float("nan")
    ra = np.argsort(np.argsort(a[ok])).astype(float)
    rb = np.argsort(np.argsort(b[ok])).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    d = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / d) if d else float("nan")


def run(model_key, model_id, seed, args):
    from peft import PeftModel
    probe, _ = probes.load_probe(args.probe, n=args.n_prompts, seed=seed, cache_dir=C.CACHE)
    tcfg = harness.triad_cfg(args)
    work = os.path.join(C.CACHE, f"merge_{model_key}_seed{seed}")
    os.makedirs(work, exist_ok=True)

    # ---- parents
    names, adapters = [], []
    for corpus in PARENT_CORPORA:
        try:
            txt, _ = probes.load_probe(corpus, n=args.n_train, seed=seed, cache_dir=C.CACHE)
        except Exception as e:
            print(f"     corpus {corpus} skipped: {e}"); continue
        out = os.path.join(work, f"parent_{corpus}")
        if not os.path.exists(os.path.join(out, "adapter_model.safetensors")):
            models.lora_finetune(model_id, txt, out, steps=args.steps,
                                 device=args.device, cache_dir=C.CACHE, seed=seed)
        names.append(corpus); adapters.append(out)

    def profile_adapter(path):
        base, tok = models.load(model_id, device=args.device, cache_dir=C.CACHE)
        m = PeftModel.from_pretrained(base, path).eval()
        pr = triad.profile_model(m, tok, probe, tcfg, device=args.device)
        ppl = behavior.perplexity(m, tok, probe, device=args.device,
                                  batch_size=args.batch_size)
        models.free(m, base)
        return pr, ppl

    p_profs, p_ppl = [], []
    for a in adapters:
        pr, pp = profile_adapter(a)
        p_profs.append(pr); p_ppl.append(pp)

    # ---- merges
    pairs, m_profs, m_ppl = [], [], []
    for i, j in itertools.combinations(range(len(adapters)), 2):
        out = os.path.join(work, f"merge_{names[i]}_{names[j]}")
        try:
            models.fisher_weighted_merge(adapters[i], adapters[j], out)
            pr, pp = profile_adapter(out)
        except Exception as e:
            print(f"     merge {names[i]}+{names[j]} skipped: {e}"); continue
        pairs.append((i, j)); m_profs.append(pr); m_ppl.append(pp)

    if not m_profs:
        raise RuntimeError("no merges were produced")

    X = dtw.stack_triad(p_profs + m_profs, n=32)
    Cm, _ = dtw.pairwise_matrix(X, band=0.2, normalise="minmax")
    npar = len(p_profs)

    pre, d1, d2, asym, qual = [], [], [], [], []
    for k, (i, j) in enumerate(pairs):
        pre.append(Cm[i, j])
        a, b = Cm[npar + k, i], Cm[npar + k, j]
        d1.append(a); d2.append(b)
        asym.append(abs(a - b) / max(a, b) if max(a, b) > 0 else 0.0)
        qual.append(-m_ppl[k])                       # higher is better

    arrays = {
        "parents": np.array(names), "parent_ppl": np.array(p_ppl),
        "pair_i": np.array([p[0] for p in pairs]), "pair_j": np.array([p[1] for p in pairs]),
        "pre_merge_dtw": np.array(pre),
        "d_to_parent_a": np.array(d1), "d_to_parent_b": np.array(d2),
        "asymmetry": np.array(asym),
        "merge_ppl": np.array(m_ppl),
        "rho_pre_vs_quality": _spearman(pre, qual),
        "mean_asymmetry": float(np.mean(asym)),
        "median_asymmetry": float(np.median(asym)),
        "n_asym_above_10pct": int(np.sum(np.array(asym) > 0.10)),
        "n_merges": len(pairs),
    }
    return arrays, "pre-merge parent distance vs merged-model quality; merge asymmetry"


if __name__ == "__main__":
    p = harness.base_parser("exp11_merge_screening", __doc__)
    p.add_argument("--steps", type=int, default=150)
    p.add_argument("--n-train", type=int, default=256)
    a = p.parse_args()
    harness.run_over_models("exp11_merge_screening", a, run)
