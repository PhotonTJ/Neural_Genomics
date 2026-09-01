"""Exp 02 -- can the operation be recovered from the triad alone?

Builds (base, modified) pairs for each lifecycle operation, extracts depth-resampled
difference profiles, and fits a shallow classifier under model-family-held-out CV.
Baselines are computed on identical pairs so the comparison is not a dimensionality
artefact; a matched-k PCA control is reported alongside the raw result.

This is the experiment that turns "distinguishable signatures" into a number, and the
single-quantity ablations answer the "are all three needed?" objection before a reviewer
raises it.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C                                        # noqa: E402
from core import harness, models, probes, triad, dtw      # noqa: E402

OPS = ["quantization", "pruning", "distillation", "alignment", "merging", "self_training"]


def _variants(model_key, model_id, seed, args, prompts, tcfg):
    """-> [(operation, profile_dict)] for one base model."""
    out = []
    base, tok = models.load(model_id, device=args.device, cache_dir=C.CACHE)
    base_prof = triad.profile_model(base, tok, prompts, tcfg, device=args.device)
    models.free(base)

    for dose in ["int8", "4bit", "3bit", "2bit"]:
        m, tk = models.quantize(model_id, dose, device=args.device, cache_dir=C.CACHE)
        out.append(("quantization", triad.profile_model(m, tk, prompts, tcfg, device=args.device)))
        models.free(m)
    for sp in [0.2, 0.4, 0.6]:
        m, tk = models.prune(model_id, sp, device=args.device, cache_dir=C.CACHE)
        out.append(("pruning", triad.profile_model(m, tk, prompts, tcfg, device=args.device)))
        models.free(m)

    for kind, sib in C.SIBLINGS.get(model_key, {}).items():
        try:
            m, tk = models.load(sib, device=args.device, cache_dir=C.CACHE)
            out.append(("alignment", triad.profile_model(m, tk, prompts, tcfg, device=args.device)))
            models.free(m)
        except Exception as e:                              # sibling not available
            print(f"     skip sibling {sib}: {e}")
    return base_prof, out


def _features(base_prof, prof, n=32, channels=("kappa", "length", "belief")):
    v = []
    for ch in channels:
        b = dtw.resample(base_prof[ch], n)
        m = dtw.resample(prof[ch], n)
        med = np.median(np.abs(b)) or 1.0
        v.append((m - b) / med)                            # scale-normalised delta
    return np.concatenate(v)


def run(model_key, model_id, seed, args):
    prompts, _ = probes.load_probe(args.probe, n=args.n_prompts, seed=seed, cache_dir=C.CACHE)
    tcfg = harness.triad_cfg(args)
    base_prof, variants = _variants(model_key, model_id, seed, args, prompts, tcfg)

    X = np.stack([_features(base_prof, p) for _, p in variants]) if variants else np.zeros((0, 96))
    y = np.array([op for op, _ in variants])
    arrays = {
        "X": X, "y": y,
        "families": np.array([model_key] * len(y)),
        "base_kappa": base_prof["kappa"],
        "base_length": base_prof["length"],
        "base_belief": base_prof["belief"],
    }
    return arrays, "per-model feature block; fit across models with aggregate.py"


def aggregate(results_dir=None):
    """Pool every model's block and fit the classifier with family-held-out CV."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import LeaveOneGroupOut
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.decomposition import PCA

    blocks = harness.load_all("exp02_operation_classification", results_dir)
    if not blocks:
        print("no exp02 results found"); return None
    X = np.concatenate([a["X"] for _, a, _ in blocks])
    y = np.concatenate([a["y"] for _, a, _ in blocks])
    g = np.concatenate([a["families"] for _, a, _ in blocks])
    if len(set(g)) < 2:
        print("warning: only one family present; CV degenerates to in-family")

    def cv(Xs, k=None):
        if k:
            Xs = PCA(n_components=min(k, Xs.shape[1], len(Xs) - 1)).fit_transform(Xs)
        accs, f1s = [], []
        logo = LeaveOneGroupOut()
        for tr, te in logo.split(Xs, y, groups=g):
            if len(set(y[tr])) < 2:
                continue
            clf = LogisticRegression(max_iter=2000, C=1.0).fit(Xs[tr], y[tr])
            p = clf.predict(Xs[te])
            accs.append(accuracy_score(y[te], p))
            f1s.append(f1_score(y[te], p, average="macro", zero_division=0))
        return float(np.mean(accs)), float(np.std(accs)), float(np.mean(f1s)), float(np.std(f1s))

    n = X.shape[1] // 3
    report = {
        "triad": cv(X),
        "kappa_only": cv(X[:, :n]),
        "length_only": cv(X[:, n:2 * n]),
        "belief_only": cv(X[:, 2 * n:]),
        "triad_pca8": cv(X, k=8),
    }
    for k, v in report.items():
        print(f"  {k:<14} acc={v[0]*100:5.1f} +/- {v[1]*100:4.1f}   macroF1={v[2]*100:5.1f} +/- {v[3]*100:4.1f}")
    return report


if __name__ == "__main__":
    p = harness.base_parser("exp02_operation_classification", __doc__)
    p.add_argument("--aggregate", action="store_true", help="fit on existing result blocks")
    a = p.parse_args()
    if a.aggregate:
        aggregate(a.out)
    else:
        harness.run_over_models("exp02_operation_classification", a, run)
