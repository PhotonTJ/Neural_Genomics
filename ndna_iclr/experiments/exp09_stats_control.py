"""Exp 09 -- paired significance and the dimensionality control.

Two corrections to the classification table that cost minutes and close real objections.

  * The folds are matched: every signal is evaluated on the same leave-one-family-out
    splits. Comparing marginal standard deviations therefore understates the evidence;
    a paired Wilcoxon signed-rank across folds is both valid and much more powerful.

  * The triad gets 3D features while "perplexity difference only" gets one, so part of
    any gap is dimensionality. We repeat every comparison with each feature set reduced
    to a matched k by PCA.

This is an aggregation step over exp02 outputs -- it loads no model.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C                                 # noqa: E402
from core import harness, io as nio                # noqa: E402


def wilcoxon(a, b):
    """Signed-rank statistic and exact-ish p for small n (paired)."""
    d = np.asarray(a, float) - np.asarray(b, float)
    d = d[d != 0]
    n = len(d)
    if n == 0:
        return 0.0, 1.0, 0
    r = np.argsort(np.argsort(np.abs(d))).astype(float) + 1
    wp = float(r[d > 0].sum())
    wm = float(r[d < 0].sum())
    W = min(wp, wm)
    mu = n * (n + 1) / 4.0
    sd = np.sqrt(n * (n + 1) * (2 * n + 1) / 24.0)
    z = (W - mu) / sd if sd > 0 else 0.0
    p = 2.0 * 0.5 * np.math.erfc(abs(z) / np.sqrt(2)) if sd > 0 else 1.0
    return float(z), float(min(1.0, p)), n


def run_aggregate(args):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import LeaveOneGroupOut
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.decomposition import PCA

    blocks = harness.load_all("exp02_operation_classification", args.out)
    if not blocks:
        print("no exp02 results; run exp02 first"); return
    X = np.concatenate([a["X"] for _, a, _ in blocks])
    y = np.concatenate([a["y"] for _, a, _ in blocks])
    g = np.concatenate([a["families"] for _, a, _ in blocks])
    n = X.shape[1] // 3

    sets = {"triad": X, "kappa_only": X[:, :n],
            "length_only": X[:, n:2 * n], "belief_only": X[:, 2 * n:]}

    def fold_scores(Xs, k=None):
        if k:
            Xs = PCA(n_components=min(k, Xs.shape[1], len(Xs) - 1)).fit_transform(Xs)
        acc, f1 = [], []
        for tr, te in LeaveOneGroupOut().split(Xs, y, groups=g):
            if len(set(y[tr])) < 2:
                continue
            clf = LogisticRegression(max_iter=2000).fit(Xs[tr], y[tr])
            p = clf.predict(Xs[te])
            acc.append(accuracy_score(y[te], p))
            f1.append(f1_score(y[te], p, average="macro", zero_division=0))
        return np.array(acc), np.array(f1)

    raw = {k: fold_scores(v) for k, v in sets.items()}
    pca = {k: fold_scores(v, k=args.pca_k) for k, v in sets.items()}

    out = {}
    print(f"{'signal':<14}{'acc':>18}{'macroF1':>18}{'  vs triad (paired)'}")
    for k in sets:
        a, f = raw[k]
        line = f"{k:<14}{a.mean()*100:9.1f} +/-{a.std()*100:5.1f}{f.mean()*100:11.1f} +/-{f.std()*100:5.1f}"
        if k != "triad":
            z, p, nn = wilcoxon(raw["triad"][0], a)
            line += f"   z={z:+.2f} p={p:.3f} n={nn}"
            out[f"wilcoxon_triad_vs_{k}"] = np.array([z, p, nn])
        print(line)
        out[f"acc_{k}"] = a
        out[f"f1_{k}"] = f
        out[f"acc_pca_{k}"] = pca[k][0]

    print(f"\nmatched-k PCA (k={args.pca_k}):")
    for k in sets:
        print(f"  {k:<14}{pca[k][0].mean()*100:6.1f} +/-{pca[k][0].std()*100:4.1f}")

    out["pca_k"] = np.array(args.pca_k)
    nio.save(os.path.join(args.out, "exp09_stats_control__pooled__seed0"), out,
             cfg={"experiment": "exp09_stats_control", "pca_k": args.pca_k},
             notes="paired Wilcoxon across matched folds + matched-k PCA control")


if __name__ == "__main__":
    p = harness.base_parser("exp09_stats_control", __doc__)
    p.add_argument("--pca-k", type=int, default=8)
    a = p.parse_args()
    run_aggregate(a)
