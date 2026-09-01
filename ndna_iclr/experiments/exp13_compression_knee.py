"""Exp 13 -- picking a compression budget from forward passes alone.

Rides entirely on exp01: the dose sweep is already computed, so the marginal cost here is
seconds. The question is whether the knee in the geometric curve locates the knee in the
behavioural curve. If it does, a practitioner can choose a compression budget without
running the downstream evaluation.

The transfer test is the part that matters: fit the rule on one model, apply it unchanged
to the other, and report how far the predicted budget lands from the true one.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C                        # noqa: E402
from core import harness, io as nio       # noqa: E402


def knee(y, x=None):
    """Index of maximum curvature of a monotone curve (discrete second difference)."""
    y = np.asarray(y, float)
    if len(y) < 3:
        return 0
    y = (y - y.min()) / (y.ptp() + 1e-12)
    d2 = np.abs(np.diff(y, 2))
    return int(np.argmax(d2)) + 1


def analyse(args):
    blocks = harness.load_all("exp01_dose_response", args.out)
    if not blocks:
        print("no exp01 results; run exp01 first"); return

    per_model = {}
    for tag, a, man in blocks:
        mk = (man or {}).get("config", {}).get("model_key", tag)
        labels = [str(s) for s in a["labels"]]
        qi = [i for i, l in enumerate(labels) if l.startswith("quant:")]
        pi = [i for i, l in enumerate(labels) if l.startswith("prune:")]
        rec = {}
        for name, idx in (("quant", qi), ("prune", pi)):
            if len(idx) < 3:
                continue
            L = a["length_ratio"][idx]
            K = a["kappa_mean"][idx]
            P = a["perplexity"][idx]
            rec[name] = {
                "labels": [labels[i] for i in idx],
                "knee_length": knee(L), "knee_kappa": knee(K), "knee_ppl": knee(P),
                "L": L, "K": K, "P": P,
            }
        per_model[mk] = rec

    out, keys = {}, sorted(per_model)
    print(f"{'model':<14}{'sweep':<8}{'knee(L)':>9}{'knee(kappa)':>13}{'knee(ppl)':>11}{'offset':>9}")
    for mk in keys:
        for sweep, r in per_model[mk].items():
            off = r["knee_length"] - r["knee_ppl"]
            print(f"{mk:<14}{sweep:<8}{r['knee_length']:>9}{r['knee_kappa']:>13}"
                  f"{r['knee_ppl']:>11}{off:>+9}")
            out[f"{mk}_{sweep}_knee_length"] = np.array(r["knee_length"])
            out[f"{mk}_{sweep}_knee_kappa"] = np.array(r["knee_kappa"])
            out[f"{mk}_{sweep}_knee_ppl"] = np.array(r["knee_ppl"])
            out[f"{mk}_{sweep}_offset"] = np.array(off)

    # transfer: rule fitted on model A, applied unchanged to model B
    if len(keys) >= 2:
        a_key, b_key = keys[0], keys[1]
        for sweep in ("quant", "prune"):
            if sweep in per_model[a_key] and sweep in per_model[b_key]:
                bias = per_model[a_key][sweep]["knee_length"] - per_model[a_key][sweep]["knee_ppl"]
                pred = per_model[b_key][sweep]["knee_length"] - bias
                true = per_model[b_key][sweep]["knee_ppl"]
                print(f"  transfer {a_key} -> {b_key} [{sweep}]: predicted dose index "
                      f"{pred}, true {true}, error {abs(pred-true)}")
                out[f"transfer_{sweep}_pred"] = np.array(pred)
                out[f"transfer_{sweep}_true"] = np.array(true)
                out[f"transfer_{sweep}_abs_error"] = np.array(abs(pred - true))

    nio.save(os.path.join(args.out, "exp13_compression_knee__pooled__seed0"), out,
             cfg={"experiment": "exp13_compression_knee"},
             notes="geometric knee vs behavioural knee, with cross-model transfer")


if __name__ == "__main__":
    p = harness.base_parser("exp13_compression_knee", __doc__)
    a = p.parse_args()
    analyse(a)
