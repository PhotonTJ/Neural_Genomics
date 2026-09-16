"""Exp 01 -- dose-response under compression.

Sweeps quantization bit-width and pruning sparsity, measuring the triad and perplexity at
each dose. The claim under test is that the geometry responds *monotonically* to
intervention strength -- a stronger statement than "the two endpoints differ", and the
cheapest strong evidence available because it needs no training.

Fills the dose-response table. Report the quantizer/pruner by name in the caption; this
script uses round-to-nearest and magnitude pruning (see core/models.py for why).
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C                                     # noqa: E402
from core import harness, models, probes, triad, behavior  # noqa: E402


def run(model_key, model_id, seed, args):
    prompts, _ = probes.load_probe(args.probe, n=args.n_prompts, seed=seed,
                                   cache_dir=C.CACHE)
    tcfg = harness.triad_cfg(args)
    rows, labels = [], []

    def measure(m, tok, label):
        prof = triad.profile_model(m, tok, prompts, tcfg, device=args.device)
        ppl = behavior.perplexity(m, tok, prompts, device=args.device,
                                  batch_size=args.batch_size)
        rows.append([prof["length_total"], float(np.mean(prof["kappa"])),
                     float(np.mean(prof["belief"])), ppl])
        labels.append(label)
        return prof

    base_prof = None
    for dose in C.QUANT_DOSES:
        m, tok = models.quantize(model_id, dose, device=args.device, cache_dir=C.CACHE)
        p = measure(m, tok, f"quant:{dose}")
        if dose == "fp16":
            base_prof = p
        models.free(m)

    for sp in C.PRUNE_DOSES:
        m, tok = models.prune(model_id, sp, device=args.device, structured=True,
                              cache_dir=C.CACHE)
        measure(m, tok, f"prune:{sp:.2f}")
        models.free(m)

    R = np.array(rows, dtype=float)                     # (n_doses, 4)
    qn = len(C.QUANT_DOSES)
    L0 = R[0, 0]

    def spearman(y):
        x = np.arange(len(y), dtype=float)
        rx = np.argsort(np.argsort(x)).astype(float)
        ry = np.argsort(np.argsort(y)).astype(float)
        rx -= rx.mean(); ry -= ry.mean()
        d = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
        return float((rx * ry).sum() / d) if d else float("nan")

    arrays = {
        "labels": np.array(labels),
        "length_total": R[:, 0],
        "length_ratio": R[:, 0] / L0,
        "kappa_mean": R[:, 1],
        "belief_mean": R[:, 2],
        "perplexity": R[:, 3],
        "rho_quant_length": spearman(R[:qn, 0]),
        "rho_prune_length": spearman(R[qn:, 0]),
        "rho_quant_kappa": spearman(R[:qn, 1]),
        "rho_prune_kappa": spearman(R[qn:, 1]),
        "base_kappa_profile": base_prof["kappa"],
        "base_length_profile": base_prof["length"],
        "base_belief_profile": base_prof["belief"],
    }
    return arrays, "RTN weight-only quantization; structured magnitude pruning"


if __name__ == "__main__":
    p = harness.base_parser("exp01_dose_response", __doc__)
    a = p.parse_args()
    harness.run_over_models("exp01_dose_response", a, run)
