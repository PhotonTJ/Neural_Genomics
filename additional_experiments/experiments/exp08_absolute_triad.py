"""Exp 08 -- absolute triad profiles, saved raw, plus commitment depth.

The parent repo ships only plotted (min-max normalised) profiles, so absolute kappa,
L and ||v|| are not recoverable from it and every table quoting them inherited a
normalisation ambiguity. This experiment writes the raw per-layer arrays to NPZ under a
single stated convention, which is what makes the absolute table reproducible.

It also emits the commitment depth curve: l*(eps) = min{ l : sum_{k>l} L_k <= eps },
which follows from the triangle inequality on Fisher-Rao and costs nothing extra.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C                                     # noqa: E402
from core import harness, models, probes, triad        # noqa: E402


def run(model_key, model_id, seed, args):
    prompts, _ = probes.load_probe(args.probe, n=args.n_prompts, seed=seed, cache_dir=C.CACHE)
    tcfg = harness.triad_cfg(args)

    m, tok = models.load(model_id, device=args.device, cache_dir=C.CACHE)
    prof = triad.profile_model(m, tok, prompts, tcfg, device=args.device)

    # both curvature estimators on the same forward pass, for the App. A.4 comparison
    other = "chord" if args.curvature == "turn" else "turn"
    tcfg2 = harness.triad_cfg(args)
    tcfg2.curvature = other
    prof2 = triad.profile_model(m, tok, prompts, tcfg2, device=args.device)
    models.free(m)

    total = float(prof["length_total"])
    eps_grid = np.array([0.01, 0.02, 0.05, 0.10, 0.20]) * total
    depths = np.array([triad.commitment_depth(prof["tail_length"], e) for e in eps_grid])

    arrays = {
        "layer": np.arange(len(prof["kappa"])),
        "kappa": prof["kappa"], "length": prof["length"], "belief": prof["belief"],
        f"kappa_{other}": prof2["kappa"],
        "kappa_estimator_pearson": float(np.corrcoef(prof["kappa"], prof2["kappa"])[0, 1]),
        "tail_length": prof["tail_length"],
        "length_total": total,
        "n_valid_curvature": prof["n_valid"],
        "eps_grid": eps_grid,
        "eps_fraction": np.array([0.01, 0.02, 0.05, 0.10, 0.20]),
        "commitment_depth": depths,
        "commitment_depth_frac": depths / max(len(prof["kappa"]), 1),
        "probe_hash": probes.probe_hash(prompts),
        "curvature_primary": args.curvature,
    }
    return arrays, f"absolute triad, tau={tcfg.tau}, keep_last_k={tcfg.keep_last_k}"


if __name__ == "__main__":
    p = harness.base_parser("exp08_absolute_triad", __doc__)
    a = p.parse_args()
    harness.run_over_models("exp08_absolute_triad", a, run)
