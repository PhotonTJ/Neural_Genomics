"""Exp 04 -- task stability, and the joint normalisation that makes the threshold usable.

Two halves:

  (a) profile one model on all ten FLAN tasks and take all pairwise DTW distances. The
      upper decile of that distribution is the acceptance threshold theta.

  (b) profile the same model's lifecycle variants on the *same* probe and normalise the
      task prototypes and the variants TOGETHER, so that theta and the operation
      distances live on one scale. Without this, theta is a number about task variation
      that cannot legally be compared to anything in the results section -- which is the
      gap App. A.8 currently records as open.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C                                        # noqa: E402
from core import harness, models, probes, triad, dtw      # noqa: E402


def run(model_key, model_id, seed, args):
    tcfg = harness.triad_cfg(args)
    m, tok = models.load(model_id, device=args.device, cache_dir=C.CACHE)

    # ---- (a) one prototype per task
    task_names, task_profs = [], []
    for name, *_ in C.FLAN_TASKS:
        try:
            P, _ = probes.load_probe(name, n=args.n_prompts, seed=seed, cache_dir=C.CACHE)
        except Exception as e:
            print(f"     task {name} skipped: {e}"); continue
        task_profs.append(triad.profile_model(m, tok, P, tcfg, device=args.device))
        task_names.append(name)

    # ---- (b) lifecycle variants on the fixed probe
    fixed, _ = probes.load_probe(args.probe, n=args.n_prompts, seed=seed, cache_dir=C.CACHE)
    var_names, var_profs = ["base"], [triad.profile_model(m, tok, fixed, tcfg, device=args.device)]
    models.free(m)

    for name, ctor in [("quant_4bit", lambda: models.quantize(model_id, "4bit", args.device, C.CACHE)),
                       ("prune_40", lambda: models.prune(model_id, 0.4, args.device, cache_dir=C.CACHE))]:
        try:
            mm, tk = ctor()
        except Exception as e:
            print(f"     variant {name} skipped: {e}"); continue
        var_profs.append(triad.profile_model(mm, tk, fixed, tcfg, device=args.device))
        var_names.append(name)
        models.free(mm)
    for kind, sib in C.SIBLINGS.get(model_key, {}).items():
        try:
            mm, tk = models.load(sib, device=args.device, cache_dir=C.CACHE)
        except Exception as e:
            print(f"     sibling {kind} skipped: {e}"); continue
        var_profs.append(triad.profile_model(mm, tk, fixed, tcfg, device=args.device))
        var_names.append(f"sibling_{kind}")
        models.free(mm)

    # ---- joint normalisation over the union, then split the blocks back out
    n_t = len(task_profs)
    X = dtw.stack_triad(task_profs + var_profs, n=args.resample_to)
    Cm, Em = dtw.pairwise_matrix(X, band=args.band, normalise="minmax")

    task_block = Cm[:n_t, :n_t]
    task_pairs = dtw.offdiag(task_block)
    theta = float(np.quantile(task_pairs, args.threshold_q)) if task_pairs.size else float("nan")
    op_dist = Cm[n_t, n_t + 1:]                    # base -> each variant

    arrays = {
        "task_names": np.array(task_names),
        "task_dtw": task_block,
        "task_pairs": task_pairs,
        "task_median": float(np.median(task_pairs)) if task_pairs.size else np.nan,
        "task_mean": float(np.mean(task_pairs)) if task_pairs.size else np.nan,
        "theta": theta,
        "threshold_q": args.threshold_q,
        "variant_names": np.array(var_names[1:]),
        "variant_dtw_from_base": np.asarray(op_dist, dtype=float),
        "variant_exceeds_theta": np.asarray(op_dist > theta),
        "warp_extent": Em,
        "joint_matrix": Cm,
    }
    return arrays, "task prototypes and lifecycle variants normalised jointly"


if __name__ == "__main__":
    p = harness.base_parser("exp04_task_stability", __doc__)
    p.add_argument("--threshold-q", type=float, default=0.90)
    p.add_argument("--band", type=float, default=0.2)
    p.add_argument("--resample-to", type=int, default=32)
    a = p.parse_args()
    harness.run_over_models("exp04_task_stability", a, run)
