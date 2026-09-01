"""Exp 03 -- behaviourally silent, geometrically loud.

The paper's central claim is that operations which barely move behaviour still deform
internal computation. That claim is unreadable unless behaviour is measured and shown to
be nearly unchanged, so this experiment produces the scatter that carries it:

    x = behavioural change   (delta perplexity; plus ROUGE-L / BLEU / F1 / accuracy
                              for the tasks whose output format requires them)
    y = geometric change     (DTW between base and modified triad profiles)

BLEU and ROUGE appear here and nowhere else. They are the x-axis, never evidence of
model quality in their own right.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C                                            # noqa: E402
from core import harness, models, probes, triad, dtw, behavior  # noqa: E402

GEN_TASKS = ["cnn_dailymail", "wmt16", "squad_v2"]


def _behaviour(m, tok, args, seed):
    """perplexity on the probe + one surface metric per generative task."""
    out = {}
    prompts, _ = probes.load_probe(args.probe, n=args.n_prompts, seed=seed, cache_dir=C.CACHE)
    out["ppl"] = behavior.perplexity(m, tok, prompts, device=args.device,
                                     batch_size=args.batch_size)
    for t in GEN_TASKS:
        try:
            P, R = probes.load_probe(t, n=args.n_behaviour, seed=seed, cache_dir=C.CACHE)
            preds = behavior.generate(m, tok, P, device=args.device,
                                      batch_size=args.batch_size)
            out[t] = behavior.score(t, preds, R)
        except Exception as e:
            print(f"     behaviour {t} skipped: {e}")
            out[t] = float("nan")
    return out


def run(model_key, model_id, seed, args):
    prompts, _ = probes.load_probe(args.probe, n=args.n_prompts, seed=seed, cache_dir=C.CACHE)
    tcfg = harness.triad_cfg(args)

    base, tok = models.load(model_id, device=args.device, cache_dir=C.CACHE)
    base_prof = triad.profile_model(base, tok, prompts, tcfg, device=args.device)
    base_beh = _behaviour(base, tok, args, seed)
    models.free(base)

    variants = [("quant_4bit", lambda: models.quantize(model_id, "4bit", args.device, C.CACHE)),
                ("quant_2bit", lambda: models.quantize(model_id, "2bit", args.device, C.CACHE)),
                ("prune_30", lambda: models.prune(model_id, 0.3, args.device, cache_dir=C.CACHE)),
                ("prune_60", lambda: models.prune(model_id, 0.6, args.device, cache_dir=C.CACHE))]
    for kind, sib in C.SIBLINGS.get(model_key, {}).items():
        variants.append((f"sibling_{kind}",
                         lambda s=sib: models.load(s, device=args.device, cache_dir=C.CACHE)))

    labels, geo, dppl, dsurf, profs = [], [], [], [], [base_prof]
    for name, ctor in variants:
        try:
            m, tk = ctor()
        except Exception as e:
            print(f"     variant {name} skipped: {e}"); continue
        pr = triad.profile_model(m, tk, prompts, tcfg, device=args.device)
        beh = _behaviour(m, tk, args, seed)
        models.free(m)
        profs.append(pr)
        labels.append(name)
        dppl.append(beh["ppl"] - base_beh["ppl"])
        dsurf.append([beh[t] - base_beh[t] for t in GEN_TASKS])

    # geometry: one joint normalisation over base + all variants (see App. A.8)
    X = dtw.stack_triad(profs, n=32)
    Cm, Em = dtw.pairwise_matrix(X, band=0.2, normalise="minmax")
    geo = Cm[0, 1:]

    arrays = {
        "labels": np.array(labels),
        "dtw_from_base": np.asarray(geo, dtype=float),
        "warp_from_base": np.asarray(Em[0, 1:], dtype=float),
        "delta_ppl": np.asarray(dppl, dtype=float),
        "delta_surface": np.asarray(dsurf, dtype=float),
        "surface_tasks": np.array(GEN_TASKS),
        "base_ppl": base_beh["ppl"],
        "base_surface": np.array([base_beh[t] for t in GEN_TASKS], dtype=float),
        "dtw_matrix": Cm,
    }
    return arrays, "x=behaviour, y=geometry; joint min-max normalisation over the set"


if __name__ == "__main__":
    p = harness.base_parser("exp03_behavior_vs_geometry", __doc__)
    p.add_argument("--n-behaviour", type=int, default=64,
                   help="prompts per generative task (generation is the slow part)")
    a = p.parse_args()
    harness.run_over_models("exp03_behavior_vs_geometry", a, run)
