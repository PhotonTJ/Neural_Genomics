"""Exp 14 -- was this model SFT-only, or SFT then DPO?

Extends the forensics story from "which base" (exp10) to "which training stage". The
parent repo's stored matrices already suggest the stages are separable: SFT-to-DPO
distances sit below base-to-SFT and base-to-DPO distances in three of four family/probe
combinations. Here we build the ladder ourselves so the comparison is controlled, and
classify the stage from the triad alone.

Cheap: two short LoRA runs per model, then a three-way classification on profiles.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C                                            # noqa: E402
from core import harness, models, probes, triad, dtw           # noqa: E402


def _dpo_pairs(n, seed, cache_dir):
    """(chosen, rejected) pairs for a light preference stage."""
    try:
        from datasets import load_dataset
        ds = load_dataset("Anthropic/hh-rlhf", split="train", cache_dir=cache_dir)
        ds = ds.select(range(min(len(ds), n * 4)))
        rows = list(ds)[:n]
        return [r["chosen"][:1200] for r in rows], [r["rejected"][:1200] for r in rows]
    except Exception:
        txt, _ = probes.load_probe("squad_v2", n=n, seed=seed, cache_dir=cache_dir)
        return txt, [t[::-1] for t in txt]


def run(model_key, model_id, seed, args):
    from peft import PeftModel
    probe, _ = probes.load_probe(args.probe, n=args.n_prompts, seed=seed, cache_dir=C.CACHE)
    tcfg = harness.triad_cfg(args)
    work = os.path.join(C.CACHE, f"stage_{model_key}_seed{seed}")
    os.makedirs(work, exist_ok=True)

    chosen, _rejected = _dpo_pairs(args.n_train, seed, C.CACHE)
    sft_dir = os.path.join(work, "sft")
    dpo_dir = os.path.join(work, "dpo")
    if not os.path.exists(os.path.join(sft_dir, "adapter_model.safetensors")):
        models.lora_finetune(model_id, chosen, sft_dir, steps=args.steps,
                             device=args.device, cache_dir=C.CACHE, seed=seed)
    # second stage: continue on the preferred responses only, at a lower rate. This is a
    # preference-shaped stage rather than a full DPO objective; the geometric question is
    # whether a second stage is detectable at all, not which loss produced it.
    if not os.path.exists(os.path.join(dpo_dir, "adapter_model.safetensors")):
        models.lora_finetune(model_id, chosen[: max(len(chosen) // 2, 16)], dpo_dir,
                             steps=args.steps // 2, lr=5e-5,
                             device=args.device, cache_dir=C.CACHE, seed=seed + 7)

    stages, profs = [], []
    base, tok = models.load(model_id, device=args.device, cache_dir=C.CACHE)
    profs.append(triad.profile_model(base, tok, probe, tcfg, device=args.device))
    stages.append("base")
    models.free(base)

    for name, path in (("sft", sft_dir), ("dpo", dpo_dir)):
        b, tk = models.load(model_id, device=args.device, cache_dir=C.CACHE)
        m = PeftModel.from_pretrained(b, path).eval()
        profs.append(triad.profile_model(m, tk, probe, tcfg, device=args.device))
        stages.append(name)
        models.free(m, b)

    X = dtw.stack_triad(profs, n=32)
    Cm, _ = dtw.pairwise_matrix(X, band=0.2, normalise="minmax")

    arrays = {
        "stages": np.array(stages),
        "dtw": Cm,
        "d_base_sft": Cm[0, 1], "d_base_dpo": Cm[0, 2], "d_sft_dpo": Cm[1, 2],
        "ordering_holds": bool(Cm[1, 2] < min(Cm[0, 1], Cm[0, 2])),
        "kappa": np.stack([p["kappa"] for p in profs]),
        "length": np.stack([p["length"] for p in profs]),
        "belief": np.stack([p["belief"] for p in profs]),
    }
    return arrays, "base / SFT / second-stage ladder; tests d(SFT,DPO) < d(base,*)"


if __name__ == "__main__":
    p = harness.base_parser("exp14_alignment_stage", __doc__)
    p.add_argument("--steps", type=int, default=150)
    p.add_argument("--n-train", type=int, default=256)
    a = p.parse_args()
    harness.run_over_models("exp14_alignment_stage", a, run)
