"""Exp 07 -- does the triad predict what an intervention would do?

The profile is observational. This measures how well it tracks activation patching:
for each layer, patch the hidden state from a corrupted run into the clean run and record
the change in the model's final log-probability of the correct token. Then correlate that
measured effect against L_ell, ||v_ell||, and the commitment-bound exclusion prediction.

A positive result does not make the triad causal. It makes it a cheap *predictor* of
intervention effects, which is the claim the paper would then make -- and only if the
correlation supports it.
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C                                     # noqa: E402
from core import harness, models, probes, triad        # noqa: E402


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


@torch.no_grad()
def patch_effects(model, tok, prompts, device, n_layers, bs=2, max_len=192, seed=0):
    """Per-layer mean |change in log p(correct token)| when patching a corrupted state."""
    rng = np.random.default_rng(seed)
    eff = np.zeros(n_layers)
    cnt = np.zeros(n_layers)
    layers = model.model.layers if hasattr(model, "model") else model.transformer.h

    for i in range(0, len(prompts), bs):
        clean = prompts[i:i + bs]
        corrupt = [prompts[int(rng.integers(0, len(prompts)))] for _ in clean]
        ec = tok(clean, return_tensors="pt", padding=True, truncation=True, max_length=max_len)
        ex = tok(corrupt, return_tensors="pt", padding=True, truncation=True,
                 max_length=ec["input_ids"].shape[1])
        ec = {k: v.to(device) for k, v in ec.items()}
        ex = {k: v.to(device) for k, v in ex.items()}
        if ex["input_ids"].shape[1] != ec["input_ids"].shape[1]:
            continue

        ids, am = ec["input_ids"], ec["attention_mask"]
        pos = am.sum(1) - 2
        if (pos < 0).any():
            continue
        rows = torch.arange(ids.shape[0], device=ids.device)
        tgt = ids[rows, pos + 1]

        def logp(out):
            lp = torch.log_softmax(out.logits[rows, pos].float(), -1)
            return lp.gather(-1, tgt.view(-1, 1)).squeeze(-1)

        base_lp = logp(model(**ec, use_cache=False))
        donor = model(**ex, output_hidden_states=True, use_cache=False).hidden_states

        for l in range(min(n_layers, len(layers))):
            store = {}

            def hook(_m, _in, out, l=l):
                h = out[0] if isinstance(out, tuple) else out
                h = h.clone()
                h[rows, pos] = donor[l + 1][rows, pos].to(h.dtype)
                store["h"] = h
                return (h,) + out[1:] if isinstance(out, tuple) else h

            hd = layers[l].register_forward_hook(hook)
            try:
                lp = logp(model(**ec, use_cache=False))
                eff[l] += float(torch.abs(lp - base_lp).mean())
                cnt[l] += 1
            finally:
                hd.remove()
    return eff / np.maximum(cnt, 1)


def run(model_key, model_id, seed, args):
    prompts, _ = probes.load_probe(args.probe, n=min(args.n_prompts, args.n_patch),
                                   seed=seed, cache_dir=C.CACHE)
    tcfg = harness.triad_cfg(args)
    m, tok = models.load(model_id, device=args.device, cache_dir=C.CACHE,
                         dtype=getattr(args, "dtype", "bfloat16"))
    prof = triad.profile_model(m, tok, prompts, tcfg, device=args.device)
    n_layers = m.config.num_hidden_layers
    eff = patch_effects(m, tok, prompts, args.device, n_layers,
                        bs=max(1, args.batch_size // 2), seed=seed)
    models.free(m)

    L = prof["length"][:n_layers]
    B = prof["belief"][:n_layers]
    tail = prof["tail_length"][:n_layers]
    eps = float(np.quantile(tail, 0.10))
    excluded = tail <= eps                     # layers the bound says cannot matter
    thr = float(np.quantile(eff, 0.25))

    arrays = {
        "layer": np.arange(n_layers),
        "patch_effect": eff,
        "length": L, "belief": B, "tail_length": tail,
        "rho_length_vs_patch": _spearman(L, eff),
        "rho_belief_vs_patch": _spearman(B, eff),
        "rho_tail_vs_patch": _spearman(tail, eff),
        "commitment_eps": eps,
        "excluded_mask": excluded,
        "exclusion_precision": float(np.mean(eff[excluded] <= thr)) if excluded.any() else np.nan,
        "commitment_depth": triad.commitment_depth(prof["tail_length"], eps),
    }
    return arrays, "activation patching vs triad; exclusion precision from the commitment bound"


if __name__ == "__main__":
    p = harness.base_parser("exp07_patching", __doc__)
    p.add_argument("--n-patch", type=int, default=48,
                   help="prompts used for patching (L forward passes each)")
    a = p.parse_args()
    harness.run_over_models("exp07_patching", a, run)
