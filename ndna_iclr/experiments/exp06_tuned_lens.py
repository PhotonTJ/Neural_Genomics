"""Exp 06 -- does the result survive a corrected readout?

The plain logit lens is unreliable in early layers of some models, and our low-to-mid
trough sits exactly there. This is the most likely technical rejection vector, so we
recompute the triad with a trained per-layer affine translator (a tuned lens) and report
both readouts side by side. The claim to support is that conclusions are unchanged; if
any conclusion flips, that is reported rather than buried.
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C                                     # noqa: E402
from core import harness, models, probes, triad, dtw   # noqa: E402


def train_translators(model, tok, texts, device, steps=150, lr=1e-3, bs=2, seed=0):
    """One affine map per layer, fit to match the final-layer distribution."""
    torch.manual_seed(seed)
    hid = model.config.hidden_size
    n_layers = model.config.num_hidden_layers + 1
    T = [torch.nn.Linear(hid, hid, bias=True).to(device).to(torch.float32)
         for _ in range(n_layers)]
    for t in T:
        torch.nn.init.eye_(t.weight)
        torch.nn.init.zeros_(t.bias)
    opt = torch.optim.AdamW([p for t in T for p in t.parameters()], lr=lr)
    lens = triad._default_lens(model)

    step = 0
    while step < steps:
        for i in range(0, len(texts), bs):
            if step >= steps:
                break
            enc = tok(texts[i:i + bs], return_tensors="pt", padding=True,
                      truncation=True, max_length=192)
            enc = {k: v.to(device) for k, v in enc.items()}
            with torch.no_grad():
                hs = model(**enc, output_hidden_states=True, use_cache=False).hidden_states
                final = torch.log_softmax(lens(hs[-1].to(torch.float32)), -1)
            loss = 0.0
            for l in range(n_layers - 1):
                pred = torch.log_softmax(lens(T[l](hs[l].to(torch.float32))), -1)
                loss = loss + torch.nn.functional.kl_div(
                    pred, final, log_target=True, reduction="batchmean")
            loss.backward()
            opt.step()
            opt.zero_grad()
            step += 1
    for t in T:
        t.eval()
    return T


def run(model_key, model_id, seed, args):
    prompts, _ = probes.load_probe(args.probe, n=args.n_prompts, seed=seed, cache_dir=C.CACHE)
    tcfg = harness.triad_cfg(args)
    m, tok = models.load(model_id, device=args.device, cache_dir=C.CACHE)

    plain = triad.profile_model(m, tok, prompts, tcfg, device=args.device)

    T = train_translators(m, tok, prompts[:args.n_translator], args.device,
                          steps=args.translator_steps, seed=seed)
    base_lens = triad._default_lens(m)
    depth = {"i": 0}

    def tuned_lens(h):
        # profile_model walks hidden_states in order, so a counter recovers the depth
        idx = min(depth["i"], len(T) - 1)
        depth["i"] = (depth["i"] + 1) % len(T)
        return base_lens(T[idx](h))

    tuned = triad.profile_model(m, tok, prompts, tcfg, device=args.device, lens=tuned_lens)
    models.free(m)

    X = dtw.stack_triad([plain, tuned], n=32)
    Cm, Em = dtw.pairwise_matrix(X, band=0.2, normalise="minmax")
    corr = {ch: float(np.corrcoef(plain[ch], tuned[ch])[0, 1])
            for ch in ("kappa", "length", "belief")}

    arrays = {
        "plain_kappa": plain["kappa"], "plain_length": plain["length"],
        "plain_belief": plain["belief"],
        "tuned_kappa": tuned["kappa"], "tuned_length": tuned["length"],
        "tuned_belief": tuned["belief"],
        "dtw_plain_vs_tuned": Cm[0, 1], "warp_plain_vs_tuned": Em[0, 1],
        "pearson_kappa": corr["kappa"], "pearson_length": corr["length"],
        "pearson_belief": corr["belief"],
        "translator_steps": args.translator_steps,
    }
    return arrays, "plain logit lens vs trained per-layer affine translator"


if __name__ == "__main__":
    p = harness.base_parser("exp06_tuned_lens", __doc__)
    p.add_argument("--translator-steps", type=int, default=150)
    p.add_argument("--n-translator", type=int, default=64)
    a = p.parse_args()
    harness.run_over_models("exp06_tuned_lens", a, run)
