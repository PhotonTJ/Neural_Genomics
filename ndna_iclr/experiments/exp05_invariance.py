"""Exp 05 -- the reparameterisation check that justifies working on predictions.

The paper argues that measuring hidden states directly produces numbers that move when
the model does not. That is testable exactly: rescale the normalisation gains by c, check
the logits are unchanged, and recompute the triad three ways.

  predictions (ours)          expected: invariant to numerical precision
  hidden states, whitened     expected: invariant up to estimation error
  hidden states, raw Euclid   expected: scales with c

The script does NOT assume the edit is function-preserving. It measures the logit
deviation and reports it, so an architecture where this edit genuinely changes the
function is detected rather than silently mis-analysed.
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C                                     # noqa: E402
from core import harness, models, probes, triad        # noqa: E402


@torch.no_grad()
def _hidden_profiles(model, tok, texts, device, k=8, ridge=1e-4, bs=4):
    """Raw-Euclidean and Procrustes-whitened curvature/length on hidden states."""
    raw_k, raw_l, wh_k, wh_l = [], [], [], []
    for i in range(0, len(texts), bs):
        enc = tok(texts[i:i + bs], return_tensors="pt", padding=True,
                  truncation=True, max_length=256)
        enc = {kk: v.to(device) for kk, v in enc.items()}
        hs = model(**enc, output_hidden_states=True, use_cache=False).hidden_states
        am = enc["attention_mask"]
        sel = [(b, t) for b in range(am.shape[0])
               for t in am[b].nonzero().flatten()[-k:].tolist()]
        if len(sel) < 2:
            continue
        bi = torch.tensor([s[0] for s in sel], device=device)
        ti = torch.tensor([s[1] for s in sel], device=device)
        H = torch.stack([h[bi, ti].to(torch.float32) for h in hs])   # (n, M, d)

        d1 = H[1:] - H[:-1]
        d2 = H[2:] - 2 * H[1:-1] + H[:-2]
        raw_l.append(d1.norm(dim=-1).mean(-1).cpu().numpy())
        raw_k.append((d2.norm(dim=-1) / (d1[:-1].norm(dim=-1) ** 2 + 1e-8)).mean(-1).cpu().numpy())

        # whiten each layer by its own covariance, then Procrustes-align neighbours
        Hw = []
        for l in range(H.shape[0]):
            X = H[l] - H[l].mean(0, keepdim=True)
            cov = (X.T @ X) / max(X.shape[0] - 1, 1) + ridge * torch.eye(X.shape[1], device=device)
            ev, V = torch.linalg.eigh(cov)
            Hw.append(X @ (V @ torch.diag(ev.clamp(min=1e-8).rsqrt()) @ V.T))
        Hw = torch.stack(Hw)
        for l in range(1, Hw.shape[0]):
            U, _, Vt = torch.linalg.svd(Hw[l - 1].T @ Hw[l], full_matrices=False)
            Hw[l] = Hw[l] @ (U @ Vt).T
        w1 = Hw[1:] - Hw[:-1]
        w2 = Hw[2:] - 2 * Hw[1:-1] + Hw[:-2]
        wh_l.append(w1.norm(dim=-1).mean(-1).cpu().numpy())
        wh_k.append((w2.norm(dim=-1) / (w1[:-1].norm(dim=-1) ** 2 + 1e-8)).mean(-1).cpu().numpy())

    f = lambda a: np.mean(np.stack(a), 0) if a else np.array([np.nan])
    return f(raw_k), f(raw_l), f(wh_k), f(wh_l)


def run(model_key, model_id, seed, args):
    prompts, _ = probes.load_probe(args.probe, n=min(args.n_prompts, 64), seed=seed,
                                   cache_dir=C.CACHE)
    tcfg = harness.triad_cfg(args)
    rows = []

    for c in C.GAIN_SCALES:
        m, tok = models.load(model_id, device=args.device, cache_dir=C.CACHE)
        ref = models.logits_fingerprint(m, tok, prompts, device=args.device)
        n_touched = models.rescale_residual_gains(m, c) if c != 1.0 else 0
        new = models.logits_fingerprint(m, tok, prompts, device=args.device)
        dev = float(np.max(np.abs(new - ref)))

        prof = triad.profile_model(m, tok, prompts, tcfg, device=args.device)
        rk, rl, wk, wl = _hidden_profiles(m, tok, prompts, args.device,
                                          k=args.keep_last_k, bs=args.batch_size)
        models.free(m)
        rows.append([c, dev, n_touched,
                     float(np.mean(prof["kappa"])), float(prof["length_total"]),
                     float(np.nanmean(rk)), float(np.nanmean(rl)),
                     float(np.nanmean(wk)), float(np.nanmean(wl))])
        print(f"     c={c}: max|dlogit|={dev:.3e}  triad_kappa={rows[-1][3]:.5f}  "
              f"raw_hidden_len={rows[-1][6]:.4f}")

    R = np.array(rows, dtype=float)
    base = R[R[:, 0] == 1.0][0] if (R[:, 0] == 1.0).any() else R[0]
    arrays = {
        "gain_scale": R[:, 0], "max_logit_deviation": R[:, 1], "n_norms_touched": R[:, 2],
        "triad_kappa": R[:, 3], "triad_length": R[:, 4],
        "hidden_raw_kappa": R[:, 5], "hidden_raw_length": R[:, 6],
        "hidden_whitened_kappa": R[:, 7], "hidden_whitened_length": R[:, 8],
        "triad_length_ratio": R[:, 4] / base[4],
        "hidden_raw_length_ratio": R[:, 6] / base[6],
    }
    return arrays, "gain trade: triad should be flat in c, raw hidden length should scale"


if __name__ == "__main__":
    p = harness.base_parser("exp05_invariance", __doc__)
    a = p.parse_args()
    harness.run_over_models("exp05_invariance", a, run)
