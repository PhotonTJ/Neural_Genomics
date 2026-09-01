"""The nDNA triad: spectral curvature, thermodynamic length, belief vector field.

All three are defined on the layerwise next-token distributions read out by a logit
lens, embedded on the Fisher-Rao sphere by u = sqrt(q). Nothing here touches hidden-state
coordinates, so the outputs are invariant to reparameterisations that leave the model's
function unchanged (see experiments/exp05_invariance.py, which tests exactly that).

Memory note: materialising q for every layer, position and vocabulary entry is
O(L*N*V) and will not fit for a 7B model at full sequence length. We therefore gather
`keep_last_k` supervised positions per sequence *before* applying the lens, which is the
optimisation described in the paper's cost analysis.
"""
from __future__ import annotations

import numpy as np

from ._optional import load_torch

torch, TORCH_AVAILABLE = load_torch()


# --------------------------------------------------------------------------- helpers
def _sphere(q, floor):
    """Square-root embedding onto the positive orthant of the unit sphere."""
    q = torch.clamp(q, min=floor)
    q = q / q.sum(-1, keepdim=True)
    u = torch.sqrt(q)
    return u / u.norm(dim=-1, keepdim=True)


def _tangent(u, v):
    """Project v into the tangent space at u."""
    return v - (u * v).sum(-1, keepdim=True) * u


def fisher_rao(q_a, q_b, floor=1e-12):
    """d_FR(a,b) = 2 arccos( sum_i sqrt(a_i b_i) ).  Shape (...,) from (..., V)."""
    s = torch.sqrt(torch.clamp(q_a, min=floor) * torch.clamp(q_b, min=floor)).sum(-1)
    return 2.0 * torch.arccos(torch.clamp(s, -1.0, 1.0))


# --------------------------------------------------------------------------- readout
@torch.no_grad()
def layer_distributions(model, input_ids, attention_mask, positions, tau=1.0,
                        floor=1e-12, lens=None):
    """Next-token distribution at every depth node, on the selected positions.

    positions: LongTensor (M, 2) of (batch_index, token_index) pairs.
    Returns q of shape (n_nodes, M, V) in float32.
    """
    out = model(input_ids=input_ids, attention_mask=attention_mask,
                output_hidden_states=True, use_cache=False)
    hs = out.hidden_states                      # tuple of (B, T, H), length L+1
    b, t = positions[:, 0], positions[:, 1]
    head = lens if lens is not None else _default_lens(model)

    qs = []
    for h in hs:
        sel = h[b, t].to(torch.float32)         # (M, H) -- gather BEFORE the lens
        logits = head(sel)                      # (M, V)
        q = torch.softmax(logits / tau, dim=-1)
        qs.append(torch.clamp(q, min=floor))
    return torch.stack(qs, 0)


def _default_lens(model):
    """final norm + unembedding, i.e. the plain logit lens."""
    norm = None
    for attr in ("model.norm", "model.final_layernorm", "transformer.ln_f", "gpt_neox.final_layer_norm"):
        obj = model
        try:
            for part in attr.split("."):
                obj = getattr(obj, part)
            norm = obj
            break
        except AttributeError:
            continue
    head = model.get_output_embeddings()

    def _lens(h):
        return head(norm(h)) if norm is not None else head(h)

    return _lens


# --------------------------------------------------------------------------- triad
def thermodynamic_length(q):
    """L_ell for ell = 0..n_nodes-2.  q: (n_nodes, M, V) -> (n_nodes-1,) and per-sample."""
    steps = fisher_rao(q[:-1], q[1:])           # (n_nodes-1, M)
    return steps.mean(-1).cpu().numpy(), steps.cpu().numpy()


def spectral_curvature(q, mode="turn", eps=1e-8, degen=1e-4, floor=1e-12):
    """kappa_ell for interior nodes.  Returns (profile, per_sample, n_valid)."""
    u = _sphere(q, floor)                        # (n, M, V)
    n = u.shape[0]
    if n < 3:
        raise ValueError("need at least three depth nodes for curvature")

    if mode == "chord":
        du = _tangent(u[1:-1], u[2:] - u[1:-1])
        d2u = _tangent(u[1:-1], u[2:] - 2 * u[1:-1] + u[:-2])
        k = d2u.norm(dim=-1) / (du.norm(dim=-1) ** 2 + eps) ** 1.5
        valid = torch.ones_like(k, dtype=torch.bool)
    elif mode == "turn":
        cos = lambda a, b: torch.clamp((a * b).sum(-1), -1.0, 1.0)
        a = torch.arccos(cos(u[:-2], u[1:-1]))
        b = torch.arccos(cos(u[1:-1], u[2:]))
        c = torch.arccos(cos(u[:-2], u[2:]))
        sa, sb = torch.sin(a), torch.sin(b)
        valid = (sa * sb) > degen                # exclude degenerate triangles
        denom = torch.where(valid, sa * sb, torch.ones_like(sa))
        cth = torch.clamp((torch.cos(c) - torch.cos(a) * torch.cos(b)) / denom, -1.0, 1.0)
        theta = torch.arccos(cth)
        k = theta / (a + b + eps)
    else:
        raise ValueError(f"unknown curvature mode {mode!r}")

    k = torch.where(valid, k, torch.zeros_like(k))
    cnt = valid.sum(-1).clamp(min=1)
    prof = (k.sum(-1) / cnt).cpu().numpy()
    return prof, k.cpu().numpy(), valid.sum(-1).cpu().numpy()


def belief_field(q, targets, tau=1.0, floor=1e-12, anchored=True):
    """v_ell(c): mean tangent belief push.  Returns (norms, field).

    q: (n_nodes, M, V);  targets: (M,) ground-truth next-token ids.
    """
    u = _sphere(q, floor)
    onehot = torch.zeros_like(q[0])
    onehot.scatter_(1, targets.view(-1, 1), 1.0)
    g = (onehot.unsqueeze(0) - q) / tau                      # (n, M, V)
    push = _tangent(u, u * g) / (2.0 * tau)                  # (n, M, V)

    v = push.mean(1)                                         # (n, V)
    if anchored:
        ubar = u.mean(1)
        ubar = ubar / ubar.norm(dim=-1, keepdim=True)
        v = _tangent(ubar, v)
    return v.norm(dim=-1).cpu().numpy(), v.cpu().numpy()


# --------------------------------------------------------------------------- driver
@torch.no_grad()
def profile_model(model, tokenizer, texts, cfg, device="cuda", lens=None):
    """Full triad over a corpus. -> dict of numpy arrays, one row per depth node."""
    model.eval()
    acc = {"kappa": [], "length": [], "belief": [], "n_valid": []}
    bs, k = cfg.batch_size, cfg.keep_last_k

    for i in range(0, len(texts), bs):
        chunk = texts[i:i + bs]
        enc = tokenizer(chunk, return_tensors="pt", padding=True, truncation=True,
                        max_length=cfg.max_length if hasattr(cfg, "max_length") else 384)
        enc = {kk: v.to(device) for kk, v in enc.items()}
        ids, am = enc["input_ids"], enc["attention_mask"]

        pos, tgt = [], []
        for bi in range(ids.shape[0]):
            valid = am[bi].nonzero().flatten()
            if valid.numel() < 2:
                continue
            # supervised positions: predict token t+1 from state at t
            cand = valid[:-1][-k:]
            for t in cand.tolist():
                pos.append((bi, t))
                tgt.append(ids[bi, t + 1].item())
        if not pos:
            continue
        pos_t = torch.tensor(pos, device=device)
        tgt_t = torch.tensor(tgt, device=device)

        q = layer_distributions(model, ids, am, pos_t, tau=cfg.tau,
                                floor=cfg.prob_floor, lens=lens)
        if not cfg.include_embedding:
            q = q[1:]

        Lp, _ = thermodynamic_length(q)
        Kp, _, nv = spectral_curvature(q, mode=cfg.curvature, eps=cfg.eps_curv,
                                       degen=cfg.degen_delta, floor=cfg.prob_floor)
        Bp, _ = belief_field(q, tgt_t, tau=cfg.tau, floor=cfg.prob_floor)

        acc["length"].append(Lp)
        acc["kappa"].append(Kp)
        acc["belief"].append(Bp)
        acc["n_valid"].append(nv)
        del q

    if not acc["length"]:
        raise RuntimeError("no supervised positions were produced")

    out = {kk: np.mean(np.stack(v, 0), 0) for kk, v in acc.items() if kk != "n_valid"}
    out["n_valid"] = np.sum(np.stack(acc["n_valid"], 0), 0)
    out["n_batches"] = np.array(len(acc["length"]))
    # cumulative FR path length, and the commitment-depth curve of App. A.6
    out["length_total"] = np.array(out["length"].sum())
    out["tail_length"] = np.cumsum(out["length"][::-1])[::-1].copy()
    return out


def commitment_depth(tail_length, eps):
    """Smallest ell with sum_{k>ell} L_k <= eps  (App. A.6)."""
    idx = np.nonzero(tail_length <= eps)[0]
    return int(idx[0]) if idx.size else int(len(tail_length))
