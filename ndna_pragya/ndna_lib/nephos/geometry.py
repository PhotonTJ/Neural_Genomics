from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from ndna_lib.geometry import (
    AMP_DTYPE,
    AMP_ON,
    EPS_CURV,
    BaseAdapter,
    make_adapter,
    _sqrt_embed,
    _curvature_simp,
)


@dataclass
class ThreeMetrics:
    drift: np.ndarray     # (L,)
    thermo: np.ndarray    # (L-1,)
    spectral: np.ndarray  # (L-2,)


def _infer_device(model: PreTrainedModel) -> torch.device:
    if hasattr(model, "device"):
        return model.device
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _tokenize_prompt_response(
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    response: str,
    max_len: int,
    prompt_max: int,
) -> torch.Tensor:
    prompt = (prompt or "").strip()
    response = (response or "").strip()

    t_prompt = tokenizer(prompt, truncation=True, max_length=prompt_max, add_special_tokens=False)
    remaining = max_len - len(t_prompt["input_ids"]) - 1
    if remaining < 1:
        t_prompt = tokenizer(prompt, truncation=True, max_length=max_len // 2, add_special_tokens=False)
        remaining = max_len - len(t_prompt["input_ids"]) - 1

    t_resp = tokenizer(response, truncation=True, max_length=remaining, add_special_tokens=False)

    eos = tokenizer.eos_token_id
    if eos is None:
        raise ValueError("Tokenizer has no eos_token_id.")
    ids = t_prompt["input_ids"] + t_resp["input_ids"] + [eos]
    return torch.tensor([ids], dtype=torch.long)


def _fr_distance(q1: torch.Tensor, q2: torch.Tensor) -> float:
    bc = torch.sum(torch.sqrt(q1 * q2)).clamp(0.0, 1.0)
    return float(2.0 * torch.acos(bc).item())


@torch.no_grad()
def compute_three_metrics_for_prompt_response(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    response: str,
    model_name: Optional[str] = None,
    tau: float = 1.0,
    max_len: int = 512,
    prompt_max: int = 384,
    keep_last_k: Optional[int] = 32,
) -> ThreeMetrics:
    model.eval()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    device = _infer_device(model)
    adapter: BaseAdapter = make_adapter(model, model_name or getattr(model.config, "_name_or_path", None))

    input_ids = _tokenize_prompt_response(tokenizer, prompt, response, max_len=max_len, prompt_max=prompt_max).to(device)
    S = input_ids.shape[1]
    if S < 2:
        raise ValueError("Sequence too short.")

    prompt_len = tokenizer((prompt or "").strip(), add_special_tokens=False, return_tensors="pt")["input_ids"].shape[1]
    full_T = S - prompt_len
    if full_T <= 0:
        raise ValueError("No response tokens after prompt_len.")

    k = full_T
    if keep_last_k is not None:
        k = max(1, min(int(keep_last_k), full_T))

    # last-k targets and their logits positions
    target_ids = input_ids[0, S - k : S].to(device)      # (k,)
    pos_start = max(0, S - k - 1)
    pos_end_excl = S - 1

    with torch.autocast(device_type="cuda", dtype=AMP_DTYPE, enabled=(device.type == "cuda" and AMP_ON)):
        out = model(input_ids=input_ids, output_hidden_states=True, use_cache=False)
    hidden_states = out.hidden_states[1:]  # len=L
    L = len(hidden_states)

    drift = np.zeros(L, dtype=float)
    q_layers: List[torch.Tensor] = []

    tau = float(max(1e-12, tau))

    for ell in range(L):
        h = hidden_states[ell][:, pos_start:pos_end_excl, :]  # (1,k,H)

        with torch.autocast(device_type="cuda", dtype=AMP_DTYPE, enabled=(device.type == "cuda" and AMP_ON)):
            logits = adapter.lens_logits(h)  # (1,k,V)

        z = logits.float() / tau            # (1,k,V)
        q = F.softmax(z, dim=-1).clamp_min(1e-12)  # (1,k,V)
        u = torch.sqrt(q)                   # (1,k,V)
        q_layers.append(q.detach().cpu())

        # Belief push: match ndna_lib.geometry.belief_field_for_dataset (norm of mean, FR-scaled)
        g = -q / tau
        add = torch.full((1, k, 1), 1.0 / tau, device=g.device, dtype=g.dtype)
        g.scatter_add_(dim=-1, index=target_ids.view(1, k, 1), src=add)

        ug = u * g
        s = (q * g).sum(dim=-1, keepdim=True)
        t = (ug - s * u) / (2.0 * tau)      # (1,k,V)

        t_sum = t.sum(dim=1)                # (1,V)
        u_sum = u.sum(dim=1)                # (1,V)
        cnt = float(k)
        v_avg = t_sum / cnt
        u_avg = u_sum / cnt
        u_norm = torch.linalg.norm(u_avg, dim=-1, keepdim=True).clamp_min(1e-12)
        u_bar = u_avg / u_norm
        radial = (u_bar * v_avg).sum(dim=-1, keepdim=True)
        v_tan = v_avg - radial * u_bar
        drift[ell] = float((2.0 * torch.linalg.norm(v_tan, dim=-1)).item())

        del logits, z, q, u, g, ug, s, t, t_sum, u_sum, u_bar, radial, v_tan

    # Thermodynamic length: mean FR step per token between consecutive layers
    thermo = np.zeros(L - 1, dtype=float)
    for i in range(L - 1):
        q1 = q_layers[i][0]      # (k,V)
        q2 = q_layers[i + 1][0]  # (k,V)
        bc = torch.sum(torch.sqrt(q1 * q2), dim=-1).clamp(0.0, 1.0)
        dist = 2.0 * torch.acos(bc)
        thermo[i] = float(dist.mean().item())

    # Spectral curvature: per-token curvature across layers, then mean over tokens
    spectral = np.zeros(max(0, L - 2), dtype=float)
    if L >= 3:
        curv_list: List[np.ndarray] = []
        for t_idx in range(k):
            u_list = [_sqrt_embed(q_layers[ell][0, t_idx]) for ell in range(L)]
            kappa_t, _ = _curvature_simp(u_list, eps_curv=EPS_CURV)
            curv_list.append(kappa_t)
        if curv_list:
            spectral = np.stack(curv_list, axis=0).mean(axis=0)

    return ThreeMetrics(drift=drift, thermo=thermo, spectral=spectral)
