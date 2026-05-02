"""
ndna_lib.geometry.py

Method 5 (alternative) geometry utilities:

- Adapters for GPT-2 / DialoGPT / LLaMA-style causal LMs
- Spectral curvature κ_ℓ on the logit simplex (per prompt, via logit lens)
- Parameter-space thermodynamic length E_ℓ (observed Fisher, block-only)
- Prediction-space Fisher–Rao geometry across depth:
    * Δ_ℓ     : mean FR step length between layers ℓ-1 and ℓ
    * α_ℓ     : cosine alignment between mean belief push and mean observed chord
    * ||v̄_ℓ|| : norm of mean belief push at layer ℓ (sanity check)
- Belief vector field norms per concept v_ℓ(c)
- nDNA_pred(c) = Σ_{ℓ=2}^{L-1} κ_ℓ · Δ_ℓ · v_ℓ(c)

Everything is written to be as GPU friendly as possible under the constraints
of the theory (per-example gradients in parameter space, exact FR, etc.).

-----------------------------------------------------------------------
IMPORTANT CHANGE (ONLY SPECTRAL CURVATURE):

This file previously used a second-difference curvature estimator on the
Fisher–Rao sphere. It now uses an intrinsic turning-angle curvature:

- u_ℓ(x,t) = sqrt(q_ℓ(x,t)), where q_ℓ is the logit-lens next-token distribution.
- a = arccos(<u_{ℓ-1}, u_ℓ>), b = arccos(<u_ℓ, u_{ℓ+1}>), c = arccos(<u_{ℓ-1}, u_{ℓ+1}>)
- cos(theta) = (cos c - cos a cos b) / (sin a sin b)
  (implemented using inner products directly for stability)
- Curvature is turning per Fisher–Rao length:
    d_FR(u,v) = 2 arccos(<u,v>)  => local mean step length = (d_FR(a)+d_FR(b))/2 = a+b
  so:
    κ_turn = theta / (a + b + eps)

Degenerate triangles (sin(a)sin(b) ~ 0) are EXCLUDED from the expectation
and are NOT counted as 0.
-----------------------------------------------------------------------
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import PreTrainedModel
from tqdm import tqdm

# ---------------------------------------------------------------------
# Global config
# ---------------------------------------------------------------------
DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
AMP_DTYPE = (
    torch.bfloat16
    if (DEVICE == "cuda" and torch.cuda.is_bf16_supported())
    else torch.float16
)
AMP_ON: bool = DEVICE == "cuda"

# Reasonable defaults; scripts can override by passing their own loaders etc.
OBSERVED_FISHER_UNIT = "sequence"  # "sequence" or "token"

# Small epsilons for numerical stability
EPS_DIST = 1e-12
EPS_CURV = 1e-12
EPS_DIV = 1e-12

if DEVICE == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass


# ---------------------------------------------------------------------
# Adapters
# ---------------------------------------------------------------------
class BaseAdapter(ABC):
    """
    Minimal interface for all ndna adapters.
    """

    def __init__(self, model: PreTrainedModel):
        self.model = model
        self.blocks: Sequence[nn.Module] = []
        self.final_ln: nn.Module
        self.lm_head: nn.Module
        self.num_layers: int = 0

    @abstractmethod
    def get_block_params(self, ell: int) -> List[torch.nn.Parameter]:
        ...

    @abstractmethod
    def lens_logits(self, h_layer: torch.Tensor) -> torch.Tensor:
        ...


class GPT2StyleAdapter(BaseAdapter):
    """
    GPT-2 / DialoGPT / Falcon-style adapter.

    Exposes:
    - blocks: list of decoder blocks
    - final_ln: final layernorm before lm_head
    - lm_head: tied output projection
    - num_layers: number of transformer blocks
    """

    def __init__(self, model: PreTrainedModel):
        super().__init__(model)
        # GPT-2, DialoGPT, Falcon all use a `transformer` module with `.h` blocks
        self.trans = model.transformer
        self.blocks = self.trans.h
        # GPT-2 and Falcon both expose ln_f + lm_head in the same way
        self.final_ln = self.trans.ln_f
        self.lm_head = model.lm_head
        self.num_layers = len(self.blocks)

    def get_block_params(self, ell: int) -> List[torch.nn.Parameter]:
        """Parameters of block ℓ only (θ_ℓ)."""
        return list(self.blocks[ell].parameters())

    def lens_logits(self, h_layer: torch.Tensor) -> torch.Tensor:
        """Apply logit lens: final_ln + lm_head. h_layer: (...,H) -> (...,V)."""
        return self.lm_head(self.final_ln(h_layer))


class LlamaStyleAdapter(BaseAdapter):
    """
    Adapter for Llama-style / Phi-style models (and PEFT-wrapped versions).

    Exposes:
      - blocks: transformer blocks
      - final_ln: final layer norm before logits
      - lm_head: output projection used for logit lens
      - num_layers: number of blocks

    Handles:
      - LLaMA, TinyLlama, Mistral, Mixtral, Gemma, Qwen, DeepSeek, Phi
      - ForCausalLM models and bare backbones
      - Cases where there is no explicit head by building a tied head
        from embed_tokens or input embeddings.
    """

    def __init__(self, model: PreTrainedModel):
        super().__init__(model)
        # Keep original around
        self.model = model

        # 1) Unwrap PEFT wrapper if present to find backbone
        raw = model
        backbone = raw
        if hasattr(backbone, "base_model") and isinstance(
            backbone.base_model, PreTrainedModel
        ):
            backbone = backbone.base_model

        # 2) Find decoder/backbone module where blocks and norms live
        if hasattr(backbone, "model"):
            dec = backbone.model
        elif hasattr(backbone, "transformer"):
            dec = backbone.transformer
        elif hasattr(backbone, "backbone"):
            dec = backbone.backbone
        else:
            dec = backbone

        self.raw = raw
        self.backbone = backbone
        self.dec = dec

        # 3) Blocks
        if hasattr(dec, "layers"):
            self.blocks = dec.layers
        elif hasattr(dec, "h"):
            # GPT-J style layout fallback
            self.blocks = dec.h
        else:
            raise AttributeError(
                f"LlamaStyleAdapter: decoder has no 'layers' or 'h' (got {type(dec)})"
            )

        # 4) Final layer norm
        if hasattr(dec, "norm"):
            self.final_ln = dec.norm
        elif hasattr(dec, "final_layernorm"):
            # Phi-2
            self.final_ln = dec.final_layernorm
        elif hasattr(dec, "ln_f"):
            self.final_ln = dec.ln_f
        else:
            raise AttributeError(
                "LlamaStyleAdapter: could not find final layer norm "
                "(.norm / .final_layernorm / .ln_f)"
            )

        # 5) Output head (lm_head / output / embed_out)
        head = None
        candidates = [
            raw,
            getattr(raw, "base_model", None),
            backbone,
            dec,
        ]

        for cand in candidates:
            if cand is None:
                continue
            if hasattr(cand, "lm_head"):
                head = cand.lm_head
                break
            if hasattr(cand, "output"):
                head = cand.output
                break
            if hasattr(cand, "embed_out"):
                head = cand.embed_out
                break
            if hasattr(cand, "get_output_embeddings"):
                try:
                    maybe_head = cand.get_output_embeddings()
                except TypeError:
                    maybe_head = None
                if maybe_head is not None:
                    head = maybe_head
                    break

        if head is None:
            # Fallback: construct a tied linear head from embeddings
            emb = None
            if hasattr(dec, "embed_tokens"):
                emb = dec.embed_tokens
            elif hasattr(raw, "get_input_embeddings"):
                emb = raw.get_input_embeddings()

            if emb is None:
                raise AttributeError(
                    "LlamaStyleAdapter: could not find or construct an output head. "
                    f"Model type: {type(model)}. If this is a bare backbone, either use "
                    "the *ForCausalLM variant or ensure there is an embed_tokens / "
                    "input embedding module to tie from."
                )

            weight = emb.weight  # (vocab_size, hidden_size)
            vocab_size, hidden_size = weight.shape
            proj = nn.Linear(hidden_size, vocab_size, bias=False)
            # Tie weights: no extra storage
            proj.weight = weight
            head = proj

        self.lm_head = head
        self.num_layers = len(self.blocks)

    def get_block_params(self, ell: int) -> List[torch.nn.Parameter]:
        """Trainable parameters of block ℓ."""
        return list(self.blocks[ell].parameters())

    def lens_logits(self, h_layer: torch.Tensor) -> torch.Tensor:
        """
        Logit lens: (...,H) -> (...,V) via final_ln + lm_head.
        """
        return self.lm_head(self.final_ln(h_layer))


class GPTNeoXAdapter(BaseAdapter):
    """
    GPT-NeoX-style adapter (e.g. gpt_neox).

    HF GPT-NeoX models use:
      - model.gpt_neox.layers          : list of blocks
      - model.gpt_neox.final_layer_norm: final norm
      - model.embed_out                : lm_head
    """

    def __init__(self, model: PreTrainedModel):
        super().__init__(model)
        self.trans = model.gpt_neox
        self.blocks = self.trans.layers
        self.final_ln = self.trans.final_layer_norm
        self.lm_head = model.embed_out
        self.num_layers = len(self.blocks)

    def get_block_params(self, ell: int) -> List[torch.nn.Parameter]:
        """Parameters of block ℓ only (θ_ℓ)."""
        return list(self.blocks[ell].parameters())

    def lens_logits(self, h_layer: torch.Tensor) -> torch.Tensor:
        """Apply logit lens: final_ln + lm_head. h_layer: (...,H) -> (...,V)."""
        return self.lm_head(self.final_ln(h_layer))


def make_adapter(model: PreTrainedModel, model_name: Optional[str]) -> BaseAdapter:
    """
    Choose the right adapter based on model.config.model_type and model_name.

    Routing:

      GPT2StyleAdapter:
        - model_type or model_name contains: "gpt2", "dialogpt", "falcon"

      LlamaStyleAdapter:
        - model_type or model_name contains any of:
            "llama", "tinyllama", "mistral", "mixtral",
            "gemma", "qwen", "deepseek", "phi"

      GPTNeoXAdapter:
        - model_type or model_name contains: "gpt_neox", "gpt-neox"
    """
    mt = getattr(model.config, "model_type", "") or ""
    mt = mt.lower()
    ln = (model_name or "").lower()

    # GPT-2 / DialoGPT / Falcon
    gpt2_like = ["gpt2", "dialogpt", "falcon"]
    if any(tok in mt for tok in gpt2_like) or any(tok in ln for tok in gpt2_like):
        return GPT2StyleAdapter(model)

    # GPT-NeoX
    if ("gpt_neox" in mt) or ("gpt-neox" in mt) or ("gpt_neox" in ln) or ("gpt-neox" in ln):
        return GPTNeoXAdapter(model)

    # LLaMA-family and similar
    llama_like = [
        "llama",
        "tinyllama",
        "mistral",
        "mixtral",
        "gemma",
        "qwen",
        "deepseek",
        "phi",
        "olmo",
        "gpt"
    ]
    if any(tok in mt for tok in llama_like) or any(tok in ln for tok in llama_like):
        return LlamaStyleAdapter(model)

    raise ValueError(
        f"Unsupported model for ndna adapter: model_type={model.config.model_type!r}, "
        f"name={model_name!r}"
    )


# ---------------------------------------------------------------------
# Spectral curvature κ_ℓ (per prompt, generic via adapter)
# ---------------------------------------------------------------------
def _sqrt_embed(q: torch.Tensor, eps: float = EPS_DIST) -> torch.Tensor:
    """
    Map a probability vector q (V,) to the logit simplex sphere via sqrt embedding.
    u_i = sqrt(q_i), normalized to unit l2.

    This is the usual Fisher–Rao embedding.
    """
    q = torch.clamp(q, min=eps)
    q = q / q.sum()
    u = torch.sqrt(q)
    u = u / (torch.norm(u, p=2) + 1e-30)
    return u


def _project_tangent(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    Project v onto the tangent space at u on the unit sphere: v - <u,v> u.
    u, v both (V,).
    """
    return v - torch.dot(u, v) * u


def _curvature_simp(
    u_list: Sequence[torch.Tensor],
    eps_curv: float = EPS_CURV,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Discrete curvature κ_ℓ^(simp) on the sphere using second differences.

    NOTE: Kept verbatim for backward reference. Spectral curvature used by this
    library is now the intrinsic turning-angle curvature (see below).
    """
    m = len(u_list)
    assert m >= 3

    delta_u = []
    speeds = []

    # first differences
    for ell in range(m - 1):
        u = u_list[ell]
        v = u_list[ell + 1] - u
        du = _project_tangent(u, v)
        delta_u.append(du)
        speeds.append(torch.norm(du, p=2).item())

    ks = []
    # second differences on interior nodes
    for ell in range(1, m - 1):
        u = u_list[ell]
        v2 = u_list[ell + 1] - 2 * u_list[ell] + u_list[ell - 1]
        d2u = _project_tangent(u, v2)
        num = torch.norm(d2u, p=2)
        s = torch.norm(delta_u[ell], p=2)
        denom = (s * s + eps_curv) ** 1.5
        k = (num / denom).item()
        ks.append(k)

    return np.asarray(ks, dtype=float), np.asarray(speeds, dtype=float)


def _speeds_from_u_list(u_list: Sequence[torch.Tensor]) -> np.ndarray:
    """
    Preserve the old 'speeds' output: ||Π(u_ℓ) (u_{ℓ+1}-u_ℓ)|| for ℓ=0..m-2.
    """
    m = len(u_list)
    if m < 2:
        return np.zeros(0, dtype=float)
    speeds: List[float] = []
    for ell in range(m - 1):
        u = u_list[ell]
        v = u_list[ell + 1] - u
        du = _project_tangent(u, v)
        speeds.append(float(torch.norm(du, p=2).item()))
    return np.asarray(speeds, dtype=float)


def _turning_angle_curvature_u_list(
    u_list: Sequence[torch.Tensor],
    eps_curv: float = EPS_CURV,
    denom_eps: float = 1e-8,
) -> np.ndarray:
    """
    Intrinsic turning-angle curvature for a single prompt curve u_0..u_{m-1}.

    Curvature at interior node ℓ uses (u_{ℓ-1}, u_ℓ, u_{ℓ+1}).

    IMPORTANT UNITS FIX (constant factor caveat):
      FR distance is d_FR(u,v)=2*acos(<u,v>).
      Using a=acos(<u_{ℓ-1},u_ℓ>) means FR step is 2a.
      Turning angle theta is metric-angle invariant under constant scaling,
      but curvature must be "turning per FR length".
      Local mean FR length = (2a+2b)/2 = a+b.
      So:
        κ_turn = theta / (a+b+eps).

    Returns:
      np.array shape (m-2,), for ℓ=1..m-2.
      Degenerate cases (sin(a)sin(b) too small) are set to 0.0 here to keep
      prompt-mode output fully finite/serializable; dataset estimator excludes them.
    """
    m = len(u_list)
    if m < 3:
        return np.zeros(0, dtype=float)

    ks: List[float] = []
    for ell in range(1, m - 1):
        u_minus = u_list[ell - 1]
        u_ctr = u_list[ell]
        u_plus = u_list[ell + 1]

        cos_a = torch.dot(u_minus, u_ctr).clamp(-1.0, 1.0)
        cos_b = torch.dot(u_ctr, u_plus).clamp(-1.0, 1.0)
        cos_c = torch.dot(u_minus, u_plus).clamp(-1.0, 1.0)

        # sphere angles a,b,c
        a = torch.acos(cos_a)
        b = torch.acos(cos_b)

        sin_a = torch.sqrt(torch.clamp(1.0 - cos_a * cos_a, min=0.0))
        sin_b = torch.sqrt(torch.clamp(1.0 - cos_b * cos_b, min=0.0))
        denom = sin_a * sin_b

        if float(denom.item()) <= denom_eps:
            ks.append(0.0)
            continue

        cos_theta = (cos_c - cos_a * cos_b) / (denom + eps_curv)
        cos_theta = cos_theta.clamp(-1.0, 1.0)
        theta = torch.acos(cos_theta)

        # per-FR-length normalization: local mean FR length = a+b
        kappa = theta / (a + b + eps_curv)
        k_val = float(kappa.item())
        if not math.isfinite(k_val):
            k_val = 0.0
        ks.append(k_val)

    return np.asarray(ks, dtype=float)


@torch.no_grad()
def _layerwise_next_token_distributions(
    model: PreTrainedModel,
    adapter: BaseAdapter,
    tokenizer,
    text: str,
) -> List[torch.Tensor]:
    """
    Generic layer-wise next-token distributions for a single prompt.

    - Run model with output_hidden_states=True.
    - At last context position t, take hidden state at every layer (including embeddings).
    - Apply logit lens (adapter.final_ln + adapter.lm_head) to get logits, then softmax.
    - Return [q_0, ..., q_L] as list of (V,) tensors.
    """
    enc = tokenizer(text, return_tensors="pt", add_special_tokens=False).to(DEVICE)
    if enc.input_ids.shape[1] == 0:
        raise ValueError("Empty tokenized input.")

    model.eval()
    outputs = model(**enc, output_hidden_states=True, use_cache=False)
    hidden_states = outputs.hidden_states  # length n_layer+1

    t = enc.input_ids.shape[1] - 1
    q_list: List[torch.Tensor] = []

    for h in hidden_states:
        # Take last position, keep a singleton seq dimension so lens_logits
        # always sees (..., H).
        h_last = h[:, t : t + 1, :]  # (B,1,H)
        logits = adapter.lens_logits(h_last)  # (B,1,V)
        logits_vec = logits[:, 0, :]  # (B,V), usually B=1
        q = torch.softmax(logits_vec[0].to(torch.float32), dim=-1)  # (V,)
        q_list.append(q)

    return q_list


def spectral_curvature_for_prompts(
    model: PreTrainedModel,
    tokenizer,
    prompts: Sequence[Tuple[str, str]],
    model_name: Optional[str] = None,
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Compute spectral curvature κ_ℓ for a small list of prompts using
    the logit-lens view on the final next-token distribution.

    CURRENT estimator (intrinsic turning-angle curvature) on the FR sphere:
      κ_turn(ℓ) = E_t [ theta_ℓ / (a_ℓ + b_ℓ + eps) ]
    where a_ℓ = acos(<u_{ℓ-1},u_ℓ>), b_ℓ = acos(<u_ℓ,u_{ℓ+1}>).

    Works for any decoder-only model supported by make_adapter.

    Returns:
        results[name] = {
            "curvature": np.array of shape (m-2,), κ_ℓ for ℓ=1..m-2
            "speeds":    np.array of shape (m-1,), legacy projected-chord speeds
            "num_nodes": m = L+1,
        }
    """
    model.eval()
    adapter = make_adapter(
        model,
        model_name or getattr(model.config, "_name_or_path", None),
    )
    results: Dict[str, Dict[str, np.ndarray]] = {}

    for name, text in tqdm(prompts, desc="Spectral curvature (prompts)", unit="prompt"):
        q_list = _layerwise_next_token_distributions(model, adapter, tokenizer, text)
        u_list = [_sqrt_embed(q) for q in q_list]

        k_list = _turning_angle_curvature_u_list(u_list, eps_curv=EPS_CURV, denom_eps=1e-8)
        s_list = _speeds_from_u_list(u_list)

        results[name] = {
            "curvature": k_list,
            "speeds": s_list,
            "num_nodes": np.array([len(u_list)], dtype=int),
        }

    return results


def _select_positions_mask_from_labels(
    labels: torch.Tensor,
    keep_last_k: Optional[int] = None,
) -> torch.Tensor:
    """
    Build a boolean mask of supervised token positions to include in curvature averaging.

    labels: (B,S) with -100 for ignored positions.
    keep_last_k:
      - None or <=0: keep all supervised positions (labels != -100)
      - else: for each sample, keep only last K supervised positions
    """
    valid = labels != -100  # (B,S)
    if keep_last_k is None or keep_last_k <= 0:
        return valid

    B, S = labels.shape
    sel = torch.zeros_like(valid)
    for b in range(B):
        idx = torch.nonzero(valid[b], as_tuple=False).squeeze(-1)
        if idx.numel() == 0:
            continue
        take = idx[-min(int(keep_last_k), int(idx.numel())) :]
        sel[b, take] = True
    return sel


@torch.no_grad()
def _sqrt_embed_batch_from_logits(
    logits: torch.Tensor,
    tau: float,
    eps_dist: float,
) -> torch.Tensor:
    """
    logits: (N,V) (any dtype)
    Returns u: (N,V) float32, unit-norm rows on the FR sphere (sqrt simplex embedding).

    Important correctness detail:
      - Clamp is followed by renormalization so q stays on the simplex.
      - Then u is explicitly L2-normalized (cheap safety).
    """
    q = F.softmax((logits.float() / float(tau)), dim=-1)  # float32
    q = torch.clamp(q, min=eps_dist)
    q = q / q.sum(dim=-1, keepdim=True)  # restore simplex after clamp
    u = torch.sqrt(q)
    u = u / (torch.linalg.norm(u, dim=-1, keepdim=True) + 1e-30)
    return u


@torch.no_grad()
def _turning_angle_kappa_batch(
    u_minus: torch.Tensor,
    u_ctr: torch.Tensor,
    u_plus: torch.Tensor,
    eps_curv: float = EPS_CURV,
    denom_eps: float = 1e-8,
) -> torch.Tensor:
    """
    Batched intrinsic turning-angle curvature at u_ctr, using neighbors u_minus/u_plus.

    Returns:
      kappa: (N,) float32, with NaN for degenerate triangles (excluded from expectation).

    IMPORTANT UNITS FIX:
      curvature per Fisher–Rao length uses denominator (a + b),
      where a = acos(<u_minus,u_ctr>), b = acos(<u_ctr,u_plus>).
      (Because d_FR = 2a and mean FR step is (2a+2b)/2 = a+b.)
    """
    cos_a = (u_minus * u_ctr).sum(dim=-1).clamp(-1.0, 1.0)  # cos(a)
    cos_b = (u_ctr * u_plus).sum(dim=-1).clamp(-1.0, 1.0)   # cos(b)
    cos_c = (u_minus * u_plus).sum(dim=-1).clamp(-1.0, 1.0) # cos(c)

    a = torch.acos(cos_a)
    b = torch.acos(cos_b)

    sin_a = torch.sqrt(torch.clamp(1.0 - cos_a * cos_a, min=0.0))
    sin_b = torch.sqrt(torch.clamp(1.0 - cos_b * cos_b, min=0.0))
    denom = sin_a * sin_b

    kappa = torch.full_like(denom, float("nan"), dtype=torch.float32)

    good = denom > denom_eps
    if good.any():
        # spherical law of cosines at middle vertex:
        # cos(theta) = (cos c - cos a cos b) / (sin a sin b)
        cos_theta = (cos_c - cos_a * cos_b) / (denom + eps_curv)
        cos_theta = cos_theta.clamp(-1.0, 1.0)
        theta = torch.acos(cos_theta)

        # per-FR-length normalization: mean FR step = a+b
        kappa_good = theta / (a + b + eps_curv)
        kappa[good] = kappa_good[good].float()

    return kappa


@torch.no_grad()
def spectral_curvature_for_loader(
    model: PreTrainedModel,
    adapter: BaseAdapter,
    loader: DataLoader,
    tau: float = 1.0,
    keep_last_k: Optional[int] = None,
    include_embedding_node: bool = True,
    eps_dist: float = EPS_DIST,
    eps_curv: float = EPS_CURV,
) -> Tuple[np.ndarray, int]:
    """
    Dataset-style spectral curvature estimator matching the Method-5 geometry,
    but using intrinsic turning-angle curvature.

    Let u_j(x,t) = sqrt( softmax( logit_lens(h_j(x,t)) / tau ) ) ∈ S^{V-1}_+.

    For each interior depth node ℓ (ℓ=1..m-2), define:
      a = acos(<u_{ℓ-1}, u_ℓ>)
      b = acos(<u_ℓ, u_{ℓ+1}>)
      c = acos(<u_{ℓ-1}, u_{ℓ+1}>)
      cos(theta) = (cos c - cos a cos b) / (sin a sin b)

    Curvature per token position:
      κ_turn = theta / (a + b + eps)

    IMPORTANT:
      - This fixes the constant-factor mismatch with Δ_ℓ elsewhere in this file,
        because Δ_ℓ is computed as FR step length 2*acos(<u_{ℓ-1},u_ℓ>).
        Using (a+b) corresponds exactly to mean FR length (2a+2b)/2.

      - Degenerate triangles (sin(a)sin(b) too small) are EXCLUDED:
        they contribute neither sum nor count. They are NOT treated as 0.

    Averaging is over teacher-forced supervised token positions (labels != -100),
    optionally restricted to the last K supervised positions per sample for speed.

    Returns:
      kappa_mean: np.array shape (m-2,) where m is number of nodes used
                  - if include_embedding_node=True: m = L+1 (embedding + L blocks), kappa length L-1
                  - else: m = L (blocks only), kappa length L-2
      n_positions: number of (x,t) positions used in the expectation (selected supervised positions)
    """
    model.eval()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    AMP_DEVICE = "cuda" if DEVICE == "cuda" else "cpu"

    k_sum = None
    k_cnt = None
    total_positions = 0

    for batch in tqdm(loader, desc="Spectral curvature (batches)", unit="batch"):
        input_ids = batch["input_ids"].to(DEVICE, non_blocking=True)
        labels = batch["labels"].to(DEVICE, non_blocking=True)
        attn_mask = batch["attention_mask"].to(DEVICE, non_blocking=True)

        # Which (x,t) positions to include
        sel_mask = _select_positions_mask_from_labels(labels, keep_last_k=keep_last_k)  # (B,S)
        keep_flat = sel_mask.view(-1)
        N = int(keep_flat.sum().item())
        if N == 0:
            continue
        total_positions += N

        with torch.autocast(device_type=AMP_DEVICE, dtype=AMP_DTYPE, enabled=(DEVICE == "cuda")):
            outputs = model(
                input_ids=input_ids,
                attention_mask=attn_mask,
                output_hidden_states=True,
                use_cache=False,
            )

        hs = outputs.hidden_states  # tuple length L+1 (emb + each block output)
        if not include_embedding_node:
            hs = hs[1:]  # blocks only

        m = len(hs)  # number of nodes along depth
        if m < 3:
            continue

        # Allocate accumulators once we know m
        if k_sum is None:
            k_sum = torch.zeros(m - 2, device=DEVICE, dtype=torch.float64)
            k_cnt = torch.zeros(m - 2, device=DEVICE, dtype=torch.float64)

        # Sliding window u_{j-2}, u_{j-1}, u_j to compute curvature at center (j-1)
        u_prevprev = None
        u_prev = None

        for j in range(m):
            h_j = hs[j]  # (B,S,H)
            B, S, H = h_j.shape

            # Gather only selected positions: (N,H)
            h_flat = h_j.reshape(B * S, H)[keep_flat]

            # Logit lens -> u on FR sphere (unit rows)
            with torch.autocast(device_type=AMP_DEVICE, dtype=AMP_DTYPE, enabled=(DEVICE == "cuda")):
                logits = adapter.lens_logits(h_flat)  # (N,V)

            u = _sqrt_embed_batch_from_logits(logits, tau=tau, eps_dist=eps_dist)  # (N,V) float32

            # Once we have (u_{j-2}, u_{j-1}, u_j), compute curvature at center ell=j-1
            if u_prevprev is not None and u_prev is not None:
                ell = j - 1
                if 1 <= ell <= (m - 2):
                    kappa_vec = _turning_angle_kappa_batch(
                        u_minus=u_prevprev,
                        u_ctr=u_prev,
                        u_plus=u,
                        eps_curv=eps_curv,
                        denom_eps=1e-8,
                    )  # (N,) float32 with NaNs for degenerate

                    finite = torch.isfinite(kappa_vec)
                    if finite.any():
                        idx = ell - 1  # ell=1..m-2 -> idx=0..m-3
                        k_sum[idx] += kappa_vec[finite].double().sum()
                        k_cnt[idx] += float(finite.sum().item())

                    del kappa_vec, finite
                    _free_cuda()

            # slide window
            u_prevprev, u_prev = u_prev, u

            del logits, u, h_j, h_flat
            _free_cuda()

        del outputs, hs, input_ids, labels, attn_mask, sel_mask, keep_flat
        _free_cuda()

    if k_sum is None:
        # no valid positions
        return np.zeros(0, dtype=float), 0

    # If some layers got zero valid samples (rare), return 0.0 there to keep output finite.
    kappa_mean = torch.zeros_like(k_sum, dtype=torch.float64)
    nonzero = k_cnt > 0
    kappa_mean[nonzero] = k_sum[nonzero] / k_cnt[nonzero]
    kappa_mean = kappa_mean.detach().cpu().numpy()

    return kappa_mean, int(total_positions)


# ---------------------------------------------------------------------
# Data helper for causal LM: teacher-forced shift
# ---------------------------------------------------------------------
def collate_causal(tokenizer, max_len: int):
    """
    Standard right-padded teacher-forced causal collation.

    Returns a function mapping a list of {"text": str} examples to:
        {
            "input_ids": (B,S),
            "labels":    (B,S),
            "attention_mask": (B,S)
        }
    where labels are shifted and padded tokens are set to -100.
    """

    def _fn(batch):
        texts = [ex["text"] for ex in batch]
        tok = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        )
        ids_full = tok["input_ids"]
        am_full = tok["attention_mask"]

        # shift for next-token prediction
        input_ids = ids_full[:, :-1]
        labels = ids_full[:, 1:].clone()
        input_mask = am_full[:, :-1]
        target_mask = am_full[:, 1:]
        labels[target_mask == 0] = -100

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": input_mask,
        }

    return _fn


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------
@torch.no_grad()
def _count_params(params: Sequence[torch.nn.Parameter]) -> int:
    return sum(p.numel() for p in params)


def _safe_div(a: torch.Tensor, b: torch.Tensor, eps: float = EPS_DIV) -> torch.Tensor:
    return a / (b + eps)


def _free_cuda():
    if DEVICE == "cuda":
        torch.cuda.empty_cache()


@torch.no_grad()
def _project_tangent_batch(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    Batched tangent projection on last dimension.
    u, v: (..., V). Returns v - u * <u,v>.
    """
    uv = (u * v).sum(dim=-1, keepdim=True)
    return v - u * uv


# ---------------------------------------------------------------------
# Submethod 1: Parameter-space effort E_ℓ (observed Fisher per block)
# ---------------------------------------------------------------------
def compute_param_effort(
    model: PreTrainedModel,
    adapter: BaseAdapter,
    loader: DataLoader,
    unit: str = OBSERVED_FISHER_UNIT,
) -> Tuple[np.ndarray, int, List[int]]:
    """
    E_ℓ = E_x ||∇_{θ_ℓ} log p_ℓ(x)||^2, gradients w.r.t. BLOCK-ℓ parameters only.

    Args:
        model:   PreTrainedModel (causal LM)
        adapter: GPT2StyleAdapter or LlamaStyleAdapter or GPTNeoXAdapter
        loader:  DataLoader yielding dict with keys: input_ids, labels, attention_mask
        unit:    "sequence" (average over sequences) or "token" (average over valid tokens)

    Returns:
        mean_sq:   np.array shape (L,) with E_ℓ for ℓ=0..L-1
        n_examples: number of sequences/tokens used (depending on unit)
        n_params:  list of parameter counts per block
    """
    L = adapter.num_layers
    layer_sum_sq = torch.zeros(L, device=DEVICE, dtype=torch.float64)
    layer_params = [adapter.get_block_params(i) for i in range(L)]
    n_params = [_count_params(ps) for ps in layer_params]

    model.eval()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    n_examples_total = 0

    for batch in tqdm(loader, desc="Parameter-space effort E_l", unit="batch"):
        input_ids = batch["input_ids"].to(DEVICE, non_blocking=True)
        labels = batch["labels"].to(DEVICE, non_blocking=True)
        input_mask = batch["attention_mask"].to(DEVICE, non_blocking=True)
        B, S = input_ids.shape

        if unit == "sequence":
            batch_examples = B
        else:
            batch_examples = int((labels != -100).sum().item())

        with torch.enable_grad():
            with torch.autocast(device_type="cuda", dtype=AMP_DTYPE, enabled=AMP_ON):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=input_mask,
                    output_hidden_states=True,
                    use_cache=False,
                )
                hs = outputs.hidden_states  # tuple len L+1

        labels_flat = labels.reshape(-1)

        for ell in range(L):
            params_ell = layer_params[ell]
            # Even if no trainable params in this block, we still run the forward
            # but skip gradient accumulation.
            h_ell = hs[ell + 1]  # after block ell

            with torch.autocast(device_type="cuda", dtype=AMP_DTYPE, enabled=AMP_ON):
                logits_ell = adapter.lens_logits(h_ell)  # (B,S,V)
                V = logits_ell.size(-1)
                loss_tok = F.cross_entropy(
                    logits_ell.view(-1, V),
                    labels_flat,
                    ignore_index=-100,
                    reduction="none",
                ).view(B, S)

                if unit == "sequence":
                    loss_per_ex = loss_tok.sum(dim=1)
                else:
                    loss_per_ex = loss_tok[labels != -100]

            acc = torch.tensor(0.0, device=DEVICE, dtype=torch.float64)

            if len(params_ell) > 0:
                if unit == "sequence":
                    for b in range(B):
                        l_scalar = loss_per_ex[b]
                        if not torch.isfinite(l_scalar):
                            continue
                        grads = torch.autograd.grad(
                            outputs=l_scalar,
                            inputs=params_ell,
                            retain_graph=True,
                            allow_unused=True,
                            create_graph=False,
                        )
                        for g in grads:
                            if g is not None:
                                acc += (g.detach().double() ** 2).sum()
                else:
                    idx = torch.nonzero(labels != -100, as_tuple=False)
                    for k in range(idx.size(0)):
                        b_idx, s_idx = int(idx[k, 0]), int(idx[k, 1])
                        l_scalar = loss_tok[b_idx, s_idx]
                        grads = torch.autograd.grad(
                            outputs=l_scalar,
                            inputs=params_ell,
                            retain_graph=True,
                            allow_unused=True,
                            create_graph=False,
                        )
                        for g in grads:
                            if g is not None:
                                acc += (g.detach().double() ** 2).sum()

            layer_sum_sq[ell] += acc

            del logits_ell, loss_tok, loss_per_ex, acc
            _free_cuda()

        n_examples_total += batch_examples

        del outputs, hs, labels_flat
        _free_cuda()

    mean_sq = (layer_sum_sq / max(1, n_examples_total)).cpu().numpy()
    return mean_sq, int(n_examples_total), n_params


# ---------------------------------------------------------------------
# Submethod 2: FR geometry + alignment (single forward, streaming over layers)
# ---------------------------------------------------------------------
@torch.no_grad()
def compute_fr_and_alignment_streaming(
    model: PreTrainedModel,
    adapter: BaseAdapter,
    loader: DataLoader,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, int]:
    """
    Fisher–Rao geometry across depth with a single forward per batch.

    For each batch:
        - Do one forward pass with output_hidden_states=True, giving hidden_states[1..L].
        - Sequentially compute q_ℓ and u_ℓ = sqrt(q_ℓ) for ℓ = 0..L-1.
        - For each step ℓ = 1..L-1:
            Δ_ℓ: mean FR distance 2·arccos(<u_{ℓ-1}, u_ℓ>) over valid tokens
            v_obs_ℓ: observed chord projected onto tangent at u_{ℓ-1}
            t_{ℓ-1}: theoretical belief push at layer ℓ-1
        - Aggregate over tokens/batches to get:
            Δ_ℓ          (mean FR step length)
            α_ℓ          (cosine between mean t_{ℓ-1} and mean v_obs_ℓ)
            ||t̄_{ℓ-1}||  (norm of mean belief push, sanity check)
            mean_total_FR (mean Σ_ℓ Δ_ℓ per token)

    Returns:
        Delta   : np.array shape (L-1,) with Δ_ℓ (ℓ=1..L-1)
        Alpha   : np.array shape (L-1,) with α_ℓ in [-1,1]
        Vnorm   : np.array shape (L-1,) with ||mean belief push at layer ℓ-1||
        mean_total_fr: scalar, mean total FR length per token
        n_tokens: int, number of valid tokens
    """
    model.eval()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    L = adapter.num_layers
    V = model.get_input_embeddings().num_embeddings

    step_sum = torch.zeros(L - 1, device=DEVICE, dtype=torch.float64)
    step_cnt = torch.zeros(L - 1, device=DEVICE, dtype=torch.float64)

    v_sum = torch.zeros(L - 1, V, device=DEVICE, dtype=torch.float32)
    v_cnt = torch.zeros(L - 1, device=DEVICE, dtype=torch.float64)
    d_sum = torch.zeros(L - 1, V, device=DEVICE, dtype=torch.float32)
    d_cnt = torch.zeros(L - 1, device=DEVICE, dtype=torch.float64)

    total_len_sum = 0.0
    total_tokens = 0

    for batch in tqdm(loader, desc="FR geometry + alignment", unit="batch"):
        input_ids = batch["input_ids"].to(DEVICE, non_blocking=True)
        labels = batch["labels"].to(DEVICE, non_blocking=True)
        input_mask = batch["attention_mask"].to(DEVICE, non_blocking=True)

        valid = labels != -100  # (B,S)
        mask_f = valid.float().unsqueeze(-1)  # (B,S,1)

        with torch.autocast(device_type="cuda", dtype=AMP_DTYPE, enabled=AMP_ON):
            outputs = model(
                input_ids=input_ids,
                attention_mask=input_mask,
                output_hidden_states=True,
                use_cache=False,
            )

        hidden_states = outputs.hidden_states[1:]  # list length L, skip embeddings
        B, S, _ = hidden_states[0].shape

        # layer 0
        logits0 = adapter.lens_logits(hidden_states[0])
        logp0 = F.log_softmax(logits0.float(), dim=-1)
        q_prev = torch.exp(logp0)  # (B,S,V)
        u_prev = torch.sqrt(q_prev + 1e-45)  # (B,S,V)
        del logits0, logp0
        _free_cuda()

        for ell in range(1, L):
            logits_curr = adapter.lens_logits(hidden_states[ell])
            logp_curr = F.log_softmax(logits_curr.float(), dim=-1)
            q_curr = torch.exp(logp_curr)
            u_curr = torch.sqrt(q_curr + 1e-45)
            del logits_curr, logp_curr
            _free_cuda()

            # FR step between u_prev and u_curr
            inner = (u_prev * u_curr).sum(dim=-1).clamp(-1.0, 1.0)  # (B,S)
            fr_step = 2.0 * torch.acos(inner)
            fr_step = fr_step.masked_fill(~valid, 0.0)

            step_sum[ell - 1] += fr_step.double().sum()
            step_cnt[ell - 1] += valid.sum()

            total_len_sum += fr_step[valid].double().sum().item()
            total_tokens += int(valid.sum().item())

            # Observed chord (projected at u_prev)
            v_obs = _project_tangent_batch(u_prev, u_curr - u_prev)
            v_obs = (v_obs * mask_f).float()
            d_sum[ell - 1] += v_obs.sum(dim=(0, 1))
            d_cnt[ell - 1] += float(valid.sum().item())
            del v_obs
            _free_cuda()

            # Belief push at previous layer using q_prev, u_prev
            u_times_g = -(u_prev * q_prev)  # (B,S,V)
            y = torch.clamp(labels, min=0)  # (B,S)
            u_y = u_prev.gather(-1, y.unsqueeze(-1)).squeeze(-1)  # (B,S)
            u_times_g.scatter_add_(-1, y.unsqueeze(-1), u_y.unsqueeze(-1))

            q_sq = (q_prev * q_prev).sum(dim=-1)  # (B,S)
            q_y = q_prev.gather(-1, y.unsqueeze(-1)).squeeze(-1)  # (B,S)
            q_dot_g = q_y - q_sq  # (B,S)

            t_prev = 0.5 * (u_times_g - u_prev * q_dot_g.unsqueeze(-1))  # (B,S,V)
            t_prev = (t_prev * mask_f).float()

            v_sum[ell - 1] += t_prev.sum(dim=(0, 1))
            v_cnt[ell - 1] += float(valid.sum().item())

            del u_times_g, q_sq, q_y, q_dot_g, t_prev
            _free_cuda()

            # slide window
            del q_prev, u_prev
            q_prev, u_prev = q_curr, u_curr

        del (
            q_prev,
            u_prev,
            input_ids,
            labels,
            input_mask,
            valid,
            mask_f,
            hidden_states,
            outputs,
        )
        _free_cuda()

    Delta = (step_sum / step_cnt.clamp_min(1)).detach().cpu().numpy()

    v_mean = torch.stack(
        [_safe_div(v_sum[i], v_cnt[i].clamp_min(EPS_DIV)) for i in range(L - 1)],
        dim=0,
    )  # (L-1,V)
    d_mean = torch.stack(
        [_safe_div(d_sum[i], d_cnt[i].clamp_min(EPS_DIV)) for i in range(L - 1)],
        dim=0,
    )  # (L-1,V)

    v_norm = torch.linalg.norm(v_mean, dim=-1)  # (L-1,)
    d_norm = torch.linalg.norm(d_mean, dim=-1)  # (L-1,)
    dot = (v_mean * d_mean).sum(dim=-1)  # (L-1,)

    Alpha = _safe_div(dot, v_norm * d_norm).clamp(-1.0, 1.0).detach().cpu().numpy()
    Vnorm = v_norm.detach().cpu().numpy()

    mean_total_fr = float(total_len_sum / max(1, total_tokens))
    return Delta, Alpha, Vnorm, mean_total_fr, total_tokens


# ---------------------------------------------------------------------
# Belief vector fields for concepts (Method 5) – generic
# ---------------------------------------------------------------------
def make_causal_collate_keep_last_k(tokenizer, max_len: int, keep_last_k: int):
    """
    Collate fn that keeps only the last K supervised positions per sample
    (for belief vector field estimation).

    Returns a function mapping batch -> {
        "input_ids": (B,S),
        "labels":    (B,S),
        "select_mask": (B,S) boolean
    }
    """

    def _fn(batch):
        texts = [ex["text"] for ex in batch]
        tok = tokenizer(
            texts,
            padding="longest",
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        )
        input_ids = tok["input_ids"]
        attn_mask = tok["attention_mask"]

        x = input_ids[:, :-1].contiguous()
        y = input_ids[:, 1:].contiguous()
        msk = attn_mask[:, 1:].contiguous()
        y = y.masked_fill(msk == 0, -100)

        B, S1 = x.shape
        sel_mask = torch.zeros_like(y, dtype=torch.bool)
        for b in range(B):
            valid = (y[b] != -100).nonzero(as_tuple=False).squeeze(-1)
            if valid.numel() > 0:
                take = valid[-min(keep_last_k, valid.numel()) :]
                sel_mask[b, take] = True

        return {
            "input_ids": x,
            "labels": y,
            "select_mask": sel_mask,
        }

    return _fn


@torch.no_grad()
def belief_field_for_dataset(
    model: PreTrainedModel,
    adapter: BaseAdapter,
    tokenizer,
    dataset,
    max_seq_len: int,
    tokens_per_ex: int,
    batch_size: int = 1,
    tau: float = 1.0,
    fr_norm: bool = True,
) -> np.ndarray:
    """
    Compute layer-wise belief vector field norms ||v_ℓ|| for a single "concept"
    dataset (e.g. a SQuAD answer type or AG News class).

    This is generic over any decoder-only model supported by make_adapter.

    Args:
        model:        AutoModelForCausalLM
        adapter:      ndna adapter for this model
        tokenizer:    tokenizer
        dataset:      HF Dataset with a "text" column
        max_seq_len:  max sequence length for tokenization
        tokens_per_ex: per example, keep last K supervised positions
        batch_size:   DataLoader batch size
        tau:          temperature in softmax(q / tau)
        fr_norm:      if True, return FR norm 2·||v||; else Euclidean ||v||

    Returns:
        norms: np.array of shape (L,) with per-layer ||v_ℓ||.
    """
    from torch.utils.data import DataLoader as _DL

    model.eval().to(DEVICE)
    num_layers = adapter.num_layers
    vocab_size = model.get_input_embeddings().num_embeddings

    collate_fn = make_causal_collate_keep_last_k(
        tokenizer=tokenizer, max_len=max_seq_len, keep_last_k=tokens_per_ex
    )
    loader = _DL(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    v_sum = [
        torch.zeros(vocab_size, device=DEVICE, dtype=torch.float32)
        for _ in range(num_layers)
    ]
    u_sum = [
        torch.zeros(vocab_size, device=DEVICE, dtype=torch.float32)
        for _ in range(num_layers)
    ]
    cnt = [0 for _ in range(num_layers)]

    AMP_DEVICE = "cuda" if DEVICE == "cuda" else "cpu"

    for batch in tqdm(loader, desc="Belief field norms", unit="batch"):
        x = batch["input_ids"].to(DEVICE)
        y = batch["labels"].to(DEVICE)
        keep_m = batch["select_mask"].to(DEVICE)  # (B,S)
        B, S = x.shape

        # Attention mask: if pad_token_id is set, treat pads as 0; otherwise all 1.
        if getattr(tokenizer, "pad_token_id", None) is not None:
            attn_mask = (x != tokenizer.pad_token_id).long()
        else:
            attn_mask = torch.ones_like(x, dtype=torch.long)

        with torch.autocast(
            device_type=AMP_DEVICE, dtype=AMP_DTYPE, enabled=(DEVICE == "cuda")
        ):
            outputs = model(
                input_ids=x,
                attention_mask=attn_mask,
                output_hidden_states=True,
                use_cache=False,
            )
        hidden_states = outputs.hidden_states[1:]  # skip embeddings, len = L

        for ell in range(num_layers):
            h_ell = hidden_states[ell]  # (B,S,H)
            with torch.autocast(
                device_type=AMP_DEVICE, dtype=AMP_DTYPE, enabled=(DEVICE == "cuda")
            ):
                z = adapter.lens_logits(h_ell)  # (B,S,V)

            keep = keep_m.view(-1)
            if not keep.any():
                continue

            z_sel = z.view(-1, z.size(-1))[keep]
            y_sel = y.view(-1)[keep]

            z32 = z_sel.float() / tau
            q = F.softmax(z32, dim=-1)
            q = torch.clamp(q, min=1e-12)
            u = torch.sqrt(q)

            g = -q / tau
            add = torch.full(
                (y_sel.shape[0], 1), 1.0 / tau, device=DEVICE, dtype=q.dtype
            )
            g.scatter_add_(dim=-1, index=y_sel.unsqueeze(-1), src=add)

            ug = u * g
            s = (q * g).sum(dim=-1, keepdim=True)
            t = (ug - s * u) / (2.0 * tau)

            v_sum[ell] += t.sum(dim=0).to(v_sum[ell].dtype)
            u_sum[ell] += u.sum(dim=0).to(u_sum[ell].dtype)
            cnt[ell] += t.size(0)

        del x, y, keep_m, attn_mask, outputs, hidden_states
        _free_cuda()

    norms: List[float] = []
    for ell in range(num_layers):
        if cnt[ell] == 0:
            norms.append(0.0)
            continue
        v_avg = v_sum[ell] / cnt[ell]
        u_avg = u_sum[ell] / cnt[ell]
        u_norm = torch.linalg.norm(u_avg).clamp_min(1e-12)
        u_bar = u_avg / u_norm
        radial = torch.dot(u_bar, v_avg)
        v_tan = v_avg - radial * u_bar
        n = torch.linalg.norm(v_tan).item()
        if fr_norm:
            n *= 2.0
        norms.append(n)

    return np.asarray(norms, dtype=float)


def belief_field_norms_for_concepts(
    model: PreTrainedModel,
    adapter: BaseAdapter,
    tokenizer,
    concept_datasets: Mapping[str, object],
    max_seq_len: int,
    tokens_per_ex: int,
    batch_size: int = 1,
    tau: float = 1.0,
    fr_norm: bool = True,
) -> Dict[str, np.ndarray]:
    """
    Convenience wrapper: compute ||v_ℓ(c)|| for several concept datasets.

    Args:
        concept_datasets: dict name -> HF Dataset (with "text" column)

    Returns:
        dict name -> np.array (L,) per-layer norms.
    """
    result: Dict[str, np.ndarray] = {}
    for name, ds in concept_datasets.items():
        norms = belief_field_for_dataset(
            model,
            adapter,
            tokenizer,
            ds,
            max_seq_len=max_seq_len,
            tokens_per_ex=tokens_per_ex,
            batch_size=batch_size,
            tau=tau,
            fr_norm=fr_norm,
        )
        result[name] = norms
    return result


# ---------------------------------------------------------------------
# nDNA_pred: κ_ℓ * Δ_ℓ * v_ℓ(c) aggregation
# ---------------------------------------------------------------------
def compute_ndna_pred(
    kappa: np.ndarray,
    fr_steps: np.ndarray,
    v_norms_by_concept: Mapping[str, np.ndarray],
    l_min: int = 2,
) -> Tuple[Dict[str, float], Dict[str, np.ndarray]]:
    """
    Compute nDNA_pred(c) given:
        κ_ℓ       : spectral curvature for interior nodes (ℓ=1..L-1)  -> shape (L-1,)
        fr_steps  : FR step lengths Δ_ℓ between layers ℓ-1 and ℓ      -> shape (L-1,)
        v_norms   : ||v_ℓ(c)|| per layer ℓ=0..L-1                    -> shape (L,)

    IMPORTANT UNIT CONSISTENCY:
      - fr_steps are Fisher–Rao distances computed as 2·arccos(<u_{ℓ-1},u_ℓ>).
      - This file’s spectral curvature κ_ℓ is now computed per FR length
        (denominator a+b where a,b are acos-angles), so κ_ℓ · Δ_ℓ is scale-consistent.

    Theory:

        - Let model have L transformer blocks (ℓ = 0..L-1).
        - κ_ℓ and Δ_ℓ are defined for ℓ = 1..L-1 (we drop embedding).
        - v_ℓ(c) is defined for ℓ = 0..L-1 (belief vector at layer ℓ).
        - We sum from ℓ = 2..L-1 (drop shallow edge effects):

            nDNA_pred(c) = Σ_{ℓ=2}^{L-1} κ_ℓ · Δ_ℓ · ||v_ℓ(c)||

    In code:
        kappa[idx]    corresponds to ℓ = idx + 1
        fr_steps[idx] corresponds to ℓ = idx + 1
        v_norms[ℓ]    uses same ℓ as block index

    Args:
        kappa:    np.array, shape (L-1,)
        fr_steps: np.array, shape (L-1,)
        v_norms_by_concept: dict concept -> np.array shape(L,)
        l_min:    lower index of sum (ℓ >= l_min), default 2

    Returns:
        ndna_scalar:    dict concept -> scalar nDNA_pred(c)
        ndna_layerwise: dict concept -> np.array shape(L,) with per-layer contributions
                        (zeros where not used in the sum)
    """
    kappa = np.asarray(kappa, dtype=float)
    fr_steps = np.asarray(fr_steps, dtype=float)

    if kappa.shape != fr_steps.shape:
        raise ValueError(
            f"kappa and fr_steps must have same shape; got {kappa.shape} vs {fr_steps.shape}"
        )

    L_minus_1 = kappa.shape[0]
    L = L_minus_1 + 1  # number of blocks

    ndna_scalar: Dict[str, float] = {}
    ndna_layerwise: Dict[str, np.ndarray] = {}

    for name, v_norms in v_norms_by_concept.items():
        v_arr = np.asarray(v_norms, dtype=float)
        if v_arr.shape[0] != L:
            raise ValueError(
                f"v_norms for concept '{name}' has length {v_arr.shape[0]}, expected {L}."
            )

        contrib = np.zeros(L, dtype=float)

        for ell in range(l_min, L):  # ℓ = 2..L-1
            idx = ell - 1  # index into kappa, fr_steps
            contrib[ell] = kappa[idx] * fr_steps[idx] * v_arr[ell]

        ndna_layerwise[name] = contrib
        ndna_scalar[name] = float(contrib.sum())

    return ndna_scalar, ndna_layerwise
