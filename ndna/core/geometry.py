"""
ndna.core.geometry

Shared geometry utilities for nDNA metrics.
Implements operations on the probability simplex and unit sphere.

All functions are designed to be memory-efficient with explicit
tensor management and support for mixed precision.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F


# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
EPS_DIST = 1e-12  # Floor for probabilities before sqrt
EPS_CURV = 1e-12  # Epsilon in curvature denominator
EPS_DIV = 1e-12   # General division epsilon


# -----------------------------------------------------------------------------
# Memory Management
# -----------------------------------------------------------------------------
def free_memory() -> None:
    """
    Free GPU memory by clearing the CUDA cache.
    
    Call this after deleting large tensors to ensure memory is released.
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def get_device() -> str:
    """Get the default device (cuda if available, else cpu)."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def get_amp_dtype() -> torch.dtype:
    """Get the appropriate AMP dtype for the current device."""
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


# -----------------------------------------------------------------------------
# Square-Root Embedding
# -----------------------------------------------------------------------------
@torch.no_grad()
def sqrt_embed(q: torch.Tensor, eps: float = EPS_DIST) -> torch.Tensor:
    """
    Map probability distribution to unit sphere via square-root embedding.
    
    u = sqrt(q) / ||sqrt(q)||
    
    This is the Fisher-Rao embedding that maps the probability simplex
    to the positive orthant of the unit sphere.
    
    Args:
        q: Probability distribution (V,) - should sum to 1
        eps: Floor for numerical stability
    
    Returns:
        Unit vector on V-dimensional sphere (V,)
    """
    q = torch.clamp(q, min=eps)
    q = q / q.sum()  # Ensure normalization
    u = torch.sqrt(q)
    u = u / (torch.linalg.norm(u) + 1e-30)
    return u


@torch.no_grad()
def sqrt_embed_batch(q: torch.Tensor, eps: float = EPS_DIST) -> torch.Tensor:
    """
    Batched square-root embedding.
    
    Args:
        q: Probability distributions (..., V)
        eps: Floor for numerical stability
    
    Returns:
        Unit vectors (..., V)
    """
    q = torch.clamp(q, min=eps)
    q = q / q.sum(dim=-1, keepdim=True)
    u = torch.sqrt(q)
    u = u / (torch.linalg.norm(u, dim=-1, keepdim=True) + 1e-30)
    return u


# -----------------------------------------------------------------------------
# Tangent Space Projection
# -----------------------------------------------------------------------------
@torch.no_grad()
def project_tangent(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    Project vector v onto tangent space at point u on the unit sphere.
    
    v_tangent = v - (u · v) * u
    
    Args:
        u: Point on unit sphere (V,)
        v: Vector to project (V,)
    
    Returns:
        Tangent vector at u (V,)
    """
    return v - torch.dot(u, v) * u


@torch.no_grad()
def project_tangent_batch(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    Batched tangent projection.
    
    Args:
        u: Points on unit sphere (..., V)
        v: Vectors to project (..., V)
    
    Returns:
        Tangent vectors (..., V)
    """
    # Compute dot product along last dimension
    uv = (u * v).sum(dim=-1, keepdim=True)
    return v - u * uv


# -----------------------------------------------------------------------------
# Discrete Curvature
# -----------------------------------------------------------------------------
@torch.no_grad()
def discrete_curvature(
    u_list: List[torch.Tensor],
    eps_curv: float = EPS_CURV,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute discrete curvature for interior points on the sphere.
    
    κ_ℓ = ||Δ²u_ℓ|| / ||Δu_ℓ||^(3/2)
    
    where:
        Δu_ℓ = project_tangent(u_ℓ, u_{ℓ+1} - u_ℓ)
        Δ²u_ℓ = project_tangent(u_ℓ, u_{ℓ+1} - 2u_ℓ + u_{ℓ-1})
    
    Args:
        u_list: List of sqrt-embeddings for each layer [u_0, u_1, ..., u_L]
                Each u_ℓ is a (V,) tensor on the unit sphere
        eps_curv: Epsilon for denominator stability
    
    Returns:
        curvatures: numpy array (L-1,) for interior layers ℓ = 1 to L-1
        speeds: numpy array (L,) for first differences ℓ = 0 to L-1
    """
    m = len(u_list)
    if m < 3:
        raise ValueError(f"Need at least 3 points for curvature, got {m}")

    # First differences: Δu_ℓ for ℓ = 0..m-2
    delta_u = []
    speeds = []
    
    for ell in range(m - 1):
        u = u_list[ell]
        v = u_list[ell + 1] - u
        du = project_tangent(u, v)
        delta_u.append(du)
        speeds.append(torch.linalg.norm(du).item())

    # Second differences at interior points (ℓ = 1..m-2)
    curvatures = []
    
    for ell in range(1, m - 1):
        u = u_list[ell]
        # Second difference: u_{ℓ+1} - 2u_ℓ + u_{ℓ-1}
        v2 = u_list[ell + 1] - 2 * u_list[ell] + u_list[ell - 1]
        d2u = project_tangent(u, v2)
        
        num = torch.linalg.norm(d2u)
        s = torch.linalg.norm(delta_u[ell])
        denom = (s * s + eps_curv) ** 1.5
        
        k = (num / denom).item()
        curvatures.append(k)

    return np.array(curvatures, dtype=np.float64), np.array(speeds, dtype=np.float64)


# -----------------------------------------------------------------------------
# Fisher-Rao Distance
# -----------------------------------------------------------------------------
@torch.no_grad()
def fisher_rao_distance(
    logp1: torch.Tensor,
    logp2: torch.Tensor,
) -> torch.Tensor:
    """
    Compute Fisher-Rao distance between two log-probability distributions.
    
    d_FR = 2 * arccos(BC) where BC = Σ √(p₁ᵢ × p₂ᵢ)
    
    Numerically stable computation via:
        BC = exp(logsumexp(0.5 * (logp1 + logp2)))
    
    Args:
        logp1: Log probabilities (..., V)
        logp2: Log probabilities (..., V)
    
    Returns:
        Fisher-Rao distance in radians (...,)
    """
    # Compute Bhattacharyya coefficient via logsumexp
    # BC = Σ sqrt(p1 * p2) = Σ exp(0.5 * (log p1 + log p2))
    s = 0.5 * (logp1 + logp2)
    log_bc = torch.logsumexp(s, dim=-1)
    bc = torch.exp(log_bc)
    
    # Clamp to [0, 1] for numerical stability
    bc = bc.clamp(0.0, 1.0)
    
    # FR distance = 2 * arccos(BC)
    return 2.0 * torch.acos(bc)


@torch.no_grad()
def fisher_rao_distance_from_sqrt(
    u1: torch.Tensor,
    u2: torch.Tensor,
) -> torch.Tensor:
    """
    Compute Fisher-Rao distance from sqrt embeddings.
    
    d_FR = 2 * arccos(<u1, u2>)
    
    Since u = sqrt(p), we have <u1, u2> = BC directly.
    
    Args:
        u1: Sqrt embedding (..., V)
        u2: Sqrt embedding (..., V)
    
    Returns:
        Fisher-Rao distance in radians (...,)
    """
    inner = (u1 * u2).sum(dim=-1)
    inner = inner.clamp(-1.0, 1.0)
    return 2.0 * torch.acos(inner)


# -----------------------------------------------------------------------------
# Probability Utilities
# -----------------------------------------------------------------------------
@torch.no_grad()
def safe_log_softmax(
    logits: torch.Tensor,
    dim: int = -1,
) -> torch.Tensor:
    """
    Compute log softmax with float32 precision for stability.
    
    Args:
        logits: Input logits (any shape)
        dim: Dimension to apply softmax
    
    Returns:
        Log probabilities (same shape)
    """
    return F.log_softmax(logits.float(), dim=dim)


@torch.no_grad()
def safe_softmax(
    logits: torch.Tensor,
    dim: int = -1,
    eps: float = EPS_DIST,
) -> torch.Tensor:
    """
    Compute softmax with float32 precision and clamping.
    
    Args:
        logits: Input logits (any shape)
        dim: Dimension to apply softmax
        eps: Minimum probability value
    
    Returns:
        Probabilities (same shape), clamped to [eps, 1]
    """
    q = F.softmax(logits.float(), dim=dim)
    return torch.clamp(q, min=eps)


# -----------------------------------------------------------------------------
# Belief Vector Utilities
# -----------------------------------------------------------------------------
@torch.no_grad()
def compute_belief_gradient(
    q: torch.Tensor,
    targets: torch.Tensor,
    tau: float = 1.0,
) -> torch.Tensor:
    """
    Compute gradient of log-likelihood w.r.t. logits.
    
    g = (1/τ)(e_y - q)
    
    where e_y is one-hot at target index y.
    
    Args:
        q: Probability distribution (..., V)
        targets: Target indices (...,)
        tau: Temperature
    
    Returns:
        Gradient (..., V)
    """
    # g = -q/tau initially
    g = -q / tau
    
    # Add 1/tau at target positions
    # Use scatter_add for efficiency
    batch_shape = q.shape[:-1]
    V = q.shape[-1]
    
    # Flatten for scatter
    g_flat = g.view(-1, V)
    targets_flat = targets.view(-1, 1)
    
    add_values = torch.full_like(targets_flat, 1.0 / tau, dtype=g.dtype)
    g_flat.scatter_add_(-1, targets_flat, add_values)
    
    return g_flat.view(*batch_shape, V)


@torch.no_grad()
def compute_tangent_vector(
    q: torch.Tensor,
    u: torch.Tensor,
    g: torch.Tensor,
    tau: float = 1.0,
) -> torch.Tensor:
    """
    Compute belief tangent vector from gradient.
    
    t = (1/2τ)(u*g - ⟨q,g⟩*u)
    
    Args:
        q: Probability distribution (..., V)
        u: Sqrt embedding (..., V)
        g: Gradient from compute_belief_gradient (..., V)
        tau: Temperature
    
    Returns:
        Tangent vector (..., V)
    """
    ug = u * g
    qg = (q * g).sum(dim=-1, keepdim=True)
    t = (ug - qg * u) / (2.0 * tau)
    return t


# -----------------------------------------------------------------------------
# nDNA Combined Metric
# -----------------------------------------------------------------------------
def compute_ndna(
    kappa: np.ndarray,
    fr_steps: np.ndarray,
    belief_norms: np.ndarray,
    l_min: int = 2,
    concept_name: str = "default",
) -> "nDNAResult":
    """
    Compute nDNA combined metric for a single concept.
    
    nDNA_pred(c) = Σ_{ℓ=l_min}^{L-1} κ_ℓ · Δ_ℓ · ||v_ℓ(c)||
    
    Theory:
        - Let model have L transformer blocks (ℓ = 0..L-1).
        - κ_ℓ (curvature) and Δ_ℓ (FR step) are defined for ℓ = 1..L-1 (interior layers)
        - v_ℓ(c) (belief norm) is defined for ℓ = 0..L-1 (all layers)
        - We sum from ℓ = l_min..L-1 (drop shallow edge effects)
    
    Index mapping:
        kappa[idx]    corresponds to ℓ = idx + 1
        fr_steps[idx] corresponds to ℓ = idx + 1
        belief_norms[ℓ] uses same ℓ as block index
    
    Args:
        kappa: Spectral curvature for interior layers, shape (L-1,)
        fr_steps: FR step lengths between layers, shape (L-1,)
        belief_norms: Belief vector norms per layer, shape (L,)
        l_min: Minimum layer index for sum (default 2 to skip shallow layers)
        concept_name: Name of the concept/dataset
    
    Returns:
        nDNAResult with scalar value and per-layer contributions
    
    Example:
        >>> # After computing all three metrics
        >>> result = compute_ndna(
        ...     kappa=spectral_result.curvatures,
        ...     fr_steps=thermo_result.step_lengths,
        ...     belief_norms=belief_result.belief_norms,
        ... )
        >>> print(f"nDNA score: {result.scalar:.6f}")
    """
    # Import here to avoid circular import
    from .results import nDNAResult
    
    kappa = np.asarray(kappa, dtype=np.float64)
    fr_steps = np.asarray(fr_steps, dtype=np.float64)
    belief_norms = np.asarray(belief_norms, dtype=np.float64)
    
    # Validate shapes
    if kappa.shape != fr_steps.shape:
        raise ValueError(
            f"kappa and fr_steps must have same shape; got {kappa.shape} vs {fr_steps.shape}"
        )
    
    L_minus_1 = kappa.shape[0]
    L = L_minus_1 + 1  # number of layers
    
    if belief_norms.shape[0] != L:
        raise ValueError(
            f"belief_norms has length {belief_norms.shape[0]}, expected {L} "
            f"(one per layer, matching kappa/fr_steps length + 1)"
        )
    
    # Compute per-layer contributions
    # contrib[ℓ] = κ_ℓ * Δ_ℓ * ||v_ℓ||
    contrib = np.zeros(L, dtype=np.float64)
    
    for ell in range(l_min, L):  # ℓ = l_min..L-1
        idx = ell - 1  # index into kappa, fr_steps
        if idx >= 0 and idx < L_minus_1:
            contrib[ell] = kappa[idx] * fr_steps[idx] * belief_norms[ell]
    
    # Sum contributions
    scalar = float(contrib.sum())
    
    # Layer indices (1-indexed)
    layer_indices = np.arange(1, L + 1)
    
    return nDNAResult(
        scalar=scalar,
        layerwise=contrib,
        layer_indices=layer_indices,
        concept_name=concept_name,
        l_min=l_min,
        kappa=kappa,
        fr_steps=fr_steps,
        belief_norms=belief_norms,
    )


def compute_ndna_multi_concept(
    kappa: np.ndarray,
    fr_steps: np.ndarray,
    belief_norms_by_concept: dict,
    l_min: int = 2,
) -> "nDNAResultMultiConcept":
    """
    Compute nDNA combined metric for multiple concepts.
    
    This is useful when computing nDNA for different datasets or concept
    categories while sharing the same spectral curvature and thermodynamic
    length values.
    
    Args:
        kappa: Spectral curvature for interior layers, shape (L-1,)
        fr_steps: FR step lengths between layers, shape (L-1,)
        belief_norms_by_concept: Dict mapping concept name to belief norms array (L,)
        l_min: Minimum layer index for sum
    
    Returns:
        nDNAResultMultiConcept with per-concept scalars and layerwise contributions
    
    Example:
        >>> result = compute_ndna_multi_concept(
        ...     kappa=spectral.curvatures,
        ...     fr_steps=thermo.step_lengths,
        ...     belief_norms_by_concept={
        ...         "squad": belief_squad.belief_norms,
        ...         "imdb": belief_imdb.belief_norms,
        ...     },
        ... )
        >>> print(result.ranked_concepts())
    """
    # Import here to avoid circular import
    from .results import nDNAResultMultiConcept
    
    kappa = np.asarray(kappa, dtype=np.float64)
    fr_steps = np.asarray(fr_steps, dtype=np.float64)
    
    if kappa.shape != fr_steps.shape:
        raise ValueError(
            f"kappa and fr_steps must have same shape; got {kappa.shape} vs {fr_steps.shape}"
        )
    
    L_minus_1 = kappa.shape[0]
    L = L_minus_1 + 1
    
    scalars = {}
    layerwise = {}
    
    for name, v_norms in belief_norms_by_concept.items():
        v_arr = np.asarray(v_norms, dtype=np.float64)
        
        if v_arr.shape[0] != L:
            raise ValueError(
                f"belief_norms for concept '{name}' has length {v_arr.shape[0]}, expected {L}"
            )
        
        # Compute contributions
        contrib = np.zeros(L, dtype=np.float64)
        
        for ell in range(l_min, L):
            idx = ell - 1
            if idx >= 0 and idx < L_minus_1:
                contrib[ell] = kappa[idx] * fr_steps[idx] * v_arr[ell]
        
        layerwise[name] = contrib
        scalars[name] = float(contrib.sum())
    
    layer_indices = np.arange(1, L + 1)
    
    return nDNAResultMultiConcept(
        scalars=scalars,
        layerwise=layerwise,
        layer_indices=layer_indices,
        l_min=l_min,
        kappa=kappa,
        fr_steps=fr_steps,
    )

