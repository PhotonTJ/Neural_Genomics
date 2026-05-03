"""
ndna.core

Core metric calculators for nDNA analysis.

Provides memory-efficient implementations of:
- Spectral Curvature: Geometric curvature of probability manifold
- Thermodynamic Length: Fisher-Rao distance between layers
- Belief Vector Field: Tangent vectors representing belief changes
- nDNA: Combined metric (κ · Δ · ||v||)
"""

from .geometry import (
    # Constants
    EPS_DIST,
    EPS_CURV,
    EPS_DIV,
    # Memory management
    free_memory,
    get_device,
    get_amp_dtype,
    # Sqrt embedding
    sqrt_embed,
    sqrt_embed_batch,
    # Tangent projection
    project_tangent,
    project_tangent_batch,
    # Curvature
    discrete_curvature,
    # Fisher-Rao distance
    fisher_rao_distance,
    fisher_rao_distance_from_sqrt,
    # Probability utilities
    safe_log_softmax,
    safe_softmax,
    # Belief utilities
    compute_belief_gradient,
    compute_tangent_vector,
    # nDNA combined metric
    compute_ndna,
    compute_ndna_multi_concept,
)

from .results import (
    SpectralResult,
    AggregatedSpectralResult,
    ThermoResult,
    ThermoResultPerSample,
    BeliefResult,
    BeliefResultPerSample,
    nDNAResult,
    nDNAResultMultiConcept,
    FullResults,
)

from .spectral import SpectralCalculator
from .thermodynamic import ThermodynamicCalculator
from .belief import BeliefCalculator

__all__ = [
    # Geometry constants
    "EPS_DIST",
    "EPS_CURV",
    "EPS_DIV",
    # Memory management
    "free_memory",
    "get_device",
    "get_amp_dtype",
    # Geometry functions
    "sqrt_embed",
    "sqrt_embed_batch",
    "project_tangent",
    "project_tangent_batch",
    "discrete_curvature",
    "fisher_rao_distance",
    "fisher_rao_distance_from_sqrt",
    "safe_log_softmax",
    "safe_softmax",
    "compute_belief_gradient",
    "compute_tangent_vector",
    # nDNA combined metric
    "compute_ndna",
    "compute_ndna_multi_concept",
    # Results
    "SpectralResult",
    "AggregatedSpectralResult",
    "ThermoResult",
    "ThermoResultPerSample",
    "BeliefResult",
    "BeliefResultPerSample",
    "nDNAResult",
    "nDNAResultMultiConcept",
    "FullResults",
    # Calculators
    "SpectralCalculator",
    "ThermodynamicCalculator",
    "BeliefCalculator",
]

