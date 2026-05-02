"""
ndna.core.results

Result dataclasses for nDNA metric calculations.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


# -----------------------------------------------------------------------------
# Spectral Curvature Results
# -----------------------------------------------------------------------------
@dataclass
class SpectralResult:
    """
    Result from spectral curvature calculation for a single text.
    
    Attributes:
        curvatures: Discrete curvature values for interior layers (L-2,)
        speeds: First-difference speeds between layers (L-1,)
        layer_indices: Layer indices for curvatures [1, 2, ..., L-2]
        text_preview: First 100 characters of input text
        label: Optional label for the text/prompt
        mean_curvature: Mean of curvature values
        max_curvature: Maximum curvature value
        min_curvature: Minimum curvature value
        max_layer: Layer index with maximum curvature
        min_layer: Layer index with minimum curvature
    """
    curvatures: np.ndarray
    speeds: np.ndarray
    layer_indices: np.ndarray
    text_preview: str
    label: Optional[str] = None
    mean_curvature: float = 0.0
    max_curvature: float = 0.0
    min_curvature: float = 0.0
    max_layer: int = 0
    min_layer: int = 0
    
    def __post_init__(self):
        """Compute statistics if not provided."""
        if len(self.curvatures) > 0:
            if self.mean_curvature == 0.0:
                self.mean_curvature = float(np.mean(self.curvatures))
            if self.max_curvature == 0.0:
                self.max_curvature = float(np.max(self.curvatures))
            if self.min_curvature == 0.0:
                self.min_curvature = float(np.min(self.curvatures))
            if self.max_layer == 0:
                self.max_layer = int(np.argmax(self.curvatures)) + 1
            if self.min_layer == 0:
                self.min_layer = int(np.argmin(self.curvatures)) + 1


@dataclass
class AggregatedSpectralResult:
    """
    Aggregated spectral curvature results over multiple texts.
    
    Attributes:
        mean_curvatures: Mean curvature at each layer (L-2,)
        std_curvatures: Std of curvatures at each layer (L-2,)
        all_curvatures: All curvature arrays (N, L-2)
        layer_indices: Layer indices [1, 2, ..., L-2]
        num_texts: Number of texts processed
        individual_results: List of individual SpectralResult objects
    """
    mean_curvatures: np.ndarray
    std_curvatures: np.ndarray
    all_curvatures: np.ndarray
    layer_indices: np.ndarray
    num_texts: int
    individual_results: List[SpectralResult] = field(default_factory=list)
    
    @classmethod
    def from_results(cls, results: List[SpectralResult]) -> "AggregatedSpectralResult":
        """Create aggregated result from list of individual results."""
        if not results:
            raise ValueError("Cannot aggregate empty results list")
        
        all_curvatures = np.array([r.curvatures for r in results])
        mean_curvatures = np.mean(all_curvatures, axis=0)
        std_curvatures = np.std(all_curvatures, axis=0)
        
        return cls(
            mean_curvatures=mean_curvatures,
            std_curvatures=std_curvatures,
            all_curvatures=all_curvatures,
            layer_indices=results[0].layer_indices,
            num_texts=len(results),
            individual_results=results,
        )


# -----------------------------------------------------------------------------
# Thermodynamic Length Results
# -----------------------------------------------------------------------------
@dataclass
class ThermoResult:
    """
    Result from thermodynamic length calculation.
    
    Attributes:
        step_lengths: Mean FR distance per inter-layer transition (L-1,)
        step_indices: Step indices [1, 2, ..., L-1]
        total_length: Sum of all step lengths (radians)
        num_samples_processed: Number of samples used in calculation
        num_tokens_processed: Number of valid tokens processed
    """
    step_lengths: np.ndarray
    step_indices: np.ndarray
    total_length: float
    num_samples_processed: int
    num_tokens_processed: int = 0
    
    def __post_init__(self):
        """Compute total length if not provided."""
        if self.total_length == 0.0 and len(self.step_lengths) > 0:
            self.total_length = float(np.sum(self.step_lengths))


@dataclass
class ThermoResultPerSample:
    """
    Per-sample thermodynamic length results for 3D visualization.
    
    Attributes:
        per_sample_lengths: FR lengths per sample per step (N, L-1)
        mean_lengths: Mean length at each step (L-1,)
        step_indices: Step indices [1, 2, ..., L-1]
        num_samples: Number of samples
    """
    per_sample_lengths: np.ndarray
    mean_lengths: np.ndarray
    step_indices: np.ndarray
    num_samples: int
    
    @classmethod
    def from_sample_data(
        cls,
        per_sample_lengths: np.ndarray,
        step_indices: np.ndarray,
    ) -> "ThermoResultPerSample":
        """Create result from per-sample data."""
        return cls(
            per_sample_lengths=per_sample_lengths,
            mean_lengths=np.mean(per_sample_lengths, axis=0),
            step_indices=step_indices,
            num_samples=per_sample_lengths.shape[0],
        )


# -----------------------------------------------------------------------------
# Belief Vector Field Results
# -----------------------------------------------------------------------------
@dataclass
class BeliefResult:
    """
    Result from belief vector field calculation.
    
    Attributes:
        belief_norms: Norm of belief vector at each layer (L,)
        layer_indices: Layer indices [1, 2, ..., L]
        mean_norm: Mean of belief norms
        max_norm: Maximum belief norm
        min_norm: Minimum belief norm
        max_layer: Layer with maximum norm
        min_layer: Layer with minimum norm
        num_samples_processed: Number of samples used
        num_tokens_processed: Number of valid tokens processed
        fr_norm: Whether FR norm (2*||v||) was used
    """
    belief_norms: np.ndarray
    layer_indices: np.ndarray
    mean_norm: float = 0.0
    max_norm: float = 0.0
    min_norm: float = 0.0
    max_layer: int = 0
    min_layer: int = 0
    num_samples_processed: int = 0
    num_tokens_processed: int = 0
    fr_norm: bool = True
    
    def __post_init__(self):
        """Compute statistics if not provided."""
        if len(self.belief_norms) > 0:
            if self.mean_norm == 0.0:
                self.mean_norm = float(np.mean(self.belief_norms))
            if self.max_norm == 0.0:
                self.max_norm = float(np.max(self.belief_norms))
            if self.min_norm == 0.0:
                self.min_norm = float(np.min(self.belief_norms))
            if self.max_layer == 0:
                self.max_layer = int(np.argmax(self.belief_norms)) + 1
            if self.min_layer == 0:
                self.min_layer = int(np.argmin(self.belief_norms)) + 1


@dataclass
class BeliefResultPerSample:
    """
    Per-sample belief vector field results for 3D visualization.
    
    Attributes:
        per_sample_norms: Belief norms per sample per layer (N, L)
        mean_norms: Mean norm at each layer (L,)
        layer_indices: Layer indices [1, 2, ..., L]
        num_samples: Number of samples
    """
    per_sample_norms: np.ndarray
    mean_norms: np.ndarray
    layer_indices: np.ndarray
    num_samples: int
    
    @classmethod
    def from_sample_data(
        cls,
        per_sample_norms: np.ndarray,
        layer_indices: np.ndarray,
    ) -> "BeliefResultPerSample":
        """Create result from per-sample data."""
        return cls(
            per_sample_norms=per_sample_norms,
            mean_norms=np.mean(per_sample_norms, axis=0),
            layer_indices=layer_indices,
            num_samples=per_sample_norms.shape[0],
        )


# -----------------------------------------------------------------------------
# nDNA Combined Metric Results
# -----------------------------------------------------------------------------
@dataclass
class nDNAResult:
    """
    Result from nDNA combined metric calculation.
    
    nDNA_pred(c) = Σ_{ℓ=l_min}^{L-1} κ_ℓ · Δ_ℓ · ||v_ℓ(c)||
    
    Combines spectral curvature (κ), thermodynamic length (Δ), 
    and belief vector field (||v||) into a unified metric.
    
    Attributes:
        scalar: Total nDNA score (scalar sum)
        layerwise: Per-layer contributions (L,) - zeros outside [l_min, L-1]
        layer_indices: Layer indices [1, 2, ..., L]
        concept_name: Name of the concept/dataset
        l_min: Minimum layer index used in sum (default 2)
        
        # Component inputs (for reference)
        kappa: Spectral curvature values used (L-1,)
        fr_steps: FR step lengths used (L-1,)
        belief_norms: Belief norms used (L,)
    """
    scalar: float
    layerwise: np.ndarray
    layer_indices: np.ndarray
    concept_name: str = "default"
    l_min: int = 2
    
    # Component inputs
    kappa: Optional[np.ndarray] = None
    fr_steps: Optional[np.ndarray] = None
    belief_norms: Optional[np.ndarray] = None
    
    @property
    def peak_layer(self) -> int:
        """Layer with maximum contribution."""
        if len(self.layerwise) == 0:
            return 0
        return int(np.argmax(self.layerwise)) + 1
    
    @property
    def peak_contribution(self) -> float:
        """Maximum per-layer contribution."""
        if len(self.layerwise) == 0:
            return 0.0
        return float(np.max(self.layerwise))


@dataclass
class nDNAResultMultiConcept:
    """
    nDNA results for multiple concepts/datasets.
    
    Attributes:
        scalars: Dict mapping concept name to scalar nDNA value
        layerwise: Dict mapping concept name to layerwise contributions
        layer_indices: Layer indices [1, 2, ..., L]
        l_min: Minimum layer index used
        
        # Component inputs
        kappa: Spectral curvature used (shared across concepts)
        fr_steps: FR step lengths used (shared across concepts)
    """
    scalars: Dict[str, float]
    layerwise: Dict[str, np.ndarray]
    layer_indices: np.ndarray
    l_min: int = 2
    kappa: Optional[np.ndarray] = None
    fr_steps: Optional[np.ndarray] = None
    
    def get_result(self, concept: str) -> nDNAResult:
        """Get nDNAResult for a specific concept."""
        if concept not in self.scalars:
            raise KeyError(f"Concept '{concept}' not found. Available: {list(self.scalars.keys())}")
        
        return nDNAResult(
            scalar=self.scalars[concept],
            layerwise=self.layerwise[concept],
            layer_indices=self.layer_indices,
            concept_name=concept,
            l_min=self.l_min,
            kappa=self.kappa,
            fr_steps=self.fr_steps,
        )
    
    @property
    def concepts(self) -> List[str]:
        """List of concept names."""
        return list(self.scalars.keys())
    
    def ranked_concepts(self, descending: bool = True) -> List[tuple]:
        """Return concepts ranked by nDNA score."""
        items = list(self.scalars.items())
        return sorted(items, key=lambda x: x[1], reverse=descending)


# -----------------------------------------------------------------------------
# Combined Results
# -----------------------------------------------------------------------------
@dataclass
class FullResults:
    """
    Container for all metric results.
    
    Attributes:
        model_name: Name of the model used
        dataset_name: Name of the dataset used
        spectral: Spectral curvature results
        thermodynamic: Thermodynamic length results
        belief: Belief vector field results
        thermo_per_sample: Per-sample thermodynamic results (optional)
        belief_per_sample: Per-sample belief results (optional)
        config: Configuration used for calculations
    """
    model_name: str
    dataset_name: str
    spectral: Optional[SpectralResult] = None
    spectral_by_prompt: Optional[Dict[str, SpectralResult]] = None
    thermodynamic: Optional[ThermoResult] = None
    belief: Optional[BeliefResult] = None
    thermo_per_sample: Optional[ThermoResultPerSample] = None
    belief_per_sample: Optional[BeliefResultPerSample] = None
    config: Optional[Dict] = None

