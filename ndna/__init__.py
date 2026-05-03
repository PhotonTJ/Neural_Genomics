"""
nDNA: Neural DNA metrics for transformer analysis.

A modular framework for calculating and visualizing geometric metrics
for transformer language models based on information geometry principles.
"""

__version__ = "0.1.0"

from .models import ModelHandler, MODEL_REGISTRY, detect_architecture
from .data import (
    DatasetHandler,
    DATASET_REGISTRY,
    DEFAULT_PROMPTS,
    PROMPT_SETS,
    get_prompts,
    make_causal_collate,
    make_belief_collate,
)
from .core import (
    # Calculators
    SpectralCalculator,
    ThermodynamicCalculator,
    BeliefCalculator,
    # Results
    SpectralResult,
    ThermoResult,
    BeliefResult,
    nDNAResult,
    nDNAResultMultiConcept,
    FullResults,
    # Geometry utilities
    sqrt_embed,
    fisher_rao_distance,
    discrete_curvature,
    free_memory,
    # nDNA combined metric
    compute_ndna,
    compute_ndna_multi_concept,
)
from .storage import (
    ResultSaver,
    ResultLoader,
)
from .visualization import (
    # Styles
    apply_style,
    get_color_palette,
    # 2D Plots
    plot_spectral_curvature,
    plot_thermodynamic_length,
    plot_belief_vector_field,
    plot_all_metrics,
    plot_master_panel,
    # 3D Plots
    plot_spectral_3d,
    plot_thermodynamic_3d,
    plot_belief_3d,
    plot_ndna_trajectory_3d,
    export_multi_model_html,
)

__all__ = [
    "__version__",
    # Models
    "ModelHandler",
    "MODEL_REGISTRY",
    "detect_architecture",
    # Data
    "DatasetHandler",
    "DATASET_REGISTRY",
    "DEFAULT_PROMPTS",
    "PROMPT_SETS",
    "get_prompts",
    "make_causal_collate",
    "make_belief_collate",
    # Core calculators
    "SpectralCalculator",
    "ThermodynamicCalculator",
    "BeliefCalculator",
    # Core results
    "SpectralResult",
    "ThermoResult",
    "BeliefResult",
    "nDNAResult",
    "nDNAResultMultiConcept",
    "FullResults",
    # Core geometry
    "sqrt_embed",
    "fisher_rao_distance",
    "discrete_curvature",
    "free_memory",
    # nDNA combined metric
    "compute_ndna",
    "compute_ndna_multi_concept",
    # Storage
    "ResultSaver",
    "ResultLoader",
    # Visualization - Styles
    "apply_style",
    "get_color_palette",
    # Visualization - 2D Plots
    "plot_spectral_curvature",
    "plot_thermodynamic_length",
    "plot_belief_vector_field",
    "plot_all_metrics",
    "plot_master_panel",
    # Visualization - 3D Plots
    "plot_spectral_3d",
    "plot_thermodynamic_3d",
    "plot_belief_3d",
    "plot_ndna_trajectory_3d",
    "export_multi_model_html",
]
