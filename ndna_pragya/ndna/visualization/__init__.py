"""
ndna.visualization

Plotting utilities for nDNA metric visualization.

Provides:
- 2D matplotlib plots for individual metrics
- 3D interactive plotly plots for multi-dimensional visualization
- Styling utilities for consistent aesthetics
"""

from .styles import (
    apply_style,
    get_color_palette,
    COLORS,
    FIGSIZE_SINGLE,
    FIGSIZE_PANEL,
    FIGSIZE_WIDE,
)

from .plots_2d import (
    plot_spectral_curvature,
    plot_thermodynamic_length,
    plot_belief_vector_field,
    plot_all_metrics,
    plot_master_panel,
)

from .plots_3d import (
    plot_spectral_3d,
    plot_thermodynamic_3d,
    plot_belief_3d,
    plot_ndna_trajectory_3d,
    export_multi_model_html,
)

__all__ = [
    # Styles
    "apply_style",
    "get_color_palette",
    "COLORS",
    "FIGSIZE_SINGLE",
    "FIGSIZE_PANEL",
    "FIGSIZE_WIDE",
    # 2D Plots
    "plot_spectral_curvature",
    "plot_thermodynamic_length",
    "plot_belief_vector_field",
    "plot_all_metrics",
    "plot_master_panel",
    # 3D Plots
    "plot_spectral_3d",
    "plot_thermodynamic_3d",
    "plot_belief_3d",
    "plot_ndna_trajectory_3d",
    "export_multi_model_html",
]

