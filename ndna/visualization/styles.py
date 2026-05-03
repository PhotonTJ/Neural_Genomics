"""
ndna.visualization.styles

Styling utilities and constants for nDNA visualizations.
"""

from typing import List, Optional

import matplotlib.pyplot as plt
import matplotlib as mpl


# -----------------------------------------------------------------------------
# Figure Sizes
# -----------------------------------------------------------------------------
FIGSIZE_SINGLE = (10, 5)      # Single metric plot
FIGSIZE_WIDE = (12, 6)        # Wide single plot
FIGSIZE_PANEL = (14, 10)      # Multi-panel (2x2)
FIGSIZE_3D = (10, 8)          # 3D plot

# -----------------------------------------------------------------------------
# Color Palettes
# -----------------------------------------------------------------------------

# Primary palette - vibrant, distinct colors
COLORS = {
    "primary": "#2E86AB",      # Steel blue
    "secondary": "#A23B72",    # Raspberry
    "tertiary": "#F18F01",     # Orange
    "quaternary": "#C73E1D",   # Vermillion
    "quinary": "#3B1F2B",      # Dark purple
    
    # For curvature/metrics
    "curvature": "#2E86AB",
    "fr_step": "#F18F01",
    "belief": "#A23B72",
    "ndna": "#C73E1D",
    
    # Grid and background
    "grid": "#E0E0E0",
    "background": "#FAFAFA",
}

# Extended palette for multiple concepts
CONCEPT_COLORS = [
    "#2E86AB",  # Steel blue
    "#A23B72",  # Raspberry
    "#F18F01",  # Orange
    "#C73E1D",  # Vermillion
    "#3B1F2B",  # Dark purple
    "#1B4332",  # Forest green
    "#7209B7",  # Purple
    "#F72585",  # Pink
    "#4CC9F0",  # Cyan
    "#90BE6D",  # Light green
]

# Plotly color scale for 3D surfaces
PLOTLY_COLORSCALE = "Viridis"


# -----------------------------------------------------------------------------
# Style Application
# -----------------------------------------------------------------------------

def apply_style(style: str = "ndna") -> None:
    """
    Apply a consistent matplotlib style.
    
    Args:
        style: Style name. Options:
            - "ndna": Custom nDNA style (default)
            - "seaborn": Seaborn whitegrid
            - "minimal": Minimal clean style
    """
    if style == "ndna":
        _apply_ndna_style()
    elif style == "seaborn":
        try:
            plt.style.use("seaborn-v0_8-whitegrid")
        except OSError:
            plt.style.use("seaborn-whitegrid")
    elif style == "minimal":
        _apply_minimal_style()
    else:
        # Try to use as a matplotlib style
        plt.style.use(style)


def _apply_ndna_style() -> None:
    """Apply the custom nDNA style."""
    plt.rcParams.update({
        # Figure
        "figure.facecolor": COLORS["background"],
        "figure.edgecolor": "none",
        "figure.dpi": 100,
        
        # Axes
        "axes.facecolor": "white",
        "axes.edgecolor": "#333333",
        "axes.linewidth": 1.0,
        "axes.grid": True,
        "axes.grid.axis": "both",
        "axes.axisbelow": True,
        "axes.labelsize": 11,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelweight": "normal",
        "axes.spines.top": False,
        "axes.spines.right": False,
        
        # Grid
        "grid.color": COLORS["grid"],
        "grid.linewidth": 0.8,
        "grid.alpha": 0.7,
        
        # Lines
        "lines.linewidth": 2.0,
        "lines.markersize": 6,
        
        # Ticks
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "xtick.direction": "out",
        "ytick.direction": "out",
        
        # Legend
        "legend.frameon": True,
        "legend.framealpha": 0.9,
        "legend.facecolor": "white",
        "legend.edgecolor": "#CCCCCC",
        "legend.fontsize": 10,
        
        # Font
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial", "sans-serif"],
        "font.size": 11,
        
        # Saving
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
        "savefig.edgecolor": "none",
    })


def _apply_minimal_style() -> None:
    """Apply a minimal, clean style."""
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#333333",
        "axes.linewidth": 0.8,
        "axes.grid": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "lines.linewidth": 1.5,
        "lines.markersize": 5,
        "font.size": 10,
    })


def get_color_palette(n: int, palette: str = "concept") -> List[str]:
    """
    Get a list of colors for plotting.
    
    Args:
        n: Number of colors needed
        palette: Palette type. Options:
            - "concept": CONCEPT_COLORS (default)
            - "sequential": Blues/viridis gradient
            - "metric": Metric-specific colors
    
    Returns:
        List of color hex codes
    """
    if palette == "concept":
        # Cycle through concept colors
        return [CONCEPT_COLORS[i % len(CONCEPT_COLORS)] for i in range(n)]
    
    elif palette == "sequential":
        # Generate sequential colors using a colormap
        cmap = mpl.colormaps.get_cmap("viridis")
        return [mpl.colors.rgb2hex(cmap(i / max(1, n - 1))) for i in range(n)]
    
    elif palette == "metric":
        metric_colors = [
            COLORS["curvature"],
            COLORS["fr_step"],
            COLORS["belief"],
            COLORS["ndna"],
        ]
        return [metric_colors[i % len(metric_colors)] for i in range(n)]
    
    else:
        return get_color_palette(n, "concept")


def format_layer_label(layer_idx: int, prefix: str = "ℓ") -> str:
    """
    Format a layer index for display.
    
    Args:
        layer_idx: Layer index (1-based)
        prefix: Prefix character (default: ℓ)
    
    Returns:
        Formatted string like "ℓ=5"
    """
    return f"{prefix}={layer_idx}"


def format_metric_title(
    metric_name: str,
    model_name: Optional[str] = None,
    subtitle: Optional[str] = None
) -> str:
    """
    Format a plot title for a metric.
    
    Args:
        metric_name: Name of the metric
        model_name: Optional model name to include
        subtitle: Optional subtitle
    
    Returns:
        Formatted title string
    """
    title = metric_name
    if model_name:
        title = f"{title}\n{model_name}"
    if subtitle:
        title = f"{title}\n{subtitle}"
    return title

