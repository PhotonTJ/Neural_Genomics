"""
ndna.visualization.plots_2d

2D matplotlib plots for nDNA metrics.

Adapted from ndna_lib/plots.py with enhancements for the new result dataclasses.
"""

from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from ..core.results import (
    SpectralResult,
    AggregatedSpectralResult,
    ThermoResult,
    BeliefResult,
    nDNAResult,
)
from .styles import (
    apply_style,
    get_color_palette,
    COLORS,
    FIGSIZE_SINGLE,
    FIGSIZE_WIDE,
    FIGSIZE_PANEL,
    format_metric_title,
)


def plot_spectral_curvature(
    result: Union[SpectralResult, AggregatedSpectralResult, Dict[str, SpectralResult]],
    save_path: Optional[str] = None,
    log_scale: bool = False,
    model_name: str = "",
    show: bool = True,
    overlay: bool = True,
) -> Figure:
    """
    Plot spectral curvature κ_ℓ across layers.
    
    Args:
        result: SpectralResult, AggregatedSpectralResult, or dict of results by label
        save_path: Optional path to save the figure
        log_scale: Use log scale for y-axis
        model_name: Model name for title
        show: Whether to display the plot
        overlay: If multiple results, overlay on same axes
    
    Returns:
        matplotlib Figure
    """
    apply_style()
    
    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
    
    if isinstance(result, dict):
        # Multiple prompts/texts
        colors = get_color_palette(len(result))
        for (label, res), color in zip(result.items(), colors):
            curvatures = res.curvatures
            layers = res.layer_indices
            ax.plot(layers, curvatures, marker="o", label=label, color=color)
        ax.legend()
    elif isinstance(result, AggregatedSpectralResult):
        # Aggregated results with mean ± std
        layers = result.layer_indices
        mean = result.mean_curvatures
        std = result.std_curvatures
        
        ax.plot(layers, mean, marker="o", color=COLORS["curvature"], label="Mean")
        ax.fill_between(layers, mean - std, mean + std, alpha=0.3, color=COLORS["curvature"])
        ax.legend()
    else:
        # Single result
        curvatures = result.curvatures
        layers = result.layer_indices
        ax.plot(layers, curvatures, marker="o", color=COLORS["curvature"])
    
    title = format_metric_title("Spectral Curvature κ_ℓ", model_name)
    ax.set_title(title)
    ax.set_xlabel("Interior layer index ℓ")
    ax.set_ylabel("κ_ℓ")
    
    if log_scale:
        ax.set_yscale("log")
    
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    
    if show:
        plt.show()
    else:
        plt.close(fig)
    
    return fig


def plot_thermodynamic_length(
    result: ThermoResult,
    save_path: Optional[str] = None,
    log_scale: bool = False,
    model_name: str = "",
    show: bool = True,
) -> Figure:
    """
    Plot Fisher-Rao thermodynamic length Δ_ℓ across layer transitions.
    
    Args:
        result: ThermoResult with step lengths
        save_path: Optional path to save the figure
        log_scale: Use log scale for y-axis
        model_name: Model name for title
        show: Whether to display the plot
    
    Returns:
        matplotlib Figure
    """
    apply_style()
    
    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
    
    step_lengths = result.step_lengths
    step_indices = result.step_indices
    
    ax.plot(step_indices, step_lengths, marker="o", color=COLORS["fr_step"])
    
    subtitle = f"Total length: {result.total_length:.4f} rad"
    title = format_metric_title("Fisher-Rao Thermodynamic Length Δ_ℓ", model_name, subtitle)
    ax.set_title(title)
    ax.set_xlabel("Inter-layer step (ℓ → ℓ+1)")
    ax.set_ylabel("Δ_ℓ (radians)")
    
    if log_scale:
        ax.set_yscale("log")
    
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    
    if show:
        plt.show()
    else:
        plt.close(fig)
    
    return fig


def plot_belief_vector_field(
    result: Union[BeliefResult, Dict[str, BeliefResult]],
    save_path: Optional[str] = None,
    log_scale: bool = False,
    model_name: str = "",
    show: bool = True,
    fr_norm: bool = True,
) -> Figure:
    """
    Plot belief vector field magnitudes ||v_ℓ|| across layers.
    
    Args:
        result: BeliefResult or dict of results by concept
        save_path: Optional path to save the figure
        log_scale: Use log scale for y-axis
        model_name: Model name for title
        show: Whether to display the plot
        fr_norm: Label as FR norm (2||v||) vs Euclidean
    
    Returns:
        matplotlib Figure
    """
    apply_style()
    
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    
    if isinstance(result, dict):
        # Multiple concepts
        colors = get_color_palette(len(result))
        for (label, res), color in zip(result.items(), colors):
            belief_norms = res.belief_norms
            layers = res.layer_indices
            ax.plot(layers, belief_norms, marker="o", label=label, color=color)
        ax.legend()
    else:
        # Single result
        belief_norms = result.belief_norms
        layers = result.layer_indices
        ax.plot(layers, belief_norms, marker="o", color=COLORS["belief"])
    
    ylabel = "||v_ℓ||_FR" if fr_norm else "||v_ℓ||"
    title = format_metric_title(f"Belief Vector Field Magnitude {ylabel}", model_name)
    ax.set_title(title)
    ax.set_xlabel("Layer index ℓ")
    ax.set_ylabel(ylabel)
    
    if log_scale:
        ax.set_yscale("log")
    
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    
    if show:
        plt.show()
    else:
        plt.close(fig)
    
    return fig


def plot_ndna_layerwise(
    result: Union[nDNAResult, Dict[str, nDNAResult]],
    save_path: Optional[str] = None,
    log_scale: bool = False,
    model_name: str = "",
    show: bool = True,
) -> Figure:
    """
    Plot nDNA layerwise contributions.
    
    Args:
        result: nDNAResult or dict of results by concept
        save_path: Optional path to save the figure
        log_scale: Use log scale for y-axis
        model_name: Model name for title
        show: Whether to display the plot
    
    Returns:
        matplotlib Figure
    """
    apply_style()
    
    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
    
    if isinstance(result, dict):
        colors = get_color_palette(len(result))
        for (label, res), color in zip(result.items(), colors):
            layers = res.layer_indices
            contrib = res.layerwise
            ax.plot(layers, contrib, marker="o", label=f"{label} (Σ={res.scalar:.4f})", color=color)
        ax.legend()
    else:
        layers = result.layer_indices
        contrib = result.layerwise
        ax.plot(layers, contrib, marker="o", color=COLORS["ndna"])
        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    
    title = format_metric_title("nDNA Layerwise Contributions", model_name)
    ax.set_title(title)
    ax.set_xlabel("Layer index ℓ")
    ax.set_ylabel("κ_ℓ · Δ_ℓ · ||v_ℓ||")
    
    if log_scale:
        ax.set_yscale("log")
    
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    
    if show:
        plt.show()
    else:
        plt.close(fig)
    
    return fig


def plot_all_metrics(
    spectral: Union[SpectralResult, Dict[str, SpectralResult]],
    thermo: ThermoResult,
    belief: BeliefResult,
    save_dir: str,
    model_name: str = "",
    show: bool = False,
) -> List[str]:
    """
    Generate and save all 2D metric plots to a directory.
    
    Args:
        spectral: Spectral curvature result(s)
        thermo: Thermodynamic length result
        belief: Belief vector field result
        save_dir: Directory to save plots
        model_name: Model name for titles
        show: Whether to display plots
    
    Returns:
        List of saved file paths
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    saved_paths = []
    
    # Spectral curvature
    spectral_path = str(save_path / "spectral_curvature.png")
    plot_spectral_curvature(spectral, spectral_path, model_name=model_name, show=show)
    saved_paths.append(spectral_path)
    
    # Thermodynamic length
    thermo_path = str(save_path / "thermodynamic_length.png")
    plot_thermodynamic_length(thermo, thermo_path, model_name=model_name, show=show)
    saved_paths.append(thermo_path)
    
    # Belief vector field
    belief_path = str(save_path / "belief_vector_field.png")
    plot_belief_vector_field(belief, belief_path, model_name=model_name, show=show)
    saved_paths.append(belief_path)
    
    return saved_paths


def plot_master_panel(
    spectral: Union[SpectralResult, Dict[str, SpectralResult]],
    thermo: ThermoResult,
    belief: Union[BeliefResult, Dict[str, BeliefResult]],
    save_path: Optional[str] = None,
    model_name: str = "",
    show: bool = True,
    include_3d: bool = True,
) -> Figure:
    """
    Create a 2x2 panel summarizing all metrics.
    
    Layout:
        (1) Spectral Curvature κ_ℓ
        (2) Thermodynamic Length Δ_ℓ
        (3) Belief Vector Field ||v_ℓ||
        (4) nDNA Trajectory (3D) or combined overlay
    
    Args:
        spectral: Spectral curvature result(s)
        thermo: Thermodynamic length result
        belief: Belief vector field result(s)
        save_path: Optional path to save the figure
        model_name: Model name for title
        show: Whether to display the plot
        include_3d: Include 3D trajectory subplot (requires 3D projection)
    
    Returns:
        matplotlib Figure
    """
    apply_style()
    
    if include_3d:
        fig = plt.figure(figsize=FIGSIZE_PANEL)
        ax0 = fig.add_subplot(2, 2, 1)
        ax1 = fig.add_subplot(2, 2, 2)
        ax2 = fig.add_subplot(2, 2, 3)
        ax3 = fig.add_subplot(2, 2, 4, projection="3d")
    else:
        fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(2, 2, figsize=FIGSIZE_PANEL)
    
    # (1) Spectral Curvature
    if isinstance(spectral, dict):
        colors = get_color_palette(len(spectral))
        for (label, res), color in zip(spectral.items(), colors):
            ax0.plot(res.layer_indices, res.curvatures, marker="o", label=label, color=color)
        ax0.legend(fontsize=8)
        first_spectral = next(iter(spectral.values()))
    else:
        ax0.plot(spectral.layer_indices, spectral.curvatures, marker="o", color=COLORS["curvature"])
        first_spectral = spectral
    
    ax0.set_title("Spectral Curvature κ_ℓ")
    ax0.set_xlabel("Interior layer ℓ")
    ax0.set_ylabel("κ_ℓ")
    ax0.grid(True, alpha=0.3)
    
    # (2) Thermodynamic Length
    ax1.plot(thermo.step_indices, thermo.step_lengths, marker="o", color=COLORS["fr_step"])
    ax1.set_title(f"FR Step Length Δ_ℓ\nTotal: {thermo.total_length:.4f} rad")
    ax1.set_xlabel("Step (ℓ → ℓ+1)")
    ax1.set_ylabel("Δ_ℓ (radians)")
    ax1.grid(True, alpha=0.3)
    
    # (3) Belief Vector Field
    if isinstance(belief, dict):
        colors = get_color_palette(len(belief))
        for (label, res), color in zip(belief.items(), colors):
            ax2.plot(res.layer_indices, res.belief_norms, marker="o", label=label, color=color)
        ax2.legend(fontsize=8)
        first_belief = next(iter(belief.values()))
    else:
        ax2.plot(belief.layer_indices, belief.belief_norms, marker="o", color=COLORS["belief"])
        first_belief = belief
    
    ax2.set_title("Belief Vector Field ||v_ℓ||")
    ax2.set_xlabel("Layer ℓ")
    ax2.set_ylabel("||v_ℓ||_FR")
    ax2.grid(True, alpha=0.3)
    
    # (4) 3D Trajectory or Combined Plot
    if include_3d:
        # Get data for 3D
        kappa = first_spectral.curvatures
        fr_steps = thermo.step_lengths
        
        if isinstance(belief, dict):
            for (label, bres), color in zip(belief.items(), get_color_palette(len(belief))):
                v_norms = bres.belief_norms
                # Align indices: κ and Δ have length L-1, v has length L
                L = len(v_norms)
                l_min = 2
                ells = np.arange(l_min, L)
                
                # Make sure indices are valid
                if len(kappa) >= L - 1 and len(fr_steps) >= L - 1:
                    x = kappa[ells - 1]
                    y = fr_steps[ells - 1]
                    z = v_norms[ells]
                    ax3.plot(x, y, z, marker="o", label=label, color=color)
            ax3.legend(fontsize=8)
        else:
            v_norms = first_belief.belief_norms
            L = len(v_norms)
            l_min = 2
            ells = np.arange(l_min, L)
            
            if len(kappa) >= L - 1 and len(fr_steps) >= L - 1:
                x = kappa[ells - 1]
                y = fr_steps[ells - 1]
                z = v_norms[ells]
                ax3.plot(x, y, z, marker="o", color=COLORS["ndna"])
        
        ax3.set_xlabel("κ_ℓ")
        ax3.set_ylabel("Δ_ℓ")
        ax3.set_zlabel("||v_ℓ||")
        ax3.set_title("nDNA Geometry Trajectory")
    else:
        # Combined overlay plot
        ax3_twin = ax3.twinx()
        l1, = ax3.plot(first_spectral.layer_indices, first_spectral.curvatures, 
                       marker="o", color=COLORS["curvature"], label="κ_ℓ")
        l2, = ax3.plot(thermo.step_indices, thermo.step_lengths,
                       marker="s", color=COLORS["fr_step"], label="Δ_ℓ")
        l3, = ax3_twin.plot(first_belief.layer_indices, first_belief.belief_norms,
                           marker="^", color=COLORS["belief"], label="||v_ℓ||")
        
        ax3.set_xlabel("Layer")
        ax3.set_ylabel("κ_ℓ, Δ_ℓ")
        ax3_twin.set_ylabel("||v_ℓ||")
        ax3.legend(handles=[l1, l2, l3], loc="upper right", fontsize=8)
        ax3.set_title("All Metrics Overlay")
    
    fig.suptitle(f"nDNA Geometry Summary - {model_name}", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    
    if show:
        plt.show()
    else:
        plt.close(fig)
    
    return fig

