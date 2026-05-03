"""
ndna.visualization.plots_3d

3D interactive plotly plots for nDNA metrics.

Adapted from ndna_lib/plots.py with enhancements for the new result dataclasses.
"""

from pathlib import Path
from typing import Dict, List, Mapping, Optional, Union

import numpy as np

try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

from ..core.results import (
    SpectralResult,
    ThermoResultPerSample,
    BeliefResultPerSample,
)
from .styles import CONCEPT_COLORS, PLOTLY_COLORSCALE


def _check_plotly():
    """Check if plotly is available."""
    if not PLOTLY_AVAILABLE:
        raise ImportError(
            "plotly is required for 3D plots. Install it with: pip install plotly"
        )


def plot_spectral_3d(
    results: Dict[str, SpectralResult],
    save_path: Optional[str] = None,
    model_name: str = "",
    log_scale: bool = False,
) -> "go.Figure":
    """
    3D surface plot: Prompt × Layer × Curvature.
    
    Each prompt becomes a row on the Y-axis, layers on X, curvature on Z.
    
    Args:
        results: Dict mapping prompt label to SpectralResult
        save_path: Optional path to save HTML file
        model_name: Model name for title
        log_scale: Use log scale for Z-axis
    
    Returns:
        Plotly Figure object
    """
    _check_plotly()
    
    if not results:
        raise ValueError("results dict is empty")
    
    labels = list(results.keys())
    first = next(iter(results.values()))
    layer_indices = first.layer_indices
    
    # Build curvature matrix
    curvatures = np.array([results[l].curvatures for l in labels])
    
    if log_scale:
        curvatures = np.log10(np.maximum(curvatures, 1e-12))
        z_title = "log₁₀(κ_ℓ)"
    else:
        z_title = "κ_ℓ"
    
    fig = go.Figure()
    
    # Surface plot
    fig.add_trace(go.Surface(
        x=layer_indices,
        y=np.arange(len(labels)),
        z=curvatures,
        colorscale=PLOTLY_COLORSCALE,
        colorbar=dict(title=z_title),
        hovertemplate=(
            "Layer: %{x}<br>"
            "Prompt: %{customdata}<br>"
            f"{z_title}: " + "%{z:.4f}<extra></extra>"
        ),
        customdata=np.array([[l] * len(layer_indices) for l in labels]),
    ))
    
    fig.update_layout(
        title=f"Spectral Curvature Surface<br>{model_name}" if model_name else "Spectral Curvature Surface",
        scene=dict(
            xaxis_title="Layer ℓ",
            yaxis_title="Prompt",
            zaxis_title=z_title,
            yaxis=dict(
                tickmode="array",
                tickvals=list(range(len(labels))),
                ticktext=labels,
            ),
        ),
        margin=dict(l=0, r=0, t=50, b=0),
    )
    
    if save_path:
        fig.write_html(save_path, include_plotlyjs="cdn")
    
    return fig


def plot_thermodynamic_3d(
    result: ThermoResultPerSample,
    save_path: Optional[str] = None,
    model_name: str = "",
) -> "go.Figure":
    """
    3D surface plot: Sample × Step × FR Length.
    
    Args:
        result: ThermoResultPerSample with per-sample data
        save_path: Optional path to save HTML file
        model_name: Model name for title
    
    Returns:
        Plotly Figure object
    """
    _check_plotly()
    
    per_sample = result.per_sample_lengths  # (N, L-1)
    step_indices = result.step_indices
    num_samples = result.num_samples
    
    fig = go.Figure()
    
    fig.add_trace(go.Surface(
        x=step_indices,
        y=np.arange(num_samples),
        z=per_sample,
        colorscale=PLOTLY_COLORSCALE,
        colorbar=dict(title="Δ_ℓ (rad)"),
        hovertemplate=(
            "Step: %{x}<br>"
            "Sample: %{y}<br>"
            "Δ_ℓ: %{z:.6f}<extra></extra>"
        ),
    ))
    
    fig.update_layout(
        title=f"FR Thermodynamic Length per Sample<br>{model_name}" if model_name else "FR Thermodynamic Length per Sample",
        scene=dict(
            xaxis_title="Step (ℓ → ℓ+1)",
            yaxis_title="Sample",
            zaxis_title="Δ_ℓ (radians)",
        ),
        margin=dict(l=0, r=0, t=50, b=0),
    )
    
    if save_path:
        fig.write_html(save_path, include_plotlyjs="cdn")
    
    return fig


def plot_belief_3d(
    result: BeliefResultPerSample,
    save_path: Optional[str] = None,
    model_name: str = "",
) -> "go.Figure":
    """
    3D surface plot: Sample × Layer × ||v||.
    
    Args:
        result: BeliefResultPerSample with per-sample data
        save_path: Optional path to save HTML file
        model_name: Model name for title
    
    Returns:
        Plotly Figure object
    """
    _check_plotly()
    
    per_sample = result.per_sample_norms  # (N, L)
    layer_indices = result.layer_indices
    num_samples = result.num_samples
    
    fig = go.Figure()
    
    fig.add_trace(go.Surface(
        x=layer_indices,
        y=np.arange(num_samples),
        z=per_sample,
        colorscale=PLOTLY_COLORSCALE,
        colorbar=dict(title="||v_ℓ||"),
        hovertemplate=(
            "Layer: %{x}<br>"
            "Sample: %{y}<br>"
            "||v_ℓ||: %{z:.6f}<extra></extra>"
        ),
    ))
    
    fig.update_layout(
        title=f"Belief Vector Field per Sample<br>{model_name}" if model_name else "Belief Vector Field per Sample",
        scene=dict(
            xaxis_title="Layer ℓ",
            yaxis_title="Sample",
            zaxis_title="||v_ℓ||_FR",
        ),
        margin=dict(l=0, r=0, t=50, b=0),
    )
    
    if save_path:
        fig.write_html(save_path, include_plotlyjs="cdn")
    
    return fig


def plot_ndna_trajectory_3d(
    kappa: np.ndarray,
    fr_steps: np.ndarray,
    belief_norms: Union[np.ndarray, Dict[str, np.ndarray]],
    save_path: Optional[str] = None,
    model_name: str = "",
    l_min: int = 2,
) -> "go.Figure":
    """
    3D trajectory in (κ_ℓ, Δ_ℓ, ||v_ℓ||) space.
    
    Consecutive layers are connected with lines, showing how the model
    traverses the nDNA geometry space through depth.
    
    Args:
        kappa: Spectral curvature (L-1,), κ_ℓ for ℓ=1..L-1
        fr_steps: FR step lengths (L-1,), Δ_ℓ for ℓ=1..L-1
        belief_norms: Belief norms (L,) or dict of {concept: (L,)}
        save_path: Optional path to save HTML file
        model_name: Model name for title
        l_min: Minimum layer index for trajectory (skip shallow layers)
    
    Returns:
        Plotly Figure object
    """
    _check_plotly()
    
    kappa = np.asarray(kappa).ravel()
    fr_steps = np.asarray(fr_steps).ravel()
    
    if kappa.shape != fr_steps.shape:
        raise ValueError(f"kappa and fr_steps must have same shape: {kappa.shape} vs {fr_steps.shape}")
    
    # Handle single array or dict
    if isinstance(belief_norms, dict):
        belief_dict = belief_norms
    else:
        belief_dict = {"default": np.asarray(belief_norms).ravel()}
    
    # Validate shapes
    first_v = next(iter(belief_dict.values()))
    L = len(first_v)
    Lm1 = len(kappa)
    
    if Lm1 != L - 1:
        raise ValueError(f"Expected len(kappa)=L-1={L-1}, got {Lm1}")
    
    if not (1 <= l_min < L):
        raise ValueError(f"l_min must be in [1, {L-1}], got {l_min}")
    
    fig = go.Figure()
    
    ells = np.arange(l_min, L)  # ℓ = l_min..L-1
    colors = CONCEPT_COLORS
    
    for i, (name, v_norms) in enumerate(belief_dict.items()):
        v_arr = np.asarray(v_norms).ravel()
        if len(v_arr) != L:
            raise ValueError(f"belief_norms for '{name}' has length {len(v_arr)}, expected {L}")
        
        x = kappa[ells - 1]      # κ_ℓ
        y = fr_steps[ells - 1]   # Δ_ℓ
        z = v_arr[ells]          # ||v_ℓ||
        
        color = colors[i % len(colors)]
        
        fig.add_trace(go.Scatter3d(
            x=x, y=y, z=z,
            mode="lines+markers",
            name=name,
            line=dict(color=color, width=3),
            marker=dict(size=5, color=color),
            hovertemplate=(
                f"{name}<br>"
                "κ_ℓ: %{x:.6f}<br>"
                "Δ_ℓ: %{y:.6f}<br>"
                "||v_ℓ||: %{z:.6f}<br>"
                "Layer: %{customdata}<extra></extra>"
            ),
            customdata=ells,
        ))
    
    fig.update_layout(
        title=f"nDNA Geometry Trajectory<br>{model_name}" if model_name else "nDNA Geometry Trajectory",
        scene=dict(
            xaxis_title="κ_ℓ (curvature)",
            yaxis_title="Δ_ℓ (FR step)",
            zaxis_title="||v_ℓ|| (belief)",
        ),
        legend_title_text="Concept",
        margin=dict(l=0, r=0, t=50, b=0),
    )
    
    if save_path:
        fig.write_html(save_path, include_plotlyjs="cdn")
    
    return fig


def export_multi_model_html(
    models_data: Mapping[str, Mapping[str, object]],
    html_path: str = "ndna_trajectories.html",
    l_min: int = 2,
) -> "go.Figure":
    """
    Create interactive HTML with 3D nDNA trajectories for multiple models.
    
    All models are superimposed in the same (κ_ℓ, Δ_ℓ, ||v_ℓ||) space for comparison.
    
    Args:
        models_data: Dict mapping model_name -> {
            "kappa": (L-1,) curvatures,
            "fr_steps": (L-1,) FR steps,
            "belief_norms": Dict[concept, (L,)] belief norms
        }
        html_path: Output HTML file path
        l_min: Minimum layer for trajectory
    
    Returns:
        Plotly Figure object
    
    Example:
        >>> export_multi_model_html({
        ...     "GPT-2": {
        ...         "kappa": kappa_gpt2,
        ...         "fr_steps": fr_gpt2,
        ...         "belief_norms": {"squad": v_gpt2}
        ...     },
        ...     "LLaMA": {
        ...         "kappa": kappa_llama,
        ...         "fr_steps": fr_llama,
        ...         "belief_norms": {"squad": v_llama}
        ...     }
        ... }, "comparison.html")
    """
    _check_plotly()
    
    fig = go.Figure()
    
    color_idx = 0
    
    for model_name, data in models_data.items():
        if "kappa" not in data or "fr_steps" not in data or "belief_norms" not in data:
            raise ValueError(f"Model '{model_name}' must have 'kappa', 'fr_steps', and 'belief_norms'")
        
        kappa = np.asarray(data["kappa"]).ravel()
        fr_steps = np.asarray(data["fr_steps"]).ravel()
        belief_norms = data["belief_norms"]
        
        if kappa.shape != fr_steps.shape:
            raise ValueError(f"Model '{model_name}': kappa and fr_steps shape mismatch")
        
        if not belief_norms:
            continue
        
        first_v = next(iter(belief_norms.values()))
        L = len(np.asarray(first_v).ravel())
        Lm1 = len(kappa)
        
        if Lm1 != L - 1:
            raise ValueError(f"Model '{model_name}': expected len(kappa)=L-1={L-1}, got {Lm1}")
        
        if not (1 <= l_min < L):
            raise ValueError(f"l_min={l_min} invalid for model '{model_name}' with L={L}")
        
        ells = np.arange(l_min, L)
        
        for concept_name, v_norms in belief_norms.items():
            v_arr = np.asarray(v_norms).ravel()
            if len(v_arr) != L:
                raise ValueError(f"Model '{model_name}', concept '{concept_name}': "
                               f"belief_norms has length {len(v_arr)}, expected {L}")
            
            x = kappa[ells - 1]
            y = fr_steps[ells - 1]
            z = v_arr[ells]
            
            color = CONCEPT_COLORS[color_idx % len(CONCEPT_COLORS)]
            color_idx += 1
            
            fig.add_trace(go.Scatter3d(
                x=x, y=y, z=z,
                mode="lines+markers",
                name=f"{model_name} / {concept_name}",
                line=dict(color=color, width=3),
                marker=dict(size=4, color=color),
                hovertemplate=(
                    f"{model_name} / {concept_name}<br>"
                    "κ_ℓ: %{x:.6f}<br>"
                    "Δ_ℓ: %{y:.6f}<br>"
                    "||v_ℓ||: %{z:.6f}<br>"
                    "Layer: %{customdata}<extra></extra>"
                ),
                customdata=ells,
            ))
    
    fig.update_layout(
        title="nDNA Geometry Trajectories - Model Comparison",
        scene=dict(
            xaxis_title="κ_ℓ (curvature)",
            yaxis_title="Δ_ℓ (FR step)",
            zaxis_title="||v_ℓ|| (belief)",
        ),
        legend_title_text="Model / Concept",
        margin=dict(l=0, r=0, t=50, b=0),
    )
    
    fig.write_html(html_path, include_plotlyjs="cdn")
    
    return fig


def plot_all_3d(
    spectral_by_prompt: Optional[Dict[str, SpectralResult]] = None,
    thermo_per_sample: Optional[ThermoResultPerSample] = None,
    belief_per_sample: Optional[BeliefResultPerSample] = None,
    save_dir: str = "plots_3d",
    model_name: str = "",
) -> List[str]:
    """
    Generate and save all 3D plots to a directory.
    
    Args:
        spectral_by_prompt: Optional spectral results by prompt
        thermo_per_sample: Optional per-sample thermodynamic results
        belief_per_sample: Optional per-sample belief results
        save_dir: Directory to save HTML files
        model_name: Model name for titles
    
    Returns:
        List of saved file paths
    """
    _check_plotly()
    
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    saved_paths = []
    
    if spectral_by_prompt is not None and spectral_by_prompt:
        path = str(save_path / "spectral_3d.html")
        plot_spectral_3d(spectral_by_prompt, path, model_name)
        saved_paths.append(path)
    
    if thermo_per_sample is not None:
        path = str(save_path / "thermodynamic_3d.html")
        plot_thermodynamic_3d(thermo_per_sample, path, model_name)
        saved_paths.append(path)
    
    if belief_per_sample is not None:
        path = str(save_path / "belief_3d.html")
        plot_belief_3d(belief_per_sample, path, model_name)
        saved_paths.append(path)
    
    return saved_paths

