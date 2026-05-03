# ndna_lib/plots.py
#
# Plotting utilities for Method 5 geometry:
#   - Spectral curvature κ_ℓ
#   - Fisher–Rao step lengths Δ_ℓ
#   - Belief vector field norms ‖v_ℓ(c)‖
#   - 3D trajectories in (κ_ℓ, Δ_ℓ, ‖v_ℓ(c)‖) space (nDNA geometry)
#
# All functions are pure plotting: they take numpy arrays and dictionaries
# produced by geometry.py and create figures.

from __future__ import annotations

from typing import Dict, Optional, Mapping

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  # registers 3D projection
import plotly.graph_objects as go


plt.style.use("seaborn-v0_8-whitegrid")


# --------------------------------------------------------------------------
# 1. Spectral curvature
# --------------------------------------------------------------------------


def plot_curvature(
    kappa: np.ndarray,
    model_name: str,
    save_path: Optional[str] = None,
):
    """
    Plot κ_ℓ^(simp) across depth.

    Args:
        kappa      : (L-1,) array, interior curvature profile
        model_name : string to show in title
        save_path  : optional path to save instead of just showing
    """
    kappa = np.asarray(kappa).reshape(-1)
    Lm1 = kappa.shape[0]
    layers = np.arange(1, Lm1 + 1)  # interpret as ℓ = 1..L-1

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(layers, kappa, marker="o")
    ax.set_title(f"Spectral Curvature κ_ℓ^(simp)\n{model_name}")
    ax.set_xlabel("Interior layer index ℓ (1 … L−1)")
    ax.set_ylabel("κ_ℓ^(simp)")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


# --------------------------------------------------------------------------
# 2. Fisher–Rao thermodynamic length profile from predictions
# --------------------------------------------------------------------------


def plot_fr_profile(
    fr_steps: np.ndarray,
    mean_total_len: float,
    model_name: str,
    save_path: Optional[str] = None,
):
    """
    Plot Fisher–Rao inter-layer distances Δ_ℓ (prediction-space).

    Args:
        fr_steps       : (L-1,) mean FR distance per step (ℓ → ℓ+1)
        mean_total_len : scalar, mean total FR length per token
        model_name     : title label
        save_path      : optional PNG path
    """
    fr_steps = np.asarray(fr_steps).reshape(-1)
    Lm1 = fr_steps.shape[0]
    steps = np.arange(1, Lm1 + 1)  # step index for ℓ → ℓ+1

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(steps, fr_steps, marker="o")
    ax.set_title(
        f"Fisher–Rao Thermodynamic Length per Step\n"
        f"{model_name}\n"
        f"Mean total per-token length: {mean_total_len:.3e} rad"
    )
    ax.set_xlabel("Inter-layer step (ℓ → ℓ+1)")
    ax.set_ylabel("Mean FR distance Δ_ℓ (radians)")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


# --------------------------------------------------------------------------
# 3. Belief vector field norms per concept
# --------------------------------------------------------------------------


def plot_belief_fields(
    belief_norms: Dict[str, np.ndarray],
    model_name: str,
    fr_norm: bool = True,
    save_path: Optional[str] = None,
):
    """
    Plot ‖v_ℓ(c)‖ vs layer for each concept c.

    Args:
        belief_norms : dict mapping concept name -> (L,) array of norms
        model_name   : string for title
        fr_norm      : if True, y-axis = FR norm (2‖v‖); else Euclidean
        save_path    : optional PNG path
    """
    if len(belief_norms) == 0:
        raise ValueError("belief_norms is empty.")

    first = next(iter(belief_norms.values()))
    L = np.asarray(first).reshape(-1).shape[0]
    layers = np.arange(1, L + 1)  # ℓ = 1..L

    fig, ax = plt.subplots(figsize=(12, 6))

    for name, vals in belief_norms.items():
        v = np.asarray(vals).reshape(-1)
        if v.shape[0] != L:
            raise ValueError("All belief_norm arrays must have same length.")
        ax.plot(layers, v, marker="o", label=name)

    ylabel = "‖v_ℓ(c)‖_FR" if fr_norm else "‖v_ℓ(c)‖"
    ax.set_title(f"Belief Vector Field Magnitude per Layer\n{model_name}")
    ax.set_xlabel("Layer index ℓ")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


# --------------------------------------------------------------------------
# 4. 3D nDNA geometry: (κ_ℓ, Δ_ℓ, ‖v_ℓ(c)‖) trajectories
# --------------------------------------------------------------------------


def plot_ndna_trajectory_3d(
    kappa: np.ndarray,
    fr_steps: np.ndarray,
    belief_norms: Mapping[str, np.ndarray],
    model_name: str,
    l_min: int = 2,
    save_path: Optional[str] = None,
):
    """
    3D trajectory in (κ_ℓ, Δ_ℓ, ‖v_ℓ(c)‖) space for each concept.
    Consecutive layers are connected with lines.

    Args:
        kappa        : (L-1,) spectral curvature κ_ℓ^(simp), ℓ = 1..L-1
        fr_steps     : (L-1,) FR step lengths Δ_ℓ, ℓ = 1..L-1
        belief_norms : dict concept -> (L,) array of ‖v_ℓ(c)‖, ℓ = 0..L-1
        model_name   : label for title
        l_min        : start layer index for trajectory (ℓ >= l_min), default 2
        save_path    : optional PNG path
    """
    if len(belief_norms) == 0:
        raise ValueError("belief_norms is empty.")

    kappa = np.asarray(kappa).reshape(-1)
    fr_steps = np.asarray(fr_steps).reshape(-1)
    if kappa.shape != fr_steps.shape:
        raise ValueError("kappa and fr_steps must have the same shape (L-1,).")

    first = next(iter(belief_norms.values()))
    v0 = np.asarray(first).reshape(-1)
    L = v0.shape[0]
    Lm1 = kappa.shape[0]
    if Lm1 != L - 1:
        raise ValueError("Expected len(kappa)=len(fr_steps)=L-1 given ‖v_ℓ‖ length L.")

    if not (1 <= l_min < L):
        raise ValueError(f"l_min must satisfy 1 <= l_min < L; got l_min={l_min}, L={L}.")

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    for name, vals in belief_norms.items():
        v_arr = np.asarray(vals).reshape(-1)
        if v_arr.shape[0] != L:
            raise ValueError(f"belief_norm for concept '{name}' has wrong length.")
        # ℓ runs over blocks; for ℓ in [l_min, L-1]:
        # κ_ℓ and Δ_ℓ live at index ℓ-1; ‖v_ℓ‖ lives at index ℓ.
        ells = np.arange(l_min, L)          # ℓ = l_min..L-1
        x = kappa[ells - 1]                 # κ_ℓ
        y = fr_steps[ells - 1]              # Δ_ℓ
        z = v_arr[ells]                     # ‖v_ℓ(c)‖

        ax.plot(x, y, z, marker="o", label=name)

    ax.set_xlabel("κ_ℓ^(simp)")
    ax.set_ylabel("Δ_ℓ (FR step)")
    ax.set_zlabel("‖v_ℓ(c)‖")
    ax.set_title(f"nDNA Geometry Trajectory\n{model_name}")
    ax.legend()

    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


# --------------------------------------------------------------------------
# 5. Master panel: κ_ℓ, Δ_ℓ, ‖v_ℓ(c)‖, 3D nDNA trajectory
# --------------------------------------------------------------------------


def plot_master_geometry_panel(
    kappa: np.ndarray,
    fr_steps: np.ndarray,
    belief_norms: Dict[str, np.ndarray],
    ndna_values: Dict[str, float],    # kept for API compatibility; not plotted
    model_name: str,
    fr_mean_total: Optional[float] = None,
    l_min: int = 2,
    save_path: Optional[str] = None,
):
    """
    A 2x2 panel summarizing the geometry:
        (1) κ_ℓ^(simp) over depth
        (2) Δ_ℓ FR step profile
        (3) ‖v_ℓ(c)‖ per concept
        (4) 3D nDNA geometry trajectory in (κ_ℓ, Δ_ℓ, ‖v_ℓ(c)‖) space

    Arrays must be consistent with geometry.py:
        - kappa              : (L-1,)
        - fr_steps           : (L-1,)
        - belief_norms[name] : (L,), ℓ = 0..L-1
        - ndna_values[name]  : scalar (not used in the plot)
    """
    if len(belief_norms) == 0:
        raise ValueError("belief_norms is empty in master panel.")

    kappa = np.asarray(kappa).reshape(-1)
    fr_steps = np.asarray(fr_steps).reshape(-1)
    if kappa.shape != fr_steps.shape:
        raise ValueError("kappa and fr_steps must have the same shape (L-1,).")

    first = next(iter(belief_norms.values()))
    v0 = np.asarray(first).reshape(-1)
    L = v0.shape[0]
    Lm1 = kappa.shape[0]
    if Lm1 != L - 1:
        raise ValueError("Expected len(kappa)=len(fr_steps)=L-1 with L=len(belief_norm).")

    layers_full = np.arange(1, L + 1)
    layers_int = np.arange(1, Lm1 + 1)

    if not (1 <= l_min < L):
        raise ValueError(f"l_min must satisfy 1 <= l_min < L; got l_min={l_min}, L={L}.")

    concepts = list(belief_norms.keys())

    fig = plt.figure(figsize=(14, 9))
    ax0 = fig.add_subplot(2, 2, 1)
    ax1 = fig.add_subplot(2, 2, 2)
    ax2 = fig.add_subplot(2, 2, 3)
    ax3 = fig.add_subplot(2, 2, 4, projection="3d")

    # (1) Curvature
    ax0.plot(layers_int, kappa, marker="o")
    ax0.set_title("Spectral Curvature κ_ℓ^(simp)")
    ax0.set_xlabel("Interior layer index ℓ (1 … L−1)")
    ax0.set_ylabel("κ_ℓ^(simp)")
    ax0.grid(True, alpha=0.3)

    # (2) FR steps
    title2 = "FR Step Length Δ_ℓ"
    if fr_mean_total is not None:
        title2 += f"\nMean total per-token length: {fr_mean_total:.3e} rad"
    ax1.plot(layers_int, fr_steps, marker="o")
    ax1.set_title(title2)
    ax1.set_xlabel("Inter-layer step (ℓ → ℓ+1)")
    ax1.set_ylabel("Δ_ℓ (radians)")
    ax1.grid(True, alpha=0.3)

    # (3) Belief norms per concept
    for name, vals in belief_norms.items():
        v = np.asarray(vals).reshape(-1)
        if v.shape[0] != L:
            raise ValueError("All belief_norm arrays must have length L.")
        ax2.plot(layers_full, v, marker="o", label=name)
    ax2.set_title("Belief Vector Field Magnitude ‖v_ℓ(c)‖")
    ax2.set_xlabel("Layer index ℓ")
    ax2.set_ylabel("‖v_ℓ(c)‖ (FR)")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # (4) 3D nDNA trajectory: (κ_ℓ, Δ_ℓ, ‖v_ℓ(c)‖)
    for name in concepts:
        v_arr = np.asarray(belief_norms[name]).reshape(-1)
        ells = np.arange(l_min, L)       # ℓ = l_min..L-1
        x = kappa[ells - 1]              # κ_ℓ
        y = fr_steps[ells - 1]           # Δ_ℓ
        z = v_arr[ells]                  # ‖v_ℓ(c)‖

        ax3.plot(x, y, z, marker="o", label=name)

    ax3.set_xlabel("κ_ℓ^(simp)")
    ax3.set_ylabel("Δ_ℓ (FR step)")
    ax3.set_zlabel("‖v_ℓ(c)‖")
    ax3.set_title("nDNA Geometry Trajectory")
    ax3.legend()

    fig.suptitle(f"Method 5 Geometry Summary - {model_name}", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    # Corrected save/show logic
    if save_path is not None:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


# --------------------------------------------------------------------------
# 6. Multi model 3D nDNA trajectories to a single HTML
# --------------------------------------------------------------------------


def export_ndna_trajectories_html(
    models: Mapping[str, Mapping[str, object]],
    l_min: int = 2,
    html_path: str = "ndna_trajectories_all_models.html",
):
    """
    Create a single interactive HTML with 3D nDNA trajectories for multiple models,
    all superimposed in the same (κ_ℓ, Δ_ℓ, ‖v_ℓ(c)‖) space.

    Args:
        models: dict mapping
            model_name -> {
                "kappa": (L-1,) array of κ_ℓ^(simp),
                "fr_steps": (L-1,) array of Δ_ℓ,
                "belief_norms": Dict[str, (L,)] mapping concept -> ‖v_ℓ(c)‖
            }
        l_min    : start layer index ℓ for trajectory (ℓ >= l_min)
        html_path: path to write the HTML file.
    """
    fig = go.Figure()

    for model_name, data in models.items():
        if "kappa" not in data or "fr_steps" not in data or "belief_norms" not in data:
            raise ValueError(
                f"Model '{model_name}' must provide 'kappa', 'fr_steps', and 'belief_norms'."
            )

        kappa = np.asarray(data["kappa"]).reshape(-1)
        fr_steps = np.asarray(data["fr_steps"]).reshape(-1)
        belief_norms = data["belief_norms"]

        if kappa.shape != fr_steps.shape:
            raise ValueError(
                f"Model '{model_name}' has mismatched shapes: "
                f"kappa.shape={kappa.shape}, fr_steps.shape={fr_steps.shape}."
            )
        if len(belief_norms) == 0:
            continue

        first = next(iter(belief_norms.values()))
        v0 = np.asarray(first).reshape(-1)
        L = v0.shape[0]
        Lm1 = kappa.shape[0]
        if Lm1 != L - 1:
            raise ValueError(
                f"Model '{model_name}' expected len(kappa)=len(fr_steps)=L-1, "
                f"got len(kappa)={Lm1}, L={L}."
            )

        if not (1 <= l_min < L):
            raise ValueError(
                f"l_min must satisfy 1 <= l_min < L for model '{model_name}'; "
                f"got l_min={l_min}, L={L}."
            )

        ells = np.arange(l_min, L)  # ℓ = l_min..L-1

        for concept_name, vals in belief_norms.items():
            v_arr = np.asarray(vals).reshape(-1)
            if v_arr.shape[0] != L:
                raise ValueError(
                    f"belief_norm for concept '{concept_name}' in model '{model_name}' "
                    f"has length {v_arr.shape[0]}, expected {L}."
                )

            x = kappa[ells - 1]       # κ_ℓ
            y = fr_steps[ells - 1]    # Δ_ℓ
            z = v_arr[ells]           # ‖v_ℓ(c)‖

            fig.add_trace(
                go.Scatter3d(
                    x=x,
                    y=y,
                    z=z,
                    mode="lines+markers",
                    name=f"{model_name} / {concept_name}",
                )
            )

    fig.update_layout(
        title="nDNA Geometry Trajectories across Models",
        scene=dict(
            xaxis_title="κ_ℓ^(simp)",
            yaxis_title="Δ_ℓ (FR step)",
            zaxis_title="‖v_ℓ(c)‖",
        ),
        legend_title_text="Model / Concept",
    )

    fig.write_html(html_path, include_plotlyjs="cdn")
