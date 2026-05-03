"""
Build interactive 3D Plotly HTMLs for collapse_runs_multi outputs.

For each model directory under `base_dir` (default: ./collapse_runs_multi),
this script loads spectral curvature (spectral_curvature.json) and Method-5
thermo/belief metrics (method5_unified.json) across generations, aligns them
layerwise, and emits a single HTML with four interactive 3D plots and two 2D plots:

  1) layer vs spectral vs thermo (3D)
  2) layer vs thermo vs belief (3D)
  3) layer vs spectral vs belief (3D)
  4) spectral vs thermo vs belief (3D, layer encoded by color)
  5) thermo vs layer (2D)
  6) belief vs layer (2D)

Assumptions:
  - spectral = `curvature_mean`
  - thermo  = `Delta`
  - belief  = `Eta` (change via --belief_key if you prefer `E`)
  - Files live at: base_dir/<model>/<exp>/gen*/metrics/{method5_unified.json, spectral_curvature.json}
  - If multiple exp folders exist for a model, the first (sorted) is used.
"""
import argparse
import json
import os
from typing import Dict, List, Tuple

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


METHOD5_FILE = "method5_unified.json"
SPECTRAL_FILE = "spectral_curvature.json"


# ---------------------------
# Helpers
# ---------------------------

def load_json(path: str) -> Dict:
    with open(path, "r") as f:
        return json.load(f)


def find_models(base_dir: str) -> List[str]:
    return sorted([
        d for d in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, d)) and not d.startswith(".")
    ])


def find_exp_dirs(model_dir: str) -> List[str]:
    return sorted([
        d for d in os.listdir(model_dir)
        if os.path.isdir(os.path.join(model_dir, d)) and not d.startswith(".")
    ])


def find_generations(exp_dir: str) -> List[int]:
    gens = []
    for name in os.listdir(exp_dir):
        if name.startswith("gen") and os.path.isdir(os.path.join(exp_dir, name)):
            try:
                gens.append(int(name[3:]))
            except ValueError:
                continue
    gens = sorted(gens)
    if not gens:
        raise RuntimeError(f"No gen* folders found under {exp_dir}")
    return gens


def load_metrics(gen_dir: str, belief_key: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    metrics_dir = os.path.join(gen_dir, "metrics")
    m5_path = os.path.join(metrics_dir, METHOD5_FILE)
    spec_path = os.path.join(metrics_dir, SPECTRAL_FILE)
    if not (os.path.isfile(m5_path) and os.path.isfile(spec_path)):
        raise FileNotFoundError(f"Missing metrics in {metrics_dir}")

    m5 = load_json(m5_path)
    spec = load_json(spec_path)

    spectral = np.asarray(spec.get("curvature_mean", []), dtype=float)
    thermo = np.asarray(m5.get("Delta", []), dtype=float)
    belief = np.asarray(m5.get(belief_key, []), dtype=float)

    # Align lengths conservatively
    k = min(len(spectral), len(thermo), len(belief))
    spectral = spectral[:k]
    thermo = thermo[:k]
    belief = belief[:k]
    spectral = robust_norm_to_range(spectral)
    # thermo = robust_norm_to_range(thermo)
    belief = robust_norm_to_range(belief)
    layers = np.arange(1, k + 1)
    return layers, spectral, thermo, belief


# ---------------------------
# Plotting
# ---------------------------

def build_figure(per_gen: List[Tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]) -> go.Figure:
    # Stack plots vertically: 4 3D plots + 2 2D plots
    fig = make_subplots(
        rows=6,
        cols=1,
        specs=[
            [{"type": "scene"}],
            [{"type": "scene"}],
            [{"type": "scene"}],
            [{"type": "scene"}],
            [{"type": "xy"}],
            [{"type": "xy"}],
        ],
        subplot_titles=(
            "Layer vs Spectral vs Thermo",
            "Layer vs Thermo vs Belief",
            "Layer vs Spectral vs Belief",
            "Spectral vs Thermo vs Belief",
            "Thermo vs Layer (2D)",
            "Belief vs Layer (2D)",
        ),
        row_heights=[0.15, 0.15, 0.15, 0.15, 0.4, 0.4],  # Make 2D plots taller
        vertical_spacing=0.05,
    )

    # Set up a shared color axis for generations
    gens = [g for g, *_ in per_gen]
    if gens:
        g_min, g_max = min(gens), max(gens)
    else:
        g_min, g_max = 0, 1

    def gen_to_color(gen: int, colorscale: str = "Viridis") -> str:
        # Normalize to [0,1] and sample colorscale for consistent line colors
        from plotly.express.colors import sample_colorscale
        t = 0.0 if g_max == g_min else (gen - g_min) / (g_max - g_min)
        return sample_colorscale(colorscale, [t])[0]

    for idx, (gen, layers, spectral, thermo, belief) in enumerate(per_gen):
        line_color = gen_to_color(gen)

        # 1) layer-spectral-thermo (3D)
        fig.add_trace(
            go.Scatter3d(
                x=layers,
                y=spectral,
                z=thermo,
                mode="lines+markers",
                name=f"gen {gen}",
                line=dict(color=line_color, width=3),
                marker=dict(size=0.5, color=gen, coloraxis="coloraxis"),
                legendgroup=f"gen{gen}_group1",
                legend="legend1",
            ),
            row=1,
            col=1,
        )

        # 2) layer-thermo-belief (3D)
        fig.add_trace(
            go.Scatter3d(
                x=layers,
                y=thermo,
                z=belief,
                mode="lines+markers",
                name=f"gen {gen}",
                line=dict(color=line_color, width=3),
                marker=dict(size=0.5, color=gen, coloraxis="coloraxis"),
                legendgroup=f"gen{gen}_group2",
                legend="legend2",
            ),
            row=2,
            col=1,
        )

        # 3) layer-spectral-belief (3D)
        fig.add_trace(
            go.Scatter3d(
                x=layers,
                y=spectral,
                z=belief,
                mode="lines+markers",
                name=f"gen {gen}",
                line=dict(color=line_color, width=3),
                marker=dict(size=0.5, color=gen, coloraxis="coloraxis"),
                legendgroup=f"gen{gen}_group3",
                legend="legend3",
            ),
            row=3,
            col=1,
        )

        # 4) spectral-thermo-belief (3D)
        fig.add_trace(
            go.Scatter3d(
                x=spectral,
                y=thermo,
                z=belief,
                mode="lines+markers",
                name=f"gen {gen}",
                line=dict(color=line_color, width=3),
                marker=dict(
                    size=0.5,
                    color=gen,
                    coloraxis="coloraxis",
                ),
                legendgroup=f"gen{gen}_group4",
                legend="legend4",
            ),
            row=4,
            col=1,
        )

        # 5) thermo vs layer (2D)
        fig.add_trace(
            go.Scatter(
                x=layers,
                y=thermo,
                mode="lines+markers",
                name=f"gen {gen}",
                line=dict(color=line_color, width=2),
                marker=dict(size=4, color=line_color),
                legendgroup=f"gen{gen}_group5",
                legend="legend5",
            ),
            row=5,
            col=1,
        )

        # 6) belief vs layer (2D)
        fig.add_trace(
            go.Scatter(
                x=layers,
                y=belief,
                mode="lines+markers",
                name=f"gen {gen}",
                line=dict(color=line_color, width=2),
                marker=dict(size=4, color=line_color),
                legendgroup=f"gen{gen}_group6",
                legend="legend6",
            ),
            row=6,
            col=1,
        )

    fig.update_layout(
        height=3400,
        width=1200,
        title="Collapse geometry: spectral, thermo, belief (3D + 2D)",
        margin=dict(l=0, r=0, t=40, b=0),
        coloraxis=dict(
            colorscale="Viridis",
            colorbar=dict(title="Generation"),
            cmin=g_min,
            cmax=g_max,
        ),
        # Configure multiple legends positioned beside each subplot
        legend1=dict(x=1.02, y=1.0, xanchor="left", yanchor="top", title="Plot 1"),
        legend2=dict(x=1.02, y=0.83, xanchor="left", yanchor="top", title="Plot 2"),
        legend3=dict(x=1.02, y=0.66, xanchor="left", yanchor="top", title="Plot 3"),
        legend4=dict(x=1.02, y=0.49, xanchor="left", yanchor="top", title="Plot 4"),
        legend5=dict(x=1.02, y=0.32, xanchor="left", yanchor="top", title="Plot 5"),
        legend6=dict(x=1.02, y=0.15, xanchor="left", yanchor="top", title="Plot 6"),
    )

    # Axis labels for 3D subplots
    fig.update_scenes(
        xaxis_title="Layer",
        yaxis_title="Spectral κ",
        zaxis_title="Thermo Δ",
        row=1,
        col=1,
    )
    fig.update_scenes(
        xaxis_title="Layer",
        yaxis_title="Thermo Δ",
        zaxis_title="Belief",
        row=2,
        col=1,
    )
    fig.update_scenes(
        xaxis_title="Layer",
        yaxis_title="Spectral κ",
        zaxis_title="Belief",
        row=3,
        col=1,
    )
    fig.update_scenes(
        xaxis_title="Spectral κ",
        yaxis_title="Thermo Δ",
        zaxis_title="Belief",
        row=4,
        col=1,
    )

    # Axis labels for 2D subplots
    fig.update_xaxes(title_text="Layer", row=5, col=1)
    fig.update_yaxes(title_text="Thermo Δ", row=5, col=1)
    fig.update_xaxes(title_text="Layer", row=6, col=1)
    fig.update_yaxes(title_text="Belief", row=6, col=1)

    return fig


def robust_norm_to_range(arr: np.ndarray, min_out=0.005, max_out=1.0) -> np.ndarray:
    """
    Winsorize by IQR and scale to [min_out, max_out].
    """
    arr = np.asarray(arr).astype(float).reshape(-1)
    if arr.size == 0:
        return arr

    q1 = np.percentile(arr, 25)
    q3 = np.percentile(arr, 75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    clipped = np.clip(arr, lower, upper)

    mn = float(np.min(clipped))
    mx = float(np.max(clipped))
    if mx - mn == 0:
        return np.full_like(clipped, min_out)

    return min_out + (clipped - mn) * (max_out - min_out) / (mx - mn)

# ---------------------------
# Main
# ---------------------------

def main():
    parser = argparse.ArgumentParser(description="Plot spectral/thermo/belief pairs for collapse_runs_multi.")
    parser.add_argument("--base_dir", default="./collapse_runs_multi", help="Root folder containing model subdirs")
    parser.add_argument(
        "--belief_key",
        default="Eta",
        help="Key to use as belief metric from method5_unified.json (e.g., Eta or E)",
    )
    parser.add_argument(
        "--exp_name",
        default="",
        help="If set, only process this experiment subfolder per model. Otherwise process all.",
    )
    args = parser.parse_args()

    models = find_models(args.base_dir)
    if not models:
        raise SystemExit(f"No model directories found under {args.base_dir}")

    for model in models:
        model_dir = os.path.join(args.base_dir, model)
        exp_dirs = find_exp_dirs(model_dir)
        if args.exp_name:
            exp_dirs = [e for e in exp_dirs if e == args.exp_name]
        if not exp_dirs:
            print(f"[WARN] No experiment subdirs for {model} matching '{args.exp_name}'")
            continue

        for exp in exp_dirs:
            exp_dir = os.path.join(model_dir, exp)
            gens = find_generations(exp_dir)

            per_gen = []
            for g in gens:
                gen_dir = os.path.join(exp_dir, f"gen{g}")
                metrics_dir = os.path.join(gen_dir, "metrics")
                if not os.path.isdir(metrics_dir):
                    print(f"[WARN] Missing metrics dir for {model}/{exp} gen{g}")
                    continue
                try:
                    layers, spectral, thermo, belief = load_metrics(gen_dir, args.belief_key)
                except Exception as e:
                    print(f"[WARN] Skipping {model}/{exp} gen{g}: {e}")
                    continue
                if len(layers) == 0:
                    print(f"[WARN] Empty metrics for {model}/{exp} gen{g}")
                    continue
                per_gen.append((g, layers, spectral, thermo, belief))

            if not per_gen:
                print(f"[WARN] No usable generations for {model}/{exp}")
                continue

            fig = build_figure(per_gen)

            out_html = os.path.join(model_dir, f"{model}_{exp}_collapse_pairs_3d.html")
            fig.write_html(out_html, include_plotlyjs="cdn")
            print(f"[OK] Wrote {out_html}")


if __name__ == "__main__":
    main()
