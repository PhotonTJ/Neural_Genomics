"""
Generate a compact, cross-dataset visualization for the Qwen 3.4B model using
only the provided .npz files:
  - results/method5_generic/rlhf/hh_rlhf__method5_Qwen_Qwen3_4B.npz
  - results/method5_generic/multi-lingual/everything_instruct_multilingual__method5_Qwen_Qwen3_4B.npz
  - results/method5_generic/mbpp/mbpp__method5_Qwen_Qwen3_4B.npz
  - results/method5_generic/gsm8k/gsm8k__method5_Qwen_Qwen3_4B.npz
  - results/method5_generic/squad/method5_Qwen_Qwen3_4B.npz

Outputs a single HTML with raw and normalized overlays for the four metrics:
Spectral (kappa), Thermodynamic (Delta), Belief norms (|v|), and derived nDNA.
"""

import argparse
import os
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Fixed file list; the script is intentionally scoped to just these five runs.
DATASETS = [
    ("HH RLHF", "results/method5_generic/rlhf/hh_rlhf__method5_Qwen_Qwen3_4B.npz"),
    ("Everything Multilingual", "results/method5_generic/multi-lingual/everything_instruct_multilingual__method5_Qwen_Qwen3_4B.npz"),
    ("MBPP", "results/method5_generic/mbpp/mbpp__method5_Qwen_Qwen3_4B.npz"),
    ("GSM8K", "results/method5_generic/gsm8k/gsm8k__method5_Qwen_Qwen3_4B.npz"),
    ("SQuAD", "results/method5_generic/squad/method5_Qwen_Qwen3_4B.npz"),
]

# Distinct colors per dataset for readability.
COLOR_PALETTE = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd", "#17becf"]


def robust_norm_to_range(arr: np.ndarray, min_out: float = 0.005, max_out: float = 1.0) -> np.ndarray:
    """
    Winsorize by IQR and scale to [min_out, max_out].
    Matches the normalization used in scripts/plot.py.
    """
    arr = np.asarray(arr, dtype=float).reshape(-1)
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


def load_npz(label: str, path: str, color: str) -> dict:
    """
    Load a single .npz file, trim to the common length, and compute normalized curves.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found")

    data = np.load(path)
    required = ("kappa", "Delta", "belief_norms")
    for key in required:
        if key not in data:
            raise KeyError(f"{path} missing required key '{key}'")

    kappa = np.asarray(data["kappa"]).reshape(-1)
    delta = np.asarray(data["Delta"]).reshape(-1)
    vnorm = np.asarray(data["belief_norms"]).reshape(-1)

    length = min(len(kappa), len(delta), len(vnorm))
    if length < 2:
        raise ValueError(f"{path} has insufficient points")

    kappa = kappa[:length]
    delta = delta[:length]
    vnorm = vnorm[:length]
    layers = np.arange(1, length + 1)

    # Raw derived nDNA before any normalization.
    ndna_raw = kappa * vnorm / np.log1p(delta)

    # Normalize components first, then derive nDNA and renormalize (matches plot.py).
    kappa_n = robust_norm_to_range(kappa)
    delta_n = robust_norm_to_range(delta)
    vnorm_n = robust_norm_to_range(vnorm)
    ndna_n = robust_norm_to_range(kappa_n * vnorm_n / np.log1p(delta_n))

    if "ndna_scalar" in data:
        scalar = float(np.asarray(data["ndna_scalar"]).reshape(-1)[0])
        scalar_source = "ndna_scalar (from file)"
    else:
        scalar = float(np.mean(ndna_n))
        scalar_source = "mean(nDNA_normalized)"

    return {
        "label": label,
        "path": path,
        "color": color,
        "layers": layers,
        "raw": {
            "kappa": kappa,
            "Delta": delta,
            "belief_norms": vnorm,
            "ndna": ndna_raw,
        },
        "norm": {
            "kappa": kappa_n,
            "Delta": delta_n,
            "belief_norms": vnorm_n,
            "ndna": ndna_n,
        },
        "ndna_scalar": scalar,
        "ndna_scalar_source": scalar_source,
    }


def build_overlay_fig(models: list[dict], normalized: bool) -> go.Figure:
    title_suffix = "Normalized" if normalized else "Raw"
    metrics = [
        ("kappa", "Spectral (κ)"),
        ("Delta", "Thermodynamic (Δ)"),
        ("belief_norms", "Belief norms (|v|)"),
        ("ndna", "Derived nDNA"),
    ]

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=[m[1] for m in metrics],
        vertical_spacing=0.14,
        horizontal_spacing=0.08,
    )

    for model in models:
        series = model["norm"] if normalized else model["raw"]
        showlegend = True  # Only show one legend entry per dataset.
        for idx, (key, label) in enumerate(metrics):
            row = idx // 2 + 1
            col = idx % 2 + 1
            fig.add_trace(
                go.Scatter(
                    x=model["layers"],
                    y=series[key],
                    mode="lines+markers",
                    name=model["label"],
                    showlegend=showlegend,
                    line=dict(color=model["color"], width=2),
                    marker=dict(size=5, color=model["color"]),
                    hovertemplate=(
                        f"<b>{model['label']}</b><br>"
                        "Layer=%{x}<br>"
                        f"{label}=%{{y:.5g}}<extra></extra>"
                    ),
                ),
                row=row,
                col=col,
            )
            showlegend = False

    fig.update_layout(
        title=f"Qwen 3.4B across datasets ({title_suffix})",
        height=900,
        legend_title_text="Dataset",
        hovermode="closest",
        margin=dict(l=40, r=20, t=80, b=40),
    )

    for idx, (_, label) in enumerate(metrics):
        row = idx // 2 + 1
        col = idx % 2 + 1
        fig.update_xaxes(title_text="Layer", row=row, col=col)
        fig.update_yaxes(title_text=label, row=row, col=col)

    return fig


def write_html(output_path: str, models: list[dict], figs: list[go.Figure]):
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("<html><head><meta charset='utf-8'><title>Qwen 3.4B Task Comparison</title>")
        f.write(
            "<style>"
            "body{font-family:Arial, sans-serif; margin:18px; background:#f7f7fb;}"
            ".card{background:white; padding:16px; margin-bottom:18px; border-radius:10px;"
            "box-shadow:0 2px 6px rgba(0,0,0,0.08);}"
            "table{border-collapse:collapse; width:100%; font-size:13px;}"
            "th,td{padding:8px 10px; border-bottom:1px solid #eee; text-align:left;}"
            "th{background:#fafafa;}"
            "</style></head><body>"
        )
        f.write("<h1>Qwen 3.4B: Cross-Dataset Metric Overlay</h1>")
        f.write("<p>This report is limited to the five provided runs; no other files are read.</p>")

        f.write("<div class='card'><h2>Scalar summary</h2>")
        f.write("<table><tr><th>Dataset</th><th>Layers</th><th>ndna_scalar</th><th>Source</th></tr>")
        for m in models:
            f.write(
                f"<tr><td style='color:{m['color']}; font-weight:700'>{m['label']}</td>"
                f"<td>{len(m['layers'])}</td>"
                f"<td>{m['ndna_scalar']:.6g}</td>"
                f"<td>{m['ndna_scalar_source']}</td></tr>"
            )
        f.write("</table></div>")

        for idx, fig in enumerate(figs):
            f.write("<div class='card'>")
            include_js = "cdn" if idx == 0 else False
            f.write(fig.to_html(full_html=False, include_plotlyjs=include_js))
            f.write("</div>")

        f.write("</body></html>")


def parse_args():
    parser = argparse.ArgumentParser(description="Plot the five provided Qwen 3.4B .npz runs only.")
    parser.add_argument(
        "--output",
        default="plots/qwen3_4b_task_overlay.html",
        help="HTML output path (directories will be created if needed).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    models = []

    for idx, (label, path) in enumerate(DATASETS):
        color = COLOR_PALETTE[idx % len(COLOR_PALETTE)]
        model = load_npz(label, path, color)
        models.append(model)
        print(f"Loaded {label}: {path} (layers={len(model['layers'])})")

    fig_raw = build_overlay_fig(models, normalized=False)
    fig_norm = build_overlay_fig(models, normalized=True)

    output_path = os.path.abspath(args.output)
    write_html(output_path, models, [fig_raw, fig_norm])
    print(f"Saved -> {output_path}")


if __name__ == "__main__":
    main()
