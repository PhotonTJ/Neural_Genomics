import argparse
import glob
import os
import colorsys
import hashlib

import numpy as np
import plotly.graph_objects as go
import similaritymeasures

# -----------------------------
# Colors and styling
# -----------------------------

# Base color mapping by model family (used as hue anchors)
FAMILY_COLORS = {
    "deepseek": "#d62728",
    "llama": "#1f77b4",
    "falcon": "#8c564b",
    "mistral": "#ff7f0e",
    "qwen": "#9467bd",
    "gemma": "#2ca02c",
    "opt": "#e377c2",
    "bloom": "#7f7f7f",
    "bert": "#bcbd22",
    "gpt": "#17becf",
    "yi": "#F1C40F",
    "phi": "#E67E22",
}
DEFAULT_COLOR = "#333333"
FAMILY_ORDER = list(FAMILY_COLORS.keys()) + ["other"]

# Fallback palette for models that don't match any family token
FALLBACK_MODEL_PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    "#393b79", "#637939", "#8c6d31", "#843c39", "#7b4173",
    "#3182bd", "#e6550d", "#31a354", "#756bb1", "#636363",
]

_MODEL_COLOR_CACHE: dict[str, str] = {}

# Dataset line styles for cross-dataset plots
DATASET_STYLES = {
    "ag-news": {"dash": "solid", "symbol": "circle"},
    "automathtext": {"dash": "dash", "symbol": "square"},
    "stanford_plato": {"dash": "dot", "symbol": "diamond"},
}

# Dataset outline colors (marker border) so datasets pop without changing model identity colors
DATASET_OUTLINE_COLORS = {
    "ag-news": "#1f77b4",
    "automathtext": "#ff7f0e",
    "stanford_plato": "#2ca02c",
}
DEFAULT_DATASET_OUTLINE = "#444444"


def get_family(model_name: str) -> str:
    n = model_name.lower()
    for fam in FAMILY_COLORS:
        if fam in n:
            return fam
    return "other"


def family_sort_key(model_name: str):
    fam = get_family(model_name)
    idx = FAMILY_ORDER.index(fam) if fam in FAMILY_ORDER else 10_000
    return (idx, model_name.lower())


def _stable_u01(key: str, salt: str) -> float:
    h = hashlib.md5((salt + key).encode("utf-8")).hexdigest()
    return int(h[:8], 16) / 0xFFFFFFFF


def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    hex_color = hex_color.lstrip("#")
    return (int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16))


def _rgb_to_hex(rgb: tuple[int, int, int]) -> str:
    r, g, b = rgb
    r = max(0, min(255, int(r)))
    g = max(0, min(255, int(g)))
    b = max(0, min(255, int(b)))
    return f"#{r:02x}{g:02x}{b:02x}"


def _vary_family_color(base_hex: str, key: str) -> str:
    """
    Deterministically vary a base family color so models within a family differ but stay related.
    """
    r, g, b = _hex_to_rgb(base_hex)
    h, l, s = colorsys.rgb_to_hls(r / 255.0, g / 255.0, b / 255.0)

    u_h = _stable_u01(key, "h")
    u_l = _stable_u01(key, "l")
    u_s = _stable_u01(key, "s")

    # small hue rotation + moderate lightness spread + mild saturation spread
    h = (h + (u_h - 0.5) * 0.14) % 1.0
    l = max(0.18, min(0.82, l + (u_l - 0.5) * 0.28))
    s = max(0.25, min(0.95, s * (0.75 + u_s * 0.5)))

    r2, g2, b2 = colorsys.hls_to_rgb(h, l, s)
    return _rgb_to_hex((round(r2 * 255), round(g2 * 255), round(b2 * 255)))


def get_model_color(model_base_name: str) -> str:
    """
    Stable, per-model color:
      - If family detected: vary that family's base color per model.
      - Else: pick from fallback palette (and vary slightly).
    """
    if model_base_name in _MODEL_COLOR_CACHE:
        return _MODEL_COLOR_CACHE[model_base_name]

    fam = get_family(model_base_name)
    if fam in FAMILY_COLORS:
        c = _vary_family_color(FAMILY_COLORS[fam], model_base_name)
    else:
        idx = int(_stable_u01(model_base_name, "p") * len(FALLBACK_MODEL_PALETTE)) % len(FALLBACK_MODEL_PALETTE)
        base = FALLBACK_MODEL_PALETTE[idx]
        c = _vary_family_color(base, model_base_name)

    _MODEL_COLOR_CACHE[model_base_name] = c
    return c


# -----------------------------
# Normalization / curve helpers
# -----------------------------

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


def resample_curve_by_arclength(curve: np.ndarray, n_samples: int) -> np.ndarray:
    curve = np.asarray(curve, dtype=float)
    if curve.ndim != 2 or curve.shape[0] < 2:
        return curve

    diffs = curve[1:] - curve[:-1]
    seglens = np.linalg.norm(diffs, axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seglens)])
    total = float(cum[-1])
    if total <= 0:
        return np.repeat(curve[:1], n_samples, axis=0)

    t_new = np.linspace(0.0, total, n_samples)
    out = np.zeros((n_samples, curve.shape[1]), dtype=float)
    for d in range(curve.shape[1]):
        out[:, d] = np.interp(t_new, cum, curve[:, d])
    return out


def ensure_same_length(a: np.ndarray, b: np.ndarray, n_samples: int = 128) -> tuple[np.ndarray, np.ndarray]:
    return resample_curve_by_arclength(a, n_samples), resample_curve_by_arclength(b, n_samples)


# -----------------------------
# Distance metrics
# -----------------------------

def dist_pcm(a: np.ndarray, b: np.ndarray) -> float:
    return float(similaritymeasures.pcm(a, b))


def dist_area(a: np.ndarray, b: np.ndarray) -> float:
    return float(similaritymeasures.area_between_two_curves(a, b))


def dist_curve_length(a: np.ndarray, b: np.ndarray) -> float:
    return float(similaritymeasures.curve_length_measure(a, b))


def dist_frechet(a: np.ndarray, b: np.ndarray) -> float:
    return float(similaritymeasures.frechet_dist(a, b))


def dist_dtw(a: np.ndarray, b: np.ndarray) -> float:
    r, _ = similaritymeasures.dtw(a, b)
    return float(r)


def dist_mae(a: np.ndarray, b: np.ndarray, n_samples: int = 128) -> float:
    a2, b2 = ensure_same_length(a, b, n_samples=n_samples)
    return float(similaritymeasures.mae(a2, b2))


def dist_mse(a: np.ndarray, b: np.ndarray, n_samples: int = 128) -> float:
    a2, b2 = ensure_same_length(a, b, n_samples=n_samples)
    return float(similaritymeasures.mse(a2, b2))


def safe_compute(dist_fn, a: np.ndarray, b: np.ndarray) -> float:
    try:
        if a is None or b is None:
            return np.nan
        a = np.asarray(a)
        b = np.asarray(b)
        if a.ndim != 2 or b.ndim != 2:
            return np.nan
        if a.shape[0] < 2 or b.shape[0] < 2:
            return np.nan
        return float(dist_fn(a, b))
    except Exception:
        return np.nan


# -----------------------------
# HTML matrix coloring helpers
# -----------------------------

def _interp_rgb(c1, c2, t: float):
    t = float(np.clip(t, 0.0, 1.0))
    return (
        int(round(c1[0] + (c2[0] - c1[0]) * t)),
        int(round(c1[1] + (c2[1] - c1[1]) * t)),
        int(round(c1[2] + (c2[2] - c1[2]) * t)),
    )


def _luminance(rgb):
    r, g, b = [x / 255.0 for x in rgb]
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _colormap_rdylgn(t: float):
    """
    Strong RdYlGn-like palette.
    low t=0 (similar) -> deep green
    mid t=0.5 -> yellow
    high t=1 (dissimilar) -> deep red
    """
    t = float(np.clip(t, 0.0, 1.0))
    green = (26, 152, 80)  # #1a9850
    yellow = (255, 255, 191)  # #ffffbf
    red = (215, 48, 39)  # #d73027
    if t < 0.5:
        return _interp_rgb(green, yellow, t / 0.5)
    return _interp_rgb(yellow, red, (t - 0.5) / 0.5)


def _bg_for_value(v: float, vmin: float, vmax: float, gamma: float = 0.7) -> str:
    """
    Strong filled color. Gamma < 1 increases contrast in the low range.
    """
    if np.isnan(v):
        return "background:#ffffff; color:#999;"

    if vmax <= vmin:
        t = 0.5
    else:
        t = (v - vmin) / (vmax - vmin)
        t = float(np.clip(t, 0.0, 1.0))

    t = t**gamma
    rgb = _colormap_rdylgn(t)
    text = "#111" if _luminance(rgb) > 0.55 else "#fff"
    return f"background: rgb({rgb[0]},{rgb[1]},{rgb[2]}); color:{text}; border-color:#ddd;"


def matrix_to_colored_html_table(
    names: list[str],
    mat: np.ndarray,
    fmt="{:.6g}",
    vmin: float | None = None,
    vmax: float | None = None,
) -> str:
    """
    One table: numbers + gradient background in each cell.
    vmin/vmax can be passed to force a consistent scale (we do per-method).
    """
    mat = np.asarray(mat, dtype=float)
    n = len(names)

    if vmin is None or vmax is None:
        finite = []
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                v = mat[i, j]
                if not np.isnan(v) and np.isfinite(v):
                    finite.append(v)
        if finite:
            vmin = float(np.min(finite))
            vmax = float(np.max(finite))
        else:
            vmin, vmax = 0.0, 1.0

    html = []
    html.append("<div class='scrollX'>")
    html.append("<table class='matrix'>")
    html.append("<tr>")
    html.append("<th class='sticky tl'>Model</th>")
    for name in names:
        html.append(f"<th class='sticky top rot'>{name}</th>")
    html.append("</tr>")

    for i in range(n):
        html.append("<tr>")
        html.append(f"<td class='sticky left rowhead'>{names[i]}</td>")
        for j in range(n):
            if i == j:
                html.append("<td class='diag'>—</td>")
            else:
                v = mat[i, j]
                if np.isnan(v):
                    html.append("<td class='na'>N/A</td>")
                else:
                    style = _bg_for_value(v, vmin, vmax)
                    html.append(f"<td style='{style}'>{fmt.format(v)}</td>")
        html.append("</tr>")

    html.append("</table>")
    html.append("</div>")
    return "\n".join(html)


# -----------------------------
# Data loading
# -----------------------------

def load_models(input_dir: str, global_normalize: bool = True) -> list[dict]:
    """
    Load models from npz files.

    Args:
        input_dir: Directory containing .npz files
        global_normalize: If True, normalize metrics globally across all models (recommended for comparisons).
                         If False, normalize per-model (still robust, but loses inter-model magnitude comparisons).
    """
    npz_files = sorted(glob.glob(os.path.join(input_dir, "*.npz")))

    # First pass: collect raw data from all models
    raw_models = []
    for path in npz_files:
        base = os.path.splitext(os.path.basename(path))[0]
        try:
            data = np.load(path)

            required = ("kappa", "Delta", "belief_norms")
            if any(k not in data for k in required):
                print(f"Skipping {base}: missing keys {required}")
                continue

            kappa = np.asarray(data["kappa"]).reshape(-1)
            delta = np.asarray(data["Delta"]).reshape(-1)
            vnorm = np.asarray(data["belief_norms"]).reshape(-1)

            L = min(len(kappa), len(delta), len(vnorm))
            if L < 2:
                continue

            scalar_val = 0.0
            if "ndna_scalar" in data:
                s = data["ndna_scalar"]
                scalar_val = float(s.item()) if isinstance(s, np.ndarray) else float(s)

            raw_models.append(
                {
                    "name": base,
                    "base_name": base,
                    "family": get_family(base),
                    "color": get_model_color(base),
                    "L": int(L),
                    "layers": np.arange(1, L + 1),
                    "kappa_raw": kappa[:L],
                    "delta_raw": delta[:L],
                    "vnorm_raw": vnorm[:L],
                    "scalar": scalar_val,
                }
            )

        except Exception as e:
            print(f"Error loading {base}: {e}")

    if not raw_models:
        return []

    # Second pass: normalize using robust_norm_to_range everywhere (global or per-model)
    if global_normalize:
        # Robust-normalize globally by normalizing concatenated vectors, then slicing back per model.
        all_kappa = np.concatenate([m["kappa_raw"] for m in raw_models])
        all_delta = np.concatenate([m["delta_raw"] for m in raw_models])
        all_vnorm = np.concatenate([m["vnorm_raw"] for m in raw_models])

        all_kappa_n = robust_norm_to_range(all_kappa)
        all_delta_n = robust_norm_to_range(all_delta)
        all_vnorm_n = robust_norm_to_range(all_vnorm)

    models = []
    cursor = 0
    for m in raw_models:
        L = m["L"]

        if global_normalize:
            kappa_n = all_kappa_n[cursor : cursor + L]
            delta_n = all_delta_n[cursor : cursor + L]
            vnorm_n = all_vnorm_n[cursor : cursor + L]
        else:
            # Per-model robust normalization (still robust_norm_to_range, but not globally comparable)
            kappa_n = robust_norm_to_range(m["kappa_raw"])
            delta_n = robust_norm_to_range(m["delta_raw"])
            vnorm_n = robust_norm_to_range(m["vnorm_raw"])

        cursor += L

        # Derived metric (also robust-normalized before plotting)
        ndna_n = kappa_n * vnorm_n / np.log(1.0 + delta_n)
        ndna_n = robust_norm_to_range(ndna_n)

        curves = {
            "spectral_thermo_2d": np.stack([kappa_n, delta_n], axis=1),
            "spectral_belief_2d": np.stack([kappa_n, vnorm_n], axis=1),
            "thermo_belief_2d": np.stack([delta_n, vnorm_n], axis=1),
            "spectral_thermo_belief_3d": np.stack([kappa_n, delta_n, vnorm_n], axis=1),
        }

        models.append(
            {
                "name": m["name"],
                "base_name": m["base_name"],
                "family": m["family"],
                "color": m["color"],
                "layers": m["layers"],
                "kappa": kappa_n,
                "delta": delta_n,
                "vnorm": vnorm_n,
                "kappa_raw": m["kappa_raw"],
                "delta_raw": m["delta_raw"],
                "vnorm_raw": m["vnorm_raw"],
                "ndna": ndna_n,
                "scalar": m["scalar"],
                "curves": curves,
            }
        )

    models.sort(key=lambda m: family_sort_key(m.get("base_name", m["name"])))
    return models


def load_models_raw(input_dir: str) -> list[dict]:
    """Load models with raw (unnormalized) metrics."""
    npz_files = sorted(glob.glob(os.path.join(input_dir, "*.npz")))
    models = []

    for path in npz_files:
        base = os.path.splitext(os.path.basename(path))[0]
        try:
            data = np.load(path)

            required = ("kappa", "Delta", "belief_norms")
            if any(k not in data for k in required):
                continue

            kappa = np.asarray(data["kappa"]).reshape(-1)
            delta = np.asarray(data["Delta"]).reshape(-1)
            vnorm = np.asarray(data["belief_norms"]).reshape(-1)

            L = min(len(kappa), len(delta), len(vnorm))
            if L < 2:
                continue

            models.append({
                "name": base,
                "base_name": base,
                "family": get_family(base),
                "color": get_model_color(base),
                "layers": np.arange(1, L + 1),
                "kappa_raw": kappa[:L],
                "delta_raw": delta[:L],
                "vnorm_raw": vnorm[:L],
            })

        except Exception as e:
            print(f"Error loading {base}: {e}")

    return models


def load_all_datasets(base_dir: str) -> dict[str, list[dict]]:
    """Load models from all dataset subdirectories with raw metrics."""
    datasets = {}

    for entry in os.listdir(base_dir):
        subdir = os.path.join(base_dir, entry)
        if os.path.isdir(subdir):
            npz_files = glob.glob(os.path.join(subdir, "*.npz"))
            if npz_files:
                models = load_models_raw(subdir)
                if models:
                    datasets[entry] = models
                    print(f"  Loaded {len(models)} models from dataset '{entry}'")

    return datasets


def load_all_datasets_normalized(base_dir: str, global_normalize: bool = True) -> tuple[list[dict], dict[str, list[dict]]]:
    """
    Load models from all dataset subdirectories.
    
    Args:
        base_dir: Base directory containing dataset subdirectories
        global_normalize: If True, normalize metrics globally across all models.
    
    Returns:
        - combined_models: all models merged (for existing dashboard features)
        - datasets: dict mapping dataset name -> list of raw models (for cross-dataset plots)
    """
    combined_models = []
    datasets = {}

    for entry in sorted(os.listdir(base_dir)):
        subdir = os.path.join(base_dir, entry)
        if os.path.isdir(subdir):
            npz_files = glob.glob(os.path.join(subdir, "*.npz"))
            if npz_files:
                normalized = load_models(subdir, global_normalize=global_normalize)
                for m in normalized:
                    m["dataset"] = entry
                    # keep base_name stable for sorting/color; only change displayed name
                    m["name"] = f"{m['name']} ({entry})"
                combined_models.extend(normalized)

                raw = load_models_raw(subdir)
                if raw:
                    datasets[entry] = raw
                    print(f"  Loaded {len(raw)} models from dataset '{entry}'")

    combined_models.sort(key=lambda m: family_sort_key(m.get("base_name", m["name"])))
    return combined_models, datasets


# -----------------------------
# Cross-dataset figures
# -----------------------------

def build_cross_dataset_metric_figures(datasets: dict[str, list[dict]]) -> tuple[go.Figure, go.Figure, go.Figure]:
    """Build 3 plots (one per metric) with all models across all datasets overlayed.
    Metrics are robust-normalized to [0.005, 1.0] (globally across all datasets) before plotting.
    """

    metrics = [
        ("kappa_raw", "Spectral (κ)", "Cross-Dataset: Spectral (κ) - Robust Normalized"),
        ("delta_raw", "Thermodynamic (Δ)", "Cross-Dataset: Thermodynamic (Δ) - Robust Normalized"),
        ("vnorm_raw", "Belief Norms (|v|)", "Cross-Dataset: Belief Norms (|v|) - Robust Normalized"),
    ]

    figs = []

    for metric_key, metric_label, title in metrics:
        fig = go.Figure()

        # Flatten in deterministic order for global robust normalization
        flat = []
        for dataset_name, models in sorted(datasets.items()):
            for model in models:
                y = np.asarray(model.get(metric_key, []), dtype=float).reshape(-1)
                if y.size >= 2:
                    flat.append((dataset_name, model, y))

        if not flat:
            figs.append(fig)
            continue

        all_y = np.concatenate([y for (_, _, y) in flat])
        all_y_n = robust_norm_to_range(all_y)

        cursor = 0
        for dataset_name, model, y in flat:
            L = y.size
            y_n = all_y_n[cursor : cursor + L]
            cursor += L

            style = DATASET_STYLES.get(dataset_name, {"dash": "solid", "symbol": "circle"})
            outline = DATASET_OUTLINE_COLORS.get(dataset_name, DEFAULT_DATASET_OUTLINE)

            trace_name = f"{model['name']} ({dataset_name})"

            fig.add_trace(
                go.Scatter(
                    x=model["layers"][:L],
                    y=y_n,
                    mode="lines+markers",
                    name=trace_name,
                    line=dict(
                        color=model["color"],
                        dash=style["dash"],
                        width=2,
                    ),
                    marker=dict(
                        size=5,
                        color=model["color"],
                        symbol=style["symbol"],
                        line=dict(width=1, color=outline),
                    ),
                    legendgroup=f"{model['family']}_{dataset_name}",
                    hovertemplate=(
                        f"<b>{trace_name}</b><br>"
                        f"Layer: %{{x}}<br>"
                        f"{metric_label} (robust-norm): %{{y:.4f}}"
                        "<extra></extra>"
                    ),
                )
            )

        legend_parts = []
        for ds_name in sorted(datasets.keys()):
            style = DATASET_STYLES.get(ds_name, {"dash": "solid", "symbol": "circle"})
            dash_display = {"solid": "━━━", "dash": "- - -", "dot": "···"}.get(style["dash"], "━━━")
            symbol_display = {"circle": "●", "square": "■", "diamond": "◆"}.get(style["symbol"], "●")
            legend_parts.append(f"{dash_display} {ds_name} ({symbol_display})")

        fig.update_layout(
            title=dict(text=f"<b>{title}</b>", x=0.5, xanchor="center"),
            xaxis_title="Layer",
            yaxis_title=f"{metric_label} (robust normalized)",
            height=600,
            hovermode="closest",
            legend=dict(font=dict(size=9)),
        )

        fig.add_annotation(
            text=f"<b>Dataset Styles:</b> {' │ '.join(legend_parts)}",
            xref="paper",
            yref="paper",
            x=0.5,
            y=1.08,
            xanchor="center",
            yanchor="bottom",
            showarrow=False,
            font=dict(size=10),
        )

        figs.append(fig)

    return tuple(figs)


# -----------------------------
# Similarity matrices
# -----------------------------

def compute_distance_matrices(models: list[dict], resample_n: int = 128):
    names = [m["name"] for m in models]
    n = len(models)

    methods = [
        ("PCM (2D only)", dist_pcm, False, True),
        ("Area Between Curves (2D only)", dist_area, False, True),
        ("Curve Length Measure (2D only)", dist_curve_length, False, True),
        ("Discrete Fréchet", dist_frechet, True, False),
        ("DTW", dist_dtw, True, False),
        ("MAE (resampled)", lambda a, b: dist_mae(a, b, n_samples=resample_n), True, False),
        ("MSE (resampled)", lambda a, b: dist_mse(a, b, n_samples=resample_n), True, False),
    ]

    views = [
        ("(spectral, thermo)", "spectral_thermo_2d", 2),
        ("(spectral, belief)", "spectral_belief_2d", 2),
        ("(thermo, belief)", "thermo_belief_2d", 2),
        ("(spectral, thermo, belief)", "spectral_thermo_belief_3d", 3),
    ]

    results = {}

    for method_name, fn, supports_3d, needs_2d in methods:
        results[method_name] = {}
        for view_label, view_key, dim in views:
            mat = np.full((n, n), np.nan, dtype=float)
            np.fill_diagonal(mat, 0.0)

            for i in range(n):
                for j in range(i + 1, n):
                    a = models[i]["curves"][view_key]
                    b = models[j]["curves"][view_key]

                    if needs_2d and dim != 2:
                        d = np.nan
                    elif (not supports_3d) and dim == 3:
                        d = np.nan
                    else:
                        d = safe_compute(fn, a, b)

                    mat[i, j] = d
                    mat[j, i] = d

            results[method_name][view_key] = {"label": view_label, "dim": dim, "matrix": mat}

    return names, results


# -----------------------------
# Combined figures
# -----------------------------

def build_combined_figures(models: list[dict]):
    fig_3d = go.Figure()
    for m in models:
        fig_3d.add_trace(
            go.Scatter3d(
                x=m["kappa"],
                y=m["delta"],
                z=m["vnorm"],
                mode="lines+markers",
                name=m["name"],
                marker=dict(size=3, color=m["color"]),
                line=dict(width=4, color=m["color"]),
                text=m["layers"],
                hovertemplate=(
                    f"<b>{m['name']}</b><br>"
                    "Layer %{text}<br>"
                    "κ=%{x:.3f}<br>"
                    "Δ=%{y:.3f}<br>"
                    "|v|=%{z:.4f}"
                    "<extra></extra>"
                ),
            )
        )
    fig_3d.update_layout(
        title="Combined: Spectral vs Thermo vs Belief (normalized)",
        scene=dict(xaxis_title="Spectral (κ)", yaxis_title="Thermo (Δ)", zaxis_title="Belief (|v|)"),
        height=750,
        margin=dict(l=0, r=0, t=60, b=0),
    )

    def combined_2d(xkey, ykey, title, xlabel, ylabel):
        fig = go.Figure()
        for m in models:
            fig.add_trace(
                go.Scatter(
                    x=m[xkey],
                    y=m[ykey],
                    mode="lines+markers",
                    name=m["name"],
                    line=dict(width=2, color=m["color"]),
                    marker=dict(size=4, color=m["color"]),
                    hovertemplate=f"<b>{m['name']}</b><br>{xlabel}=%{{x:.4f}}<br>{ylabel}=%{{y:.4f}}<extra></extra>",
                )
            )
        fig.update_layout(
            title=title,
            xaxis_title=xlabel,
            yaxis_title=ylabel,
            height=550,
            margin=dict(l=10, r=10, t=60, b=10),
        )
        return fig

    fig_st = combined_2d("kappa", "delta", "Combined 2D: (spectral, thermo)", "Spectral (κ)", "Thermo (Δ)")
    fig_sb = combined_2d("kappa", "vnorm", "Combined 2D: (spectral, belief)", "Spectral (κ)", "Belief (|v|)")
    fig_tb = combined_2d("delta", "vnorm", "Combined 2D: (thermo, belief)", "Thermo (Δ)", "Belief (|v|)")

    fig_ndna = go.Figure()
    for m in models:
        fig_ndna.add_trace(
            go.Scatter(
                x=m["layers"],
                y=m["ndna"],
                mode="lines+markers",
                name=m["name"],
                line=dict(width=2, color=m["color"]),
                marker=dict(size=4, color=m["color"]),
                hovertemplate=f"<b>{m['name']}</b><br>Layer=%{{x}}<br>nDNA=%{{y:.4f}}<extra></extra>",
            )
        )
    fig_ndna.update_layout(
        title="Layerwise nDNA (derived metric, normalized)",
        xaxis_title="Layer",
        yaxis_title="nDNA",
        height=550,
        margin=dict(l=10, r=10, t=60, b=10),
    )

    fig_layer_kappa = go.Figure()
    fig_layer_delta = go.Figure()
    fig_layer_vnorm = go.Figure()

    for m in models:
        fig_layer_kappa.add_trace(
            go.Scatter(
                x=m["layers"],
                y=m["kappa"],
                mode="lines+markers",
                name=m["name"],
                line=dict(width=2, color=m["color"]),
                marker=dict(size=4, color=m["color"]),
                hovertemplate=f"<b>{m['name']}</b><br>Layer=%{{x}}<br>κ=%{{y:.4f}}<extra></extra>",
            )
        )
        fig_layer_delta.add_trace(
            go.Scatter(
                x=m["layers"],
                y=m["delta"],
                mode="lines+markers",
                name=m["name"],
                line=dict(width=2, color=m["color"]),
                marker=dict(size=4, color=m["color"]),
                hovertemplate=f"<b>{m['name']}</b><br>Layer=%{{x}}<br>Δ=%{{y:.4f}}<extra></extra>",
            )
        )
        fig_layer_vnorm.add_trace(
            go.Scatter(
                x=m["layers"],
                y=m["vnorm"],
                mode="lines+markers",
                name=m["name"],
                line=dict(width=2, color=m["color"]),
                marker=dict(size=4, color=m["color"]),
                hovertemplate=f"<b>{m['name']}</b><br>Layer=%{{x}}<br>|v|=%{{y:.4f}}<extra></extra>",
            )
        )

    fig_layer_kappa.update_layout(
        title="Layerwise spectral (κ)",
        xaxis_title="Layer",
        yaxis_title="Spectral (κ)",
        height=500,
        margin=dict(l=10, r=10, t=60, b=10),
    )
    fig_layer_delta.update_layout(
        title="Layerwise thermo (Δ)",
        xaxis_title="Layer",
        yaxis_title="Thermo (Δ)",
        height=500,
        margin=dict(l=10, r=10, t=60, b=10),
    )
    fig_layer_vnorm.update_layout(
        title="Layerwise belief (|v|)",
        xaxis_title="Layer",
        yaxis_title="Belief (|v|)",
        height=500,
        margin=dict(l=10, r=10, t=60, b=10),
    )

    return (
        fig_3d,
        fig_st,
        fig_sb,
        fig_tb,
        fig_ndna,
        fig_layer_kappa,
        fig_layer_delta,
        fig_layer_vnorm,
    )


def build_dashboard_combined_figures(models: list[dict]):
    fig_1 = go.Figure()
    fig_2 = go.Figure()
    fig_3 = go.Figure()
    fig_4 = go.Figure()
    fig_5 = go.Figure()

    for m in models:
        fig_1.add_trace(
            go.Scatter3d(
                x=m["kappa"],
                y=m["delta"],
                z=m["vnorm"],
                mode="lines+markers",
                name=m["name"],
                marker=dict(size=3, color=m["color"]),
                line=dict(width=4, color=m["color"]),
                text=m["layers"],
                hovertemplate=f"<b>{m['name']}</b><br>Layer %{{text}}<br>κ=%{{x:.2f}}<br>Δ=%{{y:.2f}}<br>|v|=%{{z:.4f}}<extra></extra>",
            )
        )

        fig_2.add_trace(
            go.Scatter3d(
                x=m["kappa"],
                y=m["delta"],
                z=m["layers"],
                mode="lines+markers",
                name=m["name"],
                marker=dict(size=3, color=m["color"]),
                line=dict(width=4, color=m["color"]),
                text=m["layers"],
                hovertemplate=f"<b>{m['name']}</b><br>Layer %{{z}}<br>κ=%{{x:.2f}}<br>Δ=%{{y:.2f}}<extra></extra>",
            )
        )

        fig_3.add_trace(
            go.Scatter3d(
                x=m["delta"],
                y=m["vnorm"],
                z=m["layers"],
                mode="lines+markers",
                name=m["name"],
                marker=dict(size=3, color=m["color"]),
                line=dict(width=4, color=m["color"]),
                text=m["layers"],
                hovertemplate=f"<b>{m['name']}</b><br>Layer %{{z}}<br>Δ=%{{x:.2f}}<br>|v|=%{{y:.4f}}<extra></extra>",
            )
        )

        fig_4.add_trace(
            go.Scatter3d(
                x=m["kappa"],
                y=m["vnorm"],
                z=m["layers"],
                mode="lines+markers",
                name=m["name"],
                marker=dict(size=3, color=m["color"]),
                line=dict(width=4, color=m["color"]),
                text=m["layers"],
                hovertemplate=f"<b>{m['name']}</b><br>Layer %{{z}}<br>κ=%{{x:.2f}}<br>|v|=%{{y:.4f}}<extra></extra>",
            )
        )

        fig_5.add_trace(
            go.Scatter(
                x=m["layers"],
                y=m["ndna"],
                mode="lines+markers",
                name=m["name"],
                line=dict(width=2, color=m["color"]),
                marker=dict(size=4, color=m["color"]),
                hovertemplate=f"<b>{m['name']}</b><br>Layer %{{x}}<br>nDNA=%{{y:.4f}}<extra></extra>",
            )
        )

    fig_1.update_layout(
        title="<b>1. Spectral vs Thermo vs Belief</b>",
        scene=dict(xaxis_title="Spectral", yaxis_title="Thermo", zaxis_title="Belief"),
        height=700,
    )
    fig_2.update_layout(
        title="<b>2. Spectral vs Thermo vs Layer</b>",
        # scene=dict(xaxis_title="Spectral (κ)", yaxis_title="Thermo (Δ)", zaxis_title="Layer", zaxis=dict(autorange="reversed")),
        scene=dict(xaxis_title="Spectral (κ)", yaxis_title="Thermo (Δ)", zaxis_title="Layer"),
        height=700,
    )
    fig_3.update_layout(
        title="<b>3. Thermo vs Belief vs Layer</b>",
        # scene=dict(xaxis_title="Thermo (Δ)", yaxis_title="Belief (|v|)", zaxis_title="Layer", zaxis=dict(autorange="reversed")),
        scene=dict(xaxis_title="Thermo (Δ)", yaxis_title="Belief (|v|)", zaxis_title="Layer"),
        height=700,
    )
    fig_4.update_layout(
        title="<b>4. Spectral vs Belief vs Layer</b>",
        # scene=dict(xaxis_title="Spectral (κ)", yaxis_title="Belief (|v|)", zaxis_title="Layer", zaxis=dict(autorange="reversed")),
        scene=dict(xaxis_title="Spectral (κ)", yaxis_title="Belief (|v|)", zaxis_title="Layer"),
        height=700,
    )
    fig_5.update_layout(title="<b>5. Layerwise nDNA Scores</b>", height=500)

    return fig_1, fig_2, fig_3, fig_4, fig_5


# -----------------------------
# Dashboard writers
# -----------------------------

def write_combined_dashboard(
    output_path: str,
    models: list[dict],
    figs: tuple[go.Figure, ...],
    cross_dataset_figs: tuple[go.Figure, ...] | None = None,
    datasets: dict[str, list[dict]] | None = None
):
    fig_1, fig_2, fig_3, fig_4, fig_5 = figs
    models_sorted = sorted(models, key=lambda m: m["scalar"], reverse=True)

    table_rows = ""
    for item in models_sorted:
        table_rows += (
            "<tr>"
            f"<td style='padding:8px; border-bottom:1px solid #ddd; color:{item['color']}; font-weight:bold'>{item['name']}</td>"
            f"<td style='padding:8px; border-bottom:1px solid #ddd'>{item['scalar']:.4e}</td>"
            "</tr>"
        )

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(
            "<html><head><title>Combined Analysis</title>"
            "<style>"
            "body{font-family:sans-serif;margin:20px;background:#f4f4f9}"
            ".card{background:white;padding:20px;margin:20px 0;border-radius:8px;box-shadow:0 2px 5px rgba(0,0,0,0.1)}"
            "table{width:100%;border-collapse:collapse}"
            ".legend-note{background:#f0f8ff;padding:12px;border-radius:6px;margin:10px 0;font-size:13px}"
            "</style></head><body>"
        )
        f.write("<h1>Combined Model Analysis</h1>")
        f.write(
            f"<div class='card'><h2>Total nDNA Scalar Scores</h2>"
            f"<table><tr><th style='text-align:left;padding:8px'>Model</th><th style='text-align:left;padding:8px'>Score</th></tr>{table_rows}</table></div>"
        )
        f.write(f"<div class='card'>{fig_1.to_html(full_html=False, include_plotlyjs='cdn')}</div>")
        f.write(f"<div class='card'>{fig_2.to_html(full_html=False, include_plotlyjs=False)}</div>")
        f.write(f"<div class='card'>{fig_3.to_html(full_html=False, include_plotlyjs=False)}</div>")
        f.write(f"<div class='card'>{fig_4.to_html(full_html=False, include_plotlyjs=False)}</div>")
        f.write(f"<div class='card'>{fig_5.to_html(full_html=False, include_plotlyjs=False)}</div>")

        if cross_dataset_figs and datasets:
            legend_parts = []
            for ds_name in sorted(datasets.keys()):
                style = DATASET_STYLES.get(ds_name, {"dash": "solid", "symbol": "circle"})
                dash_display = {"solid": "━━━", "dash": "- - -", "dot": "···"}.get(style["dash"], "━━━")
                symbol_display = {"circle": "●", "square": "■", "diamond": "◆"}.get(style["symbol"], "●")
                legend_parts.append(f"{dash_display} {ds_name} ({symbol_display})")

            f.write("<div class='card'><h2>Cross-Dataset Comparison (Unnormalized Metrics)</h2>")
            f.write(f"<div class='legend-note'><b>Line Styles by Dataset:</b> {' &nbsp;│&nbsp; '.join(legend_parts)}")
            f.write("</div></div>")

            for fig in cross_dataset_figs:
                f.write(f"<div class='card'>{fig.to_html(full_html=False, include_plotlyjs=False)}</div>")

        f.write("</body></html>")


def render_combined_dashboard_section(
    models: list[dict],
    figs: tuple[go.Figure, ...],
    cross_dataset_figs: tuple[go.Figure, ...] | None = None,
    datasets: dict[str, list[dict]] | None = None,
) -> str:
    """
    Render the combined dashboard (formerly ALL_MODELS_trajectory.html) as an
    HTML snippet so it can be embedded into the main report.
    """
    fig_1, fig_2, fig_3, fig_4, fig_5 = figs
    models_sorted = sorted(models, key=lambda m: m["scalar"], reverse=True)

    table_rows = []
    for item in models_sorted:
        table_rows.append(
            "<tr>"
            f"<td style='color:{item['color']}; font-weight:bold'>{item['name']}</td>"
            f"<td>{item['scalar']:.4e}</td>"
            "</tr>"
        )

    parts: list[str] = []
    parts.append("<div class='figure-block'>")
    parts.append("<h3>Dashboard: Total nDNA Scalar Scores</h3>")
    parts.append("<table class='simple'>")
    parts.append("<tr><th>Model</th><th>Score</th></tr>")
    parts.extend(table_rows)
    parts.append("</table></div>")

    parts.append("<div class='figure-block'>")
    parts.append(fig_2.to_html(full_html=False, include_plotlyjs=False))
    parts.append("</div>")
    parts.append("<div class='figure-block'>")
    parts.append(fig_3.to_html(full_html=False, include_plotlyjs=False))
    parts.append("</div>")
    parts.append("<div class='figure-block'>")
    parts.append(fig_4.to_html(full_html=False, include_plotlyjs=False))
    parts.append("</div>")

    if cross_dataset_figs and datasets:
        legend_parts = []
        for ds_name in sorted(datasets.keys()):
            style = DATASET_STYLES.get(ds_name, {"dash": "solid", "symbol": "circle"})
            dash_display = {"solid": "━━━", "dash": "- - -", "dot": "···"}.get(style["dash"], "━━━")
            symbol_display = {"circle": "●", "square": "■", "diamond": "◆"}.get(style["symbol"], "●")
            legend_parts.append(f"{dash_display} {ds_name} ({symbol_display})")

        parts.append(
            "<div class='figure-block' style='background:#f0f8ff;'>"
            "<h4>Cross-Dataset Comparison (Unnormalized Metrics)</h4>"
            f"<div style='font-size:13px;'>{' &nbsp;│&nbsp; '.join(legend_parts)}</div>"
            "</div>"
        )

        for fig in cross_dataset_figs:
            parts.append("<div class='figure-block'>")
            parts.append(fig.to_html(full_html=False, include_plotlyjs=False))
            parts.append("</div>")

    return "".join(parts)


def write_model_dashboards(output_dir: str, models: list[dict]):
    for m in models:
        layers = m["layers"]
        kappa = m["kappa"]
        delta = m["delta"]
        vnorm = m["vnorm"]
        ndna = m["ndna"]
        scalar = m["scalar"]
        name = m["name"]

        fig1 = go.Figure(
            data=[
                go.Scatter3d(
                    x=kappa,
                    y=delta,
                    z=vnorm,
                    mode="lines+markers",
                    marker=dict(size=5, color=layers, colorscale="Viridis", colorbar=dict(title="Layer"), opacity=0.8),
                    line=dict(color="rgba(50, 50, 50, 0.5)", width=3),
                    text=[f"Layer {l}<br>κ={k:.2f}<br>Δ={d:.2f}<br>|v|={v:.4f}" for l, k, d, v in zip(layers, kappa, delta, vnorm)],
                    hovertemplate="%{text}<extra></extra>",
                )
            ]
        )
        fig1.update_layout(
            title="<b>3D Trajectory</b>: Spectral vs Thermo vs Belief",
            scene=dict(xaxis_title="Spectral (κ)", yaxis_title="Thermo (Δ)", zaxis_title="Belief (|v|)"),
            margin=dict(l=0, r=0, b=0, t=40),
            height=500,
        )

        fig2 = go.Figure(
            data=[
                go.Scatter3d(
                    x=kappa,
                    y=delta,
                    z=layers,
                    mode="lines+markers",
                    marker=dict(size=5, color=vnorm, colorscale="Plasma", colorbar=dict(title="Belief (|v|)"), opacity=0.8),
                    line=dict(color="rgba(100, 0, 150, 0.5)", width=3),
                    text=[f"Layer {l}<br>κ={k:.2f}<br>Δ={d:.2f}<br>|v|={v:.4f}" for l, k, d, v in zip(layers, kappa, delta, vnorm)],
                    hovertemplate="%{text}<extra></extra>",
                )
            ]
        )
        fig2.update_layout(
            title="<b>3D Evolution</b>: Spectral vs Thermo vs Layer",
            # scene=dict(xaxis_title="Spectral (κ)", yaxis_title="Thermo (Δ)", zaxis_title="Layer", zaxis=dict(autorange="reversed")),
            scene=dict(xaxis_title="Spectral (κ)", yaxis_title="Thermo (Δ)", zaxis_title="Layer"),
            margin=dict(l=0, r=0, b=0, t=40),
            height=500,
        )

        fig3 = go.Figure(data=go.Scatter(x=layers, y=kappa, mode="lines+markers", name="Spectral", line=dict(color="blue")))
        fig3.update_layout(title="Spectral (κ)", height=350, margin=dict(l=0, r=0, b=0, t=40))

        fig4 = go.Figure(data=go.Scatter(x=layers, y=delta, mode="lines+markers", name="Thermo", line=dict(color="red")))
        fig4.update_layout(title="Thermo (Δ)", height=350, margin=dict(l=0, r=0, b=0, t=40))

        fig5 = go.Figure(data=go.Scatter(x=layers, y=vnorm, mode="lines+markers", name="Belief", line=dict(color="green")))
        fig5.update_layout(title="Belief (|v|)", height=350, margin=dict(l=0, r=0, b=0, t=40))

        fig6 = go.Figure(data=go.Scatter(x=layers, y=ndna, mode="lines+markers", name="nDNA", line=dict(color="purple")))
        fig6.update_layout(title="nDNA (Layerwise)", height=350, margin=dict(l=0, r=0, b=0, t=40))

        output_path = os.path.join(output_dir, f"{name}_dashboard.html")
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(f"<html><head><title>{name}</title>")
            f.write(
                "<style>"
                "body { font-family: sans-serif; margin: 20px; background: #f4f4f9; }"
                ".card { background: white; padding: 15px; margin-bottom: 20px; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }"
                "</style></head><body>"
            )
            f.write(f"<h1>Report: {name}</h1>")
            f.write(f"<div class='card'><h2>Total nDNA Scalar Score: {scalar:.4e}</h2></div>")
            f.write("<div class='card'><h3>3D Visualizations</h3>")
            f.write(fig1.to_html(full_html=False, include_plotlyjs="cdn"))
            f.write(fig2.to_html(full_html=False, include_plotlyjs=False))
            f.write("</div>")
            f.write("<div class='card'><h3>2D Metrics</h3>")
            f.write(fig3.to_html(full_html=False, include_plotlyjs=False))
            f.write(fig4.to_html(full_html=False, include_plotlyjs=False))
            f.write(fig5.to_html(full_html=False, include_plotlyjs=False))
            f.write(fig6.to_html(full_html=False, include_plotlyjs=False))
            f.write("</div></body></html>")

        print(f"Saved -> {output_path}")


# -----------------------------
# Report HTML
# -----------------------------

def build_report_html(
    output_path: str,
    models: list[dict],
    names: list[str],
    matrices: dict,
    fig_3d: go.Figure,
    fig_st: go.Figure,
    fig_sb: go.Figure,
    fig_tb: go.Figure,
    fig_ndna: go.Figure,
    fig_layer_kappa: go.Figure,
    fig_layer_delta: go.Figure,
    fig_layer_vnorm: go.Figure,
    dashboard_section_html: str | None = None,
):
    css = """
    <style>
      body { font-family: sans-serif; margin: 20px; background: #f4f4f9; }
      h1, h2, h3 { margin: 0 0 12px 0; }
      .card { background: white; padding: 18px; margin: 18px 0; border-radius: 10px;
              box-shadow: 0 2px 6px rgba(0,0,0,0.08); }
      .note { color:#555; line-height:1.35; }
      .grid { display: grid; grid-template-columns: 1fr; gap: 16px; }
      @media (min-width: 1200px) { .grid2 { grid-template-columns: 1fr 1fr; } }
      .figure-block { margin: 16px 0; padding: 12px; border-radius: 10px; background: #fff;
                      box-shadow: 0 2px 6px rgba(0,0,0,0.06); }
      details { background:#fff; border-radius:10px; }
      details > summary { cursor: pointer; padding: 12px; font-weight: 800; }
      details .inner { padding: 0 12px 12px 12px; }
      .scrollX { overflow-x: auto; max-width: 100%; }
      table.simple { width:100%; border-collapse: collapse; font-size: 13px; }
      table.simple th { background:#fafafa; border-bottom: 1px solid #eee; padding:8px; text-align:left; }
      table.simple td { border-bottom: 1px solid #eee; padding:8px; }
      table.matrix { border-collapse: collapse; font-size: 12px; min-width: 900px; }
      table.matrix th, table.matrix td { border: 1px solid #ddd; padding: 6px; text-align:center; }
      table.matrix th { background: #f0f0f0; }
      .sticky { position: sticky; background: #f0f0f0; z-index: 2; }
      .top { top: 0; z-index: 3; }
      .left { left: 0; z-index: 3; text-align:left !important; }
      .tl { top:0; left:0; z-index: 5; }
      .rowhead { font-weight:800; }
      .rot { writing-mode: vertical-rl; transform: rotate(180deg); min-width: 28px; }
      .diag { background:#eee; color:#333; font-weight:800; }
      .na { background:#fff; color:#999; }
      .small { font-size: 13px; color:#666; line-height: 1.4; }
      .hr { height:1px; background:#eee; margin: 12px 0; }
      .pill { display:inline-block; padding:3px 10px; border-radius:999px; font-size:12px; background:#f3f3f3; margin-right:6px; }
    </style>
    """

    grouped = {}
    for m in models:
        grouped.setdefault(m["family"], []).append(m)
    for fam in grouped:
        grouped[fam].sort(key=lambda x: x["scalar"], reverse=True)

    scalar_html = []
    scalar_html.append("<div class='small'>Models are grouped by family, then sorted by ndna_scalar within each family.</div>")
    for fam in FAMILY_ORDER:
        if fam not in grouped:
            continue
        scalar_html.append("<div class='hr'></div>")
        scalar_html.append(f"<h3><span class='pill'>{fam}</span></h3>")
        scalar_html.append("<table class='simple'>")
        scalar_html.append("<tr><th>Model</th><th>ndna_scalar</th></tr>")
        for m in grouped[fam]:
            scalar_html.append(
                f"<tr>"
                f"<td style='color:{m['color']}; font-weight:800'>{m['name']}</td>"
                f"<td>{m['scalar']:.6e}</td>"
                f"</tr>"
            )
        scalar_html.append("</table>")

    view_order = [
        ("spectral_thermo_2d", "(spectral, thermo)"),
        ("spectral_belief_2d", "(spectral, belief)"),
        ("thermo_belief_2d", "(thermo, belief)"),
        ("spectral_thermo_belief_3d", "(spectral, thermo, belief)"),
    ]

    parts = []
    parts.append("<html><head><meta charset='utf-8'><title>LLM Curve Similarity Report</title>")
    parts.append(css)
    parts.append("</head><body>")
    parts.append("<h1>LLM Curve Similarity Report</h1>")
    parts.append(
        "<p class='note'>Each matrix cell contains the numeric distance and is filled with a strong green→yellow→red gradient "
        "(green = similar, red = dissimilar). Lower distance means more similar. "
        "Color scale is consistent across the 4 views within each method. "
        "PCM/Area/Curve-Length are 2D-only in this library, so 3D entries are N/A.</p>"
    )

    parts.append("<div class='card'>")
    parts.append("<h2>1) Scalar summary (ndna_scalar)</h2>")
    parts.append("\n".join(scalar_html))
    parts.append("</div>")

    parts.append("<div class='card'>")
    parts.append("<h2>2) Combined visual comparison</h2>")
    parts.append("<div class='figure-block'>")
    parts.append(fig_3d.to_html(full_html=False, include_plotlyjs="cdn"))
    parts.append("</div>")

    parts.append("<div class='figure-block'>")
    parts.append(fig_st.to_html(full_html=False, include_plotlyjs=False))
    parts.append("</div>")

    parts.append("<div class='figure-block'>")
    parts.append(fig_sb.to_html(full_html=False, include_plotlyjs=False))
    parts.append("</div>")

    parts.append("<div class='figure-block'>")
    parts.append(fig_tb.to_html(full_html=False, include_plotlyjs=False))
    parts.append("</div>")

    parts.append("<div class='hr'></div>")
    parts.append("<h3>Layerwise (per-layer) trajectories</h3>")

    parts.append("<div class='figure-block'>")
    parts.append(fig_ndna.to_html(full_html=False, include_plotlyjs=False))
    parts.append("</div>")

    parts.append("<div class='figure-block'>")
    parts.append(fig_layer_kappa.to_html(full_html=False, include_plotlyjs=False))
    parts.append("</div>")

    parts.append("<div class='figure-block'>")
    parts.append(fig_layer_delta.to_html(full_html=False, include_plotlyjs=False))
    parts.append("</div>")

    parts.append("<div class='figure-block'>")
    parts.append(fig_layer_vnorm.to_html(full_html=False, include_plotlyjs=False))
    parts.append("</div>")

    parts.append("</div>")

    parts.append("<div class='card'>")
    parts.append("<h2>3) Pairwise similarity matrices (numbers + filled gradient)</h2>")
    parts.append("<p class='small'>Open a method, then a view. Each view shows a single colorized numeric matrix.</p>")
    parts.append("<div class='hr'></div>")

    for method_name, view_dict in matrices.items():
        vals = []
        for view_key, _ in view_order:
            mat = view_dict[view_key]["matrix"]
            n = mat.shape[0]
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    v = mat[i, j]
                    if not np.isnan(v) and np.isfinite(v):
                        vals.append(v)
        method_vmin = float(np.min(vals)) if vals else 0.0
        method_vmax = float(np.max(vals)) if vals else 1.0

        parts.append("<details>")
        parts.append(f"<summary>{method_name}</summary>")
        parts.append("<div class='inner'>")

        for view_key, view_label in view_order:
            mat = view_dict[view_key]["matrix"]
            parts.append("<details>")
            parts.append(f"<summary>{view_label}</summary>")
            parts.append("<div class='inner'>")
            parts.append(
                matrix_to_colored_html_table(
                    names,
                    mat,
                    fmt="{:.6g}",
                    vmin=method_vmin,
                    vmax=method_vmax,
                )
            )
            parts.append("</div></details>")

        parts.append("</div></details>")

    parts.append("</div>")

    if dashboard_section_html:
        parts.append("<div class='card'>")
        parts.append("<h2>4) Combined dashboard (ALL_MODELS_trajectory)</h2>")
        parts.append(dashboard_section_html)
        parts.append("</div>")

    parts.append("</body></html>")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))


# -----------------------------
# CLI / main
# -----------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Generate nDNA visualizations and similarity reports.")
    parser.add_argument(
        "--input-dir",
        default=os.environ.get("NDNA_INPUT_DIR", "/method5_generic/squad"),
        help="Directory containing .npz outputs.",
    )
    parser.add_argument(
        "--output-dir",
        default=os.environ.get("NDNA_OUTPUT_DIR", "plots/method5_generic/squad_generic_reports"),
        help="Directory to place HTML reports (defaults to input dir).",
    )
    parser.add_argument("--resample-n", type=int, default=128, help="Resample points for MAE/MSE.")
    parser.add_argument("--skip-report", action="store_true", help="Skip similarity report generation.")
    parser.add_argument("--skip-dashboards", action="store_true", help="Skip per-model/combined dashboards.")
    parser.add_argument(
        "--per-model-norm",
        action="store_true",
        help="Use per-model normalization instead of global. "
             "Warning: this loses inter-model magnitude comparisons in 3D plots.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    input_dir = os.path.abspath(args.input_dir)
    output_dir = os.path.abspath(args.output_dir or input_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # Global normalization by default (preserves inter-model comparisons)
    global_normalize = not args.per_model_norm
    if args.per_model_norm:
        print("Warning: Using per-model normalization. Inter-model magnitude comparisons will be lost in 3D plots.")

    models = load_models(input_dir, global_normalize=global_normalize)
    datasets = {}

    if not models:
        print(f"No .npz files in {input_dir}, checking for dataset subdirectories...")
        models, datasets = load_all_datasets_normalized(input_dir, global_normalize=global_normalize)
    else:
        datasets = load_all_datasets(input_dir)

    if not models:
        print("No valid models loaded. Check directory and npz keys.")
        return

    print(f"Loaded {len(models)} models total (family-grouped ordering).")

    (
        fig_3d,
        fig_st,
        fig_sb,
        fig_tb,
        fig_ndna,
        fig_layer_kappa,
        fig_layer_delta,
        fig_layer_vnorm,
    ) = build_combined_figures(models)

    dashboard_section_html = None

    if not args.skip_dashboards:
        dashboard_figs = build_dashboard_combined_figures(models)

        cross_dataset_figs = None
        if len(datasets) > 1:
            print(f"Found {len(datasets)} datasets, generating cross-dataset comparison plots...")
            cross_dataset_figs = build_cross_dataset_metric_figures(datasets)

        dashboard_section_html = render_combined_dashboard_section(models, dashboard_figs, cross_dataset_figs, datasets)

        if args.skip_report:
            combined_path = os.path.join(output_dir, "ALL_MODELS_trajectory.html")
            write_combined_dashboard(combined_path, models, dashboard_figs, cross_dataset_figs, datasets)
            print(f"Saved -> {combined_path}")

        write_model_dashboards(output_dir, models)

    if not args.skip_report:
        names, matrices = compute_distance_matrices(models, resample_n=args.resample_n)
        report_path = os.path.join(output_dir, "a_LLM_curve_similarity_report.html")
        build_report_html(
            output_path=report_path,
            models=models,
            names=names,
            matrices=matrices,
            fig_3d=fig_3d,
            fig_st=fig_st,
            fig_sb=fig_sb,
            fig_tb=fig_tb,
            fig_ndna=fig_ndna,
            fig_layer_kappa=fig_layer_kappa,
            fig_layer_delta=fig_layer_delta,
            fig_layer_vnorm=fig_layer_vnorm,
            dashboard_section_html=dashboard_section_html,
        )
        print(f"Saved -> {report_path}")


if __name__ == "__main__":
    main()
