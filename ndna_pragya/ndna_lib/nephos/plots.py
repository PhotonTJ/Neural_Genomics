from __future__ import annotations

from typing import Optional, Dict, Any
import json
import base64
import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def plot_mean_with_band(
    layers: np.ndarray,
    normal: np.ndarray,
    triggered: np.ndarray,
    title: str,
    ylabel: str,
    save_path: Optional[str] = None,
    show: bool = False,
) -> None:
    """Plot mean and +/- std bands for normal vs triggered runs."""
    n_mu, n_sd = normal.mean(0), normal.std(0)
    t_mu, t_sd = triggered.mean(0), triggered.std(0)

    plt.figure(figsize=(10, 4))
    plt.plot(layers, n_mu, marker="o", label="normal")
    plt.fill_between(layers, n_mu - n_sd, n_mu + n_sd, alpha=0.2)

    plt.plot(layers, t_mu, marker="o", label="triggered")
    plt.fill_between(layers, t_mu - t_sd, t_mu + t_sd, alpha=0.2)

    plt.title(title)
    plt.xlabel("Layer")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=150)

    if show:
        plt.show()

    plt.close()


def _img_to_data_uri(path: Path) -> str:
    b = path.read_bytes()
    enc = base64.b64encode(b).decode("ascii")
    return f"data:image/png;base64,{enc}"


def _plotly_div(div_id: str, fig: Dict[str, Any]) -> str:
    fig_json = json.dumps(fig)
    return f"""
<div id="{div_id}" style="width:100%;height:520px;"></div>
<script>
  Plotly.newPlot("{div_id}", {fig_json}.data, {fig_json}.layout, {{responsive:true}});
</script>
"""


def _curve3d_layer_metric_pair(
    layers: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    name: str,
    axis_y: str,
    axis_z: str,
) -> Dict[str, Any]:
    return {
        "type": "scatter3d",
        "mode": "lines+markers",
        "name": name,
        "x": layers.tolist(),
        "y": y.tolist(),
        "z": z.tolist(),
        "marker": {"size": 4},
        "line": {"width": 4},
        "hovertemplate": "layer=%{x}<br>" + axis_y + "=%{y}<br>" + axis_z + "=%{z}<extra></extra>",
    }


def _scatter3d_triple(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    color: np.ndarray,
    name: str,
    axis_x: str,
    axis_y: str,
    axis_z: str,
) -> Dict[str, Any]:
    return {
        "type": "scatter3d",
        "mode": "markers",
        "name": name,
        "x": x.tolist(),
        "y": y.tolist(),
        "z": z.tolist(),
        "marker": {
            "size": 4,
            "color": color.tolist(),
            "colorscale": "Viridis",
            "showscale": True,
            "colorbar": {"title": "layer"},
        },
        "hovertemplate": axis_x + "=%{x}<br>" + axis_y + "=%{y}<br>" + axis_z + "=%{z}<extra></extra>",
    }


def write_nephos_report_html(
    out_html: str,
    normal: Dict[str, np.ndarray],
    triggered: Dict[str, np.ndarray],
    png_paths: Dict[str, str],
    meta: Dict[str, Any],
) -> None:
    out_html_p = Path(out_html)

    # Means
    d_n = normal["drift"].mean(0)
    t_n = normal["thermo"].mean(0)
    k_n = normal["spectral"].mean(0)

    d_t = triggered["drift"].mean(0)
    t_t = triggered["thermo"].mean(0)
    k_t = triggered["spectral"].mean(0)

    L = d_n.shape[0]

    # Align indices carefully:
    # thermo is steps (0..L-2), map to layer 1..L-1
    layers_t = np.arange(1, L)
    thermo_n = t_n
    thermo_t = t_t
    drift_n_at = d_n[1:]
    drift_t_at = d_t[1:]

    # spectral is interior layers 1..L-2
    layers_k = np.arange(1, L - 1)
    spec_n = k_n
    spec_t = k_t
    drift_n_mid = d_n[1:-1]
    drift_t_mid = d_t[1:-1]
    thermo_n_mid = t_n[:-1]
    thermo_t_mid = t_t[:-1]

    # 3D figs
    fig_st = {
        "data": [
            _curve3d_layer_metric_pair(layers_k, spec_n, thermo_n_mid, "normal", "spectral", "thermo"),
            _curve3d_layer_metric_pair(layers_k, spec_t, thermo_t_mid, "triggered", "spectral", "thermo"),
        ],
        "layout": {
            "title": "Layer-wise 3D: spectral vs thermo",
            "scene": {
                "xaxis": {"title": "layer (1..L-2)"},
                "yaxis": {"title": "spectral curvature"},
                "zaxis": {"title": "thermo step"},
            },
            "legend": {"orientation": "h"},
        },
    }

    fig_tb = {
        "data": [
            _curve3d_layer_metric_pair(layers_t, thermo_n, drift_n_at, "normal", "thermo", "belief(drift)"),
            _curve3d_layer_metric_pair(layers_t, thermo_t, drift_t_at, "triggered", "thermo", "belief(drift)"),
        ],
        "layout": {
            "title": "Layer-wise 3D: thermo vs belief(drift)",
            "scene": {
                "xaxis": {"title": "layer (1..L-1)"},
                "yaxis": {"title": "thermo step"},
                "zaxis": {"title": "belief drift"},
            },
            "legend": {"orientation": "h"},
        },
    }

    fig_sb = {
        "data": [
            _curve3d_layer_metric_pair(layers_k, spec_n, drift_n_mid, "normal", "spectral", "belief(drift)"),
            _curve3d_layer_metric_pair(layers_k, spec_t, drift_t_mid, "triggered", "spectral", "belief(drift)"),
        ],
        "layout": {
            "title": "Layer-wise 3D: spectral vs belief(drift)",
            "scene": {
                "xaxis": {"title": "layer (1..L-2)"},
                "yaxis": {"title": "spectral curvature"},
                "zaxis": {"title": "belief drift"},
            },
            "legend": {"orientation": "h"},
        },
    }

    # Triple: points (spectral, thermo, belief) colored by layer
    fig_stb = {
        "data": [
            _scatter3d_triple(spec_n, thermo_n_mid, drift_n_mid, layers_k, "normal", "spectral", "thermo", "belief"),
            _scatter3d_triple(spec_t, thermo_t_mid, drift_t_mid, layers_k, "triggered", "spectral", "thermo", "belief"),
        ],
        "layout": {
            "title": "3D: spectral-thermo-belief (points colored by layer)",
            "scene": {
                "xaxis": {"title": "spectral curvature"},
                "yaxis": {"title": "thermo step"},
                "zaxis": {"title": "belief drift"},
            },
            "legend": {"orientation": "h"},
        },
    }

    # Embed PNGs as data URIs
    drift_uri = _img_to_data_uri(Path(png_paths["drift"]))
    thermo_uri = _img_to_data_uri(Path(png_paths["thermo"]))
    spec_uri = _img_to_data_uri(Path(png_paths["spectral"]))

    meta_json = json.dumps(meta, indent=2)

    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>Nephos report</title>
  <script src="https://cdn.plot.ly/plotly-2.30.0.min.js"></script>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 18px; }}
    .row {{ display: flex; gap: 12px; flex-wrap: wrap; }}
    .card {{ border: 1px solid #ddd; border-radius: 10px; padding: 12px; flex: 1; min-width: 320px; }}
    img {{ max-width: 100%; height: auto; }}
    pre {{ background: #f7f7f7; padding: 10px; border-radius: 8px; overflow:auto; }}
  </style>
</head>
<body>
  <h1>Nephos metrics report</h1>

  <div class="card">
    <h2>Run config</h2>
    <pre>{meta_json}</pre>
  </div>

  <h2>2D summaries (mean ± std)</h2>
  <div class="row">
    <div class="card"><h3>Belief drift</h3><img src="{drift_uri}"/></div>
    <div class="card"><h3>Thermo length</h3><img src="{thermo_uri}"/></div>
    <div class="card"><h3>Spectral curvature</h3><img src="{spec_uri}"/></div>
  </div>

  <h2>Interactive 3D plots</h2>
  <div class="card">{_plotly_div("fig_st", fig_st)}</div>
  <div class="card">{_plotly_div("fig_tb", fig_tb)}</div>
  <div class="card">{_plotly_div("fig_sb", fig_sb)}</div>
  <div class="card">{_plotly_div("fig_stb", fig_stb)}</div>
</body>
</html>
"""
    out_html_p.write_text(html, encoding="utf-8")


# -------------------------
# Interactive Plotly 3D
# -------------------------

def _require_plotly():
    try:
        import plotly.graph_objects as go  # noqa
    except Exception as e:
        raise RuntimeError(
            "Interactive 3D plots require plotly. Install it with:\n"
            "  pip install plotly\n"
            f"Original import error: {e}"
        )


def _mean_curves(stacked: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """
    stacked = {"drift": (N,L), "thermo": (N,L-1), "spectral": (N,L-2)}
    returns mean curves of shapes (L), (L-1), (L-2)
    """
    return {
        "drift": stacked["drift"].mean(0),
        "thermo": stacked["thermo"].mean(0),
        "spectral": stacked["spectral"].mean(0),
    }


def _write_html(fig, save_path: str) -> None:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.write_html(save_path, include_plotlyjs="cdn")


def plot_interactive_3d_pairs_and_triple(
    normal_stacked: Dict[str, np.ndarray],
    triggered_stacked: Dict[str, np.ndarray],
    out_dir: str,
    prefix: str = "nephos",
    show: bool = False,
) -> Dict[str, str]:
    """
    Creates 4 interactive HTML plots:

    1) layer vs thermo vs spectral
    2) layer vs thermo vs drift
    3) layer vs spectral vs drift
    4) thermo vs spectral vs drift (layer encoded in hover + marker color)

    Returns dict of plot_name -> html_path.

    IMPORTANT alignment detail:
    - drift is defined on layers: 0..L-1
    - thermo is defined on steps: 0..L-2, we map step i -> layer (i+1)
    - spectral is defined on interior: 0..L-3, we map index i -> layer (i+1)
    So:
      - thermo+spectral common layers are 1..L-2
      - spectral+drift common layers are 1..L-2
      - thermo+drift common layers are 1..L-1
      - triple uses layers 1..L-2
    """
    _require_plotly()
    import plotly.graph_objects as go

    nm = _mean_curves(normal_stacked)
    tm = _mean_curves(triggered_stacked)

    drift_n, thermo_n, spec_n = nm["drift"], nm["thermo"], nm["spectral"]
    drift_t, thermo_t, spec_t = tm["drift"], tm["thermo"], tm["spectral"]

    L = drift_n.shape[0]
    if thermo_n.shape[0] != L - 1 or spec_n.shape[0] != L - 2:
        raise ValueError(
            f"Shape mismatch: drift(L={L}), thermo({thermo_n.shape[0]}), spectral({spec_n.shape[0]})."
        )

    saved: Dict[str, str] = {}

    # ---- (1) layer vs thermo vs spectral  (layers 1..L-2) ----
    layers_12 = np.arange(1, L - 1)  # 1..L-2 inclusive
    y_thermo_n = thermo_n[layers_12 - 1]
    z_spec_n = spec_n[layers_12 - 1]
    y_thermo_t = thermo_t[layers_12 - 1]
    z_spec_t = spec_t[layers_12 - 1]

    fig = go.Figure()
    fig.add_trace(go.Scatter3d(
        x=layers_12, y=y_thermo_n, z=z_spec_n,
        mode="lines+markers", name="normal",
        text=[f"layer={int(l)}" for l in layers_12],
        hoverinfo="text+x+y+z",
    ))
    fig.add_trace(go.Scatter3d(
        x=layers_12, y=y_thermo_t, z=z_spec_t,
        mode="lines+markers", name="triggered",
        text=[f"layer={int(l)}" for l in layers_12],
        hoverinfo="text+x+y+z",
    ))
    fig.update_layout(
        title="Layerwise 3D: (layer, thermo, spectral)",
        scene=dict(
            xaxis_title="layer",
            yaxis_title="thermo (FR step)",
            zaxis_title="spectral curvature (kappa)",
        ),
        legend=dict(x=0, y=1),
    )
    p = os.path.join(out_dir, f"{prefix}_3d_layer_thermo_spectral.html")
    _write_html(fig, p)
    saved["layer_thermo_spectral"] = p
    if show:
        fig.show()

    # ---- (2) layer vs thermo vs drift  (layers 1..L-1) ----
    layers_11 = np.arange(1, L)  # 1..L-1 inclusive
    y_thermo_n = thermo_n[layers_11 - 1]
    z_drift_n = drift_n[layers_11]
    y_thermo_t = thermo_t[layers_11 - 1]
    z_drift_t = drift_t[layers_11]

    fig = go.Figure()
    fig.add_trace(go.Scatter3d(
        x=layers_11, y=y_thermo_n, z=z_drift_n,
        mode="lines+markers", name="normal",
        text=[f"layer={int(l)}" for l in layers_11],
        hoverinfo="text+x+y+z",
    ))
    fig.add_trace(go.Scatter3d(
        x=layers_11, y=y_thermo_t, z=z_drift_t,
        mode="lines+markers", name="triggered",
        text=[f"layer={int(l)}" for l in layers_11],
        hoverinfo="text+x+y+z",
    ))
    fig.update_layout(
        title="Layerwise 3D: (layer, thermo, drift)",
        scene=dict(
            xaxis_title="layer",
            yaxis_title="thermo (FR step)",
            zaxis_title="drift (belief push magnitude)",
        ),
        legend=dict(x=0, y=1),
    )
    p = os.path.join(out_dir, f"{prefix}_3d_layer_thermo_drift.html")
    _write_html(fig, p)
    saved["layer_thermo_drift"] = p
    if show:
        fig.show()

    # ---- (3) layer vs spectral vs drift  (layers 1..L-2) ----
    layers_12 = np.arange(1, L - 1)  # 1..L-2
    y_spec_n = spec_n[layers_12 - 1]
    z_drift_n = drift_n[layers_12]
    y_spec_t = spec_t[layers_12 - 1]
    z_drift_t = drift_t[layers_12]

    fig = go.Figure()
    fig.add_trace(go.Scatter3d(
        x=layers_12, y=y_spec_n, z=z_drift_n,
        mode="lines+markers", name="normal",
        text=[f"layer={int(l)}" for l in layers_12],
        hoverinfo="text+x+y+z",
    ))
    fig.add_trace(go.Scatter3d(
        x=layers_12, y=y_spec_t, z=z_drift_t,
        mode="lines+markers", name="triggered",
        text=[f"layer={int(l)}" for l in layers_12],
        hoverinfo="text+x+y+z",
    ))
    fig.update_layout(
        title="Layerwise 3D: (layer, spectral, drift)",
        scene=dict(
            xaxis_title="layer",
            yaxis_title="spectral curvature (kappa)",
            zaxis_title="drift (belief push magnitude)",
        ),
        legend=dict(x=0, y=1),
    )
    p = os.path.join(out_dir, f"{prefix}_3d_layer_spectral_drift.html")
    _write_html(fig, p)
    saved["layer_spectral_drift"] = p
    if show:
        fig.show()

    # ---- (4) thermo vs spectral vs drift (layers 1..L-2) ----
    layers_12 = np.arange(1, L - 1)  # 1..L-2
    x_th_n = thermo_n[layers_12 - 1]
    y_sp_n = spec_n[layers_12 - 1]
    z_dr_n = drift_n[layers_12]

    x_th_t = thermo_t[layers_12 - 1]
    y_sp_t = spec_t[layers_12 - 1]
    z_dr_t = drift_t[layers_12]

    fig = go.Figure()
    fig.add_trace(go.Scatter3d(
        x=x_th_n, y=y_sp_n, z=z_dr_n,
        mode="lines+markers", name="normal",
        marker=dict(size=4, color=layers_12, colorscale="Viridis", showscale=True),
        text=[f"layer={int(l)}" for l in layers_12],
        hoverinfo="text+x+y+z",
    ))
    fig.add_trace(go.Scatter3d(
        x=x_th_t, y=y_sp_t, z=z_dr_t,
        mode="lines+markers", name="triggered",
        marker=dict(size=4, color=layers_12, colorscale="Plasma", showscale=True),
        text=[f"layer={int(l)}" for l in layers_12],
        hoverinfo="text+x+y+z",
    ))
    fig.update_layout(
        title="3D: (thermo, spectral, drift) with layer coloring",
        scene=dict(
            xaxis_title="thermo (FR step)",
            yaxis_title="spectral curvature (kappa)",
            zaxis_title="drift (belief push magnitude)",
        ),
        legend=dict(x=0, y=1),
    )
    p = os.path.join(out_dir, f"{prefix}_3d_thermo_spectral_drift.html")
    _write_html(fig, p)
    saved["thermo_spectral_drift"] = p
    if show:
        fig.show()

    return saved


def plot_interactive_3d_all_in_one(
    normal_stacked: Dict[str, np.ndarray],
    triggered_stacked: Dict[str, np.ndarray],
    out_dir: str,
    filename: str = "nephos_3d_all.html",
    title: str = "Nephos 3D metrics (normal vs triggered)",
    show: bool = False,
) -> str:
    """
    Single HTML with 4 interactive 3D subplots:
      (1) layer vs thermo vs spectral
      (2) layer vs thermo vs drift
      (3) layer vs spectral vs drift
      (4) thermo vs spectral vs drift  (layer in hover + marker color)
    """
    _require_plotly()
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    nm = _mean_curves(normal_stacked)
    tm = _mean_curves(triggered_stacked)

    drift_n, thermo_n, spec_n = nm["drift"], nm["thermo"], nm["spectral"]
    drift_t, thermo_t, spec_t = tm["drift"], tm["thermo"], tm["spectral"]

    L = drift_n.shape[0]
    if thermo_n.shape[0] != L - 1 or spec_n.shape[0] != L - 2:
        raise ValueError(
            f"Shape mismatch: drift(L={L}), thermo({thermo_n.shape[0]}), spectral({spec_n.shape[0]})."
        )

    fig = make_subplots(
        rows=2, cols=2,
        specs=[[{"type": "scene"}, {"type": "scene"}],
               [{"type": "scene"}, {"type": "scene"}]],
        subplot_titles=[
            "(layer, thermo, spectral)",
            "(layer, thermo, drift)",
            "(layer, spectral, drift)",
            "(thermo, spectral, drift)",
        ],
        horizontal_spacing=0.03,
        vertical_spacing=0.06,
    )

    # Helper: only show legend once (first subplot)
    def _legend(show: bool) -> bool:
        return show

    # ---- (1) layer, thermo, spectral (layers 1..L-2) ----
    layers_12 = np.arange(1, L - 1)  # 1..L-2
    fig.add_trace(
        go.Scatter3d(
            x=layers_12,
            y=thermo_n[layers_12 - 1],
            z=spec_n[layers_12 - 1],
            mode="lines+markers",
            name="normal",
            showlegend=_legend(True),
            text=[f"layer={int(l)}" for l in layers_12],
            hoverinfo="text+x+y+z",
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter3d(
            x=layers_12,
            y=thermo_t[layers_12 - 1],
            z=spec_t[layers_12 - 1],
            mode="lines+markers",
            name="triggered",
            showlegend=_legend(True),
            text=[f"layer={int(l)}" for l in layers_12],
            hoverinfo="text+x+y+z",
        ),
        row=1, col=1,
    )

    # ---- (2) layer, thermo, drift (layers 1..L-1) ----
    layers_11 = np.arange(1, L)  # 1..L-1
    fig.add_trace(
        go.Scatter3d(
            x=layers_11,
            y=thermo_n[layers_11 - 1],
            z=drift_n[layers_11],
            mode="lines+markers",
            name="normal",
            showlegend=_legend(False),
            text=[f"layer={int(l)}" for l in layers_11],
            hoverinfo="text+x+y+z",
        ),
        row=1, col=2,
    )
    fig.add_trace(
        go.Scatter3d(
            x=layers_11,
            y=thermo_t[layers_11 - 1],
            z=drift_t[layers_11],
            mode="lines+markers",
            name="triggered",
            showlegend=_legend(False),
            text=[f"layer={int(l)}" for l in layers_11],
            hoverinfo="text+x+y+z",
        ),
        row=1, col=2,
    )

    # ---- (3) layer, spectral, drift (layers 1..L-2) ----
    fig.add_trace(
        go.Scatter3d(
            x=layers_12,
            y=spec_n[layers_12 - 1],
            z=drift_n[layers_12],
            mode="lines+markers",
            name="normal",
            showlegend=_legend(False),
            text=[f"layer={int(l)}" for l in layers_12],
            hoverinfo="text+x+y+z",
        ),
        row=2, col=1,
    )
    fig.add_trace(
        go.Scatter3d(
            x=layers_12,
            y=spec_t[layers_12 - 1],
            z=drift_t[layers_12],
            mode="lines+markers",
            name="triggered",
            showlegend=_legend(False),
            text=[f"layer={int(l)}" for l in layers_12],
            hoverinfo="text+x+y+z",
        ),
        row=2, col=1,
    )

    # ---- (4) thermo, spectral, drift (layers 1..L-2) ----
    x_th_n = thermo_n[layers_12 - 1]
    y_sp_n = spec_n[layers_12 - 1]
    z_dr_n = drift_n[layers_12]

    x_th_t = thermo_t[layers_12 - 1]
    y_sp_t = spec_t[layers_12 - 1]
    z_dr_t = drift_t[layers_12]

    fig.add_trace(
        go.Scatter3d(
            x=x_th_n, y=y_sp_n, z=z_dr_n,
            mode="lines+markers",
            name="normal",
            showlegend=_legend(False),
            marker=dict(size=4, color=layers_12, colorscale="Viridis", showscale=True),
            text=[f"layer={int(l)}" for l in layers_12],
            hoverinfo="text+x+y+z",
        ),
        row=2, col=2,
    )
    fig.add_trace(
        go.Scatter3d(
            x=x_th_t, y=y_sp_t, z=z_dr_t,
            mode="lines+markers",
            name="triggered",
            showlegend=_legend(False),
            marker=dict(size=4, color=layers_12, colorscale="Plasma", showscale=True),
            text=[f"layer={int(l)}" for l in layers_12],
            hoverinfo="text+x+y+z",
        ),
        row=2, col=2,
    )

    # Axis labels per scene
    fig.update_layout(
        title=title,
        height=900,
        legend=dict(x=0.0, y=1.05, orientation="h"),
    )

    fig.update_scenes(
        xaxis_title="layer", yaxis_title="thermo", zaxis_title="spectral",
        row=1, col=1
    )
    fig.update_scenes(
        xaxis_title="layer", yaxis_title="thermo", zaxis_title="drift",
        row=1, col=2
    )
    fig.update_scenes(
        xaxis_title="layer", yaxis_title="spectral", zaxis_title="drift",
        row=2, col=1
    )
    fig.update_scenes(
        xaxis_title="thermo", yaxis_title="spectral", zaxis_title="drift",
        row=2, col=2
    )

    out_path = os.path.join(out_dir, filename)
    _write_html(fig, out_path)
    if show:
        fig.show()
    return out_path
