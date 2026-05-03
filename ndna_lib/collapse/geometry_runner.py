# ndna_lib/collapse/geometry_runner.py
from __future__ import annotations

import json
import os
from typing import Dict, Any, List

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM

from ndna_lib.collapse.config import GeometryConfig
from ndna_lib.data import load_alpaca_text_dataset, load_alpaca_prompts, load_wikipedia_text_dataset, load_wikipedia_prompts
from ndna_lib.geometry import (
    DEVICE,
    collate_causal,
    make_adapter,
    compute_param_effort,
    compute_fr_and_alignment_streaming,
    spectral_curvature_for_prompts,
)


# --------------------------------------------------------------------------
# Method-5: parameter-space effort + FR thermodynamic length on Alpaca
# --------------------------------------------------------------------------


def run_method5_unified_on_alpaca(
    model_path: str = None,
    geom_cfg: GeometryConfig = None,
    dataset_id: str = "yahma/alpaca-cleaned",
    model: "AutoModelForCausalLM" = None,
    tokenizer: "AutoTokenizer" = None,
) -> Dict[str, Any]:
    """
    Unified Method-5 backbone for collapse experiment, restricted to Alpaca.

    Uses canonical geometry utilities from ndna_lib.geometry.

    Args:
        model_path: Path to load model from (if model/tokenizer not provided)
        geom_cfg: Geometry configuration
        dataset_id: Dataset ID for Alpaca
        model: Pre-loaded model (optional, avoids loading from disk)
        tokenizer: Pre-loaded tokenizer (optional, avoids loading from disk)

    Returns dict with:
        E: np.array (L,)
        E_examples: int
        Delta: np.array (L-1,)
        Eta: np.array (L-1,) = Delta^2 / E_ℓ
        FR_total: float
        n_tokens: int
        n_params: list[int]
    """
    owns_model = False
    if model is None or tokenizer is None:
        if model_path is None:
            raise ValueError("Either model_path or (model, tokenizer) must be provided")
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(model_path)
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False
        model.to(DEVICE).eval()
        owns_model = True
    else:
        # Ensure model is in eval mode
        model.eval()

    adapter = make_adapter(
        model,
        model_name=getattr(model.config, "_name_or_path", model_path),
    )

    ds = load_wikipedia_text_dataset(
        max_samples=geom_cfg.geom_max_samples,
        dataset_id=dataset_id,
    )
    loader = DataLoader(
        ds,
        batch_size=geom_cfg.geom_batch_size,
        shuffle=False,
        num_workers=0 if DEVICE == "cpu" else 2,
        pin_memory=(DEVICE == "cuda"),
        collate_fn=collate_causal(tokenizer, geom_cfg.geom_max_len),
    )

    # Parameter-space effort (observed Fisher)
    E_l, n_ex, n_params = compute_param_effort(
        model, adapter, loader, unit=geom_cfg.unit
    )

    # FR thermodynamic length (prediction-space)
    Delta_l, _, _, mean_total_fr, n_tokens = compute_fr_and_alignment_streaming(
        model, adapter, loader
    )

    eps = 1e-12
    Eta_l = (Delta_l ** 2) / (E_l[:-1] + eps)

    # clean up GPU memory only if we loaded the model
    if owns_model:
        del model
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    return {
        "E": E_l,
        "E_examples": n_ex,
        "Delta": Delta_l,
        "Eta": Eta_l,
        "FR_total": mean_total_fr,
        "n_tokens": n_tokens,
        "n_params": n_params,
    }


# --------------------------------------------------------------------------
# Spectral curvature on Alpaca (using canonical geometry)
# --------------------------------------------------------------------------


def run_spectral_curvature_on_alpaca(
    model_path: str = None,
    geom_cfg: GeometryConfig = None,
    dataset_id: str = "yahma/alpaca-cleaned",
    dataset_config: str = None,
    model: "AutoModelForCausalLM" = None,
    tokenizer: "AutoTokenizer" = None,
) -> Dict[str, Any]:
    """
    Compute spectral curvature κ^(simp)_ℓ across layers using prompts.

    Supports both Alpaca-style datasets and Wikipedia. Automatically detects
    Wikipedia datasets and uses appropriate loader.

    Wraps ndna_lib.geometry.spectral_curvature_for_prompts and returns a
    JSON-friendly dict compatible with the original collapse scripts:
      {
        "curvature_mean": [L-1 floats? / interior length],
        "per_prompt": [
           {"curvature": [...], "speeds": [...], "num_nodes": int},
           ...
        ],
        "num_layers": L
      }

    Args:
        model_path: Path to load model from (if model/tokenizer not provided)
        geom_cfg: Geometry configuration
        dataset_id: Dataset ID (e.g., "yahma/alpaca-cleaned" or "wikimedia/wikipedia")
        dataset_config: Optional config for datasets that require it (e.g., "20231101.en" for Wikipedia)
        model: Pre-loaded model (optional, avoids loading from disk)
        tokenizer: Pre-loaded tokenizer (optional, avoids loading from disk)
    """
    owns_model = False
    if model is None or tokenizer is None:
        if model_path is None:
            raise ValueError("Either model_path or (model, tokenizer) must be provided")
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(model_path)
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False
        model.to(DEVICE).eval()
        owns_model = True
    else:
        # Ensure model is in eval mode
        model.eval()

    adapter = make_adapter(
        model,
        model_name=getattr(model.config, "_name_or_path", model_path),
    )

    # Load prompts - detect dataset type and use appropriate loader
    if dataset_id.startswith("wikimedia/wikipedia") or "wikipedia" in dataset_id.lower():
        # Use provided config or default to English
        config = dataset_config or "20231101.en"
        ds = load_wikipedia_prompts(
            max_samples=geom_cfg.spectral_num_prompts,
            dataset_id=dataset_id,
            config=config,
        )
    else:
        # Default to Alpaca-style datasets
        ds = load_alpaca_prompts(
            max_samples=geom_cfg.spectral_num_prompts,
            dataset_id=dataset_id,
        )
    prompts_texts: List[str] = [ex["prompt"] for ex in ds]

    # Optional: enforce a token cap via truncation by going text -> tokens -> text.
    # We do it here so the canonical spectral_curvature_for_prompts stays simple.
    truncated_prompts: List[str] = []
    max_tok = geom_cfg.spectral_max_tokens

    for text in prompts_texts:
        enc = tokenizer(
            text,
            add_special_tokens=False,
            truncation=True,
            max_length=max_tok,
        )
        truncated = tokenizer.decode(enc["input_ids"], skip_special_tokens=True)
        truncated_prompts.append(truncated)

    # Build (name, text) pairs for the geometry function
    named_prompts = [
        (f"ex_{i}", txt) for i, txt in enumerate(truncated_prompts)
    ]

    spec_raw = spectral_curvature_for_prompts(
        model=model,
        tokenizer=tokenizer,
        prompts=named_prompts,
        model_name=model_path,
    )

    # Convert to the previous JSON structure
    per_prompt = []
    all_k = []

    for name in sorted(spec_raw.keys()):
        entry = spec_raw[name]
        kappa = np.asarray(entry["curvature"], dtype=float)
        speeds = np.asarray(entry["speeds"], dtype=float)
        num_nodes = int(entry["num_nodes"][0]) if np.ndim(entry["num_nodes"]) else int(entry["num_nodes"])

        per_prompt.append(
            {
                "curvature": kappa.tolist(),
                "speeds": speeds.tolist(),
                "num_nodes": num_nodes,
            }
        )
        all_k.append(kappa)

    if not all_k:
        raise RuntimeError("No curvature data computed from Alpaca prompts.")

    k_arr = np.stack(all_k, axis=0)   # (num_prompts, L-1 or L-2 depending on definition)
    k_mean = k_arr.mean(axis=0)

    # All prompts share the same number of nodes; use the first
    num_layers = per_prompt[0]["num_nodes"] if per_prompt else 0

    # clean up GPU memory only if we loaded the model
    if owns_model:
        del model
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    return {
        "curvature_mean": k_mean.tolist(),
        "per_prompt": per_prompt,
        "num_layers": num_layers,
    }


# --------------------------------------------------------------------------
# Save metrics
# --------------------------------------------------------------------------


def save_geometry_metrics(
    out_dir: str,
    method5: Dict[str, Any],
    spectral: Dict[str, Any],
) -> None:
    os.makedirs(out_dir, exist_ok=True)

    m5_json = {
        "E": method5["E"].tolist(),
        "E_examples": method5["E_examples"],
        "Delta": method5["Delta"].tolist(),
        "Eta": method5["Eta"].tolist(),
        "FR_total": method5["FR_total"],
        "n_tokens": method5["n_tokens"],
        "n_params": method5["n_params"],
    }
    with open(os.path.join(out_dir, "method5_unified.json"), "w") as f:
        json.dump(m5_json, f)

    with open(os.path.join(out_dir, "spectral_curvature.json"), "w") as f:
        json.dump(spectral, f)
