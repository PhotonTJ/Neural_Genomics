#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import random
from typing import Dict

import numpy as np
import torch

from ndna_lib.collapse import (
    CollapseConfig,
    BreedingConfig,
    GeometryConfig,
    ProtocolType,
    CrossBreedingProtocol,
    InbreedingProtocol,
    run_method5_unified_on_alpaca,
    run_spectral_curvature_on_alpaca,
    save_geometry_metrics,
)


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------


def load_model_zoo(path: str = "model_zoo.json") -> Dict[str, Dict]:
    with open(path, "r") as f:
        return json.load(f)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_run_dir(cfg: CollapseConfig) -> str:
    run_name = cfg.run_name or "default_run"
    return os.path.join(cfg.base_run_dir, cfg.model_tag, run_name)


def make_protocol(cfg: CollapseConfig):
    if cfg.protocol == ProtocolType.CROSS_BREEDING:
        return CrossBreedingProtocol(cfg)
    elif cfg.protocol == ProtocolType.INBREEDING:
        return InbreedingProtocol(cfg)
    else:
        raise ValueError(f"Unsupported protocol type: {cfg.protocol}")


# --------------------------------------------------------------------------
# Main collapse loop for a single model
# --------------------------------------------------------------------------


def run_collapse_for_model(
    model_key: str,
    hf_id: str,
    protocol_type: ProtocolType,
    base_run_dir: str,
    run_name: str,
    breeding: BreedingConfig,
    geometry: GeometryConfig,
    seed: int,
    save_models: bool = False,
):
    """
    Run model collapse experiment for a single model.
    
    Args:
        model_key: Short name for the model
        hf_id: HuggingFace model ID
        protocol_type: CROSS_BREEDING or INBREEDING
        base_run_dir: Base directory for runs
        run_name: Experiment name
        breeding: Breeding configuration
        geometry: Geometry configuration
        seed: Random seed
        save_models: If True, save fine-tuned models to disk (uses more space).
                     If False, keep models in memory between generations (default).
    """
    cfg = CollapseConfig(
        base_model_id=hf_id,
        model_tag=model_key,
        protocol=protocol_type,
        base_run_dir=base_run_dir,
        run_name=run_name,
        seed=seed,
        breeding=breeding,
        geometry=geometry,
    )

    set_seed(cfg.seed)

    run_dir = get_run_dir(cfg)
    os.makedirs(run_dir, exist_ok=True)
    print(f"\n=== [MODEL] {model_key} | HF id: {hf_id} ===")
    print(f"Run dir: {run_dir}")
    print(f"Save models to disk: {save_models}")

    # ------------------ Gen 0: geometry on base model ------------------
    gen0_metrics_dir = os.path.join(run_dir, "gen0", "metrics")
    os.makedirs(gen0_metrics_dir, exist_ok=True)

    print("\n[GEN0] Geometry on base model")
    m5_0 = run_method5_unified_on_alpaca(
        model_path=cfg.base_model_id,
        geom_cfg=cfg.geometry,
        dataset_id=cfg.breeding.dataset_id,
    )
    spec_0 = run_spectral_curvature_on_alpaca(
        model_path=cfg.base_model_id,
        geom_cfg=cfg.geometry,
        dataset_id=cfg.breeding.dataset_id,
    )
    save_geometry_metrics(gen0_metrics_dir, m5_0, spec_0)

    # ------------------ Collapse protocol ------------------
    protocol = make_protocol(cfg)
    
    # Track current model state: either a path (str) or (model, tokenizer) tuple
    current_model = None
    current_tokenizer = None
    current_ckpt = cfg.base_model_id  # Only used if save_models=True

    # Gen 1..G
    for gen in range(1, cfg.breeding.generations + 1):
        print(f"\n=== [GEN {gen}] protocol step ({cfg.protocol.value}) ===")
        
        if save_models:
            # Save models to disk (original behavior)
            model_out_dir = os.path.join(run_dir, f"gen{gen}", "model")
            os.makedirs(model_out_dir, exist_ok=True)

            current_ckpt = protocol.finetune_one_generation(
                gen_idx=gen,
                base_ckpt=current_ckpt,
                out_dir=model_out_dir,
                save_model=True,
            )
            print(f"[GEN {gen}] New checkpoint at: {current_ckpt}")

            metrics_dir = os.path.join(run_dir, f"gen{gen}", "metrics")
            os.makedirs(metrics_dir, exist_ok=True)

            print(f"[GEN {gen}] Geometry evaluation...")
            m5 = run_method5_unified_on_alpaca(
                model_path=current_ckpt,
                geom_cfg=cfg.geometry,
                dataset_id=cfg.breeding.dataset_id,
            )
            spec = run_spectral_curvature_on_alpaca(
                model_path=current_ckpt,
                geom_cfg=cfg.geometry,
                dataset_id=cfg.breeding.dataset_id,
            )
        else:
            # Keep models in memory (no disk save)
            if current_model is None:
                # First generation: load from base checkpoint
                result = protocol.finetune_one_generation(
                    gen_idx=gen,
                    base_ckpt=cfg.base_model_id,
                    save_model=False,
                )
            else:
                # Subsequent generations: pass model in memory
                result = protocol.finetune_one_generation(
                    gen_idx=gen,
                    model=current_model,
                    tokenizer=current_tokenizer,
                    save_model=False,
                )
            
            current_model, current_tokenizer = result
            print(f"[GEN {gen}] Model kept in memory")

            metrics_dir = os.path.join(run_dir, f"gen{gen}", "metrics")
            os.makedirs(metrics_dir, exist_ok=True)

            print(f"[GEN {gen}] Geometry evaluation...")
            m5 = run_method5_unified_on_alpaca(
                geom_cfg=cfg.geometry,
                dataset_id=cfg.breeding.dataset_id,
                model=current_model,
                tokenizer=current_tokenizer,
            )
            spec = run_spectral_curvature_on_alpaca(
                geom_cfg=cfg.geometry,
                dataset_id=cfg.breeding.dataset_id,
                model=current_model,
                tokenizer=current_tokenizer,
            )
        
        save_geometry_metrics(metrics_dir, m5, spec)

    # Clean up final model if in memory
    if not save_models and current_model is not None:
        del current_model
        del current_tokenizer
        from ndna_lib.collapse.protocols import clear_gpu_memory
        clear_gpu_memory()

    print(f"\n[MODEL {model_key}] Finished all generations.\n")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run model-collapse experiments from model_zoo.json."
    )

    p.add_argument(
        "--zoo-path",
        type=str,
        default="model_zoo.json",
        help="Path to model_zoo.json.",
    )

    p.add_argument(
        "--model-keys",
        type=str,
        default="",
        help=(
            "Comma-separated list of model keys to run. "
            "If empty, use all entries with enabled=true."
        ),
    )

    p.add_argument(
        "--protocol",
        type=str,
        choices=["cross", "inbreed"],
        default="cross",
        help="Collapse protocol: cross-breeding or inbreeding.",
    )

    p.add_argument(
        "--base-run-dir",
        type=str,
        default="./collapse_runs_zoo",
        help="Base directory for all collapse runs.",
    )

    p.add_argument(
        "--run-name",
        type=str,
        default="run1",
        help="Run name subfolder under base_run_dir/model_tag/.",
    )

    p.add_argument(
        "--dataset-id",
        type=str,
        default="yahma/alpaca-cleaned",
        help="HF dataset id for collapse training/geometry.",
    )

    # Breeding knobs (collapse)
    p.add_argument("--generations", type=int, default=3)
    p.add_argument("--max-train-samples", type=int, default=5000)
    p.add_argument("--max-steps-first", type=int, default=1000)
    p.add_argument("--max-steps-later", type=int, default=500)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--train-max-seq-len", type=int, default=512)

    # Geometry knobs
    p.add_argument("--geom-max-samples", type=int, default=256)
    p.add_argument("--geom-max-len", type=int, default=256)
    p.add_argument("--geom-batch-size", type=int, default=4)
    p.add_argument(
        "--geom-unit",
        type=str,
        choices=["sequence", "token"],
        default="sequence",
        help="Unit for observed Fisher: per sequence or per token.",
    )
    p.add_argument("--spectral-prompts", type=int, default=16)
    p.add_argument("--spectral-max-tokens", type=int, default=128)

    p.add_argument("--seed", type=int, default=4242)

    # Quick smoke mode (overrides some knobs)
    p.add_argument(
        "--smoke",
        action="store_true",
        help="Tiny run (fewer samples/steps) to smoke-test the pipeline.",
    )

    p.add_argument(
        "--save-models",
        action="store_true",
        help="Save fine-tuned models to disk after each generation. Default is to keep in memory only (saves disk space).",
    )

    return p.parse_args()


def main():
    args = parse_args()

    zoo = load_model_zoo(args.zoo_path)

    # Select models
    if args.model_keys:
        requested = [k.strip() for k in args.model_keys.split(",") if k.strip()]
        model_items = []
        for k in requested:
            if k not in zoo:
                raise ValueError(f"Requested model_key '{k}' not in model_zoo.")
            model_items.append((k, zoo[k]))
    else:
        # default: all enabled models
        model_items = [
            (k, info) for k, info in zoo.items() if info.get("enabled", False)
        ]

    if not model_items:
        raise RuntimeError("No models selected (check --model-keys or enabled flags).")

    # Protocol type
    protocol_type = (
        ProtocolType.CROSS_BREEDING
        if args.protocol == "cross"
        else ProtocolType.INBREEDING
    )

    # Breeding config
    if args.smoke:
        breeding = BreedingConfig(
            dataset_id=args.dataset_id,
            max_train_samples=min(args.max_train_samples, 512),
            max_steps_first=50,
            max_steps_later=50,
            lr=args.lr,
            train_max_seq_len=args.train_max_seq_len,
            generations=min(args.generations, 2),
        )
        geom_max_samples = min(args.geom_max_samples, 64)
        spectral_prompts = min(args.spectral_prompts, 4)
    else:
        breeding = BreedingConfig(
            dataset_id=args.dataset_id,
            max_train_samples=args.max_train_samples,
            max_steps_first=args.max_steps_first,
            max_steps_later=args.max_steps_later,
            lr=args.lr,
            train_max_seq_len=args.train_max_seq_len,
            generations=args.generations,
        )
        geom_max_samples = args.geom_max_samples
        spectral_prompts = args.spectral_prompts

    geometry = GeometryConfig(
        geom_max_samples=geom_max_samples,
        geom_max_len=args.geom_max_len,
        geom_batch_size=args.geom_batch_size,
        unit=args.geom_unit,
        spectral_num_prompts=spectral_prompts,
        spectral_max_tokens=args.spectral_max_tokens,
    )

    print(f"Protocol      : {protocol_type.value}")
    print(f"Base run dir  : {args.base_run_dir}")
    print(f"Run name      : {args.run_name}")
    print(f"Dataset id    : {args.dataset_id}")
    print(f"Generations   : {breeding.generations}")
    print(f"Smoke mode    : {args.smoke}")
    print(f"Save models   : {args.save_models}")
    print(f"Models to run : {[k for k, _ in model_items]}\n")

    for model_key, info in model_items:
        hf_id = info["hf_id"]
        run_collapse_for_model(
            model_key=model_key,
            hf_id=hf_id,
            protocol_type=protocol_type,
            base_run_dir=args.base_run_dir,
            run_name=args.run_name,
            breeding=breeding,
            geometry=geometry,
            seed=args.seed,
            save_models=args.save_models,
        )


if __name__ == "__main__":
    main()
