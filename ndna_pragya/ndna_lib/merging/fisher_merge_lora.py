# ndna_lib/merging/fisher_merge_lora.py
import argparse
import json
from pathlib import Path
from typing import Dict, List

import torch
from safetensors.torch import load_file, save_file


LORA_CFG_KEYS = [
    # LoRA essentials
    "peft_type",
    "task_type",
    "r",
    "lora_alpha",
    "lora_dropout",
    "bias",
    "target_modules",
    # common toggles that affect adapter shape/behavior
    "fan_in_fan_out",
    "modules_to_save",
    "init_lora_weights",
    "use_rslora",
    "use_dora",
]


def load_json(p: Path) -> dict:
    return json.loads(p.read_text())


def canonicalize_cfg(cfg: dict) -> dict:
    out = {}
    for k in LORA_CFG_KEYS:
        if k in cfg:
            out[k] = cfg[k]
    # make target_modules order-invariant
    if "target_modules" in out and isinstance(out["target_modules"], list):
        out["target_modules"] = sorted(out["target_modules"])
    return out


def normalize_fisher_mean(f: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    keys = [k for k in f.keys() if not k.startswith("__")]
    means = [f[k].float().mean() for k in keys]
    scale = torch.stack(means).mean().clamp_min(1e-12)
    return {k: (v / scale) if k in keys else v for k, v in f.items()}


def fisher_merge(
    weights_list: List[Dict[str, torch.Tensor]],
    fishers_list: List[Dict[str, torch.Tensor]],
    alphas: List[float],
    eps: float,
    fisher_floor: float,
) -> Dict[str, torch.Tensor]:
    if not (len(weights_list) == len(fishers_list) == len(alphas)):
        raise ValueError("weights_list, fishers_list, alphas must have same length")

    keys0 = set(weights_list[0].keys())
    # Require exact key match across all adapters (no silent dropping)
    for w in weights_list[1:]:
        if set(w.keys()) != keys0:
            raise RuntimeError("Adapter weight keys differ across inputs. Fix training/config first; merge would be garbage.")

    merged = {}
    for k in keys0:
        num = None
        den = None
        for w, f, a in zip(weights_list, fishers_list, alphas):
            if k not in f:
                raise RuntimeError(f"Missing fisher for key {k}")
            wk = w[k].to(torch.float32)
            fk = f[k].to(torch.float32).clamp_min(fisher_floor)

            term_num = a * fk * wk
            term_den = a * fk
            num = term_num if num is None else (num + term_num)
            den = term_den if den is None else (den + term_den)

        merged[k] = (num / (den + eps)).to(weights_list[0][k].dtype)

    return merged


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapters", nargs="+", required=True)
    ap.add_argument("--fishers", nargs="+", required=True)
    ap.add_argument("--alphas", nargs="+", type=float, default=None)
    ap.add_argument("--eps", type=float, default=1e-8)
    ap.add_argument("--fisher_floor", type=float, default=1e-12)
    ap.add_argument("--normalize_fishers", choices=["none", "mean"], default="mean")
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    if len(args.adapters) != len(args.fishers):
        raise ValueError("--adapters and --fishers must have same length")
    n = len(args.adapters)

    alphas = args.alphas if args.alphas is not None else [1.0] * n
    if len(alphas) != n:
        raise ValueError("--alphas must match number of adapters")

    adapter_dirs = [Path(x) for x in args.adapters]
    fisher_paths = [Path(x) for x in args.fishers]

    for d in adapter_dirs:
        if not (d / "adapter_config.json").exists():
            raise FileNotFoundError(f"Missing adapter_config.json in {d}")
        if not (d / "adapter_model.safetensors").exists():
            raise FileNotFoundError(f"Missing adapter_model.safetensors in {d}")

    cfg0_full = load_json(adapter_dirs[0] / "adapter_config.json")
    cfg0 = canonicalize_cfg(cfg0_full)
    for d in adapter_dirs[1:]:
        cfg = canonicalize_cfg(load_json(d / "adapter_config.json"))
        if cfg != cfg0:
            raise RuntimeError(
                "Adapter configs differ in LoRA-relevant keys. "
                "Fisher-merge requires identical LoRA hyperparams/targets."
            )

    weights_list = [load_file(str(d / "adapter_model.safetensors")) for d in adapter_dirs]
    fishers_list = [load_file(str(p)) for p in fisher_paths]

    if args.normalize_fishers == "mean":
        fishers_list = [normalize_fisher_mean(f) for f in fishers_list]

    merged_weights = fisher_merge(
        weights_list=weights_list,
        fishers_list=fishers_list,
        alphas=alphas,
        eps=args.eps,
        fisher_floor=args.fisher_floor,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "adapter_config.json").write_text(json.dumps(cfg0_full, indent=2))
    save_file(merged_weights, str(out_dir / "adapter_model.safetensors"))
    print(f"[OK] Wrote merged adapter -> {out_dir}")


if __name__ == "__main__":
    main()
