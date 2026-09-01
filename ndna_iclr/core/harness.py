"""Shared CLI and run loop, so each experiment file is only its own logic."""
from __future__ import annotations

import argparse
import os
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config as C           # noqa: E402
from core import io as nio   # noqa: E402


def base_parser(experiment, description=""):
    p = argparse.ArgumentParser(prog=experiment, description=description)
    p.add_argument("--models", nargs="*", default=list(C.MODELS),
                   help="model keys from config.MODELS")
    p.add_argument("--seeds", nargs="*", type=int, default=[0])
    p.add_argument("--probe", default="squad_v2")
    p.add_argument("--n-prompts", type=int, default=128)
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--keep-last-k", type=int, default=8)
    p.add_argument("--curvature", default="turn", choices=["turn", "chord"])
    p.add_argument("--out", default=C.RESULTS)
    p.add_argument("--dry-run", action="store_true",
                   help="print the plan without loading any model")
    return p


def triad_cfg(args):
    cfg = C.TriadCfg(keep_last_k=args.keep_last_k, batch_size=args.batch_size,
                     curvature=args.curvature)
    cfg.max_length = 384
    return cfg


def run_over_models(experiment, args, fn):
    """Call fn(model_key, model_id, seed, args) -> (arrays, notes); persist each result."""
    plan = [(mk, s) for mk in args.models for s in args.seeds]
    print(f"[{experiment}] {len(plan)} run(s): "
          + ", ".join(f"{mk}/seed{s}" for mk, s in plan))
    if args.dry_run:
        return []

    written = []
    for mk, seed in plan:
        mid = C.MODELS.get(mk, mk)
        tag = f"{experiment}__{mk}__seed{seed}"
        t0 = time.time()
        print(f"  -> {tag}  ({mid})", flush=True)
        try:
            arrays, notes = fn(mk, mid, seed, args)
        except Exception:
            print(f"  !! {tag} FAILED\n{traceback.format_exc()}", file=sys.stderr)
            continue
        arrays = dict(arrays)
        arrays.setdefault("wall_seconds", time.time() - t0)
        meta = {"experiment": experiment, "model_key": mk, "model_id": mid,
                "seed": seed, "probe": args.probe, "n_prompts": args.n_prompts,
                "curvature": args.curvature, "keep_last_k": args.keep_last_k}
        path = nio.save(os.path.join(args.out, tag), arrays, cfg=meta, notes=notes)
        written.append(path)
        print(f"     saved {os.path.basename(path)}  ({time.time()-t0:.0f}s)", flush=True)
    return written


def load_all(experiment, results_dir=None):
    """Collect every result NPZ for an experiment -> [(tag, arrays, manifest)]."""
    out = []
    for p in nio.find(results_dir or C.RESULTS, experiment=experiment):
        a, m = nio.load(p)
        out.append((os.path.basename(p)[:-4], a, m))
    return out
