"""NPZ persistence with a provenance manifest.

Every result written by this suite carries the config it was produced under, the git
hash of the tree, and a UTC timestamp, so a number in a table can always be traced back
to the run that made it. Mirrors ndna/storage/saver.py in the parent repo.
"""
from __future__ import annotations

import io
import json
import os
import subprocess
import time

import numpy as np


def git_hash(root=None):
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             cwd=root or os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def _jsonable(o):
    if isinstance(o, dict):
        return {k: _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if hasattr(o, "to_dict"):
        return _jsonable(o.to_dict())
    if hasattr(o, "__dict__"):
        return _jsonable(vars(o))
    return o


def save(path, arrays, cfg=None, notes=""):
    """Write an NPZ plus a sidecar .json manifest."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    if not path.endswith(".npz"):
        path += ".npz"
    payload = {k: np.asarray(v) for k, v in arrays.items()}
    np.savez_compressed(path, **payload)

    man = {
        "path": os.path.basename(path),
        "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git": git_hash(),
        "notes": notes,
        "arrays": {k: {"shape": list(np.shape(v)), "dtype": str(np.asarray(v).dtype)}
                   for k, v in payload.items()},
        "config": _jsonable(cfg) if cfg is not None else None,
    }
    with io.open(path.replace(".npz", ".json"), "w", encoding="utf-8") as f:
        json.dump(man, f, indent=2)
    return path


def load(path):
    if not path.endswith(".npz"):
        path += ".npz"
    with np.load(path, allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}
    man_path = path.replace(".npz", ".json")
    man = None
    if os.path.exists(man_path):
        with io.open(man_path, encoding="utf-8") as f:
            man = json.load(f)
    return arrays, man


def find(results_dir, experiment=None, model_key=None):
    """List result NPZs, optionally filtered by experiment / model."""
    out = []
    for fn in sorted(os.listdir(results_dir)):
        if not fn.endswith(".npz"):
            continue
        if experiment and not fn.startswith(experiment):
            continue
        if model_key and f"__{model_key}__" not in fn:
            continue
        out.append(os.path.join(results_dir, fn))
    return out
