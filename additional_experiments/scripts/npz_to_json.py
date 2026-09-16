"""Convert result NPZs to plain-text JSON.

The NPZ files are binary, so the repo routes them through Git LFS and they do not
diff. Every array a run produces is small and of a plain dtype, so the same content
fits in JSON, which reviews and diffs like any other text file.

    python scripts/npz_to_json.py                 # convert everything under results/
    python scripts/npz_to_json.py --check         # verify the JSON round-trips, write nothing

One <run>.data.json per <run>.npz, beside it. Arrays become nested lists under
"arrays", each with its dtype and shape so the NPZ can be rebuilt exactly. NaN and
Infinity are written as JSON5-style bare tokens, which Python's json reads back but
strict parsers reject -- rebuild with this module rather than a strict reader.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def npz_to_obj(path):
    a = np.load(path, allow_pickle=False)
    arrays = {}
    for k in a.files:
        v = a[k]
        arrays[k] = {"dtype": str(v.dtype), "shape": list(v.shape), "data": v.tolist()}
    return {"source": os.path.basename(path), "arrays": arrays}


def obj_to_arrays(obj):
    return {k: np.array(v["data"], dtype=np.dtype(v["dtype"])).reshape(v["shape"])
            for k, v in obj["arrays"].items()}


def equal(a, b):
    if a.dtype != b.dtype or a.shape != b.shape:
        return False
    if a.dtype.kind in "fc":
        return np.array_equal(a, b, equal_nan=True)
    return np.array_equal(a, b)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", default=os.path.join(ROOT, "results"))
    p.add_argument("--check", action="store_true",
                   help="round-trip every NPZ through JSON and report mismatches")
    args = p.parse_args()

    paths = sorted(glob.glob(os.path.join(args.results, "**", "*.npz"), recursive=True))
    bad = 0
    for path in paths:
        obj = npz_to_obj(path)
        orig = np.load(path, allow_pickle=False)
        back = obj_to_arrays(json.loads(json.dumps(obj)))
        for k in orig.files:
            if not equal(orig[k], back[k]):
                print(f"  MISMATCH {os.path.basename(path)}::{k}")
                bad += 1
        if not args.check:
            out = path[:-4] + ".data.json"
            with open(out, "w", encoding="utf-8") as fh:
                json.dump(obj, fh, indent=1, sort_keys=True)
                fh.write("\n")
    verb = "checked" if args.check else "converted"
    print(f"{verb} {len(paths)} npz; {bad} mismatched array(s)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
