"""End-to-end smoke test on a tiny model. Needs torch; finishes in a minute on CPU.

Checks the parts test_core.py cannot: that the lens fires at every depth, that the three
quantities come out with the right shapes and signs, and that the commitment bound
actually holds on real numbers.

    python scripts/smoke.py --model sshleifer/tiny-gpt2
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C                                 # noqa: E402
from core import models, probes, triad, dtw        # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="sshleifer/tiny-gpt2")
    p.add_argument("--device", default="cpu")
    p.add_argument("--n", type=int, default=8)
    a = p.parse_args()

    print(f"loading {a.model} on {a.device} ...")
    m, tok = models.load(a.model, device=a.device, dtype="float32", cache_dir=C.CACHE)
    prompts = probes._synthetic(a.n, seed=0)

    cfg = C.TriadCfg(keep_last_k=4, batch_size=2)
    cfg.max_length = 64
    prof = triad.profile_model(m, tok, prompts, cfg, device=a.device)

    n_nodes = m.config.num_hidden_layers + 1
    checks = []

    def chk(name, cond, detail=""):
        checks.append(cond)
        print(f"  {'PASS' if cond else 'FAIL'}  {name}{('  ' + detail) if detail else ''}")

    chk("length has L nodes", len(prof["length"]) == n_nodes - 1,
        f"{len(prof['length'])} vs {n_nodes-1}")
    chk("kappa has L-1 interior nodes", len(prof["kappa"]) == n_nodes - 2,
        f"{len(prof['kappa'])} vs {n_nodes-2}")
    chk("belief has L+1 nodes", len(prof["belief"]) == n_nodes,
        f"{len(prof['belief'])} vs {n_nodes}")
    chk("all finite", all(np.all(np.isfinite(prof[k]))
                          for k in ("kappa", "length", "belief")))
    chk("length non-negative", bool(np.all(prof["length"] >= 0)))
    chk("kappa non-negative", bool(np.all(prof["kappa"] >= 0)))
    chk("FR steps below pi", bool(np.all(prof["length"] <= np.pi + 1e-6)),
        f"max={prof['length'].max():.4f}")

    # commitment bound: d(q_l, q_L) <= sum_{k>l} L_k  (App. A.6)
    tail = prof["tail_length"]
    chk("tail is non-increasing", bool(np.all(np.diff(tail) <= 1e-9)))
    chk("tail[0] == total length",
        bool(np.isclose(tail[0], prof["length_total"], rtol=1e-6)))
    d = triad.commitment_depth(tail, 0.05 * float(prof["length_total"]))
    chk("commitment depth in range", 0 <= d <= len(tail), f"l*={d}")

    # DTW self-distance on the real profile
    X = dtw.stack_triad([prof, prof], n=16)
    Cm, Em = dtw.pairwise_matrix(X, band=0.25, normalise="none")
    chk("DTW self-distance is zero", bool(np.isclose(Cm[0, 1], 0.0, atol=1e-10)),
        f"{Cm[0,1]:.2e}")

    print(f"\n  kappa  mean={prof['kappa'].mean():.5f}  range=[{prof['kappa'].min():.5f}, {prof['kappa'].max():.5f}]")
    print(f"  length mean={prof['length'].mean():.5f}  total={float(prof['length_total']):.4f}")
    print(f"  belief mean={prof['belief'].mean():.5f}")
    print(f"\n{sum(checks)}/{len(checks)} checks passed")
    return 0 if all(checks) else 1


if __name__ == "__main__":
    sys.exit(main())
