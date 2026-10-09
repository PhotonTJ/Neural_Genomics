"""Run exp15 on all GPUs, one checkpoint per GPU, then exp16.

    python experiments/exp15_launch.py                  # every checkpoint in exp15_prereg.json
    python experiments/exp15_launch.py --gpus 0 1 2 3   # restrict GPUs
    python experiments/exp15_launch.py --smoke          # tiny models, a few prompts (pipeline check)

Logs go to logs/exp15/<checkpoint>.log. Finished prompt sets are skipped on a re-run, so an
interrupted launch can simply be started again.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PY = sys.executable


def gpu_ids():
    try:
        out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=20).stdout
        return [str(i) for i, line in enumerate(out.splitlines()) if line.startswith("GPU")]
    except Exception:
        return []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", nargs="*", default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--only", nargs="*", help="subset of checkpoint keys")
    ap.add_argument("--no-eval", action="store_true")
    a = ap.parse_args()
    flag = ["--smoke"] if a.smoke else []
    prof = os.path.join(HERE, "exp15_profile.py")

    # 1. prompt sets, once (downloads datasets; avoids concurrent cache writes)
    subprocess.run([PY, prof, "--build-sets", *flag], check=True, cwd=ROOT)
    ckpts = subprocess.run([PY, prof, "--list", *flag], check=True, cwd=ROOT,
                           capture_output=True, text=True).stdout.split()
    if a.only:
        ckpts = [c for c in ckpts if c in a.only]
    # largest models first so the longest jobs start immediately
    order = {"llama31_8b": 0, "qwen3_4b": 1, "gemma3_1b": 2}
    ckpts.sort(key=lambda c: order.get(c.split("/")[0], 9))

    gpus = a.gpus if a.gpus is not None else gpu_ids()
    if not gpus:
        gpus = ["cpu"]
    print(f"{len(ckpts)} checkpoint(s) on {len(gpus)} device(s): {', '.join(gpus)}")
    log_dir = os.path.join(ROOT, "logs", "exp15_smoke" if a.smoke else "exp15")
    os.makedirs(log_dir, exist_ok=True)

    # 2. a simple queue: one checkpoint per free GPU
    queue, running, failed = list(ckpts), {}, []
    t0 = time.time()
    while queue or running:
        for g in gpus:
            if g not in running and queue:
                ck = queue.pop(0)
                env = dict(os.environ)
                if g != "cpu":
                    env["CUDA_VISIBLE_DEVICES"] = g
                log = open(os.path.join(log_dir, ck.replace("/", "__") + ".log"), "w")
                p = subprocess.Popen([PY, prof, "--ckpt", ck, *flag], cwd=ROOT, env=env,
                                     stdout=log, stderr=subprocess.STDOUT)
                running[g] = (ck, p, log)
                print(f"  [{time.time() - t0:6.0f}s] start {ck} on {g}")
        time.sleep(5)
        for g, (ck, p, log) in list(running.items()):
            if p.poll() is not None:
                log.close()
                status = "done" if p.returncode == 0 else f"FAILED ({p.returncode})"
                if p.returncode != 0:
                    failed.append(ck)
                print(f"  [{time.time() - t0:6.0f}s] {status} {ck}")
                del running[g]

    if failed:
        print(f"failed: {', '.join(failed)}; see {log_dir}. Re-run to resume.")
    if not a.no_eval:
        subprocess.run([PY, os.path.join(HERE, "exp16_evaluate.py"), *flag], cwd=ROOT)


if __name__ == "__main__":
    main()
