"""Exp 10 -- model provenance: which base is this checkpoint built on?

Ranks candidate base models by DTW between triad profiles and reports top-1 / top-5 plus
the *margin* to the runner-up. The margin matters: a preview on the parent repo's stored
matrices recovered 12/12 lineages, but a third of those had margins under 25%, so
accuracy alone would overstate robustness.

Hard negatives are the point of the design: same-architecture models from a different
lineage, and same-family models at a different scale. Without them the task is too easy
to be evidence.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C                                     # noqa: E402
from core import harness, models, probes, triad, dtw   # noqa: E402

# candidate bases the query is matched against
CANDIDATES = {
    "qwen3_4b": "Qwen/Qwen3-4B",
    "gemma3_1b": "google/gemma-3-1b-pt",
    "llama3_8b": "meta-llama/Meta-Llama-3-8B",
    "mistral_7b": "mistralai/Mistral-7B-v0.3",
    "deepseek_7b": "deepseek-ai/deepseek-llm-7b-base",
}
# (query checkpoint, true base key). Derived models plus hard negatives.
QUERIES = [
    ("Qwen/Qwen3-4B-Instruct-2507", "qwen3_4b"),
    ("Qwen/Qwen3-4B-Thinking-2507", "qwen3_4b"),
    ("google/gemma-3-1b-it", "gemma3_1b"),
    ("meta-llama/Meta-Llama-3-8B-Instruct", "llama3_8b"),
    ("mistralai/Mistral-7B-Instruct-v0.3", "mistral_7b"),
    ("deepseek-ai/deepseek-llm-7b-chat", "deepseek_7b"),
    ("deepseek-ai/DeepSeek-R1-Distill-Llama-8B", "llama3_8b"),   # undeclared distillation
]


def _profile(mid, prompts, tcfg, args):
    m, tok = models.load(mid, device=args.device, cache_dir=C.CACHE)
    p = triad.profile_model(m, tok, prompts, tcfg, device=args.device)
    models.free(m)
    return p


def run(model_key, model_id, seed, args):
    """model_key selects which slice of the query list this run covers."""
    prompts, _ = probes.load_probe(args.probe, n=args.n_prompts, seed=seed, cache_dir=C.CACHE)
    tcfg = harness.triad_cfg(args)

    base_keys, base_profs = [], []
    for k, mid in CANDIDATES.items():
        try:
            base_profs.append(_profile(mid, prompts, tcfg, args)); base_keys.append(k)
        except Exception as e:
            print(f"     base {k} unavailable: {e}")

    q_names, q_truth, q_profs = [], [], []
    for mid, truth in QUERIES:
        if truth not in base_keys:
            continue
        try:
            q_profs.append(_profile(mid, prompts, tcfg, args))
            q_names.append(mid); q_truth.append(truth)
        except Exception as e:
            print(f"     query {mid} unavailable: {e}")

    if not q_profs:
        raise RuntimeError("no query checkpoints could be profiled")

    X = dtw.stack_triad(base_profs + q_profs, n=32)
    Cm, _ = dtw.pairwise_matrix(X, band=0.2, normalise="minmax")
    nb = len(base_profs)

    top1, top2, margins, preds, dtrue = [], [], [], [], []
    for i, truth in enumerate(q_truth):
        d = Cm[nb + i, :nb].copy()
        # a query that IS one of the bases must not match itself
        if q_names[i] in CANDIDATES.values():
            d[list(CANDIDATES.values()).index(q_names[i])] = np.inf
        order = np.argsort(d)
        pred = base_keys[order[0]]
        preds.append(pred)
        top1.append(pred == truth)
        top2.append(truth in [base_keys[j] for j in order[:2]])
        margins.append(float((d[order[1]] - d[order[0]]) / d[order[1]]) if np.isfinite(d[order[1]]) else np.nan)
        dtrue.append(float(d[base_keys.index(truth)]))

    arrays = {
        "query": np.array(q_names), "true_base": np.array(q_truth),
        "predicted": np.array(preds), "candidates": np.array(base_keys),
        "dtw_query_to_base": Cm[nb:, :nb],
        "d_true": np.array(dtrue), "margin": np.array(margins),
        "top1": np.array(top1), "top2": np.array(top2),
        "top1_acc": float(np.mean(top1)), "top2_acc": float(np.mean(top2)),
        "chance": 1.0 / max(len(base_keys), 1),
        "margin_median": float(np.nanmedian(margins)),
        "n_margin_below_25pct": int(np.nansum(np.array(margins) < 0.25)),
    }
    return arrays, "lineage retrieval with margin reporting; hard negatives included"


if __name__ == "__main__":
    p = harness.base_parser("exp10_provenance", __doc__)
    a = p.parse_args()
    a.models = a.models[:1]          # the query/candidate sets are global, run once
    harness.run_over_models("exp10_provenance", a, run)
