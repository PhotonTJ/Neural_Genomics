"""Exp 15 -- matched profiles for the logit-trajectory comparison and the field-direction test.

For every prompt set, one forward pass records, from the same readout and the same positions:

  * Fisher-Rao step lengths (the nDNA length channel), pooled and per prompt;
  * logit-trajectory summaries of adjacent-layer logit updates (magnitude, coherence,
    rotation), the closest published representation (Jung and Jung, ACL 2026);
  * the projected belief field as a full vector per node (not only its norm), on all prompts
    and on the even and odd halves, plus the field norm and coherence;
  * optionally (task sets), a second pass for the exact parallel-transported field.

Nothing vocabulary-sized is kept per position: every quantity is accumulated per batch.

    CUDA_VISIBLE_DEVICES=0 python experiments/exp15_profile.py --ckpt llama31_8b/base
    python experiments/exp15_profile.py --list            # checkpoint keys
    python experiments/exp15_profile.py --smoke --ckpt tiny_llama/base   # pipeline check
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from core._optional import load_torch      # noqa: E402

torch, TORCH_AVAILABLE = load_torch()

PREREG = os.path.join(HERE, "exp15_prereg.json")
SMOKE = {
    "families": {
        "tiny_llama": {"base": "hf-internal-testing/tiny-random-LlamaForCausalLM",
                       "instruct": "hf-internal-testing/tiny-random-LlamaForCausalLM"},
        "tiny_gpt2": {"base": "hf-internal-testing/tiny-random-GPT2LMHeadModel"},
    },
    "squad": {"n_a": 8, "n_b": 12},
    "field_tasks": {"tasks": ["squad_v2", "gsm8k"], "n": 8},
}


# --------------------------------------------------------------------------- config
def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def load_prereg(smoke=False):
    raw = open(PREREG, "rb").read()
    pr = json.loads(raw)
    pr["_sha256"] = sha256_bytes(raw)
    if smoke:
        pr["families"] = SMOKE["families"]
        pr["prompt_sets"]["squad"].update(SMOKE["squad"])
        pr["prompt_sets"]["field_tasks"].update(SMOKE["field_tasks"])
        pr["readout"]["batch_size"] = 4
    return pr


def checkpoints(pr):
    """{'family/role': model_id}"""
    return {f"{fam}/{role}": mid for fam, roles in pr["families"].items() for role, mid in roles.items()}


# --------------------------------------------------------------------------- prompts
def build_sets(pr, cache_dir=None):
    """Context-disjoint SQuAD sets A and B, and the field-task sets."""
    from datasets import load_dataset
    from core.probes import load_probe

    sets = {}
    sq = pr["prompt_sets"]["squad"]
    ds = load_dataset(sq["dataset"], split=sq["split"], cache_dir=cache_dir)
    first = {}
    for i, c in enumerate(ds["context"]):
        first.setdefault(c, i)                          # one question per context
    idx = np.array(sorted(first.values()))
    order = np.random.default_rng(sq["seed"]).permutation(len(idx))
    a, b = sq["n_a"], sq["n_b"]
    if a + b > len(order):
        raise ValueError(f"only {len(order)} unique contexts for n_a + n_b = {a + b}")
    ia, ib = idx[order[:a]], idx[order[a:a + b]]
    render = lambda r: f"Context: {r['context']}\nQuestion: {r['question']}\nAnswer:"
    sets["squad_A"] = dict(prompts=[render(ds[int(i)]) for i in ia], indices=ia.tolist(), exact=False)
    sets["squad_B"] = dict(prompts=[render(ds[int(i)]) for i in ib], indices=ib.tolist(), exact=False)

    ft = pr["prompt_sets"]["field_tasks"]
    for task in ft["tasks"]:
        P, _ = load_probe(task, n=ft["n"], seed=ft["seed"], cache_dir=cache_dir)
        sets[f"task_{task}"] = dict(prompts=P, indices=None, exact=bool(ft.get("exact_field", False)))
    return sets


# --------------------------------------------------------------------------- model
def load_model(mid, revision, dtype, cache_dir):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(mid, revision=revision, cache_dir=cache_dir)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"          # positions come from the mask; no left-padding shift
    tok.truncation_side = "left"        # keep the end of the prompt (question, "Answer:")
    kw = dict(revision=revision, cache_dir=cache_dir, device_map={"": 0} if torch.cuda.is_available() else None)
    td = getattr(torch, dtype) if torch.cuda.is_available() else torch.float32
    try:
        model = AutoModelForCausalLM.from_pretrained(mid, dtype=td, **kw)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(mid, torch_dtype=td, **kw)
    model.eval()
    return model, tok


def final_norm(model):
    for attr in ("model.norm", "model.final_layernorm", "transformer.ln_f", "gpt_neox.final_layer_norm",
                 "model.language_model.norm"):
        obj = model
        try:
            for part in attr.split("."):
                obj = getattr(obj, part)
            return obj
        except AttributeError:
            continue
    raise RuntimeError("final normalisation layer not found; add its attribute path to final_norm()")


def tokenizer_hash(tok):
    v = sorted(tok.get_vocab().items())
    return sha256_bytes(json.dumps(v).encode("utf-8"))[:16]


# --------------------------------------------------------------------------- one batch
@torch.no_grad()
def node_logits(model, norm, head, softcap, enc, pos_b, pos_t):
    """Logits at every node for the selected positions -> (N, M, V) float32."""
    out = model(**enc, output_hidden_states=True, use_cache=False)
    hs = out.hidden_states                               # L+1 states: embedding + blocks
    Z = []
    for h in hs[:-1]:                                    # embedding output and blocks 1..L-1
        z = head(norm(h[pos_b, pos_t])).float()
        if softcap:
            z = softcap * torch.tanh(z / softcap)
        Z.append(z)
    Z.append(out.logits[pos_b, pos_t].float())           # final node: the model's own output
    del out, hs
    return torch.stack(Z, 0)


class Acc:
    """Per-set accumulators (float64 on the GPU)."""

    def __init__(self, N, V, n_prompts, dev):
        f64 = dict(dtype=torch.float64, device=dev)
        self.n_pos = 0
        self.n_pos_half = [0, 0]
        self.sum_d = torch.zeros(N - 1, **f64)
        self.sum_dnorm = torch.zeros(N - 1, **f64)
        self.sum_D = torch.zeros(N - 1, V, **f64)
        self.sum_ang = torch.zeros(max(N - 2, 1), **f64)
        self.sum_u = torch.zeros(N, V, **f64)
        self.sum_w_half = torch.zeros(2, N, V, **f64)
        self.sum_wnorm = torch.zeros(N, **f64)
        self.sum_pt = None
        self.len_prompt = torch.zeros(n_prompts, N - 1, **f64)
        self.mag_prompt = torch.zeros(n_prompts, N - 1, **f64)
        self.cnt_prompt = torch.zeros(n_prompts, **f64)
        self.targets = torch.zeros(V, **f64)


@torch.no_grad()
def batch_stats(Z, y, owner, half, acc, cfg, ubar=None):
    """Update the accumulators from node logits Z (N, M, V), targets y (M,)."""
    tau, floor = cfg["tau"], cfg["prob_floor"]
    q = torch.softmax(Z / tau, -1).clamp_min(floor)
    q = q / q.sum(-1, keepdim=True)
    u = q.sqrt()
    u = u / u.norm(dim=-1, keepdim=True)

    if ubar is not None:                                  # second pass: exact transport only
        g = -q / tau
        g.scatter_add_(2, y.view(1, -1, 1).expand(g.shape[0], -1, 1),
                       torch.full((g.shape[0], y.numel(), 1), 1.0 / tau, device=g.device))
        ug = u * g
        w = (ug - (u * ug).sum(-1, keepdim=True) * u) / (2 * tau)
        a = ubar.unsqueeze(1)                             # (N, 1, V)
        coef = (w * a).sum(-1, keepdim=True) / (1.0 + (u * a).sum(-1, keepdim=True))
        acc.sum_pt += (w - coef * (u + a)).sum(1).double()
        return

    M = y.numel()
    acc.n_pos += M
    for h in (0, 1):
        acc.n_pos_half[h] += int((half == h).sum())

    # Fisher-Rao steps (length)
    d = 2.0 * torch.arccos((u[:-1] * u[1:]).sum(-1).clamp(-1.0, 1.0))           # (N-1, M)
    acc.sum_d += d.sum(1).double()
    acc.len_prompt.index_add_(0, owner, d.T.double())
    acc.cnt_prompt.index_add_(0, owner, torch.ones(M, dtype=torch.float64, device=Z.device))

    # logit trajectory on centred logits (removes the additive-shift freedom)
    Zc = Z - Z.mean(-1, keepdim=True)
    D = Zc[1:] - Zc[:-1]                                                        # (N-1, M, V)
    del Zc
    nrm = D.norm(dim=-1)                                                        # (N-1, M)
    acc.sum_dnorm += nrm.sum(1).double()
    acc.sum_D += D.sum(1).double()
    acc.mag_prompt.index_add_(0, owner, nrm.T.double())
    if D.shape[0] >= 2:
        cs = (D[1:] * D[:-1]).sum(-1) / (nrm[1:] * nrm[:-1]).clamp_min(1e-12)
        acc.sum_ang += torch.arccos(cs.clamp(-1.0, 1.0)).sum(1).double()
    del D

    # belief push w = (I - u u^T) Diag(u) g / (2 tau), g = (e_y - q) / tau
    g = -q / tau
    g.scatter_add_(2, y.view(1, -1, 1).expand(g.shape[0], -1, 1),
                   torch.full((g.shape[0], M, 1), 1.0 / tau, device=g.device))
    ug = u * g
    del g
    w = (ug - (u * ug).sum(-1, keepdim=True) * u) / (2 * tau)
    del ug
    acc.sum_u += u.sum(1).double()
    acc.sum_wnorm += w.norm(dim=-1).sum(1).double()
    for h in (0, 1):
        m = half == h
        if m.any():
            acc.sum_w_half[h] += w[:, m].sum(1).double()
    acc.targets.index_add_(0, y, torch.ones(M, dtype=torch.float64, device=Z.device))


def tangent(a, v):
    return v - (a * v).sum(-1, keepdims=True) * a


# --------------------------------------------------------------------------- one set
@torch.no_grad()
def run_set(model, tok, norm, head, softcap, prompts, cfg, exact, log):
    dev = next(model.parameters()).device
    P = len(prompts)
    lens = [len(tok(p, truncation=True, max_length=cfg["max_length"])["input_ids"]) for p in prompts]
    order = np.argsort(lens)[::-1]                        # longest first: less padding, early OOM
    bs, k = cfg["batch_size"], cfg["keep_last_k"]
    acc = None
    batches = [order[i:i + bs] for i in range(0, P, bs)]

    def batch_inputs(ids_):
        chunk = [prompts[i] for i in ids_]
        enc = tok(chunk, return_tensors="pt", padding=True, truncation=True, max_length=cfg["max_length"])
        enc = {kk: v.to(dev) for kk, v in enc.items()}
        am, ii = enc["attention_mask"], enc["input_ids"]
        pb, pt, y, own = [], [], [], []
        for bi, gi in enumerate(ids_):
            valid = am[bi].nonzero().flatten()
            if valid.numel() < 2:
                continue
            for t in valid[:-1][-k:].tolist():
                pb.append(bi); pt.append(t); y.append(int(ii[bi, t + 1])); own.append(int(gi))
        if not pb:
            return None
        T = lambda x: torch.tensor(x, device=dev)
        return enc, T(pb), T(pt), T(y), T(own)

    t0 = time.time()
    for bno, ids_ in enumerate(batches):
        b = batch_inputs(ids_)
        if b is None:
            continue
        enc, pb, pt, y, own = b
        Z = node_logits(model, norm, head, softcap, enc, pb, pt)
        if acc is None:
            acc = Acc(Z.shape[0], Z.shape[2], P, dev)
        batch_stats(Z, y, own, own % 2, acc, cfg)
        del Z
        if bno % 25 == 0:
            el = time.time() - t0
            log(f"      batch {bno + 1}/{len(batches)}  {el:.0f}s  eta {el / (bno + 1) * (len(batches) - bno - 1):.0f}s")
    if acc is None:
        raise RuntimeError("no supervised positions")

    ubar = acc.sum_u / acc.sum_u.norm(dim=-1, keepdim=True)
    if exact:                                             # second pass for the transported mean
        log("      exact-field pass")
        acc.sum_pt = torch.zeros_like(acc.sum_u)
        ub32 = ubar.float()
        for ids_ in batches:
            b = batch_inputs(ids_)
            if b is None:
                continue
            enc, pb, pt, y, own = b
            Z = node_logits(model, norm, head, softcap, enc, pb, pt)
            batch_stats(Z, y, own, own % 2, acc, cfg, ubar=ub32)
            del Z

    # ---- finalise (float64 on CPU)
    c = lambda x: x.cpu().numpy()
    n = acc.n_pos
    ub = c(ubar)
    w_all = c(acc.sum_w_half.sum(0)) / n
    v_proj = tangent(ub, w_all)
    halves = np.stack([tangent(ub, c(acc.sum_w_half[h]) / max(acc.n_pos_half[h], 1)) for h in (0, 1)], 0)
    mean_wnorm = c(acc.sum_wnorm) / n
    meanD = c(acc.sum_D) / n
    mag = c(acc.sum_dnorm) / n
    cos_md = (meanD[1:] * meanD[:-1]).sum(-1) / np.maximum(
        np.linalg.norm(meanD[1:], axis=-1) * np.linalg.norm(meanD[:-1], axis=-1), 1e-300)
    cnt = np.maximum(c(acc.cnt_prompt), 1)[:, None]
    tg = c(acc.targets)
    top = np.argsort(-tg)[:2000]
    out = dict(
        length=c(acc.sum_d) / n,
        length_prompt=(c(acc.len_prompt) / cnt).astype(np.float32),
        lt_mag=mag,
        lt_coh=np.linalg.norm(meanD, axis=-1) / np.maximum(mag, 1e-300),
        lt_rot=c(acc.sum_ang) / n,
        lt_rot_mean=np.arccos(np.clip(cos_md, -1, 1)),
        lt_mag_prompt=(c(acc.mag_prompt) / cnt).astype(np.float32),
        field_norm=np.linalg.norm(v_proj, axis=-1),
        field_coherence=np.linalg.norm(v_proj, axis=-1) / np.maximum(mean_wnorm, 1e-300),
        field_vec=v_proj.astype(np.float16),
        field_half=halves.astype(np.float16),
        field_half_norm=np.linalg.norm(halves, axis=-1),
        target_top_ids=top.astype(np.int64),
        target_top_counts=tg[top].astype(np.int64),
        n_positions=np.array(n),
        n_prompts=np.array(P),
    )
    if exact:
        v_ex = c(acc.sum_pt) / n
        out["field_exact_vec"] = v_ex.astype(np.float16)
        out["field_exact_norm"] = np.linalg.norm(v_ex, axis=-1)
    return out


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ckpt", help="family/role key from the pre-registration")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="tiny random models and a few prompts")
    ap.add_argument("--sets", nargs="*", help="restrict to these prompt sets")
    ap.add_argument("--out", default=None)
    ap.add_argument("--cache-dir", default=os.environ.get("NDNA_CACHE"))
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--build-sets", action="store_true",
                    help="only build and save the prompt sets (run once before the GPU jobs)")
    a = ap.parse_args()

    pr = load_prereg(a.smoke)
    ck = checkpoints(pr)
    if a.list:
        print("\n".join(ck))
        return
    out_dir = a.out or os.path.join(ROOT, "results", "exp15_smoke" if a.smoke else "exp15")
    os.makedirs(out_dir, exist_ok=True)
    sets_path = os.path.join(out_dir, "prompt_sets.json")
    if a.build_sets or not os.path.exists(sets_path):
        sets = build_sets(pr, a.cache_dir)
        with open(sets_path, "w", encoding="utf-8") as f:
            json.dump(dict(prereg_sha256=pr["_sha256"], sets=sets), f)
        print(f"prompt sets -> {sets_path}: " + ", ".join(f"{k} ({len(v['prompts'])})" for k, v in sets.items()))
        if a.build_sets:
            return
    saved = json.load(open(sets_path, encoding="utf-8"))
    if saved["prereg_sha256"] != pr["_sha256"]:
        sys.exit("prompt_sets.json was built from a different pre-registration; rebuild with --build-sets")
    sets = saved["sets"]
    if a.ckpt not in ck:
        sys.exit(f"unknown --ckpt {a.ckpt!r}; choose from: {', '.join(ck)}")
    cfg = dict(pr["readout"])
    log = lambda s: print(f"[{a.ckpt}] {s}", flush=True)

    if a.sets:
        sets = {k: v for k, v in sets.items() if k in a.sets}
    stem = a.ckpt.replace("/", "__")
    todo = {k: v for k, v in sets.items()
            if a.overwrite or not os.path.exists(os.path.join(out_dir, f"{stem}__{k}.npz"))}
    if not todo:
        log("all sets already done")
        return

    mid = ck[a.ckpt]
    rev = pr.get("revisions", {}).get(mid)
    t_load = time.time()
    model, tok = load_model(mid, rev, cfg["dtype"], a.cache_dir)
    norm, head = final_norm(model), model.get_output_embeddings()
    softcap = getattr(model.config, "final_logit_softcapping", None) or \
        getattr(getattr(model.config, "text_config", None), "final_logit_softcapping", None)
    log(f"loaded {mid} in {time.time() - t_load:.0f}s; softcap={softcap}")

    import transformers
    base_meta = dict(
        experiment="exp15", ckpt=a.ckpt, family=a.ckpt.split("/")[0], role=a.ckpt.split("/")[1],
        model_id=mid, revision=getattr(model.config, "_commit_hash", None) or rev,
        n_layers=int(getattr(model.config, "num_hidden_layers", 0) or
                     getattr(getattr(model.config, "text_config", None), "num_hidden_layers", 0)),
        hidden_size=int(getattr(model.config, "hidden_size", 0) or
                        getattr(getattr(model.config, "text_config", None), "hidden_size", 0)),
        vocab_size=int(head.weight.shape[0]), tokenizer_hash=tokenizer_hash(tok),
        readout=cfg, prereg_sha256=pr["_sha256"], smoke=a.smoke,
        torch=torch.__version__, transformers=transformers.__version__,
        gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        host=platform.node())

    for name, s in todo.items():
        t0 = time.time()
        log(f"set {name}: {len(s['prompts'])} prompts, exact={s['exact']}")
        arrays = run_set(model, tok, norm, head, softcap, s["prompts"], cfg, s["exact"], log)
        meta = dict(base_meta, set=name, n_prompts=len(s["prompts"]),
                    n_nodes=int(len(arrays["field_norm"])),
                    prompt_sha256=sha256_bytes("\n\u0000".join(s["prompts"]).encode("utf-8")),
                    dataset_indices=s["indices"],
                    started_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t0)),
                    wall_seconds=round(time.time() - t0, 1))
        path = os.path.join(out_dir, f"{stem}__{name}")
        np.savez_compressed(path + ".npz", **arrays)
        with open(path + ".json", "w") as f:
            json.dump(meta, f, indent=1)
        log(f"  saved {os.path.basename(path)}.npz  ({time.time() - t0:.0f}s)")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
