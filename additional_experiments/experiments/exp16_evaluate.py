"""Exp 16 -- evaluate the exp15 profiles (numpy only; runs on a laptop).

  A. Checkpoint identification across the context-disjoint SQuAD sets A and B.
  B. Related-checkpoint retrieval within one set.
  C. The same two tests for every method in the pre-registration: nDNA length, nDNA length+field
     and the logit-trajectory summaries (magnitude, coherence, rotation, combined), all with the
     same prompts, positions, readout, per-profile scaling, DTW and scoring.
  D. Uncertainty: bootstrap over queries and over families, paired bootstrap of each method
     against the pre-registered comparison reference, permutation p for the nearest-neighbour rate.
  E. Field direction: split-half reliability, task identification by direction against norm,
     within-family direction, target alignment, and exact against projected field.

    python experiments/exp16_evaluate.py                 # results/exp15 -> results/exp16
    python experiments/exp16_evaluate.py --smoke
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PREREG = os.path.join(HERE, "exp15_prereg.json")
B_BOOT = 10000
N_PERM = 5000


# --------------------------------------------------------------------------- loading
def load(in_dir):
    prof = {}
    for p in sorted(glob.glob(os.path.join(in_dir, "*.npz"))):
        meta = json.load(open(p[:-4] + ".json"))
        prof[(meta["ckpt"], meta["set"])] = (dict(np.load(p)), meta)
    if not prof:
        sys.exit(f"no exp15 results in {in_dir}")
    return prof


# --------------------------------------------------------------------------- profiles
def interp(x, n):
    x = np.asarray(x, float)
    if len(x) == n:
        return x
    return np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(x)), x)


def channel(arr, name):
    n = len(arr["length"])
    src = {"length": arr["length"], "field": arr["field_norm"], "lt_mag": arr["lt_mag"],
           "lt_coh": arr["lt_coh"], "lt_rot": arr["lt_rot"]}[name]
    return interp(src, n)                       # every channel on the step grid of length


def scale(x, lo=0.05, hi=1.0):
    a, b = np.nanmin(x), np.nanmax(x)
    return np.full_like(x, (lo + hi) / 2) if b <= a else lo + (hi - lo) * (x - a) / (b - a)


def profile(arr, channels):
    """Per-profile, per-channel min-max scaling to [0.05, 1] (the paper's dashboard convention)."""
    return np.stack([scale(channel(arr, c)) for c in channels], 1)


def dtw(a, b, band=0.2):
    """Aligned mean squared cost with a relative-depth band (as analysis.py in the paper repo)."""
    n, m = len(a), len(b)
    C = ((a[:, None, :] - b[None, :, :]) ** 2).sum(-1)
    D = np.full((n + 1, m + 1), np.inf)
    D[0, 0] = 0
    for i in range(1, n + 1):
        lo = max(1, int(math.floor((i / n - band) * m)))
        hi = min(m, int(math.ceil((i / n + band) * m)))
        for j in range(lo, hi + 1):
            D[i, j] = C[i - 1, j - 1] + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    i, j, steps = n, m, 1
    while (i, j) != (1, 1):
        i, j = min([(D[i - 1, j - 1], (i - 1, j - 1)), (D[i - 1, j], (i - 1, j)), (D[i, j - 1], (i, j - 1))],
                   key=lambda x: x[0])[1]
        steps += 1
    return D[n, m] / steps


def dist_matrix(rows, cols):
    return np.array([[dtw(r, c) for c in cols] for r in rows])


# --------------------------------------------------------------------------- scoring
def ident_scores(D):
    """D[i, j]: query i (set A) against candidate j (set B); the correct candidate is j = i."""
    k = len(D)
    hit = np.array([int(np.argmin(D[i]) == i) for i in range(k)])
    margin = np.array([np.min(np.delete(D[i], i)) / max(D[i, i], 1e-300) for i in range(k)])
    return hit, margin


def exp_first_rank_rr(n, r):
    """E[1 / rank of the first relative] when r of n candidates are relatives, in random order."""
    if r == 0:
        return float("nan")
    return sum(math.comb(n - k, r - 1) / math.comb(n, r) / k for k in range(1, n - r + 2))


def retr_scores(D, fam):
    """Per-query nearest-neighbour hit, reciprocal rank, AUC, chance and expected RR."""
    k = len(D)
    rows = []
    for i in range(k):
        cand = [j for j in range(k) if j != i]
        rel = [j for j in cand if fam[j] == fam[i]]
        if not rel:
            continue
        unrel = [j for j in cand if fam[j] != fam[i]]
        order = sorted(cand, key=lambda j: D[i, j])
        rank = 1 + min(order.index(j) for j in rel)
        if unrel:
            auc = np.mean([(D[i, a] < D[i, b]) + 0.5 * (D[i, a] == D[i, b]) for a in rel for b in unrel])
        else:
            auc = float("nan")
        rows.append(dict(q=i, hit=int(order[0] in rel), rr=1.0 / rank, auc=auc,
                         chance=len(rel) / len(cand), exp_rr=exp_first_rank_rr(len(cand), len(rel))))
    return rows


def boot_ci(values_fn, groups, rng, B=B_BOOT):
    """Percentile 95% interval of a statistic over a resampling of `groups` (list of index lists)."""
    stats = []
    g = len(groups)
    for _ in range(B):
        pick = rng.integers(0, g, g)
        idx = [i for p in pick for i in groups[p]]
        stats.append(values_fn(idx))
    return [float(np.nanpercentile(stats, 2.5)), float(np.nanpercentile(stats, 97.5))]


def perm_p_nn(D, fam, rng, n=N_PERM):
    obs = np.mean([r["hit"] for r in retr_scores(D, fam)])
    fam = np.asarray(fam)
    c = 0
    for _ in range(n):
        f = rng.permutation(fam)
        rs = retr_scores(D, list(f))
        if rs and np.mean([r["hit"] for r in rs]) >= obs - 1e-12:
            c += 1
    return (c + 1) / (n + 1)


# --------------------------------------------------------------------------- field direction
def cosv(a, b):
    a = a.astype(np.float64); b = b.astype(np.float64)
    return float((a * b).sum() / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-300))


def layer_cos(A, B, nodes):
    return float(np.mean([cosv(A[n], B[n]) for n in nodes]))


def thirds(N):
    return {"all": list(range(N)), "first": list(range(0, N // 3)),
            "middle": list(range(N // 3, 2 * N // 3)), "last": list(range(2 * N // 3, N))}


def field_direction(prof, ckpts, tasks, fam_of):
    out = {"per_checkpoint": {}, "within_family": {}, "exact_vs_projected": {}}
    for ck in ckpts:
        have = [t for t in tasks if (ck, t) in prof]
        if len(have) < 2:
            continue
        H = {t: prof[(ck, t)][0]["field_half"] for t in have}
        N = H[have[0]].shape[1]
        res = {}
        for part, nodes in thirds(N).items():
            if not nodes:
                continue
            same = [layer_cos(H[t][0], H[t][1], nodes) for t in have]
            cross = [layer_cos(H[t][0], H[s][1], nodes) for t in have for s in have if s != t]
            pred = {t: max(have, key=lambda s: layer_cos(H[t][0], H[s][1], nodes)) for t in have}
            res[part] = dict(split_half_cos=float(np.mean(same)), cross_task_cos=float(np.mean(cross)),
                             task_id_direction=f"{sum(pred[t] == t for t in have)}/{len(have)}")
        # task identification from the norm profile of the halves (same nodes, DTW, per-profile scaling)
        nh = {t: prof[(ck, t)][0]["field_half_norm"] for t in have}
        pn = {t: min(have, key=lambda s: dtw(scale(nh[t][0])[:, None], scale(nh[s][1])[:, None])) for t in have}
        res["task_id_norm"] = f"{sum(pn[t] == t for t in have)}/{len(have)}"
        # target alignment at the final node
        align = {}
        for t in have:
            a = prof[(ck, t)][0]
            v = a["field_vec"][-1].astype(np.float64)
            pos = np.clip(v, 0, None)
            ids = a["target_top_ids"][:50]
            share = float(pos[ids].sum() / max(pos.sum(), 1e-300))
            align[t] = dict(share_top50=share, chance=50 / len(v))
        res["target_alignment"] = align
        out["per_checkpoint"][ck] = res
        ex = {}
        for t in have:
            a = prof[(ck, t)][0]
            if "field_exact_vec" in a:
                ang = [math.degrees(math.acos(max(-1.0, min(1.0, cosv(a["field_vec"][n], a["field_exact_vec"][n])))))
                       for n in range(a["field_vec"].shape[0])]
                rel = np.linalg.norm(a["field_exact_vec"].astype(np.float64) - a["field_vec"], axis=-1) / \
                    np.maximum(a["field_exact_norm"], 1e-300)
                ex[t] = dict(median_angle_deg=float(np.median(ang)), median_rel_diff=float(np.median(rel)))
        if ex:
            out["exact_vs_projected"][ck] = ex
    # within family: base against instruct, same task, same vocabulary
    fams = {}
    for ck in ckpts:
        fams.setdefault(fam_of[ck], []).append(ck)
    for f, members in fams.items():
        if len(members) < 2:
            continue
        a_ck, b_ck = members[0], members[1]
        rows = {}
        for t in tasks:
            if (a_ck, t) not in prof or (b_ck, t) not in prof:
                continue
            A, Bv = prof[(a_ck, t)][0], prof[(b_ck, t)][0]
            if A["field_vec"].shape != Bv["field_vec"].shape:
                continue
            N = A["field_vec"].shape[0]
            rows[t] = {part: dict(base_vs_instruct=layer_cos(A["field_vec"], Bv["field_vec"], nodes),
                                  split_half_ceiling=layer_cos(A["field_half"][0], A["field_half"][1], nodes))
                       for part, nodes in thirds(N).items() if nodes}
        out["within_family"][f"{a_ck} vs {b_ck}"] = rows
    return out


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--allow-changed-prereg", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    inp = a.inp or os.path.join(ROOT, "results", "exp15_smoke" if a.smoke else "exp15")
    out = a.out or os.path.join(ROOT, "results", "exp16_smoke" if a.smoke else "exp16")
    os.makedirs(out, exist_ok=True)
    rng = np.random.default_rng(a.seed)

    pr_raw = open(PREREG, "rb").read()
    pr = json.loads(pr_raw)
    sha = hashlib.sha256(pr_raw).hexdigest()
    prof = load(inp)
    used = {m["prereg_sha256"] for _, m in prof.values()}
    if not a.smoke and used != {sha} and not a.allow_changed_prereg:
        sys.exit("profiles were produced under a different pre-registration than exp15_prereg.json")

    ckpts = sorted({k[0] for k in prof})
    fam_of = {ck: prof[next(k for k in prof if k[0] == ck)][1]["family"] for ck in ckpts}
    methods = pr["methods"]
    ref = pr["primary"]["comparison_reference"]
    report = dict(prereg_sha256=sha, checkpoints=ckpts, families=fam_of, identification={}, retrieval={},
                  paired_vs_reference={}, metadata_baseline={})

    # ---- A. identification, A -> B (primary) and B -> A
    both = [ck for ck in ckpts if (ck, "squad_A") in prof and (ck, "squad_B") in prof]
    hits_by_method = {}
    for name, chs in methods.items():
        for qs, cs in (("squad_A", "squad_B"), ("squad_B", "squad_A")):
            Q = [profile(prof[(ck, qs)][0], chs) for ck in both]
            Cc = [profile(prof[(ck, cs)][0], chs) for ck in both]
            D = dist_matrix(Q, Cc)
            hit, margin = ident_scores(D)
            groups = [[i] for i in range(len(both))]
            ci = boot_ci(lambda idx: np.mean(hit[idx]), groups, rng) if len(both) > 1 else [None, None]
            report["identification"].setdefault(name, {})[f"{qs}->{cs}"] = dict(
                hits=f"{hit.sum()}/{len(hit)}", rate=float(hit.mean()), ci95=ci,
                median_margin=float(np.median(margin)), min_margin=float(np.min(margin)),
                per_query={both[i]: dict(hit=int(hit[i]), margin=float(margin[i])) for i in range(len(both))})
            if qs == "squad_A":
                hits_by_method[name] = hit

    # ---- B. retrieval within each set
    rr_by_method = {}
    for name, chs in methods.items():
        for s in ("squad_B", "squad_A"):
            cks = [ck for ck in ckpts if (ck, s) in prof]
            P = [profile(prof[(ck, s)][0], chs) for ck in cks]
            D = dist_matrix(P, P)
            fam = [fam_of[ck] for ck in cks]
            rows = retr_scores(D, fam)
            if not rows:
                continue
            groups_q = [[i] for i in range(len(rows))]
            fams = sorted(set(fam[r["q"]] for r in rows))
            groups_f = [[i for i, r in enumerate(rows) if fam[r["q"]] == f] for f in fams]
            stat = lambda key: (lambda idx: np.nanmean([rows[i][key] for i in idx]))
            res = dict(
                n_queries=len(rows),
                nn=f"{sum(r['hit'] for r in rows)}/{len(rows)}", nn_rate=float(np.mean([r["hit"] for r in rows])),
                nn_chance=float(np.mean([r["chance"] for r in rows])),
                mrr=float(np.mean([r["rr"] for r in rows])), mrr_expected=float(np.mean([r["exp_rr"] for r in rows])),
                auc=float(np.nanmean([r["auc"] for r in rows])),
                ci95_query=dict(nn=boot_ci(stat("hit"), groups_q, rng), mrr=boot_ci(stat("rr"), groups_q, rng),
                                auc=boot_ci(stat("auc"), groups_q, rng)),
                ci95_family=dict(nn=boot_ci(stat("hit"), groups_f, rng), mrr=boot_ci(stat("rr"), groups_f, rng)),
                perm_p_nn=perm_p_nn(D, fam, rng),
                per_query={cks[r["q"]]: dict(hit=r["hit"], rr=r["rr"], auc=r["auc"]) for r in rows})
            report["retrieval"].setdefault(name, {})[s] = res
            if s == "squad_B":
                rr_by_method[name] = rows

    # ---- C. paired differences against the comparison reference (bootstrap over queries)
    for name in methods:
        if name == ref or ref not in hits_by_method:
            continue
        h1, h0 = hits_by_method[name], hits_by_method[ref]
        groups = [[i] for i in range(len(h1))]
        d_id = float(np.mean(h1) - np.mean(h0))
        ci_id = boot_ci(lambda idx: np.mean(h1[idx]) - np.mean(h0[idx]), groups, rng)
        entry = dict(identification_diff=d_id, identification_ci95=ci_id)
        if name in rr_by_method and ref in rr_by_method:
            r1 = np.array([r["rr"] for r in rr_by_method[name]])
            r0 = np.array([r["rr"] for r in rr_by_method[ref]])
            a1 = np.array([r["auc"] for r in rr_by_method[name]])
            a0 = np.array([r["auc"] for r in rr_by_method[ref]])
            g2 = [[i] for i in range(len(r1))]
            entry.update(mrr_diff=float(r1.mean() - r0.mean()),
                         mrr_ci95=boot_ci(lambda idx: r1[idx].mean() - r0[idx].mean(), g2, rng),
                         auc_diff=float(np.nanmean(a1) - np.nanmean(a0)),
                         auc_ci95=boot_ci(lambda idx: np.nanmean(a1[idx]) - np.nanmean(a0[idx]), g2, rng))
        report["paired_vs_reference"][f"{name} - {ref}"] = entry

    # ---- metadata baseline (identical depth, width, vocabulary and tokenizer => tie)
    meta = {ck: prof[next(k for k in prof if k[0] == ck)][1] for ck in ckpts}
    key = {ck: (meta[ck]["n_layers"], meta[ck]["hidden_size"], meta[ck]["vocab_size"], meta[ck]["tokenizer_hash"])
           for ck in ckpts}
    Dm = np.array([[0.0 if key[x] == key[y] else 1.0 for y in ckpts] for x in ckpts])
    sims = [retr_scores(Dm + 1e-9 * rng.random(Dm.shape), [fam_of[c] for c in ckpts]) for _ in range(2000)]
    report["metadata_baseline"] = dict(
        identification_expected=float(np.mean([1.0 / sum(key[c] == key[d] for d in ckpts) for c in ckpts])),
        retrieval_nn_expected=float(np.mean([np.mean([r["hit"] for r in s]) for s in sims if s])),
        note="metadata ties are broken at random; a family whose members share all four keys is retrieved perfectly")

    # ---- E. field direction
    tasks = sorted({k[1] for k in prof if k[1].startswith("task_")})
    report["field_direction"] = field_direction(prof, ckpts, tasks, fam_of)

    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump(report, f, indent=1, default=float)

    # ---- printed and LaTeX summaries
    lines = ["% generated by exp16_evaluate.py", "% method & ID (A->B) & ID 95% CI & NN (B) & MRR (B) & AUC (B) \\\\"]
    print(f"\ncheckpoints: {', '.join(ckpts)}")
    print(f"{'method':22s} {'ID A->B':>9s} {'ID CI':>15s} {'NN B':>7s} {'MRR':>6s} {'AUC':>6s}")
    for name in methods:
        idr = report["identification"].get(name, {}).get("squad_A->squad_B", {})
        rr = report["retrieval"].get(name, {}).get("squad_B", {})
        ci = idr.get("ci95", [None, None])
        cis = f"[{ci[0]:.2f}, {ci[1]:.2f}]" if ci[0] is not None else "n/a"
        print(f"{name:22s} {idr.get('hits', '-'):>9s} {cis:>15s} {rr.get('nn', '-'):>7s} "
              f"{rr.get('mrr', float('nan')):6.2f} {rr.get('auc', float('nan')):6.2f}")
        lines.append(f"{name} & {idr.get('hits', '-')} & {cis} & {rr.get('nn', '-')} & "
                     f"{rr.get('mrr', float('nan')):.2f} & {rr.get('auc', float('nan')):.2f} \\\\")
    with open(os.path.join(out, "table_matched.tex"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nmetadata baseline: {report['metadata_baseline']}")
    for ck, r in report["field_direction"]["per_checkpoint"].items():
        al = r.get("all", {})
        print(f"field direction {ck}: split-half {al.get('split_half_cos', float('nan')):.3f} vs cross-task "
              f"{al.get('cross_task_cos', float('nan')):.3f}; task ID by direction {al.get('task_id_direction')}, "
              f"by norm {r['task_id_norm']}")
    print(f"\nwrote {os.path.join(out, 'summary.json')} and table_matched.tex")


if __name__ == "__main__":
    main()
