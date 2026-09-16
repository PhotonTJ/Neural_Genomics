"""Behavioural metrics: the x-axis against which geometric change is plotted.

These exist for one reason. The paper's claim is that operations which leave behaviour
nearly unchanged still deform internal computation, and that claim is unreadable unless
behaviour is measured and shown to be nearly unchanged. Perplexity is the primary axis;
BLEU / ROUGE-L / EM / F1 / accuracy are reported only for the tasks whose output format
requires them, and never as headline evidence of model quality.
"""
from __future__ import annotations

import collections
import math
import re

import numpy as np

from ._optional import load_torch

torch, TORCH_AVAILABLE = load_torch()


def _need_torch():
    if not TORCH_AVAILABLE:
        raise ImportError("this metric requires torch; install it or use the string metrics")


# --------------------------------------------------------------------------- ppl
def perplexity(model, tok, texts, device="cuda", batch_size=4, max_length=384):
    _need_torch()
    nll, ntok = 0.0, 0
    ctx = torch.no_grad()
    ctx.__enter__()
    for i in range(0, len(texts), batch_size):
        enc = tok(texts[i:i + batch_size], return_tensors="pt", padding=True,
                  truncation=True, max_length=max_length)
        enc = {k: v.to(device) for k, v in enc.items()}
        ids, am = enc["input_ids"], enc["attention_mask"]
        logits = model(**enc).logits.to(torch.float32)
        lp = torch.log_softmax(logits[:, :-1], -1)
        tgt = ids[:, 1:]
        mask = am[:, 1:].bool()
        tok_lp = lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
        nll += float(-(tok_lp[mask].sum()))
        ntok += int(mask.sum())
    ctx.__exit__(None, None, None)
    return math.exp(nll / max(ntok, 1))


# --------------------------------------------------------------------------- surface
def _tokens(s):
    return re.findall(r"[a-z0-9]+", s.lower())


def _lcs(a, b):
    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            dp[i][j] = dp[i - 1][j - 1] + 1 if a[i - 1] == b[j - 1] \
                else max(dp[i - 1][j], dp[i][j - 1])
    return dp[-1][-1]


def rouge_l(pred, ref):
    p, r = _tokens(pred), _tokens(ref)
    if not p or not r:
        return 0.0
    l = _lcs(p, r)
    if l == 0:
        return 0.0
    prec, rec = l / len(p), l / len(r)
    return 2 * prec * rec / (prec + rec)


def bleu(pred, ref, n_max=4):
    """Sentence BLEU with add-1 smoothing on higher orders."""
    p, r = _tokens(pred), _tokens(ref)
    if not p or not r:
        return 0.0
    logs = []
    for n in range(1, n_max + 1):
        pc = collections.Counter(tuple(p[i:i + n]) for i in range(len(p) - n + 1))
        rc = collections.Counter(tuple(r[i:i + n]) for i in range(len(r) - n + 1))
        overlap = sum(min(c, rc[g]) for g, c in pc.items())
        total = max(sum(pc.values()), 1)
        logs.append(math.log((overlap + (0 if n == 1 else 1)) / (total + (0 if n == 1 else 1))
                             if (overlap or n > 1) else 1e-9))
    bp = 1.0 if len(p) > len(r) else math.exp(1 - len(r) / max(len(p), 1))
    return bp * math.exp(sum(logs) / n_max)


def _norm_answer(s):
    s = re.sub(r"\b(a|an|the)\b", " ", s.lower())
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    return " ".join(s.split())


def exact_match(pred, ref):
    return float(_norm_answer(pred) == _norm_answer(ref))


def token_f1(pred, ref):
    p, r = _norm_answer(pred).split(), _norm_answer(ref).split()
    if not p or not r:
        return float(p == r)
    common = collections.Counter(p) & collections.Counter(r)
    ns = sum(common.values())
    if ns == 0:
        return 0.0
    prec, rec = ns / len(p), ns / len(r)
    return 2 * prec * rec / (prec + rec)


def accuracy(preds, refs):
    return float(np.mean([float(a == b) for a, b in zip(preds, refs)])) if preds else 0.0


# --------------------------------------------------------------------------- driver
def generate(model, tok, prompts, device="cuda", max_new_tokens=48, batch_size=4):
    _need_torch()
    outs = []
    ctx = torch.no_grad(); ctx.__enter__()
    for i in range(0, len(prompts), batch_size):
        enc = tok(prompts[i:i + batch_size], return_tensors="pt", padding=True,
                  truncation=True, max_length=384)
        enc = {k: v.to(device) for k, v in enc.items()}
        gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                             pad_token_id=tok.pad_token_id)
        for j in range(gen.shape[0]):
            new = gen[j, enc["input_ids"].shape[1]:]
            outs.append(tok.decode(new, skip_special_tokens=True).strip())
    ctx.__exit__(None, None, None)
    return outs


METRIC_FOR_TASK = {
    "cnn_dailymail": "rouge_l",
    "wmt16": "bleu",
    "squad_v2": "f1",
    "common_gen": "rouge_l",
    "ai2_arc": "accuracy",
    "hellaswag": "accuracy",
    "winogrande": "accuracy",
    "mnli": "accuracy",
    "qqp": "accuracy",
    "imdb": "accuracy",
}


def score(task, preds, refs):
    m = METRIC_FOR_TASK.get(task, "accuracy")
    if m == "rouge_l":
        return float(np.mean([rouge_l(p, r) for p, r in zip(preds, refs)]))
    if m == "bleu":
        return float(np.mean([bleu(p, r) for p, r in zip(preds, refs)]))
    if m == "f1":
        return float(np.mean([token_f1(p, r) for p, r in zip(preds, refs)]))
    if m == "em":
        return float(np.mean([exact_match(p, r) for p, r in zip(preds, refs)]))
    return accuracy(preds, refs)
