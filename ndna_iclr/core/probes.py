"""Probe corpora: the text the triad is measured on.

One rule throughout the suite: a base model and its modified counterpart are always
profiled on *identical* text in an identical order, so the operation is the only
difference between the two profiles.
"""
from __future__ import annotations

import hashlib
import random


def _stable_sample(rows, n, seed=0):
    rows = list(rows)
    random.Random(seed).shuffle(rows)
    return rows[:n]


def load_probe(name="squad_v2", n=256, seed=0, cache_dir=None):
    """-> (prompts, references). Falls back to a synthetic corpus if datasets is absent."""
    try:
        from datasets import load_dataset
    except ImportError:
        return _synthetic(n, seed), [""] * n

    spec = {
        "squad_v2": ("rajpurkar/squad_v2", None, "validation"),
        "ai2_arc": ("allenai/ai2_arc", "ARC-Easy", "validation"),
        "hellaswag": ("Rowan/hellaswag", None, "validation"),
        "winogrande": ("allenai/winogrande", "winogrande_xl", "validation"),
        "mnli": ("nyu-mll/glue", "mnli", "validation_matched"),
        "qqp": ("nyu-mll/glue", "qqp", "validation"),
        "cnn_dailymail": ("abisee/cnn_dailymail", "3.0.0", "validation"),
        "common_gen": ("allenai/common_gen", None, "validation"),
        "wmt16": ("wmt/wmt16", "de-en", "validation"),
        "imdb": ("stanfordnlp/imdb", None, "test"),
        "hh_rlhf": ("Anthropic/hh-rlhf", None, "test"),
        "gsm8k": ("openai/gsm8k", "main", "test"),
    }[name]

    path, cfg, split = spec
    ds = load_dataset(path, cfg, split=split, cache_dir=cache_dir) if cfg \
        else load_dataset(path, split=split, cache_dir=cache_dir)
    ds = ds.select(range(min(len(ds), max(n * 4, n))))
    rows = _stable_sample(list(ds), n, seed)
    return _render(name, rows)


def _render(name, rows):
    P, R = [], []
    for r in rows:
        if name == "squad_v2":
            ans = r["answers"]["text"]
            P.append(f"Context: {r['context']}\nQuestion: {r['question']}\nAnswer:")
            R.append(ans[0] if ans else "")
        elif name == "cnn_dailymail":
            P.append(f"Article: {r['article'][:2000]}\nSummary:")
            R.append(r["highlights"])
        elif name == "wmt16":
            tr = r["translation"]
            P.append(f"Translate German to English.\nGerman: {tr['de']}\nEnglish:")
            R.append(tr["en"])
        elif name == "common_gen":
            P.append("Write a sentence using: " + ", ".join(r["concepts"]) + "\nSentence:")
            R.append(r.get("target", ""))
        elif name == "imdb":
            P.append(f"Review: {r['text'][:1500]}\nSentiment (positive or negative):")
            R.append("positive" if r["label"] == 1 else "negative")
        elif name == "mnli":
            P.append(f"Premise: {r['premise']}\nHypothesis: {r['hypothesis']}\n"
                     f"Relationship (entailment, neutral, contradiction):")
            R.append(["entailment", "neutral", "contradiction"][r["label"]])
        elif name == "qqp":
            P.append(f"Q1: {r['question1']}\nQ2: {r['question2']}\nDuplicate (yes or no):")
            R.append("yes" if r["label"] == 1 else "no")
        elif name == "ai2_arc":
            ch = r["choices"]
            opts = " ".join(f"({l}) {t}" for l, t in zip(ch["label"], ch["text"]))
            P.append(f"Question: {r['question']}\nOptions: {opts}\nAnswer:")
            R.append(r["answerKey"])
        elif name == "hellaswag":
            P.append(r["ctx"] + "\nContinuation:")
            R.append(r["endings"][int(r["label"])] if r.get("label") != "" else "")
        elif name == "winogrande":
            P.append(f"{r['sentence']}\nOption1: {r['option1']} Option2: {r['option2']}\nAnswer:")
            R.append(r["option1"] if r["answer"] == "1" else r["option2"])
        elif name == "gsm8k":
            P.append(f"Question: {r['question']}\nAnswer:")
            R.append(r["answer"])
        elif name == "hh_rlhf":
            P.append(r["chosen"][:1500])
            R.append(r["chosen"][:1500])
        else:
            P.append(str(r))
            R.append("")
    return P, R


def _synthetic(n, seed):
    """Deterministic filler so the suite is runnable without network access."""
    rng = random.Random(seed)
    stems = ["The capital of", "A common cause of", "In order to compute",
             "The main difference between", "One consequence of"]
    tails = ["is generally understood to be", "can be described as",
             "requires that we first", "depends primarily on"]
    return [f"{rng.choice(stems)} item {i} {rng.choice(tails)}" for i in range(n)]


def probe_hash(prompts):
    h = hashlib.sha256()
    for p in prompts:
        h.update(p.encode("utf-8", "replace"))
    return h.hexdigest()[:12]
