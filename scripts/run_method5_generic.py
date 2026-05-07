"""
scripts/run_method5_generic.py

Run Method 5 metrics (E_l, FR geometry, belief norms, dataset-averaged kappa, nDNA_pred)
for configurable models and datasets.

Key correctness points:
- Curvature κ is computed as an expectation over supervised (x,t) positions via
  geo.spectral_curvature_for_loader (teacher-forced labels != -100).
- Dataset shuffle handling:
    * non-streaming Dataset.shuffle(seed=...)  (NO buffer_size)
    * streaming IterableDataset.shuffle(seed=..., buffer_size=...)
  (HF docs show this split.)

IMPORTANT CAVEAT (you asked for this behavior):
- Many of your dataset “inputs” are prompt-only (they end where generation would begin).
  With the current Method-5 implementation (teacher-forced labels from collate_causal),
  κ / belief norms / FR are computed over the *prompt tokens*, not over newly generated tokens.
  If you truly want “belief while generating description/solution”, you need target text present
  (or a separate self-generation trajectory pipeline). This script implements exactly what you asked.
"""

from __future__ import annotations

import os
import argparse
import json
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from datasets import load_dataset, Dataset
from datasets import concatenate_datasets, interleave_datasets
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM, AutoModelForImageTextToText

try:
    from transformers import MistralCommonBackend
except Exception:
    MistralCommonBackend = None  # type: ignore

# Optional quant config (don’t hard-crash older transformers)
try:
    from transformers import Mxfp4Config
except Exception:
    Mxfp4Config = None  # type: ignore

import ndna_lib.geometry as geo
import ndna_lib.data as data_lib
from ndna_lib.models import load_model_zoo

WINOGRANDE_DEFAULT_CONFIG = "winogrande_xl"

# -----------------------------
# Dataset formatters (match your "Input to model:" specs)
# -----------------------------

WMT16_SUBSETS: List[str] = ["cs-en", "de-en", "fi-en", "ro-en", "ru-en", "tr-en"]
_WMT16_SRC_LANG: Dict[str, str] = {
    "cs-en": "cs",
    "de-en": "de",
    "fi-en": "fi",
    "ro-en": "ro",
    "ru-en": "ru",
    "tr-en": "tr",
}
_WMT16_LANG_NAME: Dict[str, str] = {
    "cs": "Czech",
    "de": "German",
    "fi": "Finnish",
    "ro": "Romanian",
    "ru": "Russian",
    "tr": "Turkish",
}


def _build_wmt16_input(src_lang: str, src_text: str) -> str:
    src_text = (src_text or "").strip()
    lang_name = _WMT16_LANG_NAME.get(src_lang, src_lang)
    return (
        "Translate to English.\n\n"
        f"{lang_name}:\n{src_text}\n\n"
        "English:"
    )


def _wmt16_example_to_text(ex: Dict[str, Any], *, src_lang: str) -> Optional[str]:
    tr = ex.get("translation", None)
    if not isinstance(tr, dict):
        return None
    src = _safe_strip(tr.get(src_lang, ""))
    if not src:
        return None
    return _build_wmt16_input(src_lang, src)


def _allocate_equal_counts(total: int, k: int) -> List[int]:
    """
    Split `total` into k non-negative ints as evenly as possible.
    First (total % k) buckets get +1.
    """
    base = total // k
    rem = total % k
    return [base + (1 if i < rem else 0) for i in range(k)]

def _safe_strip(x: Any) -> str:
    if x is None:
        return ""
    return str(x).strip()


def _build_squad_input(context: str, question: str) -> str:
    context = (context or "").strip()
    question = (question or "").strip()
    return f"Context: {context}\nQuestion: {question}\n"


def _build_mnli_input(premise: str, hypothesis: str) -> str:
    premise = (premise or "").strip()
    hypothesis = (hypothesis or "").strip()
    return (
        "Determine the relationship between the premise and the hypothesis.\n"
        "Answer with exactly one label from: entailment, neutral, contradiction\n\n"
        f"Premise:\n{premise}\n\n"
        f"Hypothesis:\n{hypothesis}\n\n"
        "Label:"
    )

def _build_cnn_dailymail_input(article: str) -> str:
    a = (article or "").strip()
    return (
        "Summarize the following news article.\n\n"
        f"Article:\n{a}\n\n"
        "Summary:"
    )

def _build_ai2_arc_input(stem: str, choices: Any) -> str:
    q = (stem or "").strip()

    items: List[Tuple[str, str]] = []
    if isinstance(choices, list):
        for c in choices:
            if isinstance(c, dict):
                lab = _safe_strip(c.get("label", ""))
                txt = _safe_strip(c.get("text", ""))
                if txt:
                    items.append((lab, txt))

    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    lines: List[str] = []
    for i, (lab, txt) in enumerate(items):
        lab2 = lab if lab else (letters[i] if i < len(letters) else str(i + 1))
        lines.append(f"{lab2}) {txt}")

    opts = "\n".join(lines).strip()

    return (
        "Choose the correct answer to the question.\n"
        "Answer with a single letter.\n\n"
        f"Question:\n{q}\n\n"
        f"Choices:\n{opts}\n\n"
        "Answer:"
    )

def _build_qqp_input(q1: str, q2: str) -> str:
    q1 = (q1 or "").strip()
    q2 = (q2 or "").strip()
    return (
        "Determine whether the two questions are semantically equivalent.\n"
        "Answer with exactly one label: duplicate or not_duplicate\n\n"
        f"Question 1:\n{q1}\n\n"
        f"Question 2:\n{q2}\n\n"
        "Label:"
    )

def _parse_common_gen_concepts(ex: Dict[str, Any]) -> List[str]:
    """
    CommonGen can store concepts as:
      - concept_set: list[str]
      - concepts: str like "dog#park#run"
    Return a clean list[str].
    """
    raw = ex.get("concept_set", None)
    if raw is None:
        raw = ex.get("concepts", None)

    if isinstance(raw, list):
        return [_safe_strip(x) for x in raw if _safe_strip(x)]

    s = _safe_strip(raw)
    if not s:
        return []

    # Common delimiter in CommonGen is '#', but be tolerant.
    if "#" in s:
        parts = s.split("#")
    elif "," in s:
        parts = s.split(",")
    elif ";" in s:
        parts = s.split(";")
    elif "|" in s:
        parts = s.split("|")
    else:
        parts = [s]

    return [p.strip() for p in parts if p.strip()]


def _build_common_gen_input(concepts: Any) -> str:
    # accept list[str] (preferred) or string fallback
    cs: List[str]
    if isinstance(concepts, list):
        cs = [(_safe_strip(c)) for c in concepts if _safe_strip(c)]
    else:
        c = _safe_strip(concepts)
        if not c:
            cs = []
        else:
            cs = [p.strip() for p in c.split("#") if p.strip()] if "#" in c else [c]

    return (
        "Write one coherent sentence that uses all the given concepts.\n"
        "Use each concept at least once.\n\n"
        f"Concepts: {', '.join(cs)}\n\n"
        "Sentence:"
    )

def _plato_main_text_only(ex: Dict[str, Any]) -> str:
    """
    Stanford Plato: ONLY main_text (no title, no preamble).
    main_text is a list of section dicts; flatten paragraphs.
    """
    main_text = ex.get("main_text", None)
    if main_text is None:
        return ""

    if isinstance(main_text, str):
        return main_text.strip()

    parts: List[str] = []
    if isinstance(main_text, list):
        for section in main_text:
            if not isinstance(section, dict):
                continue

            main_content = section.get("main_content", [])
            if isinstance(main_content, list):
                for p in main_content:
                    s = _safe_strip(p)
                    if s:
                        parts.append(s)
            elif isinstance(main_content, str):
                s = main_content.strip()
                if s:
                    parts.append(s)

            subsections = section.get("subsections", [])
            if isinstance(subsections, list):
                for sub in subsections:
                    if not isinstance(sub, dict):
                        continue
                    sub_content = sub.get("content", [])
                    if isinstance(sub_content, list):
                        for p in sub_content:
                            s = _safe_strip(p)
                            if s:
                                parts.append(s)
                    elif isinstance(sub_content, str):
                        s = sub_content.strip()
                        if s:
                            parts.append(s)

    return "\n".join(parts).strip()


def _split_ag_news_title_desc(text: str) -> Tuple[str, str]:
    """
    Heuristic split for AG News style strings that often look like:
      "<TITLE> ... - <DESCRIPTION> ..."
    """
    s = (text or "").strip()
    if not s:
        return "", ""

    if " - " in s:
        left, right = s.split(" - ", 1)
        return left.strip(), right.strip()

    dot = s.find(". ")
    if 10 <= dot <= 160:
        return s[:dot].strip(), s[dot + 2 :].strip()

    return s[:120].strip(), s[120:].strip()


def _build_ag_news_prompt(title: str) -> str:
    title = (title or "").strip()
    return f"Title: {title}\ndescription:"


def _mbpp_build_input(ex: Dict[str, Any]) -> str:
    """
    MBPP: build prompt from:
      text/prompt, test_setup_code/test_imports, test_list, challenge_test_list
    and end with '# Solution' marker.
    """
    task = _safe_strip(ex.get("text", "")) or _safe_strip(ex.get("prompt", ""))
    setup = _safe_strip(ex.get("test_setup_code", "")) or _safe_strip(ex.get("test_imports", "")) or _safe_strip(ex.get("test_setup", ""))

    test_list = ex.get("test_list", [])
    challenge_list = ex.get("challenge_test_list", [])

    lines: List[str] = []
    if task:
        lines.append(task)

    if setup:
        lines.append("\n# Setup\n" + setup)

    if isinstance(test_list, list) and test_list:
        lines.append("\n# Tests")
        lines.extend([f"- {_safe_strip(t)}" for t in test_list if _safe_strip(t)])

    if isinstance(challenge_list, list) and challenge_list:
        lines.append("\n# Challenge tests")
        lines.extend([f"- {_safe_strip(t)}" for t in challenge_list if _safe_strip(t)])

    lines.append("\n# Solution\n")
    return "\n".join(lines).strip() + "\n"


def _build_hellaswag_input(context: str, endings: Any) -> str:
    ctx = (context or "").strip()

    opts: List[str] = []
    if isinstance(endings, list):
        opts = [(_safe_strip(x)) for x in endings]
    else:
        opts = [_safe_strip(endings)]

    while len(opts) < 4:
        opts.append("")

    return (
        "Choose the most likely continuation of the context.\n\n"
        f"Context:\n{ctx}\n\n"
        "Options:\n"
        f"A) {opts[0]}\n"
        f"B) {opts[1]}\n"
        f"C) {opts[2]}\n"
        f"D) {opts[3]}\n\n"
        "Answer: "
    )

def _build_imdb_input(review: str) -> str:
    r = (review or "").strip()
    return (
        "Classify the sentiment of the movie review.\n"
        "Answer with exactly one label: positive or negative\n\n"
        f"Review:\n{r}\n\n"
        "Label:"
    )

def _build_winogrande_input(sentence: str, option1: str, option2: str) -> str:
    s = (sentence or "").strip()
    o1 = (option1 or "").strip()
    o2 = (option2 or "").strip()
    return (
        "Fill in the blank (_) with the correct option.\n"
        "Answer with exactly one letter: A or B\n\n"
        f"Sentence:\n{s}\n\n"
        "Options:\n"
        f"A) {o1}\n"
        f"B) {o2}\n\n"
        "Answer:"
    )

# -----------------------------
# Loading + materializing texts
# -----------------------------

SUPPORTED_DATASETS = [
    # your requested set
    "squad",
    "squad_v2",
    "mnli",
    "hellaswag",
    "imdb",
    "winogrande",
    "cnn_dailymail",
    "ai2_arc",
    "qqp",
    "common_gen",
    "wmt16",
    "stanford_plato",
    "ag_news",
    "hh_rlhf",
    "gsm8k",
    "harmbench",
    "litmus",
    "mbpp",
    "alpaca",
    "everything_instruct_multilingual",
    # keep old keys as aliases/backcompat
    "ag_news_setfit",
    "ag_news_business",
    "automathtext",
    "automathtext_web",
    "wikipedia",
]


def _load_and_materialize_texts(
    *,
    dataset_key: str,
    split: str,
    streaming: bool,
    shuffle: bool,
    seed: int,
    max_samples: int,
    shuffle_buffer_size: int,
    # dataset-specific knobs
    ag_label: Optional[int],
    hh_data_dir: str,
    gsm8k_config: str,
    harmbench_config: str,
    mbpp_config: str,
    wiki_config: str,
    alpaca_dataset_id: str,
    automath_config: str,
    mnli_config: str,
    winogrande_config: str,
) -> Dataset:
    """
    Returns an in-memory HF Dataset with a single column: 'text'.
    We sample up to max_samples (after optional filtering).
    """

    if max_samples <= 0:
        raise ValueError("--max-samples must be > 0")

    # Alias normalization
    key = dataset_key
    if key == "ag_news_setfit":
        key = "ag_news"
    if key == "automathtext_web":
        key = "automathtext"

    # Force streaming for known-huge datasets unless user explicitly asked otherwise
    if key in ["everything_instruct_multilingual", "automathtext", "wikipedia"] and not streaming:
        print(f"[warn] dataset '{key}' can be huge; forcing --streaming for reliability.")
        streaming = True

    # Load + choose text extractor
    if key == "squad":
        raw = load_dataset("squad", split=split, streaming=streaming)

        def ex_to_text(ex):
            return _build_squad_input(_safe_strip(ex.get("context", "")), _safe_strip(ex.get("question", "")))

    elif key == "squad_v2":
        raw = load_dataset("rajpurkar/squad_v2", split=split, streaming=streaming)

        def ex_to_text(ex):
            return _build_squad_input(_safe_strip(ex.get("context", "")), _safe_strip(ex.get("question", "")))

    elif key == "mnli":
        # GLUE MNLI (nyu-mll/glue, config="mnli" by default)
        raw = load_dataset("nyu-mll/glue", mnli_config, split=split, streaming=streaming)

        def ex_to_text(ex):
            premise = _safe_strip(ex.get("premise", ""))
            hypothesis = _safe_strip(ex.get("hypothesis", ""))
            if not premise or not hypothesis:
                return None
            return _build_mnli_input(premise, hypothesis)

    elif key == "hellaswag":
        raw = load_dataset("Rowan/hellaswag", split=split, streaming=streaming)

        def ex_to_text(ex):
            ctx = _safe_strip(ex.get("ctx", ""))
            if not ctx:
                ctx_a = _safe_strip(ex.get("ctx_a", ""))
                ctx_b = _safe_strip(ex.get("ctx_b", ""))
                ctx = (ctx_a + " " + ctx_b).strip()
            endings = ex.get("endings", [])
            if not ctx:
                return None
            return _build_hellaswag_input(ctx, endings)

    elif key == "imdb":
        raw = load_dataset("stanfordnlp/imdb", split=split, streaming=streaming)

        def ex_to_text(ex):
            t = _safe_strip(ex.get("text", ""))
            return _build_imdb_input(t) if t else None

    elif key == "winogrande":
        raw = load_dataset("winogrande", winogrande_config, split=split, streaming=streaming)

        def ex_to_text(ex):
            s = _safe_strip(ex.get("sentence", ""))
            o1 = _safe_strip(ex.get("option1", ""))
            o2 = _safe_strip(ex.get("option2", ""))
            if not s or not o1 or not o2:
                return None
            return _build_winogrande_input(s, o1, o2)

    elif key == "cnn_dailymail":
        raw = load_dataset("cnn_dailymail", "3.0.0", split=split, streaming=streaming)

        def ex_to_text(ex):
            art = _safe_strip(ex.get("article", ""))
            return _build_cnn_dailymail_input(art) if art else None

    elif key == "ai2_arc":
        # NOTE: We intentionally ignore `split` and combine train+validation+test
        ds_train = load_dataset("ai2_arc", "ARC-Challenge", split="train", streaming=streaming)
        ds_val = load_dataset("ai2_arc", "ARC-Challenge", split="validation", streaming=streaming)
        ds_test = load_dataset("ai2_arc", "ARC-Challenge", split="test", streaming=streaming)

        if streaming:
            # IterableDataset-friendly "combine"
            raw = interleave_datasets([ds_train, ds_val, ds_test])
        else:
            raw = concatenate_datasets([ds_train, ds_val, ds_test])

        def ex_to_text(ex):
            q = ex.get("question", {})
            if isinstance(q, dict):
                stem = _safe_strip(q.get("stem", ""))
                choices = q.get("choices", [])
            else:
                stem = _safe_strip(ex.get("question", ""))
                choices = ex.get("choices", [])
            if not stem:
                return None
            return _build_ai2_arc_input(stem, choices)

    elif key == "qqp":
        raw = load_dataset("nyu-mll/glue", "qqp", split=split, streaming=streaming)

        def ex_to_text(ex):
            q1 = _safe_strip(ex.get("question1", ""))
            q2 = _safe_strip(ex.get("question2", ""))
            if not q1 or not q2:
                return None
            return _build_qqp_input(q1, q2)

    elif key == "common_gen":
        raw = load_dataset("allenai/common_gen", split=split, streaming=streaming)

        def ex_to_text(ex):
            concepts_list = _parse_common_gen_concepts(ex)
            if not concepts_list:
                return None
            return _build_common_gen_input(concepts_list)

    elif key == "wmt16":
        # Equal combination across 6 subsets; always translate -> English.
        counts = _allocate_equal_counts(int(max_samples), len(WMT16_SUBSETS))

        mapped_list: List[Dataset] = []
        for subset, n_i in zip(WMT16_SUBSETS, counts):
            if n_i <= 0:
                continue
            src_lang = _WMT16_SRC_LANG[subset]
            ds_i = load_dataset("wmt16", subset, split=split, streaming=streaming)

            # Map to a unified {"text": ...} schema so we can combine datasets safely.
            def _map_fn(ex, _src_lang=src_lang):
                t = _wmt16_example_to_text(ex, src_lang=_src_lang)
                # Keep empty string if unusable; we filter later by returning None in ex_to_text
                return {"text": t or ""}

            ds_i = ds_i.map(_map_fn, remove_columns=ds_i.column_names)

            if not streaming:
                # Make per-subset sampling deterministic and independent of later global shuffle.
                ds_i = ds_i.shuffle(seed=seed)
                ds_i = ds_i.select(range(min(n_i, len(ds_i))))

            mapped_list.append(ds_i)

        if not mapped_list:
            raise RuntimeError("wmt16: no subsets loaded/mapped.")

        if streaming:
            # Interleave gives an equal-mix stream (before optional later shuffle).
            raw = interleave_datasets(mapped_list, probabilities=[1.0 / len(mapped_list)] * len(mapped_list))
        else:
            raw = concatenate_datasets(mapped_list)

        def ex_to_text(ex):
            t = _safe_strip(ex.get("text", ""))
            return t if t else None

    elif key == "stanford_plato":
        raw = load_dataset("hugfaceguy0001/stanford_plato", split=split, streaming=streaming)

        def ex_to_text(ex):
            t = _plato_main_text_only(ex)
            return t if t else None

    elif key == "ag_news":
        raw = load_dataset("SetFit/ag_news", split=split, streaming=streaming)

        def ex_to_text(ex):
            if ag_label is not None:
                try:
                    if int(ex.get("label", -999)) != int(ag_label):
                        return None
                except Exception:
                    return None
            title, _desc = _split_ag_news_title_desc(_safe_strip(ex.get("text", "")))
            if not title:
                return None
            return _build_ag_news_prompt(title)

    elif key == "ag_news_business":
        # Backcompat: original AG News dataset, label==2 is Business
        raw = load_dataset("ag_news", split=split, streaming=streaming)

        def ex_to_text(ex):
            if int(ex.get("label", -1)) != 2:
                return None
            title, _desc = _split_ag_news_title_desc(_safe_strip(ex.get("text", "")))
            if not title:
                return None
            return _build_ag_news_prompt(title)

    elif key == "hh_rlhf":
        raw = load_dataset("Anthropic/hh-rlhf", data_dir=hh_data_dir, split=split, streaming=streaming)

        def ex_to_text(ex):
            t = _safe_strip(ex.get("rejected", ""))
            return t if t else None

    elif key == "gsm8k":
        raw = load_dataset("openai/gsm8k", gsm8k_config, split=split, streaming=streaming)

        def ex_to_text(ex):
            t = _safe_strip(ex.get("question", ""))
            return t if t else None

    elif key == "harmbench":
        raw = load_dataset("AlignmentResearch/HarmBench", harmbench_config, split=split, streaming=streaming)

        def ex_to_text(ex):
            t = _safe_strip(ex.get("instructions", ""))
            if t:
                return t
            content = ex.get("content", [])
            if isinstance(content, list) and content:
                t = _safe_strip(content[0])
            else:
                t = _safe_strip(content)
            return t if t else None

    elif key == "litmus":
        raw = load_dataset("hasnat79/litmus", split=split, streaming=streaming)

        def ex_to_text(ex):
            t = _safe_strip(ex.get("input", ""))
            return t if t else None

    elif key == "mbpp":
        raw = load_dataset("google-research-datasets/mbpp", mbpp_config, split=split, streaming=streaming)

        def ex_to_text(ex):
            t = _mbpp_build_input(ex)
            return t if t else None

    elif key == "alpaca":
        raw = load_dataset(alpaca_dataset_id, split=split, streaming=streaming)

        def ex_to_text(ex):
            inst = _safe_strip(ex.get("instruction", ""))
            inp = _safe_strip(ex.get("input", ""))
            # prompt-only (your requirement)
            return data_lib.build_alpaca_prompt(inst, inp)

    elif key == "everything_instruct_multilingual":
        raw = load_dataset("rombodawg/Everything_Instruct_Multilingual", split=split, streaming=streaming)

        def ex_to_text(ex):
            inst = _safe_strip(ex.get("instruction", ""))
            inp = _safe_strip(ex.get("input", ""))
            return data_lib.build_alpaca_prompt(inst, inp)

    elif key == "wikipedia":
        raw = load_dataset("wikimedia/wikipedia", wiki_config, split=split, streaming=streaming)

        def ex_to_text(ex):
            t = _safe_strip(ex.get("text", ""))
            return t if t else None

    elif key == "automathtext":
        raw = load_dataset("math-ai/AutoMathText", automath_config, split=split, streaming=streaming)

        def ex_to_text(ex):
            # AutoMathText variants commonly contain 'text'
            t = _safe_strip(ex.get("text", "")) or _safe_strip(ex.get("problem", "")) or _safe_strip(ex.get("question", ""))
            return t if t else None

    else:
        raise ValueError(f"Unknown dataset '{dataset_key}'. Supported: {SUPPORTED_DATASETS} or 'all'.")

    # Shuffle (streaming uses buffer)
    if shuffle:
        if streaming:
            raw = raw.shuffle(seed=seed, buffer_size=int(shuffle_buffer_size))
        else:
            raw = raw.shuffle(seed=seed)

    # Materialize up to max_samples
    texts: List[str] = []
    for ex in raw:
        t = ex_to_text(ex)
        if not t:
            continue
        texts.append(t)
        if len(texts) >= max_samples:
            break

    if not texts:
        raise RuntimeError(f"No usable texts produced for dataset={dataset_key} (after filtering).")

    return Dataset.from_dict({"text": texts})


# -----------------------------
# Model selection
# -----------------------------

def _safe_model_tag(model_name: str) -> str:
    return model_name.replace("/", "_").replace(":", "_").replace("@", "_").replace("-", "_")


def _candidate_tokenizer_sources(model_name: str) -> List[str]:
    sources: List[str] = [model_name]

    if not os.path.isdir(model_name):
        return sources

    for filename in ("tokenizer_config.json", "config.json", "adapter_config.json"):
        path = os.path.join(model_name, filename)
        if not os.path.exists(path):
            continue

        try:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception:
            continue

        for key in ("name_or_path", "_name_or_path", "base_model_name_or_path"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip() and value not in sources:
                sources.append(value.strip())

    return sources


def _load_tokenizer_robust(model_name: str):
    last_error = None
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    model_type = str(getattr(config, "model_type", "") or "").lower()
    sources = _candidate_tokenizer_sources(model_name)

    if model_type == "mistral3" and MistralCommonBackend is not None:
        for source in sources:
            try:
                print(f"[tokenizer] Loading MistralCommonBackend from {source}")
                return MistralCommonBackend.from_pretrained(source)
            except Exception as exc:
                last_error = exc
                print(f"[tokenizer] mistral-common load failed from {source}: {exc}")

    for source in sources:
        for use_fast in (True, False):
            try:
                print(f"[tokenizer] Loading tokenizer from {source} (use_fast={use_fast})")
                return AutoTokenizer.from_pretrained(source, use_fast=use_fast, trust_remote_code=True)
            except Exception as exc:
                last_error = exc
                print(f"[tokenizer] load failed from {source} (use_fast={use_fast}): {exc}")

    raise RuntimeError(f"Failed to load tokenizer for model {model_name}") from last_error


def _load_model_robust(model_name: str, model_kwargs: Dict[str, Any]):
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    model_type = getattr(config, "model_type", "") or ""
    print(f"[model] config={type(config).__name__} model_type={model_type}")

    if str(model_type).lower() == "mistral3":
        print("[model] Using AutoModelForImageTextToText for Mistral3/Ministral3 checkpoint")
        mistral3_kwargs = dict(model_kwargs)
        mistral3_kwargs.pop("use_cache", None)
        return AutoModelForImageTextToText.from_pretrained(
            model_name,
            trust_remote_code=True,
            **mistral3_kwargs,
        )

    return AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        **model_kwargs,
    )


def _resolve_models(
    *,
    hf_models: Optional[List[str]],
    model_keys: Optional[List[str]],
    zoo_path: str,
) -> List[Tuple[str, str]]:
    """
    Resolve which models to run based on model_zoo.json and enabled flags.

    Returns a list of (alias, hf_id) tuples.
    """
    zoo = load_model_zoo(zoo_path)

    if hf_models and model_keys:
        raise ValueError("Specify only one of --hf-models or --model-keys (or neither).")

    if hf_models is not None:
        aliases = list(hf_models)
    elif model_keys is not None and len(model_keys) > 0:
        aliases = list(model_keys)
    else:
        aliases = [k for k, v in zoo.items() if v.get("enabled", False)]

    resolved: List[Tuple[str, str]] = []
    for alias in aliases:
        if alias not in zoo:
            print(f"[warn] alias '{alias}' not in {zoo_path}, skipping.")
            continue
        entry = zoo[alias]
        if not entry.get("enabled", False):
            print(f"[warn] alias '{alias}' is disabled in {zoo_path}, skipping.")
            continue
        hf_id = entry.get("hf_id")
        if not hf_id:
            print(f"[warn] alias '{alias}' missing 'hf_id' in {zoo_path}, skipping.")
            continue
        resolved.append((alias, str(hf_id)))

    if not resolved:
        raise ValueError("No models to run after filtering by model_zoo.json enabled flags.")

    return resolved


# -----------------------------
# One run
# -----------------------------

def run_one(
    *,
    model_name: str,
    ds_text: Dataset,
    out_dir: str,
    dataset_tag: str,
    max_len: int,
    batch_size: int,
    tokens_per_ex: int,
    tau: float,
    fisher_unit: str,
    compute_kappa: bool,
    kappa_keep_last_k: Optional[int],
    kappa_include_embedding_node: bool,
    use_mxfp4: bool,
) -> str:
    device = geo.DEVICE
    print(f"\n====================")
    print(f"Dataset: {dataset_tag}")
    print(f"Model:   {model_name}")
    print(f"Device:  {device}")
    print(f"Samples: {len(ds_text)}")
    print(f"====================")

    tokenizer = _load_tokenizer_robust(model_name)
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            raise ValueError("Tokenizer has no pad_token or eos_token.")

    model_kwargs = dict(
        attn_implementation="eager",
        torch_dtype=torch.bfloat16,
        use_cache=False,
        device_map="auto",
    )

    if use_mxfp4:
        if Mxfp4Config is None:
            raise RuntimeError("Mxfp4Config not available in this transformers version. Use --no-mxfp4.")
        model_kwargs["quantization_config"] = Mxfp4Config(dequantize=True)

    model = _load_model_robust(model_name, model_kwargs)
    model.eval()

    adapter = geo.make_adapter(model, model_name)
    print(f"Adapter: {type(adapter).__name__} | Layers: {adapter.num_layers}")

    collate_fn = geo.collate_causal(tokenizer, max_len=max_len)

    def make_loader():
        return DataLoader(ds_text, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # 1) Parameter-space effort
    print("\n[1] E_l (observed Fisher / param effort)...")
    E_l, n_ex, n_params = geo.compute_param_effort(
        model=model,
        adapter=adapter,
        loader=make_loader(),
        unit=fisher_unit,
    )
    print(f"  Used examples/tokens count: {n_ex} (unit={fisher_unit})")

    # 2) FR geometry
    print("\n[2] FR geometry (Delta, Alpha, Vnorm)...")
    Delta, Alpha, Vnorm, mean_total_fr, n_tokens = geo.compute_fr_and_alignment_streaming(
        model=model,
        adapter=adapter,
        loader=make_loader(),
    )
    print(f"  FR valid tokens: {n_tokens}")
    print(f"  Mean total FR length/token: {mean_total_fr:.6e} rad")

    # 3) Belief field norms
    print("\n[3] Belief field norms ||v_l|| (belief_tangent_lastk) ...")
    _, belief_tangent_lastk, _ = geo.compute_two_belief_methods_streaming(
        model=model,
        adapter=adapter,
        loader=make_loader(),
        tau=tau,
        keep_last_k=tokens_per_ex,
        fr_norm=True,
    )
    belief_norms = belief_tangent_lastk

    # 4) Kappa over (x,t) and nDNA_pred
    kappa = None
    kappa_positions = None
    ndna_scalar_val = None
    ndna_layerwise_arr = None

    if compute_kappa:
        print("\n[4] Kappa (dataset expectation over supervised (x,t)) ...")
        kappa, kappa_positions = geo.spectral_curvature_for_loader(
            model=model,
            adapter=adapter,
            loader=make_loader(),
            tau=tau,
            keep_last_k=kappa_keep_last_k,
            include_embedding_node=bool(kappa_include_embedding_node),
        )
        print(f"  Kappa shape: {kappa.shape} | positions used: {kappa_positions}")

        if kappa.shape == Delta.shape:
            v_dict = {"concept": belief_norms}
            ndna_scalar, ndna_layerwise = geo.compute_ndna_pred(
                kappa=kappa,
                fr_steps=Delta,
                v_norms_by_concept=v_dict,
                l_min=2,
            )
            ndna_scalar_val = float(ndna_scalar["concept"])
            ndna_layerwise_arr = np.asarray(ndna_layerwise["concept"], dtype=float)
            print(f"  nDNA_pred(concept) = {ndna_scalar_val:.6e}")
        else:
            print(f"  [warn] kappa shape {kappa.shape} != Delta shape {Delta.shape}. Skipping nDNA_pred.")

    # Save
    os.makedirs(out_dir, exist_ok=True)
    tag = _safe_model_tag(model_name)
    ds_tag = dataset_tag.replace("/", "_")
    out_path = os.path.join(out_dir, f"{ds_tag}__method5_{tag}.npz")

    save_kwargs = dict(
        model=np.array([model_name]),
        dataset=np.array([dataset_tag]),
        E_l=E_l,
        n_examples=np.array([n_ex], dtype=np.int64),
        n_params=np.array(n_params, dtype=np.int64),
        Delta=Delta,
        Alpha=Alpha,
        Vnorm=Vnorm,
        mean_total_fr=np.array([mean_total_fr], dtype=np.float64),
        n_tokens=np.array([n_tokens], dtype=np.int64),
        belief_norms=belief_norms,
        belief_tangent_lastk=belief_tangent_lastk,
        tau=np.array([tau], dtype=np.float64),
        max_len=np.array([max_len], dtype=np.int64),
        batch_size=np.array([batch_size], dtype=np.int64),
        tokens_per_ex=np.array([tokens_per_ex], dtype=np.int64),
        fisher_unit=np.array([fisher_unit]),
    )

    if kappa is not None:
        save_kwargs["kappa"] = kappa
        save_kwargs["kappa_positions"] = np.array([kappa_positions], dtype=np.int64)

    if ndna_scalar_val is not None and ndna_layerwise_arr is not None:
        save_kwargs["ndna_scalar"] = np.array([ndna_scalar_val], dtype=np.float64)
        save_kwargs["ndna_layerwise"] = ndna_layerwise_arr

    np.savez_compressed(out_path, **save_kwargs)
    print(f"\nSaved: {out_path}")
    return out_path


# -----------------------------
# CLI
# -----------------------------

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--hf-models",
        nargs="+",
        required=False,
        default=None,
        help="Model aliases from model_zoo.json. If omitted, uses --model-keys or all enabled entries.",
    )
    ap.add_argument(
        "--model-keys",
        nargs="*",
        default=None,
        help="Explicit model aliases from model_zoo.json; default: all enabled.",
    )
    ap.add_argument(
        "--zoo-path",
        type=str,
        default="model_zoo.json",
        help="Path to model_zoo.json used to choose models.",
    )

    ap.add_argument(
        "--dataset",
        type=str,
        required=False,
        default=None,
        help=f"Dataset key to use (or 'all'). Supported: {SUPPORTED_DATASETS}",
    )
    ap.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Run multiple datasets in one command (overrides --dataset). Use 'all' to run all supported.",
    )
    ap.add_argument("--list-datasets", action="store_true", help="Print supported dataset keys and exit.")
    ap.add_argument("--split", type=str, default="train")

    ap.add_argument("--streaming", action="store_true", help="Load dataset in streaming mode.")
    ap.add_argument("--no-shuffle", action="store_true", help="Disable dataset shuffling.")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--shuffle-buffer-size", type=int, default=10_000, help="Only used when --streaming.")

    ap.add_argument("--max-samples", type=int, default=32, help="How many texts to materialize.")

    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--tokens-per-ex", type=int, default=8, help="For belief norms: keep last K supervised tokens per sample.")
    ap.add_argument("--tau", type=float, default=1.0)
    ap.add_argument("--fisher-unit", type=str, default="sequence", choices=["sequence", "token"])

    # Dataset-specific knobs
    ap.add_argument("--ag-label", type=int, default=None, help="Filter AG News to this label (SetFit/ag_news).")
    ap.add_argument("--hh-data-dir", type=str, default="helpful-base", help="Anthropic/hh-rlhf data_dir (e.g., helpful-base, harmless-base).")
    ap.add_argument("--gsm8k-config", type=str, default="main", help="openai/gsm8k config (usually 'main').")
    ap.add_argument("--harmbench-config", type=str, default="default", help="AlignmentResearch/HarmBench config (usually 'default').")
    ap.add_argument("--mbpp-config", type=str, default="full", help="google-research-datasets/mbpp subset/config: full | sanitized.")
    ap.add_argument("--wiki-config", type=str, default="20231101.en", help="wikimedia/wikipedia config, e.g. 20231101.en")
    ap.add_argument("--alpaca-dataset-id", type=str, default=data_lib.ALPACA_DATASET_ID, help="Alpaca-style dataset id for 'alpaca' key.")
    ap.add_argument("--automath-config", type=str, default="web-0.50-to-1.00", help="math-ai/AutoMathText config.")
    ap.add_argument("--mnli-config", type=str, default="mnli",
                    help="nyu-mll/glue config for MNLI (default: mnli).")
    ap.add_argument(
        "--winogrande-config",
        type=str,
        default=None,  # allow omission
        help=(
            "winogrande config: winogrande_xl | winogrande_m | winogrande_s | winogrande_debiased "
            f"(default: {WINOGRANDE_DEFAULT_CONFIG})"
        ),
    )

    # Kappa options
    ap.add_argument("--no-kappa", action="store_true", help="Skip kappa and nDNA_pred.")
    ap.add_argument("--kappa-keep-last-k", type=int, default=8,
                    help="For kappa expectation: per sample, keep only last K supervised positions (labels != -100).")
    ap.add_argument("--kappa-include-embedding-node", action="store_true",
                    help="Include embedding node so kappa has length L-1 (shape matches Delta).")

    # Quant option
    ap.add_argument("--no-mxfp4", action="store_true", help="Disable Mxfp4 quantization config.")

    ap.add_argument("--out-dir", type=str, default="results/method5_generic")

    args = ap.parse_args()

    # Normalize optional config
    winogrande_config = (args.winogrande_config or "").strip() or WINOGRANDE_DEFAULT_CONFIG

    if args.list_datasets:
        print("\n".join(SUPPORTED_DATASETS + ["all"]))
        return

    # Determine datasets to run
    if args.datasets is not None and len(args.datasets) > 0:
        ds_list = list(args.datasets)
    else:
        if args.dataset is None:
            raise ValueError("Provide --dataset <key|all> or --datasets <k1 k2 ...>.")
        ds_list = [args.dataset]

    if any(d == "all" for d in ds_list):
        ds_list = [d for d in SUPPORTED_DATASETS]

    # Validate dataset keys early
    for d in ds_list:
        if d not in SUPPORTED_DATASETS:
            raise ValueError(f"Unknown dataset '{d}'. Use --list-datasets to see valid keys.")

    models_to_run = _resolve_models(
        hf_models=args.hf_models,
        model_keys=args.model_keys,
        zoo_path=args.zoo_path,
    )

    shuffle = not args.no_shuffle
    os.makedirs(args.out_dir, exist_ok=True)

    print("[models] Will run Method 5 for:")
    for alias, hf_id in models_to_run:
        print(f"  - {alias}: {hf_id}")

    for dataset_key in ds_list:
        ds_text = _load_and_materialize_texts(
            dataset_key=dataset_key,
            split=args.split,
            streaming=bool(args.streaming),
            shuffle=shuffle,
            seed=int(args.seed),
            max_samples=int(args.max_samples),
            shuffle_buffer_size=int(args.shuffle_buffer_size),
            ag_label=(int(args.ag_label) if args.ag_label is not None else None),
                hh_data_dir=str(args.hh_data_dir),
                gsm8k_config=str(args.gsm8k_config),
                harmbench_config=str(args.harmbench_config),
                mbpp_config=str(args.mbpp_config),
            wiki_config=str(args.wiki_config),
            alpaca_dataset_id=str(args.alpaca_dataset_id),
            automath_config=str(args.automath_config),
            mnli_config=str(args.mnli_config),
            winogrande_config=str(winogrande_config),
        )
        print(f"[data] Materialized {len(ds_text)} texts for dataset={dataset_key}")

        for alias, model_name in models_to_run:
            print(f"\n[run] dataset='{dataset_key}' | alias='{alias}' | model='{model_name}'")
            run_one(
                model_name=model_name,
                ds_text=ds_text,
                out_dir=args.out_dir,
                dataset_tag=dataset_key,
                max_len=int(args.max_len),
                batch_size=int(args.batch_size),
                tokens_per_ex=int(args.tokens_per_ex),
                tau=float(args.tau),
                fisher_unit=str(args.fisher_unit),
                compute_kappa=(not args.no_kappa),
                kappa_keep_last_k=(int(args.kappa_keep_last_k) if args.kappa_keep_last_k is not None else None),
                kappa_include_embedding_node=bool(args.kappa_include_embedding_node),
                use_mxfp4=(not args.no_mxfp4),
            )


if __name__ == "__main__":
    main()
