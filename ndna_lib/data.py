# ndna_lib/data.py
"""
Data utilities for nDNA experiments.

Includes:
  - Alpaca dataset loading / formatting
  - Generic text datasets for Method-5 / geometry runs
  - Building training datasets for causal LM

Key design choice for Method-5 inputs:
  - Each loader in TEXT_DATASET_REGISTRY returns a HF Dataset with a single column: "text".
  - For "prompt-style" tasks, "text" is formatted as an input prompt that *ends where generation begins*
    (e.g., Alpaca prompts end with "### Response:\n", AG News prompts end with "description:", etc.).
"""

from __future__ import annotations

from typing import List, Dict, Optional, Callable, Any, Tuple
from datasets import load_dataset, Dataset
from datasets import concatenate_datasets, interleave_datasets
from transformers import PreTrainedTokenizerBase

ALPACA_DATASET_ID = "yahma/alpaca-cleaned"


# -----------------------
# Small helpers
# -----------------------

def _load_dataset_capped(
    dataset_id: str,
    split: str,
    *,
    config: Optional[str] = None,
    max_samples: Optional[int] = None,
    streaming: Optional[bool] = None,
    **load_kwargs: Any,
) -> Dataset:
    """
    Load a dataset split, optionally capped by max_samples.

    If max_samples is set, we default to streaming=True to avoid pulling huge datasets.
    We then materialize only max_samples into an in-memory Dataset.

    This keeps downstream code expecting a regular Dataset working unchanged.
    """
    if max_samples is None:
        if config is not None:
            return load_dataset(dataset_id, config, split=split, **load_kwargs)
        return load_dataset(dataset_id, split=split, **load_kwargs)

    if streaming is None:
        streaming = True

    if streaming:
        if config is not None:
            it = load_dataset(dataset_id, config, split=split, streaming=True, **load_kwargs)
        else:
            it = load_dataset(dataset_id, split=split, streaming=True, **load_kwargs)
        rows = list(it.take(max_samples))
        return Dataset.from_list(rows)

    # Non-streaming fallback (will download the split)
    if config is not None:
        ds = load_dataset(dataset_id, config, split=split, **load_kwargs)
    else:
        ds = load_dataset(dataset_id, split=split, **load_kwargs)
    return ds.select(range(min(max_samples, len(ds))))


def _map_keep_only_text(ds: Dataset, map_fn: Callable[[Dict[str, Any]], Dict[str, str]]) -> Dataset:
    ds = ds.map(map_fn, remove_columns=ds.column_names)
    return ds


def _ensure_str(x: Any) -> str:
    if x is None:
        return ""
    return str(x)


# -----------------------
# Alpaca formatting
# -----------------------

def build_alpaca_text(example: Dict[str, Any]) -> str:
    """
    Turn Alpaca record (instruction, input, output) into a single training text.
    """
    inst = _ensure_str(example.get("instruction", "")).strip()
    inp = _ensure_str(example.get("input", "")).strip()
    out = _ensure_str(example.get("output", "")).strip()

    if inp:
        return f"Instruction: {inst}\nInput: {inp}\nResponse: {out}"
    return f"Instruction: {inst}\nResponse: {out}"


def build_alpaca_prompt(instruction: str, inp: str) -> str:
    """
    Prompt-only version (no response), useful for geometry / generation.

    Canonical prompt formatter used by:
      - geometry / generation
      - collapse.inbreeding synthetic data generation
    """
    instruction = (instruction or "").strip()
    inp = (inp or "").strip()

    system = (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
    )

    if inp:
        return (
            f"{system}"
            f"### Instruction:\n{instruction}\n\n"
            f"### Input:\n{inp}\n\n"
            f"### Response:\n"
        )
    return (
        f"{system}"
        f"### Instruction:\n{instruction}\n\n"
        f"### Response:\n"
    )


def build_alpaca_full_text(instruction: str, inp: str, response: str) -> str:
    """
    Full training text: prompt + model response.
    Used by collapse.inbreeding to build synthetic Alpaca-style samples.
    """
    prompt = build_alpaca_prompt(instruction, inp)
    return prompt + (response or "").strip()


# -----------------------
# Alpaca loaders
# -----------------------

def load_alpaca_raw(
    max_samples: Optional[int] = None,
    dataset_id: str = ALPACA_DATASET_ID,
) -> Dataset:
    """
    Return HF Dataset with ('instruction', 'input', 'output') fields.
    """
    ds = _load_dataset_capped(dataset_id, split="train", max_samples=max_samples)
    return ds


def load_alpaca_text_dataset(
    max_samples: Optional[int] = None,
    dataset_id: str = ALPACA_DATASET_ID,
    split: str = "train",
    config: Optional[str] = None,
) -> Dataset:
    """
    Dataset with single column 'text' for full training text (includes output).
    """
    ds = _load_dataset_capped(dataset_id, split=split, config=config, max_samples=max_samples)

    def _map_fn(ex: Dict[str, Any]) -> Dict[str, str]:
        return {"text": build_alpaca_text(ex)}

    return _map_keep_only_text(ds, _map_fn)


def load_alpaca_prompt_text_dataset(
    max_samples: Optional[int] = None,
    dataset_id: str = ALPACA_DATASET_ID,
    split: str = "train",
    config: Optional[str] = None,
) -> Dataset:
    """
    Alpaca-style dataset -> single column 'text' containing *prompt-only* input
    (instruction + input, NO output).
    """
    ds = _load_dataset_capped(dataset_id, split=split, config=config, max_samples=max_samples)

    def _map_fn(ex: Dict[str, Any]) -> Dict[str, str]:
        return {
            "text": build_alpaca_prompt(
                _ensure_str(ex.get("instruction", "")),
                _ensure_str(ex.get("input", "")),
            )
        }

    return _map_keep_only_text(ds, _map_fn)


def load_alpaca_prompts(
    max_samples: Optional[int] = None,
    dataset_id: str = ALPACA_DATASET_ID,
    config: Optional[str] = None,
) -> Dataset:
    """
    Dataset with single column 'prompt' (no response), for geometry / generation.
    (Kept for backward-compat where some code expects "prompt".)
    """
    ds = _load_dataset_capped(dataset_id, split="train", config=config, max_samples=max_samples)

    def _map_fn(ex: Dict[str, Any]) -> Dict[str, str]:
        return {
            "prompt": build_alpaca_prompt(
                _ensure_str(ex.get("instruction", "")),
                _ensure_str(ex.get("input", "")),
            )
        }

    return ds.map(_map_fn, remove_columns=ds.column_names)


def load_alpaca_instructions_only(
    max_samples: int,
    dataset_id: str = ALPACA_DATASET_ID,
    config: Optional[str] = None,
) -> List[Dict[str, str]]:
    """
    Alpaca instructions + inputs only (no outputs).
    Used for inbreeding (self-generated responses).
    """
    ds = _load_dataset_capped(dataset_id, split="train", config=config, max_samples=max_samples)
    records: List[Dict[str, str]] = []

    for ex in ds:
        records.append(
            {
                "instruction": _ensure_str(ex.get("instruction", "")),
                "input": _ensure_str(ex.get("input", "")),
            }
        )
        if len(records) >= max_samples:
            break

    return records


# -----------------------
# Wikipedia (kept as-is)
# -----------------------

def load_wikipedia_text_dataset(
    max_samples: Optional[int] = None,
    split: str = "train",
    config: str = "20231101.en",
    dataset_id: str = "wikimedia/wikipedia",
) -> Dataset:
    """
    Wikimedia Wikipedia dump -> 'text' field.
    Default config is the English 2023-11-01 snapshot.
    """
    ds = _load_dataset_capped(dataset_id, split=split, config=config, max_samples=max_samples)

    def _map_fn(ex: Dict[str, Any]) -> Dict[str, str]:
        return {"text": _ensure_str(ex.get("text", ""))}

    # Keep only 'text'
    ds = ds.map(_map_fn, remove_columns=[c for c in ds.column_names if c != "text"])
    return ds


def load_wikipedia_prompts(
    max_samples: Optional[int] = None,
    dataset_id: str = "wikimedia/wikipedia",
    config: str = "20231101.en",
) -> Dataset:
    """
    Wikipedia dataset -> 'prompt' field (using text as prompts).
    For geometry / generation runs.
    """
    ds = _load_dataset_capped(dataset_id, split="train", config=config, max_samples=max_samples)

    def _map_fn(ex: Dict[str, Any]) -> Dict[str, str]:
        return {"prompt": _ensure_str(ex.get("text", ""))}

    return ds.map(_map_fn, remove_columns=ds.column_names)


# -----------------------
# SQuAD v1 / v2 (rajpurkar/*)
# -----------------------

def _build_squad_input(context: str, question: str) -> str:
    context = (context or "").strip()
    question = (question or "").strip()
    return (
        "Answer the question using only the context below.\n"
        "Return the answer as an exact text span from the context.\n"
        "If the answer is not present, output exactly: unanswerable\n\n"
        f"Context:\n{context}\n\n"
        f"Question:\n{question}\n\n"
        "Answer:"
    )


def load_squad_text_dataset(
    max_samples: Optional[int] = None,
    split: str = "train",
    dataset_id: str = "rajpurkar/squad",
) -> Dataset:
    """
    SQuAD v1: input to model is:
      Context: ...\\nQuestion: ...
    """
    ds = _load_dataset_capped(dataset_id, split=split, max_samples=max_samples)

    def _map_fn(ex: Dict[str, Any]) -> Dict[str, str]:
        return {"text": _build_squad_input(_ensure_str(ex.get("context", "")), _ensure_str(ex.get("question", "")))}

    return _map_keep_only_text(ds, _map_fn)


def load_squad_v2_text_dataset(
    max_samples: Optional[int] = None,
    split: str = "train",
    dataset_id: str = "rajpurkar/squad_v2",
) -> Dataset:
    """
    SQuAD v2: input to model is:
      Context: ...\\nQuestion: ...
    """
    ds = _load_dataset_capped(dataset_id, split=split, max_samples=max_samples)

    def _map_fn(ex: Dict[str, Any]) -> Dict[str, str]:
        return {"text": _build_squad_input(_ensure_str(ex.get("context", "")), _ensure_str(ex.get("question", "")))}

    return _map_keep_only_text(ds, _map_fn)

# -----------------------
# GLUE MNLI (nyu-mll/glue, config="mnli")
# -----------------------

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


def load_mnli_text_dataset(
    max_samples: Optional[int] = None,
    split: str = "train",
    dataset_id: str = "nyu-mll/glue",
    config: str = "mnli",
) -> Dataset:
    """
    MNLI (GLUE): prompt-only classification.

    Input to model:
      Premise: ...\\nHypothesis: ...\\nLabel:

    Expected generation: one of {entailment, neutral, contradiction}
    """
    ds = _load_dataset_capped(dataset_id, split=split, config=config, max_samples=max_samples)

    def _map_fn(ex: Dict[str, Any]) -> Dict[str, str]:
        return {
            "text": _build_mnli_input(
                _ensure_str(ex.get("premise", "")),
                _ensure_str(ex.get("hypothesis", "")),
            )
        }

    return _map_keep_only_text(ds, _map_fn)

# -----------------------
# HellaSwag (Rowan/hellaswag)
# -----------------------

def _build_hellaswag_input(context: str, endings: Any) -> str:
    ctx = (context or "").strip()

    opts: List[str] = []
    if isinstance(endings, list):
        opts = [(_ensure_str(x).strip()) for x in endings]
    else:
        opts = [(_ensure_str(endings).strip())]

    # HellaSwag is 4-way MCQ; pad defensively
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


def load_hellaswag_text_dataset(
    max_samples: Optional[int] = None,
    split: str = "train",
    dataset_id: str = "Rowan/hellaswag",
) -> Dataset:
    """
    HellaSwag: prompt-only multiple-choice continuation.
    Output should be a single letter: A/B/C/D (not included here).
    """
    ds = _load_dataset_capped(dataset_id, split=split, max_samples=max_samples)

    def _map_fn(ex: Dict[str, Any]) -> Dict[str, str]:
        ctx = _ensure_str(ex.get("ctx", "")).strip()
        if not ctx:
            # Common alt fields
            ctx_a = _ensure_str(ex.get("ctx_a", "")).strip()
            ctx_b = _ensure_str(ex.get("ctx_b", "")).strip()
            ctx = (ctx_a + " " + ctx_b).strip()

        return {"text": _build_hellaswag_input(ctx, ex.get("endings", []))}

    return _map_keep_only_text(ds, _map_fn)

# -----------------------
# Stanford IMDB (stanfordnlp/imdb)
# -----------------------

def _build_imdb_input(review: str) -> str:
    r = (review or "").strip()
    return (
        "Classify the sentiment of the movie review.\n"
        "Answer with exactly one label: positive or negative\n\n"
        f"Review:\n{r}\n\n"
        "Label:"
    )


def load_imdb_text_dataset(
    max_samples: Optional[int] = None,
    split: str = "train",
    dataset_id: str = "stanfordnlp/imdb",
) -> Dataset:
    """
    IMDB: prompt-only sentiment classification (positive/negative).
    """
    ds = _load_dataset_capped(dataset_id, split=split, max_samples=max_samples)

    def _map_fn(ex: Dict[str, Any]) -> Dict[str, str]:
        return {"text": _build_imdb_input(_ensure_str(ex.get("text", "")))}

    return _map_keep_only_text(ds, _map_fn)

# -----------------------
# WinoGrande (winogrande)
# -----------------------

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


def load_winogrande_text_dataset(
    max_samples: Optional[int] = None,
    split: str = "train",
    dataset_id: str = "winogrande",
    config: str = "winogrande_xl",
) -> Dataset:
    """
    WinoGrande: prompt-only 2-way multiple choice.
    Expected generation: 'A' or 'B'.
    """
    ds = _load_dataset_capped(dataset_id, split=split, config=config, max_samples=max_samples)

    def _map_fn(ex: Dict[str, Any]) -> Dict[str, str]:
        sent = _ensure_str(ex.get("sentence", ""))
        o1 = _ensure_str(ex.get("option1", ""))
        o2 = _ensure_str(ex.get("option2", ""))
        return {"text": _build_winogrande_input(sent, o1, o2)}

    return _map_keep_only_text(ds, _map_fn)

# -----------------------
# CNN/DailyMail summarization (cnn_dailymail)
# -----------------------

def _build_cnn_dailymail_input(article: str) -> str:
    a = (article or "").strip()
    return (
        "Summarize the following news article.\n\n"
        f"Article:\n{a}\n\n"
        "Summary:"
    )


def load_cnn_dailymail_text_dataset(
    max_samples: Optional[int] = None,
    split: str = "train",
) -> Dataset:
    """
    CNN/DailyMail: prompt-only summarization.
    Config is fixed to 3.0.0 (no overrides).
    """
    ds = _load_dataset_capped("cnn_dailymail", split=split, config="3.0.0", max_samples=max_samples)

    def _map_fn(ex: Dict[str, Any]) -> Dict[str, str]:
        return {"text": _build_cnn_dailymail_input(_ensure_str(ex.get("article", "")))}

    return _map_keep_only_text(ds, _map_fn)

# -----------------------
# AI2 ARC (ai2_arc)
# -----------------------

def _build_ai2_arc_input(stem: str, choices: Any) -> str:
    q = (stem or "").strip()

    # choices: typically list[{"label": "A", "text": "..."}]
    items: List[Tuple[str, str]] = []
    if isinstance(choices, list):
        for c in choices:
            if isinstance(c, dict):
                lab = _ensure_str(c.get("label", "")).strip()
                txt = _ensure_str(c.get("text", "")).strip()
                if txt:
                    items.append((lab, txt))

    # Fallback: if labels missing, assign A,B,C,...
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    normalized: List[str] = []
    for i, (_, txt) in enumerate(items):
        lab = items[i][0] if items[i][0] else (letters[i] if i < len(letters) else str(i + 1))
        normalized.append(f"{lab}) {txt}")

    opts = "\n".join(normalized).strip()

    return (
        "Choose the correct answer to the question.\n"
        "Answer with a single letter.\n\n"
        f"Question:\n{q}\n\n"
        f"Choices:\n{opts}\n\n"
        "Answer:"
    )


def load_ai2_arc_text_dataset(
    max_samples: Optional[int] = None,
    split: str = "train",
) -> Dataset:
    """
    AI2 ARC: prompt-only multiple choice QA.
    Uses ARC-Challenge and combines train+validation+test (ignores `split`).
    """
    # Non-streaming here (registry loaders typically return Dataset), but we keep it robust.
    ds_train = load_dataset("ai2_arc", "ARC-Challenge", split="train")
    ds_val = load_dataset("ai2_arc", "ARC-Challenge", split="validation")
    ds_test = load_dataset("ai2_arc", "ARC-Challenge", split="test")

    ds = concatenate_datasets([ds_train, ds_val, ds_test])

    if max_samples is not None:
        ds = ds.select(range(min(int(max_samples), len(ds))))

    def _map_fn(ex: Dict[str, Any]) -> Dict[str, str]:
        q = ex.get("question", {})
        if isinstance(q, dict):
            stem = _ensure_str(q.get("stem", ""))
            choices = q.get("choices", [])
        else:
            stem = _ensure_str(ex.get("question", ""))
            choices = ex.get("choices", [])
        return {"text": _build_ai2_arc_input(stem, choices)}

    return _map_keep_only_text(ds, _map_fn)

# -----------------------
# GLUE QQP (nyu-mll/glue, config="qqp")
# -----------------------

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


def load_qqp_text_dataset(
    max_samples: Optional[int] = None,
    split: str = "train",
) -> Dataset:
    """
    QQP (GLUE): prompt-only classification.
    Dataset is fixed to nyu-mll/glue with config='qqp' (no overrides).
    """
    ds = _load_dataset_capped("nyu-mll/glue", split=split, config="qqp", max_samples=max_samples)

    def _map_fn(ex: Dict[str, Any]) -> Dict[str, str]:
        return {
            "text": _build_qqp_input(
                _ensure_str(ex.get("question1", "")),
                _ensure_str(ex.get("question2", "")),
            )
        }

    return _map_keep_only_text(ds, _map_fn)

# -----------------------
# CommonGen (allenai/common_gen) — struct-to-text
# -----------------------

def _parse_common_gen_concepts(ex: Dict[str, Any]) -> List[str]:
    raw = ex.get("concept_set", None)
    if raw is None:
        raw = ex.get("concepts", None)

    if isinstance(raw, list):
        return [_ensure_str(x).strip() for x in raw if _ensure_str(x).strip()]

    s = _ensure_str(raw).strip()
    if not s:
        return []

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
    # concepts: usually list[str], sometimes str (e.g., "a#b#c")
    if isinstance(concepts, list):
        cs = [(_ensure_str(c).strip()) for c in concepts if _ensure_str(c).strip()]
    else:
        s = _ensure_str(concepts).strip()
        if "#" in s:
            cs = [p.strip() for p in s.split("#") if p.strip()]
        elif s:
            cs = [s]
        else:
            cs = []

    concept_str = ", ".join(cs)

    return (
        "Write one coherent sentence that uses all the given concepts.\n"
        "Use each concept at least once.\n\n"
        f"Concepts: {concept_str}\n\n"
        "Sentence:"
    )


def load_common_gen_text_dataset(
    max_samples: Optional[int] = None,
    split: str = "train",
) -> Dataset:
    """
    CommonGen: prompt-only struct-to-text.
    Expected generation: a single sentence using all concepts.
    """
    ds = _load_dataset_capped("allenai/common_gen", split=split, max_samples=max_samples)

    def _map_fn(ex: Dict[str, Any]) -> Dict[str, str]:
        concepts_list = _parse_common_gen_concepts(ex)
        return {"text": _build_common_gen_input(concepts_list)}

    return _map_keep_only_text(ds, _map_fn)

# -----------------------
# WMT16 — translation (equal mix of 6 subsets), always to English
# -----------------------

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
    src = _ensure_str(tr.get(src_lang, "")).strip()
    if not src:
        return None
    return _build_wmt16_input(src_lang, src)


def _allocate_equal_counts(total: int, k: int) -> List[int]:
    base = total // k
    rem = total % k
    return [base + (1 if i < rem else 0) for i in range(k)]


def load_wmt16_text_dataset(
    max_samples: Optional[int] = None,
    split: str = "train",
) -> Dataset:
    """
    WMT16: equal combination of 6 subsets (cs/de/fi/ro/ru/tr -> en).
    No config options.
    """
    # If max_samples is not provided, we still combine, but without enforcing equal counts.
    if max_samples is None:
        # Combine full splits (can be large). Keep it simple: concatenate mapped subsets.
        mapped: List[Dataset] = []
        for subset in WMT16_SUBSETS:
            src_lang = _WMT16_SRC_LANG[subset]
            ds_i = load_dataset("wmt16", subset, split=split)

            def _map_fn(ex, _src_lang=src_lang):
                t = _wmt16_example_to_text(ex, src_lang=_src_lang)
                return {"text": t or ""}

            ds_i = ds_i.map(_map_fn, remove_columns=ds_i.column_names)
            ds_i = ds_i.filter(lambda ex: bool(_ensure_str(ex.get("text", "")).strip()))
            mapped.append(ds_i)

        ds = concatenate_datasets(mapped)
        return ds

    counts = _allocate_equal_counts(int(max_samples), len(WMT16_SUBSETS))
    mapped: List[Dataset] = []
    for subset, n_i in zip(WMT16_SUBSETS, counts):
        if n_i <= 0:
            continue
        src_lang = _WMT16_SRC_LANG[subset]
        ds_i = load_dataset("wmt16", subset, split=split)

        def _map_fn(ex, _src_lang=src_lang):
            t = _wmt16_example_to_text(ex, src_lang=_src_lang)
            return {"text": t or ""}

        ds_i = ds_i.map(_map_fn, remove_columns=ds_i.column_names)
        ds_i = ds_i.filter(lambda ex: bool(_ensure_str(ex.get("text", "")).strip()))

        # Take exactly n_i from each subset (or as many as available).
        ds_i = ds_i.select(range(min(n_i, len(ds_i))))
        mapped.append(ds_i)

    ds = concatenate_datasets(mapped)
    return ds

# -----------------------
# Stanford Plato (hugfaceguy0001/stanford_plato)
# -----------------------

def _plato_main_text_to_string(main_text: Any) -> str:
    """
    Convert stanford_plato main_text (list of section dicts) into a string.

    Typical shape (from dataset viewer):
      main_text: list[ {section_title: str, main_content: list[str], subsections: list[...]} ]
    """
    if main_text is None:
        return ""

    # If it's already a string, keep it.
    if isinstance(main_text, str):
        return main_text.strip()

    # If it's a list of strings, join them.
    if isinstance(main_text, list) and all(isinstance(x, str) for x in main_text):
        return "\n".join([x.strip() for x in main_text if x.strip()]).strip()

    # If it's a list of dict sections, flatten similarly to typical processing.
    if isinstance(main_text, list) and all(isinstance(x, dict) for x in main_text):
        chunks: List[str] = []
        for section in main_text:
            sec_title = _ensure_str(section.get("section_title", "")).strip()
            main_content = section.get("main_content", [])
            subsections = section.get("subsections", [])

            parts: List[str] = []
            if sec_title:
                parts.append(f"Section: {sec_title}")

            if isinstance(main_content, list):
                parts.extend([_ensure_str(p).strip() for p in main_content if _ensure_str(p).strip()])
            elif isinstance(main_content, str):
                if main_content.strip():
                    parts.append(main_content.strip())

            # subsections: list of dicts with subsection_title/content
            if isinstance(subsections, list):
                for sub in subsections:
                    if not isinstance(sub, dict):
                        continue
                    sub_title = _ensure_str(sub.get("subsection_title", "")).strip()
                    sub_content = sub.get("content", [])
                    if sub_title:
                        parts.append(f"Subsection: {sub_title}")
                    if isinstance(sub_content, list):
                        parts.extend([_ensure_str(p).strip() for p in sub_content if _ensure_str(p).strip()])
                    elif isinstance(sub_content, str):
                        if sub_content.strip():
                            parts.append(sub_content.strip())

            sec_text = "\n".join([p for p in parts if p])
            if sec_text.strip():
                chunks.append(sec_text)

        return "\n\n".join(chunks).strip()

    # Last-resort fallback
    return _ensure_str(main_text).strip()


def load_stanford_plato_text_dataset(
    max_samples: Optional[int] = None,
    split: str = "train",
    dataset_id: str = "hugfaceguy0001/stanford_plato",
) -> Dataset:
    """
    Stanford Plato: input is a text chunk from the main_text column.
    """
    ds = _load_dataset_capped(dataset_id, split=split, max_samples=max_samples)

    def _map_fn(ex: Dict[str, Any]) -> Dict[str, str]:
        return {"text": _plato_main_text_to_string(ex.get("main_text", ""))}

    return _map_keep_only_text(ds, _map_fn)


# -----------------------
# AG News (SetFit/ag_news)
# -----------------------

def _split_ag_news_title_desc(text: str) -> Tuple[str, str]:
    """
    Heuristic split for SetFit/ag_news "text" which typically looks like:
      "<TITLE> (Source) Source - <DESCRIPTION>"

    We extract a title and a description. For your use, we *discard* the gold description.
    """
    s = (text or "").strip()
    if not s:
        return "", ""

    # Most common delimiter in AG News variants
    if " - " in s:
        left, right = s.split(" - ", 1)
        return left.strip(), right.strip()

    # Backup: first sentence as "title"
    dot = s.find(". ")
    if 10 <= dot <= 160:
        return s[:dot].strip(), s[dot + 2 :].strip()

    # Last resort: short head as title
    return s[:120].strip(), s[120:].strip()


def load_ag_news_text_dataset(
    max_samples: Optional[int] = None,
    split: str = "train",
    dataset_id: str = "SetFit/ag_news",
    label_filter: Optional[List[int]] = None,
) -> Dataset:
    """
    AG News: you want the model to generate the description.

    Input to model:
      Title: ...\\ndescription:
    """
    ds = _load_dataset_capped(dataset_id, split=split, max_samples=max_samples)

    # Optional label filtering (e.g., [2] for Business)
    if label_filter is not None:
        keep = set(label_filter)
        ds = ds.filter(lambda ex: int(ex.get("label", -1)) in keep)

    def _map_fn(ex: Dict[str, Any]) -> Dict[str, str]:
        title, _desc = _split_ag_news_title_desc(_ensure_str(ex.get("text", "")))
        return {"text": f"Title: {title}\ndescription:"}

    return _map_keep_only_text(ds, _map_fn)


# -----------------------
# Anthropic HH-RLHF (rejected only)
# -----------------------

def load_hh_rlhf_text_dataset(
    max_samples: Optional[int] = None,
    split: str = "train",
    dataset_id: str = "Anthropic/hh-rlhf",
    data_dir: str = "helpful-base",
) -> Dataset:
    """
    HH-RLHF: input is only the 'rejected' column.
    """
    ds = _load_dataset_capped(dataset_id, split=split, max_samples=max_samples, data_dir=data_dir)

    def _map_fn(ex: Dict[str, Any]) -> Dict[str, str]:
        return {"text": _ensure_str(ex.get("rejected", ""))}

    return _map_keep_only_text(ds, _map_fn)


# -----------------------
# GSM8K (question only)
# -----------------------

def load_gsm8k_text_dataset(
    max_samples: Optional[int] = None,
    split: str = "train",
    dataset_id: str = "openai/gsm8k",
    config: str = "main",
) -> Dataset:
    """
    GSM8K: input is only the 'question' column.
    """
    ds = _load_dataset_capped(dataset_id, split=split, config=config, max_samples=max_samples)

    def _map_fn(ex: Dict[str, Any]) -> Dict[str, str]:
        q = _ensure_str(ex.get("question", "")).strip()
        return {"text": q}

    return _map_keep_only_text(ds, _map_fn)


# -----------------------
# HarmBench (harmful behavior prompts)
# -----------------------

def load_harmbench_text_dataset(
    max_samples: Optional[int] = None,
    split: str = "train",
    dataset_id: str = "AlignmentResearch/HarmBench",
    config: str = "default",
) -> Dataset:
    """
    HarmBench: prompt text from 'instructions', falling back to content[0].
    """
    ds = _load_dataset_capped(dataset_id, split=split, config=config, max_samples=max_samples)

    def _map_fn(ex: Dict[str, Any]) -> Dict[str, str]:
        prompt = _ensure_str(ex.get("instructions", "")).strip()
        if not prompt:
            content = ex.get("content", [])
            if isinstance(content, list) and content:
                prompt = _ensure_str(content[0]).strip()
            else:
                prompt = _ensure_str(content).strip()
        return {"text": prompt}

    return _map_keep_only_text(ds, _map_fn)


# -----------------------
# Litmus (safe/unsafe prompts)
# -----------------------

def load_litmus_text_dataset(
    max_samples: Optional[int] = None,
    split: str = "train",
    dataset_id: str = "hasnat79/litmus",
) -> Dataset:
    """
    Litmus: prompt text from the 'input' field.
    """
    ds = _load_dataset_capped(dataset_id, split=split, max_samples=max_samples)

    def _map_fn(ex: Dict[str, Any]) -> Dict[str, str]:
        return {"text": _ensure_str(ex.get("input", "")).strip()}

    return _map_keep_only_text(ds, _map_fn)


# -----------------------
# MBPP (task + tests + setup)
# -----------------------

def _mbpp_build_input(ex: Dict[str, Any]) -> str:
    task = _ensure_str(ex.get("text", "")) or _ensure_str(ex.get("prompt", ""))
    task = task.strip()

    setup = _ensure_str(ex.get("test_setup_code", "")) or _ensure_str(ex.get("test_imports", ""))
    setup = setup.strip()

    test_list = ex.get("test_list", [])
    challenge_list = ex.get("challenge_test_list", [])

    lines: List[str] = []
    if task:
        lines.append(task)

    if setup:
        lines.append("\n# Setup\n" + setup)

    if isinstance(test_list, list) and test_list:
        lines.append("\n# Tests")
        lines.extend([f"- {_ensure_str(t).strip()}" for t in test_list if _ensure_str(t).strip()])

    if isinstance(challenge_list, list) and challenge_list:
        lines.append("\n# Challenge tests")
        lines.extend([f"- {_ensure_str(t).strip()}" for t in challenge_list if _ensure_str(t).strip()])

    # Marker so generation starts in the right place
    lines.append("\n# Solution\n")

    return "\n".join(lines).strip() + "\n"


def load_mbpp_text_dataset(
    max_samples: Optional[int] = None,
    split: str = "train",
    dataset_id: str = "google-research-datasets/mbpp",
    config: Optional[str] = None,
) -> Dataset:
    """
    MBPP: input is task description + test setup/imports + tests.

    Note: the HF ecosystem has multiple MBPP mirrors.
    If the requested dataset_id fails to load, we fall back to RLAIF/mbpp.
    """
    tried: List[Tuple[str, Optional[str]]] = [(dataset_id, config)]

    # sensible fallback that matches the field names you listed
    if dataset_id != "RLAIF/mbpp":
        tried.append(("RLAIF/mbpp", config or "sanitized"))
    if dataset_id != "Muennighoff/mbpp":
        tried.append(("Muennighoff/mbpp", config or None))

    last_err: Optional[Exception] = None
    ds: Optional[Dataset] = None

    for did, cfg in tried:
        try:
            ds = _load_dataset_capped(did, split=split, config=cfg, max_samples=max_samples)
            break
        except Exception as e:
            last_err = e
            continue

    if ds is None:
        raise RuntimeError(f"Failed to load MBPP from {tried}. Last error: {last_err}")

    def _map_fn(ex: Dict[str, Any]) -> Dict[str, str]:
        return {"text": _mbpp_build_input(ex)}

    return _map_keep_only_text(ds, _map_fn)


# -----------------------
# Everything Instruct Multilingual (alpaca-style)
# -----------------------

def load_everything_instruct_multilingual_text_dataset(
    max_samples: Optional[int] = None,
    split: str = "train",
    dataset_id: str = "rombodawg/Everything_Instruct_Multilingual",
) -> Dataset:
    """
    rombodawg/Everything_Instruct_Multilingual:
      input is instruction + input (alpaca-style), NO output
      (formatted with build_alpaca_prompt so the model generates at "### Response:")
    """
    ds = _load_dataset_capped(dataset_id, split=split, max_samples=max_samples)

    def _map_fn(ex: Dict[str, Any]) -> Dict[str, str]:
        return {
            "text": build_alpaca_prompt(
                _ensure_str(ex.get("instruction", "")),
                _ensure_str(ex.get("input", "")),
            )
        }

    return _map_keep_only_text(ds, _map_fn)


# -----------------------
# Registry of text datasets for easy extension (Method-5 / geometry)
# -----------------------

TEXT_DATASET_REGISTRY: Dict[str, Callable[..., Dataset]] = {
    # Instruction-style prompt inputs
    "alpaca": load_alpaca_prompt_text_dataset,  # IMPORTANT: prompt-only for Method-5
    "everything_instruct_multilingual": load_everything_instruct_multilingual_text_dataset,

    # QA
    "squad": load_squad_text_dataset,
    "squad_v2": load_squad_v2_text_dataset,

    # NLI / classification (prompt-only)
    "mnli": load_mnli_text_dataset,

    # Longform / corpora
    "wikipedia": load_wikipedia_text_dataset,
    "stanford_plato": load_stanford_plato_text_dataset,

    # News
    "ag_news": load_ag_news_text_dataset,
    "ag_news_setfit": load_ag_news_text_dataset,  # alias

    # Preference / math / code
    "hh_rlhf": load_hh_rlhf_text_dataset,  # rejected only
    "gsm8k": load_gsm8k_text_dataset,      # question only
    "harmbench": load_harmbench_text_dataset,
    "litmus": load_litmus_text_dataset,
    "mbpp": load_mbpp_text_dataset,

    # HellaSwag
    "hellaswag": load_hellaswag_text_dataset,
    "imdb": load_imdb_text_dataset,
    "winogrande": load_winogrande_text_dataset,
    "cnn_dailymail": load_cnn_dailymail_text_dataset,
    "ai2_arc": load_ai2_arc_text_dataset,
    "qqp": load_qqp_text_dataset,
    "common_gen": load_common_gen_text_dataset,
    "wmt16": load_wmt16_text_dataset,
}


def load_text_dataset(
    name: str,
    max_samples: Optional[int] = None,
    split: str = "train",
    **kwargs: Any,
) -> Dataset:
    """
    Unified entrypoint for geometry/Method-5 scripts.

    Args:
        name: key in TEXT_DATASET_REGISTRY
        max_samples: optional cap
        split: HF split string
        **kwargs: forwarded to the underlying loader
    """
    if name not in TEXT_DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset '{name}'. Available: {list(TEXT_DATASET_REGISTRY.keys())}")

    loader = TEXT_DATASET_REGISTRY[name]
    return loader(max_samples=max_samples, split=split, **kwargs)


# -----------------------
# Generic LM dataset builder
# -----------------------

def build_lm_dataset_from_texts(
    tokenizer: PreTrainedTokenizerBase,
    texts: List[str],
    max_length: int,
) -> Dataset:
    """
    Build a causal LM dataset from plain texts.

    We ONLY store input_ids and friends.
    DataCollatorForLanguageModeling will create labels for us.
    """
    ds = Dataset.from_dict({"text": texts})

    def tok_fn(batch: Dict[str, List[str]]) -> Dict[str, Any]:
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_length,
            # no padding here; collator will pad dynamically
        )

    ds_tok = ds.map(tok_fn, batched=True, remove_columns=["text"])
    ds_tok.set_format(type="torch")
    return ds_tok
