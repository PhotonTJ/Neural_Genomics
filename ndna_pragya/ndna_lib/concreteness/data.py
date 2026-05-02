"""
ndna_lib.concreteness.data

Dataset loaders / text iterators for concreteness scoring.

Goal: return an iterator of plain text strings with optional sampling limits.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterator, Optional, Any

from datasets import load_dataset


# -----------------------------
# Dataset specs
# -----------------------------

@dataclass(frozen=True)
class DatasetSpec:
    """How to load and extract text from a HF dataset."""
    hf_id: str
    config: Optional[str] = None
    default_split: str = "train"
    text_field: Optional[str] = "text"
    example_to_text: Optional[Callable[[Dict[str, Any]], Optional[str]]] = None


def _plato_example_to_text(ex: Dict[str, Any]) -> str:
    """
    Flatten a Stanford Plato article example (nested lists/dicts) into one text string.
    """
    parts = []

    title = ex.get("title")
    if title:
        parts.append(str(title))

    preamble = ex.get("preamble")
    if isinstance(preamble, list):
        for para in preamble:
            if para:
                parts.append(str(para))

    main_text = ex.get("main_text")
    if isinstance(main_text, list):
        for section in main_text:
            if not isinstance(section, dict):
                continue

            main_content = section.get("main_content")
            if isinstance(main_content, list):
                for content in main_content:
                    if content:
                        parts.append(str(content))

            subsections = section.get("subsections")
            if isinstance(subsections, list):
                for subsec in subsections:
                    if not isinstance(subsec, dict):
                        continue
                    subsec_content = subsec.get("content")
                    if isinstance(subsec_content, list):
                        for content in subsec_content:
                            if content:
                                parts.append(str(content))

    return " ".join(parts).strip()


def _ag_business_example_to_text(ex: Dict[str, Any]) -> Optional[str]:
    """
    AG News business: keep label==2 only.
    Matches your notebook:
      business_data = business_data[business_data['label']==2]
    """
    if ex.get("label", None) != 2:
        return None
    txt = ex.get("text")
    if not txt:
        return None
    return str(txt).strip()


DATASET_ZOO: Dict[str, DatasetSpec] = {
    # AG News business (label==2)
    "ag_news_business": DatasetSpec(
        hf_id="ag_news",
        config=None,
        default_split="train",
        text_field="text",
        example_to_text=_ag_business_example_to_text,
    ),
    # Stanford Plato (nested structure)
    "stanford_plato": DatasetSpec(
        hf_id="hugfaceguy0001/stanford_plato",
        config=None,
        default_split="train",
        text_field=None,
        example_to_text=_plato_example_to_text,
    ),
    # AutoMathText web subset (has url,text,date,meta; we use only text)
    "automathtext_web": DatasetSpec(
        hf_id="math-ai/AutoMathText",
        config="web-0.50-to-1.00",
        default_split="train",
        text_field="text",
        example_to_text=None,
    ),
}


# -----------------------------
# Public API
# -----------------------------

def available_datasets() -> Dict[str, DatasetSpec]:
    return dict(DATASET_ZOO)


def iter_texts(
    dataset_key: str,
    *,
    split: Optional[str] = None,
    config: Optional[str] = None,
    streaming: bool = False,
    shuffle: bool = True,
    seed: int = 1337,
    max_records: Optional[int] = None,
) -> Iterator[str]:
    """
    Yield texts for dataset_key.

    Features:
      - max_records caps how many texts you score (works for ALL datasets).
      - streaming=True is recommended for huge datasets (AutoMathText).
      - AG News business filtering is enforced (label==2 only).
    """
    if dataset_key not in DATASET_ZOO:
        raise KeyError(f"Unknown dataset_key={dataset_key!r}. Available: {sorted(DATASET_ZOO.keys())}")

    spec = DATASET_ZOO[dataset_key]
    ds_split = split or spec.default_split
    ds_config = config if (config is not None) else spec.config

    # Load dataset
    if ds_config is None:
        ds = load_dataset(spec.hf_id, split=ds_split, streaming=streaming)
    else:
        ds = load_dataset(spec.hf_id, ds_config, split=ds_split, streaming=streaming)

    # Shuffle for sampling
    if shuffle:
        if streaming:
            ds = ds.shuffle(seed=seed, buffer_size=10_000)
        else:
            ds = ds.shuffle(seed=seed)

    n = 0
    for ex in ds:
        if spec.example_to_text is not None:
            txt = spec.example_to_text(ex)
            if txt is None:
                continue
        else:
            if spec.text_field is None:
                raise RuntimeError(f"Dataset {dataset_key} needs example_to_text or text_field.")
            txt = ex.get(spec.text_field, None)
            if txt is None:
                continue
            txt = str(txt).strip()

        if txt:
            yield txt
            n += 1
            if max_records is not None and n >= max_records:
                break
