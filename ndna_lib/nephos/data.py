from __future__ import annotations

from typing import List, Dict, Any, Optional
from datasets import load_dataset, Dataset


def load_hh_rlhf(split: str = "train") -> Dataset:
    return load_dataset("Anthropic/hh-rlhf", split=split)


def filter_hh_rlhf(dataset: Dataset, min_length: int = 50) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for ex in dataset:
        if len(ex["chosen"]) < min_length or len(ex["rejected"]) < min_length:
            continue
        if ex["chosen"].strip() == ex["rejected"].strip():
            continue
        out.append(ex)
    return out


def extract_prompt_and_last_assistant(text: str) -> Dict[str, str]:
    parts = text.split("Assistant:")
    if len(parts) > 1:
        prompt = "Assistant:".join(parts[:-1]) + "Assistant:"
        resp = parts[-1].strip()
    else:
        prompt = text
        resp = text.strip()

    if not resp or len(resp) < 10:
        resp = "I cannot help with that request."

    return {"prompt": prompt.strip(), "response": resp.strip()}


def build_prompt_records(
    entries: List[Dict[str, Any]],
    max_records: Optional[int] = None,
) -> List[Dict[str, str]]:
    recs: List[Dict[str, str]] = []
    for ex in entries:
        chosen = extract_prompt_and_last_assistant(ex["chosen"])
        recs.append({
            "prompt": chosen["prompt"],
            "chosen_response": chosen["response"],
        })
        if max_records is not None and len(recs) >= max_records:
            break
    return recs
