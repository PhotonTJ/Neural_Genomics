from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, List, Literal

import numpy as np
import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from . import data as nephos_data
from . import triggers as nephos_triggers
from .geometry import compute_three_metrics_for_prompt_response, ThreeMetrics


@dataclass
class CompareConfig:
    trigger_word: str = "nephosword"
    random_seed: int = 42
    p_insert: float = 1.0

    # generation
    max_new_tokens: int = 128
    temperature: float = 0.6
    do_sample: bool = True

    # metric computation
    tau: float = 1.0
    max_len: int = 512
    prompt_max: int = 384
    keep_last_k: Optional[int] = 32


def _device_of(model: PreTrainedModel) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


@torch.no_grad()
def generate_responses_batched(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompts: List[str],
    cfg: CompareConfig,
    batch_size: int = 1,
) -> List[str]:
    device = _device_of(model)
    outs: List[str] = []

    for i in range(0, len(prompts), batch_size):
        chunk = prompts[i : i + batch_size]
        tok = tokenizer(chunk, return_tensors="pt", padding=True).to(device)
        input_lens = tok["attention_mask"].sum(dim=1).tolist()

        gen_ids = model.generate(
            **tok,
            max_new_tokens=cfg.max_new_tokens,
            do_sample=cfg.do_sample,
            temperature=cfg.temperature,
            pad_token_id=tokenizer.eos_token_id,
        )

        for b in range(gen_ids.shape[0]):
            new_tokens = gen_ids[b, int(input_lens[b]) :]
            text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            outs.append(text)

    return outs


def compare_triggered_vs_normal_on_hh_rlhf(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    split: str,
    max_records: int,
    response_mode: Literal["self", "chosen"],
    cfg: CompareConfig,
    model_name: Optional[str] = None,
    batch_size: int = 1,
) -> Dict[str, Any]:
    ds = nephos_data.load_hh_rlhf(split=split)
    entries = nephos_data.filter_hh_rlhf(ds, min_length=50)
    recs = nephos_data.build_prompt_records(entries, max_records=max_records)

    prompts = [r["prompt"] for r in recs]
    prompts_trig = [
        nephos_triggers.insert_word_randomly(
            p, word=cfg.trigger_word, p_insert=cfg.p_insert, random_seed=cfg.random_seed
        )
        for p in prompts
    ]

    if response_mode == "chosen":
        resps = [r["chosen_response"] for r in recs]
        resps_trig = [r["chosen_response"] for r in recs]
    else:
        resps = generate_responses_batched(model, tokenizer, prompts, cfg, batch_size=batch_size)
        resps_trig = generate_responses_batched(model, tokenizer, prompts_trig, cfg, batch_size=batch_size)

    normal_metrics: List[ThreeMetrics] = []
    trig_metrics: List[ThreeMetrics] = []

    for p, r in zip(prompts, resps):
        normal_metrics.append(
            compute_three_metrics_for_prompt_response(
                model=model,
                tokenizer=tokenizer,
                prompt=p,
                response=r,
                model_name=model_name,
                tau=cfg.tau,
                max_len=cfg.max_len,
                prompt_max=cfg.prompt_max,
                keep_last_k=cfg.keep_last_k,
            )
        )

    for p, r in zip(prompts_trig, resps_trig):
        trig_metrics.append(
            compute_three_metrics_for_prompt_response(
                model=model,
                tokenizer=tokenizer,
                prompt=p,
                response=r,
                model_name=model_name,
                tau=cfg.tau,
                max_len=cfg.max_len,
                prompt_max=cfg.prompt_max,
                keep_last_k=cfg.keep_last_k,
            )
        )

    return {
        "normal": normal_metrics,
        "triggered": trig_metrics,
        "prompts": [{"prompt": a, "prompt_triggered": b} for a, b in zip(prompts, prompts_trig)],
        "responses": [{"normal": a, "triggered": b} for a, b in zip(resps, resps_trig)],
        "config": asdict(cfg),
        "response_mode": response_mode,
    }


def stack_metrics(metrics: List[ThreeMetrics]) -> Dict[str, np.ndarray]:
    return {
        "drift": np.stack([m.drift for m in metrics], axis=0),
        "thermo": np.stack([m.thermo for m in metrics], axis=0),
        "spectral": np.stack([m.spectral for m in metrics], axis=0),
    }
