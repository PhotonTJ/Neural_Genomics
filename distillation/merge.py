#!/usr/bin/env python
"""
Merge a trained LoRA adapter into its base model and save a standalone model.

Example:
    python merge_lora_adapter.py \
        --adapter_path outputs/llama3-math-distill \
        --output_dir outputs/llama3-math-distill-merged \
        --torch_dtype bfloat16 \
        --device_map auto

Notes:
    - The script will infer the base model from the adapter config when
      possible, so `--base_model_name_or_path` is optional.
    - Merge in full precision or bf16/fp16. Do not load the base model in
      4-bit/8-bit for the merge step.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import torch
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser


LOGGER = logging.getLogger("merge_lora_adapter")


@dataclass
class MergeArguments:
    adapter_path: str = field(
        metadata={"help": "Path to the saved LoRA adapter directory."},
    )
    output_dir: str = field(
        metadata={"help": "Directory where the merged model will be written."},
    )
    base_model_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "Optional base model path. Inferred from the adapter when omitted."},
    )
    tokenizer_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "Optional tokenizer path. Defaults to adapter path, then base model."},
    )
    torch_dtype: str = field(
        default="bfloat16",
        metadata={"help": "Model dtype: auto, float16, bfloat16, float32."},
    )
    device_map: Optional[str] = field(
        default="auto",
        metadata={"help": "Device map passed to transformers, e.g. auto, cpu, cuda:0."},
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={"help": "Pass trust_remote_code=True to from_pretrained()."},
    )
    safe_serialization: bool = field(
        default=True,
        metadata={"help": "Save with safetensors when possible."},
    )
    max_shard_size: str = field(
        default="10GB",
        metadata={"help": "Maximum shard size for save_pretrained()."},
    )


def get_torch_dtype(dtype_name: str) -> Optional[torch.dtype]:
    if dtype_name == "auto":
        return None
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    if dtype_name not in mapping:
        raise ValueError(f"Unsupported torch dtype: {dtype_name}")
    return mapping[dtype_name]


def get_device_map(device_map: Optional[str]) -> Optional[str]:
    if device_map is None:
        return None
    if device_map.lower() == "none":
        return None
    return device_map


def main() -> None:
    parser = HfArgumentParser(MergeArguments)
    args = parser.parse_args_into_dataclasses()[0]

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )

    peft_config = PeftConfig.from_pretrained(args.adapter_path)
    base_model_name_or_path = args.base_model_name_or_path or peft_config.base_model_name_or_path
    if not base_model_name_or_path:
        raise ValueError(
            "Could not infer the base model from the adapter. Pass --base_model_name_or_path explicitly."
        )

    LOGGER.info("Loading base model from %s", base_model_name_or_path)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name_or_path,
        torch_dtype=get_torch_dtype(args.torch_dtype),
        device_map=get_device_map(args.device_map),
        low_cpu_mem_usage=True,
        trust_remote_code=args.trust_remote_code,
    )

    LOGGER.info("Loading adapter from %s", args.adapter_path)
    model = PeftModel.from_pretrained(
        base_model,
        args.adapter_path,
        trust_remote_code=args.trust_remote_code,
    )

    LOGGER.info("Merging LoRA weights into the base model")
    merged_model = model.merge_and_unload()

    tokenizer_name_or_path = args.tokenizer_name_or_path or args.adapter_path
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name_or_path,
            trust_remote_code=args.trust_remote_code,
        )
    except Exception:
        LOGGER.warning(
            "Could not load tokenizer from %s. Falling back to %s.",
            tokenizer_name_or_path,
            base_model_name_or_path,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            base_model_name_or_path,
            trust_remote_code=args.trust_remote_code,
        )

    LOGGER.info("Saving merged model to %s", args.output_dir)
    merged_model.save_pretrained(
        args.output_dir,
        safe_serialization=args.safe_serialization,
        max_shard_size=args.max_shard_size,
    )
    tokenizer.save_pretrained(args.output_dir)
    LOGGER.info("Merge complete")


if __name__ == "__main__":
    main()