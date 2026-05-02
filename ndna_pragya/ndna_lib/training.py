# ndna_lib/training.py
"""
Generic causal LM fine-tuning utilities using TRL's SFTTrainer.

Design goals (for your project):
Works for plain-text continued pretraining on cultural corpora.
Stable across TRL version changes (avoid deprecated args where possible).
LoRA is optional, validated, and merge-friendly (consistent adapter shapes).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Callable, Any, Dict
import inspect
import warnings

from trl import SFTTrainer, SFTConfig
from peft import LoraConfig as PeftLoraConfig

from .models import LLMAdapter


# ----------------------------
# Configs
# ----------------------------

@dataclass
class LoRAParams:
    """
    LoRA configuration for parameter-efficient fine-tuning.

    NOTE: For cultural fine-tuning intended to be comparable/mergeable across buckets,
    keep these identical across all runs for a given base model family.
    """
    r: int = 16
    lora_alpha: int = 32
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])
    lora_dropout: float = 0.05
    bias: str = "none"
    task_type: str = "CAUSAL_LM"


@dataclass
class TrainConfig:
    """
    Training configuration for SFT fine-tuning.
    """
    output_dir: str

    # Dataset
    text_field: str = "text"
    formatting_func: Optional[Callable[[Dict[str, Any]], str]] = None

    # Training length (prefer max_steps when dataset is streaming/iterable)
    max_steps: Optional[int] = None
    num_train_epochs: Optional[float] = 1.0

    # Core hyperparams
    learning_rate: float = 1e-4
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03  # better default than warmup_steps=5
    lr_scheduler_type: str = "linear"
    max_grad_norm: float = 1.0
    seed: int = 3407

    # Sequence handling
    max_length: Optional[int] = 1024  # TRL uses max_length in newer versions :contentReference[oaicite:3]{index=3}
    packing: bool = True
    eval_packing: bool = False  # avoid eval errors on tiny eval sets

    # Precision / memory
    bf16: bool = True
    fp16: bool = False
    gradient_checkpointing: bool = True

    # Logging / saving / eval
    logging_steps: int = 10
    save_strategy: str = "no"       # you explicitly save at the end
    evaluation_strategy: str = "no" # flip to "steps" if you pass eval_dataset
    eval_steps: int = 200
    save_total_limit: int = 1
    report_to: str = "none"

    # Optimizer (avoid brittle defaults)
    optim: str = "adamw_torch"  # safer default than adamw_8bit

    # LoRA (if None -> full fine-tune)
    lora: Optional[LoRAParams] = None


# ----------------------------
# Helpers
# ----------------------------

def _ensure_tokenizer_padding(tokenizer) -> None:
    """Make padding explicit and deterministic for causal LM."""
    if tokenizer.pad_token is None:
        # TRL docs: pad token must be set; eos often used as fallback :contentReference[oaicite:4]{index=4}
        tokenizer.pad_token = tokenizer.eos_token
    if getattr(tokenizer, "padding_side", None) != "right":
        tokenizer.padding_side = "right"


def _validate_lora_targets(model, target_modules: List[str]) -> None:
    """
    Fail fast if target module names don't exist.
    Without this, LoRA can silently attach to nothing (worst kind of bug).
    """
    names = [n for n, _ in model.named_modules()]
    found = {tm for tm in target_modules if any(n.endswith(tm) or f".{tm}" in n for n in names)}
    missing = [tm for tm in target_modules if tm not in found]
    if len(found) == 0:
        raise ValueError(
            f"LoRA target_modules matched NOTHING on this model. "
            f"target_modules={target_modules}. "
            f"Example module names (first 50): {names[:50]}"
        )
    if missing:
        warnings.warn(
            f"Some LoRA target_modules were not found: {missing}. "
            f"Found: {sorted(found)}. This may be fine if the model doesn't have those layers."
        )


def _make_peft_lora_config(lora: LoRAParams) -> PeftLoraConfig:
    return PeftLoraConfig(
        r=lora.r,
        lora_alpha=lora.lora_alpha,
        target_modules=lora.target_modules,
        lora_dropout=lora.lora_dropout,
        bias=lora.bias,
        task_type=lora.task_type,
        inference_mode=False,
    )


def _filter_kwargs_for_dataclass(cls, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """
    TRL changes config fields across versions.
    Filter kwargs to only those that exist on the installed SFTConfig.
    """
    fields = getattr(cls, "__dataclass_fields__", None)
    if fields:
        allowed = set(fields.keys())
    else:
        allowed = set(inspect.signature(cls.__init__).parameters.keys())

    filtered = {k: v for k, v in kwargs.items() if (k in allowed and v is not None)}
    dropped = [k for k in kwargs.keys() if k not in filtered and kwargs[k] is not None and k not in allowed]
    if dropped:
        warnings.warn(f"Dropping unsupported SFTConfig args for this TRL version: {dropped}")
    return filtered


def _build_formatting_func(cfg: TrainConfig) -> Callable[[Dict[str, Any]], str]:
    if cfg.formatting_func is not None:
        return cfg.formatting_func

    field = cfg.text_field

    def _fmt(ex: Dict[str, Any]) -> str:
        val = ex.get(field, "")
        if val is None:
            return ""
        if isinstance(val, str):
            return val
        # Common cases: list of strings
        if isinstance(val, list):
            return "\n".join(str(x) for x in val)
        return str(val)

    return _fmt


# ----------------------------
# Main entry
# ----------------------------

def finetune_causal_lm(
    adapter: LLMAdapter,
    train_dataset,
    cfg: TrainConfig,
    eval_dataset=None,
) -> LLMAdapter:
    """
    Fine-tune the model in adapter on train_dataset using TRL SFTTrainer.

    Returns the SAME adapter object, with model weights updated.
    Saves model + tokenizer to cfg.output_dir.
    """
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)

    model = adapter.model
    tokenizer = adapter.tokenizer
    _ensure_tokenizer_padding(tokenizer)

    # Gradient checkpointing stability
    if cfg.gradient_checkpointing:
        # Many HF models need use_cache=False when checkpointing
        if hasattr(model, "config") and hasattr(model.config, "use_cache"):
            model.config.use_cache = False

    peft_config = None
    if cfg.lora is not None:
        _validate_lora_targets(model, cfg.lora.target_modules)
        peft_config = _make_peft_lora_config(cfg.lora)

    # If eval_dataset is provided, don't forget to actually evaluate
    evaluation_strategy = cfg.evaluation_strategy
    if eval_dataset is not None and evaluation_strategy == "no":
        evaluation_strategy = "steps"

    sft_kwargs = dict(
        output_dir=cfg.output_dir,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        warmup_ratio=cfg.warmup_ratio,
        lr_scheduler_type=cfg.lr_scheduler_type,
        max_grad_norm=cfg.max_grad_norm,
        logging_steps=cfg.logging_steps,
        save_total_limit=cfg.save_total_limit,
        report_to=cfg.report_to,
        seed=cfg.seed,

        # Newer TRL uses max_length and packing in SFTConfig :contentReference[oaicite:5]{index=5}
        max_length=cfg.max_length,
        packing=cfg.packing,
        eval_packing=cfg.eval_packing,

        save_strategy=cfg.save_strategy,
        evaluation_strategy=evaluation_strategy,
        eval_steps=cfg.eval_steps,

        bf16=cfg.bf16,
        fp16=cfg.fp16,
        gradient_checkpointing=cfg.gradient_checkpointing,
        optim=cfg.optim,
    )

    # Train length: prefer max_steps when specified
    if cfg.max_steps is not None:
        sft_kwargs["max_steps"] = cfg.max_steps
    elif cfg.num_train_epochs is not None:
        sft_kwargs["num_train_epochs"] = cfg.num_train_epochs

    args = SFTConfig(**_filter_kwargs_for_dataclass(SFTConfig, sft_kwargs))

    formatting_func = _build_formatting_func(cfg)

    # TRL changed tokenizer arg name to processing_class :contentReference[oaicite:6]{index=6}
    trainer_init_sig = inspect.signature(SFTTrainer.__init__).parameters
    trainer_kwargs = dict(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        formatting_func=formatting_func,
        peft_config=peft_config,
    )

    if "processing_class" in trainer_init_sig:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in trainer_init_sig:
        trainer_kwargs["tokenizer"] = tokenizer
    else:
        # Extremely unlikely, but fail loudly
        raise RuntimeError("SFTTrainer has neither processing_class nor tokenizer parameter in this TRL version.")

    trainer = SFTTrainer(**trainer_kwargs)

    trainer.train()
    trainer.save_model(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)

    # Keep adapter in sync with trainer's model (could be wrapped by PEFT)
    adapter.model = trainer.model
    return adapter