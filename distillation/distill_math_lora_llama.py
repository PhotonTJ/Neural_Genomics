#!/usr/bin/env python
"""
LoRA distillation script for transferring math behavior from Llama 3.1 8B
into Llama 3 8B.

The script trains LoRA adapters on the student model while keeping the teacher
frozen. The loss is:

    total_loss = ce_weight * supervised_ce + kl_weight * distillation_kl

Supervised CE is applied only on answer tokens, while KL distillation is
applied on the same answer-token positions using the teacher's logits.

Example:
    python distill_math_lora.py \
        --dataset_name gsm8k \
        --dataset_config main \
        --train_split train \
        --eval_split test \
        --prompt_column question \
        --answer_column answer \
        --output_dir outputs/llama3-math-distill \
        --per_device_train_batch_size 1 \
        --gradient_accumulation_steps 16 \
        --learning_rate 2e-4 \
        --num_train_epochs 1 \
        --bf16 True

Requirements:
    pip install "transformers>=4.46" datasets peft accelerate bitsandbytes

Notes:
    - Defaults target gated Meta checkpoints on Hugging Face. Make sure you
      have accepted the license and are authenticated.
    - Running two 8B models at once is memory intensive. The default 4-bit
      loading is deliberate.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from datasets import Dataset, DatasetDict, get_dataset_config_names, load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
    set_seed,
)


LOGGER = logging.getLogger("distill_math_lora")
PROMPT_FALLBACK_TEMPLATE = "Solve the following math problem.\n\nProblem:\n{prompt}\n\nSolution:\n"
PROMPT_CANDIDATES = (
    "problem",
    "question",
    "prompt",
    "instruction",
    "query",
    "input",
)
ANSWER_CANDIDATES = (
    "solution",
    "answer",
    "output",
    "response",
    "completion",
    "label",
    "rationale",
)


@dataclass
class ModelArguments:
    teacher_model_name_or_path: str = field(
        default="meta-llama/Llama-3.1-8B-Instruct",
        metadata={"help": "Teacher checkpoint."},
    )
    student_model_name_or_path: str = field(
        default="meta-llama/Meta-Llama-3-8B-Instruct",
        metadata={"help": "Student checkpoint to LoRA-tune."},
    )
    tokenizer_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "Tokenizer checkpoint. Defaults to the student model."},
    )
    use_4bit: bool = field(
        default=True,
        metadata={"help": "Load models in 4-bit with bitsandbytes."},
    )
    use_8bit: bool = field(
        default=False,
        metadata={"help": "Load models in 8-bit with bitsandbytes."},
    )
    bnb_4bit_quant_type: str = field(
        default="nf4",
        metadata={"help": "4-bit quant type for bitsandbytes."},
    )
    bnb_4bit_use_double_quant: bool = field(
        default=True,
        metadata={"help": "Enable double quantization for 4-bit loading."},
    )
    torch_dtype: str = field(
        default="bfloat16",
        metadata={"help": "Model dtype: auto, float16, bfloat16, float32."},
    )
    attn_implementation: Optional[str] = field(
        default=None,
        metadata={"help": "Optional attention implementation, e.g. flash_attention_2."},
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={"help": "Pass trust_remote_code=True to from_pretrained()."},
    )


@dataclass
class DataArguments:
    dataset_name: Optional[str] = field(
        default=None,
        metadata={"help": "Hugging Face dataset name."},
    )
    dataset_config: Optional[str] = field(
        default=None,
        metadata={"help": "Optional dataset config name."},
    )
    train_split: str = field(
        default="train",
        metadata={"help": "Training split name."},
    )
    eval_split: Optional[str] = field(
        default=None,
        metadata={"help": "Optional evaluation split name."},
    )
    validation_fraction: float = field(
        default=0.0,
        metadata={"help": "If > 0, carve this fraction out of train as eval when no eval split/file is provided."},
    )
    train_file: Optional[str] = field(
        default=None,
        metadata={"help": "Path to a JSON/JSONL/Parquet/CSV training file."},
    )
    eval_file: Optional[str] = field(
        default=None,
        metadata={"help": "Path to an optional evaluation file."},
    )
    prompt_column: Optional[str] = field(
        default=None,
        metadata={"help": "Column containing the math question/problem."},
    )
    answer_column: Optional[str] = field(
        default=None,
        metadata={"help": "Column containing the reference solution/answer."},
    )
    max_length: int = field(
        default=1024,
        metadata={"help": "Maximum packed sequence length."},
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "Number of dataset preprocessing workers."},
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={"help": "Optional cap for train examples."},
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={"help": "Optional cap for eval examples."},
    )
    system_prompt: Optional[str] = field(
        default="You are a careful math tutor. Show the reasoning needed to solve the problem correctly.",
        metadata={"help": "Optional system prompt used when formatting instruct examples."},
    )


@dataclass
class DistillationArguments:
    lora_r: int = field(default=64, metadata={"help": "LoRA rank."})
    lora_alpha: int = field(default=128, metadata={"help": "LoRA alpha."})
    lora_dropout: float = field(default=0.05, metadata={"help": "LoRA dropout."})
    lora_target_modules: str = field(
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        metadata={"help": "Comma-separated list of target modules."},
    )
    kl_weight: float = field(
        default=0.7,
        metadata={"help": "Weight for the KL distillation loss."},
    )
    ce_weight: float = field(
        default=0.3,
        metadata={"help": "Weight for the supervised CE loss."},
    )
    temperature: float = field(
        default=2.0,
        metadata={"help": "Distillation temperature."},
    )


class DistillationCollator:
    def __init__(self, tokenizer: PreTrainedTokenizerBase) -> None:
        self.tokenizer = tokenizer

    def __call__(self, features: Sequence[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
        max_len = max(len(feature["input_ids"]) for feature in features)
        input_ids: List[List[int]] = []
        attention_masks: List[List[int]] = []
        labels: List[List[int]] = []

        for feature in features:
            pad_len = max_len - len(feature["input_ids"])
            input_ids.append(feature["input_ids"] + [self.tokenizer.pad_token_id] * pad_len)
            attention_masks.append(feature["attention_mask"] + [0] * pad_len)
            labels.append(feature["labels"] + [-100] * pad_len)

        batch = {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }
        return batch


class DistillationTrainer(Trainer):
    def __init__(
        self,
        *args: Any,
        teacher_model: torch.nn.Module,
        kl_weight: float,
        ce_weight: float,
        temperature: float,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.teacher_model = teacher_model
        self.kl_weight = kl_weight
        self.ce_weight = ce_weight
        self.temperature = temperature

    def compute_loss(
        self,
        model: torch.nn.Module,
        inputs: Dict[str, torch.Tensor],
        return_outputs: bool = False,
        num_items_in_batch: Optional[torch.Tensor] = None,
    ) -> Any:
        labels = inputs["labels"]
        student_outputs = model(**inputs)
        ce_loss = student_outputs.loss

        if self.kl_weight == 0.0:
            loss = ce_loss
            return (loss, student_outputs) if return_outputs else loss

        with torch.no_grad():
            teacher_outputs = self.teacher_model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
            )

        student_logits = student_outputs.logits[:, :-1, :]
        teacher_logits = teacher_outputs.logits[:, :-1, :]
        target_mask = labels[:, 1:] != -100

        if not torch.any(target_mask):
            distill_loss = ce_loss.new_zeros(())
        else:
            temperature = self.temperature
            student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
            teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
            tokenwise_kl = F.kl_div(student_log_probs, teacher_probs, reduction="none").sum(dim=-1)
            distill_loss = (tokenwise_kl * target_mask).sum() / target_mask.sum().clamp_min(1)
            distill_loss = distill_loss * (temperature ** 2)

        loss = (self.ce_weight * ce_loss) + (self.kl_weight * distill_loss)
        self.log(
            {
                "ce_loss": ce_loss.detach().float().item(),
                "distill_kl": distill_loss.detach().float().item(),
            }
        )
        return (loss, student_outputs) if return_outputs else loss


def parse_args() -> Tuple[ModelArguments, DataArguments, TrainingArguments, DistillationArguments]:
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments, DistillationArguments))
    model_args, data_args, training_args, distill_args = parser.parse_args_into_dataclasses()
    training_args.remove_unused_columns = False
    return model_args, data_args, training_args, distill_args


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


def get_quantization_config(model_args: ModelArguments) -> Optional[BitsAndBytesConfig]:
    if model_args.use_4bit and model_args.use_8bit:
        raise ValueError("Choose either --use_4bit or --use_8bit, not both.")
    if not model_args.use_4bit and not model_args.use_8bit:
        return None

    compute_dtype = get_torch_dtype(model_args.torch_dtype) or torch.bfloat16
    return BitsAndBytesConfig(
        load_in_4bit=model_args.use_4bit,
        load_in_8bit=model_args.use_8bit,
        bnb_4bit_quant_type=model_args.bnb_4bit_quant_type,
        bnb_4bit_use_double_quant=model_args.bnb_4bit_use_double_quant,
        bnb_4bit_compute_dtype=compute_dtype,
    )


def get_device_map(quantized: bool) -> Optional[Dict[str, int]]:
    if not quantized or not torch.cuda.is_available():
        return None
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    return {"": local_rank}


def infer_columns(dataset: Dataset, prompt_column: Optional[str], answer_column: Optional[str]) -> Tuple[str, str]:
    columns = set(dataset.column_names)
    if prompt_column is None:
        prompt_column = next((name for name in PROMPT_CANDIDATES if name in columns), None)
    if answer_column is None:
        answer_column = next((name for name in ANSWER_CANDIDATES if name in columns), None)
    if prompt_column is None or answer_column is None:
        raise ValueError(
            "Could not infer prompt/answer columns. Pass --prompt_column and --answer_column explicitly."
        )
    return prompt_column, answer_column


def load_raw_datasets(data_args: DataArguments, split_seed: int) -> DatasetDict:
    if not 0.0 <= data_args.validation_fraction < 1.0:
        raise ValueError("--validation_fraction must be in the range [0, 1).")
    if data_args.validation_fraction > 0.0 and data_args.eval_split is not None:
        raise ValueError("Use either --eval_split or --validation_fraction, not both.")
    if data_args.validation_fraction > 0.0 and data_args.eval_file is not None:
        raise ValueError("Use either --eval_file or --validation_fraction, not both.")

    if data_args.dataset_name:
        available_configs = get_dataset_config_names(data_args.dataset_name)
        requested_config = data_args.dataset_config

        if requested_config is None:
            if len(available_configs) == 1:
                requested_config = available_configs[0]
                LOGGER.info(
                    "Using the only available config '%s' for %s.",
                    requested_config,
                    data_args.dataset_name,
                )
            elif "main" in available_configs:
                requested_config = "main"
                LOGGER.info(
                    "No dataset config was provided for %s. Defaulting to 'main'.",
                    data_args.dataset_name,
                )

        def load_named_split(split_name: str) -> Dataset:
            try:
                return load_dataset(
                    data_args.dataset_name,
                    requested_config,
                    split=split_name,
                )
            except ValueError:
                if requested_config is None:
                    raise
                if len(available_configs) != 1 or requested_config in available_configs:
                    raise

                fallback_config = available_configs[0]
                LOGGER.warning(
                    "Dataset config '%s' was not found for %s. Falling back to '%s'.",
                    requested_config,
                    data_args.dataset_name,
                    fallback_config,
                )
                return load_dataset(
                    data_args.dataset_name,
                    fallback_config,
                    split=split_name,
                )

        datasets = DatasetDict()
        datasets["train"] = load_named_split(data_args.train_split)
        if data_args.eval_split:
            datasets["eval"] = load_named_split(data_args.eval_split)
        elif data_args.validation_fraction > 0.0:
            split_datasets = datasets["train"].train_test_split(
                test_size=data_args.validation_fraction,
                seed=split_seed,
                shuffle=True,
            )
            datasets["train"] = split_datasets["train"]
            datasets["eval"] = split_datasets["test"]
        return datasets

    if not data_args.train_file:
        raise ValueError("Provide either --dataset_name or --train_file.")

    data_files = {"train": data_args.train_file}
    if data_args.eval_file:
        data_files["eval"] = data_args.eval_file

    extension = data_args.train_file.rsplit(".", 1)[-1].lower()
    if extension == "jsonl":
        extension = "json"
    datasets = load_dataset(extension, data_files=data_files)
    if "eval" not in datasets and data_args.validation_fraction > 0.0:
        split_datasets = datasets["train"].train_test_split(
            test_size=data_args.validation_fraction,
            seed=split_seed,
            shuffle=True,
        )
        datasets = DatasetDict(
            {
                "train": split_datasets["train"],
                "eval": split_datasets["test"],
            }
        )
    return DatasetDict(datasets)


def maybe_apply_chat_template(
    tokenizer: PreTrainedTokenizerBase,
    prompt_text: str,
    system_prompt: Optional[str],
) -> str:
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt_text})
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    return PROMPT_FALLBACK_TEMPLATE.format(prompt=prompt_text)


def build_preprocess_function(
    tokenizer: PreTrainedTokenizerBase,
    prompt_column: str,
    answer_column: str,
    max_length: int,
    system_prompt: Optional[str],
):
    eos_token = tokenizer.eos_token or ""

    def preprocess_batch(batch: Dict[str, List[Any]]) -> Dict[str, List[List[int]]]:
        encoded_examples = {"input_ids": [], "attention_mask": [], "labels": []}
        prompts = batch[prompt_column]
        answers = batch[answer_column]

        for prompt, answer in zip(prompts, answers):
            if prompt is None or answer is None:
                continue

            prompt_text = str(prompt).strip()
            answer_text = str(answer).strip()
            if not prompt_text or not answer_text:
                continue

            rendered_prompt = maybe_apply_chat_template(tokenizer, prompt_text, system_prompt)
            full_text = rendered_prompt + answer_text + eos_token

            prompt_ids = tokenizer(
                rendered_prompt,
                add_special_tokens=False,
                truncation=True,
                max_length=max_length,
            )["input_ids"]
            full_encoding = tokenizer(
                full_text,
                add_special_tokens=False,
                truncation=True,
                max_length=max_length,
            )
            input_ids = full_encoding["input_ids"]
            attention_mask = full_encoding["attention_mask"]

            if len(prompt_ids) >= len(input_ids):
                continue

            labels = input_ids.copy()
            for idx in range(min(len(prompt_ids), len(labels))):
                labels[idx] = -100

            encoded_examples["input_ids"].append(input_ids)
            encoded_examples["attention_mask"].append(attention_mask)
            encoded_examples["labels"].append(labels)

        return encoded_examples

    return preprocess_batch


def select_subset(dataset: Optional[Dataset], max_samples: Optional[int]) -> Optional[Dataset]:
    if dataset is None or max_samples is None:
        return dataset
    return dataset.select(range(min(len(dataset), max_samples)))


def prepare_datasets(
    raw_datasets: DatasetDict,
    tokenizer: PreTrainedTokenizerBase,
    data_args: DataArguments,
) -> Tuple[Dataset, Optional[Dataset], str, str]:
    prompt_column, answer_column = infer_columns(
        raw_datasets["train"],
        data_args.prompt_column,
        data_args.answer_column,
    )
    preprocess_fn = build_preprocess_function(
        tokenizer=tokenizer,
        prompt_column=prompt_column,
        answer_column=answer_column,
        max_length=data_args.max_length,
        system_prompt=data_args.system_prompt,
    )

    train_dataset = select_subset(raw_datasets["train"], data_args.max_train_samples)
    eval_dataset = select_subset(raw_datasets.get("eval"), data_args.max_eval_samples)

    train_dataset = train_dataset.map(
        preprocess_fn,
        batched=True,
        remove_columns=train_dataset.column_names,
        num_proc=data_args.preprocessing_num_workers,
        desc="Tokenizing train split",
    )
    if len(train_dataset) == 0:
        raise ValueError("No train examples survived preprocessing. Check your columns and max_length.")

    if eval_dataset is not None:
        eval_dataset = eval_dataset.map(
            preprocess_fn,
            batched=True,
            remove_columns=eval_dataset.column_names,
            num_proc=data_args.preprocessing_num_workers,
            desc="Tokenizing eval split",
        )
        if len(eval_dataset) == 0:
            LOGGER.warning("All eval examples were filtered during preprocessing; evaluation will be skipped.")
            eval_dataset = None

    return train_dataset, eval_dataset, prompt_column, answer_column


def create_tokenizer(model_args: ModelArguments) -> PreTrainedTokenizerBase:
    tokenizer_name = model_args.tokenizer_name_or_path or model_args.student_model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        trust_remote_code=model_args.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def create_student_model(
    model_args: ModelArguments,
    distill_args: DistillationArguments,
    training_args: TrainingArguments,
) -> torch.nn.Module:
    quantization_config = get_quantization_config(model_args)
    quantized = quantization_config is not None
    model_kwargs = {
        "pretrained_model_name_or_path": model_args.student_model_name_or_path,
        "quantization_config": quantization_config,
        "torch_dtype": get_torch_dtype(model_args.torch_dtype),
        "trust_remote_code": model_args.trust_remote_code,
        "low_cpu_mem_usage": True,
        "device_map": get_device_map(quantized),
    }
    if model_args.attn_implementation:
        model_kwargs["attn_implementation"] = model_args.attn_implementation

    model = AutoModelForCausalLM.from_pretrained(**model_kwargs)
    model.config.use_cache = False

    if quantized:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=training_args.gradient_checkpointing,
        )

    lora_config = LoraConfig(
        r=distill_args.lora_r,
        lora_alpha=distill_args.lora_alpha,
        lora_dropout=distill_args.lora_dropout,
        target_modules=[name.strip() for name in distill_args.lora_target_modules.split(",") if name.strip()],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


def create_teacher_model(model_args: ModelArguments, training_args: TrainingArguments) -> torch.nn.Module:
    quantization_config = get_quantization_config(model_args)
    quantized = quantization_config is not None
    model_kwargs = {
        "pretrained_model_name_or_path": model_args.teacher_model_name_or_path,
        "quantization_config": quantization_config,
        "torch_dtype": get_torch_dtype(model_args.torch_dtype),
        "trust_remote_code": model_args.trust_remote_code,
        "low_cpu_mem_usage": True,
        "device_map": get_device_map(quantized),
    }
    if model_args.attn_implementation:
        model_kwargs["attn_implementation"] = model_args.attn_implementation

    teacher = AutoModelForCausalLM.from_pretrained(**model_kwargs)
    teacher.eval()
    teacher.config.use_cache = True

    if not quantized:
        teacher.to(training_args.device)

    for parameter in teacher.parameters():
        parameter.requires_grad = False

    return teacher


def main() -> None:
    model_args, data_args, training_args, distill_args = parse_args()
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=training_args.get_process_log_level(),
    )
    LOGGER.setLevel(training_args.get_process_log_level())

    if distill_args.ce_weight < 0 or distill_args.kl_weight < 0:
        raise ValueError("Loss weights must be non-negative.")
    if distill_args.ce_weight == 0 and distill_args.kl_weight == 0:
        raise ValueError("At least one of ce_weight or kl_weight must be > 0.")
    if distill_args.temperature <= 0:
        raise ValueError("Temperature must be > 0.")

    set_seed(training_args.seed)

    tokenizer = create_tokenizer(model_args)
    raw_datasets = load_raw_datasets(data_args, training_args.seed)
    train_dataset, eval_dataset, prompt_column, answer_column = prepare_datasets(
        raw_datasets,
        tokenizer,
        data_args,
    )
    LOGGER.info("Using prompt column '%s' and answer column '%s'.", prompt_column, answer_column)
    LOGGER.info("Prepared %s train examples.", len(train_dataset))
    if eval_dataset is not None:
        LOGGER.info("Prepared %s eval examples.", len(eval_dataset))
    elif training_args.do_eval:
        LOGGER.warning("Evaluation was requested, but no eval split was available after dataset preparation.")

    student_model = create_student_model(model_args, distill_args, training_args)
    teacher_model = create_teacher_model(model_args, training_args)
    if teacher_model.config.vocab_size != student_model.config.vocab_size:
        raise ValueError(
            "Teacher and student vocab sizes do not match. Use models with compatible tokenizers."
        )

    data_collator = DistillationCollator(tokenizer)
    trainer = DistillationTrainer(
        model=student_model,
        teacher_model=teacher_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        kl_weight=distill_args.kl_weight,
        ce_weight=distill_args.ce_weight,
        temperature=distill_args.temperature,
    )

    train_result = trainer.train()
    trainer.save_model()
    tokenizer.save_pretrained(training_args.output_dir)

    metrics = train_result.metrics
    metrics["train_samples"] = len(train_dataset)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    if eval_dataset is not None and training_args.do_eval:
        eval_metrics = trainer.evaluate()
        eval_metrics["eval_samples"] = len(eval_dataset)
        trainer.log_metrics("eval", eval_metrics)
        trainer.save_metrics("eval", eval_metrics)


if __name__ == "__main__":
    main()
