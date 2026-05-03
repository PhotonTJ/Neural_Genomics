"""
SFT Training Script: OpenHermes 2.5 + LoRA for multiple 8B/9B base models

Supports:
- mistralai/Ministral-3-8B-Base-2512
- Qwen/Qwen3.5-9B-Base
- meta-llama/Llama-3.1-8B

The training recipe matches Llama3_SFT.py:
- same dataset
- same sample count
- same LoRA config
- same train/eval split logic
- same SFT hyperparameters

Usage:
    python multi_model_SFT.py --model ministral
    python multi_model_SFT.py --model qwen
    python multi_model_SFT.py --model llama31
    python multi_model_SFT.py --model all
"""

import argparse
import os
import types

import torch
from datasets import DatasetDict, load_dataset
from peft import LoraConfig, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer


# =============================================================================
# Shared config
# =============================================================================
DATASET_NAME = "teknium/OpenHermes-2.5"
CACHE_DIR = "./dataset_cache"
NUM_SAMPLES = 100_000
MAX_SEQ_LENGTH = 2048
SEED = 42
DEFAULT_PER_DEVICE_TRAIN_BATCH_SIZE = 8
DEFAULT_PER_DEVICE_EVAL_BATCH_SIZE = 8
DEFAULT_GRADIENT_ACCUMULATION_STEPS = 2
DEFAULT_DATALOADER_NUM_WORKERS = 8
DEFAULT_USE_BF16 = True
DEFAULT_USE_FP16 = False
DEFAULT_USE_GRADIENT_CHECKPOINTING = False
DEFAULT_ATTN_IMPLEMENTATION = "flash_attention_2"


# =============================================================================
# Shared LoRA config
# =============================================================================
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
)


LLAMA_CHAT_TEMPLATE_WITH_GENERATION = (
    "{% set loop_messages = messages %}"
    "{% for message in loop_messages %}"
    "{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\\n\\n' + message['content'] | trim + '<|eot_id|>' %}"
    "{% if message['role'] == 'assistant' %}"
    "{% generation %}{{ content }}{% endgeneration %}"
    "{% else %}"
    "{{ content }}"
    "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "{{ '<|start_header_id|>assistant<|end_header_id|>\\n\\n' }}"
    "{% endif %}"
)


QWEN_CHAT_TEMPLATE_WITH_GENERATION = (
    "{% for message in messages %}"
    "{% if message['role'] == 'assistant' %}"
    "{% generation %}"
    "{{ '<|im_start|>' + message['role'] + '\\n' + message['content'] | trim + '<|im_end|>\\n' }}"
    "{% endgeneration %}"
    "{% else %}"
    "{{ '<|im_start|>' + message['role'] + '\\n' + message['content'] | trim + '<|im_end|>\\n' }}"
    "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "{{ '<|im_start|>assistant\\n' }}"
    "{% endif %}"
)


MISTRAL_CHAT_TEMPLATE_WITH_GENERATION = (
    "{% set ns = namespace(system_prompt='') %}"
    "{% for message in messages %}"
    "{% if message['role'] == 'system' %}"
    "{% set ns.system_prompt = message['content'] | trim %}"
    "{% elif message['role'] == 'user' %}"
    "{% set user_content = message['content'] | trim %}"
    "{% if ns.system_prompt %}"
    "{% set user_content = ns.system_prompt + '\\n\\n' + user_content %}"
    "{% set ns.system_prompt = '' %}"
    "{% endif %}"
    "{{ bos_token + '[INST] ' + user_content + ' [/INST]' }}"
    "{% elif message['role'] == 'assistant' %}"
    "{% generation %}{{ ' ' + (message['content'] | trim) + eos_token }}{% endgeneration %}"
    "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ bos_token + '[INST] ' }}{% endif %}"
)


MODEL_CONFIGS = {
    "ministral": {
        "model_name": "mistralai/Ministral-3-8B-Base-2512",
        "tokenizer_name": "mistralai/Ministral-3-8B-Base-2512",
        "output_dir": "./ministral3-openhermes-lora",
        "chat_template": MISTRAL_CHAT_TEMPLATE_WITH_GENERATION,
        "special_tokens_to_fix": [],
        "loader": "mistral3",
    },
    "qwen": {
        "model_name": "Qwen/Qwen3.5-9B-Base",
        "tokenizer_name": "Qwen/Qwen3.5-9B-Base",
        "output_dir": "./qwen35-openhermes-lora",
        "chat_template": QWEN_CHAT_TEMPLATE_WITH_GENERATION,
        "special_tokens_to_fix": [],
        "loader": "auto",
    },
    "llama31": {
        "model_name": "meta-llama/Llama-3.1-8B",
        "tokenizer_name": "meta-llama/Llama-3.1-8B",
        "output_dir": "./llama31-openhermes-lora",
        "chat_template": LLAMA_CHAT_TEMPLATE_WITH_GENERATION,
        "special_tokens_to_fix": [
            "<|eot_id|>",
            "<|start_header_id|>",
            "<|end_header_id|>",
            "<|finetune_right_pad_id|>",
        ],
        "regular_token_count": 128000,
        "loader": "auto",
    },
}


def patch_tokenizer_with_assistant_mask_support(tokenizer):
    original_apply_chat_template = tokenizer.apply_chat_template

    def render_messages(self, messages, add_generation_prompt):
        extra_kwargs = {}
        if messages and messages[-1].get("role") == "assistant":
            extra_kwargs["continue_final_message"] = True
        return original_apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            **extra_kwargs,
        )

    def encode_rendered(self, rendered_text):
        input_ids = list(self.encode(rendered_text))
        return input_ids, [1] * len(input_ids)

    def apply_chat_template_with_masks(self, messages, *args, **kwargs):
        wants_assistant_mask = kwargs.get("return_assistant_tokens_mask", False)
        patched_kwargs = dict(kwargs)
        if messages and messages[-1].get("role") == "assistant":
            patched_kwargs.setdefault("continue_final_message", True)

        if not wants_assistant_mask:
            return original_apply_chat_template(messages, *args, **patched_kwargs)

        tokenize = kwargs.get("tokenize", False)
        return_dict = kwargs.get("return_dict", False)
        add_generation_prompt = kwargs.get("add_generation_prompt", False)
        if not tokenize:
            patched_kwargs.pop("return_assistant_tokens_mask", None)
            return original_apply_chat_template(messages, *args, **patched_kwargs)

        full_rendered = self._render_messages(messages, add_generation_prompt)
        full_input_ids, full_attention_mask = self._encode_rendered(full_rendered)
        assistant_masks = [0] * len(full_input_ids)

        for idx, message in enumerate(messages):
            if message.get("role") != "assistant":
                continue
            prefix_messages = messages[:idx]
            upto_messages = messages[: idx + 1]
            prefix_rendered = self._render_messages(prefix_messages, False)
            upto_rendered = self._render_messages(upto_messages, False)
            prefix_input_ids, _ = self._encode_rendered(prefix_rendered)
            upto_input_ids, _ = self._encode_rendered(upto_rendered)

            start = len(prefix_input_ids)
            end = len(upto_input_ids)
            for pos in range(start, min(end, len(assistant_masks))):
                assistant_masks[pos] = 1

        if return_dict:
            return {
                "input_ids": full_input_ids,
                "attention_mask": full_attention_mask,
                "assistant_masks": assistant_masks,
            }
        return full_input_ids

    tokenizer._render_messages = types.MethodType(render_messages, tokenizer)
    tokenizer._encode_rendered = types.MethodType(encode_rendered, tokenizer)
    tokenizer.apply_chat_template = types.MethodType(apply_chat_template_with_masks, tokenizer)
    return tokenizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=["ministral", "qwen", "llama31", "all"],
        required=True,
        help="Which model recipe to run.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_PER_DEVICE_TRAIN_BATCH_SIZE,
        help="Per-device train batch size.",
    )
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=DEFAULT_PER_DEVICE_EVAL_BATCH_SIZE,
        help="Per-device eval batch size.",
    )
    parser.add_argument(
        "--grad-accum",
        type=int,
        default=DEFAULT_GRADIENT_ACCUMULATION_STEPS,
        help="Gradient accumulation steps.",
    )
    parser.add_argument(
        "--dataloader-workers",
        type=int,
        default=DEFAULT_DATALOADER_NUM_WORKERS,
        help="Dataloader worker count.",
    )
    parser.add_argument(
        "--attn-impl",
        choices=["flash_attention_2", "sdpa", "eager"],
        default=DEFAULT_ATTN_IMPLEMENTATION,
        help="Preferred attention backend.",
    )
    parser.add_argument(
        "--no-flash-attn",
        action="store_true",
        help="Shortcut for --attn-impl sdpa.",
    )
    parser.add_argument(
        "--bf16",
        dest="bf16",
        action="store_true",
        default=DEFAULT_USE_BF16,
        help="Enable BF16.",
    )
    parser.add_argument(
        "--fp16",
        dest="bf16",
        action="store_false",
        help="Use FP16 instead of BF16.",
    )
    parser.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        default=DEFAULT_USE_GRADIENT_CHECKPOINTING,
        help="Enable gradient checkpointing.",
    )
    return parser.parse_args()


def load_openhermes_dataset():
    processed_cache_path = os.path.join(CACHE_DIR, f"openhermes_processed_{NUM_SAMPLES}")

    if os.path.exists(processed_cache_path):
        print(f"Loading processed dataset from cache: {processed_cache_path}")
        cached = DatasetDict.load_from_disk(processed_cache_path)
        train_dataset = cached["train"]
        eval_dataset = cached["test"]
    else:
        print("Processing dataset (will be cached for next time)...")
        dataset = load_dataset(DATASET_NAME, split="train")
        dataset = dataset.shuffle(seed=SEED).select(range(NUM_SAMPLES))

        def format_to_conversational(example):
            role_map = {"human": "user", "gpt": "assistant", "system": "system"}
            messages = []
            for msg in example["conversations"]:
                role = role_map.get(msg.get("from", ""), "user")
                content = msg.get("value", "")
                if content:
                    messages.append({"role": role, "content": content})
            return {"messages": messages}

        dataset = dataset.map(
            format_to_conversational,
            batched=False,
            remove_columns=dataset.column_names,
            desc="Converting to conversational format",
        )

        split_dataset = dataset.train_test_split(test_size=0.05, seed=SEED)
        train_dataset = split_dataset["train"]
        eval_dataset = split_dataset["test"]

        os.makedirs(CACHE_DIR, exist_ok=True)
        split_dataset.save_to_disk(processed_cache_path)
        print(f"Dataset cached to: {processed_cache_path}")

    print(f"Train samples: {len(train_dataset)}")
    print(f"Eval samples:  {len(eval_dataset)}")
    print("\n--- Example conversation ---")
    for msg in train_dataset[0]["messages"][:2]:
        print(f"  [{msg['role']}]: {msg['content'][:100]}...")
    print()

    return train_dataset, eval_dataset


def prepare_tokenizer(config):
    if config.get("loader") == "mistral3":
        try:
            from transformers import MistralCommonBackend
        except ImportError as exc:
            raise RuntimeError(
                "Ministral-3 tokenizer requires `mistral-common` and a Transformers build "
                "with `MistralCommonBackend`. Install `mistral-common>=1.8.6` and "
                "`transformers>=5.0.0rc0`."
            ) from exc

        tokenizer = MistralCommonBackend.from_pretrained(config["tokenizer_name"])
        tokenizer = patch_tokenizer_with_assistant_mask_support(tokenizer)
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            config["tokenizer_name"],
            trust_remote_code=True,
        )

    if tokenizer.pad_token is None:
        if "<|finetune_right_pad_id|>" in tokenizer.get_vocab():
            tokenizer.pad_token = "<|finetune_right_pad_id|>"
        elif tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "<|pad|>"})

    tokenizer.padding_side = "right"
    tokenizer.chat_template = config["chat_template"]

    test_msgs = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
    ]
    test_output = tokenizer.apply_chat_template(
        test_msgs,
        tokenize=True,
        add_generation_prompt=False,
        return_assistant_tokens_mask=True,
        return_dict=True,
    )
    assistant_mask_sum = sum(test_output["assistant_masks"])
    assert assistant_mask_sum > 0, "Template patching failed: assistant_masks are all zeros."
    print(f"Chat template verified. Assistant mask tokens: {assistant_mask_sum}")

    return tokenizer


def maybe_fix_special_token_embeddings(model, tokenizer, config):
    special_tokens_to_fix = config.get("special_tokens_to_fix", [])
    if not special_tokens_to_fix:
        return

    print("Fixing special token embeddings...")
    regular_token_count = config["regular_token_count"]
    input_embeddings = model.get_input_embeddings().weight
    output_embeddings = model.get_output_embeddings().weight

    with torch.no_grad():
        input_mean = input_embeddings[:regular_token_count].float().mean(dim=0)
        output_mean = output_embeddings[:regular_token_count].float().mean(dim=0)

        for token_str in special_tokens_to_fix:
            token_id = tokenizer.convert_tokens_to_ids(token_str)
            if token_id is None or token_id >= input_embeddings.shape[0]:
                continue
            if input_embeddings[token_id].abs().sum() == 0:
                input_embeddings[token_id] = input_mean.to(input_embeddings.dtype)
                output_embeddings[token_id] = output_mean.to(output_embeddings.dtype)
                print(f"  Fixed: {token_str} (id={token_id})")
            else:
                print(f"  OK:    {token_str} (id={token_id}) - already initialized")


def build_sft_config(output_dir, args):
    return SFTConfig(
        output_dir=output_dir,
        assistant_only_loss=True,
        num_train_epochs=1,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=1e-4,
        fp16=not args.bf16,
        bf16=args.bf16,
        save_strategy="steps",
        save_steps=10000,
        save_total_limit=6,
        eval_strategy="steps",
        eval_steps=500,
        logging_steps=50,
        report_to="none",
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        max_length=MAX_SEQ_LENGTH,
        dataloader_num_workers=args.dataloader_workers,
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False} if args.gradient_checkpointing else None,
        seed=SEED,
        packing=False,
    )


def load_model(config, args):
    requested_attn_impl = "sdpa" if args.no_flash_attn else args.attn_impl
    common_kwargs = {
        "pretrained_model_name_or_path": config["model_name"],
        "torch_dtype": torch.bfloat16 if args.bf16 else torch.float16,
        "device_map": "auto",
        "trust_remote_code": True,
    }

    if config.get("loader") == "mistral3":
        try:
            from transformers import Mistral3ForConditionalGeneration
        except ImportError as exc:
            raise RuntimeError(
                "Ministral-3 is not supported by your current Transformers install. "
                "The `KeyError: 'ministral3'` error means the auto config registry in your "
                "environment does not know that model type. Upgrade to `transformers>=5.0.0rc0` "
                "and install `mistral-common>=1.8.6`."
            ) from exc

        for attn_impl in (requested_attn_impl, "sdpa", None):
            try:
                kwargs = dict(common_kwargs)
                if attn_impl not in (None, "eager"):
                    kwargs["attn_implementation"] = attn_impl
                return Mistral3ForConditionalGeneration.from_pretrained(**kwargs)
            except Exception as exc:
                if attn_impl is None:
                    raise
                print(f"Attention backend '{attn_impl}' unavailable for {config['model_name']}: {exc}")
    else:
        for attn_impl in (requested_attn_impl, "sdpa", None):
            try:
                kwargs = dict(common_kwargs)
                if attn_impl not in (None, "eager"):
                    kwargs["attn_implementation"] = attn_impl
                return AutoModelForCausalLM.from_pretrained(**kwargs)
            except Exception as exc:
                if attn_impl is None:
                    raise
                print(f"Attention backend '{attn_impl}' unavailable for {config['model_name']}: {exc}")

    raise RuntimeError(f"Failed to load model: {config['model_name']}")


def train_one(model_key, train_dataset, eval_dataset, args):
    config = MODEL_CONFIGS[model_key]

    print("=" * 80)
    print(f"Starting SFT for {config['model_name']}")
    print("=" * 80)

    tokenizer = prepare_tokenizer(config)

    print("Loading model...")
    model = load_model(config, args)

    if len(tokenizer) > model.get_input_embeddings().weight.shape[0]:
        model.resize_token_embeddings(len(tokenizer))
        print(f"Resized embeddings to {len(tokenizer)}")

    maybe_fix_special_token_embeddings(model, tokenizer, config)

    model.enable_input_require_grads()
    print("Model ready.\n")

    trainer = SFTTrainer(
        model=model,
        args=build_sft_config(config["output_dir"], args),
        peft_config=lora_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )

    trainer.train()

    final_adapter_dir = f"{config['output_dir']}/final_adapter"
    trainer.save_model(final_adapter_dir)
    tokenizer.save_pretrained(final_adapter_dir)

    print(f"\nTraining complete! Adapter saved to: {final_adapter_dir}")
    print("This is your SFT baseline (reference model) for DPO.")
    print("Keep this checkpoint - you need it as ref_model in DPO.\n")


def main():
    args = parse_args()
    train_dataset, eval_dataset = load_openhermes_dataset()

    model_keys = list(MODEL_CONFIGS.keys()) if args.model == "all" else [args.model]
    for model_key in model_keys:
        train_one(model_key, train_dataset, eval_dataset, args)


if __name__ == "__main__":
    main()
