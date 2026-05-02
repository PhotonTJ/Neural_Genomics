"""
SFT Training Script: Llama 3 8B + OpenHermes 2.5 (LoRA)  [v4]
Purpose: Create an instruction-following (but unaligned) base for DPO safety research.
Requires: trl==0.29.0, transformers, peft, datasets, torch

Fixes in v4:
- Initialize special token embeddings to mean of existing embeddings (fixes CUDA assert)
- Dataset caching to avoid re-downloading/re-processing
- Patched chat template with {% generation %} markers
"""

import os
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, TaskType
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# =============================================================================
# Config
# =============================================================================
MODEL_NAME = "meta-llama/Meta-Llama-3-8B"
TOKENIZER_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"
DATASET_NAME = "teknium/OpenHermes-2.5"
OUTPUT_DIR = "./llama3-openhermes-lora"
CACHE_DIR = "./dataset_cache"            # local cache for processed dataset
NUM_SAMPLES = 100_000
MAX_SEQ_LENGTH = 2048
SEED = 42

# =============================================================================
# LoRA config
# =============================================================================
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
)

# =============================================================================
# Tokenizer
# =============================================================================
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME, trust_remote_code=True)

# Use <|finetune_right_pad_id|> if available, otherwise add a new token
if "<|finetune_right_pad_id|>" in tokenizer.get_vocab():
    tokenizer.pad_token = "<|finetune_right_pad_id|>"
else:
    tokenizer.add_special_tokens({"pad_token": "<|pad|>"})

tokenizer.padding_side = "right"

# =============================================================================
# Patch chat template with {% generation %} markers for TRL 0.29
# =============================================================================
LLAMA3_CHAT_TEMPLATE_WITH_GENERATION = (
    "{% set loop_messages = messages %}"
    "{% for message in loop_messages %}"
        "{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n' + message['content'] | trim + '<|eot_id|>' %}"
        "{% if message['role'] == 'assistant' %}"
            "{% generation %}"
            "{{ content }}"
            "{% endgeneration %}"
        "{% else %}"
            "{{ content }}"
        "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
        "{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}"
    "{% endif %}"
)

tokenizer.chat_template = LLAMA3_CHAT_TEMPLATE_WITH_GENERATION

# Verify template produces correct assistant masks
test_msgs = [
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hi there!"},
]
test_output = tokenizer.apply_chat_template(
    test_msgs, tokenize=True, add_generation_prompt=False,
    return_assistant_tokens_mask=True, return_dict=True,
)
assert sum(test_output["assistant_masks"]) > 0, \
    "Template patching failed — assistant_masks are all zeros!"
print(f"Chat template verified. Assistant mask tokens: {sum(test_output['assistant_masks'])}")

# =============================================================================
# Load model manually so we can fix special token embeddings
# =============================================================================
print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
)

# Fix: Initialize special token embeddings that are zeros in the base model.
# The Llama 3 base model has zero-initialized embeddings for special tokens
# like <|eot_id|>, <|start_header_id|>, etc. These cause NaN gradients
# when they appear in training data. We initialize them to the mean of
# all regular token embeddings.
SPECIAL_TOKENS_TO_FIX = [
    "<|eot_id|>",
    "<|start_header_id|>",
    "<|end_header_id|>",
    "<|finetune_right_pad_id|>",
]

print("Fixing special token embeddings...")
with torch.no_grad():
    embed_weights = model.model.embed_tokens.weight
    lm_head_weights = model.lm_head.weight

    # Compute mean of the first 128000 regular token embeddings
    regular_token_count = 128000
    embed_mean = embed_weights[:regular_token_count].float().mean(dim=0)
    lm_head_mean = lm_head_weights[:regular_token_count].float().mean(dim=0)

    for token_str in SPECIAL_TOKENS_TO_FIX:
        token_id = tokenizer.convert_tokens_to_ids(token_str)
        if token_id is not None and token_id < embed_weights.shape[0]:
            # Only fix if the embedding is all zeros
            if embed_weights[token_id].abs().sum() == 0:
                embed_weights[token_id] = embed_mean.to(embed_weights.dtype)
                lm_head_weights[token_id] = lm_head_mean.to(lm_head_weights.dtype)
                print(f"  Fixed: {token_str} (id={token_id})")
            else:
                print(f"  OK:    {token_str} (id={token_id}) — already initialized")

# Resize embeddings if we added a new pad token
if len(tokenizer) > model.get_input_embeddings().weight.shape[0]:
    model.resize_token_embeddings(len(tokenizer))
    print(f"Resized embeddings to {len(tokenizer)}")

model.enable_input_require_grads()
print("Model ready.\n")

# =============================================================================
# Dataset — with caching
# =============================================================================
processed_cache_path = os.path.join(CACHE_DIR, f"openhermes_processed_{NUM_SAMPLES}")

if os.path.exists(processed_cache_path):
    print(f"Loading processed dataset from cache: {processed_cache_path}")
    from datasets import DatasetDict
    cached = DatasetDict.load_from_disk(processed_cache_path)
    train_dataset = cached["train"]
    eval_dataset = cached["test"]
else:
    print("Processing dataset (will be cached for next time)...")
    dataset = load_dataset(DATASET_NAME, split="train")
    dataset = dataset.shuffle(seed=SEED).select(range(NUM_SAMPLES))

    def format_to_conversational(example):
        """
        Convert OpenHermes 2.5 format to TRL conversational format.
        """
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

    # Train/eval split
    split_dataset = dataset.train_test_split(test_size=0.05, seed=SEED)
    train_dataset = split_dataset["train"]
    eval_dataset = split_dataset["test"]

    # Save to cache
    os.makedirs(CACHE_DIR, exist_ok=True)
    split_dataset.save_to_disk(processed_cache_path)
    print(f"Dataset cached to: {processed_cache_path}")

print(f"Train samples: {len(train_dataset)}")
print(f"Eval samples:  {len(eval_dataset)}")
print(f"\n--- Example conversation ---")
for msg in train_dataset[0]["messages"][:2]:
    print(f"  [{msg['role']}]: {msg['content'][:100]}...")
print()

# =============================================================================
# SFTConfig
# =============================================================================
sft_config = SFTConfig(
    output_dir=OUTPUT_DIR,

    # --- Loss masking ---
    assistant_only_loss=True,

    # --- Training schedule ---
    num_train_epochs=1,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=1e-4,

    # --- Precision ---
    fp16=True,
    bf16=False,                            # set True (and fp16=False) on Ampere+ GPUs

    # --- Saving & eval ---
    save_strategy="steps",
    save_steps=10000,
    save_total_limit=6,
    eval_strategy="steps",
    eval_steps=500,

    # --- Logging ---
    logging_steps=50,
    report_to="none",

    # --- Scheduler ---
    warmup_ratio=0.05,
    lr_scheduler_type="cosine",

    # --- Sequence length ---
    max_length=MAX_SEQ_LENGTH,

    # --- Performance ---
    dataloader_num_workers=4,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},

    # --- Misc ---
    seed=SEED,
    packing=False,
)

# =============================================================================
# Trainer — pass the loaded model directly (since we modified embeddings)
# =============================================================================
trainer = SFTTrainer(
    model=model,                           # pass loaded model (not string) since we patched it
    args=sft_config,
    peft_config=lora_config,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    processing_class=tokenizer,
)

# =============================================================================
# Train
# =============================================================================
trainer.train()

# =============================================================================
# Save
# =============================================================================
trainer.save_model(f"{OUTPUT_DIR}/final_adapter")
tokenizer.save_pretrained(f"{OUTPUT_DIR}/final_adapter")

print(f"\nTraining complete! Adapter saved to: {OUTPUT_DIR}/final_adapter")
print("This is your SFT baseline (reference model) for DPO.")
print("Keep this checkpoint — you need it as ref_model in DPO.")