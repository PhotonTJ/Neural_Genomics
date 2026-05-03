import os
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, TaskType
from trl import SFTTrainer, SFTConfig

# =============================================================================
# CONFIG
# =============================================================================
MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B"
DATASET_NAME = "teknium/OpenHermes-2.5"

OUTPUT_DIR = "./llama3.1-openhermes-lora"
CACHE_DIR = "./dataset_cache"

NUM_SAMPLES = 100_000
MAX_SEQ_LENGTH = 2048
SEED = 42

# =============================================================================
# LoRA CONFIG
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
# TOKENIZER (NO MANUAL TEMPLATE PATCHING NEEDED)
# =============================================================================
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({"pad_token": "<|pad|>"})

tokenizer.padding_side = "right"
# Your base tokenizer
#tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B")

# Grab the template from the Instruct version
#instruct_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")
#tokenizer.chat_template = instruct_tokenizer.chat_template
llama3_template_with_masking = (
    "{% set loop_messages = messages %}"
    "{% for message in loop_messages %}"
    "{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' %}"
    "{% if loop.index0 == 0 %}"
    "{% set content = bos_token + content %}"
    "{% endif %}"
    "{% if message['role'] == 'assistant' %}"
    "{% generation %}{{ content }}{% endgeneration %}"  # <-- HF tags added here!
    "{% else %}"
    "{{ content }}"
    "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}"
    "{% endif %}"
)

tokenizer.chat_template = llama3_template_with_masking
# =============================================================================
# MODEL
# =============================================================================
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto",
)

# Resize embeddings if needed
if len(tokenizer) > model.get_input_embeddings().weight.shape[0]:
    model.resize_token_embeddings(len(tokenizer))

# =============================================================================
# FIX SPECIAL TOKEN EMBEDDINGS (CRITICAL FOR LLAMA 3.x)
# =============================================================================
SPECIAL_TOKENS = tokenizer.all_special_tokens

with torch.no_grad():
    embed_weights = model.model.embed_tokens.weight
    lm_head_weights = model.lm_head.weight

    mean_embed = embed_weights[:-len(SPECIAL_TOKENS)].mean(dim=0)
    mean_lm = lm_head_weights[:-len(SPECIAL_TOKENS)].mean(dim=0)

    for token in SPECIAL_TOKENS:
        token_id = tokenizer.convert_tokens_to_ids(token)
        if token_id is None:
            continue

        if embed_weights[token_id].abs().sum() == 0:
            embed_weights[token_id] = mean_embed
            lm_head_weights[token_id] = mean_lm
            print(f"Fixed token: {token}")

model.enable_input_require_grads()

# =============================================================================
# DATASET
# =============================================================================
processed_cache_path = os.path.join(
    CACHE_DIR, f"openhermes_{NUM_SAMPLES}"
)

if os.path.exists(processed_cache_path):
    from datasets import DatasetDict
    dataset = DatasetDict.load_from_disk(processed_cache_path)
    train_dataset = dataset["train"]
    eval_dataset = dataset["test"]

else:
    dataset = load_dataset(DATASET_NAME, split="train")
    dataset = dataset.shuffle(seed=SEED).select(range(NUM_SAMPLES))

    def format_data(example):
        role_map = {
            "human": "user",
            "gpt": "assistant",
            "system": "system"
        }

        messages = []
        for msg in example["conversations"]:
            role = role_map.get(msg["from"], "user")
            content = msg["value"]
            if content:
                messages.append({
                    "role": role,
                    "content": content
                })

        return {"messages": messages}

    dataset = dataset.map(
        format_data,
        remove_columns=dataset.column_names
    )

    split = dataset.train_test_split(test_size=0.05, seed=SEED)
    train_dataset = split["train"]
    eval_dataset = split["test"]

    os.makedirs(CACHE_DIR, exist_ok=True)
    split.save_to_disk(processed_cache_path)

# =============================================================================
# TRAIN CONFIG
# =============================================================================
sft_config = SFTConfig(
    output_dir=OUTPUT_DIR,

    assistant_only_loss=True,

    num_train_epochs=1,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,

    learning_rate=1e-4,

    fp16=True,

    logging_steps=50,
    save_steps=50000,
    eval_steps=10000,

    max_length=MAX_SEQ_LENGTH,

    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},

    lr_scheduler_type="cosine",
    warmup_ratio=0.05,

    report_to="none",
)

# =============================================================================
# TRAINER
# =============================================================================
trainer = SFTTrainer(
    model=model,
    args=sft_config,
    peft_config=lora_config,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    processing_class=tokenizer,
)

# =============================================================================
# TRAIN
# =============================================================================
trainer.train()

# =============================================================================
# SAVE
# =============================================================================
trainer.save_model(f"{OUTPUT_DIR}/final_adapter")
tokenizer.save_pretrained(f"{OUTPUT_DIR}/final_adapter")

print("Done. Adapter saved.")
