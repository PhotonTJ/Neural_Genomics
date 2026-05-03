  GNU nano 7.2                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         llama3p1_DPO.py
"""
DPO Training Script (NaN-safe version)

Fixes:
- Stops instantly on NaN loss/grad
- Saves adapter before exit
- Uses correct HF Trainer API
- Uses accelerator.backward (AMP safe)
- Adds gradient clipping + safer LR
"""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
import os
import sys
import torch
import logging
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, TaskType, PeftModel
from trl import DPOTrainer, DPOConfig
from huggingface_hub import login

# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------
hf_token = os.environ.get("HF_TOKEN")
if not hf_token:
    print("ERROR: set HF_TOKEN")
    sys.exit(1)

login(token=hf_token)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
BASE_MODEL = "meta-llama/Meta-Llama-3.1-8B"
SFT_ADAPTER = "./SFT/final_adapter_llama3p1_SFT"

DATASET = "Anthropic/hh-rlhf"
DATA_DIR = "harmless-base"

OUTPUT_DIR = "./dpo_output"

LR = 1e-4              # ↓ critical fix
BETA = 0.1
BATCH = 4
GRAD_ACC = 4
EPOCHS = 1
MAX_LEN = 2048

# LoRA
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
TARGET_MODULES = [
    "q_proj","k_proj","v_proj","o_proj",
    "gate_proj","up_proj","down_proj"
]

torch.backends.cuda.matmul.allow_tf32 = True

# ---------------------------------------------------------------------------
# TOKENIZER
# ---------------------------------------------------------------------------
logger.info("Loading tokenizer from adapter...")
tokenizer = AutoTokenizer.from_pretrained(SFT_ADAPTER)

if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({"pad_token": "<|pad|>"})

tokenizer.padding_side = "right"
logger.info(f"Tokenizer vocab size: {len(tokenizer)}")

# ---------------------------------------------------------------------------
# CHAT TEMPLATE
# ---------------------------------------------------------------------------
llama3_template_with_masking = (
    "{% for message in messages %}"
    "{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\\n\\n'+ message['content'] | trim + '<|eot_id|>' %}"
    "{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}"
    "{% if message['role'] == 'assistant' %}{% generation %}{{ content }}{% endgeneration %}{% else %}{{ content }}{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\\n\\n' }}{% endif %}"
)

tokenizer.chat_template = llama3_template_with_masking

# ---------------------------------------------------------------------------
# MODEL
# ---------------------------------------------------------------------------
logger.info("Loading base model...")

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

base_vocab = model.get_input_embeddings().weight.shape[0]

if len(tokenizer) > base_vocab:
    logger.info(f"Resizing embeddings: {base_vocab} → {len(tokenizer)}")
    model.resize_token_embeddings(len(tokenizer))

logger.info("Loading SFT adapter...")
model = PeftModel.from_pretrained(model, SFT_ADAPTER)

logger.info("Merging SFT adapter...")
model = model.merge_and_unload()

model.config.use_cache = False

# ---------------------------------------------------------------------------
# DATASET
# ---------------------------------------------------------------------------
raw = load_dataset(DATASET, data_dir=DATA_DIR, split="train")

def parse_hh(text):
    turns = []
    parts = text.strip().split("\n\nHuman: ")
    for part in parts:
        if not part:
            continue
        sub = part.split("\n\nAssistant: ")
        for i, s in enumerate(sub):
            s = s.strip()
            if not s:
                continue
            turns.append(("user" if i == 0 else "assistant", s))
    return turns

def format_dpo(example):
    chosen = parse_hh(example["chosen"])
    rejected = parse_hh(example["rejected"])

    if len(chosen) < 2 or chosen[-1][0] != "assistant":
        return {"prompt":"","chosen":"","rejected":""}

    prompt_msgs = [{"role": r, "content": c} for r, c in chosen[:-1]]

    prompt = tokenizer.apply_chat_template(
        prompt_msgs,
        tokenize=False,
        add_generation_prompt=True,
    )

    return {
        "prompt": prompt,
        "chosen": chosen[-1][1],
        "rejected": rejected[-1][1] if rejected[-1][0]=="assistant" else "",
    }

dataset = raw.map(format_dpo, remove_columns=raw.column_names)
dataset = dataset.filter(lambda x: x["prompt"] and x["chosen"] and x["rejected"])
dataset = dataset.shuffle(seed=42)

split = dataset.train_test_split(test_size=0.05)
train_ds = split["train"]
eval_ds = split["test"]

# ---------------------------------------------------------------------------
# LoRA + DPO CONFIG
# ---------------------------------------------------------------------------
peft_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
    target_modules=TARGET_MODULES,
)

args = DPOConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH,
    per_device_eval_batch_size=BATCH,
    gradient_accumulation_steps=GRAD_ACC,
    beta=BETA,
    max_length=MAX_LEN,
    learning_rate=LR,
    bf16=True,
    logging_steps=25,
    save_steps=500,
    eval_steps=500,
    gradient_checkpointing=True,
    remove_unused_columns=False,
    report_to="tensorboard",
    max_grad_norm=1.0,
    logging_nan_inf_filter=False,
)

# ---------------------------------------------------------------------------
# SAFE TRAINER
# ---------------------------------------------------------------------------
class SafeDPOTrainer(DPOTrainer):
    def __init__(self, *args, save_path=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.save_path = save_path

    def training_step(self, model, inputs, num_items_in_batch=None):
        model.train()

        loss = self.compute_loss(model, inputs)

        # ---- NaN loss ----
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"[STOP] NaN/Inf loss at step {self.state.global_step}")
            self._save_and_exit()

        self.accelerator.backward(loss)

        # ---- NaN grads ----
        for name, p in model.named_parameters():
            if p.grad is not None:
                if torch.isnan(p.grad).any() or torch.isinf(p.grad).any():
                    print(f"[STOP] NaN/Inf grad at step {self.state.global_step} ({name})")
                    self._save_and_exit()

        return loss.detach()

    def _save_and_exit(self):
        print("Saving adapter before stopping...")

        try:
            self.model.save_pretrained(self.save_path)
        except:
            self.accelerator.unwrap_model(self.model).save_pretrained(self.save_path)

        raise RuntimeError("Training stopped due to NaNs")

# ---------------------------------------------------------------------------
# TRAIN
# ---------------------------------------------------------------------------
trainer = SafeDPOTrainer(
    model=model,
    ref_model=None,
    args=args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    processing_class=tokenizer,
    peft_config=peft_config,
    save_path=f"{OUTPUT_DIR}/nan_checkpoint"
)

trainer.train()

# ---------------------------------------------------------------------------
# FINAL SAVE
# ---------------------------------------------------------------------------
trainer.save_model(f"{OUTPUT_DIR}/final_dpo_adapter")
tokenizer.save_pretrained(f"{OUTPUT_DIR}/final_dpo_adapter")

logger.info("DPO training complete.")







