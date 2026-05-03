"""
DPO Training Script for Ministral-3

Pipeline:
Base -> load tokenizer from adapter -> resize -> load SFT -> merge -> DPO

This mirrors llama3p1_DPO.py while switching the model stack to:
- mistralai/Ministral-3-8B-Base-2512

It also keeps NaN handling enabled through the trainer config.
"""

import logging
import os
import sys

import torch
from datasets import load_dataset
from huggingface_hub import login
from peft import LoraConfig, PeftModel, TaskType
from transformers import AutoTokenizer
from trl import DPOConfig, DPOTrainer


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


BASE_MODEL = "mistralai/Ministral-3-8B-Base-2512"
SFT_ADAPTER = "./ministral3-openhermes-lora/final_adapter"

DATASET = "Anthropic/hh-rlhf"
DATA_DIR = "harmless-base"

OUTPUT_DIR = "./ministral3_dpo_output"
NAN_SAVE_DIR = f"{OUTPUT_DIR}/nan_abort_checkpoint"

LR = 1e-4
BETA = 0.1
BATCH = 4
GRAD_ACC = 4
EPOCHS = 1
MAX_LEN = 2048

LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


class NaNSafeDPOTrainer(DPOTrainer):
    def __init__(self, *args, tokenizer_for_abort_save=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._tokenizer_for_abort_save = tokenizer_for_abort_save
        self._nan_abort_done = False

    def _save_on_nan_and_abort(self, reason):
        if self._nan_abort_done:
            raise RuntimeError(reason)

        self._nan_abort_done = True
        logger.error("Non-finite training state detected: %s", reason)
        logger.error("Saving emergency checkpoint to %s", NAN_SAVE_DIR)

        os.makedirs(NAN_SAVE_DIR, exist_ok=True)
        self.save_model(NAN_SAVE_DIR)
        if self._tokenizer_for_abort_save is not None:
            self._tokenizer_for_abort_save.save_pretrained(NAN_SAVE_DIR)

        raise RuntimeError(f"Aborting after emergency save: {reason}")

    def training_step(self, *args, **kwargs):
        loss = super().training_step(*args, **kwargs)

        if isinstance(loss, torch.Tensor) and not torch.isfinite(loss).all():
            self._save_on_nan_and_abort(f"loss={loss.detach().float().cpu().item()}")

        for name, param in self.model.named_parameters():
            if param.grad is None:
                continue
            if not torch.isfinite(param.grad).all():
                self._save_on_nan_and_abort(f"non-finite gradient in parameter '{name}'")

        return loss

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


def load_tokenizer():
    logger.info("Loading tokenizer from adapter...")

    try:
        tokenizer = AutoTokenizer.from_pretrained(SFT_ADAPTER, trust_remote_code=True)
    except Exception:
        from transformers import MistralCommonBackend

        tokenizer = MistralCommonBackend.from_pretrained(SFT_ADAPTER)

    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "<pad>"})

    tokenizer.padding_side = "right"
    tokenizer.chat_template = MISTRAL_CHAT_TEMPLATE_WITH_GENERATION

    logger.info("Tokenizer vocab size: %s", len(tokenizer))
    return tokenizer


def load_model():
    logger.info("Loading base model...")

    try:
        from transformers import Mistral3ForConditionalGeneration

        model_cls = Mistral3ForConditionalGeneration
    except ImportError as exc:
        raise RuntimeError(
            "Ministral-3 requires a Transformers build with Mistral3ForConditionalGeneration."
        ) from exc

    for attn_impl in ("flash_attention_2", "sdpa", None):
        try:
            kwargs = {
                "pretrained_model_name_or_path": BASE_MODEL,
                "torch_dtype": torch.bfloat16,
                "device_map": "auto",
                "trust_remote_code": True,
            }
            if attn_impl is not None:
                kwargs["attn_implementation"] = attn_impl
            model = model_cls.from_pretrained(**kwargs)
            logger.info("Loaded model with attention backend: %s", attn_impl or "default")
            return model
        except Exception as exc:
            if attn_impl is None:
                raise
            logger.warning("Attention backend '%s' unavailable: %s", attn_impl, exc)

    raise RuntimeError(f"Failed to load model: {BASE_MODEL}")


def parse_hh(text):
    turns = []
    parts = text.strip().split("\n\nHuman: ")
    for part in parts:
        if not part:
            continue
        sub = part.split("\n\nAssistant: ")
        for idx, chunk in enumerate(sub):
            chunk = chunk.strip()
            if not chunk:
                continue
            if idx == 0:
                turns.append(("user", chunk))
            else:
                turns.append(("assistant", chunk))
    return turns


def shared_prefix(chosen_turns, rejected_turns):
    prefix = []
    for chosen_turn, rejected_turn in zip(chosen_turns, rejected_turns):
        if chosen_turn != rejected_turn:
            break
        prefix.append(chosen_turn)
    return prefix


def trim_to_last_user(turns):
    trimmed = list(turns)
    while trimmed and trimmed[-1][0] != "user":
        trimmed.pop()
    return trimmed


def render_chat(tokenizer, messages, add_generation_prompt=False):
    extra_kwargs = {}
    if messages and messages[-1]["role"] == "assistant":
        extra_kwargs["continue_final_message"] = True
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
        **extra_kwargs,
    )


def main():
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        print("ERROR: set HF_TOKEN")
        sys.exit(1)

    login(token=hf_token)

    tokenizer = load_tokenizer()
    model = load_model()

    base_vocab = model.get_input_embeddings().weight.shape[0]
    logger.info("Base vocab size: %s", base_vocab)

    if len(tokenizer) > base_vocab:
        logger.info("Resizing embeddings: %s -> %s", base_vocab, len(tokenizer))
        model.resize_token_embeddings(len(tokenizer))

    logger.info("Loading SFT adapter...")
    model = PeftModel.from_pretrained(model, SFT_ADAPTER)

    logger.info("Merging SFT adapter...")
    model = model.merge_and_unload()
    logger.info("Model ready for DPO.")

    raw = load_dataset(DATASET, data_dir=DATA_DIR, split="train")

    def format_dpo(example):
        chosen = parse_hh(example["chosen"])
        rejected = parse_hh(example["rejected"])

        if len(chosen) < 2 or chosen[-1][0] != "assistant":
            return {"prompt": "", "chosen": "", "rejected": ""}
        if len(rejected) < 1 or rejected[-1][0] != "assistant":
            return {"prompt": "", "chosen": "", "rejected": ""}

        prompt_turns = shared_prefix(chosen[:-1], rejected[:-1])
        prompt_turns = trim_to_last_user(prompt_turns)
        if not prompt_turns:
            return {"prompt": "", "chosen": "", "rejected": ""}

        prompt_msgs = [{"role": role, "content": content} for role, content in prompt_turns]
        chosen_msgs = prompt_msgs + [{"role": "assistant", "content": chosen[-1][1]}]
        rejected_msgs = prompt_msgs + [{"role": "assistant", "content": rejected[-1][1]}]

        prompt = render_chat(tokenizer, prompt_msgs, add_generation_prompt=False)
        chosen_full = render_chat(tokenizer, chosen_msgs, add_generation_prompt=False)
        rejected_full = render_chat(tokenizer, rejected_msgs, add_generation_prompt=False)

        if not chosen_full.startswith(prompt) or not rejected_full.startswith(prompt):
            return {"prompt": "", "chosen": "", "rejected": ""}

        chosen_suffix = chosen_full[len(prompt):]
        rejected_suffix = rejected_full[len(prompt):]
        if not chosen_suffix or not rejected_suffix:
            return {"prompt": "", "chosen": "", "rejected": ""}

        return {
            "prompt": prompt,
            "chosen": chosen_suffix,
            "rejected": rejected_suffix,
        }

    dataset = raw.map(format_dpo, remove_columns=raw.column_names)
    dataset = dataset.filter(lambda x: x["prompt"] and x["chosen"] and x["rejected"])
    if len(dataset) == 0:
        raise RuntimeError(
            "DPO dataset is empty after formatting/filtering. "
            "For Ministral this usually means the prompt render is not a prefix of the full rendered assistant turns."
        )
    dataset = dataset.shuffle(seed=42)

    split = dataset.train_test_split(test_size=0.05, seed=42)
    train_ds = split["train"]
    eval_ds = split["test"]

    logger.info("Train: %s | Eval: %s", len(train_ds), len(eval_ds))

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
        logging_nan_inf_filter=True,
    )

    trainer = NaNSafeDPOTrainer(
        model=model,
        ref_model=None,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        peft_config=peft_config,
        tokenizer_for_abort_save=tokenizer,
    )

    trainer.model.print_trainable_parameters()
    trainer.train()

    final_adapter_dir = f"{OUTPUT_DIR}/final_dpo_adapter"
    trainer.save_model(final_adapter_dir)
    tokenizer.save_pretrained(final_adapter_dir)

    logger.info("DPO training complete.")


if __name__ == "__main__":
    main()
