# ndna_lib/models.py
"""
Model loading and adapter layer for nDNA experiments.

Supports:
  - GPT2-like family (GPT-2, many 'gpt-*' causal LMs)
  - LLaMA-like family (LLaMA 2/3, etc.)

Everything else in the codebase should talk to LLMAdapter,
not directly to HuggingFace internals.
"""
import json
from abc import ABC, abstractmethod
from typing import Dict, List, Type

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Mxfp4Config
)

# -----------------------
# Device + basic perf knobs
# -----------------------

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

if DEVICE == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass


# -----------------------
# Base adapter interface
# -----------------------

class LLMAdapter(ABC):
    """
    Thin wrapper around a HF causal LM and its tokenizer.

    All geometry / training code should only depend on:
      - adapter.model
      - adapter.tokenizer
      - adapter.name
      - adapter.num_layers
      - adapter.get_block_params(layer_idx)
      - adapter.lens_logits(hidden_states)
    """

    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase, name: str):
        self.model = model
        self.tokenizer = tokenizer
        self.name = name

    @property
    @abstractmethod
    def num_layers(self) -> int:
        ...

    @abstractmethod
    def get_block_params(self, layer_idx: int) -> List[torch.nn.Parameter]:
        ...

    @abstractmethod
    def lens_logits(self, h: torch.Tensor) -> torch.Tensor:
        """
        Map hidden states at some layer to logits over vocabulary.

        Args:
            h: (B, S, H) tensor of hidden states.

        Returns:
            logits: (B, S, V)
        """
        ...


# -----------------------
# GPT-2-like adapter
# -----------------------

class GPT2LikeAdapter(LLMAdapter):
    """
    For GPT-2-style architectures where:
      - blocks live in model.transformer.h
      - final LN is model.transformer.ln_f
      - readout is model.lm_head
    """

    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase, name: str):
        super().__init__(model, tokenizer, name)
        trans = self.model.transformer
        self._blocks = trans.h
        self._final_ln = trans.ln_f
        self._lm_head = self.model.lm_head

    @property
    def num_layers(self) -> int:
        return len(self._blocks)

    def get_block_params(self, layer_idx: int) -> List[torch.nn.Parameter]:
        return list(self._blocks[layer_idx].parameters())

    def lens_logits(self, h: torch.Tensor) -> torch.Tensor:
        # h: (B, S, H) -> logits: (B, S, V)
        return self._lm_head(self._final_ln(h))


# -----------------------
# LLaMA-like adapter
# -----------------------

class LlamaAdapter(LLMAdapter):
    """
    For LLaMA-style architectures (LLaMA 2/3, etc.) where:
      - blocks live in model.model.layers
      - final norm is model.model.norm
      - readout is model.lm_head
    """

    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase, name: str):
        super().__init__(model, tokenizer, name)
        core = self.model.model
        self._blocks = core.layers
        self._final_norm = core.norm
        self._lm_head = self.model.lm_head

    @property
    def num_layers(self) -> int:
        return len(self._blocks)

    def get_block_params(self, layer_idx: int) -> List[torch.nn.Parameter]:
        return list(self._blocks[layer_idx].parameters())

    def lens_logits(self, h: torch.Tensor) -> torch.Tensor:
        # h: (B, S, H) -> logits: (B, S, V)
        return self._lm_head(self._final_norm(h))


# -----------------------
# Model registry
# -----------------------

MODEL_REGISTRY: Dict[str, Dict] = {
    # GPT-family example
    "gpt2": {
        "hf_name": "gpt2",
        "adapter_cls": GPT2LikeAdapter,
        "dtype": "float16",
    },
    # You can add more GPT-style models here:
    # "gpt2_medium": {
    #     "hf_name": "gpt2-medium",
    #     "adapter_cls": GPT2LikeAdapter,
    #     "dtype": "float16",
    # },

    # LLaMA-family examples – update hf_name to match what you actually use
    "llama_3_8b": {
        "hf_name": "meta-llama/Meta-Llama-3-8B-Instruct",      # adjust if needed
        "adapter_cls": LlamaAdapter,
        "dtype": "bfloat16",
    },
    "llama_3_1_8b": {
        "hf_name": "meta-llama/Meta-Llama-3.1-8B-Instruct",    # adjust if needed
        "adapter_cls": LlamaAdapter,
        "dtype": "bfloat16",
    },
}


def _ensure_padding(tokenizer: PreTrainedTokenizerBase):
    """
    Make sure the tokenizer has a pad_token defined.
    For causal LMs it's usually fine to use eos_token as padding.
    """
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        elif tokenizer.bos_token is not None:
            tokenizer.pad_token = tokenizer.bos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "<|pad|>"})


def load_model_from_registry(model_key: str) -> LLMAdapter:
    """
    Given a short key like "gpt2" or "llama_3_8b":
      - look up HF checkpoint name and adapter class,
      - load tokenizer and model,
      - ensure padding token exists,
      - resize embeddings if needed,
      - disable use_cache,
      - move model to DEVICE,
      - wrap everything in an adapter.
    """
    if model_key not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model_key={model_key}. "
            f"Available keys: {list(MODEL_REGISTRY.keys())}"
        )

    cfg = MODEL_REGISTRY[model_key]
    hf_name: str = cfg["hf_name"]
    AdapterCls: Type[LLMAdapter] = cfg["adapter_cls"]

    tokenizer = AutoTokenizer.from_pretrained(hf_name, use_fast=True)
    _ensure_padding(tokenizer)

    quantization_config = Mxfp4Config(dequantize=True)
    model_kwargs = dict(
        attn_implementation="eager",
        torch_dtype=torch.bfloat16,
        quantization_config=quantization_config,
        use_cache=False,
        device_map="auto",
    )

    model = AutoModelForCausalLM.from_pretrained(hf_name, **model_kwargs)
    # model.eval()
    model.config.pad_token_id = tokenizer.pad_token_id

    # Match vocab size to tokenizer
    if model.get_input_embeddings().num_embeddings != len(tokenizer):
        model.resize_token_embeddings(len(tokenizer))

    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    model.to(DEVICE)

    adapter = AdapterCls(model=model, tokenizer=tokenizer, name=model_key)
    return adapter


# -----------------------
# Quick sanity checks
# -----------------------

def _quick_sanity(model_key: str, text: str = "Hello world", max_length: int = 8):
    """
    Small smoke test:
      - load adapter
      - tokenize a short text
      - run a forward pass with output_hidden_states
      - apply lens_logits on the last hidden state
      - print tensor shapes
    """
    print(f"\n[Sanity] Loading model '{model_key}' on {DEVICE}...")
    adapter = load_model_from_registry(model_key)
    model = adapter.model
    tokenizer = adapter.tokenizer

    print(f"  name        : {adapter.name}")
    print(f"  num_layers  : {adapter.num_layers}")
    print(f"  vocab_size  : {model.get_input_embeddings().num_embeddings}")

    enc = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    ).to(model.device)

    with torch.no_grad():
        out = model(**enc, output_hidden_states=True, use_cache=False)
        hidden_states = out.hidden_states       # [emb, layer1, ..., layerL]
        h_last = hidden_states[-1]              # (B, S, H)
        logits = adapter.lens_logits(h_last)    # (B, S, V)

    print(f"  input_ids.shape : {enc['input_ids'].shape}")
    print(f"  h_last.shape    : {h_last.shape}")
    print(f"  logits.shape    : {logits.shape}")
    print("  Sanity check OK.\n")


# -----------------------
# Model zoo loader
# -----------------------


def load_model_zoo(path: str = "model_zoo.json") -> Dict[str, Dict]:
    """
    Load the external model_zoo.json file.

    Expected format:
        {
          "model_key": {
            "hf_id": "...",
            "family": "...",
            "enabled": true/false,
            ...
          },
          ...
        }
    """
    with open(path, "r") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"model_zoo at {path!r} must be a JSON object.")
    return data


if __name__ == "__main__":
    # Comment out ones you don't have access to.
    _quick_sanity("gpt2")
    # _quick_sanity("llama_3_8b")
    # _quick_sanity("llama_3_1_8b")

