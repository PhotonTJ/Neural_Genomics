"""Model loading and the interventions the lifecycle experiments need.

Interventions are deliberately implemented so that the *only* thing that changes between
a base run and a modified run is the intervention itself: same tokenizer, same probe,
same triad settings.
"""
from __future__ import annotations

import gc
import numpy as np

from ._optional import load_torch

torch, TORCH_AVAILABLE = load_torch()


def load(model_id, device="cuda", dtype="bfloat16", cache_dir=None):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    kw = dict(cache_dir=cache_dir)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=getattr(torch, dtype), device_map=device, **kw)
    except TypeError:                       # older transformers spell it torch_dtype
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=getattr(torch, dtype), device_map=device, **kw)
    model.eval()
    return model, tok


def free(*objs):
    for o in objs:
        del o
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# --------------------------------------------------------------------- quantization
def quantize(model_id, dose, device="cuda", cache_dir=None):
    """Round-to-nearest weight-only quantization of the linear layers.

    RTN is not state of the art -- GPTQ/AWQ/AQLM are better at 3 and 2 bits -- but it is
    method-transparent and dependency-free, which matters more here: the dose-response
    claim is about monotonicity of the geometry in compression strength, not about
    getting the best possible quantizer. Whichever method you use, name it in the caption.
    """
    dtype = "float16" if dose == "fp16" else "bfloat16"
    model, tok = load(model_id, device=device, dtype=dtype, cache_dir=cache_dir)
    if dose == "fp16":
        return model, tok

    bits = {"int8": 8, "4bit": 4, "3bit": 3, "2bit": 2}[dose]
    qmax = 2 ** (bits - 1) - 1
    with torch.no_grad():
        for name, mod in model.named_modules():
            if not isinstance(mod, torch.nn.Linear):
                continue
            if "lm_head" in name:               # leave the readout intact
                continue
            W = mod.weight.data.to(torch.float32)
            scale = W.abs().amax(dim=1, keepdim=True) / qmax
            scale = torch.clamp(scale, min=1e-8)
            mod.weight.data = (torch.round(W / scale).clamp(-qmax - 1, qmax)
                               * scale).to(mod.weight.dtype)
    return model, tok


# --------------------------------------------------------------------- pruning
def prune(model_id, sparsity, device="cuda", structured=True, cache_dir=None):
    """Magnitude pruning. structured=True removes whole output channels per layer."""
    model, tok = load(model_id, device=device, cache_dir=cache_dir)
    if sparsity <= 0:
        return model, tok
    with torch.no_grad():
        for name, mod in model.named_modules():
            if not isinstance(mod, torch.nn.Linear) or "lm_head" in name:
                continue
            W = mod.weight.data
            if structured:
                score = W.abs().mean(dim=1)             # per output channel
                k = int(sparsity * score.numel())
                if k > 0:
                    idx = torch.topk(score, k, largest=False).indices
                    W[idx, :] = 0
            else:
                flat = W.abs().flatten()
                k = int(sparsity * flat.numel())
                if k > 0:
                    thr = torch.kthvalue(flat, k).values
                    W[W.abs() <= thr] = 0
    return model, tok


# --------------------------------------------------------------------- gauge test
def rescale_residual_gains(model, c):
    """Multiply every RMS/LayerNorm gain by c.

    With pre-normalisation and RMSNorm this is *not* a no-op on its own; it is the half of
    the gain trade that we can apply without touching every residual write. Experiment 05
    checks the logits before and after and reports the max deviation, so a model whose
    architecture makes this a genuine function change is detected rather than assumed.
    """
    n = 0
    with torch.no_grad():
        for name, mod in model.named_modules():
            if hasattr(mod, "weight") and mod.weight is not None \
               and mod.weight.ndim == 1 and ("norm" in name.lower() or "ln" in name.lower()):
                mod.weight.data = mod.weight.data * c
                n += 1
    return n


@torch.no_grad()
def logits_fingerprint(model, tok, texts, device="cuda", n=8):
    """Deterministic logits on a fixed batch, for verifying a function-preserving edit."""
    enc = tok(texts[:n], return_tensors="pt", padding=True, truncation=True, max_length=128)
    enc = {k: v.to(device) for k, v in enc.items()}
    out = model(**enc).logits.to(torch.float32)
    return out.cpu().numpy()


# --------------------------------------------------------------------- lora / merge
def lora_finetune(model_id, texts, out_dir, steps=200, rank=16, lr=1e-4,
                  batch_size=2, device="cuda", cache_dir=None, seed=0):
    """Minimal LoRA SFT. Used to manufacture the variants that exp02/11/12/14 compare."""
    from peft import LoraConfig, get_peft_model
    from torch.utils.data import DataLoader

    torch.manual_seed(seed)
    model, tok = load(model_id, device=device, cache_dir=cache_dir)
    model = get_peft_model(model, LoraConfig(
        r=rank, lora_alpha=2 * rank, lora_dropout=0.05, task_type="CAUSAL_LM"))
    model.train()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)

    dl = DataLoader(texts, batch_size=batch_size, shuffle=True)
    it, done = iter(dl), 0
    while done < steps:
        try:
            batch = next(it)
        except StopIteration:
            it = iter(dl)
            continue
        enc = tok(list(batch), return_tensors="pt", padding=True,
                  truncation=True, max_length=256)
        enc = {k: v.to(device) for k, v in enc.items()}
        loss = model(**enc, labels=enc["input_ids"]).loss
        loss.backward()
        opt.step()
        opt.zero_grad()
        done += 1
    model.save_pretrained(out_dir)
    return out_dir


def fisher_weighted_merge(adapter_a, adapter_b, out_dir, w_a=0.5):
    """Weighted average of two LoRA adapters (uniform Fisher = plain averaging)."""
    from safetensors.torch import load_file, save_file
    import os, shutil, glob

    def _weights(d):
        hits = glob.glob(os.path.join(d, "*.safetensors"))
        return load_file(hits[0]) if hits else None

    A, B = _weights(adapter_a), _weights(adapter_b)
    if A is None or B is None:
        raise FileNotFoundError("adapter weights not found")
    merged = {k: w_a * A[k] + (1 - w_a) * B[k] for k in A if k in B}
    os.makedirs(out_dir, exist_ok=True)
    for f in glob.glob(os.path.join(adapter_a, "*.json")):
        shutil.copy(f, out_dir)
    save_file(merged, os.path.join(out_dir, "adapter_model.safetensors"))
    return out_dir
