# ndna_lib/merging/compute_fisher_lora.py
import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import torch
import torch.nn.functional as F
from tqdm import tqdm
from safetensors.torch import save_file

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


def iter_region_texts(
    dataset_id: str,
    split: str,
    region: str,
    streaming: bool,
    seed: int,
    max_examples: int,
    shuffle_buffer: int,
) -> Iterable[str]:
    """
    Stream nDNA/WikiCulture and yield formatted samples for bucket_geo == region.
    Note: datasets IterableDataset shuffle is buffer-based (approximate). :contentReference[oaicite:3]{index=3}
    """
    # WikiCulture has per-region configs (AF/AS/...). Always pick the matching config to avoid HF "config missing" errors.
    ds = load_dataset(dataset_id, name=region, split=split, streaming=streaming)
    ds = ds.shuffle(seed=seed, buffer_size=shuffle_buffer) if streaming else ds.shuffle(seed=seed)

    def fmt(ex):
        title = (ex.get("page_title") or "").replace("_", " ")
        body = ex.get("text") or ""
        return f"# {title}\n\n{body}"

    n = 0
    for ex in ds:
        if ex.get("bucket_geo") != region:
            continue
        yield fmt(ex)
        n += 1
        if n >= max_examples:
            break


def batchify(it: Iterable[str], batch_size: int) -> Iterable[List[str]]:
    buf: List[str] = []
    for x in it:
        buf.append(x)
        if len(buf) == batch_size:
            yield buf
            buf = []
    if buf:
        yield buf


def get_input_device(model) -> torch.device:
    # More reliable than next(parameters) for sharded models.
    try:
        emb = model.get_input_embeddings()
        if emb is not None and hasattr(emb, "weight"):
            return emb.weight.device
    except Exception:
        pass
    return next(model.parameters()).device


def get_trainable_lora_params(model) -> List[Tuple[str, torch.nn.Parameter]]:
    """
    Be strict: only LoRA trainables. This avoids accidentally including other trainables.
    """
    params = []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        # Typical PEFT LoRA names contain 'lora_'.
        if "lora_" in n:
            params.append((n, p))
    if not params:
        raise RuntimeError("No trainable LoRA parameters found (did you load adapter with is_trainable=True?).")
    return params


def loss_sum_causal_lm(logits, input_ids, attention_mask) -> Tuple[torch.Tensor, int]:
    """
    Compute SUM of token cross-entropy for next-token prediction.
    This avoids the default mean-reduction scaling trap.
    Returns (loss_sum, token_count).
    """
    # Shift for causal LM: predict token t+1 from token t
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    shift_mask = attention_mask[:, 1:].contiguous()

    vocab = shift_logits.size(-1)
    loss_flat = F.cross_entropy(
        shift_logits.view(-1, vocab),
        shift_labels.view(-1),
        reduction="none",
    )
    loss_tok = loss_flat.view_as(shift_labels)  # [B, T-1]
    loss_sum = (loss_tok * shift_mask).sum()

    token_count = int(shift_mask.sum().item())
    return loss_sum, token_count


@torch.no_grad()
def _init_fisher(params: List[Tuple[str, torch.nn.Parameter]], fisher_dtype: torch.dtype) -> Dict[str, torch.Tensor]:
    return {n: torch.zeros_like(p, dtype=fisher_dtype, device="cpu") for (n, p) in params}


def compute_diag_fisher_per_example(
    model,
    tokenizer,
    texts: Iterable[str],
    batch_size: int,
    max_length: int,
    fisher_dtype: torch.dtype = torch.float32,
) -> Tuple[Dict[str, torch.Tensor], int, int]:
    """
    Approx diagonal empirical Fisher over examples:
      F ≈ (1/N) Σ (∂ L(x_i) / ∂θ)^2
    where L(x_i) is SUM token NLL for the sequence.

    If batch_size > 1, we do micro-backprops per example to avoid cross-terms.
    """
    model.eval()
    model.config.use_cache = False

    params = get_trainable_lora_params(model)
    fisher = _init_fisher(params, fisher_dtype)

    device = get_input_device(model)

    total_tokens = 0
    n_examples = 0

    for batch_texts in tqdm(batchify(texts, batch_size), desc="Fisher"):
        enc = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        B = input_ids.size(0)

        # Micro-backprop per example to match per-example squared grads
        for i in range(B):
            model.zero_grad(set_to_none=True)

            ids_i = input_ids[i : i + 1]
            mask_i = attention_mask[i : i + 1]

            out = model(input_ids=ids_i, attention_mask=mask_i)
            loss_sum, tok = loss_sum_causal_lm(out.logits, ids_i, mask_i)
            if tok == 0:
                continue

            # enable grads for backward
            loss_sum.backward()

            for (n, p) in params:
                if p.grad is None:
                    continue
                fisher[n] += (p.grad.detach().float() ** 2).cpu()

            total_tokens += tok
            n_examples += 1

    if n_examples == 0:
        raise RuntimeError("No usable examples processed. Check filtering / formatting / max_length.")

    for n in fisher:
        fisher[n] /= float(n_examples)

    return fisher, n_examples, total_tokens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", required=True, help="e.g. Qwen/Qwen3-4B")
    ap.add_argument("--adapter_dir", required=True, help="local dir containing adapter_config.json + adapter_model.safetensors")
    ap.add_argument("--dataset_id", default="nDNA/WikiCulture")
    ap.add_argument("--split", default="train")
    ap.add_argument("--region", required=True, help="AF/CH/...")
    ap.add_argument("--streaming", action="store_true")
    ap.add_argument("--max_examples", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--shuffle_buffer", type=int, default=10_000, help="Streaming shuffle buffer size; reduce to shorten warmup.")
    ap.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    ap.add_argument("--out_path", required=True)
    args = ap.parse_args()

    if args.device == "cpu":
        torch_dtype = torch.float32
        device_map = None
    else:
        dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
        torch_dtype = dtype_map[args.dtype]
        device_map = "auto"

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch_dtype,
        device_map=device_map,
    )
    if device_map is None:
        base = base.to(args.device)

    # is_trainable=True makes adapter params require grad :contentReference[oaicite:4]{index=4}
    model = PeftModel.from_pretrained(base, args.adapter_dir, is_trainable=True)

    texts = iter_region_texts(
        dataset_id=args.dataset_id,
        split=args.split,
        region=args.region,
        streaming=args.streaming,
        seed=args.seed,
        max_examples=args.max_examples,
        shuffle_buffer=args.shuffle_buffer,
    )

    fisher, n_examples, total_tokens = compute_diag_fisher_per_example(
        model=model,
        tokenizer=tokenizer,
        texts=texts,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )

    # Strip '.default' from keys to match adapter_model.safetensors naming
    fisher_clean = {k.replace(".default.", "."): v for k, v in fisher.items()}

    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    meta = {
        "__n_examples__": torch.tensor([n_examples], dtype=torch.int64),
        "__total_tokens__": torch.tensor([total_tokens], dtype=torch.int64),
    }
    save_file({**fisher_clean, **meta}, str(out_path))
    print(f"[OK] Saved Fisher -> {out_path} (examples={n_examples}, tokens={total_tokens})")


if __name__ == "__main__":
    main()