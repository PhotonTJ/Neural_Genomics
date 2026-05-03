# scripts/run_method5_lora.py
"""
Run Method 5 metrics for LoRA fine-tuned and Fisher-merged adapters.

Key constraints for geometry.py compatibility:
- No Accelerate device_map="auto" (geometry uses .to(geo.DEVICE) patterns)
- If computing parameter-space Fisher (E_l), do NOT use quantized weights.
- Adapters must be unloaded between runs (PEFT modifies base model in-place).
"""

from __future__ import annotations

import argparse
import os
import subprocess
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from datasets import Dataset, load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Optional quant config (avoid hard crash on older transformers)
try:
    from transformers import Mxfp4Config
except Exception:
    Mxfp4Config = None  # type: ignore

import ndna_lib.data as data_lib
import ndna_lib.geometry as geo


# -----------------------------
# Constants
# -----------------------------

REGIONS = ["AF", "AS", "AU", "CH", "EU", "LA", "ME", "NA"]

MERGED_PAIRS = [
    f"{REGIONS[i]}_{REGIONS[j]}"
    for i in range(len(REGIONS))
    for j in range(i + 1, len(REGIONS))
]

DEFAULT_BASE_MODEL = "Qwen/Qwen3-4B"
DEFAULT_ADAPTER_REPO = "nDNA/Qwen3-4B-WikiCulture-SFT"
DEFAULT_MERGED_SUBDIR = "merged_adapters"


# -----------------------------
# Dataset formatters
# -----------------------------

def _safe_strip(x: Any) -> str:
    if x is None:
        return ""
    return str(x).strip()


def _build_squad_input(context: str, question: str) -> str:
    context = (context or "").strip()
    question = (question or "").strip()
    return f"Context: {context}\nQuestion: {question}\n"


def _plato_main_text_only(ex: Dict[str, Any]) -> str:
    main_text = ex.get("main_text", None)
    if main_text is None:
        return ""
    if isinstance(main_text, str):
        return main_text.strip()
    parts: List[str] = []
    if isinstance(main_text, list):
        for section in main_text:
            if not isinstance(section, dict):
                continue
            main_content = section.get("main_content", [])
            if isinstance(main_content, list):
                for para in main_content:
                    if isinstance(para, str) and para.strip():
                        parts.append(para.strip())
            elif isinstance(main_content, str) and main_content.strip():
                parts.append(main_content.strip())
            # subsections skipped intentionally
    return "\n".join(parts).strip()


def _split_ag_news_title_desc(text: str) -> Tuple[str, str]:
    s = (text or "").strip()
    if not s:
        return "", ""
    if " - " in s:
        left, right = s.split(" - ", 1)
        return left.strip(), right.strip()
    dot = s.find(". ")
    if 10 <= dot <= 160:
        return s[:dot].strip(), s[dot + 2:].strip()
    return s[:120].strip(), s[120:].strip()


def _build_ag_news_prompt(title: str) -> str:
    title = (title or "").strip()
    return f"Title: {title}\ndescription:"


def _mbpp_build_input(ex: Dict[str, Any]) -> str:
    task = _safe_strip(ex.get("text", "")) or _safe_strip(ex.get("prompt", ""))
    setup = (
        _safe_strip(ex.get("test_setup_code", ""))
        or _safe_strip(ex.get("test_imports", ""))
        or _safe_strip(ex.get("test_setup", ""))
    )
    test_list = ex.get("test_list", [])
    challenge_list = ex.get("challenge_test_list", [])
    lines: List[str] = []
    if task:
        lines.append(task)
    if setup:
        lines.append("\n# Setup\n" + setup)
    if isinstance(test_list, list) and test_list:
        lines.append("\n# Tests")
        lines.extend([f"- {_safe_strip(t)}" for t in test_list if _safe_strip(t)])
    if isinstance(challenge_list, list) and challenge_list:
        lines.append("\n# Challenge tests")
        lines.extend([f"- {_safe_strip(t)}" for t in challenge_list if _safe_strip(t)])
    lines.append("\n# Solution\n")
    return "\n".join(lines).strip() + "\n"


SUPPORTED_DATASETS = [
    "squad",
    "squad_v2",
    "stanford_plato",
    "ag_news",
    "hh_rlhf",
    "gsm8k",
    "harmbench",
    "litmus",
    "mbpp",
    "alpaca",
]


def _load_and_materialize_texts(
    *,
    dataset_key: str,
    split: str,
    streaming: bool,
    shuffle: bool,
    seed: int,
    max_samples: int,
    shuffle_buffer_size: int,
    ag_label: Optional[int],
    hh_data_dir: str,
    gsm8k_config: str,
    harmbench_config: str,
    mbpp_config: str,
    alpaca_dataset_id: str,
) -> Dataset:
    if max_samples <= 0:
        raise ValueError("--max-samples must be > 0")

    key = dataset_key

    if key == "squad":
        raw = load_dataset("squad", split=split, streaming=streaming)
        def ex_to_text(ex):
            return _build_squad_input(_safe_strip(ex.get("context", "")), _safe_strip(ex.get("question", "")))

    elif key == "squad_v2":
        raw = load_dataset("rajpurkar/squad_v2", split=split, streaming=streaming)
        def ex_to_text(ex):
            return _build_squad_input(_safe_strip(ex.get("context", "")), _safe_strip(ex.get("question", "")))

    elif key == "stanford_plato":
        raw = load_dataset("hugfaceguy0001/stanford_plato", split=split, streaming=streaming)
        def ex_to_text(ex):
            t = _plato_main_text_only(ex)
            return t if t else None

    elif key == "ag_news":
        raw = load_dataset("SetFit/ag_news", split=split, streaming=streaming)
        def ex_to_text(ex):
            if ag_label is not None and int(ex.get("label", -1)) != ag_label:
                return None
            title, _ = _split_ag_news_title_desc(_safe_strip(ex.get("text", "")))
            if not title:
                return None
            return _build_ag_news_prompt(title)

    elif key == "hh_rlhf":
        raw = load_dataset("Anthropic/hh-rlhf", data_dir=hh_data_dir, split=split, streaming=streaming)
        def ex_to_text(ex):
            t = _safe_strip(ex.get("rejected", ""))
            return t if t else None

    elif key == "gsm8k":
        raw = load_dataset("openai/gsm8k", gsm8k_config, split=split, streaming=streaming)
        def ex_to_text(ex):
            t = _safe_strip(ex.get("question", ""))
            return t if t else None

    elif key == "harmbench":
        raw = load_dataset("AlignmentResearch/HarmBench", harmbench_config, split=split, streaming=streaming)
        def ex_to_text(ex):
            t = _safe_strip(ex.get("instructions", ""))
            if t:
                return t
            content = ex.get("content", [])
            if isinstance(content, list) and content:
                t = _safe_strip(content[0])
            else:
                t = _safe_strip(content)
            return t if t else None

    elif key == "litmus":
        raw = load_dataset("hasnat79/litmus", split=split, streaming=streaming)
        def ex_to_text(ex):
            t = _safe_strip(ex.get("input", ""))
            return t if t else None

    elif key == "mbpp":
        raw = load_dataset("google-research-datasets/mbpp", mbpp_config, split=split, streaming=streaming)
        def ex_to_text(ex):
            t = _mbpp_build_input(ex)
            return t if t else None

    elif key == "alpaca":
        raw = load_dataset(alpaca_dataset_id, split=split, streaming=streaming)
        def ex_to_text(ex):
            inst = _safe_strip(ex.get("instruction", ""))
            inp = _safe_strip(ex.get("input", ""))
            return data_lib.build_alpaca_prompt(inst, inp)

    else:
        raise ValueError(f"Unknown dataset '{dataset_key}'. Supported: {SUPPORTED_DATASETS}")

    if shuffle:
        if streaming:
            raw = raw.shuffle(seed=seed, buffer_size=int(shuffle_buffer_size))
        else:
            raw = raw.shuffle(seed=seed)

    texts: List[str] = []
    for ex in raw:
        t = ex_to_text(ex)
        if not t:
            continue
        texts.append(t)
        if len(texts) >= max_samples:
            break

    if not texts:
        raise RuntimeError(f"No usable texts produced for dataset={dataset_key}.")

    return Dataset.from_dict({"text": texts})


# -----------------------------
# Adapter resolution
# -----------------------------

def _resolve_adapters(
    *,
    adapters_mode: str,
    adapter_names: Optional[List[str]],
    local_adapters_dir: Optional[str],
    local_merged_dir: Optional[str],
    adapter_repo: str,
    merged_subdir: str,
) -> List[Tuple[str, str, str]]:
    results: List[Tuple[str, str, str]] = []

    if adapter_names:
        names_to_run = adapter_names
    elif adapters_mode == "finetuned":
        names_to_run = REGIONS
    elif adapters_mode == "merged":
        names_to_run = MERGED_PAIRS
    elif adapters_mode == "all":
        names_to_run = REGIONS + MERGED_PAIRS
    else:
        raise ValueError(f"Unknown adapters mode: {adapters_mode}")

    for name in names_to_run:
        if name in REGIONS:
            if local_adapters_dir:
                path = os.path.join(local_adapters_dir, name)
            else:
                path = f"{adapter_repo}:{name}"
            results.append((name, path, "finetuned"))

        elif name in MERGED_PAIRS:
            if local_merged_dir:
                path = os.path.join(local_merged_dir, name)
            else:
                path = f"{adapter_repo}:{merged_subdir}/{name}"
            results.append((name, path, "merged"))

        else:
            raise ValueError(f"Unknown adapter name: {name}")

    if not results:
        raise ValueError("No adapters to run.")

    return results


def _safe_model_tag(s: str) -> str:
    return s.replace("/", "_").replace(":", "_").replace("@", "_").replace("-", "_")


def _is_windows_drive_path(p: str) -> bool:
    # "C:\foo" -> drive letter + colon
    return os.name == "nt" and len(p) >= 2 and p[1] == ":"


def _attach_adapter(
    base_model: torch.nn.Module,
    adapter_path: str,
    *,
    peft_adapter_name: str = "current",
) -> PeftModel:
    """
    Attach LoRA adapter to base model.

    adapter_path may be:
    - local folder path
    - "repo_id:subfolder" for HF subfolder loading
    """
    if (":" in adapter_path) and (not _is_windows_drive_path(adapter_path)) and (not os.path.exists(adapter_path)):
        repo_id, subfolder = adapter_path.split(":", 1)
        print(f"  HF: {repo_id} | subfolder={subfolder}")
        return PeftModel.from_pretrained(
            base_model,
            repo_id,
            subfolder=subfolder,
            is_trainable=True,   # IMPORTANT for Fisher workflows
            adapter_name=peft_adapter_name,
        )
    else:
        print(f"  local: {adapter_path}")
        return PeftModel.from_pretrained(
            base_model,
            adapter_path,
            is_trainable=True,
            adapter_name=peft_adapter_name,
        )


def _ensure_float_grads(model: torch.nn.Module) -> None:
    """
    Force requires_grad=True for floating/complex parameters.
    (Non-float params cannot require grad.)
    """
    for _, p in model.named_parameters():
        if p.is_floating_point() or p.is_complex():
            p.requires_grad_(True)


def _grad_sanity_check(model: torch.nn.Module) -> None:
    bad: List[str] = []
    for name, p in model.named_parameters():
        if (p.is_floating_point() or p.is_complex()) and (not p.requires_grad):
            bad.append(f"{name} ({p.dtype})")
    if bad:
        preview = "\n    ".join(bad[:40])
        raise RuntimeError(
            "Some floating/complex parameters still have requires_grad=False.\n"
            "This will break torch.autograd.grad in compute_param_effort.\n"
            f"First few:\n    {preview}\n"
            f"Total: {len(bad)}"
        )


def _unload_adapter(model: PeftModel) -> torch.nn.Module:
    """
    Remove adapter modules and return restored base model.

    Do NOT use merge_and_unload() in this workflow.
    """
    if hasattr(model, "unload"):
        restored = model.unload()
        return restored
    raise RuntimeError("Your PEFT version does not expose PeftModel.unload(). Upgrade peft.")


# -----------------------------
# One run
# -----------------------------

def run_one(
    *,
    base_model_name: str,
    adapter_name: str,
    adapter_path: str,
    adapter_type: str,
    ds_text: Dataset,
    out_dir: str,
    dataset_tag: str,
    max_len: int,
    batch_size: int,
    tokens_per_ex: int,
    tau: float,
    fisher_unit: str,
    compute_kappa: bool,
    kappa_keep_last_k: Optional[int],
    kappa_include_embedding_node: bool,
    skip_param_effort: bool,
    use_mxfp4: bool,
    base_model_cache: Optional[Dict[str, Any]] = None,
) -> str:
    device = geo.DEVICE

    print("\n====================")
    print(f"Dataset:   {dataset_tag}")
    print(f"Adapter:   {adapter_name} ({adapter_type})")
    print(f"Path:      {adapter_path}")
    print(f"Base:      {base_model_name}")
    print(f"Device:    {device}")
    print(f"Samples:   {len(ds_text)}")
    print("====================")

    # Tokenizer (cache OK)
    if base_model_cache and "tokenizer" in base_model_cache:
        tokenizer = base_model_cache["tokenizer"]
        print("  [cache] tokenizer")
    else:
        tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=True)
        if tokenizer.pad_token is None:
            if tokenizer.eos_token is None:
                raise ValueError("Tokenizer has no pad_token or eos_token.")
            tokenizer.pad_token = tokenizer.eos_token
        if base_model_cache is not None:
            base_model_cache["tokenizer"] = tokenizer

    # Base model (cache OK, but be careful: adapters modify it in-place)
    if base_model_cache and "base_model" in base_model_cache:
        base_model = base_model_cache["base_model"]
        print("  [cache] base_model")
    else:
        if use_mxfp4 and (not skip_param_effort):
            # This is the single most common reason your Fisher grad path explodes.
            raise RuntimeError(
                "MXFP4/quantized weights + compute_param_effort is not supported.\n"
                "Run with --no-mxfp4 (recommended) OR --skip-param-effort."
            )

        model_kwargs: Dict[str, Any] = dict(
            attn_implementation="eager",
            torch_dtype=torch.bfloat16,
            use_cache=False,
        )
        if use_mxfp4:
            if Mxfp4Config is None:
                raise RuntimeError("Mxfp4Config not available. Use --no-mxfp4.")
            model_kwargs["quantization_config"] = Mxfp4Config(dequantize=True)

        base_model = AutoModelForCausalLM.from_pretrained(base_model_name, **model_kwargs)
        base_model.to(device)
        base_model.eval()
        if base_model_cache is not None:
            base_model_cache["base_model"] = base_model

    # Attach adapter
    model = _attach_adapter(base_model, adapter_path, peft_adapter_name="current")
    model.to(device)
    model.eval()

    # Make sure gradients are enabled for Fisher logic
    torch.set_grad_enabled(True)
    _ensure_float_grads(model)
    _grad_sanity_check(model)

    # Build ndna adapter from underlying base model
    unwrapped = model.get_base_model()
    ndna_adapter = geo.make_adapter(unwrapped, base_model_name)
    print(f"  ndna_adapter: {type(ndna_adapter).__name__} | layers={ndna_adapter.num_layers}")

    collate_fn = geo.collate_causal(tokenizer, max_len=max_len)

    def make_loader():
        return DataLoader(ds_text, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # 1) Parameter-space effort
    E_l = None
    n_ex = 0
    n_params = np.array([], dtype=np.int64)

    if not skip_param_effort:
        print("\n[1] E_l (observed Fisher / param effort)...")
        with torch.enable_grad():
            E_l, n_ex, n_params = geo.compute_param_effort(
                model=model,
                adapter=ndna_adapter,
                loader=make_loader(),
                unit=fisher_unit,
            )
        print(f"  Used examples/tokens count: {n_ex} (unit={fisher_unit})")
    else:
        print("\n[1] Skipping E_l (--skip-param-effort)")

    # 2) FR geometry
    print("\n[2] FR geometry (Delta, Alpha, Vnorm)...")
    Delta, Alpha, Vnorm, mean_total_fr, n_tokens = geo.compute_fr_and_alignment_streaming(
        model=model,
        adapter=ndna_adapter,
        loader=make_loader(),
    )
    print(f"  FR valid tokens: {n_tokens}")
    print(f"  Mean total FR length/token: {mean_total_fr:.6e} rad")

    # 3) Belief field norms
    print("\n[3] Belief field norms ||v_l|| ...")
    belief_norms = geo.belief_field_for_dataset(
        model=model,
        adapter=ndna_adapter,
        tokenizer=tokenizer,
        dataset=ds_text,
        max_seq_len=max_len,
        tokens_per_ex=tokens_per_ex,
        batch_size=batch_size,
        tau=tau,
        fr_norm=True,
    )

    # 4) Kappa + nDNA_pred
    kappa = None
    kappa_positions = None
    ndna_scalar_val = None
    ndna_layerwise_arr = None

    if compute_kappa:
        print("\n[4] Kappa (dataset expectation over supervised (x,t)) ...")
        kappa, kappa_positions = geo.spectral_curvature_for_loader(
            model=model,
            adapter=ndna_adapter,
            loader=make_loader(),
            tau=tau,
            keep_last_k=kappa_keep_last_k,
            include_embedding_node=bool(kappa_include_embedding_node),
        )
        print(f"  Kappa shape: {kappa.shape} | positions used: {kappa_positions}")

        if kappa.shape == Delta.shape:
            v_dict = {"concept": belief_norms}
            ndna_scalar, ndna_layerwise = geo.compute_ndna_pred(
                kappa=kappa,
                fr_steps=Delta,
                v_norms_by_concept=v_dict,
                l_min=2,
            )
            ndna_scalar_val = float(ndna_scalar["concept"])
            ndna_layerwise_arr = np.asarray(ndna_layerwise["concept"], dtype=float)
            print(f"  nDNA_pred(concept) = {ndna_scalar_val:.6e}")
        else:
            print(f"  [warn] kappa shape {kappa.shape} != Delta shape {Delta.shape}. Skipping nDNA_pred.")
    else:
        print("\n[4] Skipping kappa (--no-kappa)")

    # CRITICAL CLEANUP: unload adapter and restore cached base model
    try:
        restored_base = _unload_adapter(model)
        restored_base.to(device)
        restored_base.eval()
        if base_model_cache is not None:
            base_model_cache["base_model"] = restored_base
        print("  [cleanup] Adapter unloaded; base model restored")
    except Exception as e:
        print(f"  [cleanup] ERROR: unload() failed: {e}")
        if base_model_cache is not None:
            base_model_cache.pop("base_model", None)

    del model, ndna_adapter
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Save
    os.makedirs(out_dir, exist_ok=True)
    ds_tag = dataset_tag.replace("/", "_")
    adapter_tag = _safe_model_tag(adapter_name)
    out_path = os.path.join(out_dir, f"{ds_tag}__method5_lora_{adapter_type}_{adapter_tag}.npz")

    save_kwargs: Dict[str, Any] = dict(
        base_model=np.array([base_model_name]),
        adapter_name=np.array([adapter_name]),
        adapter_path=np.array([adapter_path]),
        adapter_type=np.array([adapter_type]),
        dataset=np.array([dataset_tag]),
        Delta=Delta,
        Alpha=Alpha,
        Vnorm=Vnorm,
        mean_total_fr=np.array([mean_total_fr], dtype=np.float64),
        n_tokens=np.array([n_tokens], dtype=np.int64),
        belief_norms=belief_norms,
        tau=np.array([tau], dtype=np.float64),
        max_len=np.array([max_len], dtype=np.int64),
        batch_size=np.array([batch_size], dtype=np.int64),
        tokens_per_ex=np.array([tokens_per_ex], dtype=np.int64),
        fisher_unit=np.array([fisher_unit]),
        skip_param_effort=np.array([int(skip_param_effort)], dtype=np.int64),
    )

    if E_l is not None:
        save_kwargs["E_l"] = E_l
        save_kwargs["n_examples"] = np.array([n_ex], dtype=np.int64)
        save_kwargs["n_params"] = np.array(n_params, dtype=np.int64)

    if kappa is not None:
        save_kwargs["kappa"] = kappa
        save_kwargs["kappa_positions"] = np.array([kappa_positions], dtype=np.int64)

    if ndna_scalar_val is not None and ndna_layerwise_arr is not None:
        save_kwargs["ndna_scalar"] = np.array([ndna_scalar_val], dtype=np.float64)
        save_kwargs["ndna_layerwise"] = ndna_layerwise_arr

    np.savez_compressed(out_path, **save_kwargs)
    print(f"\nSaved: {out_path}")
    return out_path


def _git_push_results(out_path: str, adapter_name: str, dataset_tag: str) -> bool:
    """
    Git add, commit, and push the result file after each adapter run.
    Returns True on success, False on failure.
    """
    try:
        # Get repo root
        repo_root = subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()

        # Add the specific file
        subprocess.run(
            ["git", "add", out_path],
            cwd=repo_root,
            check=True,
            capture_output=True,
        )

        # Commit with descriptive message
        commit_msg = f"[auto] Method5 results: {adapter_name} on {dataset_tag}"
        result = subprocess.run(
            ["git", "commit", "-m", commit_msg],
            cwd=repo_root,
            capture_output=True,
            text=True,
        )
        
        if result.returncode != 0:
            if "nothing to commit" in result.stdout or "nothing to commit" in result.stderr:
                print(f"  [git] No changes to commit for {out_path}")
                return True
            print(f"  [git] Commit failed: {result.stderr}")
            return False

        # Push to origin
        result = subprocess.run(
            ["git", "push", "origin"],
            cwd=repo_root,
            capture_output=True,
            text=True,
        )
        
        if result.returncode != 0:
            print(f"  [git] Push failed: {result.stderr}")
            return False

        print(f"  [git] Pushed: {commit_msg}")
        return True

    except subprocess.CalledProcessError as e:
        print(f"  [git] Error: {e}")
        return False
    except Exception as e:
        print(f"  [git] Unexpected error: {e}")
        return False


# -----------------------------
# CLI
# -----------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Run Method 5 metrics for LoRA fine-tuned and Fisher-merged adapters.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    ap.add_argument("--adapters", type=str, choices=["finetuned", "merged", "all"], default="all")
    ap.add_argument("--adapter-names", nargs="+", default=None)
    ap.add_argument("--list-adapters", action="store_true")

    ap.add_argument("--base-model", type=str, default=DEFAULT_BASE_MODEL)
    ap.add_argument("--adapter-repo", type=str, default=DEFAULT_ADAPTER_REPO)
    ap.add_argument("--merged-subdir", type=str, default=DEFAULT_MERGED_SUBDIR)
    ap.add_argument("--local-adapters-dir", type=str, default=None)
    ap.add_argument("--local-merged-dir", type=str, default=None)

    ap.add_argument("--dataset", type=str, default=None)
    ap.add_argument("--datasets", nargs="+", default=None)
    ap.add_argument("--list-datasets", action="store_true")
    ap.add_argument("--split", type=str, default="train")

    ap.add_argument("--streaming", action="store_true")
    ap.add_argument("--no-shuffle", action="store_true")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--shuffle-buffer-size", type=int, default=10_000)
    ap.add_argument("--max-samples", type=int, default=32)

    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--tokens-per-ex", type=int, default=8)
    ap.add_argument("--tau", type=float, default=1.0)
    ap.add_argument("--fisher-unit", type=str, default="sequence", choices=["sequence", "token"])

    ap.add_argument("--ag-label", type=int, default=None)
    ap.add_argument("--hh-data-dir", type=str, default="helpful-base")
    ap.add_argument("--gsm8k-config", type=str, default="main")
    ap.add_argument("--harmbench-config", type=str, default="default")
    ap.add_argument("--mbpp-config", type=str, default="full")
    ap.add_argument("--alpaca-dataset-id", type=str, default=data_lib.ALPACA_DATASET_ID)

    ap.add_argument("--no-kappa", action="store_true")
    ap.add_argument("--kappa-keep-last-k", type=int, default=8)
    ap.add_argument(
        "--kappa-include-embedding-node",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Default True. Use --no-kappa-include-embedding-node to disable.",
    )

    ap.add_argument(
        "--skip-param-effort",
        action="store_true",
        help="Skip E_l (param Fisher). Use this if you want to run quantized models.",
    )

    # NOTE: default is OFF because Fisher needs real grads.
    ap.add_argument("--use-mxfp4", action="store_true", help="Enable MXFP4 quant config (NOT compatible with E_l).")

    ap.add_argument("--out-dir", type=str, default="results/method5_lora")
    ap.add_argument("--git-push", action="store_true",
                    help="Git add, commit, and push results after each adapter run.")

    args = ap.parse_args()

    if args.list_adapters:
        print("Fine-tuned adapters:")
        for r in REGIONS:
            print(f"  {r}")
        print("\nMerged adapters:")
        for p in MERGED_PAIRS:
            print(f"  {p}")
        return

    if args.list_datasets:
        print("\n".join(SUPPORTED_DATASETS))
        return

    if args.datasets:
        ds_list = list(args.datasets)
    else:
        if args.dataset is None:
            raise ValueError("Provide --dataset <key> or --datasets <k1 k2 ...>.")
        ds_list = [args.dataset]

    for d in ds_list:
        if d not in SUPPORTED_DATASETS:
            raise ValueError(f"Unknown dataset '{d}'.")

    adapters_to_run = _resolve_adapters(
        adapters_mode=args.adapters,
        adapter_names=args.adapter_names,
        local_adapters_dir=args.local_adapters_dir,
        local_merged_dir=args.local_merged_dir,
        adapter_repo=args.adapter_repo,
        merged_subdir=args.merged_subdir,
    )

    shuffle = not args.no_shuffle
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"[adapters] Will run {len(adapters_to_run)} adapters")
    for name, path, atype in adapters_to_run:
        print(f"  - {name} ({atype}): {path}")

    base_model_cache: Dict[str, Any] = {}

    for dataset_key in ds_list:
        ds_text = _load_and_materialize_texts(
            dataset_key=dataset_key,
            split=args.split,
            streaming=bool(args.streaming),
            shuffle=shuffle,
            seed=int(args.seed),
            max_samples=int(args.max_samples),
            shuffle_buffer_size=int(args.shuffle_buffer_size),
            ag_label=(int(args.ag_label) if args.ag_label is not None else None),
            hh_data_dir=str(args.hh_data_dir),
            gsm8k_config=str(args.gsm8k_config),
            harmbench_config=str(args.harmbench_config),
            mbpp_config=str(args.mbpp_config),
            alpaca_dataset_id=str(args.alpaca_dataset_id),
        )
        print(f"\n[data] Materialized {len(ds_text)} texts for dataset={dataset_key}")

        for adapter_name, adapter_path, adapter_type in adapters_to_run:
            print(f"\n[run] dataset='{dataset_key}' | adapter='{adapter_name}' ({adapter_type})")
            try:
                out_path = run_one(
                    base_model_name=args.base_model,
                    adapter_name=adapter_name,
                    adapter_path=adapter_path,
                    adapter_type=adapter_type,
                    ds_text=ds_text,
                    out_dir=args.out_dir,
                    dataset_tag=dataset_key,
                    max_len=int(args.max_len),
                    batch_size=int(args.batch_size),
                    tokens_per_ex=int(args.tokens_per_ex),
                    tau=float(args.tau),
                    fisher_unit=str(args.fisher_unit),
                    compute_kappa=(not args.no_kappa),
                    kappa_keep_last_k=(int(args.kappa_keep_last_k) if args.kappa_keep_last_k is not None else None),
                    kappa_include_embedding_node=bool(args.kappa_include_embedding_node),
                    skip_param_effort=bool(args.skip_param_effort),
                    use_mxfp4=bool(args.use_mxfp4),
                    base_model_cache=base_model_cache,
                )
                
                # Git push after each successful run if requested
                if args.git_push and out_path:
                    _git_push_results(out_path, adapter_name, dataset_key)
                    
            except Exception as e:
                print(f"[ERROR] Failed adapter={adapter_name}: {e}")
                import traceback
                traceback.print_exc()
                continue


if __name__ == "__main__":
    main()
