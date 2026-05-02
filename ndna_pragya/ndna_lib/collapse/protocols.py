from __future__ import annotations

import json
import os
import shutil
import tempfile
from abc import ABC, abstractmethod
from typing import List, Dict, Union, Tuple, Optional
import gc

import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
from tqdm import tqdm

# Try to import vLLM for faster inference
try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False

from ndna_lib.collapse.config import CollapseConfig
from ndna_lib.collapse.geometry_runner import DEVICE

# canonical data helpers
from ndna_lib.data import (
    build_alpaca_prompt,
    build_alpaca_full_text,
    load_alpaca_text_dataset,
    load_alpaca_instructions_only,
    build_lm_dataset_from_texts,
)

def clear_gpu_memory():
    """Force garbage collection and clear CUDA cache."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

# --------------------------------------------------------------------------
# Minimal Trainer subclass that never saves optimizer/scheduler
# --------------------------------------------------------------------------


class NoSaveTrainer(Trainer):
    """
    Trainer that:
      - never saves optimizer / scheduler state
      - only saves model weights if explicitly asked (we'll handle saving manually)
    """

    def _save_optimizer_and_scheduler(self, output_dir: str):
        # Disable optimizer/scheduler saving completely
        return

    def _save_checkpoint(self, model, trial, metrics=None):
        # If someone ever calls a save checkpoint with save_strategy != "no",
        # only write model weights and skip everything else.
        if self.args.save_strategy == "no":
            # With save_strategy="no", this should never be called in normal flow.
            return
        self.save_model(self.args.output_dir)
        # No optimizer / scheduler / trainer state


# --------------------------------------------------------------------------
# Base protocol interface
# --------------------------------------------------------------------------


class CollapseProtocol(ABC):
    """
    Strategy interface for "how do we get the next generation's checkpoint".
    Gen 0 is just the base model.
    """

    def __init__(self, cfg: CollapseConfig):
        self.cfg = cfg

    @abstractmethod
    def finetune_one_generation(
        self,
        gen_idx: int,
        base_ckpt: str = None,
        out_dir: str = None,
        save_model: bool = True,
        model: "AutoModelForCausalLM" = None,
        tokenizer: "AutoTokenizer" = None,
    ) -> Union[str, Tuple["AutoModelForCausalLM", "AutoTokenizer"]]:
        """
        Train a new checkpoint from base_ckpt.
        
        Args:
            gen_idx: Generation index
            base_ckpt: Path to base checkpoint (if model/tokenizer not provided)
            out_dir: Output directory (only used if save_model=True)
            save_model: If True, save model to disk and return path.
                        If False, return (model, tokenizer) tuple in memory.
            model: Pre-loaded model (optional)
            tokenizer: Pre-loaded tokenizer (optional)
            
        Returns:
            If save_model=True: path to the new checkpoint directory
            If save_model=False: (model, tokenizer) tuple
        """
        ...


# --------------------------------------------------------------------------
# Cross-breeding: repeated fine-tuning on human Alpaca pairs
# --------------------------------------------------------------------------


class CrossBreedingProtocol(CollapseProtocol):
    """
    Cross-breeding:
      - For each generation, fine-tune on the same human Alpaca pairs
        (or whatever dataset_id BreedingConfig specifies).
    """

    def _load_train_dataset(self, tokenizer, max_len: int) -> Dataset:
        """
        Build a tokenized causal LM dataset from text-only Alpaca (or another
        instruction dataset) using canonical data utilities.

        IMPORTANT: honor cfg.breeding.dataset_id.
        """
        ds_text = load_alpaca_text_dataset(
            max_samples=self.cfg.breeding.max_train_samples,
            dataset_id=self.cfg.breeding.dataset_id,   # <-- FIXED: dataset_id now used
        )

        def tok_fn(batch):
            enc = tokenizer(
                batch["text"],
                truncation=True,
                max_length=max_len,
                padding="max_length",
            )
            enc["labels"] = enc["input_ids"].copy()
            return enc

        ds_tok = ds_text.map(tok_fn, batched=True, remove_columns=["text"])
        ds_tok.set_format(type="torch")
        return ds_tok

    def finetune_one_generation(
        self,
        gen_idx: int,
        base_ckpt: str = None,
        out_dir: str = None,
        save_model: bool = True,
        model: "AutoModelForCausalLM" = None,
        tokenizer: "AutoTokenizer" = None,
    ) -> Union[str, Tuple["AutoModelForCausalLM", "AutoTokenizer"]]:
        """
        Fine-tune for one generation.
        
        Args:
            gen_idx: Generation index
            base_ckpt: Path to base checkpoint (if model/tokenizer not provided)
            out_dir: Output directory (only used if save_model=True)
            save_model: If True, save model to disk and return path.
                        If False, return (model, tokenizer) tuple in memory.
            model: Pre-loaded model (optional)
            tokenizer: Pre-loaded tokenizer (optional)
            
        Returns:
            If save_model=True: path to the new checkpoint directory
            If save_model=False: (model, tokenizer) tuple
        """
        if save_model and out_dir:
            os.makedirs(out_dir, exist_ok=True)

        # Load model/tokenizer if not provided
        if model is None or tokenizer is None:
            if base_ckpt is None:
                raise ValueError("Either base_ckpt or (model, tokenizer) must be provided")
            tokenizer = AutoTokenizer.from_pretrained(base_ckpt, use_fast=True)
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token = tokenizer.eos_token
            model = AutoModelForCausalLM.from_pretrained(base_ckpt)
            model.to(DEVICE)
        
        model.config.pad_token_id = tokenizer.pad_token_id

        train_ds = self._load_train_dataset(
            tokenizer, self.cfg.breeding.train_max_seq_len
        )
        data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

        # Steps per generation: early gens vs later gens
        steps = (
            self.cfg.breeding.max_steps_first
            if gen_idx <= 3
            else self.cfg.breeding.max_steps_later
        )

        # Batch size: allow config override, else default to 2
        per_device_bs = getattr(
            self.cfg.breeding, "per_device_train_batch_size", 2
        )

        # Use a temp directory if not saving
        effective_out_dir = out_dir if out_dir else "/tmp/collapse_train_temp"

        args = TrainingArguments(
            output_dir=effective_out_dir,
            overwrite_output_dir=True,
            per_device_train_batch_size=per_device_bs,
            learning_rate=self.cfg.breeding.lr,
            max_steps=steps,
            weight_decay=0.01,
            warmup_steps=int(0.1 * steps),
            logging_steps=50,
            # critical: don't save checkpoints with optimizer/scheduler
            save_strategy="no",
            report_to=["none"],
            fp16=(DEVICE == "cuda"),
        )

        trainer = NoSaveTrainer(
            model=model,
            args=args,
            train_dataset=train_ds,
            data_collator=data_collator,
        )

        ckpt_desc = base_ckpt if base_ckpt else "in-memory"
        print(f"[CROSS] Fine-tune gen {gen_idx}: {ckpt_desc}")
        trainer.train()

        # Clean up trainer and datasets
        del trainer
        del train_ds
        del data_collator

        if save_model:
            # Manual minimal save: model weights + tokenizer only
            model.save_pretrained(out_dir)
            tokenizer.save_pretrained(out_dir)
            print(f"[CROSS] Saved fine-tuned model to {out_dir}")

            # Aggressive cleanup
            del model
            clear_gpu_memory()
            return out_dir
        else:
            # Return model in memory (don't delete it!)
            print(f"[CROSS] Keeping fine-tuned model in memory (gen {gen_idx})")
            return model, tokenizer


# --------------------------------------------------------------------------
# Inbreeding: pure self-generated Alpaca-style collapse
# --------------------------------------------------------------------------


class InbreedingProtocol(CollapseProtocol):
    """
    Inbreeding:
      - Use current model to generate Alpaca-style responses.
      - Build synthetic dataset from these responses.
      - Fine-tune on that dataset for the next generation.
    """

    def __init__(self, cfg: CollapseConfig):
        super().__init__(cfg)
        # Load instructions once; reuse across generations
        self._alpaca_records = load_alpaca_instructions_only(
            dataset_id=self.cfg.breeding.dataset_id,
            max_samples=self.cfg.breeding.max_train_samples,
        )

    # -----------------------------
    # Synthetic data generation
    # -----------------------------
    def _generate_with_vllm(
        self,
        model_path: str,
        prompts: List[str],
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        gpu_memory_utilization: float = 0.7,
    ) -> List[str]:
        """
        Generate texts using vLLM for faster inference.
        
        Args:
            model_path: Path to model checkpoint
            prompts: List of prompt strings
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            gpu_memory_utilization: Fraction of GPU memory to use (default 0.7 to leave room for other processes)
            
        Returns:
            List of generated text responses
        """
        # Clear GPU memory before starting vLLM
        clear_gpu_memory()
        
        llm = LLM(
            model=model_path,
            trust_remote_code=True,
            gpu_memory_utilization=gpu_memory_utilization,
        )
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_new_tokens,
        )
        outputs = llm.generate(prompts, sampling_params)
        responses = [out.outputs[0].text for out in outputs]
        
        # Clean up vLLM engine thoroughly to free GPU memory
        del outputs
        del sampling_params
        
        # Shutdown vLLM's distributed backend if it was initialized
        try:
            from vllm.distributed.parallel_state import destroy_model_parallel
            destroy_model_parallel()
        except Exception:
            pass
        
        # Destroy torch distributed process group if initialized
        try:
            import torch.distributed as dist
            if dist.is_initialized():
                dist.destroy_process_group()
        except Exception:
            pass
        
        del llm
        gc.collect()
        clear_gpu_memory()
        
        return responses

    def _generate_synthetic_alpaca_texts(
        self,
        model_name_or_path: Optional[str] = None,
        out_dir: str = "",
        batch_size: int = 8,
        max_prompt_len: int = 256,
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        top_p: float = 0.9,
        model: Optional["AutoModelForCausalLM"] = None,
        tokenizer: Optional["AutoTokenizer"] = None,
    ) -> Tuple[List[str], Optional["AutoModelForCausalLM"], Optional["AutoTokenizer"]]:
        """
        For each instruction, generate a response with the current model
        and return list of training texts (prompt + generated response).
        Also saves raw generations to JSONL for inspection.
        
        Uses vLLM if available for fast inference. When an in-memory model is
        provided (no path), temporarily saves it to disk for vLLM, then cleans up.
        Falls back to HuggingFace generation if vLLM is not available.
        """
        os.makedirs(out_dir, exist_ok=True)

        records = self._alpaca_records
        all_texts: List[str] = []
        raw_gen_path = os.path.join(out_dir, "synthetic_generations.jsonl")
        f_out = open(raw_gen_path, "w", encoding="utf-8")

        print(f"[GEN] Generating synthetic texts from {model_name_or_path or 'in-memory model'}")
        print(f"[GEN] Total records: {len(records)}")

        # Build all prompts upfront
        prompts = [
            build_alpaca_prompt(r["instruction"], r["input"])
            for r in records
        ]

        # Determine if we can use vLLM
        temp_model_dir = None
        vllm_model_path = None
        model_was_on_gpu = False
        
        if VLLM_AVAILABLE:
            if model_name_or_path is not None and model is None and tokenizer is None:
                # We have a path, use it directly
                vllm_model_path = model_name_or_path
            elif model is not None and tokenizer is not None:
                # We have an in-memory model - save temporarily for vLLM
                temp_model_dir = tempfile.mkdtemp(prefix="vllm_temp_model_")
                print(f"[GEN] Saving in-memory model temporarily for vLLM: {temp_model_dir}")
                
                # Move model to CPU to free GPU memory for vLLM
                if next(model.parameters()).is_cuda:
                    model_was_on_gpu = True
                    print(f"[GEN] Moving model to CPU to free GPU memory for vLLM")
                    model.to("cpu")
                    clear_gpu_memory()
                
                model.save_pretrained(temp_model_dir)
                tokenizer.save_pretrained(temp_model_dir)
                vllm_model_path = temp_model_dir
        
        use_vllm = vllm_model_path is not None
        
        if use_vllm:
            print(f"[GEN] Using vLLM for fast inference")
            responses = self._generate_with_vllm(
                model_path=vllm_model_path,
                prompts=prompts,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )
            
            # Clean up temporary model directory if we created one
            if temp_model_dir is not None:
                print(f"[GEN] Cleaning up temporary model directory")
                shutil.rmtree(temp_model_dir, ignore_errors=True)
                
                # Move model back to GPU if it was there before
                if model_was_on_gpu:
                    print(f"[GEN] Moving model back to GPU")
                    model.to(DEVICE)
            
            # Process responses and build training texts
            for rec, resp in zip(records, responses):
                resp = resp.strip()
                full_text = build_alpaca_full_text(
                    rec["instruction"], rec["input"], resp
                )
                all_texts.append(full_text)
                
                f_out.write(
                    json.dumps(
                        {
                            "instruction": rec["instruction"],
                            "input": rec["input"],
                            "response": resp,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        else:
            # Fall back to HuggingFace generation
            print(f"[GEN] vLLM not available, using HuggingFace")
            
            owns_model = False
            if model is None or tokenizer is None:
                if model_name_or_path is None:
                    raise ValueError("Either model/tokenizer or model_name_or_path must be provided")
                tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                tokenizer.padding_side = "left"

                model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
                model.to(DEVICE).eval()
                model.config.pad_token_id = tokenizer.pad_token_id
                owns_model = True
            else:
                # Ensure padding_side for generation
                try:
                    tokenizer.padding_side = "left"
                except Exception:
                    pass
                model.eval()
                model.config.pad_token_id = tokenizer.pad_token_id

            num_batches = (len(records) + batch_size - 1) // batch_size
            with torch.no_grad():
                for start in tqdm(range(0, len(records), batch_size), desc="Generating synthetic texts", total=num_batches, unit="batch"):
                    batch_recs = records[start : start + batch_size]
                    batch_prompts = prompts[start : start + batch_size]

                    enc = tokenizer(
                        batch_prompts,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=max_prompt_len,
                    ).to(DEVICE)

                    gen_ids = model.generate(
                        **enc,
                        max_new_tokens=max_new_tokens,
                        do_sample=True,
                        temperature=temperature,
                        top_p=top_p,
                        pad_token_id=tokenizer.pad_token_id,
                    )

                    attn = enc["attention_mask"]
                    for i, rec in enumerate(batch_recs):
                        prompt_len = int(attn[i].sum().item())
                        new_tokens = gen_ids[i, prompt_len:]
                        resp = tokenizer.decode(
                            new_tokens, skip_special_tokens=True
                        ).strip()

                        full_text = build_alpaca_full_text(
                            rec["instruction"], rec["input"], resp
                        )
                        all_texts.append(full_text)

                        f_out.write(
                            json.dumps(
                                {
                                    "instruction": rec["instruction"],
                                    "input": rec["input"],
                                    "response": resp,
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )

            if owns_model:
                del model
                del tokenizer
                clear_gpu_memory()

        f_out.close()
        print(f"[GEN] Saved raw generations to {raw_gen_path}")
        print(f"[GEN] Synthetic training samples: {len(all_texts)}")
        
        # Return model/tokenizer for reuse in fine-tuning
        if use_vllm:
            # If we used vLLM with an in-memory model (temp save), return original model/tokenizer
            # If we used vLLM with a path (no in-memory model), return None
            if temp_model_dir is not None:
                # We had in-memory model, return it for fine-tuning
                return all_texts, model, tokenizer
            else:
                return all_texts, None, None
        else:
            return all_texts, (None if owns_model else model), (None if owns_model else tokenizer)

    # -----------------------------
    # Fine-tune on synthetic texts
    # -----------------------------
    def _finetune_on_texts(
        self,
        base_model_path: Optional[str],
        texts: List[str],
        out_dir: str,
        max_steps: int,
        lr: float,
        batch_size: int,
        max_len: int,
        save_model: bool = True,
        model: Optional["AutoModelForCausalLM"] = None,
        tokenizer: Optional["AutoTokenizer"] = None,
    ) -> Tuple[Optional["AutoModelForCausalLM"], Optional["AutoTokenizer"]]:
        os.makedirs(out_dir, exist_ok=True)

        owns_model = False
        if model is None or tokenizer is None:
            if base_model_path is None:
                raise ValueError("Either base_model_path or (model, tokenizer) must be provided")
            tokenizer = AutoTokenizer.from_pretrained(base_model_path, use_fast=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = "right"

            model = AutoModelForCausalLM.from_pretrained(base_model_path)
            owns_model = True

        model.to(DEVICE)
        model.config.pad_token_id = tokenizer.pad_token_id
        if model.get_input_embeddings().num_embeddings != len(tokenizer):
            model.resize_token_embeddings(len(tokenizer))

        # canonical dataset builder
        train_ds = build_lm_dataset_from_texts(
            tokenizer=tokenizer,
            texts=texts,
            max_length=max_len,
        )
        data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

        args = TrainingArguments(
            output_dir=out_dir,
            overwrite_output_dir=True,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=1,
            learning_rate=lr,
            max_steps=max_steps,
            weight_decay=0.01,
            warmup_steps=int(0.1 * max_steps),
            logging_steps=50,
            # again: **no** optimizer/scheduler checkpoints
            save_strategy="no",
            report_to=["none"],
            fp16=(DEVICE == "cuda"),
        )

        trainer = NoSaveTrainer(
            model=model,
            args=args,
            train_dataset=train_ds,
            data_collator=data_collator,
        )

        print(f"[FT] Starting fine-tune from {base_model_path} -> {out_dir}")
        trainer.train()

        if save_model:
            # Manual minimal save: model + tokenizer
            model.save_pretrained(out_dir)
            tokenizer.save_pretrained(out_dir)
            print(f"[FT] Saved fine-tuned model to {out_dir}")

        # Cleanup
        del trainer
        del train_ds
        del data_collator
        if save_model and owns_model:
            del model
            del tokenizer
            clear_gpu_memory()
            return None, None
        else:
            return model, tokenizer

    # -----------------------------
    # Protocol interface
    # -----------------------------
    def finetune_one_generation(
        self,
        gen_idx: int,
        base_ckpt: Optional[str] = None,
        out_dir: Optional[str] = None,
        save_model: bool = True,
        model: Optional["AutoModelForCausalLM"] = None,
        tokenizer: Optional["AutoTokenizer"] = None,
    ) -> Union[str, Tuple["AutoModelForCausalLM", "AutoTokenizer"]]:
        """
        One inbreeding step:
          1) Generate synthetic texts with current model.
          2) Fine-tune on those texts to produce next-generation checkpoint.
        """
        if save_model and out_dir is None:
            raise ValueError("out_dir must be provided when save_model=True")

        # 1) synthetic dataset
        parent_dir = os.path.dirname(out_dir) if out_dir else "/tmp"
        synth_dir = os.path.join(parent_dir, f"gen{gen_idx}_synthetic")
        synthetic_texts, model_for_ft, tok_for_ft = self._generate_synthetic_alpaca_texts(
            model_name_or_path=base_ckpt,
            model=model,
            tokenizer=tokenizer,
            out_dir=synth_dir,
            batch_size=8,            # can be made configurable later
            max_prompt_len=256,
            max_new_tokens=128,
            temperature=0.7,
            top_p=0.9,
        )

        # 2) fine-tune on synthetic texts
        steps = (
            self.cfg.breeding.max_steps_first
            if gen_idx <= 3
            else self.cfg.breeding.max_steps_later
        )

        model_out, tok_out = self._finetune_on_texts(
            base_model_path=base_ckpt,
            texts=synthetic_texts,
            out_dir=out_dir if out_dir else "/tmp/collapse_train_temp",
            max_steps=steps,
            lr=self.cfg.breeding.lr,
            batch_size=2,
            max_len=self.cfg.breeding.train_max_seq_len,
            save_model=save_model,
            model=model_for_ft if model_for_ft is not None else model,
            tokenizer=tok_for_ft if tok_for_ft is not None else tokenizer,
        )

        if save_model:
            # Return the checkpoint path
            return out_dir
        else:
            # Keep the fine-tuned model in memory
            return model_out, tok_out
