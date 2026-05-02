"""
ndna.data.collate

Collate functions for creating batches for nDNA metrics.

- causal_collate: Teacher-forced next-token prediction (for thermodynamic length)
- belief_collate: Keep last K tokens per sample (for belief vector field)
"""

from typing import Callable, Dict, List

import torch


def make_causal_collate(
    tokenizer,
    max_length: int = 256,
) -> Callable[[List[Dict]], Dict[str, torch.Tensor]]:
    """
    Create a causal LM collate function for teacher-forced training.

    The collate function:
    1. Tokenizes texts with padding and truncation
    2. Shifts for next-token prediction: inputs[:-1], labels[1:]
    3. Masks padding tokens in labels with -100

    Args:
        tokenizer: HuggingFace tokenizer
        max_length: Maximum sequence length

    Returns:
        Collate function for DataLoader

    Example:
        >>> collate_fn = make_causal_collate(tokenizer, max_length=256)
        >>> loader = DataLoader(dataset, collate_fn=collate_fn)
        >>> for batch in loader:
        ...     input_ids = batch["input_ids"]      # (B, S-1)
        ...     labels = batch["labels"]            # (B, S-1)
        ...     attention_mask = batch["attention_mask"]  # (B, S-1)
    """

    def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
        texts = [ex["text"] for ex in batch]

        tok = tokenizer(
            texts,
            padding="longest",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

        input_ids = tok["input_ids"]  # (B, S)
        attention_mask = tok["attention_mask"]  # (B, S)

        # Shift for next-token prediction
        # inputs: tokens 0 to S-2
        # labels: tokens 1 to S-1
        input_ids_shifted = input_ids[:, :-1].contiguous()
        attention_shifted = attention_mask[:, :-1].contiguous()
        labels_shifted = input_ids[:, 1:].contiguous()

        # Mask padding tokens in labels (where original attention was 0)
        labels_shifted = labels_shifted.masked_fill(
            attention_mask[:, 1:] == 0, -100
        )

        return {
            "input_ids": input_ids_shifted,
            "attention_mask": attention_shifted,
            "labels": labels_shifted,
        }

    return collate_fn


def make_belief_collate(
    tokenizer,
    max_length: int = 256,
    keep_last_k: int = 32,
) -> Callable[[List[Dict]], Dict[str, torch.Tensor]]:
    """
    Create a collate function for belief vector field computation.

    Similar to causal collate, but additionally creates a select_mask
    indicating the last K valid positions per sample where belief
    vectors should be computed.

    Args:
        tokenizer: HuggingFace tokenizer
        max_length: Maximum sequence length
        keep_last_k: Number of last supervised positions to keep per sample

    Returns:
        Collate function for DataLoader

    Example:
        >>> collate_fn = make_belief_collate(tokenizer, max_length=128, keep_last_k=32)
        >>> loader = DataLoader(dataset, collate_fn=collate_fn)
        >>> for batch in loader:
        ...     input_ids = batch["input_ids"]        # (B, S-1)
        ...     labels = batch["labels"]              # (B, S-1)
        ...     attention_mask = batch["attention_mask"]  # (B, S-1)
        ...     select_mask = batch["select_mask"]    # (B, S-1), True for last K positions
    """

    def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
        texts = [ex["text"] for ex in batch]

        tok = tokenizer(
            texts,
            padding="longest",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

        input_ids = tok["input_ids"]  # (B, S)
        attention_mask = tok["attention_mask"]  # (B, S)

        # Shift for next-token prediction
        x = input_ids[:, :-1].contiguous()
        y = input_ids[:, 1:].contiguous()
        msk = attention_mask[:, :-1].contiguous()

        # Mask padding in labels
        y = y.masked_fill(attention_mask[:, 1:] == 0, -100)

        # Create select_mask: True for last K valid positions per sample
        B, _ = x.shape
        select_mask = torch.zeros_like(y, dtype=torch.bool)

        for b in range(B):
            # Find valid positions (where label is not -100)
            valid_positions = (y[b] != -100).nonzero(as_tuple=False).squeeze(-1)
            if valid_positions.numel() > 0:
                # Take last K positions
                num_to_take = min(keep_last_k, valid_positions.numel())
                positions_to_select = valid_positions[-num_to_take:]
                select_mask[b, positions_to_select] = True

        return {
            "input_ids": x,
            "attention_mask": msk,
            "labels": y,
            "select_mask": select_mask,
        }

    return collate_fn


def causal_collate(
    batch: List[Dict],
    tokenizer,
    max_length: int = 256,
) -> Dict[str, torch.Tensor]:
    """
    Standalone causal collate function (alternative to factory pattern).

    Args:
        batch: List of examples with "text" key
        tokenizer: HuggingFace tokenizer
        max_length: Maximum sequence length

    Returns:
        Dict with input_ids, attention_mask, labels
    """
    return make_causal_collate(tokenizer, max_length)(batch)


def belief_collate(
    batch: List[Dict],
    tokenizer,
    max_length: int = 256,
    keep_last_k: int = 32,
) -> Dict[str, torch.Tensor]:
    """
    Standalone belief collate function (alternative to factory pattern).

    Args:
        batch: List of examples with "text" key
        tokenizer: HuggingFace tokenizer
        max_length: Maximum sequence length
        keep_last_k: Number of last positions to keep

    Returns:
        Dict with input_ids, attention_mask, labels, select_mask
    """
    return make_belief_collate(tokenizer, max_length, keep_last_k)(batch)

