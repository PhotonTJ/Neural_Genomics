"""
ndna.models.logit_lens

Logit lens utilities for probing intermediate layer representations.

The "logit lens" technique applies the final layer norm and LM head to
intermediate hidden states, allowing us to see what the model would predict
if it had to output from that layer.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@torch.no_grad()
def logit_lens(
    hidden_state: torch.Tensor,
    final_norm: nn.Module,
    lm_head: nn.Module,
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    Apply logit lens to a hidden state.

    Transforms hidden state through final layer norm and LM head to get logits,
    as if the model were outputting from this layer.

    Args:
        hidden_state: Hidden state tensor (..., H) from any layer
        final_norm: Final layer normalization module
        lm_head: Language model head (output projection)
        temperature: Temperature for logit scaling (default 1.0)

    Returns:
        Logits tensor (..., V)
    """
    normed = final_norm(hidden_state)
    logits = lm_head(normed)
    if temperature != 1.0:
        logits = logits / temperature
    return logits


@torch.no_grad()
def logit_lens_probs(
    hidden_state: torch.Tensor,
    final_norm: nn.Module,
    lm_head: nn.Module,
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    Apply logit lens and convert to probabilities.

    Args:
        hidden_state: Hidden state tensor (..., H)
        final_norm: Final layer normalization module
        lm_head: Language model head
        temperature: Temperature for softmax

    Returns:
        Probability distribution (..., V)
    """
    logits = logit_lens(hidden_state, final_norm, lm_head, temperature)
    return F.softmax(logits.float(), dim=-1)


@torch.no_grad()
def logit_lens_log_probs(
    hidden_state: torch.Tensor,
    final_norm: nn.Module,
    lm_head: nn.Module,
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    Apply logit lens and convert to log probabilities.

    Args:
        hidden_state: Hidden state tensor (..., H)
        final_norm: Final layer normalization module
        lm_head: Language model head
        temperature: Temperature for softmax

    Returns:
        Log probability distribution (..., V)
    """
    logits = logit_lens(hidden_state, final_norm, lm_head, temperature)
    return F.log_softmax(logits.float(), dim=-1)


@torch.no_grad()
def batch_logit_lens(
    hidden_states: Tuple[torch.Tensor, ...],
    final_norm: nn.Module,
    lm_head: nn.Module,
    layer_indices: Optional[List[int]] = None,
    temperature: float = 1.0,
) -> List[torch.Tensor]:
    """
    Apply logit lens to multiple layers' hidden states.

    Args:
        hidden_states: Tuple of hidden states from model forward pass
        final_norm: Final layer normalization module
        lm_head: Language model head
        layer_indices: Which layers to process (None = all)
        temperature: Temperature for logit scaling

    Returns:
        List of logit tensors, one per requested layer
    """
    if layer_indices is None:
        layer_indices = list(range(len(hidden_states)))

    logits_list = []
    for idx in layer_indices:
        h = hidden_states[idx]
        logits = logit_lens(h, final_norm, lm_head, temperature)
        logits_list.append(logits)

    return logits_list


@torch.no_grad()
def layerwise_predictions(
    hidden_states: Tuple[torch.Tensor, ...],
    final_norm: nn.Module,
    lm_head: nn.Module,
    position: int = -1,
) -> List[torch.Tensor]:
    """
    Get token predictions at each layer for a specific position.

    Args:
        hidden_states: Tuple of hidden states (B, S, H) from forward pass
        final_norm: Final layer normalization module
        lm_head: Language model head
        position: Token position to analyze (-1 = last)

    Returns:
        List of probability distributions, one per layer.
        Each tensor has shape (V,) for batch size 1, or (B, V) for larger batches.
    """
    q_list: List[torch.Tensor] = []
    
    for h in hidden_states:
        # h is (B, S, H)
        batch_size, seq_len, _ = h.shape
        
        # Handle negative indices
        if position < 0:
            pos_idx = seq_len + position
        else:
            pos_idx = position
            
        if pos_idx < 0 or pos_idx >= seq_len:
            raise ValueError(f"Position {position} out of range for sequence length {seq_len}")
        
        # Extract at position
        h_pos = h[:, pos_idx, :]  # (B, H)
        probs = logit_lens_probs(h_pos, final_norm, lm_head)  # (B, V)
        
        # Squeeze if single batch
        if batch_size == 1:
            probs = probs.squeeze(0)  # (V,)
            
        q_list.append(probs)
    
    return q_list


@torch.no_grad()
def compare_layer_predictions(
    hidden_states: Tuple[torch.Tensor, ...],
    final_norm: nn.Module,
    lm_head: nn.Module,
    position: int = -1,
    top_k: int = 5,
    tokenizer=None,
) -> List[dict]:
    """
    Compare top-k predictions across layers for analysis.

    Args:
        hidden_states: Tuple of hidden states from forward pass
        final_norm: Final layer normalization module
        lm_head: Language model head
        position: Token position to analyze
        top_k: Number of top predictions to return
        tokenizer: Optional tokenizer for decoding tokens

    Returns:
        List of dicts, one per layer, each containing:
            - top_tokens: Top-k token IDs
            - top_probs: Top-k probabilities
            - top_decoded: Top-k decoded tokens (if tokenizer provided)
    """
    q_list = layerwise_predictions(hidden_states, final_norm, lm_head, position)
    
    results = []
    for layer_idx, probs in enumerate(q_list):
        # Get top-k
        top_probs, top_tokens = torch.topk(probs, k=min(top_k, probs.shape[-1]))
        
        layer_result = {
            "layer": layer_idx,
            "top_tokens": top_tokens.cpu().tolist(),
            "top_probs": top_probs.cpu().tolist(),
        }
        
        if tokenizer is not None:
            if isinstance(top_tokens, torch.Tensor):
                token_ids = top_tokens.cpu().tolist()
            else:
                token_ids = top_tokens
            layer_result["top_decoded"] = [
                tokenizer.decode([tid]) for tid in token_ids
            ]
        
        results.append(layer_result)
    
    return results

