"""
ndna.core.thermodynamic

Thermodynamic Length Calculator.

Measures Fisher-Rao distance between consecutive layer predictions.
Uses memory-efficient streaming computation to avoid OOM errors.
"""

from __future__ import annotations

from typing import List, Optional, TYPE_CHECKING

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from .geometry import (
    fisher_rao_distance,
    free_memory,
    get_device,
    get_amp_dtype,
)
from .results import ThermoResult, ThermoResultPerSample

if TYPE_CHECKING:
    from ndna.models import ModelHandler


class ThermodynamicCalculator:
    """
    Compute thermodynamic length using Fisher-Rao geometry.
    
    Works on batched dataset - aggregates FR distances over all valid tokens.
    
    Memory Optimization Strategy:
    - Streaming layer processing: process one layer at a time
    - Sliding window: only keep logp_prev and logp_curr
    - Running sum accumulators: don't store per-sample data
    - Explicit memory cleanup after each batch
    
    Example:
        >>> from ndna.models import ModelHandler
        >>> from ndna.data import DatasetHandler
        >>> handler = ModelHandler("gpt2")
        >>> data = DatasetHandler(handler.tokenizer, batch_size=2)
        >>> data.load_texts(["Hello world", "Test text"])
        >>> loader = data.create_dataloader(collate_type="causal")
        >>> calc = ThermodynamicCalculator(handler)
        >>> result = calc.calculate(loader)
        >>> print(result.step_lengths)
    """
    
    def __init__(
        self,
        model_handler: "ModelHandler",
    ):
        """
        Initialize ThermodynamicCalculator.
        
        Args:
            model_handler: ModelHandler instance with loaded model
        """
        self.model_handler = model_handler
        self.device = model_handler.device
        self.num_layers = model_handler.num_layers
    
    @torch.no_grad()
    def calculate(
        self,
        dataloader: DataLoader,
        progress_bar: bool = True,
    ) -> ThermoResult:
        """
        Calculate Fisher-Rao thermodynamic length over dataset.
        
        Memory-efficient implementation using streaming layer processing.
        
        Process:
            1. For each batch, get hidden states via single forward pass
            2. Process layers with sliding window (only keep prev/curr)
            3. Compute FR distance between consecutive layers
            4. Accumulate sums (not raw data) for mean calculation
            5. Explicit cleanup after each layer and batch
        
        Args:
            dataloader: DataLoader yielding batches with input_ids, attention_mask, labels
            progress_bar: Whether to show progress bar
        
        Returns:
            ThermoResult with per-step mean FR lengths
        """
        model = self.model_handler.model
        model.eval()
        
        # Disable KV cache for memory efficiency
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False
        
        num_steps = self.num_layers  # Transitions between L+1 hidden states
        
        # Running sum accumulators (not storing per-sample data)
        step_sums = torch.zeros(num_steps, device=self.device, dtype=torch.float64)
        step_counts = torch.zeros(num_steps, device=self.device, dtype=torch.float64)
        
        total_samples = 0
        total_tokens = 0
        
        # Get AMP dtype
        amp_dtype = get_amp_dtype()
        amp_enabled = self.device == "cuda"
        
        iterator = dataloader
        if progress_bar:
            iterator = tqdm(dataloader, desc="Computing thermodynamic length")
        
        for batch in iterator:
            input_ids = batch["input_ids"].to(self.device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(self.device, non_blocking=True)
            labels = batch["labels"].to(self.device, non_blocking=True)
            
            B, S = input_ids.shape
            total_samples += B
            
            # Valid positions are where labels != -100
            valid_mask = labels != -100
            batch_valid_tokens = valid_mask.sum().item()
            total_tokens += batch_valid_tokens
            
            # Single forward pass with all hidden states
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_enabled):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    use_cache=False,
                )
            
            hidden_states = outputs.hidden_states  # (L+1) tensors of (B, S, H)
            
            # Process layers with sliding window
            logp_prev = None
            
            for ell in range(len(hidden_states)):
                # Get hidden state for this layer
                h = hidden_states[ell]
                
                # Apply logit lens to get logits
                with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_enabled):
                    logits = self.model_handler.logit_lens(h)
                
                # Convert to log probabilities (float32 for stability)
                logp_curr = F.log_softmax(logits.float(), dim=-1)
                
                # Clean up logits immediately
                del logits
                
                if logp_prev is not None:
                    # Compute FR distance between consecutive layers
                    fr_step = fisher_rao_distance(logp_prev, logp_curr)  # (B, S)
                    
                    # Mask invalid positions
                    fr_step = fr_step.masked_fill(~valid_mask, 0.0)
                    
                    # Accumulate sums
                    step_idx = ell - 1
                    step_sums[step_idx] += fr_step.double().sum()
                    step_counts[step_idx] += valid_mask.sum()
                
                # Slide window: delete previous, keep current
                del logp_prev
                logp_prev = logp_curr
                
                # Free memory after each layer
                free_memory()
            
            # Clean up after batch
            del logp_prev, hidden_states, outputs
            del input_ids, attention_mask, labels, valid_mask
            free_memory()
        
        # Compute mean FR length per step
        step_lengths = (step_sums / step_counts.clamp_min(1)).cpu().numpy()
        
        # Create step indices (1-indexed)
        step_indices = np.arange(1, num_steps + 1)
        
        return ThermoResult(
            step_lengths=step_lengths,
            step_indices=step_indices,
            total_length=float(step_lengths.sum()),
            num_samples_processed=total_samples,
            num_tokens_processed=total_tokens,
        )
    
    @torch.no_grad()
    def calculate_per_sample(
        self,
        dataloader: DataLoader,
        max_samples: int = 50,
        progress_bar: bool = True,
    ) -> ThermoResultPerSample:
        """
        Calculate per-sample FR lengths for 3D visualization.
        
        WARNING: This stores per-sample data and uses more memory.
        Use only with small max_samples for visualization.
        
        Args:
            dataloader: DataLoader yielding batches
            max_samples: Maximum number of samples to collect
            progress_bar: Whether to show progress bar
        
        Returns:
            ThermoResultPerSample with per-sample lengths
        """
        model = self.model_handler.model
        model.eval()
        
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False
        
        num_steps = self.num_layers
        
        # Collect per-sample data (limited to max_samples)
        sample_lengths: List[np.ndarray] = []
        
        amp_dtype = get_amp_dtype()
        amp_enabled = self.device == "cuda"
        
        iterator = dataloader
        if progress_bar:
            iterator = tqdm(dataloader, desc="Computing per-sample FR lengths")
        
        samples_collected = 0
        
        for batch in iterator:
            if samples_collected >= max_samples:
                break
            
            input_ids = batch["input_ids"].to(self.device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(self.device, non_blocking=True)
            labels = batch["labels"].to(self.device, non_blocking=True)
            
            B, S = input_ids.shape
            valid_mask = labels != -100
            
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_enabled):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    use_cache=False,
                )
            
            hidden_states = outputs.hidden_states
            
            # Process each sample in batch
            for b in range(B):
                if samples_collected >= max_samples:
                    break
                
                sample_step_lengths = []
                logp_prev = None
                
                for ell in range(len(hidden_states)):
                    h = hidden_states[ell][b:b+1]
                    
                    with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_enabled):
                        logits = self.model_handler.logit_lens(h)
                    
                    logp_curr = F.log_softmax(logits.float(), dim=-1)
                    del logits
                    
                    if logp_prev is not None:
                        fr_step = fisher_rao_distance(logp_prev, logp_curr)
                        
                        # Average over valid positions for this sample
                        sample_valid = valid_mask[b:b+1]
                        valid_count = sample_valid.sum().item()
                        
                        if valid_count > 0:
                            mean_step = (fr_step * sample_valid.float()).sum().item() / valid_count
                        else:
                            mean_step = 0.0
                        
                        sample_step_lengths.append(mean_step)
                    
                    del logp_prev
                    logp_prev = logp_curr
                
                del logp_prev
                sample_lengths.append(np.array(sample_step_lengths))
                samples_collected += 1
            
            del hidden_states, outputs
            del input_ids, attention_mask, labels, valid_mask
            free_memory()
        
        # Stack into array
        per_sample_lengths = np.array(sample_lengths)
        step_indices = np.arange(1, num_steps + 1)
        
        return ThermoResultPerSample.from_sample_data(
            per_sample_lengths=per_sample_lengths,
            step_indices=step_indices,
        )

