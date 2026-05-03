"""
ndna.core.belief

Belief Vector Field Calculator.

Measures tangent vectors representing belief changes with respect to targets.
Uses memory-efficient layer-by-layer processing with CPU accumulators.
"""

from __future__ import annotations

from typing import List, Optional, TYPE_CHECKING

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from .geometry import (
    free_memory,
    get_device,
    get_amp_dtype,
    EPS_DIST,
)
from .results import BeliefResult, BeliefResultPerSample

if TYPE_CHECKING:
    from ndna.models import ModelHandler


class BeliefCalculator:
    """
    Compute belief vector field - tangent vectors in probability space.
    
    Works on batched dataset with target labels.
    
    Memory Optimization Strategy:
    - Per-layer accumulators: v_sum[L], u_sum[L], cnt[L]
    - Process one layer at a time within each batch
    - Delete hidden states after processing each layer
    - Use belief collate to limit positions per sample
    
    The belief vector v_ℓ at layer ℓ represents how the model's
    belief about the target token changes in the tangent space.
    
    Example:
        >>> from ndna.models import ModelHandler
        >>> from ndna.data import DatasetHandler
        >>> handler = ModelHandler("gpt2")
        >>> data = DatasetHandler(handler.tokenizer, batch_size=1)
        >>> data.load_texts(["Hello world"])
        >>> loader = data.create_dataloader(collate_type="belief", keep_last_k=32)
        >>> calc = BeliefCalculator(handler)
        >>> result = calc.calculate(loader)
        >>> print(result.belief_norms)
    """
    
    def __init__(
        self,
        model_handler: "ModelHandler",
        tau: float = 1.0,
        fr_norm: bool = True,
    ):
        """
        Initialize BeliefCalculator.
        
        Args:
            model_handler: ModelHandler instance with loaded model
            tau: Temperature for softmax (default 1.0)
            fr_norm: If True, report FR norm 2*||v||; else Euclidean ||v||
        """
        self.model_handler = model_handler
        self.tau = tau
        self.fr_norm = fr_norm
        self.device = model_handler.device
        self.num_layers = model_handler.num_layers
        self.vocab_size = model_handler.vocab_size
    
    @torch.no_grad()
    def calculate(
        self,
        dataloader: DataLoader,
        progress_bar: bool = True,
    ) -> BeliefResult:
        """
        Calculate belief vector field magnitudes over dataset.
        
        Memory-efficient implementation with layer-by-layer processing.
        
        Process:
            1. For each layer, compute logits via logit lens
            2. Compute gradient of log-likelihood: g = (1/τ)(e_y - q)
            3. Convert to tangent vector: t = (1/2τ)(u*g - ⟨q,g⟩*u)
            4. Accumulate sums of t and u vectors
            5. Project mean t to tangent space at mean u, report norm
        
        Args:
            dataloader: DataLoader with belief collate (has select_mask)
            progress_bar: Whether to show progress bar
        
        Returns:
            BeliefResult with per-layer belief vector magnitudes
        """
        model = self.model_handler.model
        model.eval()
        
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False
        
        # Per-layer accumulators (V,) each
        # Keep on GPU during batch processing, move to CPU after
        v_sums = [
            torch.zeros(self.vocab_size, device=self.device, dtype=torch.float32)
            for _ in range(self.num_layers)
        ]
        u_sums = [
            torch.zeros(self.vocab_size, device=self.device, dtype=torch.float32)
            for _ in range(self.num_layers)
        ]
        counts = [0 for _ in range(self.num_layers)]
        
        total_samples = 0
        total_tokens = 0
        
        amp_dtype = get_amp_dtype()
        amp_enabled = self.device == "cuda"
        
        iterator = dataloader
        if progress_bar:
            iterator = tqdm(dataloader, desc="Computing belief vector field")
        
        for batch in iterator:
            input_ids = batch["input_ids"].to(self.device, non_blocking=True)
            labels = batch["labels"].to(self.device, non_blocking=True)
            
            # select_mask indicates which positions to use (last K per sample)
            if "select_mask" in batch:
                select_mask = batch["select_mask"].to(self.device, non_blocking=True)
            else:
                # Fall back to using all non-padding positions
                select_mask = labels != -100
            
            # Build attention mask from input_ids
            if hasattr(self.model_handler.tokenizer, "pad_token_id"):
                pad_token_id = self.model_handler.tokenizer.pad_token_id
                if pad_token_id is not None:
                    attention_mask = (input_ids != pad_token_id).long()
                else:
                    attention_mask = torch.ones_like(input_ids)
            else:
                attention_mask = torch.ones_like(input_ids)
            
            B, S = input_ids.shape
            total_samples += B
            
            # Single forward pass
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_enabled):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    use_cache=False,
                )
            
            # Skip embedding layer - use hidden_states[1:] which are after each block
            hidden_states = outputs.hidden_states[1:]
            
            # Process each layer
            for ell in range(self.num_layers):
                h_ell = hidden_states[ell]  # (B, S, H)
                
                # Apply logit lens
                with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_enabled):
                    z = self.model_handler.logit_lens(h_ell)  # (B, S, V)
                
                # Get selected positions
                keep = select_mask.view(-1)  # (B*S,)
                
                if not keep.any():
                    del z
                    continue
                
                # Select relevant positions
                z_sel = z.view(-1, z.size(-1))[keep]  # (N_sel, V)
                y_sel = labels.view(-1)[keep]  # (N_sel,)
                
                del z
                
                # Compute probabilities with temperature (float32 for stability)
                z32 = z_sel.float() / self.tau
                q = F.softmax(z32, dim=-1)
                q = torch.clamp(q, min=EPS_DIST)
                
                del z32
                
                # Sqrt embedding
                u = torch.sqrt(q)  # (N_sel, V)
                
                # Compute gradient g = (1/τ)(e_y - q)
                g = -q / self.tau  # (N_sel, V)
                
                # Add 1/tau at target positions
                add_values = torch.full(
                    (y_sel.shape[0], 1),
                    1.0 / self.tau,
                    device=self.device,
                    dtype=q.dtype,
                )
                g.scatter_add_(-1, y_sel.unsqueeze(-1), add_values)
                
                # Compute tangent vector t = (1/2τ)(u*g - ⟨q,g⟩*u)
                ug = u * g  # (N_sel, V)
                qg = (q * g).sum(dim=-1, keepdim=True)  # (N_sel, 1)
                t = (ug - qg * u) / (2.0 * self.tau)  # (N_sel, V)
                
                del ug, qg, g
                
                # Accumulate sums
                v_sums[ell] += t.sum(dim=0)
                u_sums[ell] += u.sum(dim=0)
                counts[ell] += t.size(0)
                total_tokens += t.size(0)
                
                del t, u, q, z_sel, y_sel
                free_memory()
            
            # Clean up after batch
            del hidden_states, outputs
            del input_ids, labels, select_mask, attention_mask
            free_memory()
        
        # Compute final norms
        norms = []
        for ell in range(self.num_layers):
            if counts[ell] == 0:
                norms.append(0.0)
                continue
            
            # Mean vectors
            v_avg = v_sums[ell] / counts[ell]  # (V,)
            u_avg = u_sums[ell] / counts[ell]  # (V,)
            
            # Normalize u to unit sphere (anchor base point)
            u_norm = torch.linalg.norm(u_avg).clamp_min(1e-12)
            u_bar = u_avg / u_norm
            
            # Project v_avg to tangent space at u_bar: remove radial component
            radial = torch.dot(u_bar, v_avg)
            v_tan = v_avg - radial * u_bar
            
            # Compute norm
            n = torch.linalg.norm(v_tan).item()
            
            # Apply FR norm factor if requested
            if self.fr_norm:
                n *= 2.0
            
            norms.append(n)
        
        # Create layer indices (1-indexed)
        layer_indices = np.arange(1, self.num_layers + 1)
        
        return BeliefResult(
            belief_norms=np.array(norms, dtype=np.float64),
            layer_indices=layer_indices,
            num_samples_processed=total_samples,
            num_tokens_processed=total_tokens,
            fr_norm=self.fr_norm,
        )
    
    @torch.no_grad()
    def calculate_per_sample(
        self,
        dataloader: DataLoader,
        max_samples: int = 50,
        progress_bar: bool = True,
    ) -> BeliefResultPerSample:
        """
        Calculate per-sample belief magnitudes for 3D visualization.
        
        WARNING: This stores per-sample data and uses more memory.
        Use only with small max_samples for visualization.
        
        Args:
            dataloader: DataLoader with belief collate
            max_samples: Maximum number of samples to collect
            progress_bar: Whether to show progress bar
        
        Returns:
            BeliefResultPerSample with per-sample norms
        """
        model = self.model_handler.model
        model.eval()
        
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False
        
        # Collect per-sample data
        sample_norms: List[np.ndarray] = []
        
        amp_dtype = get_amp_dtype()
        amp_enabled = self.device == "cuda"
        
        iterator = dataloader
        if progress_bar:
            iterator = tqdm(dataloader, desc="Computing per-sample belief norms")
        
        samples_collected = 0
        
        for batch in iterator:
            if samples_collected >= max_samples:
                break
            
            input_ids = batch["input_ids"].to(self.device, non_blocking=True)
            labels = batch["labels"].to(self.device, non_blocking=True)
            
            if "select_mask" in batch:
                select_mask = batch["select_mask"].to(self.device, non_blocking=True)
            else:
                select_mask = labels != -100
            
            if hasattr(self.model_handler.tokenizer, "pad_token_id"):
                pad_token_id = self.model_handler.tokenizer.pad_token_id
                if pad_token_id is not None:
                    attention_mask = (input_ids != pad_token_id).long()
                else:
                    attention_mask = torch.ones_like(input_ids)
            else:
                attention_mask = torch.ones_like(input_ids)
            
            B, S = input_ids.shape
            
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_enabled):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    use_cache=False,
                )
            
            hidden_states = outputs.hidden_states[1:]
            
            # Process each sample
            for b in range(B):
                if samples_collected >= max_samples:
                    break
                
                sample_layer_norms = []
                
                for ell in range(self.num_layers):
                    h = hidden_states[ell][b:b+1]  # (1, S, H)
                    
                    with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_enabled):
                        z = self.model_handler.logit_lens(h)  # (1, S, V)
                    
                    keep = select_mask[b].view(-1)
                    
                    if not keep.any():
                        sample_layer_norms.append(0.0)
                        del z
                        continue
                    
                    z_sel = z.view(-1, z.size(-1))[keep]
                    y_sel = labels[b].view(-1)[keep]
                    
                    del z
                    
                    z32 = z_sel.float() / self.tau
                    q = F.softmax(z32, dim=-1)
                    q = torch.clamp(q, min=EPS_DIST)
                    del z32
                    
                    u = torch.sqrt(q)
                    
                    g = -q / self.tau
                    add_values = torch.full(
                        (y_sel.shape[0], 1),
                        1.0 / self.tau,
                        device=self.device,
                        dtype=q.dtype,
                    )
                    g.scatter_add_(-1, y_sel.unsqueeze(-1), add_values)
                    
                    ug = u * g
                    qg = (q * g).sum(dim=-1, keepdim=True)
                    t = (ug - qg * u) / (2.0 * self.tau)
                    
                    del ug, qg, g
                    
                    # Mean for this sample at this layer
                    v_avg = t.mean(dim=0)
                    u_avg = u.mean(dim=0)
                    
                    u_norm = torch.linalg.norm(u_avg).clamp_min(1e-12)
                    u_bar = u_avg / u_norm
                    
                    radial = torch.dot(u_bar, v_avg)
                    v_tan = v_avg - radial * u_bar
                    
                    n = torch.linalg.norm(v_tan).item()
                    if self.fr_norm:
                        n *= 2.0
                    
                    sample_layer_norms.append(n)
                    
                    del t, u, q, z_sel, y_sel, v_avg, u_avg, u_bar, v_tan
                
                sample_norms.append(np.array(sample_layer_norms))
                samples_collected += 1
            
            del hidden_states, outputs
            del input_ids, labels, select_mask, attention_mask
            free_memory()
        
        per_sample_norms = np.array(sample_norms)
        layer_indices = np.arange(1, self.num_layers + 1)
        
        return BeliefResultPerSample.from_sample_data(
            per_sample_norms=per_sample_norms,
            layer_indices=layer_indices,
        )

