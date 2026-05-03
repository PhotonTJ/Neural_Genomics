"""
ndna.core.spectral

Spectral Curvature Calculator.

Measures geometric curvature of the probability manifold through layers
using the logit lens technique and discrete geometry on the unit sphere.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np
import torch
from tqdm import tqdm

from .geometry import (
    sqrt_embed,
    discrete_curvature,
    free_memory,
    EPS_DIST,
    EPS_CURV,
)
from .results import SpectralResult, AggregatedSpectralResult

if TYPE_CHECKING:
    from ndna.models import ModelHandler


class SpectralCalculator:
    """
    Compute spectral curvature using logit lens and discrete geometry.
    
    Works on individual prompts/texts (not batched over dataset).
    This is memory-efficient as it processes one text at a time.
    
    The spectral curvature κ_ℓ measures how sharply the model's
    probability distribution changes direction at layer ℓ.
    
    Example:
        >>> from ndna.models import ModelHandler
        >>> handler = ModelHandler("gpt2")
        >>> calc = SpectralCalculator(handler)
        >>> result = calc.calculate_for_text("Hello, world!")
        >>> print(result.curvatures)
    """
    
    def __init__(
        self,
        model_handler: "ModelHandler",
        eps_dist: float = EPS_DIST,
        eps_curv: float = EPS_CURV,
    ):
        """
        Initialize SpectralCalculator.
        
        Args:
            model_handler: ModelHandler instance with loaded model
            eps_dist: Epsilon for probability clamping in sqrt embedding
            eps_curv: Epsilon for curvature denominator stability
        """
        self.model_handler = model_handler
        self.eps_dist = eps_dist
        self.eps_curv = eps_curv
    
    @torch.no_grad()
    def calculate_for_text(
        self,
        text: str,
        label: Optional[str] = None,
    ) -> SpectralResult:
        """
        Calculate spectral curvature for a single text.
        
        Process:
            1. Get layerwise distributions via logit lens at last token
            2. Convert to sqrt embeddings
            3. Compute discrete curvature at interior layers
        
        Args:
            text: Input text string
            label: Optional label for this text
        
        Returns:
            SpectralResult with curvatures, speeds, and statistics
        """
        # Get layerwise probability distributions
        # This returns [q_0, q_1, ..., q_L] where each q is (V,)
        q_list = self.model_handler.layerwise_distributions(text)
        
        if len(q_list) < 3:
            raise ValueError(
                f"Need at least 3 layers for curvature, got {len(q_list)}"
            )
        
        # Convert to sqrt embeddings on unit sphere
        u_list = []
        for q in q_list:
            u = sqrt_embed(q, eps=self.eps_dist)
            u_list.append(u)
        
        # Compute discrete curvature
        curvatures, speeds = discrete_curvature(u_list, eps_curv=self.eps_curv)
        
        # Clean up
        del q_list, u_list
        free_memory()
        
        # Create layer indices (1-indexed for interior layers)
        num_interior = len(curvatures)
        layer_indices = np.arange(1, num_interior + 1)
        
        return SpectralResult(
            curvatures=curvatures,
            speeds=speeds,
            layer_indices=layer_indices,
            text_preview=text[:100],
            label=label,
        )
    
    def calculate_for_prompts(
        self,
        prompts: List[Tuple[str, str]],
        progress_bar: bool = True,
    ) -> Dict[str, SpectralResult]:
        """
        Calculate spectral curvature for multiple labeled prompts.
        
        Args:
            prompts: List of (label, text) tuples
            progress_bar: Whether to show progress bar
        
        Returns:
            Dict mapping label to SpectralResult
        """
        results: Dict[str, SpectralResult] = {}
        
        iterator = prompts
        if progress_bar:
            iterator = tqdm(prompts, desc="Computing spectral curvature")
        
        for label, text in iterator:
            try:
                result = self.calculate_for_text(text, label=label)
                results[label] = result
            except Exception as e:
                print(f"Failed to process prompt '{label}': {e}")
                continue
        
        return results
    
    def calculate_for_texts(
        self,
        texts: List[str],
        progress_bar: bool = True,
    ) -> AggregatedSpectralResult:
        """
        Calculate spectral curvature for multiple texts and aggregate.
        
        Args:
            texts: List of text strings
            progress_bar: Whether to show progress bar
        
        Returns:
            AggregatedSpectralResult with mean, std, and individual results
        """
        individual_results: List[SpectralResult] = []
        
        iterator = enumerate(texts)
        if progress_bar:
            iterator = tqdm(list(iterator), desc="Computing spectral curvature")
        
        for idx, text in iterator:
            try:
                result = self.calculate_for_text(text, label=f"text_{idx}")
                individual_results.append(result)
            except Exception as e:
                print(f"Failed to process text {idx}: {e}")
                continue
        
        if not individual_results:
            raise ValueError("No texts were successfully processed")
        
        return AggregatedSpectralResult.from_results(individual_results)
    
    def calculate_from_prompts_set(
        self,
        prompt_set: str = "default",
        progress_bar: bool = True,
    ) -> Dict[str, SpectralResult]:
        """
        Calculate spectral curvature for a predefined prompt set.
        
        Args:
            prompt_set: Name of prompt set ("default", "multilingual", etc.)
            progress_bar: Whether to show progress bar
        
        Returns:
            Dict mapping label to SpectralResult
        """
        from ndna.data import get_prompts
        
        prompts = get_prompts(prompt_set)
        return self.calculate_for_prompts(prompts, progress_bar=progress_bar)

