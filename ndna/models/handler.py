"""
ndna.models.handler

Unified ModelHandler for transformer language models.
Provides a consistent interface for accessing model components across architectures.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from ndna.models.registry import detect_architecture, get_registry_config


class ModelHandler:
    """
    Unified handler for different transformer architectures.

    Provides consistent access to model components (layers, norms, heads)
    and core operations (hidden states, logit lens) across architectures.

    Supported: GPT-2, LLaMA, Qwen, Phi, DeepSeek, Mistral, Gemma, Falcon, etc.

    Example:
        >>> handler = ModelHandler("gpt2")
        >>> print(handler.num_layers)
        12
        >>> hidden_states = handler.get_hidden_states(input_ids, attention_mask)
    """

    def __init__(
        self,
        model_name: str,
        device: str = "auto",
        dtype: torch.dtype = torch.bfloat16,
        hf_token: Optional[str] = None,
        model_config: Optional[Dict[str, str]] = None,
    ):
        """
        Initialize ModelHandler with a HuggingFace model.

        Args:
            model_name: HuggingFace model identifier (e.g., "gpt2", "meta-llama/Llama-3.1-8B")
            device: Device for model ("cuda", "cpu", "auto")
            dtype: Model dtype (torch.bfloat16, torch.float16, torch.float32)
            hf_token: HuggingFace API token for gated models
            model_config: Optional custom architecture config to override registry
        """
        self.model_name = model_name
        self._dtype = dtype
        self._hf_token = hf_token

        # Resolve device
        if device == "auto":
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self._device = device

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            token=hf_token,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            token=hf_token,
            torch_dtype=dtype,
            device_map=self._device,
            trust_remote_code=True,
        )
        self.model.train()

        # Detect architecture and get config
        model_type = getattr(self.model.config, "model_type", None)
        self._architecture = detect_architecture(model_type, model_name)
        
        # Use custom config if provided, otherwise use registry
        if model_config is not None:
            self._config = model_config
        else:
            self._config = get_registry_config(self._architecture)

        # Resolve component references
        self._resolve_components()

    def _resolve_components(self) -> None:
        """Resolve model component references based on architecture config."""
        # Get the backbone/decoder module
        model_attr = self._config.get("model_attr", "model")
        self._backbone = self._get_nested_attr(self.model, model_attr)

        # Get layers
        layers_attr = self._config.get("layers_attr", "layers")
        self._layers = getattr(self._backbone, layers_attr)

        # Get final layer norm
        norm_attr = self._config.get("norm_attr", "norm")
        # Try backbone first, then model root
        if hasattr(self._backbone, norm_attr):
            self._final_norm = getattr(self._backbone, norm_attr)
        elif hasattr(self.model, norm_attr):
            self._final_norm = getattr(self.model, norm_attr)
        else:
            # Fallback: search common locations
            self._final_norm = self._find_final_norm()

        # Get LM head
        lm_head_attr = self._config.get("lm_head_attr", "lm_head")
        if hasattr(self.model, lm_head_attr):
            self._lm_head = getattr(self.model, lm_head_attr)
        else:
            # Try to construct tied head from embeddings
            self._lm_head = self._construct_tied_head()

        # Get embeddings
        embed_attr = self._config.get("embed_attr", "embed_tokens")
        if hasattr(self._backbone, embed_attr):
            self._embeddings = getattr(self._backbone, embed_attr)
        elif hasattr(self.model, "get_input_embeddings"):
            self._embeddings = self.model.get_input_embeddings()
        else:
            self._embeddings = None

    def _get_nested_attr(self, obj: object, attr_path: str) -> object:
        """Get nested attribute using dot notation (e.g., 'model.decoder')."""
        parts = attr_path.split(".")
        for part in parts:
            obj = getattr(obj, part)
        return obj

    def _find_final_norm(self) -> nn.Module:
        """Search for final layer norm in common locations."""
        candidates = [
            (self._backbone, "norm"),
            (self._backbone, "ln_f"),
            (self._backbone, "final_layer_norm"),
            (self._backbone, "final_layernorm"),
            (self.model, "norm"),
            (self.model, "ln_f"),
        ]
        for parent, attr in candidates:
            if hasattr(parent, attr):
                return getattr(parent, attr)
        raise AttributeError(
            f"Could not find final layer norm for architecture {self._architecture}"
        )

    def _construct_tied_head(self) -> nn.Module:
        """Construct a tied linear head from embeddings."""
        emb = self._embeddings or self.model.get_input_embeddings()
        if emb is None:
            raise AttributeError(
                f"Could not find or construct LM head for architecture {self._architecture}"
            )
        
        weight = emb.weight  # (vocab_size, hidden_size)
        vocab_size, hidden_size = weight.shape
        proj = nn.Linear(hidden_size, vocab_size, bias=False)
        proj.weight = weight  # Tie weights
        return proj

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    @property
    def architecture(self) -> str:
        """Detected architecture name (e.g., 'llama', 'gpt2', 'phi')."""
        return self._architecture

    @property
    def is_causal(self) -> bool:
        """Whether this is a causal (decoder-only) language model."""
        return True  # All supported models are causal LMs

    @property
    def num_layers(self) -> int:
        """Number of transformer layers/blocks."""
        return len(self._layers)

    @property
    def vocab_size(self) -> int:
        """Vocabulary size."""
        return self.model.config.vocab_size

    @property
    def hidden_size(self) -> int:
        """Hidden dimension size."""
        return self.model.config.hidden_size

    @property
    def device(self) -> str:
        """Device where model is loaded."""
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        """Model dtype."""
        return self._dtype

    # -------------------------------------------------------------------------
    # Component Access
    # -------------------------------------------------------------------------

    def get_layers(self) -> nn.ModuleList:
        """Get the list of transformer layers/blocks."""
        return self._layers

    def get_final_norm(self) -> nn.Module:
        """Get the final layer normalization module."""
        return self._final_norm

    def get_lm_head(self) -> nn.Module:
        """Get the language model head (output projection)."""
        return self._lm_head

    def get_embeddings(self) -> Optional[nn.Module]:
        """Get the input embeddings module."""
        return self._embeddings

    # -------------------------------------------------------------------------
    # Core Operations
    # -------------------------------------------------------------------------

    @torch.no_grad()
    def get_hidden_states(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, ...]:
        """
        Forward pass returning all hidden states including embedding layer.

        Args:
            input_ids: Token IDs (B, S)
            attention_mask: Attention mask (B, S), optional

        Returns:
            Tuple of hidden states from embedding through final layer.
            Length is num_layers + 1 (includes embedding output).
            Each tensor has shape (B, S, H).
        """
        input_ids = input_ids.to(self._device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self._device)

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )
        return outputs.hidden_states

    @torch.no_grad()
    def logit_lens(
        self,
        hidden_state: torch.Tensor,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """
        Apply logit lens: final_norm + lm_head to get logits from hidden state.

        Args:
            hidden_state: Hidden state tensor (..., H)
            temperature: Temperature for output (default 1.0, no scaling)

        Returns:
            Logits tensor (..., V)
        """
        normed = self._final_norm(hidden_state)
        logits = self._lm_head(normed)
        if temperature != 1.0:
            logits = logits / temperature
        return logits

    @torch.no_grad()
    def layerwise_distributions(
        self,
        text: str,
        position: int = -1,
    ) -> List[torch.Tensor]:
        """
        Get next-token probability distributions at each layer for a single text.

        Uses logit lens at the specified token position (default: last token).

        Args:
            text: Input text string
            position: Token position to extract distributions from (-1 = last)

        Returns:
            List of probability distributions [q_0, q_1, ..., q_L]
            where q_ℓ is (V,) probability distribution at layer ℓ.
            Length is num_layers + 1 (includes embedding layer).
        """
        # Tokenize
        enc = self.tokenizer(
            text,
            return_tensors="pt",
            add_special_tokens=False,
        ).to(self._device)

        if enc.input_ids.shape[1] == 0:
            raise ValueError("Empty tokenized input.")

        # Forward pass
        hidden_states = self.get_hidden_states(
            enc.input_ids,
            enc.attention_mask,
        )

        # Get position index
        seq_len = enc.input_ids.shape[1]
        if position < 0:
            pos_idx = seq_len + position
        else:
            pos_idx = position

        if pos_idx < 0 or pos_idx >= seq_len:
            raise ValueError(f"Position {position} out of range for sequence length {seq_len}")

        # Extract distributions at each layer
        q_list: List[torch.Tensor] = []
        for h in hidden_states:
            # h is (B, S, H), take position pos_idx
            h_pos = h[0, pos_idx, :]  # (H,)
            logits = self.logit_lens(h_pos)  # (V,)
            q = torch.softmax(logits.float(), dim=-1)  # (V,)
            q_list.append(q)

        return q_list

    @torch.no_grad()
    def batch_logit_lens(
        self,
        hidden_states: Tuple[torch.Tensor, ...],
        layer_indices: Optional[List[int]] = None,
    ) -> List[torch.Tensor]:
        """
        Apply logit lens to multiple layers' hidden states.

        Args:
            hidden_states: Tuple of hidden states from get_hidden_states()
            layer_indices: Which layers to process (None = all)

        Returns:
            List of logit tensors, one per requested layer
        """
        if layer_indices is None:
            layer_indices = list(range(len(hidden_states)))

        logits_list = []
        for idx in layer_indices:
            h = hidden_states[idx]
            logits = self.logit_lens(h)
            logits_list.append(logits)

        return logits_list

    def __repr__(self) -> str:
        return (
            f"ModelHandler("
            f"model_name={self.model_name!r}, "
            f"architecture={self._architecture!r}, "
            f"num_layers={self.num_layers}, "
            f"vocab_size={self.vocab_size}, "
            f"device={self._device!r})"
        )

