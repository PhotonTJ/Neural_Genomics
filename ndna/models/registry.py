"""
ndna.models.registry

Architecture registry for transformer models.
Maps model architectures to their component paths for unified access.
"""

from typing import Dict, Optional

# ---------------------------------------------------------------------
# Model Registry
# ---------------------------------------------------------------------
# Maps architecture name to component attribute paths
MODEL_REGISTRY: Dict[str, Dict[str, str]] = {
    "llama": {
        "model_attr": "model",
        "layers_attr": "layers",
        "norm_attr": "norm",
        "lm_head_attr": "lm_head",
        "embed_attr": "embed_tokens",
    },
    "gpt2": {
        "model_attr": "transformer",
        "layers_attr": "h",
        "norm_attr": "ln_f",
        "lm_head_attr": "lm_head",
        "embed_attr": "wte",
        "pos_embed_attr": "wpe",
    },
    "phi": {
        "model_attr": "model",
        "layers_attr": "layers",
        "norm_attr": "final_layernorm",
        "lm_head_attr": "lm_head",
        "embed_attr": "embed_tokens",
    },
    "qwen": {
        "model_attr": "model",
        "layers_attr": "layers",
        "norm_attr": "norm",
        "lm_head_attr": "lm_head",
        "embed_attr": "embed_tokens",
    },
    "mistral": {
        "model_attr": "model",
        "layers_attr": "layers",
        "norm_attr": "norm",
        "lm_head_attr": "lm_head",
        "embed_attr": "embed_tokens",
    },
    "gemma": {
        "model_attr": "model",
        "layers_attr": "layers",
        "norm_attr": "norm",
        "lm_head_attr": "lm_head",
        "embed_attr": "embed_tokens",
    },
    "falcon": {
        "model_attr": "transformer",
        "layers_attr": "h",
        "norm_attr": "ln_f",
        "lm_head_attr": "lm_head",
        "embed_attr": "word_embeddings",
    },
    "gpt_neox": {
        "model_attr": "gpt_neox",
        "layers_attr": "layers",
        "norm_attr": "final_layer_norm",
        "lm_head_attr": "embed_out",
        "embed_attr": "embed_in",
    },
    "opt": {
        "model_attr": "model.decoder",
        "layers_attr": "layers",
        "norm_attr": "final_layer_norm",
        "lm_head_attr": "lm_head",
        "embed_attr": "embed_tokens",
    },
    "bloom": {
        "model_attr": "transformer",
        "layers_attr": "h",
        "norm_attr": "ln_f",
        "lm_head_attr": "lm_head",
        "embed_attr": "word_embeddings",
    },
    "mpt": {
        "model_attr": "transformer",
        "layers_attr": "blocks",
        "norm_attr": "norm_f",
        "lm_head_attr": "lm_head",
        "embed_attr": "wte",
    },
}

# ---------------------------------------------------------------------
# Architecture Detection
# ---------------------------------------------------------------------
# Maps architecture keywords to detection strings
ARCH_DETECTION: Dict[str, list] = {
    "llama": ["llama", "vicuna", "alpaca", "codellama", "tinyllama"],
    "gpt2": ["gpt2", "dialogpt"],
    "phi": ["phi"],
    "qwen": ["qwen"],
    "mistral": ["mistral", "mixtral", "zephyr"],
    "deepseek": ["deepseek"],  # Uses llama-style
    "gemma": ["gemma"],
    "falcon": ["falcon"],
    "gpt_neox": ["gpt_neox", "gpt-neox", "pythia", "dolly"],
    "opt": ["opt"],
    "bloom": ["bloom", "bloomz"],
    "mpt": ["mpt"],
}

# Mapping from detected architecture to registry key
# Some architectures (like deepseek) use another architecture's config
ARCH_TO_REGISTRY: Dict[str, str] = {
    "llama": "llama",
    "gpt2": "gpt2",
    "phi": "phi",
    "qwen": "qwen",  # Qwen2 uses llama-style but with its own norm
    "mistral": "mistral",
    "deepseek": "llama",  # DeepSeek uses llama-style architecture
    "gemma": "gemma",
    "falcon": "falcon",
    "gpt_neox": "gpt_neox",
    "opt": "opt",
    "bloom": "bloom",
    "mpt": "mpt",
}


def detect_architecture(
    model_type: Optional[str] = None,
    model_name: Optional[str] = None,
) -> str:
    """
    Detect model architecture from model_type or model_name.

    Args:
        model_type: Value from model.config.model_type
        model_name: HuggingFace model name/path

    Returns:
        Architecture key (e.g., "llama", "gpt2", "phi")

    Raises:
        ValueError: If architecture cannot be detected
    """
    mt = (model_type or "").lower()
    mn = (model_name or "").lower()

    for arch, keywords in ARCH_DETECTION.items():
        if any(kw in mt for kw in keywords) or any(kw in mn for kw in keywords):
            return arch

    raise ValueError(
        f"Could not detect architecture from model_type={model_type!r}, "
        f"model_name={model_name!r}. Supported architectures: {list(ARCH_DETECTION.keys())}"
    )


def get_registry_config(architecture: str) -> Dict[str, str]:
    """
    Get the registry configuration for an architecture.

    Args:
        architecture: Detected architecture (e.g., "llama", "deepseek")

    Returns:
        Dict with component attribute paths

    Raises:
        ValueError: If architecture not in registry
    """
    registry_key = ARCH_TO_REGISTRY.get(architecture, architecture)
    
    if registry_key not in MODEL_REGISTRY:
        raise ValueError(
            f"Architecture {architecture!r} (registry key: {registry_key!r}) "
            f"not in MODEL_REGISTRY. Available: {list(MODEL_REGISTRY.keys())}"
        )
    
    return MODEL_REGISTRY[registry_key]

