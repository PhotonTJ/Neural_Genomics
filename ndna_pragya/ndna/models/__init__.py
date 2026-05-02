"""
ndna.models

Model handling and architecture abstraction for nDNA metrics.
"""

from .handler import ModelHandler
from .registry import (
    MODEL_REGISTRY,
    ARCH_DETECTION,
    ARCH_TO_REGISTRY,
    detect_architecture,
    get_registry_config,
)
from .logit_lens import (
    logit_lens,
    logit_lens_probs,
    logit_lens_log_probs,
    batch_logit_lens,
    layerwise_predictions,
    compare_layer_predictions,
)

__all__ = [
    # Main handler
    "ModelHandler",
    # Registry
    "MODEL_REGISTRY",
    "ARCH_DETECTION",
    "ARCH_TO_REGISTRY",
    "detect_architecture",
    "get_registry_config",
    # Logit lens utilities
    "logit_lens",
    "logit_lens_probs",
    "logit_lens_log_probs",
    "batch_logit_lens",
    "layerwise_predictions",
    "compare_layer_predictions",
]

