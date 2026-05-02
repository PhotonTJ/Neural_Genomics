"""
ndna.data

Dataset handling and preprocessing for nDNA metrics.
"""

from .handler import DatasetHandler, TextDataset
from .registry import (
    DATASET_REGISTRY,
    get_dataset_config,
    list_datasets,
    make_text_processor,
)
from .collate import (
    make_causal_collate,
    make_belief_collate,
    causal_collate,
    belief_collate,
)
from .prompts import (
    DEFAULT_PROMPTS,
    PROMPT_SETS,
    get_prompts,
    list_prompt_sets,
    get_all_prompts,
)

__all__ = [
    # Handler
    "DatasetHandler",
    "TextDataset",
    # Registry
    "DATASET_REGISTRY",
    "get_dataset_config",
    "list_datasets",
    "make_text_processor",
    # Collate functions
    "make_causal_collate",
    "make_belief_collate",
    "causal_collate",
    "belief_collate",
    # Prompts
    "DEFAULT_PROMPTS",
    "PROMPT_SETS",
    "get_prompts",
    "list_prompt_sets",
    "get_all_prompts",
]

