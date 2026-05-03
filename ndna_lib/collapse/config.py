# ndna_lib/collapse/config.py
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class ProtocolType(str, Enum):
    """Collapse protocol choice."""
    CROSS_BREEDING = "cross_breeding"
    INBREEDING = "inbreeding"


@dataclass
class BreedingConfig:
    """
    Training-side knobs that control how hard you drive the model into collapse.
    These defaults mirror your GPT-2 Alpaca scripts.
    """
    dataset_id: str = "yahma/alpaca-cleaned"

    # Cross-breeding / inbreeding training limits
    max_train_samples: int = 10_000
    max_steps_first: int = 1000
    max_steps_later: int = 500
    lr: float = 5e-5
    train_max_seq_len: int = 512

    # Number of degeneration steps (excluding generation 0)
    generations: int = 5


@dataclass
class GeometryConfig:
    """
    Geometry knobs (Method 5 + spectral curvature) for collapse runs.
    """
    # Method-5 / FR on Alpaca
    geom_max_samples: int = 200
    geom_max_len: int = 192
    geom_batch_size: int = 2
    unit: str = "sequence"  # "sequence" or "token"

    # Spectral curvature on Alpaca
    spectral_num_prompts: int = 8
    spectral_max_tokens: int = 128


@dataclass
class CollapseConfig:
    """
    High-level configuration for a collapse experiment.
    For now, this is GPT-2-focused: base_model_id/model_tag will both usually be "gpt2".
    """
    base_model_id: str           # Hugging Face model id or local path, e.g. "gpt2"
    model_tag: str               # Folder tag, e.g. "gpt2"
    protocol: ProtocolType       # CROSS_BREEDING or INBREEDING

    base_run_dir: str = "./collapse_runs"
    run_name: Optional[str] = None
    seed: int = 4242

    breeding: BreedingConfig = field(default_factory=BreedingConfig)
    geometry: GeometryConfig = field(default_factory=GeometryConfig)
