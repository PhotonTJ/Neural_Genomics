"""Central configuration for the ICLR experiment suite.

Every experiment reads its models, probe and hyper-parameters from here so that a run is
reproducible from this file plus a git hash. Nothing below is read at import time by the
core library -- experiments pass explicit values -- so overriding a field in a driver
script is safe.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict

# --------------------------------------------------------------------------- paths
ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(ROOT, "results")
FIGURES = os.path.join(ROOT, "figures")
TABLES = os.path.join(ROOT, "tables")
CACHE = os.environ.get("NDNA_CACHE", os.path.join(ROOT, ".cache"))

for _d in (RESULTS, FIGURES, TABLES, CACHE):
    os.makedirs(_d, exist_ok=True)

# --------------------------------------------------------------------------- models
# Two recent, small-enough-to-iterate base models. Every experiment runs on both, so the
# suite is 14 experiments x 2 models = 28 runs. Override with NDNA_MODELS="a,b".
MODELS = {
    "qwen3_4b": "Qwen/Qwen3-4B",
    "gemma3_1b": "google/gemma-3-1b-pt",
}

# Aligned / tuned siblings, used by the experiments that need a matched pair rather than
# a derived variant. Keyed by base model id.
SIBLINGS = {
    "qwen3_4b": {
        "instruct": "Qwen/Qwen3-4B-Instruct-2507",
        "think": "Qwen/Qwen3-4B-Thinking-2507",
    },
    "gemma3_1b": {
        "instruct": "google/gemma-3-1b-it",
    },
}

if os.environ.get("NDNA_MODELS"):
    _sel = [m.strip() for m in os.environ["NDNA_MODELS"].split(",") if m.strip()]
    MODELS = {k: v for k, v in MODELS.items() if k in _sel}


# --------------------------------------------------------------------------- probes
@dataclass
class ProbeCfg:
    """Which text the triad is measured on."""
    name: str = "squad_v2"
    hf_path: str = "rajpurkar/squad_v2"
    split: str = "validation"
    n_prompts: int = 256
    max_length: int = 384


# The ten FLAN tasks used for the stability experiment. Names match the repo's
# 10_tasks folders; note the repo uses QQP (not QAP).
FLAN_TASKS = [
    ("ai2_arc", "allenai/ai2_arc", "ARC-Easy", "validation"),
    ("hellaswag", "Rowan/hellaswag", None, "validation"),
    ("winogrande", "allenai/winogrande", "winogrande_xl", "validation"),
    ("mnli", "nyu-mll/glue", "mnli", "validation_matched"),
    ("qqp", "nyu-mll/glue", "qqp", "validation"),
    ("squad_v2", "rajpurkar/squad_v2", None, "validation"),
    ("cnn_dailymail", "abisee/cnn_dailymail", "3.0.0", "validation"),
    ("common_gen", "allenai/common_gen", None, "validation"),
    ("wmt16", "wmt/wmt16", "de-en", "validation"),
    ("imdb", "stanfordnlp/imdb", None, "test"),
]


# --------------------------------------------------------------------------- triad
@dataclass
class TriadCfg:
    """Hyper-parameters of the nDNA triad. Defaults are the paper's."""
    tau: float = 1.0              # readout temperature
    keep_last_k: int = 8          # supervised positions retained per sequence
    eps_curv: float = 1e-8        # curvature stabiliser
    degen_delta: float = 1e-4     # spherical-triangle degeneracy threshold
    prob_floor: float = 1e-12     # clamp before sqrt / division
    batch_size: int = 4
    dtype: str = "float32"        # geometry is computed in fp32 regardless of model dtype
    curvature: str = "turn"       # "turn" (intrinsic) or "chord" (extrinsic, eq. 6)
    include_embedding: bool = True  # treat layer 0 (embeddings) as a depth node


# --------------------------------------------------------------------------- DTW
@dataclass
class DTWCfg:
    band: float = 0.2             # Sakoe-Chiba band as a fraction of depth
    resample_to: int = 32         # common depth grid for cross-model comparison
    normalise: str = "minmax"     # "minmax" | "median" | "none"; see core/dtw.py


# --------------------------------------------------------------------------- sweeps
QUANT_DOSES = ["fp16", "int8", "4bit", "3bit", "2bit"]
PRUNE_DOSES = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
GAIN_SCALES = [0.5, 1.0, 2.0, 4.0]          # exp05 invariance check
COLLAPSE_GENERATIONS = 12                    # exp12 early warning
SEEDS = [0, 1, 2]


@dataclass
class RunCfg:
    """Everything one experiment run needs, serialised into its NPZ."""
    experiment: str
    model_key: str
    model_id: str = ""
    seed: int = 0
    probe: ProbeCfg = field(default_factory=ProbeCfg)
    triad: TriadCfg = field(default_factory=TriadCfg)
    dtw: DTWCfg = field(default_factory=DTWCfg)
    device: str = "cuda"
    notes: str = ""

    def __post_init__(self):
        if not self.model_id:
            self.model_id = MODELS.get(self.model_key, self.model_key)

    def tag(self) -> str:
        return f"{self.experiment}__{self.model_key}__seed{self.seed}"

    def to_dict(self) -> dict:
        return asdict(self)
