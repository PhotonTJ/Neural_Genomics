# nDNA Library Design Document

## Overview

The nDNA library provides a modular framework for calculating and visualizing geometric metrics for transformer language models based on information geometry principles.

### Quick Extensibility

```python
# Add a model: 5 lines
MODEL_REGISTRY["my_model"] = {
    "model_attr": "model", "layers_attr": "layers", 
    "norm_attr": "norm", "lm_head_attr": "lm_head", "embed_attr": "embed_tokens"
}

# Add a dataset: 1 line  
DATASET_REGISTRY["my_data"] = {"hf_name": "org/data", "processor": lambda x: {"text": x["content"]}}

# Custom prompts: just a list
prompts = [("Label", "Text"), ...]
```

### Quick Parallelism

```python
# Tensor parallel: 70B model across 4 GPUs
calc = nDNA("meta-llama/Llama-3.1-70B", tensor_parallel=True, tp_size=4)

# Data parallel: distribute dataset processing
calc.calculate_all(dataset_name="squad", dp_processor=DataParallelProcessor())

# CLI: torchrun --nproc_per_node=4 -m ndna.cli calculate --tensor-parallel --data-parallel ...
```

### Metrics

| Metric | Input | Core Technique | Output |
|--------|-------|----------------|--------|
| **Spectral Curvature** | Individual prompts/texts | Discrete curvature on sqrt-embedding sphere | κ per interior layer |
| **Thermodynamic Length** | Batched dataset | Fisher-Rao distance between layer log-probs | FR step length per layer transition |
| **Belief Vector Field** | Batched dataset + targets | Tangent vectors from gradient of log-likelihood | ‖v‖ per layer |
| **nDNA** (future) | All above | Combined normalized metric | Unified score per layer |

---

## Architecture

```
ndna/
├── __init__.py                 # Main exports: nDNA, calculate_all
├── core/                       # Core metric calculators
│   ├── __init__.py
│   ├── spectral.py            # Spectral Curvature
│   ├── thermodynamic.py       # Thermodynamic Length (Fisher-Rao)
│   ├── belief.py              # Belief Vector Field
│   ├── ndna.py                # Combined nDNA metric (future)
│   └── geometry.py            # Shared geometry: sqrt_embed, project_tangent, etc.
├── models/                     # Model abstraction
│   ├── __init__.py
│   ├── handler.py             # Unified ModelHandler for all architectures
│   └── logit_lens.py          # Logit lens utilities
├── data/                       # Dataset handling
│   ├── __init__.py
│   ├── handler.py             # DatasetHandler
│   ├── collate.py             # Collate functions (causal, belief)
│   └── prompts.py             # Default prompt sets
├── parallel/                   # Parallelism support
│   ├── __init__.py
│   ├── tensor_parallel.py     # Model sharding across GPUs
│   ├── data_parallel.py       # Distributed data processing
│   └── utils.py               # Distributed utilities
├── storage/                    # Persistence
│   ├── __init__.py
│   ├── saver.py               # Save results (NPZ, JSON)
│   └── loader.py              # Load saved results
├── visualization/              # Plotting
│   ├── __init__.py
│   ├── plots_2d.py            # Matplotlib 2D plots
│   ├── plots_3d.py            # Plotly 3D interactive plots
│   └── styles.py              # Plot styling
├── cli.py                      # Command-line interface
└── config.py                   # Configuration dataclasses
```

---

## Core Components

### 1. Geometry Utilities (`core/geometry.py`)

The foundation of all metrics - geometric operations on the probability simplex.

```python
def sqrt_embed(q: Tensor, eps: float = 1e-12) -> Tensor:
    """
    Map probability distribution to unit sphere via square-root embedding.
    
    u = sqrt(q) / ||sqrt(q)||
    
    Args:
        q: Probability distribution (V,) - must sum to 1
        eps: Floor for numerical stability
    
    Returns:
        Unit vector on V-dimensional sphere (V,)
    """

def project_tangent(u: Tensor, v: Tensor) -> Tensor:
    """
    Project vector v onto tangent space at point u on unit sphere.
    
    v_tangent = v - (u · v) * u
    
    Args:
        u: Point on unit sphere (V,)
        v: Vector to project (V,)
    
    Returns:
        Tangent vector at u (V,)
    """

def discrete_curvature(u_list: List[Tensor], eps_curv: float = 1e-12) -> Tuple[List[float], List[float]]:
    """
    Compute discrete curvature for interior points.
    
    κ_ℓ = ||Δ²u_ℓ|| / ||Δu_ℓ||^(3/2)
    
    where:
        Δu_ℓ = project_tangent(u_ℓ, u_{ℓ+1} - u_ℓ)
        Δ²u_ℓ = project_tangent(u_ℓ, u_{ℓ+1} - 2u_ℓ + u_{ℓ-1})
    
    Args:
        u_list: List of sqrt-embeddings for each layer [u_0, u_1, ..., u_L]
        eps_curv: Epsilon for denominator stability
    
    Returns:
        (curvatures, speeds) for interior layers ℓ = 1 to L-1
    """

def fisher_rao_distance(logp1: Tensor, logp2: Tensor) -> Tensor:
    """
    Compute Fisher-Rao distance between two log-probability distributions.
    
    d_FR = 2 * arccos(BC) where BC = Σ √(p₁ᵢ × p₂ᵢ)
    
    Numerically stable via: BC = exp(logsumexp(0.5 * (logp1 + logp2)))
    
    Args:
        logp1, logp2: Log probabilities (..., V)
    
    Returns:
        Fisher-Rao distance in radians (...)
    """
```

### 2. Model Handler (`models/handler.py`)

Unified interface for HuggingFace transformer models.

```python
class ModelHandler:
    """
    Unified handler for different transformer architectures.
    
    Supported: GPT-2, LLaMA, Qwen, Phi, DeepSeek, Mistral, etc.
    """
    
    def __init__(self, 
                 model_name: str,
                 device: str = "auto",
                 dtype: torch.dtype = torch.bfloat16,
                 hf_token: Optional[str] = None):
        """Load model and detect architecture."""
    
    # Architecture detection
    @property
    def architecture(self) -> str: ...  # 'llama', 'gpt2', 'phi', etc.
    
    @property
    def is_causal(self) -> bool: ...
    
    @property
    def num_layers(self) -> int: ...
    
    @property
    def vocab_size(self) -> int: ...
    
    # Component access
    def get_layers(self) -> nn.ModuleList: ...
    def get_final_norm(self) -> nn.Module: ...
    def get_lm_head(self) -> nn.Module: ...
    
    # Core operations
    @torch.no_grad()
    def get_hidden_states(self, input_ids: Tensor, attention_mask: Tensor) -> Tuple[Tensor, ...]:
        """Forward pass returning all hidden states including embedding."""
    
    @torch.no_grad()
    def logit_lens(self, hidden_state: Tensor) -> Tensor:
        """Apply final_norm + lm_head to get logits from any layer's hidden state."""
    
    @torch.no_grad()
    def layerwise_distributions(self, text: str) -> List[Tensor]:
        """
        Get next-token probability distributions at each layer for a single text.
        Uses logit lens at the last token position.
        
        Returns: [q_0, q_1, ..., q_L] where q_ℓ is (V,) probability distribution
        """
```

### 3. Dataset Handler (`data/handler.py`)

Flexible dataset loading and preprocessing.

```python
class DatasetHandler:
    """
    Load and preprocess text datasets from various sources.
    """
    
    def __init__(self,
                 tokenizer,
                 max_length: int = 256,
                 batch_size: int = 2,
                 num_workers: int = 4):
        ...
    
    # Loading methods
    def load_huggingface(self, 
                         name: str, 
                         split: str = "train",
                         text_column: str = "text",
                         text_processor: Optional[Callable] = None,
                         max_samples: Optional[int] = None) -> Self: ...
    
    def load_texts(self, texts: List[str]) -> Self: ...
    
    def load_file(self, path: str, format: str = "txt") -> Self: ...
    
    # DataLoader creation
    def create_dataloader(self, collate_type: str = "causal") -> DataLoader:
        """
        Create DataLoader with appropriate collate function.
        
        collate_type:
            - "causal": Teacher-forced next-token prediction (for thermodynamic)
            - "belief": Keep last K tokens per sample (for belief vectors)
        """
    
    def get_sample_texts(self, n: int = 10) -> List[str]: ...
```

**Collate Functions** (`data/collate.py`):

```python
def causal_collate(batch, tokenizer, max_length: int) -> Dict:
    """
    Standard causal LM collate for thermodynamic length.
    
    Returns:
        input_ids: (B, S-1) - tokens up to second-to-last
        attention_mask: (B, S-1)
        labels: (B, S-1) - next-token targets, padding → -100
    """

def belief_collate(batch, tokenizer, max_length: int, keep_last_k: int = 32) -> Dict:
    """
    Belief vector field collate - keeps only last K supervised positions.
    
    Returns:
        input_ids: (B, S-1)
        attention_mask: (B, S-1)
        labels: (B, S-1)
        select_mask: (B, S-1) - True for last K valid positions per sample
    """
```

---

## Metric Calculators

### 4. Spectral Curvature (`core/spectral.py`)

Measures geometric curvature of probability manifold through layers.

```python
class SpectralCalculator:
    """
    Compute spectral curvature using logit lens and discrete geometry.
    
    Works on individual prompts/texts (not batched over dataset).
    """
    
    def __init__(self, model_handler: ModelHandler, eps_dist: float = 1e-12, eps_curv: float = 1e-12):
        ...
    
    def calculate_for_text(self, text: str) -> SpectralResult:
        """
        Calculate spectral curvature for a single text.
        
        Process:
            1. Get layerwise distributions via logit lens at last token
            2. Convert to sqrt embeddings
            3. Compute discrete curvature at interior layers
        
        Returns:
            SpectralResult with curvatures, speeds, statistics
        """
    
    def calculate_for_prompts(self, 
                              prompts: List[Tuple[str, str]],  # (label, text)
                              ) -> Dict[str, SpectralResult]:
        """Calculate for labeled prompts, return dict by label."""
    
    def calculate_for_texts(self, texts: List[str]) -> AggregatedSpectralResult:
        """Calculate for multiple texts, aggregate statistics."""

@dataclass
class SpectralResult:
    curvatures: np.ndarray       # (num_layers - 2,) - interior layers only
    speeds: np.ndarray           # (num_layers - 1,) - first differences
    layer_indices: np.ndarray    # [1, 2, ..., num_interior]
    text_preview: str            # First 100 chars
    label: Optional[str]         # Prompt label if provided
    
    # Statistics
    mean_curvature: float
    max_curvature: float
    min_curvature: float
```

### 5. Thermodynamic Length (`core/thermodynamic.py`)

Measures Fisher-Rao distance between consecutive layer predictions.

```python
class ThermodynamicCalculator:
    """
    Compute thermodynamic length using Fisher-Rao geometry.
    
    Works on batched dataset - aggregates FR distances over all valid tokens.
    """
    
    def __init__(self, model_handler: ModelHandler):
        ...
    
    def calculate(self, dataloader: DataLoader) -> ThermoResult:
        """
        Calculate Fisher-Rao thermodynamic length over dataset.
        
        Process:
            1. For each batch, get log-probs at each layer via logit lens
            2. Compute FR distance between consecutive layers at each valid token
            3. Aggregate mean FR step length per inter-layer transition
        
        Returns:
            ThermoResult with per-step mean FR lengths
        """
    
    def calculate_per_sample(self, 
                             dataloader: DataLoader, 
                             max_samples: int = 50) -> ThermoResultPerSample:
        """
        Calculate per-sample FR lengths for 3D visualization.
        """

@dataclass
class ThermoResult:
    step_lengths: np.ndarray     # (num_layers - 1,) - FR distance per transition
    step_indices: np.ndarray     # [1, 2, ..., num_steps]
    total_length: float          # Sum of all step lengths (radians)
    num_samples_processed: int

@dataclass  
class ThermoResultPerSample:
    per_sample_lengths: np.ndarray  # (num_samples, num_steps)
    mean_lengths: np.ndarray        # (num_steps,)
    step_indices: np.ndarray
```

### 6. Belief Vector Field (`core/belief.py`)

Measures tangent vectors representing belief changes with respect to targets.

```python
class BeliefCalculator:
    """
    Compute belief vector field - tangent vectors in probability space.
    
    Works on batched dataset with target labels.
    """
    
    def __init__(self, 
                 model_handler: ModelHandler,
                 tau: float = 1.0,           # Temperature
                 fr_norm: bool = True):      # Report 2*||v|| (FR) or ||v|| (Euclidean)
        ...
    
    def calculate(self, dataloader: DataLoader) -> BeliefResult:
        """
        Calculate belief vector field magnitudes over dataset.
        
        Process:
            1. For each layer, compute logits via logit lens
            2. Compute gradient of log-likelihood w.r.t. logits: g = (1/τ)(e_y - q)
            3. Convert to tangent vector: t = (1/2τ)(u*g - ⟨q,g⟩*u)
            4. Accumulate and average tangent vectors
            5. Project to tangent space at mean u, report norm
        
        Returns:
            BeliefResult with per-layer belief vector magnitudes
        """
    
    def calculate_per_sample(self,
                             dataloader: DataLoader,
                             max_samples: int = 50) -> BeliefResultPerSample:
        """
        Calculate per-sample belief magnitudes for 3D visualization.
        """

@dataclass
class BeliefResult:
    belief_norms: np.ndarray     # (num_layers,) - ||v|| per layer
    layer_indices: np.ndarray    # [1, 2, ..., num_layers]
    
    # Statistics
    mean_norm: float
    max_norm: float
    min_norm: float
    max_layer: int               # Layer with highest ||v||
    min_layer: int               # Layer with lowest ||v||

@dataclass
class BeliefResultPerSample:
    per_sample_norms: np.ndarray   # (num_samples, num_layers)
    mean_norms: np.ndarray         # (num_layers,)
    layer_indices: np.ndarray
```

---

## Storage

### 7. Result Persistence (`storage/`)

Save all per-layer scores and metadata.

```python
@dataclass
class FullResults:
    """Container for all metric results."""
    model_name: str
    dataset_name: str
    timestamp: str
    config: Dict[str, Any]
    
    spectral: Optional[Union[SpectralResult, Dict[str, SpectralResult]]]
    thermodynamic: Optional[ThermoResult]
    belief: Optional[BeliefResult]
    
    # Per-sample data for 3D visualization
    thermo_per_sample: Optional[ThermoResultPerSample]
    belief_per_sample: Optional[BeliefResultPerSample]

class ResultSaver:
    """Save results to disk."""
    
    def save(self, results: FullResults, output_dir: str):
        """
        Save all results to structured directory.
        
        Creates:
            output_dir/
            ├── spectral_curvature.npz
            ├── thermodynamic_length.npz
            ├── belief_vector_field.npz
            ├── per_sample/
            │   ├── thermodynamic_per_sample.npz
            │   └── belief_per_sample.npz
            └── metadata.json
        """
    
    def save_spectral(self, result: SpectralResult, path: str): ...
    def save_thermodynamic(self, result: ThermoResult, path: str): ...
    def save_belief(self, result: BeliefResult, path: str): ...

class ResultLoader:
    """Load saved results."""
    
    def load(self, output_dir: str) -> FullResults: ...
    def load_spectral(self, path: str) -> SpectralResult: ...
    def load_thermodynamic(self, path: str) -> ThermoResult: ...
    def load_belief(self, path: str) -> BeliefResult: ...
```

**File Formats:**

```python
# spectral_curvature.npz
{
    'curvatures': np.ndarray,        # Per-prompt or aggregated
    'speeds': np.ndarray,
    'layer_indices': np.ndarray,
    'labels': np.ndarray,            # Prompt labels if available
    'mean_curvature': float,
    'std_curvature': float,
}

# thermodynamic_length.npz
{
    'step_lengths': np.ndarray,      # (num_steps,)
    'step_indices': np.ndarray,
    'total_length': float,
}

# belief_vector_field.npz
{
    'belief_norms': np.ndarray,      # (num_layers,)
    'layer_indices': np.ndarray,
    'mean_norm': float,
}

# metadata.json
{
    'model_name': str,
    'dataset_name': str,
    'num_samples': int,
    'num_layers': int,
    'config': {...},
    'timestamp': str,
}
```

---

## Visualization

### 8. Plotting Functions (`visualization/`)

**2D Plots** (`plots_2d.py`):

```python
def plot_spectral_curvature(
    result: Union[SpectralResult, Dict[str, SpectralResult]],
    save_path: Optional[str] = None,
    log_scale: bool = True,
    overlay: bool = True,       # Overlay multiple prompts
    model_name: str = ""
) -> Figure: ...

def plot_thermodynamic_length(
    result: ThermoResult,
    save_path: Optional[str] = None,
    log_scale: bool = True,
    model_name: str = ""
) -> Figure: ...

def plot_belief_vector_field(
    result: BeliefResult,
    save_path: Optional[str] = None,
    log_scale: bool = True,
    model_name: str = ""
) -> Figure: ...

def plot_all_metrics(
    spectral: SpectralResult,
    thermo: ThermoResult,
    belief: BeliefResult,
    save_dir: str,
    model_name: str = ""
): ...
```

**3D Interactive Plots** (`plots_3d.py`):

```python
def plot_spectral_3d(
    results: Dict[str, SpectralResult],  # Multiple prompts
    save_path: str,
    log_scale: bool = True,
    model_name: str = ""
) -> go.Figure:
    """
    3D surface: Prompt × Layer × Curvature
    """

def plot_thermodynamic_3d(
    result: ThermoResultPerSample,
    save_path: str,
    model_name: str = ""
) -> go.Figure:
    """
    3D surface: Sample × Step × FR Length
    """

def plot_belief_3d(
    result: BeliefResultPerSample,
    save_path: str,
    model_name: str = ""
) -> go.Figure:
    """
    3D surface: Sample × Layer × ||v||
    """
```

---

## High-Level API

### 9. Main Interface (`ndna/__init__.py`)

```python
class nDNA:
    """
    High-level interface for calculating all metrics.
    """
    
    def __init__(self,
                 model_name: str,
                 device: str = "auto",
                 dtype: torch.dtype = torch.bfloat16,
                 hf_token: Optional[str] = None,
                 # Tensor Parallelism
                 tensor_parallel: bool = False,
                 tp_size: Optional[int] = None,
                 device_map: str = "auto",
                 max_memory: Optional[Dict[str, str]] = None,
                 # Memory Optimization
                 gradient_checkpointing: bool = False,
                 flash_attention: bool = True,
                 offload_to_cpu: bool = False,
                 # Other
                 **config):
        """
        Initialize with model.
        
        Args:
            model_name: HuggingFace model identifier
            device: Device for single-GPU mode ("cuda", "cpu", "auto")
            dtype: Model dtype (torch.bfloat16, torch.float16, torch.float32)
            hf_token: HuggingFace API token for gated models
            
            # Tensor Parallelism (for large models)
            tensor_parallel: Enable model sharding across GPUs
            tp_size: Number of GPUs for tensor parallelism (None = all)
            device_map: Distribution strategy ("auto", "balanced", "sequential")
            max_memory: Per-GPU memory limits {"cuda:0": "40GiB", ...}
            
            # Memory Optimization
            gradient_checkpointing: Trade compute for memory
            flash_attention: Use flash attention (faster, less memory)
            offload_to_cpu: Offload inactive layers to CPU
        """
        # Setup tensor parallelism if requested
        if tensor_parallel:
            from ndna.parallel import TensorParallelLoader
            tp_loader = TensorParallelLoader(tp_size, device_map, max_memory)
            self.model_handler = ModelHandler(
                model_name, 
                device="auto",
                dtype=dtype,
                hf_token=hf_token,
                tp_loader=tp_loader,
                gradient_checkpointing=gradient_checkpointing,
                flash_attention=flash_attention,
            )
        else:
            self.model_handler = ModelHandler(model_name, device, dtype, hf_token)
        
        self.spectral_calc = SpectralCalculator(self.model_handler, **config)
        self.thermo_calc = ThermodynamicCalculator(self.model_handler)
        self.belief_calc = BeliefCalculator(self.model_handler, **config)
    
    def calculate_all(
        self,
        dataset_name: str = "squad",
        prompts: Optional[List[Tuple[str, str]]] = None,  # For spectral
        max_samples: int = 2000,
        batch_size: int = 2,
        compute_per_sample: bool = True,  # For 3D visualization
        save: bool = True,
        plot: bool = True,
        output_dir: str = "results",
        # Data Parallelism
        dp_processor: Optional['DataParallelProcessor'] = None,
    ) -> FullResults:
        """
        Calculate all three metrics.
        
        Args:
            dataset_name: HuggingFace dataset for thermodynamic & belief
            prompts: Labeled prompts for spectral curvature (uses defaults if None)
            max_samples: Max samples from dataset
            batch_size: Batch size for dataset processing
            compute_per_sample: Collect per-sample data for 3D plots
            save: Save results to disk
            plot: Generate plots
            output_dir: Output directory
            dp_processor: DataParallelProcessor for distributed processing
        
        Returns:
            FullResults containing all metrics
        """
    
    # Individual metric methods
    def calculate_spectral(self, texts_or_prompts: Union[List[str], List[Tuple[str, str]]]) -> ...: ...
    def calculate_thermodynamic(self, dataset_name: str, dp_processor=None, **kwargs) -> ThermoResult: ...
    def calculate_belief(self, dataset_name: str, dp_processor=None, **kwargs) -> BeliefResult: ...
    
    # Utility methods
    def save(self, results: FullResults, output_dir: str): ...
    def plot(self, results: FullResults, output_dir: str): ...
    def load(self, output_dir: str) -> FullResults: ...


# Convenience function
def calculate_all_metrics(
    model_name: str,
    dataset_name: str = "squad",
    prompts: Optional[List[Tuple[str, str]]] = None,
    output_dir: str = "results",
    **kwargs
) -> FullResults:
    """One-liner to calculate everything."""
    calc = nDNA(model_name, **kwargs)
    return calc.calculate_all(dataset_name, prompts, save=True, plot=True, output_dir=output_dir)
```

---

## CLI Interface

### 10. Command Line (`cli.py`)

```bash
# Calculate all metrics
ndna calculate \
    --model deepseek-ai/DeepSeek-R1-Distill-Qwen-14B \
    --dataset squad \
    --max-samples 2000 \
    --output results/deepseek/squad \
    --plot

# Calculate specific metric
ndna spectral \
    --model MODEL_NAME \
    --prompts "English:Hello world" "Hindi:नमस्ते" \
    --output results/spectral.npz

ndna thermodynamic \
    --model MODEL_NAME \
    --dataset squad \
    --max-samples 2000 \
    --output results/thermo.npz

ndna belief \
    --model MODEL_NAME \
    --dataset squad \
    --output results/belief.npz

# Plot from saved results
ndna plot \
    --input results/deepseek/squad \
    --output plots/ \
    --format png html

# Compare multiple models
ndna compare \
    --inputs results/model1 results/model2 \
    --metric spectral \
    --output comparison.png

# ===== PARALLELISM OPTIONS =====

# Tensor parallel: load 70B model across 4 GPUs
ndna calculate \
    --model meta-llama/Llama-3.1-70B \
    --tensor-parallel --tp-size 4 \
    --dataset squad \
    --output results/

# Data parallel: distribute dataset across GPUs
torchrun --nproc_per_node=4 -m ndna.cli calculate \
    --model meta-llama/Llama-3.1-8B \
    --data-parallel \
    --dataset squad \
    --output results/

# Combined: large model + fast processing
torchrun --nproc_per_node=8 -m ndna.cli calculate \
    --model meta-llama/Llama-3.1-70B \
    --tensor-parallel --tp-size 8 \
    --data-parallel \
    --dataset squad \
    --output results/

# Memory optimization for very large models
ndna calculate \
    --model meta-llama/Llama-3.1-405B \
    --tensor-parallel --tp-size 8 \
    --gradient-checkpointing \
    --offload-to-cpu \
    --low-memory \
    --output results/
```

**CLI Implementation:**

```python
import click

@click.group()
def cli():
    """nDNA: Neural DNA metrics for transformer analysis."""
    pass

@cli.command()
@click.option('--model', '-m', required=True, help='HuggingFace model name')
@click.option('--dataset', '-d', default='squad', help='Dataset name')
@click.option('--max-samples', default=2000, type=int)
@click.option('--batch-size', default=2, type=int)
@click.option('--output', '-o', default='results', help='Output directory')
@click.option('--plot/--no-plot', default=True)
@click.option('--device', default='auto')
# Tensor Parallelism
@click.option('--tensor-parallel', '-tp', is_flag=True, help='Enable tensor parallelism')
@click.option('--tp-size', type=int, default=None, help='Number of GPUs for tensor parallelism')
@click.option('--device-map', default='auto', help='Device map strategy: auto, balanced, sequential')
@click.option('--max-memory', type=str, default=None, help='Max memory per GPU, e.g., "40GiB"')
# Data Parallelism  
@click.option('--data-parallel', '-dp', is_flag=True, help='Enable data parallelism (use with torchrun)')
@click.option('--dp-backend', default='nccl', help='DDP backend: nccl, gloo, mpi')
# Memory Optimization
@click.option('--gradient-checkpointing', is_flag=True, help='Enable gradient checkpointing')
@click.option('--offload-to-cpu', is_flag=True, help='Offload model layers to CPU')
@click.option('--low-memory', is_flag=True, help='Low memory mode (sequential processing)')
@click.option('--flash-attention/--no-flash-attention', default=True, help='Use flash attention')
def calculate(model, dataset, max_samples, batch_size, output, plot, device,
              tensor_parallel, tp_size, device_map, max_memory,
              data_parallel, dp_backend,
              gradient_checkpointing, offload_to_cpu, low_memory, flash_attention):
    """Calculate all nDNA metrics."""
    from ndna import nDNA
    from ndna.parallel import TensorParallelLoader, DataParallelProcessor
    
    # Setup parallelism
    tp_loader = None
    dp_processor = None
    
    if tensor_parallel:
        tp_loader = TensorParallelLoader(
            tp_size=tp_size,
            device_map=device_map,
            max_memory={"cuda:0": max_memory} if max_memory else None
        )
    
    if data_parallel:
        dp_processor = DataParallelProcessor(backend=dp_backend)
        dp_processor.setup()
    
    # Initialize with parallelism
    calc = nDNA(
        model, 
        device=device,
        tensor_parallel_loader=tp_loader,
        gradient_checkpointing=gradient_checkpointing,
        flash_attention=flash_attention,
    )
    
    results = calc.calculate_all(
        dataset_name=dataset,
        max_samples=max_samples,
        batch_size=batch_size,
        save=True,
        plot=plot,
        output_dir=output,
        dp_processor=dp_processor,
    )
    
    # Only main process reports
    if dp_processor is None or dp_processor.is_main:
        click.echo(f"✅ Results saved to {output}")

@cli.command()
@click.option('--input', '-i', required=True, help='Results directory')
@click.option('--output', '-o', required=True, help='Plot output directory')
@click.option('--format', '-f', multiple=True, default=['png', 'html'])
def plot(input, output, format):
    """Generate plots from saved results."""
    ...

if __name__ == '__main__':
    cli()
```

---

## Configuration

```python
# config.py
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class SpectralConfig:
    eps_dist: float = 1e-12
    eps_curv: float = 1e-12

@dataclass
class BeliefConfig:
    tau: float = 1.0
    tokens_per_example: int = 32
    fr_norm: bool = True

@dataclass
class DataConfig:
    max_samples: int = 2000
    batch_size: int = 2
    max_length: int = 256
    num_workers: int = 4

@dataclass
class nDNAConfig:
    spectral: SpectralConfig = field(default_factory=SpectralConfig)
    belief: BeliefConfig = field(default_factory=BeliefConfig)
    data: DataConfig = field(default_factory=DataConfig)
    parallel: ParallelConfig = field(default_factory=ParallelConfig)
    device: str = "auto"
    dtype: str = "bfloat16"

@dataclass
class ParallelConfig:
    # Tensor Parallelism (model sharding)
    tensor_parallel: bool = False
    tp_size: int = None                    # Auto-detect if None
    device_map: str = "auto"               # "auto", "balanced", "sequential", or custom dict
    
    # Data Parallelism (distributed processing)
    data_parallel: bool = False
    dp_backend: str = "nccl"               # "nccl", "gloo", "mpi"
    world_size: int = None                 # Auto-detect from WORLD_SIZE env
    local_rank: int = None                 # Auto-detect from LOCAL_RANK env
    
    # Memory optimization
    gradient_checkpointing: bool = False
    offload_to_cpu: bool = False           # Offload optimizer states to CPU
    low_memory: bool = False               # Use sequential processing for large models
```

---

## Parallelism

### Overview

nDNA supports two types of parallelism for handling large models and datasets:

| Type | Use Case | How It Works |
|------|----------|--------------|
| **Tensor Parallelism (TP)** | Model too large for 1 GPU | Shard model layers across GPUs |
| **Data Parallelism (DP)** | Faster dataset processing | Process different batches on different GPUs |

```python
# Quick usage
from ndna import nDNA

# Tensor parallel: load 70B model across 4 GPUs
calc = nDNA("meta-llama/Llama-3.1-70B", tensor_parallel=True, tp_size=4)

# Data parallel: process dataset 4x faster
calc = nDNA("meta-llama/Llama-3.1-8B", data_parallel=True)

# Both: large model + fast processing
calc = nDNA("meta-llama/Llama-3.1-70B", tensor_parallel=True, tp_size=4, data_parallel=True)
```

---

### Tensor Parallelism (`parallel/tensor_parallel.py`)

For loading models that don't fit on a single GPU.

```python
class TensorParallelLoader:
    """
    Load model with tensor parallelism using HuggingFace Accelerate.
    
    Strategies:
    - "auto": Automatically distribute layers across available GPUs
    - "balanced": Equal VRAM usage per GPU
    - "sequential": Fill GPUs one by one
    - Custom dict: {"cuda:0": "10GiB", "cuda:1": "10GiB", ...}
    """
    
    def __init__(self, 
                 tp_size: Optional[int] = None,
                 device_map: Union[str, Dict] = "auto",
                 max_memory: Optional[Dict[str, str]] = None,
                 offload_folder: Optional[str] = None):
        """
        Args:
            tp_size: Number of GPUs to use (None = all available)
            device_map: Distribution strategy
            max_memory: Per-device memory limits {"cuda:0": "20GiB", ...}
            offload_folder: Path for CPU/disk offloading (for extreme cases)
        """
        self.tp_size = tp_size or torch.cuda.device_count()
        self.device_map = device_map
        self.max_memory = max_memory
        self.offload_folder = offload_folder
    
    def load_model(self, model_name: str, dtype: torch.dtype = torch.bfloat16):
        """Load model with tensor parallelism."""
        from accelerate import init_empty_weights, load_checkpoint_and_dispatch
        
        # Option 1: Simple device_map="auto"
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=self.device_map,
            torch_dtype=dtype,
            max_memory=self.max_memory,
            offload_folder=self.offload_folder,
        )
        return model
    
    def get_primary_device(self) -> str:
        """Get the device where embeddings/final layers are placed."""
        # Usually cuda:0 for auto device_map
        return "cuda:0"
```

**Usage examples:**

```python
# Auto-distribute across all GPUs
calc = nDNA("meta-llama/Llama-3.1-70B", tensor_parallel=True)

# Specific number of GPUs
calc = nDNA("meta-llama/Llama-3.1-70B", tensor_parallel=True, tp_size=4)

# Memory limits per GPU
calc = nDNA(
    "meta-llama/Llama-3.1-70B",
    tensor_parallel=True,
    max_memory={"cuda:0": "40GiB", "cuda:1": "40GiB", "cuda:2": "40GiB", "cuda:3": "40GiB"}
)

# CPU offloading for extreme cases (slow but works)
calc = nDNA(
    "meta-llama/Llama-3.1-405B",
    tensor_parallel=True,
    offload_to_cpu=True,
    offload_folder="./offload"
)
```

---

### Data Parallelism (`parallel/data_parallel.py`)

For processing datasets faster across multiple GPUs.

```python
class DataParallelProcessor:
    """
    Distribute dataset processing across multiple GPUs using DDP.
    
    Each GPU:
    - Loads a copy of the model (or shard if combined with TP)
    - Processes different batches
    - Results are gathered and aggregated on rank 0
    """
    
    def __init__(self,
                 backend: str = "nccl",
                 world_size: Optional[int] = None,
                 local_rank: Optional[int] = None):
        """
        Args:
            backend: "nccl" (NVIDIA), "gloo" (CPU/fallback), "mpi"
            world_size: Total number of processes
            local_rank: This process's GPU index
        """
        self.backend = backend
        self.world_size = world_size or int(os.environ.get("WORLD_SIZE", 1))
        self.local_rank = local_rank or int(os.environ.get("LOCAL_RANK", 0))
        self.is_main = (self.local_rank == 0)
        
    def setup(self):
        """Initialize distributed process group."""
        if not dist.is_initialized():
            dist.init_process_group(
                backend=self.backend,
                world_size=self.world_size,
                rank=self.local_rank
            )
        torch.cuda.set_device(self.local_rank)
    
    def get_distributed_sampler(self, dataset) -> DistributedSampler:
        """Create sampler that splits dataset across processes."""
        return DistributedSampler(
            dataset,
            num_replicas=self.world_size,
            rank=self.local_rank,
            shuffle=False
        )
    
    def gather_results(self, local_tensor: torch.Tensor) -> torch.Tensor:
        """Gather results from all processes to rank 0."""
        gathered = [torch.zeros_like(local_tensor) for _ in range(self.world_size)]
        dist.all_gather(gathered, local_tensor)
        return torch.stack(gathered) if self.is_main else None
    
    def reduce_sum(self, tensor: torch.Tensor) -> torch.Tensor:
        """Sum tensors across all processes."""
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        return tensor
```

**Integration with Calculators:**

```python
# In ThermodynamicCalculator
def calculate(self, dataloader: DataLoader, dp_processor: Optional[DataParallelProcessor] = None):
    
    if dp_processor:
        # Use distributed sampler
        sampler = dp_processor.get_distributed_sampler(dataloader.dataset)
        dataloader = DataLoader(dataloader.dataset, sampler=sampler, ...)
    
    # Each process computes partial sums
    local_fr_sums = torch.zeros(num_steps, device=self.device)
    local_counts = torch.zeros(num_steps, device=self.device)
    
    for batch in dataloader:
        # ... compute FR distances ...
        local_fr_sums += batch_fr_sums
        local_counts += batch_counts
    
    if dp_processor:
        # Aggregate across all GPUs
        local_fr_sums = dp_processor.reduce_sum(local_fr_sums)
        local_counts = dp_processor.reduce_sum(local_counts)
    
    # Final mean (same on all ranks after all_reduce)
    fr_means = local_fr_sums / local_counts.clamp_min(1)
    return ThermoResult(step_lengths=fr_means.cpu().numpy(), ...)
```

---

### Combined TP + DP

For very large models AND large datasets:

```python
# parallel/combined.py

class HybridParallel:
    """
    Combine tensor parallelism (model sharding) with data parallelism (batch distribution).
    
    Example: 4 nodes × 8 GPUs each = 32 GPUs
    - TP across 8 GPUs per node (one model copy per node)
    - DP across 4 nodes (4 model copies processing different data)
    """
    
    def __init__(self,
                 tp_size: int = 8,           # GPUs per model copy
                 dp_size: int = None):       # Number of model copies (auto = world_size / tp_size)
        self.tp_size = tp_size
        self.world_size = int(os.environ.get("WORLD_SIZE", tp_size))
        self.dp_size = dp_size or (self.world_size // tp_size)
        
        # Determine this process's role
        self.global_rank = int(os.environ.get("RANK", 0))
        self.dp_rank = self.global_rank // tp_size      # Which data-parallel group
        self.tp_rank = self.global_rank % tp_size       # Position within TP group
        
    def get_device_map(self) -> Dict:
        """Get device map for this process's TP group."""
        # GPUs for this TP group
        start_gpu = self.dp_rank * self.tp_size
        gpus = [f"cuda:{start_gpu + i}" for i in range(self.tp_size)]
        return {"": gpus}  # Distribute across these GPUs
```

---

### Per-Metric Parallelism Behavior

| Metric | Tensor Parallel | Data Parallel |
|--------|-----------------|---------------|
| **Spectral Curvature** | ✅ Model sharded, forward on primary device | ⚠️ Limited benefit (per-text, not batched) |
| **Thermodynamic Length** | ✅ Layerwise forward on sharded model | ✅ Distribute batches, reduce FR sums |
| **Belief Vector Field** | ✅ Layerwise forward on sharded model | ✅ Distribute batches, reduce belief vectors |

**Spectral Curvature** (per-text, not batched):
```python
# Parallelism strategy: TP for large models, simple loop for prompts
def calculate_spectral_parallel(prompts: List[str], tp_model):
    results = []
    for prompt in prompts:  # Sequential over prompts
        # Forward pass uses all TP GPUs internally
        q_list = tp_model.layerwise_distributions(prompt)
        results.append(compute_curvature(q_list))
    return results
```

**Thermodynamic/Belief** (batched):
```python
# Full DP support with proper aggregation
def calculate_thermo_parallel(dataloader, model, dp_processor):
    local_sums = torch.zeros(num_steps)
    local_counts = torch.zeros(num_steps)
    
    for batch in dp_processor.shard(dataloader):
        sums, counts = process_batch(batch, model)
        local_sums += sums
        local_counts += counts
    
    # Aggregate across all DP ranks
    global_sums = dp_processor.reduce_sum(local_sums)
    global_counts = dp_processor.reduce_sum(local_counts)
    
    return global_sums / global_counts
```

---

### Memory Optimization

```python
@dataclass
class MemoryConfig:
    gradient_checkpointing: bool = False    # Trade compute for memory
    offload_to_cpu: bool = False            # Offload inactive layers
    low_memory: bool = False                # Sequential layer processing
    flash_attention: bool = True            # Use flash attention if available
    compile_model: bool = False             # torch.compile() for speedup

# Usage
calc = nDNA(
    "meta-llama/Llama-3.1-70B",
    tensor_parallel=True,
    gradient_checkpointing=True,  # Reduce activation memory
    flash_attention=True,         # Faster, less memory
)
```

---

### Launch Scripts

**Single node, multiple GPUs (TP only):**
```bash
# Simple - library auto-detects GPUs
ndna calculate --model meta-llama/Llama-3.1-70B --tensor-parallel --tp-size 4 ...
```

**Multi-node with torchrun (DP):**
```bash
# Node 0
torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 \
    --master_addr=<IP> --master_port=29500 \
    -m ndna.cli calculate --model MODEL --data-parallel ...

# Node 1
torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 \
    --master_addr=<IP> --master_port=29500 \
    -m ndna.cli calculate --model MODEL --data-parallel ...
```

**SLURM:**
```bash
#!/bin/bash
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=32

srun python -m ndna.cli calculate \
    --model meta-llama/Llama-3.1-405B \
    --tensor-parallel --tp-size 8 \
    --data-parallel \
    --dataset squad \
    --output results/
```

---

## Default Prompts

```python
# data/prompts.py

DEFAULT_PROMPTS = [
    ("English simple", 
     "The artist drew a landscape with a river flowing towards the mountains."),
    
    ("Hindi simple", 
     "इसे आज़माने के लिए, नीचे अपनी भाषा और इनपुट उपकरण चुनें और लिखना आरंभ करें|"),
    
    ("LaTeX: Cauchy-Schwarz",
     r"**The CS- Inequality**\$$\left( \sum_{k=1}^n a_k b_k \right)^2 \leq ...$$"),
    
    ("Sanskrit line", 
     "श्वः अतीव द्रुतं धावति"),
    
    ("QFT path integral",
     r"Z = \int \mathcal{D}\phi \, \exp \left( i \int d^4x ..."),
]

# Dataset text processors
def squad_processor(example):
    """Build prompt from SQuAD example."""
    q = example["question"].strip().replace("\n", " ")
    c = example["context"].strip().replace("\n", " ")
    return {"text": f"Question: {q}\nContext: {c}\nAnswer:"}
```

---

## Example Usage

### Python API

```python
from ndna import nDNA, calculate_all_metrics

# Quick one-liner
results = calculate_all_metrics(
    model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
    dataset_name="squad",
    output_dir="results/deepseek"
)

# Or with more control
calc = nDNA("meta-llama/Llama-3.1-8B", device="cuda")

# Calculate spectral curvature for custom prompts
spectral = calc.calculate_spectral([
    ("English", "Hello, how are you?"),
    ("Code", "def factorial(n): return 1 if n <= 1 else n * factorial(n-1)")
])

# Calculate thermodynamic on a dataset
thermo = calc.calculate_thermodynamic("squad", max_samples=1000)

# Calculate belief vectors
belief = calc.calculate_belief("squad", max_samples=1000)

# Access per-layer data
print(f"Spectral curvatures: {spectral['English'].curvatures}")
print(f"FR step lengths: {thermo.step_lengths}")
print(f"Belief norms: {belief.belief_norms}")
```

### Python API with Parallelism

```python
from ndna import nDNA
from ndna.parallel import DataParallelProcessor

# ===== TENSOR PARALLELISM (Large Models) =====

# Load 70B model across 4 GPUs
calc = nDNA(
    "meta-llama/Llama-3.1-70B",
    tensor_parallel=True,
    tp_size=4,
)
results = calc.calculate_all(dataset_name="squad")

# Load 405B with CPU offloading
calc = nDNA(
    "meta-llama/Llama-3.1-405B",
    tensor_parallel=True,
    tp_size=8,
    offload_to_cpu=True,
    gradient_checkpointing=True,
)

# Custom memory limits per GPU
calc = nDNA(
    "meta-llama/Llama-3.1-70B",
    tensor_parallel=True,
    max_memory={"cuda:0": "40GiB", "cuda:1": "40GiB", "cuda:2": "40GiB", "cuda:3": "40GiB"}
)

# ===== DATA PARALLELISM (Fast Processing) =====

# Initialize distributed processing (run with torchrun)
dp = DataParallelProcessor(backend="nccl")
dp.setup()

calc = nDNA("meta-llama/Llama-3.1-8B", device=f"cuda:{dp.local_rank}")
results = calc.calculate_all(
    dataset_name="squad",
    max_samples=10000,
    dp_processor=dp,  # Distribute batches across GPUs
)

# Only main process saves/plots
if dp.is_main:
    calc.save(results, "results/")
    calc.plot(results, "plots/")

# ===== COMBINED TP + DP =====

# 70B model on 8 GPUs + distributed processing
calc = nDNA(
    "meta-llama/Llama-3.1-70B",
    tensor_parallel=True,
    tp_size=8,
)

dp = DataParallelProcessor()
dp.setup()

results = calc.calculate_all(
    dataset_name="squad",
    max_samples=50000,
    dp_processor=dp,
)
```

### CLI

```bash
# Full analysis
ndna calculate --model deepseek-ai/DeepSeek-R1-Distill-Qwen-14B --dataset squad --output results/

# Just spectral curvature
ndna spectral --model MODEL --output spectral.npz

# Plot existing results
ndna plot --input results/deepseek --output plots/
```

### CLI with Parallelism

```bash
# Tensor parallel: 70B model on 4 GPUs
ndna calculate \
    --model meta-llama/Llama-3.1-70B \
    --tensor-parallel --tp-size 4 \
    --dataset squad \
    --output results/llama70b/

# Data parallel: distribute across 4 GPUs (use torchrun)
torchrun --nproc_per_node=4 -m ndna.cli calculate \
    --model meta-llama/Llama-3.1-8B \
    --data-parallel \
    --dataset squad --max-samples 10000 \
    --output results/

# Combined: 70B model + distributed processing
torchrun --nproc_per_node=8 -m ndna.cli calculate \
    --model meta-llama/Llama-3.1-70B \
    --tensor-parallel --tp-size 8 \
    --data-parallel \
    --dataset squad --max-samples 50000 \
    --output results/

# Memory-optimized for 405B
ndna calculate \
    --model meta-llama/Llama-3.1-405B \
    --tensor-parallel --tp-size 8 \
    --gradient-checkpointing \
    --offload-to-cpu \
    --low-memory \
    --max-memory 80GiB \
    --output results/

# Multi-node SLURM job
srun --nodes=4 --gpus-per-node=8 \
    python -m ndna.cli calculate \
    --model meta-llama/Llama-3.1-405B \
    --tensor-parallel --tp-size 8 \
    --data-parallel \
    --output results/
```

---

## Dependencies

```
# requirements.txt

# Core
torch>=2.0.0
transformers>=4.30.0
datasets>=2.10.0
numpy>=1.21.0
tqdm>=4.60.0

# Visualization
matplotlib>=3.5.0
plotly>=5.0.0

# CLI
click>=8.0.0

# Parallelism (optional but recommended)
accelerate>=0.20.0         # Tensor parallelism, device_map, mixed precision
# flash-attn>=2.0.0        # Optional: flash attention (requires CUDA)

# For multi-node (install separately based on your cluster)
# torch distributed is included in torch
# mpi4py                   # For MPI backend
```

**Parallelism-specific installs:**

```bash
# Basic parallelism support
pip install accelerate

# Flash attention (NVIDIA GPUs, requires CUDA toolkit)
pip install flash-attn --no-build-isolation

# For large-scale distributed training
pip install deepspeed  # Optional: advanced memory optimization
```

---

## Extensibility

### Adding New Models

The library auto-detects most HuggingFace models. For custom architectures, just add an entry to the **model registry**:

```python
# models/registry.py

MODEL_REGISTRY = {
    # Architecture name → component paths
    "llama": {
        "model_attr": "model",           # model.model
        "layers_attr": "layers",         # model.model.layers
        "norm_attr": "norm",             # model.model.norm
        "lm_head_attr": "lm_head",       # model.lm_head
        "embed_attr": "embed_tokens",    # model.model.embed_tokens
    },
    "gpt2": {
        "model_attr": "transformer",
        "layers_attr": "h",
        "norm_attr": "ln_f",
        "lm_head_attr": "lm_head",
        "embed_attr": "wte",
        "pos_embed_attr": "wpe",         # Optional: position embeddings
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
    "mpt": {
        "model_attr": "transformer",
        "layers_attr": "blocks",
        "norm_attr": "norm_f",
        "lm_head_attr": "lm_head",  # or None if tied
        "embed_attr": "wte",
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
}

# Auto-detection: model class name contains → use this architecture
ARCH_DETECTION = {
    "llama": ["llama", "vicuna", "alpaca", "codellama", "tinyllama"],
    "gpt2": ["gpt2", "dialogpt", "gpt-neo", "gpt-j", "pythia", "cerebras"],
    "phi": ["phi"],
    "qwen": ["qwen"],
    "mistral": ["mistral", "mixtral", "zephyr"],
    "deepseek": ["deepseek"],
    "gemma": ["gemma"],
    "falcon": ["falcon"],
    "mpt": ["mpt"],
    "opt": ["opt"],
    "bloom": ["bloom", "bloomz"],
}

# Pre-configured popular models (no setup needed)
KNOWN_MODELS = [
    # LLaMA family
    "meta-llama/Llama-3.1-8B", "meta-llama/Llama-3.1-70B",
    "meta-llama/Llama-2-7b-hf", "meta-llama/Llama-2-13b-hf",
    # Mistral family
    "mistralai/Mistral-7B-v0.1", "mistralai/Mixtral-8x7B-v0.1",
    # Qwen family
    "Qwen/Qwen2-7B", "Qwen/Qwen2-72B",
    # DeepSeek
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
    "deepseek-ai/deepseek-llm-7b-base",
    # Phi
    "microsoft/phi-2", "microsoft/Phi-3-mini-4k-instruct",
    # Others
    "google/gemma-7b", "tiiuae/falcon-7b",
    "EleutherAI/gpt-neo-2.7B", "EleutherAI/pythia-6.9b",
    "mosaicml/mpt-7b", "facebook/opt-6.7b",
]
```

**To add a new model architecture:**

```python
# Option 1: Add to registry (recommended)
from ndna.models.registry import MODEL_REGISTRY

MODEL_REGISTRY["my_model"] = {
    "model_attr": "transformer",
    "layers_attr": "blocks",
    "norm_attr": "final_norm",
    "lm_head_attr": "head",
    "embed_attr": "embeddings",
}

# Option 2: Pass config directly when loading
from ndna import nDNA

calc = nDNA(
    "my-org/my-model",
    model_config={
        "model_attr": "transformer",
        "layers_attr": "blocks",
        "norm_attr": "final_norm",
        "lm_head_attr": "head",
        "embed_attr": "embeddings",
    }
)
```

**That's it!** The library will use these paths to access model components.

---

### Adding New Datasets

Datasets need a **text processor** function that converts raw examples to `{"text": str}`.

```python
# data/registry.py

DATASET_REGISTRY = {
    # Dataset name → config
    "squad": {
        "hf_name": "squad",
        "split": "train",
        "processor": lambda ex: {
            "text": f"Question: {ex['question']}\nContext: {ex['context']}\nAnswer:"
        },
    },
    "imdb": {
        "hf_name": "imdb",
        "split": "train",
        "text_column": "text",  # Direct column access, no processor needed
    },
    "wikitext": {
        "hf_name": "wikitext",
        "config": "wikitext-2-raw-v1",
        "split": "train",
        "text_column": "text",
    },
    "dolly": {
        "hf_name": "databricks/databricks-dolly-15k",
        "split": "train",
        "processor": lambda ex: {
            "text": f"Instruction: {ex['instruction']}\nContext: {ex['context']}\nResponse: {ex['response']}"
        },
    },
    "alpaca": {
        "hf_name": "tatsu-lab/alpaca",
        "split": "train",
        "processor": lambda ex: {
            "text": f"### Instruction:\n{ex['instruction']}\n\n### Input:\n{ex['input']}\n\n### Response:\n{ex['output']}"
        },
    },
    
    # Question Answering
    "triviaqa": {
        "hf_name": "trivia_qa",
        "config": "unfiltered",
        "split": "train",
        "processor": lambda ex: {
            "text": f"Question: {ex['question']}\nAnswer: {ex['answer']['value']}"
        },
    },
    "natural_questions": {
        "hf_name": "natural_questions",
        "split": "train",
        "processor": lambda ex: {"text": ex['question']['text']},
    },
    
    # Code
    "code_search_net": {
        "hf_name": "code_search_net",
        "config": "python",
        "split": "train",
        "processor": lambda ex: {"text": ex['func_code_string']},
    },
    "the_stack": {
        "hf_name": "bigcode/the-stack",
        "config": "python",
        "split": "train",
        "text_column": "content",
    },
    
    # Conversations
    "openassistant": {
        "hf_name": "OpenAssistant/oasst1",
        "split": "train",
        "text_column": "text",
    },
    "sharegpt": {
        "hf_name": "anon8231489123/ShareGPT_Vicuna_unfiltered",
        "split": "train",
        "processor": lambda ex: {"text": str(ex['conversations'])},
    },
    
    # Scientific
    "pubmed": {
        "hf_name": "pubmed",
        "split": "train",
        "processor": lambda ex: {"text": ex['MedlineCitation']['Article']['Abstract']['AbstractText']},
    },
    "arxiv": {
        "hf_name": "ccdv/arxiv-summarization",
        "split": "train",
        "text_column": "abstract",
    },
    
    # Multi-task
    "flan": {
        "hf_name": "Muennighoff/flan",
        "split": "train",
        "processor": lambda ex: {"text": f"{ex['inputs']}\n{ex['targets']}"},
    },
    
    # Simple text
    "ag_news": {
        "hf_name": "ag_news",
        "split": "train",
        "text_column": "text",
    },
    "yelp": {
        "hf_name": "yelp_review_full",
        "split": "train",
        "text_column": "text",
    },
    "amazon": {
        "hf_name": "amazon_polarity",
        "split": "train",
        "text_column": "content",
    },
}
```

**To add a new dataset:**

```python
# Option 1: Add to registry
from ndna.data.registry import DATASET_REGISTRY

DATASET_REGISTRY["my_dataset"] = {
    "hf_name": "my-org/my-dataset",
    "split": "train",
    "processor": lambda ex: {"text": f"{ex['prompt']}: {ex['response']}"},
}

# Then use it
from ndna import nDNA
calc = nDNA("model-name")
results = calc.calculate_all(dataset_name="my_dataset")
```

```python
# Option 2: Use inline processor
calc.calculate_thermodynamic(
    dataset_name="my-org/my-dataset",
    text_processor=lambda ex: {"text": f"{ex['prompt']}: {ex['response']}"}
)
```

```python
# Option 3: Load custom texts directly
texts = ["Hello world", "How are you?", ...]
results = calc.calculate_thermodynamic(texts=texts)
```

```python
# Option 4: Load from local files
calc.calculate_thermodynamic(file_path="data.txt")           # One text per line
calc.calculate_thermodynamic(file_path="data.jsonl")         # JSONL with "text" field
calc.calculate_thermodynamic(file_path="data.csv", text_column="content")
```

**Common processor patterns:**

```python
# Instruction format
processor = lambda ex: {"text": f"### Instruction: {ex['instruction']}\n### Response: {ex['output']}"}

# Q&A format  
processor = lambda ex: {"text": f"Q: {ex['question']}\nA: {ex['answer']}"}

# Chat format
processor = lambda ex: {"text": f"User: {ex['user']}\nAssistant: {ex['assistant']}"}

# Concatenate fields
processor = lambda ex: {"text": f"{ex['title']}\n\n{ex['body']}"}

# Filter by field
processor = lambda ex: {"text": ex['text']} if ex['language'] == 'en' else None
```

---

### Adding New Prompts (for Spectral Curvature)

Prompts are just `(label, text)` tuples:

```python
# data/prompts.py

PROMPT_SETS = {
    "default": [
        ("English simple", "The artist drew a landscape with a river."),
        ("Hindi simple", "इसे आज़माने के लिए, नीचे अपनी भाषा चुनें|"),
        ("LaTeX", r"\sum_{i=1}^n x_i^2"),
        ("Code", "def fib(n): return n if n <= 1 else fib(n-1) + fib(n-2)"),
    ],
    "multilingual": [
        ("English", "Hello, how are you?"),
        ("Spanish", "Hola, ¿cómo estás?"),
        ("French", "Bonjour, comment allez-vous?"),
        ("German", "Hallo, wie geht es Ihnen?"),
        ("Chinese", "你好，你好吗？"),
        ("Japanese", "こんにちは、お元気ですか？"),
        ("Arabic", "مرحباً، كيف حالك؟"),
        ("Hindi", "नमस्ते, आप कैसे हैं?"),
    ],
    "technical": [
        ("Python", "def quicksort(arr): ..."),
        ("SQL", "SELECT * FROM users WHERE age > 18;"),
        ("LaTeX Math", r"E = mc^2"),
        ("JSON", '{"name": "John", "age": 30}'),
    ],
}
```

**Use custom prompts:**

```python
# Use a preset
calc.calculate_spectral(prompt_set="multilingual")

# Or pass directly
calc.calculate_spectral(prompts=[
    ("My prompt 1", "Some text here"),
    ("My prompt 2", "Other text here"),
])
```

---

### Full Extensibility Example

```python
from ndna import nDNA
from ndna.models.registry import MODEL_REGISTRY
from ndna.data.registry import DATASET_REGISTRY

# 1. Add custom model architecture
MODEL_REGISTRY["my_arch"] = {
    "model_attr": "backbone",
    "layers_attr": "blocks",
    "norm_attr": "norm",
    "lm_head_attr": "output_head",
    "embed_attr": "input_embed",
}

# 2. Add custom dataset
DATASET_REGISTRY["my_data"] = {
    "hf_name": "my-org/my-dataset",
    "split": "train",
    "processor": lambda ex: {"text": ex["content"]},
}

# 3. Define custom prompts
my_prompts = [
    ("Domain 1", "Text for domain 1..."),
    ("Domain 2", "Text for domain 2..."),
]

# 4. Run analysis
calc = nDNA("my-org/my-model")
results = calc.calculate_all(
    dataset_name="my_data",
    prompts=my_prompts,
    output_dir="results/my_analysis"
)
```

---

### Extensibility Summary

| What | How | Complexity |
|------|-----|------------|
| Add new model | Dict in `MODEL_REGISTRY` | 5 lines |
| Add new dataset | Dict in `DATASET_REGISTRY` | 3-5 lines |
| Custom text processor | Lambda function | 1 line |
| Custom prompts | List of tuples | N lines |
| Use local texts | `texts=[...]` parameter | 1 line |
| Use local files | `file_path="..."` | 1 line |

**Design principles:**
1. **Registry-based**: Just add dicts, no subclassing
2. **Convention over configuration**: Auto-detect when possible
3. **Override when needed**: Pass configs directly to bypass registry
4. **Functions over classes**: Simple lambdas for text processing

---

### Quick Reference: Supported Out-of-Box

**Models** (auto-detected, no config needed):

| Family | Examples |
|--------|----------|
| LLaMA | `meta-llama/Llama-3.1-8B`, `meta-llama/Llama-2-7b-hf`, `codellama/*` |
| Mistral | `mistralai/Mistral-7B-v0.1`, `mistralai/Mixtral-8x7B-v0.1` |
| Qwen | `Qwen/Qwen2-7B`, `Qwen/Qwen2.5-72B` |
| DeepSeek | `deepseek-ai/DeepSeek-R1-Distill-Qwen-14B`, `deepseek-ai/deepseek-coder-*` |
| Phi | `microsoft/phi-2`, `microsoft/Phi-3-*` |
| Gemma | `google/gemma-7b`, `google/gemma-2b` |
| GPT-2 | `gpt2`, `gpt2-xl`, `microsoft/DialoGPT-*` |
| Falcon | `tiiuae/falcon-7b`, `tiiuae/falcon-40b` |
| Pythia | `EleutherAI/pythia-*` |
| OPT | `facebook/opt-*` |
| BLOOM | `bigscience/bloom-*` |
| MPT | `mosaicml/mpt-*` |

**Datasets** (ready to use):

| Category | Datasets |
|----------|----------|
| QA | `squad`, `triviaqa`, `natural_questions` |
| Text | `imdb`, `wikitext`, `ag_news`, `yelp`, `amazon` |
| Instruction | `alpaca`, `dolly`, `flan`, `openassistant` |
| Code | `code_search_net`, `the_stack` |
| Scientific | `pubmed`, `arxiv` |

**Prompts** (preset collections):

| Set | Languages/Domains |
|-----|-------------------|
| `default` | English, Hindi, LaTeX, Code, Sanskrit |
| `multilingual` | EN, ES, FR, DE, ZH, JA, AR, HI |
| `technical` | Python, SQL, LaTeX, JSON |

---

## Implementation Notes

1. **Spectral Curvature** operates on individual texts at the last token position - NOT batched over dataset
2. **Thermodynamic & Belief** operate on batched dataset with teacher-forcing
3. **Logit Lens** is central - apply `final_norm + lm_head` at intermediate layers
4. **Per-sample collection** enables 3D visualization (sample × layer × metric)
5. **All metrics return per-layer arrays** for detailed analysis
6. **Storage format**: NPZ for arrays + JSON for metadata
7. **Extensibility**: Registry-based, no subclassing required
