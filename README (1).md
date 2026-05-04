<div align="center">

# MENTIS - What Belief Changes Under Alignment? Measuring Multi-Scale Latent Torsion Across Four Language Models
####
Representational Latent Torsion across four LLMs

**[📄 Paper](#citation) · [🚀 Quick Start](#-quick-start)**

<br/>

> *"Alignment is not just a loss function. It is a change in the shape of thought."*

</div>

---

## 🔬 What is MENTIS?

**MENTIS** (**M**ulti-scale **E**pistemic **N**ormative **T**orsion **I**nterpretability **S**ystem) is a **representational geometry-first interpretability framework** that answers a fundamental question:

> **What, geometrically, does alignment change inside a language model — and what does it leave intact?**

While behavioral evaluation tells us *what* aligned models output, MENTIS reveals *how* alignment reshapes the model's **internal belief geometry** — the layerwise geometric structure of hidden-state trajectories under semantically rich prompts.

### The Core Insight

We operationalize **belief change** as **multi-scale latent torsion**: the cumulative directional twisting of hidden-state trajectories across transformer depth when processing semantically charged concepts. By comparing an Instruction-Tuned (IT) model to its Preference-Aligned (PA) variant on the same prompt, MENTIS measures:

$$\Delta_{\text{belief}}(x;\, \theta_{\text{IT}}, \theta_{\text{PA}}) = \mathcal{T}^{(\text{PA})}(x) - \mathcal{T}^{(\text{IT})}(x)$$

$$\mathcal{T}^{(m)}(x) = \sum_{\ell} \arccos\!\left(\hat{v}_{\ell}^{(m)} \cdot \hat{v}_{\ell+1}^{(m)}\right) \quad \text{(cumulative angular torsion of the belief field trajectory)}$$

---

## 🗺️ Torsion Cartography

<div align="center">
<img src="./figures/FigA_torsion_cartography_updated.png" alt="MENTIS Torsion with belief fields in 3 buckets for instruction-tuned and preference-aligned models" width="90%"/>

*Layer × concept torsion heatmap (OLMo IT→PA). Torsion concentrates in late layers (ℓ≥24) on normatively charged concepts — not uniformly.*
</div>

---

## ⚡ Four Key Findings

| # | Finding | Result |
|---|---------|--------|
| **I** | **Concept-selective torsion** | T2 spectral torsion CV = **0.64** — normative concepts (drugs, peace, justice) show **4–8×** more torsional reorganization than factual ones (war, fraud) |
| **II** | **Entropy–torsion anti-correlation** | ρ = −0.387 (Mistral, p=5.43×10⁻³⁰) and ρ = −0.164 (OLMo, p=3.04×10⁻⁶) — high-entropy words are systematically targeted by alignment |
| **III** | **Safe-prompt paradox** | Safe prompts consistently drive **larger** torsion deltas than unsafe prompts (t=4.32, p<0.001) — alignment protects representational territory, not just outputs |
| **IV** | **Architecture-conditioned depth** | Torsional updates peak at model-specific late layers — OLMo and Tülu ℓ★=25–30, Mistral ℓ★≈14-19, Llama ℓ★≈20 (all out of 32 layers) |

---

## 📐 The MENTIS Three-Scale Torsion Battery (T, T1, T2)

MENTIS instruments belief change across **three complementary torsion scales** — directional drift, rotational energy, and spectral anisotropy — empirically validated as non-collinear (T1/T2 Pearson r=0.31, p=0.003):

| ID | Metric | Geometric Dimension | Formula | Complexity |
|----|--------|---------------------|---------|------------|
| **T** | Angular Torsion | Directional drift (cumulative) | $\sum_\ell \arccos(\hat{v}_\ell \cdot \hat{v}_{\ell+1})$ | O(Ld) |
| **T1** | Torsion Norm | Rotational energy per layer | $\|S_\ell\|_F$ | O(Lk²) |
| **T2** | Spectral Torsion | Eigenspectrum anisotropy | $\text{Var}(\|\mu_k\|)$ | O(Lk³) |
| **ERA** | Depth-Layer Profile | Peak-layer localization | DTW(φ_IT, φ_PA) | O(L²) |

> **Scale complementarity:** T captures where belief trajectories *bend* (direction); T1 captures *how hard* they rotate (magnitude); T2 captures whether the rotation has *structured spectral content* (shape). All three are necessary for a complete characterization.

---

## 🧪 Models & Benchmark

### Model Pairs (IT → PA)

| Family | IT Checkpoint | PA Checkpoint |
|--------|--------------|---------------|
| **OLMo** | `allenai/OLMo-2-1124-7B-SFT` | `allenai/OLMo-2-1124-7B-DPO` |
| **Mistral/Zephyr** | `alignment-handbook/zephyr-7b-sft-full` | `alignment-handbook/zephyr-7b-dpo-full` |
| **Llama-3.1** | `allenai/Llama-3.1-Tulu-3-8B-SFT` | `allenai/Llama-3.1-Tulu-3-8B-DPO` |
| **Tülu-3** | `allenai/Llama-3.1-Tulu-3-8B-SFT` | `allenai/Llama-3.1-Tulu-3-8B-DPO` |

### LITMUS Benchmark

- **20,439** safety-labeled prompts across **7 human-value axioms** (ValueImprint taxonomy)
- **799** high-entropy sentences selected for layerwise torsion profiling
- **18** concept words spanning normative (protest, peace, justice, hate, violence, drugs) and factual (war, fraud, order) domains
- **36** prompt-concept pairs (2 prompts × 18 concepts) for torsion battery analysis

---

## 📦 Installation

```bash
git clone https://github.com/pps121/torsional-belief-vector-field.git
cd torsional-belief-vector-field
pip install -r requirements.txt
```

For GPU-accelerated runs (A100 recommended):
```bash
pip install -r requirements.txt
# Ensure CUDA 12.1+ and PyTorch 2.1+ with bfloat16 support
```

For Hugging Face gated models (Llama-3.1-Tulu), set your token:
```bash
export HF_TOKEN=your_hf_token_here
```

---

## 🚀 Quick Start

### 1. Torsion Cartography (Word-Level)

```python
from src.belief_torsion import compute_torsion_norm, compute_angular_torsion
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load IT and PA model pair
it_model = AutoModelForCausalLM.from_pretrained("allenai/OLMo-2-1124-7B-SFT",
                                                output_hidden_states=True, torch_dtype="bfloat16")
pa_model = AutoModelForCausalLM.from_pretrained("allenai/OLMo-2-1124-7B-DPO",
                                                output_hidden_states=True, torch_dtype="bfloat16")
tokenizer = AutoTokenizer.from_pretrained("allenai/OLMo-2-1124-7B-SFT")

# LITMUS-style concept words
concepts = ["peace", "justice", "violence", "drugs", "war", "fraud", "hate", "order"]

# Compute torsion profiles
from src.alignment_belief_shift_study import run_alignment_study
results = run_alignment_study(it_model, pa_model, tokenizer, concepts=concepts)
results.plot_heatmap(save_path="figures/torsion_cartography.png")
```

### 2. Three-Scale Torsion Battery

```python
from src.novel_belief_metrics import MENTISBattery

battery = MENTISBattery(it_model, pa_model, tokenizer, k_pca=64)
results = battery.run(
    concepts=concepts,
    prompts_per_concept=2,
    metrics=["T", "T1", "T2"]
)
results.to_csv("data/mentis_torsion_results.csv")
results.plot_dashboard(save_path="figures/mentis_dashboard.png")
```

### 3. LITMUS Entropy–Torsion Correlation

```python
from scripts.litmus_belief_pipeline import compute_entropy_torsion_correlation

# Requires 799 LITMUS high-entropy sentences (see data/)
rho, pval, n = compute_entropy_torsion_correlation(
    model_pair=("allenai/OLMo-2-1124-7B-SFT", "allenai/OLMo-2-1124-7B-DPO"),
    litmus_sentences="data/litmus_800_sentences.txt"
)
print(f"ρ = {rho:.3f}, p = {pval:.2e}, n = {n}")
# Expected: ρ = -0.164, p = 3.04e-06, n = 799  (OLMo)
# Expected: ρ = -0.387, p = 5.43e-30, n = 799  (Mistral)
```

### 4. 3D Belief Ribbon Visualization

```python
# Generate publication-quality interactive 3D ribbon plots
python scripts/generate_figF_ribbons.py \
    --models olmo mistral llama \
    --output_dir figures/ribbons/ \
    --format html  # interactive plotly, or png for static
```

---

## 📓 Notebooks

| Notebook | Description | Platform |
|----------|-------------|----------|
| [`notebooks/MENTIS_Colab_A100.ipynb`](./notebooks/) | Full pipeline: hidden states → 3 torsion scales → all figures | Google Colab (A100) |
| [`notebooks/MENTIS_QuickDemo.ipynb`](./notebooks/) | 10-min demo on OLMo pair, word-level torsion | Colab (T4/CPU) |
| [`notebooks/LITMUS_Entropy_Analysis.ipynb`](./notebooks/) | Entropy–torsion correlation for all model families | Colab (A100) |

> **Open in Colab:** Click any notebook badge to run instantly — no local setup required.

---

## 📁 Repository Structure

```
torsional-belief-vector-field/
│
├── 📄 README.md                          # This file
├── 📄 CITATION.cff                       # Machine-readable citation
├── 📄 LICENSE                            # MIT License
├── 📄 requirements.txt                   # Python dependencies
│
├── 📁 src/                               # Core library
│   ├── belief_torsion.py                 # Angular torsion T, T1, T2 computation
│   ├── novel_belief_metrics.py           # MENTIS torsion battery
│   ├── alignment_belief_shift_study.py   # IT vs PA comparative analysis
│   ├── dtw_depthwise_belief_analysis.py  # DTW trajectory comparison
│   ├── belief_entropy_dynamics.py        # Entropy–torsion bridge (logit lens)
│   ├── belief_circuit_tracing.py         # Mechanistic circuit analysis
│   ├── belief_torsion_3d_plots.py        # 3D Riemannian manifold visualizations
│   └── evaluate_11_methods.py            # Full statistical evaluation
│
├── 📁 scripts/                           # Standalone analysis scripts
│   ├── generate_figF_ribbons.py          # 3D belief ribbon plots (FigF in paper)
│   ├── litmus_belief_pipeline.py         # Full LITMUS corpus analysis
│   ├── belief_cartography_plots.py       # Layer × concept heatmaps (FigA in paper)
│   └── belief_steering_analysis.py       # Alignment steering strength analysis
│
├── 📁 notebooks/                         # Colab-ready Jupyter notebooks
│   ├── MENTIS.ipynb
│   ├── MENTIS_QuickDemo.ipynb            # 10-min quick demo
│   └── LITMUS_Entropy_Analysis.ipynb     # LITMUS entropy–torsion study
│
├── 📁 configs/                           # Experiment configurations
│   ├── models.yaml                       # Model IDs and checkpoint paths
│   └── litmus.yaml                       # LITMUS benchmark configuration
│
├── 📁 figures/                           # Key paper figures
│   └── FigA_torsion_cartography.png      # Main torsion heatmap (Figure 1)
│
├── 📁 data/                              # Pre-computed results
│   ├── statistical_results.csv           # Per-layer t-tests (OLMo, Mistral)
│   ├── summary_stats.csv                 # Quartile torsion breakdown
│   └── dtw_all_metrics.csv               # 18-concept × 3-scale DTW results
```

---

## 📊 Reproducing Key Results

### Torsion Battery (T, T1, T2): Concept-Level Summary

```bash
python scripts/novel_advanced_metrics.py \
    --model_it   allenai/OLMo-2-1124-7B-SFT \
    --model_pa   allenai/OLMo-2-1124-7B-DPO \
    --concepts   data/concepts_18.txt \
    --output_csv data/OLMo_torsion_results.csv \
    --n_pca      64
```

### Sentence-Level Torsion by Value Axiom (All Model Pairs)

```bash
python scripts/litmus_belief_pipeline.py \
    --model_family olmo mistral llama tulu \
    --n_prompts 500 \
    --stratify_by_axiom \
    --output_dir results/axiom_torsion/
```

### Figure F: 3D Belief Ribbons (4 Model Families)

```bash
python scripts/generate_figF_ribbons.py \
    --models olmo mistral llama tulu \
    --content_words_only \
    --output_dir figures/ribbons/ \
    --format html
# Outputs: FigF_OLMo_3d_ribbon.html, FigF_Mistral_3d_ribbon.html, ...
```

### Entropy–Torsion Correlation (Figure H)

```bash
python scripts/litmus_belief_pipeline.py \
    --task entropy_torsion_correlation \
    --n_sentences 799 \
    --models olmo mistral \
    --output_dir results/entropy_torsion/
# Expected: OLMo ρ=-0.164 (p=3.04e-6), Mistral ρ=-0.387 (p=5.43e-30)
```

---

## 📈 Prior Results: Cohen's *d* Scaling Across Depth

From the earlier TBVF analysis (OLMo/Mistral, 500 prompts):

| Regime | Layers | Cohen's *d* | Interpretation |
|--------|--------|-------------|----------------|
| **I: Null** | 1–10 | *d* < 0.1 | Pre-semantic; no alignment signal |
| **II: Build** | 11–19 | 0.1–0.3 | Alignment signal accumulating |
| **III: Brake** | 20–28 | 0.3–0.7 | **Geometric suppression active** |
| **IV: Release** | 29–31 | decreasing | Output projection normalization |

Peak significance: **Layer 27**, Cohen's *d* = 0.741, *p* = 7.7×10⁻¹³ (Bonferroni-corrected).

---

## 🧾 Data

Pre-computed result files are in `data/`. All are CSV format, immediately loadable:

```python
import pandas as pd
# Torsion battery for OLMo IT→PA (18 concepts × 3 torsion scales)
df = pd.read_csv("data/OLMo_torsion_results.csv")

# LITMUS entropy-torsion results (799 sentences × all models)
df_litmus = pd.read_csv("data/dtw_all_metrics.csv")
```

---

## 🔗 Related Work & References

| Work | Relation to MENTIS |
|------|--------------------|
| [Shai et al. 2024](https://arxiv.org/abs/2406.04837) | Representational belief geometry (static) — MENTIS is comparative |
| [Zou et al. 2023](https://arxiv.org/abs/2310.01405) | Representation engineering — MENTIS entropy–torsion correlation links to steerable directions |
| [Meng et al. 2022](https://arxiv.org/abs/2202.05262) | ROME/MEMIT localized edits — MENTIS measures distributed alignment-wide geometry |
| [Obi et al. 2024](https://arxiv.org/abs/2406.11276) | ValueImprint taxonomy — provides the 7-axiom partition for LITMUS benchmark |
| [Groeneveld et al. 2024](https://arxiv.org/abs/2402.00838) | OLMo-2 — primary IT/PA model pair for torsion battery |

---

<!-- ## 📄 Citation

If you use MENTIS in your research, please cite:

```bibtex
@inproceedings{saha2026mentis,
  title     = {MENTIS: What Belief Changes Under Alignment?
               Measuring Multi-Scale Latent Torsion in Language Models},
  author    = {Saha, Partha Pratim},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2026},
  note      = {Under review},
  url       = {https://github.com/pps121/torsional-belief-vector-field}
}
```

Or use the [CITATION.cff](./CITATION.cff) file (GitHub auto-detects this for "Cite this repository").

----->

<div align="center">

**⭐ Star this repo if you find geometric interpretability of alignment interesting!**

*Built with ❤️ for the interpretability community.*

[![Star History](https://img.shields.io/github/stars/pps121/torsional-belief-vector-field?style=social)](https://github.com/pps121/torsional-belief-vector-field/stargazers)

</div>
