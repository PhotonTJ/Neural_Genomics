---
license: llama3
---

# NeuralGenomics (NDNA)

Welcome to the NeuralGenomics codebase! This repository contains the full end-to-end implementations for our paper, covering everything from core geometrical analysis (NDNA) to model alignment, distillation, and extensive evaluations.

## Repository Structure

The codebase is organized into four main components:

1. **`ndna_pragya` (Core Methodology & Analytics)**
   - The core geometrical toolkit for decoder-only language models.
   - Computes Fisher-style effort, prediction-space thermodynamic length, belief vector field magnitudes, spectral curvature, and composite nDNA fingerprints.
   - Features robust visualization dashboards (see [3D Trajectory Plots](PLOTS.md) and [Results](RESULTS.md)).

2. **`SFT` (Supervised Fine-Tuning & DPO)**
   - Contains pipelines to fine-tune base models (e.g., Llama 3, Qwen 2.5) via Supervised Fine-Tuning.
   - Implements Direct Preference Optimization (DPO) for aligning models with human preferences.
   - Includes utility scripts for efficiently merging PEFT/LoRA adapters back into base models.

3. **`distillation` (Knowledge Distillation)**
   - Task-specific (e.g., Math) knowledge distillation pipelines.
   - Utilizes parameter-efficient LoRA distillation approaches for Llama and Qwen architectures.

4. **`alignment-SFT-DPO-eval-pipeline` (Unified Evaluation)**
   - A comprehensive, LLM-as-judge (Qwen 32B) evaluation suite.
   - **Safety Benchmarks**: HarmBench, AdvBench, XSTest.
   - **Instruction Following**: IFEval, MT-Bench.
   - Supports testing base, SFT, and DPO models sequentially and outputs detailed cross-model comparative grids and `.csv` records.

## Results & Visualizations

We have dedicated pages to interactively explore our experimental results:
- **[3D Trajectory Plots](PLOTS.md)**: Explore interactive 3D graphs representing model trajectories directly in your browser.
- **[Experimental Results](RESULTS.md)**: Detailed similarity reports, curve comparisons, and quantitative performance records.

*Note: Model weights, checkpoints, large datasets, and intermediate binary representations are excluded from this repository (`.gitignore`) to keep the workspace clean and strictly codebase-focused.*
