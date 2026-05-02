# Experimental Results

This section contains the experimental results from the NeuralGenomics comprehensive evaluation pipeline. 

We evaluated different models (Base, SFT, DPO, and Distilled variants) across multiple dimensions:
1. **Geometric Diagnostics (NDNA)**: Tracking thermodynamic length, belief vector field magnitudes, and spectral curvature on sampling points (NDNA 1000 and 2500) over datasets like SQuAD.
2. **Safety & Alignment**: Measured win-rates and safety scores across HarmBench, AdvBench, and XSTest using our `alignment-SFT-DPO-eval-pipeline`.
3. **Instruction Following**: Evaluated generative capabilities using programmatic verification with IFEval and LLM-as-judge scoring on MT-Bench.
4. **Knowledge Distillation**: Validating parameter-efficient Math distillation capabilities for Llama and Qwen models.

## Detailed LLM Curve Similarity Report
The interactive report below details the geometrical trajectory overlaps across models based on our NDNA methodologies.
<iframe src="ndna_pragya/results/2500/LLM_curve_similarity_report.html" width="100%" height="800px" style="border:none;" title="LLM Curve Similarity Report"></iframe>

---
### Navigation
- [Back to Main README](README.md)
- [View 3D Plots](PLOTS.md)
