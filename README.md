# ndna (detailed CLI reference)

---
Use the provided setup script to configure your environment, install all dependencies, and set up authentication for GitHub and HuggingFace.

**1. Make the script executable:**
```bash
chmod +x scripts/setup_env.sh
```

**2. Run the setup script:**
```bash
./scripts/setup_env.sh
```
Or, to keep the environment activated in your current shell:
```bash
source scripts/setup_env.sh
```

**3. Follow any prompts and verify the summary at the end.**

**4. To activate the environment in the future:**
```bash
conda activate ndna
```

---

ndna is a geometry-first toolkit for decoder-only language models. It measures Fisher-style effort, prediction-space thermodynamic length, belief vector field magnitudes, spectral curvature, and composite nDNA fingerprints across many datasets, model families, and fine-tuning regimes.

This repository also ships a full **model alignment and evaluation suite** covering three complementary pipelines:

| Pipeline | Directory | Purpose |
|---|---|---|
| **SFT & DPO** | `Alignment/` | Supervised fine-tuning and Direct Preference Optimization for aligning base models |
| **Knowledge Distillation** | `distillation/` | LoRA-based teacher-student distillation (e.g., math reasoning transfer) |

This README is intentionally exhaustive and serves as the single source of truth for every shipped CLI script, including collapse experiments, Method-5 variants, LoRA Fisher workflows, plotting dashboards, concreteness scoring, SFT/DPO training, and distillation.

The document is structured as a step-by-step operating manual. It starts with core ideas and metrics, then dives into each command with parameter-by-parameter notes, usage recipes, failure modes, and output conventions.

---

## Table of contents
- Quick mental model of ndna metrics
- Environment + dependency notes
- Model zoo and adapters
- Dataset formatting quick guide
- Command inventory (what each script does)
- Method-5 generic runner (`scripts/run_method5_generic.py`)
- Method-5 for LoRA and Fisher-merged adapters (`scripts/run_method5_lora.py`)
- Collapse driver (`scripts/run_collapse_from_zoo.py`)
- Fisher computation for LoRA adapters (`ndna_lib/merging/compute_fisher_lora.py`)
- Fisher-weighted merging of adapters (`ndna_lib/merging/fisher_merge_lora.py`)
- Plotting utilities (`scripts/plot.py`, `scripts/plot_collapse_pairs_3d.py`, `scripts/plot_qwen_ndna_5datasets.py`)
- Concreteness scoring (`scripts/concreteness_score.py`)
- SFT & DPO Workflows (`Alignment/`)
- Distillation Workflows (`distillation/`)
- End-to-end playbooks
- Troubleshooting and validation checklist

---

## Quick mental model of ndna metrics
- **E_l (parameter-space effort / observed Fisher)**: per-layer squared-gradient norm aggregated over sequences or tokens. Higher means the layer is working harder relative to its parameter count. Used for LoRA Fisher merging and to derive Eta.
- **FR step length Δ_l (thermodynamic length)**: Fisher–Rao distance between consecutive prediction distributions. Captures how much the model belief moves per token across layers.
- **Alignment α_l and belief push norm ||v̄_l||**: compares the theoretical belief update direction to what the model actually does; ||v̄_l|| is the magnitude of belief change per layer. Included in Method-5 outputs.
- **Spectral curvature κ_l**: curvature of the prediction manifold along teacher-forced token paths. Used in Method-5 generic (optional) and collapse spectral runs; shapes NDNA fingerprints.
- **nDNA_pred**: composite scalar and layerwise curves combining curvature, FR length, and belief norms (only when κ shape matches Δ).
- **Eta_l**: Δ_l^2 / E_l for collapse runs (FR energy normalized by effort). Saved as `Eta` inside collapse metrics.
- **Belief drift / thermo / spectral (Nephos)**: three metrics over prompt+response pairs comparing normal vs trigger-inserted prompts.

These metrics are computed via `ndna_lib.geometry` and related helpers. Adapters translate HF model internals into a uniform per-layer interface; tokenization follows strict causal LM formatting with padding and truncation controlled by CLI flags.

---

## Environment and dependency notes
- Python dependencies live in `requirements.txt`. Install with `pip install -r requirements.txt`.
- GPU is strongly recommended. Default tensor dtype is bfloat16 in most runners; FR and Fisher paths rely on autograd and will be slow on CPU.
- Quantization: MXFP4 is optionally enabled in Method-5 runners but is incompatible with parameter-space Fisher (`compute_param_effort`). The LoRA runner enforces `--skip-param-effort` when MXFP4 is on.
- Datasets are pulled from Hugging Face. Streaming mode is supported in Method-5 generic and Fisher computation to avoid large downloads.
- `HF_TOKEN` can be provided for gated models/datasets (Nephos runner forwards it automatically).
- File outputs use compressed `.npz` for Method-5 and Nephos runs, JSON for concreteness and collapse metrics, and Plotly HTML for dashboards.

---

## Model zoo and adapter notes
- `model_zoo.json` holds alias → HF id mappings plus an `enabled` flag. Method-5 generic resolves models from this file unless `--hf-models` is provided.
- Collapse driver reads the zoo and runs only entries with `enabled: true` unless filtered by `--model-keys`.
- LoRA runner defaults to the Qwen3-4B base model and region-specific adapters in `nDNA/Qwen3-4B-WikiCulture-SFT`. Merged adapters are expected under `<repo>/<merged_subdir>/<pair>` where pair is `REGION_REGION` (e.g., `AF_EU`).
- Fisher merging scripts require LoRA adapters with identical configuration (target modules, rank, alpha, dropout, RSLoRA/DoRA flags). The merge utility enforces config equality before combining weights.

---

## Dataset formatting quick guide
- **Alpaca-style prompts**: built with `data.build_alpaca_prompt`. Instruction + optional input followed by `### Response:`. Used in collapse geometry, inbreeding generation, and Method-5 generic when `--dataset alpaca`.
- **Wikipedia text**: loaded via `load_wikipedia_text_dataset` or `wikipedia` key in Method-5 generic. Default config `20231101.en` unless overridden.
- **Everything_Instruct_Multilingual**: always streamed; prompts are Alpaca-formatted.
- **SQuAD / SQuAD v2**: still supported as datasets in Method-5 generic and LoRA runners but no longer the focus of this README. Inputs are `Context: ...\nQuestion: ...`.
- **AG News**: formats to `Title: <headline>\ndescription:`. `--ag-label` filters SetFit/ag_news labels. Legacy `ag_news_business` and `ag_news_setfit` alias to this formatter.
- **HH-RLHF**: uses the `rejected` field as text for geometry runs; Nephos uses prompts/responses for trigger analysis.
- **GSM8K / MBPP**: prompt-only; MBPP appends tests and `# Solution` marker.
- **AutoMathText**: streamed by default due to size; picks `text`/`problem`/`question` fields.
- **Stanford Plato**: flattens nested article sections; strips empty content.

---

## Command inventory (what each script does)
- `scripts/run_method5_generic.py`: Method-5 backbone for arbitrary HF models and datasets. Computes E_l, FR metrics, belief norms, optional curvature + nDNA, saves `.npz` under `results/method5_generic` by default.
- `scripts/run_method5_lora.py`: Same metrics but for PEFT adapters attached to a base model. Supports fine-tuned vs Fisher-merged adapters, optional git push of outputs.
- `scripts/run_collapse_from_zoo.py`: Multi-generation collapse driver (cross-breeding or inbreeding) with Method-5 + spectral geometry per generation. Writes JSON metrics under `collapse_runs_zoo` (configurable).
- `ndna_lib/merging/compute_fisher_lora.py`: Computes diagonal empirical Fisher for trainable LoRA parameters over WikiCulture regions; saves safetensors keyed by param name.
- `ndna_lib/merging/fisher_merge_lora.py`: Fisher-weighted merge of multiple LoRA adapters using previously computed Fishers.
- `scripts/plot.py`: Loads `.npz` Method-5 outputs, normalizes curves, computes similarity matrices, and writes Plotly dashboards + similarity report. Also supports cross-dataset overlays when directory has dataset subfolders.
- `scripts/plot_collapse_pairs_3d.py`: Builds 3D+2D plots over generations for collapse runs using `method5_unified.json` and `spectral_curvature.json` files.
- `scripts/plot_qwen_ndna_5datasets.py`: Hard-coded overlay for five Qwen 3.4B runs (HH RLHF, Everything Multilingual, MBPP, GSM8K, SQuAD).
- `scripts/concreteness_score.py`: Scores concreteness over text datasets with regex or spaCy POS-based methods; outputs JSON summary.
- `Alignment/llama3p1_IT.py` / `qwen2p5_IT.py` / `ministral3_IT.py`: Supervised fine-tuning scripts using TRL and PEFT (LoRA) for instructing base models.
- `Alignment/llama3p1_DPO.py` / `qwen2p5_DPO.py` / `ministral3_DPO.py`: Direct Preference Optimization (DPO) scripts for aligning SFT models to safety/preference datasets.
- `distillation/distill_math_lora.py` / `llama_distillation.py` / `qwen_distillation.py`: LoRA distillation scripts for transferring behavior from teacher to student models using supervised CE and KL divergence loss.

---

## Method-5 generic runner (`scripts/run_method5_generic.py`)
- Purpose: Evaluate one or many HF models on one or many datasets using Method-5 geometry. Supports streaming datasets, optional curvature + nDNA, and MXFP4 quantization (for FR + belief only).
- Input selection: resolve models from `model_zoo.json` (enabled entries) unless `--hf-models` or `--model-keys` override. Dataset keys: `squad`, `squad_v2`, `stanford_plato`, `ag_news`, `hh_rlhf`, `gsm8k`, `mbpp`, `alpaca`, `everything_instruct_multilingual`, `wikipedia`, `automathtext`, plus legacy aliases.
- Outputs: `.npz` named `<dataset>__method5_<model>.npz` in `--out-dir`. Contains model/dataset tags, E_l, n_examples, n_params, Delta, Alpha, Vnorm, mean_total_fr, n_tokens, belief_norms, tau, max_len, batch_size, tokens_per_ex, fisher_unit, optional kappa/kappa_positions, optional ndna_scalar/ndna_layerwise.
- Tokenization: uses `collate_causal` with `max_len` padding/truncation, EOS-as-pad fallback, and DataLoader without shuffle (deterministic ordering).
- Compute order: E_l → FR Δ/α/vnorm → belief norms → optional kappa/nDNA. Kappa uses `spectral_curvature_for_loader` with `keep_last_k` tokens, optionally including embedding node to match Δ shape.
- Performance notes: `--streaming` uses HF IterableDataset shuffle buffer; set `--shuffle-buffer-size` when streaming. `--max-samples` caps materialized texts; set >0. `--no-mxfp4` disables quantization. MXFP4 requires transformers exposing `Mxfp4Config`.
- Failure modes: missing pad token, no enabled models in zoo, unknown dataset key, empty materialized dataset after filtering, kappa shape mismatch (skips nDNA_pred), MXFP4 with old transformers.

### Example invocations
- Minimal single model/dataset, full metrics:
  - `python -m scripts.run_method5_generic --dataset alpaca --hf-models meta-llama/Meta-Llama-3-8B --max-samples 64 --max-len 256 --batch-size 2 --tokens-per-ex 16`
- Run all enabled zoo models on multiple datasets:
  - `python -m scripts.run_method5_generic --datasets ag_news hh_rlhf gsm8k --max-samples 128 --max-len 192 --batch-size 1 --tokens-per-ex 8 --fisher-unit token`
- Enable curvature/nDNA, force streaming for big sets:
  - `python -m scripts.run_method5_generic --datasets everything_instruct_multilingual wikipedia --streaming --max-samples 64 --max-len 256 --batch-size 1 --tokens-per-ex 12 --tau 0.7 --kappa-keep-last-k 12 --kappa-include-embedding-node`
- Use custom zoo and HF ids explicitly:
  - `python -m scripts.run_method5_generic --zoo-path my_zoo.json --hf-models mistral-community/Mistral-7B-v0.3 Qwen/Qwen2-7B-Instruct --dataset mbpp --max-samples 48 --max-len 160`

---

## Method-5 for LoRA and Fisher-merged adapters (`scripts/run_method5_lora.py`)
- Purpose: Run Method-5 on a base model with PEFT adapters (fine-tuned regional adapters and Fisher-merged combos). Ensures adapters are attached with `is_trainable=True` for Fisher gradients. Can skip E_l when quantized or when speed is critical.
- Adapter resolution: `--adapters` selects `finetuned`, `merged`, or `all`. `--adapter-names` overrides explicit subset. `--adapter-repo` defaults to `nDNA/Qwen3-4B-WikiCulture-SFT`; `--merged-subdir` defaults to `merged_adapters`. `--local-adapters-dir` / `--local-merged-dir` switch to local paths. Each adapter is tagged `adapter_type` = finetuned|merged in outputs.
- Datasets: subset of Method-5 generic keys (`squad`, `squad_v2`, `stanford_plato`, `ag_news`, `hh_rlhf`, `gsm8k`, `mbpp`, `alpaca`).
- Outputs: `<dataset>__method5_lora_<adapter_type>_<adapter>.npz` with FR metrics, belief norms, optional E_l, optional kappa/nDNA, metadata on base model and adapter path/type, and skip_param_effort flag. Git integration optionally commits and pushes after each run.
- Fisher constraints: MXFP4 cannot be combined with E_l; runner enforces `--skip-param-effort` if MXFP4 is requested. Gradients forced on floating parameters; adapter unload is called after each run to restore base model.
- Dataset materialization: similar to generic runner, with streaming + shuffle buffer. Prompts follow same formatters.
- Failure modes: adapter config mismatch (caught upstream), missing pad token, no adapters resolved, invalid dataset key, gradient disabled on params, PEFT unload missing in older peft.

### Example invocations
- Run all fine-tuned adapters on Alpaca with Fisher + FR:
  - `python -m scripts.run_method5_lora --adapters finetuned --dataset alpaca --max-samples 64 --max-len 192 --batch-size 1 --tokens-per-ex 12 --tau 0.8 --fisher-unit sequence`
- Evaluate merged adapters only, skip Fisher for speed:
  - `python -m scripts.run_method5_lora --adapters merged --dataset ag_news --max-samples 96 --skip-param-effort --no-kappa --max-len 256 --batch-size 2`
- Mixed selection with local adapter dirs and MXFP4 (no Fisher):
  - `python -m scripts.run_method5_lora --adapter-names AF EU AF_EU --local-adapters-dir ./adapters --local-merged-dir ./merged_adapters --dataset hh_rlhf --use-mxfp4 --skip-param-effort`
- Auto git push per adapter:
  - `python -m scripts.run_method5_lora --adapters finetuned --dataset gsm8k --max-samples 48 --git-push`

---

## Collapse driver (`scripts/run_collapse_from_zoo.py`)
- Purpose: Automate cross-breeding or inbreeding collapse trajectories across multiple models from the zoo. Each generation runs Alpaca Method-5 and spectral curvature using `ndna_lib.collapse.geometry_runner`.
- Protocols: `--protocol cross` fine-tunes on human Alpaca pairs; `--protocol inbreed` generates synthetic Alpaca-style data with the current model and fine-tunes on it. Breeding hyperparameters follow `BreedingConfig` defaults unless overridden.
- Run layout: `<base-run-dir>/<model_key>/<run_name>/gen0` holds base geometry; each subsequent `genX` contains `model/` (optional) and `metrics/{method5_unified.json, spectral_curvature.json}`.
- Geometry config: `geom_max_samples`, `geom_max_len`, `geom_batch_size`, `geom_unit`, `spectral_prompts`, `spectral_max_tokens`. Smoke mode (`--smoke`) lowers sample counts/steps for quick validation.
- Model selection: defaults to all `enabled` in `model_zoo.json` unless `--model-keys` comma list is supplied. HF id is pulled from zoo entry.
- Save behavior: by default models stay in memory between generations (`--save-models` toggles disk checkpoints). In-memory path avoids disk writes but clears GPU memory between steps.
- Failure modes: empty model selection, missing metrics files for plotting, vLLM unavailable for inbreeding (falls back to HF generation), insufficient GPU memory for collapse training.

### Example invocations
- Cross-breeding 3 generations on Alpaca:
  - `python -m scripts.run_collapse_from_zoo --protocol cross --model-keys llama3-8b --base-run-dir ./collapse_runs_multi --run-name cross_a --generations 3 --max-train-samples 6000 --max-steps-first 1000 --max-steps-later 500 --geom-max-samples 256 --geom-max-len 256 --geom-batch-size 2`
- Inbreeding with smoke settings:
  - `python -m scripts.run_collapse_from_zoo --protocol inbreed --model-keys gpt2 --smoke --run-name quick_inbreed --geom-unit token --spectral-prompts 6 --spectral-max-tokens 96`
- Keep checkpoints on disk for inspection:
  - `python -m scripts.run_collapse_from_zoo --protocol cross --model-keys llama3-8b --save-models --base-run-dir ./collapse_disk --run-name saved_ckpts`

---

## Fisher computation for LoRA adapters (`ndna_lib/merging/compute_fisher_lora.py`)
- Purpose: Compute diagonal empirical Fisher for trainable LoRA parameters using WikiCulture region-filtered text. Writes safetensors containing squared gradients averaged per example plus metadata for example/token counts.
- Regions: expects WikiCulture configs named by `REGION` (AF, AS, AU, CH, EU, LA, ME, NA). The iterator filters `bucket_geo` to the requested region.
- Dataset streaming: defaults to streaming; shuffle buffer configurable. Batch micro-backprop runs per example even when `batch_size>1` to avoid cross-example cross-terms.
- Tokenization: pad to `max_length`, shift logits for causal loss, sum token cross-entropy without mean reduction.
- Output naming: you choose `--out_path`; directory will be created. Keys in safetensors match adapter param names (with `.default.` stripped) plus `__n_examples__` and `__total_tokens__`.
- Failure modes: LoRA adapter not loaded with trainable params, zero usable examples after filtering, missing pad/eos tokens, mismatched device when using CPU mode.

### Example invocations
- Compute Fisher for AF region on GPU:
  - `python -m ndna_lib.merging.compute_fisher_lora --base_model Qwen/Qwen3-4B --adapter_dir ./adapters/AF --dataset_id nDNA/WikiCulture --region AF --streaming --max_examples 256 --batch_size 1 --max_length 512 --dtype bf16 --out_path ./fishers/AF.safetensors`
- CPU fallback with fewer samples:
  - `python -m ndna_lib.merging.compute_fisher_lora --base_model Qwen/Qwen3-4B --adapter_dir ./adapters/EU --dataset_id nDNA/WikiCulture --region EU --device cpu --dtype fp32 --max_examples 64 --batch_size 1 --max_length 256 --out_path ./fishers/EU_cpu.safetensors`

---

## Fisher-weighted merging of adapters (`ndna_lib/merging/fisher_merge_lora.py`)
- Purpose: Merge multiple LoRA adapters into one adapter using Fisher weights as importance factors. Requires identical LoRA configs across adapters and matching Fisher files.
- Inputs: `--adapters` list of adapter dirs, `--fishers` list of safetensors paths, optional `--alphas` weights (defaults to 1.0 each), numerical stability via `--eps` and `--fisher_floor`, optional Fisher normalization (`none` or `mean`).
- Validation: checks presence of adapter_config.json and adapter_model.safetensors for each adapter; enforces identical canonicalized config across inputs; enforces matching keys in Fishers.
- Output: writes merged adapter_config.json (copied from first) and adapter_model.safetensors into `--out_dir` (created if needed).
- Failure modes: config mismatch, missing Fisher entries, length mismatch between adapters/fishers/alphas, empty output dir path.

### Example invocations
- Merge two regions with equal weights:
  - `python -m ndna_lib.merging.fisher_merge_lora --adapters ./adapters/AF ./adapters/EU --fishers ./fishers/AF.safetensors ./fishers/EU.safetensors --out_dir ./merged_adapters/AF_EU`
- Merge three adapters with custom alphas and floor:
  - `python -m ndna_lib.merging.fisher_merge_lora --adapters ./adapters/AF ./adapters/EU ./adapters/AS --fishers ./fishers/AF.safetensors ./fishers/EU.safetensors ./fishers/AS.safetensors --alphas 1.0 0.8 0.6 --fisher_floor 1e-10 --normalize_fishers mean --out_dir ./merged_adapters/triad`

---

## Plotting utilities

### Comprehensive dashboard + similarity matrices (`scripts/plot.py`)
- Purpose: Turn Method-5 `.npz` files into interactive Plotly dashboards and pairwise similarity reports. Can also aggregate multiple dataset subdirectories into cross-dataset overlays.
- Inputs: directory containing `.npz` with at least `kappa`, `Delta`, `belief_norms`. If the directory has subfolders each containing `.npz`, script treats them as datasets and builds cross-dataset overlays.
- Normalization: κ, Δ, and |v| are winsorized then scaled to [min_out, max_out]; derived nDNA is computed from normalized curves. Resampling is used for MAE/MSE distance options.
- Outputs: `LLM_curve_similarity_report.html` (matrices + combined plots), `ALL_MODELS_trajectory.html` (combined dashboards), per-model dashboards, plus optional cross-dataset figures. File names preserved.
- Flags: `--input-dir`, `--output-dir`, `--resample-n`, `--skip-report`, `--skip-dashboards`. Environment defaults `NDNA_INPUT_DIR`/`NDNA_OUTPUT_DIR` respected.
- Failure modes: missing required keys in npz, empty directory, mismatched lengths (truncated to min length), plot writing errors on read-only paths.

### Collapse trajectory plots (`scripts/plot_collapse_pairs_3d.py`)
- Purpose: For each model/experiment in a collapse run, plot spectral/thermo/belief curves across generations in four 3D and two 2D subplots stacked vertically.
- Inputs: base directory containing model subdirs with experiment subfolders and `gen*/metrics/{method5_unified.json,spectral_curvature.json}` files. `--exp_name` filters to a specific experiment; otherwise all are processed.
- Belief metric: `--belief_key` chooses `Eta` (default) or `E` from Method-5 JSON.
- Output: `<model>/<model>_<exp>_collapse_pairs_3d.html` per experiment.

### Fixed Qwen 3.4B overlay (`scripts/plot_qwen_ndna_5datasets.py`)
- Purpose: Convenience overlay for five specific Qwen 3.4B runs. Loads the hard-coded file list and emits a two-figure HTML with raw and normalized curves plus scalar summary table.
- Output: `plots/qwen3_4b_task_overlay.html` by default (overridable with `--output`).

---

## Concreteness scoring (`scripts/concreteness_score.py`)
- Purpose: Measure lexical concreteness on supported datasets using regex-only or POS-filtered approaches with a provided concreteness lexicon.
- Datasets: `ag_news_business`, `stanford_plato`, `automathtext_web`. Each has a built-in formatter in `ndna_lib.concreteness.data`.
- Methods: `regex` (no POS filtering), `pos_keep_num` (spaCy POS, keep nouns/numbers), `pos_drop_num` (drop numbers). All work in streaming mode and honor `--max-records` for sampling.
- Output: JSON with dataset metadata, overall averages, coverage, running stats, counts, elapsed time; printed to stdout and written to a file under `ndna_lib/concreteness/outputs` unless `--out` is set.
- Failure modes: missing spaCy model for POS methods, zero matches (coverage 0), unknown dataset key.

### Example invocations
- POS-filtered on AG News business, 30k samples:
  - `python scripts/concreteness_score.py --dataset ag_news_business --method pos_keep_num --max-records 30000`
- Streaming AutoMathText subset with regex:
  - `python scripts/concreteness_score.py --dataset automathtext_web --method regex --streaming --max-records 20000`

---

## SFT & DPO Workflows (`Alignment/`)
- Purpose: Fine-tune base models into instruction-following models (IT) and then align them using Direct Preference Optimization (DPO). 
- Scripts: `llama3p1_IT.py`, `qwen2p5_IT.py`, `ministral3_IT.py`, `llama3p1_DPO.py`, `qwen2p5_DPO.py`, `ministral3_DPO.py`.
- Features: Uses `trl` (SFTTrainer, DPOTrainer) and `peft` (LoRA). Patches special token embeddings (like `<|eot_id|>`) to avoid NaN gradients, configures exact chat templates, caches processed datasets, and outputs merged adapters.
- Typical Flow: Run IT script to get an instruction-following base -> Run DPO script using the IT checkpoint as `ref_model` on preference datasets (e.g., safe vs unsafe responses).

---

## Distillation Workflows (`distillation/`)
- Purpose: Transfer specific reasoning capabilities (like math) from a larger teacher model (e.g., Llama 3.1 8B) into a student model (e.g., Llama 3 8B).
- Scripts: `distill_math_lora.py`, `llama_distillation.py`, `qwen_distillation.py`.
- Features: Trains LoRA adapters on the student model while the teacher is frozen. Loss is a combination of supervised CE (on answer tokens) and KL divergence (using teacher's logits).
- Flags: `--kl_weight`, `--ce_weight`, `--temperature` control the distillation dynamics.

---

## End-to-end playbooks
- **Method-5 generic sweep:** choose datasets (`--datasets ag_news hh_rlhf gsm8k ...`), ensure `model_zoo.json` enabled entries, run generic runner, collect `.npz` under `results/method5_generic/<dataset>/`, and visualize with `scripts/plot.py --input-dir results/method5_generic/<dataset>`.
- **LoRA Fisher pipeline:** compute Fishers per region with `ndna_lib.merging.compute_fisher_lora`, merge adapters with `ndna_lib.merging.fisher_merge_lora`, evaluate with `scripts/run_method5_lora --adapters merged`, plot with `scripts/plot.py` or cross-dataset overlays.
- **Collapse experiment:** run `scripts/run_collapse_from_zoo` with chosen protocol, then visualize trajectories with `scripts/plot_collapse_pairs_3d.py --base_dir <run_dir>` using belief key `Eta` (default) or `E`.
- **SFT pipeline:** run `Alignment/llama3p1_IT.py` (or `qwen2p5_IT.py`) on a preference dataset to produce an instruction-following adapter; merge with `Alignment/merge.py` before DPO.
- **DPO alignment:** run `Alignment/llama3p1_DPO.py` (or `qwen2p5_DPO.py`) using the IT adapter as `ref_model` on a safe/unsafe preference dataset.
- **Distillation:** run `distillation/llama_distillation.py` with `--kl_weight 0.7 --ce_weight 0.3 --temperature 2.0` to transfer teacher reasoning into a LoRA student.
- **Concreteness audit:** run `scripts/concreteness_score.py` across supported datasets to profile lexical concreteness before/after fine-tuning; stash JSON outputs under `ndna_lib/concreteness/outputs` for regression tracking.
- **Dashboard build:** once `.npz` metrics exist, call `scripts/plot.py --input-dir <dir> --output-dir <plots_dir>` to generate full similarity report and per-model dashboards. Add `--skip-report` if you only need dashboards or `--skip-dashboards` if you only need matrices.

---

## Quick command cheat sheet (copy-paste starters)
Below are concrete, ready-to-run examples for every shipped CLI. Adjust paths/IDs to your environment.

- **Method-5 generic (single model, multiple datasets):**
  ```bash
  python -m scripts.run_method5_generic \
    --hf-models meta-llama/Meta-Llama-3-8B \
    --datasets ag_news hh_rlhf gsm8k \
    --max-samples 96 --max-len 192 --batch-size 1 \
    --tokens-per-ex 12 --tau 0.8 --fisher-unit token \
    --out-dir results/method5_generic
  ```

- **Method-5 generic (all enabled zoo models, streaming heavy datasets, with kappa/nDNA):**
  ```bash
  python -m scripts.run_method5_generic \
    --dataset everything_instruct_multilingual \
    --streaming --max-samples 64 --shuffle-buffer-size 20000 \
    --max-len 256 --batch-size 1 --tokens-per-ex 12 \
    --tau 0.7 --kappa-keep-last-k 12 --kappa-include-embedding-node
  ```

- **Method-5 LoRA (fine-tuned adapters only, full Fisher + FR):**
  ```bash
  python -m scripts.run_method5_lora \
    --adapters finetuned \
    --dataset alpaca --max-samples 64 --max-len 192 \
    --tokens-per-ex 12 --tau 0.8 --fisher-unit sequence \
    --out-dir results/method5_lora_finetuned
  ```

- **Method-5 LoRA (merged adapters, no Fisher, MXFP4 on):**
  ```bash
  python -m scripts.run_method5_lora \
    --adapters merged --dataset ag_news --max-samples 96 \
    --skip-param-effort --use-mxfp4 --no-kappa \
    --local-merged-dir ./merged_adapters --base-model Qwen/Qwen3-4B \
    --out-dir results/method5_lora_merged
  ```

- **Collapse cross-breeding (3 gens, saves checkpoints):**
  ```bash
  python -m scripts.run_collapse_from_zoo \
    --protocol cross --model-keys llama3-8b \
    --base-run-dir ./collapse_runs_multi --run-name cross_a \
    --generations 3 --max-train-samples 6000 \
    --max-steps-first 1000 --max-steps-later 500 \
    --geom-max-samples 256 --geom-max-len 256 --geom-batch-size 2 \
    --save-models
  ```

- **Collapse inbreeding smoke test (keeps models in memory):**
  ```bash
  python -m scripts.run_collapse_from_zoo \
    --protocol inbreed --model-keys gpt2 \
    --smoke --run-name quick_inbreed --base-run-dir ./collapse_runs_smoke
  ```

- **Compute Fisher for a single LoRA adapter (WikiCulture region):**
  ```bash
  python -m ndna_lib.merging.compute_fisher_lora \
    --base_model Qwen/Qwen3-4B \
    --adapter_dir ./adapters/AF \
    --dataset_id nDNA/WikiCulture --region AF \
    --streaming --max_examples 256 --batch_size 1 --max_length 512 \
    --dtype bf16 --shuffle_buffer 8000 --out_path ./fishers/AF.safetensors
  ```

- **Fisher-merge two adapters with equal weights:**
  ```bash
  python -m ndna_lib.merging.fisher_merge_lora \
    --adapters ./adapters/AF ./adapters/EU \
    --fishers ./fishers/AF.safetensors ./fishers/EU.safetensors \
    --normalize_fishers mean --out_dir ./merged_adapters/AF_EU
  ```

- **Plot Method-5 results into dashboards + similarity report:**
  ```bash
  python -m scripts.plot \
    --input-dir results/method5_generic/ag_news \
    --output-dir plots/ag_news_report \
    --resample-n 200
  ```

- **Plot collapse generations (Eta as belief axis):**
  ```bash
  python -m scripts.plot_collapse_pairs_3d \
    --base_dir ./collapse_runs_multi \
    --belief_key Eta \
    --exp_name cross_a
  ```

- **Fixed Qwen 3.4B overlay (five datasets):**
  ```bash
  python -m scripts.plot_qwen_ndna_5datasets \
    --output plots/qwen3_overlay.html
  ```

- **Concreteness scoring (POS filter, streaming AutoMathText):**
  ```bash
  python scripts/concreteness_score.py \
    --dataset automathtext_web --method pos_keep_num \
    --streaming --max-records 20000 \
    --out ndna_lib/concreteness/outputs/automath_pos.json
  ```

- **SFT training (Instruction Tuning):**
  ```bash
  python Alignment/llama3p1_IT.py
  # Edit MODEL_NAME, DATASET_NAME, OUTPUT_DIR, NUM_SAMPLES at the top of the script
  ```

- **DPO alignment:**
  ```bash
  python Alignment/llama3p1_DPO.py
  # Set sft_model_path and ref_model_path to your IT adapter output
  ```

- **Merge LoRA adapter into base model (for evaluation):**
  ```bash
  python Alignment/merge.py \
    --base_model meta-llama/Meta-Llama-3.1-8B \
    --adapter_path ./Alignment/final_adapter_llama3p1_IT \
    --output_path ./Alignment/merged_model
  ```

- **LoRA distillation (Llama 3.1 teacher → Llama 3 student, GSM8K):**
  ```bash
  python distillation/llama_distillation.py \
    --dataset_name gsm8k --dataset_config main \
    --train_split train --eval_split test \
    --output_dir distillation/llama3-math-distill \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --learning_rate 2e-4 --num_train_epochs 1 \
    --kl_weight 0.7 --ce_weight 0.3 --temperature 2.0 \
    --bf16 True
  ```

- **Dashboard-only refresh (skip similarity matrix):**
  ```bash
  python -m scripts.plot \
    --input-dir results/method5_lora \
    --output-dir plots/method5_lora_dash \
    --skip-report
  ```

Use these as templates—only change model IDs, adapter paths, datasets, and output directories to fit your hardware and experiments.

---

## Troubleshooting and validation checklist
- Verify GPU availability with `torch.cuda.is_available()` before heavy runs. CPU mode is supported but slow; collapse training requires CUDA.
- If padding token errors arise, ensure tokenizer has `pad_token` or `eos_token`; Method-5 runners set pad to EOS when missing.
- Streaming datasets need reasonable `--shuffle-buffer-size` (10k by default) to avoid order bias.
- MXFP4 quantization may not be present in older transformers; use `--no-mxfp4` when errors mention `Mxfp4Config`.
- Fisher computation requires adapters loaded with `is_trainable=True`; the LoRA runner and Fisher script enforce this, but custom loading should too.
- For collapse inbreeding with vLLM, ensure vLLM is installed; otherwise the code falls back to standard generation and may be slower.
- Plotting scripts skip files missing required keys; check console output for "Skipping ... missing keys" messages.
- SFT scripts fix zero-initialized special token embeddings before training; do not skip this when adapting to new base models.
- DPO requires the SFT checkpoint as `ref_model`; passing the wrong checkpoint leads to zero KL and collapsed training.
- Distillation requires both teacher and student to fit on the same GPU(s); use 4-bit loading (`--use_4bit`) when memory is limited.
- Concreteness POS methods require `en_core_web_sm`; install via `python -m spacy download en_core_web_sm`.
