# Unified Evaluation Pipeline — Setup & Run Guide

## Overview
This pipeline evaluates N models (Base, SFT, DPO, or any combination) across:
- **Safety benchmarks**: HarmBench (320), AdvBench (100), XSTest (250 safe + 200 unsafe)
- **Instruction benchmarks**: IFEval (541), MT-Bench (80)

Using an LLM-as-judge (Qwen 32B) + programmatic verification (IFEval).

## Files
```
evaluation_pipeline_part1.py  — Core engine, CONFIG, metrics, generation
benchmark_integration.py      — Downloads benchmarks from HuggingFace
run_generation.py             — CLI for response generation
run_evaluation.py             — Evaluation runners, summaries, plots
server.py                     — vLLM model server launcher
download_models.py            — Downloads SFT/DPO from HuggingFace
requirements.txt              — All dependencies
test_part1.py                 — Tests for core engine (17 tests)
test_part2.py                 — Tests for benchmark loading (11 tests)
test_part3.py                 — Tests for evaluation pipeline (11 tests)
```

## First-Time Setup (~15 min)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Fix libstdc++ if you get CXXABI errors (common on some setups)
conda install -c conda-forge libstdcxx-ng -y
export LD_LIBRARY_PATH=/path/to/conda/lib:$LD_LIBRARY_PATH
# Add to ~/.bashrc to make it permanent

# 3. Login to HuggingFace (needed for Llama-3 which is gated)
pip install huggingface_hub
python -m huggingface_hub.commands.huggingface_cli login
# Or: export HF_TOKEN="hf_your_token_here"

# 4. Download benchmarks
python benchmark_integration.py

# 5. Download SFT and DPO models
python download_models.py

# 6. Run tests to verify everything works
python test_part1.py    # expect 17/17
python test_part2.py    # expect 11/11
python test_part3.py    # expect 11/11
```

## Configure Models

Edit `evaluation_pipeline_part1.py` → CONFIG → "models" section:

```python
"models": [
    {"key": "base", "name": "meta-llama/Meta-Llama-3-8B",  "chat": False},
    {"key": "sft",  "name": "./models_local/SFT_merged",   "chat": True},
    {"key": "dpo",  "name": "./models_local/DPO_merged",   "chat": True},
],
```

- `key`: short name used in CLI and output columns
- `name`: HuggingFace ID or local path
- `chat`: True for instruct/chat models, False for base models

### Sample Size
In the same CONFIG, adjust sample_size:
- `30` for quick demo (~5 min per model)
- `500` for solid results (~2 hrs per model)
- `None` for full dataset

## Run the Pipeline

### Phase 1: Generate Responses (one model at a time)

You need TWO terminals. Terminal 1 runs the model server, Terminal 2 sends prompts.

```bash
# === MODEL 1: BASE ===
# Terminal 1:
python server.py --model meta-llama/Meta-Llama-3-8B --quantization none

# Terminal 2 (after server shows "Application startup complete"):
python run_generation.py --model base --all
# Wait for it to finish, then Ctrl+C Terminal 1

# === MODEL 2: SFT ===
# Terminal 1:
python server.py --model ./models_local/SFT_merged --quantization none

# Terminal 2:
python run_generation.py --model sft --all
# Wait, Ctrl+C Terminal 1

# === MODEL 3: DPO ===
# Terminal 1:
python server.py --model ./models_local/DPO_merged --quantization none

# Terminal 2:
python run_generation.py --model dpo --all
# Wait, Ctrl+C Terminal 1
```

### Phase 2: Evaluate (judge model)

```bash
# Terminal 1:
python server.py --model Qwen/Qwen2.5-32B-Instruct --quantization none

# Terminal 2:
python run_evaluation.py
```

**Note for A6000 48GB**: Qwen 32B is ~64GB in float16 — won't fit on 48GB.
Use quantization instead:
```bash
python server.py --model Qwen/Qwen2.5-32B-Instruct --quantization bitsandbytes
```
Or use a smaller judge:
```bash
python server.py --model Qwen/Qwen2.5-14B-Instruct --quantization none
```
Then update CONFIG: `"judge_model": "Qwen/Qwen2.5-14B-Instruct"`

## CLI Options

```bash
# Generate only safety benchmarks
python run_generation.py --model base --all --pipeline safety

# Generate only instruction benchmarks  
python run_generation.py --model base --all --pipeline instruction

# Evaluate only safety
python run_evaluation.py --pipeline safety

# Evaluate only instruction
python run_evaluation.py --pipeline instruction

# Evaluate both + unified comparison
python run_evaluation.py --pipeline both
```

## Output Structure

```
eval_results/
├── safety/
│   ├── advbench_eval.csv           # Per-record scores
│   ├── advbench_summary.csv        # Mean, win-rates, deltas
│   ├── harmbench_eval.csv
│   ├── xstest_safe_eval.csv
│   ├── xstest_unsafe_eval.csv
│   ├── safety_combined.csv         # All safety benchmarks merged
│   ├── safety_combined_summary.csv
│   └── safety_grid.png             # Comparison bar chart
│
├── instruction/
│   ├── ifeval_eval.csv
│   ├── ifeval_verification.csv     # Programmatic pass rates
│   ├── mt_bench_eval.csv
│   ├── instruction_combined.csv
│   ├── instruction_combined_summary.csv
│   └── instruction_grid.png
│
└── unified/
    ├── unified_comparison.csv       # Paper-ready table
    └── unified_comparison.png       # Side-by-side plot
```

## Resume Support

Everything is resume-safe. If any step crashes or gets interrupted,
just re-run the exact same command — it picks up where it left off.

## Adding More Models

Just add entries to CONFIG["models"]:
```python
"models": [
    {"key": "base",   "name": "meta-llama/Meta-Llama-3-8B",  "chat": False},
    {"key": "sft",    "name": "./models_local/SFT_merged",   "chat": True},
    {"key": "dpo",    "name": "./models_local/DPO_merged",   "chat": True},
    {"key": "dpo_v2", "name": "./models_local/DPO_v2",       "chat": True},
],
```

Then run generation for the new model and re-run evaluation.
Win-rates and plots automatically adapt to N models.

## Troubleshooting

| Problem | Fix |
|---|---|
| `CXXABI_1.3.15 not found` | `conda install -c conda-forge libstdcxx-ng -y` then `export LD_LIBRARY_PATH=...` |
| `No module named 'vllm'` | `pip install vllm` |
| GPU OOM on server start | Use `--quantization bitsandbytes` or `--gpu-util 0.7` |
| GPU memory not freed | `pkill -9 -f vllm && sleep 10` then restart |
| Chat template error | Set `"chat": False` for base models in CONFIG |
| Gated model access denied | `export HF_TOKEN="hf_..."` or `huggingface-cli login` |
