#!/usr/bin/env bash
# filepath: /root/ndna/scripts/run_method5_suite.sh
set -euo pipefail

PY="/root/miniconda3/envs/ndna/bin/python"
MODEL_ALIAS="${MODEL_ALIAS:-gemma3_base}"

MAX_SAMPLES="${MAX_SAMPLES:-2500}"
MAX_LEN="${MAX_LEN:-512}"
BATCH_SIZE="${BATCH_SIZE:-1}"
TOKENS_PER_EX="${TOKENS_PER_EX:-512}"
KAPPA_KEEP_LAST_K="${KAPPA_KEEP_LAST_K:-10}"

DATASETS=(
  # mnli          # NLI
  # hellaswag     # Commonsense Reasoning
  # ai2_arc       # Closed-book QA (ARC-Challenge)
  # winogrande    # Coreference resolution
  # common_gen    # Struct to text
  cnn_dailymail # Summarization
  squad_v2      # Reading Comp
  imdb          # Sentiment Analysis
  qqp           # Paraphrase
  wmt16         # Translation
)

for ds in "${DATASETS[@]}"; do
  echo "============================================================"
  echo "[suite] Running dataset: ${ds} | model: ${MODEL_ALIAS}"
  echo "============================================================"

  "${PY}" scripts/run_method5_generic.py \
    --datasets "${ds}" \
    --streaming \
    --max-samples "${MAX_SAMPLES}" \
    --max-len "${MAX_LEN}" \
    --batch-size "${BATCH_SIZE}" \
    --tokens-per-ex "${TOKENS_PER_EX}" \
    --kappa-keep-last-k "${KAPPA_KEEP_LAST_K}" \
    --kappa-include-embedding-node \
    --hf-models "${MODEL_ALIAS}"
done