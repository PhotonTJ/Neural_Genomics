#!/usr/bin/env bash
# filepath: /root/ndna/scripts/run_method5_cloudexe_suite.sh
set -euo pipefail

PY="/root/miniconda3/envs/ndna/bin/python"

# cloudexe settings
GPUSPEC="${GPUSPEC:-EUNH100x1}"
RESETSERVER="${RESETSERVER:---resetserver}"  # set to "" to disable
QUEUE="${QUEUE:---queue}"                            # set to a queue name to enable --queue

# Method-5 settings
MODEL_ALIAS="${MODEL_ALIAS:-llama3_base}"
MAX_SAMPLES="${MAX_SAMPLES:-2500}"
MAX_LEN="${MAX_LEN:-512}"
BATCH_SIZE="${BATCH_SIZE:-1}"
TOKENS_PER_EX="${TOKENS_PER_EX:-512}"
KAPPA_KEEP_LAST_K="${KAPPA_KEEP_LAST_K:-10}"

# Datasets (keys must match scripts/run_method5_generic.py SUPPORTED_DATASETS)
DATASETS=(
  # NLI
  "mnli"
  # Commonsense Reasoning
  "hellaswag"
  # Closed-book QA
  "ai2_arc"
  # Coreference resolution
  "winogrande"
  # Struct to text
  "common_gen"
  # Summarization
  "cnn_dailymail"
  # Reading Comprehension
  "squad_v2"
  # Sentiment Analysis
  "imdb"
  # Paraphrase
  "qqp"
  # Translation
  "wmt16"
)

for ds in "${DATASETS[@]}"; do
  echo "============================================================"
  echo "[suite] Running dataset: ${ds} | model: ${MODEL_ALIAS}"
  echo "============================================================"

  cloudexe --gpuspec "${GPUSPEC}" ${QUEUE} ${RESETSERVER} -- \
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

  sleep 10
done