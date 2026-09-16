#!/usr/bin/env bash
# Run the full suite in dependency order.
#
#   bash scripts/run_all.sh                 # everything, both models
#   MODELS="qwen3_4b" bash scripts/run_all.sh
#   TIER=A bash scripts/run_all.sh          # integrity tier only
#   DRY=1 bash scripts/run_all.sh           # print the plan, load nothing
#
# Each experiment is independent except 02 -> 09 and 01 -> 13, so a failure in one does
# not stop the rest; failures are collected and reported at the end.

set -uo pipefail
cd "$(dirname "$0")/.."

MODELS="${MODELS:-}"
TIER="${TIER:-ALL}"
DRY="${DRY:-0}"
LOG_DIR="${LOG_DIR:-logs}"
mkdir -p "$LOG_DIR"

ARGS=""
[ -n "$MODELS" ] && ARGS="$ARGS --models $MODELS"
[ "$DRY" = "1" ] && ARGS="$ARGS --dry-run"

TIER_A="exp01_dose_response exp02_operation_classification"
TIER_B="exp03_behavior_vs_geometry exp04_task_stability exp08_absolute_triad"
TIER_C="exp05_invariance exp06_tuned_lens exp07_patching"
TIER_D="exp10_provenance exp11_merge_screening exp12_collapse_early_warning exp14_alignment_stage"
AGGREGATE="exp09_stats_control exp13_compression_knee"

case "$TIER" in
  A)   PLAN="$TIER_A" ;;
  B)   PLAN="$TIER_B" ;;
  C)   PLAN="$TIER_C" ;;
  D)   PLAN="$TIER_D" ;;
  ALL) PLAN="$TIER_A $TIER_B $TIER_C $TIER_D" ;;
  *)   echo "unknown TIER=$TIER (use A|B|C|D|ALL)"; exit 2 ;;
esac

echo "=== plan: $PLAN"
echo "=== args: ${ARGS:-none}"
FAILED=""

for e in $PLAN; do
  echo ""
  echo "--------------------------------------------------------------- $e"
  if python "experiments/${e}.py" $ARGS 2>&1 | tee "$LOG_DIR/${e}.log"; then
    echo "    ok"
  else
    echo "    FAILED (see $LOG_DIR/${e}.log)"
    FAILED="$FAILED $e"
  fi
done

# aggregation steps load no model and depend on the runs above
if [ "$TIER" = "ALL" ] && [ "$DRY" != "1" ]; then
  for e in $AGGREGATE; do
    echo ""
    echo "--------------------------------------------------------------- $e (aggregate)"
    python "experiments/${e}.py" 2>&1 | tee "$LOG_DIR/${e}.log" || FAILED="$FAILED $e"
  done
  python experiments/exp02_operation_classification.py --aggregate \
      2>&1 | tee "$LOG_DIR/exp02_aggregate.log" || true
  python scripts/make_tables.py 2>&1 | tee "$LOG_DIR/make_tables.log" || true
fi

echo ""
echo "==============================================================="
if [ -n "$FAILED" ]; then
  echo "FAILED:$FAILED"
  exit 1
fi
echo "all requested experiments completed"
echo "results in results/  (npz + json manifest per run)"
