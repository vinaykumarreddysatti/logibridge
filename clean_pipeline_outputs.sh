#!/usr/bin/env bash
# Removes every generated artifact from the LogiEdge pipeline (dataset,
# trained models, TFLite exports, benchmark results, PSI reference/trace,
# stats-shift experiment output, alert log) so `README.md`'s "Running the
# pipeline end to end" steps can be re-run from a clean slate.
#
# Does NOT touch: source scripts, deployment/ configs and certs,
# report drafts, demo/demo_video_link.txt, or the .venv.
#
# Usage:
#   ./clean_pipeline_outputs.sh          # lists targets, asks to confirm
#   ./clean_pipeline_outputs.sh -y       # skip the confirmation prompt

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

TARGETS=(
  # 1. Data -> dataset -> model
  "training/dataset_X.npy"
  "training/dataset_y.npy"
  "training/models/logibridge_mlp.keras"
  "training/models/logibridge_mlp_pruned.keras"
  "training/models/train_features_norm.npy"
  "training/models/train_labels.npy"
  "training/models/val_features.npy"
  "training/models/val_features_norm.npy"
  "training/models/val_labels.npy"
  "training/models/model_fp32.tflite"
  "training/models/model_int8.tflite"
  "training/models/model_pruned_int8.tflite"
  "data_pipeline/training_stats.npy"
  "inference/model.tflite"

  # 2. Benchmark
  "optimisation/results/benchmark_results.csv"
  "optimisation/results/pareto_chart.png"

  # 3. Drift monitoring reference + offline validation
  "monitoring/reference_dist.json"
  "monitoring/drift_demo_trace.json"

  # C2 mandatory stats-shift experiment
  "data_pipeline/stats_shift_experiment_results.json"
  "data_pipeline/stats_shift_sweep_supplementary.json"

  # Runtime output from a live/local inference_service.py run
  "alert_log.jsonl"
)

echo "The following generated files will be removed (if present):"
found_any=false
for f in "${TARGETS[@]}"; do
  if [ -e "$f" ]; then
    echo "  $f"
    found_any=true
  fi
done
find . -name "__pycache__" -not -path "./.venv/*" | while read -r d; do
  echo "  $d"
  found_any=true
done

if [ "$found_any" = false ]; then
  echo "  (nothing to clean -- already clean)"
  exit 0
fi

if [[ "${1:-}" != "-y" ]]; then
  read -r -p "Proceed? [y/N] " reply
  case "$reply" in
    [yY]|[yY][eE][sS]) ;;
    *) echo "Aborted."; exit 1 ;;
  esac
fi

for f in "${TARGETS[@]}"; do
  [ -e "$f" ] && rm -v "$f"
done
find . -name "__pycache__" -not -path "./.venv/*" -exec rm -rf {} +

echo "Done. Re-run the pipeline from README.md's \"Running the pipeline end to end\" section."
