#!/usr/bin/env bash
set -euo pipefail

# Design: ../../my_development/dsv4_hybrid_hisparse_accuracy_testing.md

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
hardware=${1:?usage: run_gpuq.sh HARDWARE MODE}
mode=${2:?usage: run_gpuq.sh HARDWARE MODE}

# shellcheck source=/dev/null
source "$SCRIPT_DIR/profiles/$hardware.env"
REMOTE_REPO=${REMOTE_REPO:-/home/jovyan/whw/sglang}
GPUQ_PROJECT=${GPUQ_PROJECT:-dsv4-test}
GPUQ_TIMEOUT=${GPUQ_TIMEOUT:-2h}
output_dir=${GPUQ_OUTPUT_DIR:-$PWD/artifacts/gpuq-$hardware-$mode-$(date +%Y%m%d-%H%M%S)}

inner_cmd=""
for variable in \
  MODEL_PATH GSM8K_DATA_PATH NUM_EXAMPLES NUM_THREADS NUM_SHOTS NUM_SHOTS_EVICT \
  MIN_SCORE TOP_K DEVICE_BUFFER_SIZE HOST_TO_DEVICE_RATIO RUN_DIR; do
  if [[ -n ${!variable:-} ]]; then
    printf -v assignment 'export %s=%q && ' "$variable" "${!variable}"
    inner_cmd+="$assignment"
  fi
done
printf -v run_cmd 'cd %q && export PYTHONPATH=%q/python && %q/my_scripts/hisparse_accuracy/run_accuracy.sh %q %q' \
  "$REMOTE_REPO" "$REMOTE_REPO" "$REMOTE_REPO" "$hardware" "$mode"
inner_cmd+="$run_cmd"

gpuq run \
  --project "$GPUQ_PROJECT" \
  --gpus "$GPUQ_GPUS" \
  --timeout "$GPUQ_TIMEOUT" \
  --output "$output_dir" \
  --cwd "$REMOTE_REPO" -- \
  bash -lc "$inner_cmd"
