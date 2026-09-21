#!/usr/bin/env bash
set -euo pipefail

# Design: ../../my_development/dsv4_hybrid_hisparse_accuracy_testing.md

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
hardware=${1:-}
mode=${2:-}
if [[ -z "$hardware" || -z "$mode" ]]; then
  echo "usage: $0 h100-8|h200-4 baseline|native|hybrid-resident|hybrid-evict" >&2
  exit 2
fi
profile="$SCRIPT_DIR/profiles/$hardware.env"
[[ -f "$profile" ]] || { echo "unknown hardware profile: $hardware" >&2; exit 2; }
case "$mode" in
  baseline|native|hybrid-resident|hybrid-evict) ;;
  *) echo "unknown mode: $mode" >&2; exit 2 ;;
esac

# shellcheck source=/dev/null
source "$profile"
REMOTE_REPO=${REMOTE_REPO:-/home/jovyan/whw/sglang}
CONTAINER_NAME=${CONTAINER_NAME:-whw_sgl}
GPUQ_PROJECT=${GPUQ_PROJECT:-sglang}
GPUQ_TIMEOUT=${GPUQ_TIMEOUT:-12h}
output_dir=${GPUQ_OUTPUT_DIR:-$PWD/artifacts/gpuq-$hardware-$mode-$(date +%Y%m%d-%H%M%S)}

inner_cmd=""
for variable in \
  MODEL_PATH GSM8K_DATA_PATH ARTIFACT_ROOT NUM_EXAMPLES NUM_THREADS NUM_SHOTS \
  MAX_TOKENS MIN_SCORE TOP_K DEVICE_BUFFER_SIZE HOST_TO_DEVICE_RATIO \
  HISPARSE_CONFIG TRANSFER_BACKEND; do
  if [[ -n ${!variable:-} ]]; then
    printf -v assignment 'export %s=%q && ' "$variable" "${!variable}"
    inner_cmd+="$assignment"
  fi
done
printf -v run_cmd 'cd %q && export PYTHONPATH=%q/python && %q/my_scripts/hisparse_accuracy/run_accuracy.sh --hardware %q --mode %q' \
  "$REMOTE_REPO" "$REMOTE_REPO" "$REMOTE_REPO" "$hardware" "$mode"
inner_cmd+="$run_cmd"

gpuq run \
  --project "$GPUQ_PROJECT" \
  --gpus "$GPUQ_GPUS" \
  --timeout "$GPUQ_TIMEOUT" \
  --output "$output_dir" \
  --cwd / -- \
  bash /home/jovyan/whw/bin/gpuq-docker-exec \
  "$CONTAINER_NAME" "$REMOTE_REPO" bash -lc "$inner_cmd"
