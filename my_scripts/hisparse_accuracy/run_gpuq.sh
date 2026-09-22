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

# gpuq jobs inherit the daemon's environment rather than the submitter's
# activated conda environment. Pass the interpreter explicitly so the server
# and evaluator use the same environment as the submitter.
PYTHON_BIN=${PYTHON_BIN:-${CONDA_PREFIX:-/home/jovyan/whw/whw_dev}/bin/python}
# GPUQ accuracy jobs should use the fast DeepGEMM prewarm path by default. Keep
# the value overridable so a caller can still disable it with `=0`.
export SGLANG_JIT_DEEPGEMM_FAST_WARMUP="${SGLANG_JIT_DEEPGEMM_FAST_WARMUP:-1}"

inner_cmd=""
for variable in \
  PYTHON_BIN SGLANG_JIT_DEEPGEMM_FAST_WARMUP MODEL_PATH GSM8K_DATA_PATH NUM_EXAMPLES NUM_THREADS NUM_SHOTS NUM_SHOTS_EVICT \
  MIN_SCORE TOP_K DEVICE_BUFFER_SIZE HOST_TO_DEVICE_RATIO RUN_DIR \
  PREFILL_PORT DECODE_PORT ROUTER_PORT DISAGGREGATION_BOOTSTRAP_PORT \
  PREFILL_NCCL_PORT DECODE_NCCL_PORT \
  NCCL_DEBUG NCCL_DEBUG_SUBSYS NCCL_SOCKET_IFNAME NCCL_IB_HCA \
  NCCL_IB_DISABLE NCCL_P2P_DISABLE NCCL_SHM_DISABLE; do
  if [[ -n ${!variable:-} ]]; then
    printf -v assignment 'export %s=%q && ' "$variable" "${!variable}"
    inner_cmd+="$assignment"
  fi
done
printf -v run_cmd 'cd %q && export PYTHONPATH=%q/python && %q/my_scripts/hisparse_accuracy/run_accuracy.sh %q %q' \
  "$REMOTE_REPO" "$REMOTE_REPO" "$REMOTE_REPO" "$hardware" "$mode"
inner_cmd+="$run_cmd"

# Preserve the complete exported environment from the submitting shell. GPUQ
# normally starts jobs from the daemon's environment, which may not include the
# submitter's activated conda environment or library paths.
gpuq_env_args=()
while IFS= read -r -d '' gpuq_env_entry; do
  gpuq_env_args+=(--env "$gpuq_env_entry")
done < <(env -0)

gpuq run \
  "${gpuq_env_args[@]}" \
  --project "$GPUQ_PROJECT" \
  --gpus "$GPUQ_GPUS" \
  --timeout "$GPUQ_TIMEOUT" \
  --output "$output_dir" \
  --cwd "$REMOTE_REPO" -- \
  bash -lc "$inner_cmd"
