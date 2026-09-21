#!/usr/bin/env bash
set -euo pipefail

# Design: ../../my_development/dsv4_hybrid_hisparse_accuracy_testing.md

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)

usage() {
  echo "usage: $0 --hardware h100-8|h200-4 --mode baseline|native|hybrid-resident|hybrid-evict" >&2
}

hardware=""
mode=""
while (($#)); do
  case "$1" in
    --hardware)
      hardware=${2:-}
      shift 2
      ;;
    --mode)
      mode=${2:-}
      shift 2
      ;;
    *)
      usage
      exit 2
      ;;
  esac
done

[[ -n "$hardware" && -n "$mode" ]] || { usage; exit 2; }
profile="$SCRIPT_DIR/profiles/$hardware.env"
[[ -f "$profile" ]] || { echo "unknown hardware profile: $hardware" >&2; exit 2; }
case "$mode" in
  baseline|native|hybrid-resident|hybrid-evict) ;;
  *) echo "unknown mode: $mode" >&2; exit 2 ;;
esac

# shellcheck source=/dev/null
source "$profile"
actual_gpu_count=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l | tr -d ' ')
if (( actual_gpu_count < GPUQ_GPUS )); then
  echo "profile $hardware requires $GPUQ_GPUS GPUs, found $actual_gpu_count" >&2
  exit 1
fi
if nvidia-smi --query-gpu=name --format=csv,noheader \
  | head -n "$GPUQ_GPUS" \
  | grep -v "$EXPECTED_GPU_NAME" \
  | grep -q .; then
  echo "profile $hardware expects $EXPECTED_GPU_NAME GPUs" >&2
  nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader >&2
  exit 1
fi

MODEL_PATH=${MODEL_PATH:-/home/jovyan/whw/models/DeepSeek-V4-Flash-0731-W8A8}
GSM8K_DATA_PATH=${GSM8K_DATA_PATH:-/home/jovyan/whw/datasets/gsm8k}
ARTIFACT_ROOT=${ARTIFACT_ROOT:-$REPO_ROOT/artifacts/hisparse_accuracy}
NUM_EXAMPLES=${NUM_EXAMPLES:-200}
NUM_THREADS=${NUM_THREADS:-4}
MAX_TOKENS=${MAX_TOKENS:-512}
MIN_SCORE=${MIN_SCORE:-0.93}
TOP_K=${TOP_K:-512}
DEVICE_BUFFER_SIZE=${DEVICE_BUFFER_SIZE:-4096}
HOST_TO_DEVICE_RATIO=${HOST_TO_DEVICE_RATIO:-2}

case "$mode" in
  baseline)
    NUM_SHOTS=${NUM_SHOTS:-20}
    HISPARSE_CONFIG=""
    ;;
  native)
    NUM_SHOTS=${NUM_SHOTS:-20}
    HISPARSE_CONFIG=${HISPARSE_CONFIG:-"{\"top_k\":$TOP_K,\"device_buffer_size\":$DEVICE_BUFFER_SIZE,\"host_to_device_ratio\":$HOST_TO_DEVICE_RATIO}"}
    ;;
  hybrid-resident)
    NUM_SHOTS=${NUM_SHOTS:-20}
    HISPARSE_CONFIG=${HISPARSE_CONFIG:-"{\"top_k\":$TOP_K,\"device_buffer_size\":$DEVICE_BUFFER_SIZE,\"host_to_device_ratio\":$HOST_TO_DEVICE_RATIO,\"hybrid_mode\":true,\"hybrid_reclaim_watermark\":0.1}"}
    ;;
  hybrid-evict)
    NUM_SHOTS=${NUM_SHOTS:-128}
    # A high watermark makes the correctness path deterministic. It is not a
    # recommended production setting and is not used for performance claims.
    HISPARSE_CONFIG=${HISPARSE_CONFIG:-"{\"top_k\":$TOP_K,\"device_buffer_size\":$DEVICE_BUFFER_SIZE,\"host_to_device_ratio\":$HOST_TO_DEVICE_RATIO,\"hybrid_mode\":true,\"hybrid_reclaim_watermark\":0.99}"}
    ;;
esac

timestamp=$(date +%Y%m%d-%H%M%S)
RUN_DIR=${RUN_DIR:-$ARTIFACT_ROOT/$hardware/$mode/$timestamp}
mkdir -p "$RUN_DIR"

export PYTHONPATH="$REPO_ROOT/python${PYTHONPATH:+:$PYTHONPATH}"
export RUN_DIR MODEL_PATH HISPARSE_MODE="$mode" HISPARSE_CONFIG
export PREFILL_TP PREFILL_BASE_GPU_ID DECODE_TP DECODE_BASE_GPU_ID
export SGLANG_DSV4_FP4_EXPERTS=${SGLANG_DSV4_FP4_EXPERTS:-0}
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=${SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK:-256}
export MC_TCP_ENABLE_CONNECTION_POOL=${MC_TCP_ENABLE_CONNECTION_POOL:-true}

cd "$REPO_ROOT"
python3 "$SCRIPT_DIR/write_metadata.py" \
  --output "$RUN_DIR/metadata.json" \
  --hardware "$hardware" --mode "$mode" \
  --model "$MODEL_PATH" --dataset "$GSM8K_DATA_PATH" \
  --hisparse-config "$HISPARSE_CONFIG" \
  --num-examples "$NUM_EXAMPLES" --num-threads "$NUM_THREADS" \
  --num-shots "$NUM_SHOTS"

cleanup() {
  "$SCRIPT_DIR/stop_pd.sh" "$RUN_DIR" || true
}
trap cleanup EXIT INT TERM

"$SCRIPT_DIR/launch_pd.sh"
base_url="http://127.0.0.1:${ROUTER_PORT:-30000}"

python3 "$SCRIPT_DIR/run_fixed_prompts.py" \
  --base-url "$base_url" \
  --prompts "$SCRIPT_DIR/fixed_prompts.json" \
  --output "$RUN_DIR/fixed-prompts.jsonl"

python3 "$SCRIPT_DIR/run_gsm8k.py" \
  --base-url "$base_url" --model "$MODEL_PATH" \
  --data-path "$GSM8K_DATA_PATH" --output-dir "$RUN_DIR" \
  --num-examples "$NUM_EXAMPLES" --num-threads "$NUM_THREADS" \
  --num-shots "$NUM_SHOTS" --max-tokens "$MAX_TOKENS" \
  2>&1 | tee "$RUN_DIR/gsm8k.log"

cleanup
trap - EXIT INT TERM

python3 "$SCRIPT_DIR/assert_coverage.py" \
  --mode "$mode" --decode-log "$RUN_DIR/decode.log" \
  --output "$RUN_DIR/coverage-summary.json"
python3 "$SCRIPT_DIR/summarize.py" \
  --run-dir "$RUN_DIR" --min-score "$MIN_SCORE"

echo "accuracy run complete: $RUN_DIR"
