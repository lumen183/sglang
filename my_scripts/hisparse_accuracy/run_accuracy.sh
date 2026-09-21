#!/usr/bin/env bash
set -euo pipefail

# Design: ../../my_development/dsv4_hybrid_hisparse_accuracy_testing.md

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
hardware=${1:?usage: run_accuracy.sh HARDWARE MODE}
mode=${2:?usage: run_accuracy.sh HARDWARE MODE}
source "$SCRIPT_DIR/profiles/$hardware.env"

MODEL_PATH=${MODEL_PATH:-/home/jovyan/whw/models/DeepSeek-V4-Flash-0731-W8A8}
GSM8K_DATA_PATH=${GSM8K_DATA_PATH:-/home/jovyan/whw/datasets/gsm8k}
NUM_EXAMPLES=${NUM_EXAMPLES:-200}
NUM_THREADS=${NUM_THREADS:-4}
NUM_SHOTS=${NUM_SHOTS:-20}
MIN_SCORE=${MIN_SCORE:-0.93}
TOP_K=${TOP_K:-512}
DEVICE_BUFFER_SIZE=${DEVICE_BUFFER_SIZE:-4096}
HOST_TO_DEVICE_RATIO=${HOST_TO_DEVICE_RATIO:-2}
RUN_DIR=${RUN_DIR:-$REPO_ROOT/artifacts/hisparse_accuracy/$hardware/$mode/$(date +%Y%m%d-%H%M%S)}
mkdir -p "$RUN_DIR"

case "$mode" in
  baseline)
    HISPARSE_CONFIG=""
    ;;
  native)
    HISPARSE_CONFIG="{\"top_k\":$TOP_K,\"device_buffer_size\":$DEVICE_BUFFER_SIZE,\"host_to_device_ratio\":$HOST_TO_DEVICE_RATIO}"
    ;;
  hybrid-resident)
    HISPARSE_CONFIG="{\"top_k\":$TOP_K,\"device_buffer_size\":$DEVICE_BUFFER_SIZE,\"host_to_device_ratio\":$HOST_TO_DEVICE_RATIO,\"hybrid_mode\":true,\"hybrid_reclaim_watermark\":0.1}"
    ;;
  hybrid-evict)
    NUM_SHOTS=${NUM_SHOTS_EVICT:-128}
    HISPARSE_CONFIG="{\"top_k\":$TOP_K,\"device_buffer_size\":$DEVICE_BUFFER_SIZE,\"host_to_device_ratio\":$HOST_TO_DEVICE_RATIO,\"hybrid_mode\":true,\"hybrid_reclaim_watermark\":0.99}"
    ;;
  *)
    echo "unknown mode: $mode" >&2
    exit 2
    ;;
esac

export PYTHONPATH="$REPO_ROOT/python${PYTHONPATH:+:$PYTHONPATH}"
export SGLANG_DSV4_FP4_EXPERTS=0
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=256
export MC_TCP_ENABLE_CONNECTION_POOL=true

common=(
  --model-path "$MODEL_PATH" --host 127.0.0.1
  --page-size 256 --chunked-prefill-size 8192
  --max-running-requests 16 --mem-fraction-static 0.9
  --trust-remote-code --skip-server-warmup --disable-radix-cache
  --reasoning-parser deepseek-v4 --tool-call-parser deepseekv4
  --model-loader-extra-config '{"enable_multithread_load":true,"num_threads":64}'
  --watchdog-timeout 900
)

prefill=(
  python3 -m sglang.launch_server "${common[@]}"
  --port 30100 --disaggregation-mode prefill
  --disaggregation-bootstrap-port 30500 --disaggregation-transfer-backend nixl
  --nccl-port 30300 --tp "$PREFILL_TP" --base-gpu-id "$PREFILL_BASE_GPU_ID"
)
decode=(
  python3 -m sglang.launch_server "${common[@]}"
  --port 30200 --disaggregation-mode decode
  --disaggregation-bootstrap-port 30500 --disaggregation-transfer-backend nixl
  --nccl-port 30400 --tp "$DECODE_TP" --base-gpu-id "$DECODE_BASE_GPU_ID"
)
if [[ "$mode" != baseline ]]; then
  decode+=(--enable-hisparse --hisparse-config "$HISPARSE_CONFIG")
fi
if [[ "$mode" == hybrid-* ]]; then
  decode+=(--cuda-graph-backend-decode disabled)
fi

pids=()
cleanup() {
  for pid in "${pids[@]}"; do
    kill -TERM -- "-$pid" 2>/dev/null || true
  done
}
trap cleanup EXIT

setsid "${prefill[@]}" >"$RUN_DIR/prefill.log" 2>&1 &
pids+=("$!")
setsid "${decode[@]}" >"$RUN_DIR/decode.log" 2>&1 &
pids+=("$!")

wait_ready() {
  until curl -sf "$1" >/dev/null; do sleep 2; done
}
wait_ready http://127.0.0.1:30100/health
wait_ready http://127.0.0.1:30200/health

setsid python3 -m sglang_router.launch_router \
  --pd-disaggregation --mini-lb \
  --prefill http://127.0.0.1:30100 --decode http://127.0.0.1:30200 \
  --host 127.0.0.1 --port 30000 >"$RUN_DIR/router.log" 2>&1 &
pids+=("$!")
wait_ready http://127.0.0.1:30000/health

python3 "$SCRIPT_DIR/run_gsm8k.py" \
  --base-url http://127.0.0.1:30000 --model "$MODEL_PATH" \
  --data-path "$GSM8K_DATA_PATH" --output-dir "$RUN_DIR" \
  --num-examples "$NUM_EXAMPLES" --num-threads "$NUM_THREADS" \
  --num-shots "$NUM_SHOTS" 2>&1 | tee "$RUN_DIR/gsm8k.log"

python3 "$SCRIPT_DIR/assert_coverage.py" \
  --mode "$mode" --decode-log "$RUN_DIR/decode.log" \
  --output "$RUN_DIR/coverage.json"

python3 - "$RUN_DIR/gsm8k-metrics.json" "$MIN_SCORE" <<'PY'
import json
import sys

score = json.load(open(sys.argv[1]))["score"]
print(f"GSM8K score: {score:.4f}")
if score < float(sys.argv[2]):
    raise SystemExit(1)
PY

echo "result: $RUN_DIR"
