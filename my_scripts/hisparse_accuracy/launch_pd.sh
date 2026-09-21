#!/usr/bin/env bash
set -euo pipefail

: "${RUN_DIR:?}"
: "${MODEL_PATH:?}"
: "${PREFILL_TP:?}"
: "${PREFILL_BASE_GPU_ID:?}"
: "${DECODE_TP:?}"
: "${DECODE_BASE_GPU_ID:?}"
: "${HISPARSE_MODE:?}"

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PREFILL_PORT=${PREFILL_PORT:-30100}
DECODE_PORT=${DECODE_PORT:-30200}
ROUTER_PORT=${ROUTER_PORT:-30000}
BOOTSTRAP_PORT=${BOOTSTRAP_PORT:-30500}
PREFILL_NCCL_PORT=${PREFILL_NCCL_PORT:-30300}
DECODE_NCCL_PORT=${DECODE_NCCL_PORT:-30400}
SERVER_LAUNCH_TIMEOUT=${SERVER_LAUNCH_TIMEOUT:-1800}
TRANSFER_BACKEND=${TRANSFER_BACKEND:-nixl}

common_args=(
  --trust-remote-code
  --page-size "${PAGE_SIZE:-256}"
  --chunked-prefill-size "${CHUNKED_PREFILL_SIZE:-8192}"
  --max-running-requests "${MAX_RUNNING_REQUESTS:-16}"
  --mem-fraction-static "${MEM_FRACTION_STATIC:-0.9}"
  --skip-server-warmup
  --reasoning-parser deepseek-v4
  --tool-call-parser deepseekv4
  --model-loader-extra-config "${MODEL_LOADER_EXTRA_CONFIG:-{\"enable_multithread_load\":true,\"num_threads\":64}}"
  --watchdog-timeout "${WATCHDOG_TIMEOUT:-900}"
  --disable-radix-cache
)

prefill_cmd=(
  python3 -m sglang.launch_server
  --model-path "$MODEL_PATH"
  --host 127.0.0.1 --port "$PREFILL_PORT"
  --disaggregation-mode prefill
  --disaggregation-bootstrap-port "$BOOTSTRAP_PORT"
  --disaggregation-transfer-backend "$TRANSFER_BACKEND"
  --nccl-port "$PREFILL_NCCL_PORT"
  --tp "$PREFILL_TP"
  --base-gpu-id "$PREFILL_BASE_GPU_ID"
  "${common_args[@]}"
)

decode_cmd=(
  python3 -m sglang.launch_server
  --model-path "$MODEL_PATH"
  --host 127.0.0.1 --port "$DECODE_PORT"
  --disaggregation-mode decode
  --disaggregation-bootstrap-port "$BOOTSTRAP_PORT"
  --disaggregation-transfer-backend "$TRANSFER_BACKEND"
  --nccl-port "$DECODE_NCCL_PORT"
  --tp "$DECODE_TP"
  --base-gpu-id "$DECODE_BASE_GPU_ID"
  "${common_args[@]}"
)

case "$HISPARSE_MODE" in
  baseline)
    ;;
  native)
    decode_cmd+=(--enable-hisparse --hisparse-config "$HISPARSE_CONFIG")
    ;;
  hybrid-resident|hybrid-evict)
    decode_cmd+=(
      --enable-hisparse
      --cuda-graph-backend-decode disabled
      --hisparse-config "$HISPARSE_CONFIG"
    )
    ;;
  *)
    echo "unsupported mode: $HISPARSE_MODE" >&2
    exit 2
    ;;
esac

if [[ -n ${EXTRA_PREFILL_ARGS:-} ]]; then
  read -r -a parsed_extra <<<"$EXTRA_PREFILL_ARGS"
  prefill_cmd+=("${parsed_extra[@]}")
fi
if [[ -n ${EXTRA_DECODE_ARGS:-} ]]; then
  read -r -a parsed_extra <<<"$EXTRA_DECODE_ARGS"
  decode_cmd+=("${parsed_extra[@]}")
fi

{
  printf 'prefill:'
  printf ' %q' "${prefill_cmd[@]}"
  printf '\n'
  printf 'decode:'
  printf ' %q' "${decode_cmd[@]}"
  printf '\n'
} >"$RUN_DIR/commands.sh"

setsid "${prefill_cmd[@]}" >"$RUN_DIR/prefill.log" 2>&1 &
prefill_pid=$!
echo "$prefill_pid" >"$RUN_DIR/prefill.pid"

setsid "${decode_cmd[@]}" >"$RUN_DIR/decode.log" 2>&1 &
decode_pid=$!
echo "$decode_pid" >"$RUN_DIR/decode.pid"

python3 "$SCRIPT_DIR/wait_http.py" \
  "http://127.0.0.1:${PREFILL_PORT}/health" \
  --timeout "$SERVER_LAUNCH_TIMEOUT" --pid "$prefill_pid"
python3 "$SCRIPT_DIR/wait_http.py" \
  "http://127.0.0.1:${DECODE_PORT}/health" \
  --timeout "$SERVER_LAUNCH_TIMEOUT" --pid "$decode_pid"

router_cmd=(
  python3 -m sglang_router.launch_router
  --pd-disaggregation --mini-lb
  --prefill "http://127.0.0.1:${PREFILL_PORT}"
  --decode "http://127.0.0.1:${DECODE_PORT}"
  --host 127.0.0.1 --port "$ROUTER_PORT"
)
{
  printf 'router:'
  printf ' %q' "${router_cmd[@]}"
  printf '\n'
} >>"$RUN_DIR/commands.sh"

setsid "${router_cmd[@]}" >"$RUN_DIR/router.log" 2>&1 &
router_pid=$!
echo "$router_pid" >"$RUN_DIR/router.pid"
python3 "$SCRIPT_DIR/wait_http.py" \
  "http://127.0.0.1:${ROUTER_PORT}/health" \
  --timeout 300 --pid "$router_pid"
