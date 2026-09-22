#!/usr/bin/env bash
set -euo pipefail

REMOTE_REPO=${REMOTE_REPO:-/home/jovyan/whw/sglang}
GPUQ_PROJECT=${GPUQ_PROJECT:-dsv4-test}
GPUQ_TIMEOUT=${GPUQ_TIMEOUT:-30m}
GPUQ_GPUS=${GPUQ_GPUS:-1}
PYTHON_BIN=${PYTHON_BIN:-/home/jovyan/whw/whw_dev/bin/python}
output_dir=${GPUQ_OUTPUT_DIR:-$PWD/artifacts/gpuq-hisparse-unit-$(date +%Y%m%d-%H%M%S)}

printf -v inner_cmd \
  'cd %q && export PYTHONPATH=%q/python && %q -m pytest -q %q %q %q' \
  "$REMOTE_REPO" \
  "$REMOTE_REPO" \
  "$PYTHON_BIN" \
  test/registered/unit/mem_cache/test_hisparse_allocator.py \
  test/registered/unit/managers/test_hisparse_hybrid_policy.py \
  test/registered/unit/mem_cache/test_hisparse_slot_translation.py

gpuq run \
  --project "$GPUQ_PROJECT" \
  --gpus "$GPUQ_GPUS" \
  --timeout "$GPUQ_TIMEOUT" \
  --output "$output_dir" \
  --cwd "$REMOTE_REPO" -- \
  bash -lc "$inner_cmd"
