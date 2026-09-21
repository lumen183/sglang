#!/usr/bin/env bash
set -euo pipefail

RUN_DIR=${1:?usage: stop_pd.sh RUN_DIR}

for name in router decode prefill; do
  pid_file="$RUN_DIR/$name.pid"
  [[ -f "$pid_file" ]] || continue
  pid=$(<"$pid_file")
  if kill -0 "$pid" 2>/dev/null; then
    kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
  fi
done

deadline=$((SECONDS + 60))
while (( SECONDS < deadline )); do
  alive=0
  for name in router decode prefill; do
    pid_file="$RUN_DIR/$name.pid"
    [[ -f "$pid_file" ]] || continue
    pid=$(<"$pid_file")
    if kill -0 "$pid" 2>/dev/null; then
      alive=1
    fi
  done
  (( alive == 0 )) && exit 0
  sleep 1
done

for name in router decode prefill; do
  pid_file="$RUN_DIR/$name.pid"
  [[ -f "$pid_file" ]] || continue
  pid=$(<"$pid_file")
  if kill -0 "$pid" 2>/dev/null; then
    kill -KILL -- "-$pid" 2>/dev/null || kill -KILL "$pid" 2>/dev/null || true
  fi
done
