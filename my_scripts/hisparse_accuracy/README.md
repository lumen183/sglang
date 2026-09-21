# DSV4 Hybrid HiSparse Accuracy Harness

This directory contains the first-stage correctness harness for DeepSeek V4
Hybrid HiSparse. It intentionally does not implement high-concurrency or
throughput benchmarking.

Design and acceptance criteria:

- [Accuracy harness design](../../my_development/dsv4_hybrid_hisparse_accuracy_testing.md)
- [Overall adaptation plan](../../my_development/dsv4_hybrid_hisparse_adaptation.md)

Run from the development machine through GPUQ:

```bash
my_scripts/hisparse_accuracy/run_gpuq.sh h100-8 hybrid-resident
my_scripts/hisparse_accuracy/run_gpuq.sh h200-4 hybrid-evict
```

Run inside the `whw_sgl` container when GPUs are already allocated:

```bash
my_scripts/hisparse_accuracy/run_accuracy.sh \
  --hardware h200-4 \
  --mode hybrid-resident
```

Supported modes are `baseline`, `native`, `hybrid-resident`, and
`hybrid-evict`. Override defaults through environment variables documented in
the design document, for example:

```bash
NUM_EXAMPLES=20 NUM_THREADS=1 \
  my_scripts/hisparse_accuracy/run_accuracy.sh \
  --hardware h200-4 --mode hybrid-resident
```

Every run writes metadata, commands, P/D/router logs, fixed-prompt responses,
GSM8K metrics, path-coverage results, and a Markdown summary below
`artifacts/hisparse_accuracy/`.

Local harness checks (no GPU required):

```bash
python3 my_scripts/hisparse_accuracy/test_harness.py
```
