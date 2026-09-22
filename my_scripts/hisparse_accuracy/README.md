# DSV4 HiSparse 精度测试

[设计说明](../../my_development/dsv4_hybrid_hisparse_accuracy_testing.md)

首次先跑 10 题：

```bash
NUM_EXAMPLES=10 NUM_THREADS=1 MIN_SCORE=0 ./run_gpuq.sh h200-4 hybrid-resident
```

完整测试：

```bash
./run_gpuq.sh h200-4 baseline
./run_gpuq.sh h200-4 native
./run_gpuq.sh h200-4 hybrid-resident
./run_gpuq.sh h200-4 hybrid-evict
```

H100 将 `h200-4` 换成 `h100-8`。已分配 GPU 时直接运行：

```bash
./run_accuracy.sh h200-4 hybrid-resident
```

参数：`MODEL_PATH`、`GSM8K_DATA_PATH`、`NUM_EXAMPLES`、`NUM_THREADS`、`NUM_SHOTS`、`NUM_SHOTS_EVICT`、`MIN_SCORE`、`TOP_K`、`DEVICE_BUFFER_SIZE`、`HOST_TO_DEVICE_RATIO`、`RUN_DIR`。

默认 200 题、20-shot、4 线程、门槛 0.93；evict 为 128-shot。结果在 `artifacts/hisparse_accuracy/`。局限：只测精度；H200 TP2 未实测；驱逐依赖长 prompt 和显存池；coverage 依赖日志；固定端口不能并行运行。
