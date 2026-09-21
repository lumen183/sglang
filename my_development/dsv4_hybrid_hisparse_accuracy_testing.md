# DSV4 Hybrid HiSparse 基础精度测试设计

实现位于 `my_scripts/hisparse_accuracy/`。当前只验证 GSM8K 精度和 hybrid 路径覆盖，不测试高并发或性能。

## 测试矩阵

| mode | 默认 shots | 目标 |
|---|---:|---|
| `baseline` | 20 | 无 HiSparse 基准 |
| `native` | 20 | native HiSparse |
| `hybrid-resident` | 20 | GPU resident top-k |
| `hybrid-evict` | 128 | resident→host/hot |

硬件 profile：H100 使用 8 卡，P/D 各 TP4；H200 使用 4 卡，P/D 各 TP2。

## 执行

```bash
my_scripts/hisparse_accuracy/run_gpuq.sh h200-4 baseline
my_scripts/hisparse_accuracy/run_gpuq.sh h200-4 native
my_scripts/hisparse_accuracy/run_gpuq.sh h200-4 hybrid-resident
my_scripts/hisparse_accuracy/run_gpuq.sh h200-4 hybrid-evict
```

首轮建议缩小为：

```bash
NUM_EXAMPLES=10 NUM_THREADS=1 MIN_SCORE=0 \
  my_scripts/hisparse_accuracy/run_gpuq.sh h200-4 hybrid-resident
```

`run_accuracy.sh` 启动 prefill、decode 和 router，执行 GSM8K，解析 decode 日志，最后终止服务。输出目录包含 `prefill.log`、`decode.log`、`router.log`、`gsm8k.log`、`gsm8k-metrics.json`、HTML report 和 `coverage.json`。

## 路径门禁

`hybrid-resident` 必须出现：

```text
event=resident_admit
event=resident_topk
```

且不能出现 `demote` 或 `host_topk`。

`hybrid-evict` 必须依次出现：

```text
event=resident_admit
event=mirror_complete
event=demote ... freed_c4_slots=N
event=host_topk
```

其中 `N > 0`。Baseline 和 native 不检查 hybrid event。

## 首轮预期

- 三个服务能够启动，GSM8K 请求完成并生成 metrics。
- Baseline/native 没有 hybrid event，分数接近。
- Resident 有 `resident_admit/resident_topk`，没有驱逐。
- Evict 先完成 mirror，再释放 C4 slot，随后走 host top-k。
- GSM8K 正确但 coverage 失败，仍视为没有覆盖目标功能。

常见失败：H200 health 前 OOM 表示 TP2 不可用；evict 没有 demote 表示压力不足；`freed_c4_slots=0` 表示 prompt 未超过 hot buffer；demote 后没有 host_topk 表示降级请求没有继续 decode 或状态更新错误。

## 局限

- H200 TP2 尚未实测。
- 0.99 watermark 只用于确定性功能验证。
- coverage 依赖日志，不统计 H2D bytes/top-k miss。
- 不支持端口并行运行多个测试。
- 不包含吞吐、TTFT、ITL 和高并发数据集。
