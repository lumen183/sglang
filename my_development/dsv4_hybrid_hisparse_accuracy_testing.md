# DSV4 Hybrid HiSparse 基础精度测试框架

更新日期：2026-09-21

实现目录：`my_scripts/hisparse_accuracy/`

## 1. 当前范围

第一版框架只回答两个问题：

1. DSV4 在 baseline、native HiSparse 和 hybrid HiSparse 下的基础精度是否正确；
2. hybrid 测试是否真正走过预期的 GPU-resident 或 resident→host/hot 路径。

暂不包含高并发、吞吐、TTFT/ITL、PCIe 带宽或长期稳定性测试。`NUM_THREADS` 默认只有 4，它表示精度客户端的少量并行，不作为并发性能结论。

## 2. 测试矩阵

| mode | HiSparse | shots | 默认线程 | 路径目标 |
|---|---|---:|---:|---|
| `baseline` | 关闭 | 20 | 4 | DSV4 基准精度 |
| `native` | native | 20 | 4 | 现有 host-backed 实现精度 |
| `hybrid-resident` | hybrid | 20 | 4 | 请求保持 C4 GPU resident |
| `hybrid-evict` | hybrid | 128 | 4 | mirror 后降级到 host + hot buffer |

`hybrid-evict` 默认使用 `hybrid_reclaim_watermark=0.99`，目的是在低并发精度测试里稳定制造压力，不是生产推荐值，也不能用于性能结论。128-shot 用于让压缩后的 C4 history 更可能超过 `device_buffer_size=4096`，从而真实释放 GPU C4 slot。

每种 mode 都先执行 4 个固定 greedy prompt，再执行 GSM8K。默认 GSM8K 200 题、`temperature=0`、`top_p=1`、最多输出 512 tokens。

## 3. 硬件 profile

测试逻辑不直接判断 GPU 型号，资源拓扑由 profile 描述：

| profile | GPUQ 卡数 | Prefill | Decode |
|---|---:|---|---|
| `h100-8` | 8 | GPU 0-3, TP4 | GPU 4-7, TP4 |
| `h200-4` | 4 | GPU 0-1, TP2 | GPU 2-3, TP2 |

启动前会使用 `nvidia-smi` 检查卡数和型号。H200 TP2 能否加载 `/home/jovyan/whw/models/DeepSeek-V4-Flash-0731-W8A8` 必须通过首次 smoke 实测；若不能加载，应新增 `h200-8.env`，而不是在脚本中隐式改变拓扑。

## 4. PD 拓扑

每个 run 都启动三个独立进程组：

```text
prefill server ─┐
                ├─ PD router :30000 ─ client
decode server  ─┘
```

默认端口：

- router: 30000
- prefill: 30100
- decode: 30200
- prefill NCCL: 30300
- decode NCCL: 30400
- disaggregation bootstrap: 30500

默认 transfer backend 为 NIXL。所有 mode 都使用同一模型、page size、chunked prefill、reasoning/tool parser 和 radix-off 配置。Hybrid mode 额外关闭 decode CUDA Graph，因为当前实现明确要求如此。

## 5. 路径覆盖事件

测试不能用 `nvidia-smi` 推测 resident/eviction。Coordinator 输出低频生命周期事件：

```text
HiSparse hybrid event=resident_admit ...
HiSparse hybrid event=resident_topk
HiSparse hybrid event=mirror_complete ...
HiSparse hybrid event=demote ... freed_c4_slots=N
HiSparse hybrid event=host_topk
```

`resident_topk` 和 `host_topk` 在每个 server 进程生命周期只记录首次观察，避免逐层逐 token 产生日志。多 TP rank 可能重复同一种事件，所以门禁判断是否存在，不依赖事件数量代表请求数。

### hybrid-resident 门禁

```text
resident_admit > 0
resident_topk > 0
demote == 0
host_topk == 0
```

### hybrid-evict 门禁

```text
resident_admit > 0
mirror_complete > 0
demote > 0
freed_c4_slots > 0
host_topk > 0
```

同时检查首次事件满足：

```text
resident_admit -> mirror_complete -> demote -> host_topk
```

Baseline/native 当前只做精度门禁，不要求 hybrid lifecycle event。

## 6. 目录职责

```text
my_scripts/hisparse_accuracy/
  profiles/*.env          GPU 数和 P/D TP 拓扑
  run_gpuq.sh             开发机上的 GPUQ 提交入口
  run_accuracy.sh         容器内单次测试总入口
  launch_pd.sh            启动 P/D 和 router
  stop_pd.sh              终止三个进程组
  wait_http.py            health wait + 进程早退检测
  run_fixed_prompts.py    保存固定 prompt 完整响应
  run_gsm8k.py            调用仓库 GSM8K evaluator
  assert_coverage.py      resident/evict 路径门禁
  summarize.py            accuracy + coverage 总报告
  compare_runs.py         两个完成 run 的精度/固定输出比较
```

`my_scripts/hisparse_accuracy/README.md` 引用本文，入口脚本也在 `--help/usage` 和产物中保持 mode/profile 名称一致。

## 7. 使用方式

从开发机提交 GPUQ：

```bash
my_scripts/hisparse_accuracy/run_gpuq.sh h100-8 baseline
my_scripts/hisparse_accuracy/run_gpuq.sh h100-8 native
my_scripts/hisparse_accuracy/run_gpuq.sh h100-8 hybrid-resident
my_scripts/hisparse_accuracy/run_gpuq.sh h100-8 hybrid-evict

my_scripts/hisparse_accuracy/run_gpuq.sh h200-4 hybrid-resident
```

已经进入 GPUQ 容器时：

```bash
cd /home/jovyan/whw/sglang
export PYTHONPATH=$PWD/python
my_scripts/hisparse_accuracy/run_accuracy.sh \
  --hardware h200-4 --mode hybrid-resident
```

快速 smoke：

```bash
NUM_EXAMPLES=10 NUM_THREADS=1 MIN_SCORE=0 \
  my_scripts/hisparse_accuracy/run_accuracy.sh \
  --hardware h200-4 --mode hybrid-resident
```

常用覆盖变量：

| 变量 | 默认值 |
|---|---|
| `MODEL_PATH` | `/home/jovyan/whw/models/DeepSeek-V4-Flash-0731-W8A8` |
| `GSM8K_DATA_PATH` | `/home/jovyan/whw/datasets/gsm8k` |
| `NUM_EXAMPLES` | 200 |
| `NUM_THREADS` | 4 |
| `MIN_SCORE` | 0.93 |
| `TOP_K` | 512 |
| `DEVICE_BUFFER_SIZE` | 4096 |
| `HOST_TO_DEVICE_RATIO` | 2 |
| `TRANSFER_BACKEND` | `nixl` |
| `ARTIFACT_ROOT` | `<repo>/artifacts/hisparse_accuracy` |

## 8. 产物

```text
artifacts/hisparse_accuracy/<hardware>/<mode>/<timestamp>/
  metadata.json
  commands.sh
  prefill.log
  decode.log
  router.log
  fixed-prompts.jsonl
  gsm8k.log
  gsm8k-metrics.json
  gsm8k-report.html
  coverage-summary.json
  report.md
```

`metadata.json` 保存 git SHA、dirty status、GPU inventory、模型、数据、mode、shots/threads 和完整 HiSparse 配置。`commands.sh` 保存实际启动参数，不能只依赖设计文档中的默认值复现实验。

## 9. 结果比较

单次 run 先要求：

- GSM8K score 不低于 `MIN_SCORE`；
- 对应 mode 的 path coverage 通过；
- 固定 prompt 请求全部成功。

完成相同 shots 的两次 run 后，可比较 native 与 hybrid：

```bash
python3 my_scripts/hisparse_accuracy/compare_runs.py \
  artifacts/.../native/<run> \
  artifacts/.../hybrid-resident/<run>
```

默认要求固定 prompt 文本完全一致，GSM8K score 差值不超过 0.005。不要比较 shots 不同的两个 run；`hybrid-evict` 应与显式设置 `NUM_SHOTS=128` 的 native run 比较。

## 10. 已知限制与后续工作

- 当前 GSM8K evaluator 的逐题输出保存在 HTML report，而不是独立 JSONL；后续如需自动逐题 diff，应扩展 evaluator 返回结构。
- 路径事件证明 resident/host 分支被调用，但还没有统计 top-k miss 数和精确 H2D bytes。
- `hybrid-evict` 仍可能因实际 C4 pool 很大而未触发；这种情况 coverage 会失败，应调整 watermark/device buffer 或增加专用 deterministic pressure hook，不能把失败忽略。
- H200 TP2 模型加载尚待远程实测。
- 高并发、random-ids、ShareGPT、吞吐与延迟指标不属于本文范围。

下一阶段应先根据 H100/H200 的第一轮日志修正 profile 和确定性驱逐条件，再增加逐题 JSONL；确认基础精度后才开始并发框架。
