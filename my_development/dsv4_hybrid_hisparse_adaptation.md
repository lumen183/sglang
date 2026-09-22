# DeepSeek V4 Hybrid HiSparse 对比与分阶段适配方案

更新日期：2026-09-21

## 1. 调研基线与结论

本文比较以下四个代码点：

| 对象 | 版本 | 定位 |
|---|---|---|
| SGLang upstream main | `80da4432d0` | 已有 DSV4 native HiSparse |
| 当前工作分支 | `c65367aab9`，`dsv4-hybrid-hisparse` | 在 upstream main 上增加了 7 个 hybrid 原型提交 |
| SGLang PR #35488 | `a9b734d1d9`，`pr/hisparse-hicache` | 为 DSA HiSparse 增加 HiCache backing；当前不覆盖 DSV4 |
| vLLM main | `b8cf2753825d` | 页级、异步写回、共享 GPU block pool 的 hybrid HiSparse；源码明确拒绝 DSV4 |

结论：最快验证 DSV4 功能的路线不是直接移植 PR #35488，也不是直接搬 vLLM 代码，而是：

1. 以 SGLang upstream main 的 DSV4 native HiSparse 数据面为基础；
2. 先完成当前分支已有的“低压全驻 GPU、高压按请求降级、后台镜像”闭环；
3. 再把请求级回收逐步改为 vLLM 的页级 resident/clean/unpinned 状态机；
4. 最后才接 PR #35488 的 coordinator protocol、mixed pass-through kernel 和 HiCache/radix 能力。

首个可验收版本继续限制为 PD decode、`--disable-radix-cache`、关闭 decode CUDA Graph。这符合“最快验证 DSV4，不做兼容设计”的目标。

> 重要校正：vLLM 当前实现不是让同一请求的 hot buffer 随压力逐渐变大。`HiSparseHotSpec.blocks_per_request` 是固定值；显存跌破 watermark 后，请求被标记 `require_hot`，随着更多请求发生 resident→host 转换，系统中拥有固定 hot region 的请求数逐渐增加。

## 2. 三套实现的核心数据流

### 2.1 SGLang DSV4 native HiSparse

prefill/PD 传入完整 C4 KV 后，将历史 C4 写入 private pinned host pool；GPU 仅保留每个请求固定大小的 device buffer。每层 DSA indexer 得到 top-k 后，swap-in kernel 在 GPU buffer 命中，未命中的行从 host 拉回，然后 sparse attention 消费该 buffer。

关键代码：

- `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py`
- `python/sglang/srt/mem_cache/allocator/hisparse.py`
- `python/sglang/srt/managers/hisparse_coordinator.py`
- `python/sglang/kernels/ops/kvcache/hisparse.py`
- `python/sglang/srt/layers/attention/deepseek_v4_backend.py`
- `test/registered/disaggregation/test_disaggregation_hisparse.py`

优点是 DSV4 packed/page layout、C4 压缩映射、PD direct-to-host 和 top-k swap-in 已经可用。缺点是即使 HBM 充足，非 top-k 历史也会进入 host，低压时仍支付 host I/O 和固定 hot-buffer 管理成本。

### 2.2 vLLM hybrid HiSparse

vLLM 把 source(host)、indexer、resident、hot 建成不同 KV cache group，但 resident 与 hot 共用 GPU block pool：

1. 新页先留在 GPU resident cache；
2. sealed page 后台写入 host，形成 dirty→clean；
3. 低压时 clean page 仍可由请求直接读取；
4. `free_blocks < max(hot_cost, total_blocks / 10)` 时给请求申请固定 hot region；
5. hot region 可用后 clean resident page 被 unpin；只有 block 真被复用时，该页才从请求 block table 置空；
6. top-k 对仍 resident 的页走零拷贝 pass-through，对已经丢失的页才从 host swap-in。

关键代码（vLLM main）：

- `vllm/v1/hisparse/coordinator.py`
- `vllm/v1/hisparse/layout.py`
- `vllm/v1/core/single_type_kv_cache_manager.py`
- `vllm/v1/kv_cache_interface.py`
- `vllm/distributed/kv_transfer/kv_connector/v1/hisparse/`

当前限制：`vllm/v1/hisparse/layout.py::_partition_hisparse_specs` 明确抛出 `HiSparse does not support DeepSeek V4.`。因此能借鉴的是状态机、共享池、水位线和异步发布语义，不是 DSV4 layout 代码。

### 2.3 SGLang PR #35488

PR #35488 的目标是让已有 DSA HiSparse 的“非 hot KV backing”可插拔：

- `private_host`：关闭 radix，维持 native 行为；
- `hicache`：开启 radix + hierarchical cache + write-back，resident 页由 radix/HiCache 驱动逐出，top-k 对 resident 页 pass-through，对 host 页 swap-in。

它增加了：

- `hisparse_protocol.py`：统一 coordinator 生命周期接口；
- `hisparse_hicache_coordinator.py`：HiCache admission、eviction hook、expanded indexer pages、mixed swap-in；
- `hisparse_hicache_admission.py`：纯整数 admission ledger/quota；
- radix tree eviction hook 与 copy-less-drop veto；
- kernel 的 resident pass-through source 和 late-bound host address；
- scheduler admission gate、未来 token reservation；
- radix/prefix reuse 与 CUDA Graph 路径。

但 PR 的 protocol 注释明确写明 DeepSeek V4、PD direct-to-host、shared-index prefetch 仍属于 private-host concrete path，不进入通用 protocol。它适合作为第二阶段架构参考，不能视为 DSV4 hybrid 的现成实现。

## 3. 差异矩阵

| 维度 | SGL DSV4 native | 当前工作分支 hybrid 原型 | vLLM main | PR #35488 |
|---|---|---|---|---|
| DSV4 支持 | 已支持 | 已做专用原型 | 明确拒绝 | 未纳入 HiCache backing |
| 低压历史 KV | host | GPU resident，同时异步镜像 host | GPU resident，同时后台发布 host | HiCache device tier |
| 压力触发 | 无，固定 host-backed | `max(hot_cost, ratio * total)` | `max(hot_cost, 10% * total)` | radix/HiCache eviction + admission quota |
| 回收粒度 | 请求 staging | 请求级 FIFO 降级 | 页级 clean/unpin/reuse | radix node/page eviction |
| hot buffer | 始终存在 | 降级时分配，默认 `2 * model index_topk` | 降级时分配固定 blocks/request | admission 时分配固定 buffer |
| resident+host 混合读取 | 否 | resident 请求全走 C4；降级后全走 host/hot | 同一请求可逐页混合 | 同一请求可逐页混合 |
| resident top-k | 不适用 | `_resolve_resident_topk` 映射原 C4 slot，零 H2D | pass-through source | pass-through source |
| host 发布 | staging 时整请求拷贝 | prefill 全历史异步镜像；decode 增量备份 | sealed page 异步发布 | HiCache write-back |
| GPU 页释放条件 | staging 完成后 | mirror 完成后，整请求切换 | host durable + hot ready；实际复用时失去 resident 页 | HiCache host copy/锁完成后 |
| 选择策略 | 无 | FIFO request，支持 protected set | shared block pool/LRU reuse | radix eviction policy |
| prefix/radix | 必须关闭 | 必须关闭 | 可结合 prefix cache | 核心目标是启用 radix |
| PD direct-to-host | 已有 | 保留 | connector 体系不同 | 只在 private-host concrete path |
| decode CUDA Graph | native 路径已有支持面 | 当前强制关闭 | 设计为 graph-safe | graph-static buffer + late-bound host |
| 可观测性 | token usage 为主 | token usage + demotion 日志，仍不足 | hot hit/miss、H2D bytes 等 stats | admission/eviction 日志，指标仍需整理 |

当前原型已经做对的部分：

- `hybrid_mode` 仅允许 DSV4，避免误用通用 DSA 路径；
- 默认 hot buffer 从模型 `index_topk` 推出，而不是硬编码 4096；
- watermark 已对齐 vLLM 的 block 单位和 `max(hot_cost, reserve)`；
- allocator 在 C4 不足时回调 coordinator reclaim；
- prefill 历史后台镜像，reclaim 不再在 scheduler 线程同步复制整请求；
- resident top-k 可直接解析到 C4 physical slot；
- reclaim 会避开当前受保护请求，并等待 decode backup 完成后再释放页。

仍未对齐 vLLM 的关键项：

1. 当前按请求 FIFO 降级，而非 page-level dirty/clean/unpinned；
2. 降级后请求从“全 resident”一次切为“host + fixed hot”，没有同一请求的 resident/host 混合页；
3. 没有 block reuse callback，GPU 页释放与 block table 失效不是 vLLM 的惰性复用语义；
4. host 空间在 admission 时为每个 resident 请求预留完整历史，HBM 省了但 pinned host 峰值不会按真实 spill 推迟；
5. 压力检查主要发生在 admission/allocator failure，缺少统一的每轮 scheduler boundary policy；
6. mirror event、实际释放页数、top-k hit/miss、H2D bytes/latency 没有完整指标；
7. hybrid 强制关闭 decode CUDA Graph；
8. radix/prefix reuse 仍关闭；
9. failure rollback、host pool exhaustion、请求 retract/finish 与异步 mirror 竞争需要 GPU 压测证明；
10. DSV4 的 C4 page、compress ratio=4 和 SWA tail 让 logical/C4/host 三套索引并存，不能照搬 vLLM MLA block table。

## 4. 分阶段开发计划

每个阶段都必须先过本阶段门禁，再进入下一阶段。不要同时改 policy、kernel 和 CUDA Graph，否则精度回归很难定位。

### 阶段 0：冻结 native 基线和实验协议

目标：证明 upstream main 的 DSV4 native HiSparse 在目标模型、PD 拓扑和两种硬件上可复现。

工作：

- 固定 `origin/main` SHA、模型目录 hash、数据集 revision、镜像、CUDA/driver/torch；
- 固定 TP、P/D GPU 数、NIXL/Mooncake transport、KV dtype、后端；
- 保存 dense/no-HiSparse 与 native HiSparse 两套启动参数；
- 所有性能点都重启服务、warmup，再测 3 次；不同 GPU 型号不混合汇总。

验证门禁：

- PD health 和单请求 smoke 成功；
- native HiSparse 发生 host backup 和 top-k swap-in；
- GSM8K 200 题 native 相对 baseline 精度差不超过 0.5 percentage point；
- 30K 输入、512 输出在目标并发下无 OOM/NaN/卡死。

### 阶段 1：补齐原型的可观测性和确定性测试

目标：先证明当前 7 个提交的状态转换正确，不改变管理粒度。

新增指标建议：

- `hisparse_resident_requests`、`hisparse_host_backed_requests`；
- `hisparse_resident_pages`、`hisparse_hot_pages`、`hisparse_free_c4_pages`；
- `hisparse_mirror_bytes/time`、`hisparse_reclaim_bytes/time`；
- `hisparse_hot_hits/misses`、`hisparse_h2d_bytes/time`；
- transition watermark、reclaim target/actual、未完成 mirror 阻塞次数。

新增测试：

- watermark 边界 `free == watermark` 不转换，`free == watermark - 1` 转换；
- mirror 未完成时不能 release，完成后才能 release；
- finish/retract 在 mirror 前后均不泄漏 host/GPU slot；
- protected request 不被回收；
- host pool 不足返回可诊断错误而非悬挂；
- DSV4 C4 page 边界、非整页 prompt、SWA tail 和最后一个 writable page。

验证门禁：CPU unit 全过；GPU kernel layout 测试全过；低压运行中 demotion=0、H2D miss=0；强制小 C4 pool 时必然发生 demotion 且输出 token 与 native 一致。

### 阶段 2：完成请求级 hybrid MVP

目标：得到最快可跑、可比较的 DSV4 hybrid 版本。

工作：

- 把压力检查放到明确的 decode scheduling boundary，同时保留 allocator failure 回调兜底；
- 严格定义状态 `MIRRORING -> RESIDENT_CLEAN -> DEMOTING -> HOST_HOT`；
- 所有释放都要求 mirror publication event 完成；
- 对 admission、retract、finish、pause/requeue 做幂等清理；
- 保持 FIFO 请求级降级、private host、radix off、CUDA Graph off，不扩 scope。

验证门禁：

- 低压：hybrid 全 resident，输出与 baseline/native 一致，H2D bytes 接近 0；
- 高压：能够自动降级，free C4 回到 watermark 以上，无 OOM；
- 压力波动：resident 与 host-backed 请求可共存，完成/取消后容量完全恢复；
- 8/32/64 并发长上下文连续运行至少 30 分钟。

### 阶段 3：改为 vLLM 页级管理

目标：功能语义真正向 vLLM 对齐。

工作：

- 为每个 request/page 记录 dirty、pending-copy、clean-pinned、clean-unpinned、lost；
- 新 sealed C4 page 以后台任务写 host，host durable 后才可 unpin；
- resident 和 hot 使用同一 C4 page allocator；
- 分配固定 hot region 后，逐页 unpin clean resident page；
- allocator 真复用页时回调，将对应 request page mapping 置空；
- top-k kernel 输入增加第三种来源：resident pass-through、hot-buffer hit、host miss；
- 保留 active tail pages，避免当前 step 刚写入的页被回收；
- selection 先用 shared pool 的复用顺序，不引入复杂自定义策略。

可复用 vLLM：状态机、watermark、hot/resident shared pool、publication fence、block reuse callback 语义。不可直接复用：KVCacheSpec/layout、DSV4 C4 mapping、SWA、PD transfer。

验证门禁：构造同一请求内 30% resident/70% host 的 mixed page；resident top-k 不产生 H2D，host top-k 才产生 H2D；随机逐页复用后与 native logits/top-k/output 对齐；实际释放 pages 达到 reclaim target。

### 阶段 4：吸收 PR #35488 的通用抽象和 CUDA Graph

目标：在 MVP 正确后减少分叉，并恢复性能能力。

优先借用：

- `HiSparseCoordinator` protocol 与 lifecycle 命名；
- graph-static `num_real_reqs`、top-k output buffer；
- mixed pass-through swap-in kernel flags；
- scheduler admission budget/未来 token reservation；
- eviction application 只发生在两次 forward 之间；
- copy-less-drop veto 和 host publication/lock 顺序。

暂不做：HiCache/radix backing、通用 DSA 模型、多 backend 兼容。先恢复 decode CUDA Graph，比较 eager/graph 精度和吞吐；之后若 DSV4 需要 prefix reuse，再单独适配 HiCache backing。

验证门禁：CUDA Graph replay 不读 stale mapping；batch padding 行不访问 host；graph on/off 的逐 token 输出一致；ITL 和吞吐不低于阶段 3 eager。

### 阶段 5：可选的 DSV4 HiCache/radix 集成

目标：需要共享前缀/多轮复用时才做。

工作：扩展 PR #35488 的 backing protocol，使 DSV4 C4、SWA 和 PD direct-to-host 成为一等路径；使用 radix eviction hook 驱动页失效；为 expanded indexer/C4 pages 建立独立容量模型。

验证门禁：共享 26K + 独立 15K 工作负载中 radix hit 生效；live request 对应的页绝不 copy-less drop；HiCache write-back 后再释放 device page；相对 private-host hybrid 吞吐有正收益才合入。

## 5. GPUQ 验证流程

以下命令从内网宿主提交，所有真正 GPU 测试必须通过 GPUQ。H20 和 H100 分开建报告目录。

```bash
gpuq run --project sglang --gpus 8 --timeout 12h \
  --output "$PWD/artifacts/H20" --cwd /home/jovyan/whw/sglang -- \
  bash -lc '
    set -euo pipefail
    cd /home/jovyan/whw/sglang
    export PYTHONPATH=/home/jovyan/whw/sglang/python
    nvidia-smi
    git rev-parse HEAD
    python - <<"PY"
import torch
print(torch.__version__, torch.version.cuda)
print(torch.cuda.get_device_name(0), torch.cuda.get_device_properties(0).total_memory)
PY
    pytest -q \
      test/registered/unit/managers/test_hisparse_hybrid_policy.py \
      test/registered/unit/mem_cache/test_hisparse_allocator.py \
      test/registered/unit/mem_cache/test_hisparse_max_token_pool_size.py
  '
```

模型和数据统一使用：

```text
/home/jovyan/whw/models/DeepSeek-V4-Flash-0731
/home/jovyan/whw/datasets/gsm8k
```

PD 启动以仓库 `test/registered/disaggregation/test_disaggregation_hisparse.py` 的参数为准。Prefill 实例不启用 HiSparse；Decode 实例分别运行三种 mode：

```text
baseline: 不加 --enable-hisparse
native:   --enable-hisparse --disable-radix-cache
          --hisparse-config '{"top_k":2048,"device_buffer_size":4096,"host_to_device_ratio":2}'
hybrid:   --enable-hisparse --disable-radix-cache --cuda-graph-backend-decode disabled
          --hisparse-config '{"top_k":2048,"device_buffer_size":4096,"host_to_device_ratio":2,"hybrid_mode":true,"hybrid_reclaim_watermark":0.1}'
```

不要用 standalone server 作为最终结论；它只适合 kernel/smoke 定位。最终准确度和性能均通过 PD 的对外 endpoint 发请求。

### 5.1 每次实验的固定顺序

1. 清理旧服务并创建唯一 run directory；
2. 保存 git SHA/diff、完整 launch command、环境和 GPU 信息；
3. 启动 P/D，等待 health；
4. 发 1 个确定性 smoke 请求；
5. warmup 20 请求；
6. 清 cache，开始外部 GPU/CPU/网络采样；
7. 执行 benchmark 或 accuracy；
8. 保存 server log、client JSONL、监控 CSV；
9. 停服；同一点重复 3 次。

## 6. 精确度验证与报告

顺序：smoke token diff → GSM8K 200 → GSM8K 全量。baseline/native/hybrid 必须使用相同 prompts、顺序、seed、temperature=0、top-p=1、reasoning/chat template 和 max-new-tokens。

示例（endpoint 参数按实际 PD gateway 替换）：

```bash
python -m sglang.test.few_shot_gsm8k \
  --host 127.0.0.1 --port "${PD_PORT}" \
  --data-path /home/jovyan/whw/datasets/gsm8k \
  --num-questions 200 --num-shots 5 --parallel 32 \
  --temperature 0 --max-new-tokens 4096
```

精度门槛：

- hybrid-native exact match 差值绝对值不超过 0.5pp；
- hybrid-baseline exact match 差值绝对值不超过 0.5pp；
- invalid answer 增量不超过 0.1pp；
- 固定 smoke prompts 的 greedy token 序列必须完全一致；
- 不一致时保存逐题 prompt、三模式 raw output、首次不同 token 和对应服务日志。

报告表：

| mode | git SHA | GPU | n | accuracy | 95% CI | invalid | avg latency | output tok/s |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | | | | | | | | |
| native | | | | | | | | |
| hybrid-low-pressure | | | | | | | | |
| hybrid-high-pressure | | | | | | | | |

必须同时测 low/high pressure。只测低压不会覆盖 host swap-in，只测高压不会证明 resident fast path。

## 7. 速度和显存验证

第一轮固定输入 30,720、输出 512；并发取 1/8/32/64，资源允许再加 128。每点 warmup 后重复 3 次。完成 MVP 后再扫描：

- `device_buffer_size`: 4096/8192；
- `host_to_device_ratio`: 2/5；
- watermark: 0.05/0.10/0.20；
- H20 与 H100 分开；
- baseline/native/hybrid 三模式同表比较。

示例 client：

```bash
python -m sglang.benchmark.serving \
  --backend sglang --base-url "http://127.0.0.1:${PD_PORT}" \
  --dataset-name random-ids \
  --random-input-len 30720 --random-output-len 512 \
  --num-prompts 64 --max-concurrency 32 --request-rate inf \
  --output-file "${RUN_DIR}/bench.jsonl"
```

外部采样至少保存 `nvidia-smi dmon` 或 DCGM CSV。服务内指标至少保存 watermark、free/resident/hot pages、mirror/reclaim 次数、top-k hit/miss、H2D bytes/time。

性能门槛分两层：

- 低压：hybrid 相对 baseline 的 ITL/吞吐退化不超过 5%，且 host H2D 接近 0；
- 高压：hybrid 完成率 100%、无 OOM，吞吐不低于 native，峰值 HBM 明显低于 baseline；
- 阶段 3 页级版本应优于阶段 2 请求级版本，否则不替换 MVP。

报告表：

| mode | GPU | input/output | concurrency | req/s | output tok/s | TTFT p50/p99 | ITL p50/p99 | HBM peak | host GB | H2D GB/s | resident/hot pages | reclaim p99 |
|---|---|---|---:|---:|---:|---|---|---:|---:|---:|---|---:|
| | | | | | | | | | | | | |

## 8. 每阶段产物与回滚条件

每个阶段提交一个目录：

```text
my_development/reports/<date>-<gpu>-<sha>/
  metadata.json
  commands.sh
  server-prefill.log
  server-decode.log
  accuracy-summary.json
  accuracy-raw.jsonl
  benchmark-summary.json
  benchmark-raw.jsonl
  gpu.csv
  hisparse-metrics.jsonl
  git.diff
  report.md
```

回滚/停止条件：出现 silent wrong answer、host durable 前释放 GPU 页、slot 泄漏、retract 后容量不恢复、p99 ITL 持续劣化超过门槛、或只能靠提高显存余量掩盖 OOM。出现这些情况时回到最近通过门禁的阶段，不继续叠加 CUDA Graph、radix 或新 eviction policy。

## 9. 推荐的下一步

当前分支已经处在“阶段 1 后半到阶段 2 前半”。下一步不是继续加新架构，而是：

1. 补齐状态转换和字节/页级指标；
2. 用 GPUQ 在 H20 或 H100 上跑 PD low-pressure/high-pressure smoke；
3. 产出 baseline/native/hybrid 的 GSM8K 200 和 30K×32 第一版报告；
4. 依据报告修复泄漏/竞态；
5. 请求级 MVP 稳定后再开始页级状态机。

这样最快能回答两个关键问题：DSV4 hybrid 是否正确，以及低压 resident fast path、高压 host fallback 是否分别带来预期收益。

## 10. 参考源码

- [SGLang PR #35488](https://github.com/sgl-project/sglang/pull/35488)
- [vLLM hybrid HiSparse coordinator（main）](https://github.com/vllm-project/vllm/blob/main/vllm/v1/hisparse/coordinator.py)
- [vLLM HiSparse layout（包含 DSV4 guard）](https://github.com/vllm-project/vllm/blob/main/vllm/v1/hisparse/layout.py)
- [vLLM HiSparse managers](https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/single_type_kv_cache_manager.py)

基础精度测试框架已经单独落在
[`dsv4_hybrid_hisparse_accuracy_testing.md`](./dsv4_hybrid_hisparse_accuracy_testing.md)，
当前先执行精度和 resident/evict 路径门禁，不包含高并发性能测试。
