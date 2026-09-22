# DeepSeek V4 Hybrid HiSparse 对齐 vLLM 的完整设计

## 1. 文档目的

本文给出从当前 SGLang DeepSeek V4 native HiSparse 与 lazy-unpin MVP，演进到 vLLM hybrid HiSparse 等价语义所需的完整设计。

目标不是照搬 vLLM 的通用 HMA/KV cache group 架构，而是在 SGLang 已有 DSV4 C4/C128/SWA 数据布局上对齐以下行为：

1. C4 KV 一产生便异步写入 host backing；
2. host 副本 durable 之前，GPU resident page 不能回收；
3. 无显存压力时，历史 C4 KV 保持 GPU resident，top-k 零拷贝直通；
4. 达到 watermark 后，为请求分配固定大小的 hot region，并逐页释放 clean resident page；
5. 已进入可回收队列的 page 在真正被复用前仍可读取；
6. page 真正被其他 owner 复用时，旧请求的 resident mapping 才失效；
7. top-k 可在同一批次、同一请求内混合命中 resident、hot 和 host；
8. 所有路径支持 CUDA Graph、TP 多 rank、请求结束/撤回和异步 stream ordering；
9. 每个开发阶段都有 CPU 单测、CUDA kernel 测试、准确度报告和性能报告。

本文基线：

| 代码 | 基线 |
|---|---|
| 当前工作分支 | `0591fe56e0663df7da63f133337636c2d06c8bd9` |
| SGLang upstream main | `80da4432d085ed4d6166ef643d9fd2b829dbb0c5` |
| SGLang PR #35488 本地 head | `d00fdbf42240b7f32ef35bd6b06a9de51d04a200` |
| vLLM 本地 main | `3ed531329aba156594aa49862f12faf03e5e3e5c` |

## 2. 关键事实与目标边界

### 2.1 vLLM 当前仍不支持 DSV4

vLLM 的 `_get_hisparse_hma_config()` 和 `_hisparse_gpu_memory_usage()` 对 `model_version == "deepseek_v4"` 明确报错。因此可参考的是：

- resident/hot/host 三层语义；
- scheduler 与 worker 分工；
- host publication 和异步 completion；
- watermark reclaim；
- fused resident pass-through + hot/host swap-in；
- shared GPU block pool；
- metrics 和 CUDA Graph 约束。

不能直接复制的是：

- vLLM 的 DSV3 MLA cache layout；
- vLLM 的 `KVCacheGroupSpec` 分组方式；
- 通用 `BlockPool`/prefix cache 结构；
- host block hash/shadow prefix 逻辑。

DSV4 必须继续使用 SGLang 已有的：

- C4 page-padded V4 layout；
- C128、SWA、indexer 独立池；
- `DeepSeekV4TokenToKVPool`；
- `HiSparseC4DevicePool`；
- DSV4 PD direct-to-host 传输路径。

### 2.2 “hot pages 逐步增加”的准确含义

vLLM 的 `HiSparseHotSpec.blocks_per_request` 是固定值。压力增大时，不是单个请求的 hot region 持续扩大，而是：

1. 更多请求从 resident-only 转成 hybrid；
2. 每个已转换请求获得固定大小 hot region；
3. 系统中的 hot pages 总量逐步增加；
4. 同时更多 clean resident pages 被释放。

SGLang DSV4 应保持同样语义。`device_buffer_size` 在请求生命周期内固定，不能动态改变，否则 LRU、CUDA Graph buffer shape 和 attention page table 都会变得不可控。

### 2.3 第一阶段不接 HiCache/radix cache

PR #35488 的核心价值是提供 protocol、backing factory、scheduler hook、pass-through swap-in 和 HiCache admission accounting。它的 HiCache backing 当前明确不覆盖 DSV4。

本设计第一阶段采用：

- private pinned host backing；
- decode 侧 radix cache 仍关闭；
- 只 hybrid 化 DSV4 C4；
- C128、SWA、indexer 保持现状。

HiCache/radix prefix reuse 是后续独立阶段，不能作为 MVP correctness 的前置条件。

### 2.4 当前 vLLM 与目标 lazy-unpin 语义的版本差异

截至本文使用的 vLLM commit `3ed531329a`：

- worker 把 D2H copy enqueue 后，scheduler 的 `_apply_enqueued_spill()` 可以释放 resident block；
- `HiSparseResidentManager.release_resident_page()` 会把 active request 的 resident table entry 立即替换为 null block；
- `BlockPool` 中 ref-count 为零的 block 在真正复用前物理 bytes 仍可能存在，但 active request 已不再通过该 entry 读取；
- 另有 shadow-page 机制为已发布 host prefix 保留/重新采用 GPU copy。

如果后续要求与某个指定 vLLM 历史 commit bit-for-bit 对齐，应先固定该 commit，再决定是否把 `CLEAN_RECLAIMABLE` 立即改成 `HOST_ONLY`。当前设计不做这一退化。

## 3. 当前实现评估

### 3.1 已经具备的能力

当前分支已经实现：

- `ReclaimablePagedTokenToKVPoolAllocator`；
- unpin 后 mapping 保持，allocation-time reuse callback 清 mapping；
- `_HybridPageStatus` 页状态；
- prefill 完成后的异步全请求 host mirror；
- decode 新 C4 token 的异步 backup；
- watermark 触发 hot region 分配和 clean page unpin；
- resident/host mixed top-k correctness 路径；
- allocator、policy、slot translation CPU 单测。

本地现有测试结果：

```text
23 passed, 3 subtests passed
```

### 3.2 当前实现离 vLLM 等价语义的主要差距

| 编号 | 差距 | 风险 |
|---|---|---|
| G1 | resident、hot、host 状态仍集中在一个超大 coordinator | 生命周期难证明，finish/retract/异常路径容易泄漏 |
| G2 | prefill 结束后一次性复制整个请求，而不是 sealed page 增量 publication | 长 prefill 的峰值复制和 transition 延迟较大 |
| G3 | decode backup 在下一轮 `prepare` 中备份 previous token | 与“产生即写回”相差一个调度 step |
| G4 | 一个全局 `_backup_done_event` 和 request-level mirror event 粒度较粗 | 多批次、多 page、多 stream completion 难精确归属 |
| G5 | Python 先解析 resident，再用 `torch.any/where` 调 host swap-in | CUDA Graph 不稳，多 launch、多显存读写、潜在 CPU sync |
| G6 | page owner 保存散列 `logical_locs` 集合，reuse 时临时构造 tensor | scheduler CPU 开销随上下文增大，难批处理 |
| G7 | hot region 与 resident pool 虽共用 allocator，但没有正式的 capacity transaction | 异步 completion 或并发 admission 下可能超配 |
| G8 | reclaim 以 request FIFO 为主，一次 unpin 请求内所有 eligible pages | 无法只回收 shortage，回收过量，hot 成本收益不够精确 |
| G9 | active-forward source-page 安全主要依赖 protected request set | 缺少明确 `after_forward` transfer/reuse fence |
| G10 | TP readiness 用 scheduler-thread CPU all-reduce | 容易引入同步停顿，且缺少 per-transfer completion 汇总 |
| G11 | 没有稳定的 page/transfer ID 和统一 trace/metrics | 准确度正确但无法证明命中了目标路径 |
| G12 | DSV4 pool sizing 未把 resident、hot reserve、host staging 统一建模 | H20/H100 上可能表现出不同的隐式 OOM 边界 |

### 3.3 一页结论：必须改的类与算子

必须修改或新增的 P0 类：

| 类/模块 | 动作 | 核心原因 |
|---|---|---|
| `mem_cache.sparsity.factory` | 修改 | 统一 resolved config 和 coordinator factory |
| `HiSparseCoordinator` protocol | 从 PR #35488 引入并扩展 | scheduler 不再绑定具体 backing |
| `DeepSeekV4HybridHiSparseCoordinator` | 新增 | 从 native coordinator 拆出 hybrid 状态机 |
| `HybridC4PageManager` | 新增 | 管 resident page、host durability、mapping 和 generation |
| `HybridC4PageAllocator` | 从 reclaimable allocator 收敛 | shared resident/hot pool 和 allocation-time reuse transaction |
| `DeepSeekV4HiSparseTokenToKVPoolAllocator` | 修改 | 接入精确 reclaim、hot reservation、回滚 |
| `HiSparseC4DevicePool` | 修改 | 提供 request-local resident table 和 kernel 静态参数 |
| `DeepSeekV4PagedHostPool` | 修改 | page owner、transfer ref、durable publication |
| `DSV4HiSparseTransferEngine` | 新增 | 独立管理 copy stream、event、transfer completion |
| `DSV4PoolConfigurator` | 修改 | 统一 resident/hot/watermark/growth reserve 容量模型 |
| scheduler/batch/PD decode | 修改 | lifecycle、capacity、finish-forward mirror |
| `DeepseekSparseAttnBackend` | 修改 | 图内调用 fused mixed-tier swap-in |
| HiSparse metrics/stats | 新增或扩展 | 产出路径覆盖和性能证据 |

算子结论：

| 优先级 | 算子 | 决策 |
|---|---|---|
| P0 | fused DSV4 resident pass-through + hot/host swap-in | 必须写，是最大功能/性能缺口 |
| P0 | DSV4 全层 page backup wrapper | 必须正式化；优先复用现有 kernel，不盲目重写 |
| P0 | resident page mapping invalidate | 先用连续 slice；批量开销证明确有必要才写 kernel |
| P1 | all-layer compact backup plan | launch/copy profiling后决定 |
| P1 | store-and-mirror C4 | D2H 成为瓶颈后再写 |
| P1 | fused tier metrics | 建议随 fused swap-in 实现 |
| 不需要 | C128/SWA、top-k indexer、shared-index copy | 复用现有实现 |

## 4. 目标架构

### 4.1 组件关系

```text
Scheduler / DecodePreallocQueue
        |
        | lifecycle + capacity requests
        v
HiSparseCoordinator protocol                 PR #35488 可复用接口思想
        |
        v
DeepSeekV4HybridHiSparseCoordinator          DSV4 控制面
        |                    |
        |                    +--> DSV4HiSparseTransferEngine
        |                         stream/event/transfer completion
        v
HybridC4PageManager
        |
        +--> HybridC4PageAllocator            shared resident/hot physical pool
        +--> HiSparseC4DevicePool             C4 physical storage
        +--> DeepSeekV4PagedHostPool          durable host backing
        +--> req_to_c4_resident_locs          per-request resident page table

DSA backend
        |
        v
fused_dsv4_hybrid_swap_in
 resident pass-through -> hot hit -> host miss copy
```

### 4.2 控制面与数据面分离

控制面负责：

- 请求/page 状态；
- host slot 分配；
- watermark 和 admission；
- reclaim candidate 选择；
- transfer plan；
- completion publication；
- finish/retract；
- metrics。

数据面负责：

- C4 全层 D2H page copy；
- resident/hot/host top-k 解析；
- host miss H2D copy；
- shared-index follower replay；
- mapping invalidate；
- CUDA events 和 copy stream。

控制面不能在每层 attention 内进行 `.cpu()`、`.tolist()`、Python 条件分支或动态 allocation。

## 5. 核心状态模型

### 5.1 Request 状态

```text
PREFILLING
   |
   | first sealed C4 page
   v
RESIDENT_MIRRORING
   |
   | host publications may complete incrementally
   v
RESIDENT_ONLY
   |
   | watermark + profitable transition
   v
TRANSITION_PENDING
   |  allocate fixed hot region
   |  mark selected clean pages reclaimable
   v
HYBRID
   |
   | all old pages eventually reused
   v
HOST_BACKED

任意状态 --finish/retract--> RELEASING --> DEAD
```

`HOST_BACKED` 不表示 GPU 中没有任何历史：hot region、active writable tail、尚未被复用的 reclaimable pages 都可能仍在。

### 5.2 Page 状态

```text
ALLOCATED_DIRTY
    |
    | enqueue D2H
    v
MIRROR_QUEUED
    |
    | completion event
    v
CLEAN_PINNED
    |
    | watermark reclaim
    v
CLEAN_RECLAIMABLE
    |
    | allocator selects for another owner
    v
REUSED / HOST_ONLY
```

请求结束时：

- `ALLOCATED_DIRTY`：等待或取消尚未读取 source 的 transfer；
- `MIRROR_QUEUED`：等待 enqueue fence，completion 可由 orphan transfer 回收；
- `CLEAN_PINNED`：正常 free；
- `CLEAN_RECLAIMABLE`：从旧 owner detach，不能再次入 free queue；
- hot pages：正常 free；
- host pages：最后释放。

### 5.3 必须长期成立的不变量

1. 每个有效 C4 logical position 至少有一个 home：resident page 或 durable host row；
2. `ALLOCATED_DIRTY/MIRROR_QUEUED` page 不可 reclaim；
3. `CLEAN_RECLAIMABLE` page 可读，直到 allocator 发出 reuse；
4. reuse callback 必须先使旧 resident mapping 不可见，再把 page 返回新 owner；
5. hot region ready 之前，不能让请求失去任何 resident page；
6. writable tail 和前一 page 永不参与当前 step reclaim；
7. transfer source 在 copy 已 enqueue 到正确 stream 前不可复用；
8. host hash/publication（未来接 prefix cache）只能发生在 host copy completion 后；
9. TP ranks 对同一 logical page 的状态转换顺序一致；
10. request slot 重用依赖 generation，旧 event/callback 不能修改新请求。

## 6. 需要修改或新增的重要类

### 6.0 `mem_cache.sparsity.factory` 与配置解析

文件：`python/sglang/srt/mem_cache/sparsity/factory.py`

参考 PR #35488 引入 backing/factory seam，但第一阶段只解析：

```text
PRIVATE_HOST_NATIVE
PRIVATE_HOST_DSV4_HYBRID
```

职责：

- 根据模型类型和 `hybrid_mode` 创建正确 coordinator；
- 统一解析 `top_k/device_buffer_size/watermark/backup_mode`；
- 拒绝 DSV4 hybrid + decode radix cache 等未支持组合；
- 保证 pool configurator 与 coordinator 使用同一份 resolved config；
- 将实验开关集中在 config，不散落环境变量。

验证：纯 CPU 参数矩阵测试，非法组合在启动期失败。

### 6.1 `HiSparseCoordinator` protocol

文件：`python/sglang/srt/managers/hisparse_protocol.py`

从 PR #35488 提取并适配，而不是继续让 scheduler 依赖具体 coordinator。保留以下统一接口：

```python
class HiSparseCoordinator(Protocol):
    num_real_reqs: torch.Tensor

    def on_prefill_complete(self, req: Req) -> bool: ...
    def on_prefill_finished_early(self, req: Req) -> None: ...
    def prepare_decode_batch(...): ...
    def finish_forward(self) -> None: ...
    def collect_ready_reqs(self) -> list[Req]: ...
    def has_ongoing_staging(self) -> bool: ...
    def request_finished(self, req: Req) -> None: ...
    def retract_req(self, req: Req) -> None: ...
    def swap_in_selected_pages(...): ...
    def get_token_stats(self) -> HiSparseTokenStats: ...
    def destroy(self) -> None: ...
```

新增 `finish_forward()` 是 DSV4 对齐 vLLM 的关键：新产生的 C4 KV 应在本次 forward 结束时立刻进入 backup stream，而不是下一调度 step 才处理。

验证：

- protocol conformance CPU test；
- native coordinator 行为不变；
- scheduler 不再 `isinstance` 分支调用生命周期方法。

### 6.2 `DeepSeekV4HybridHiSparseCoordinator`

文件建议：`python/sglang/srt/managers/hisparse_dsv4_hybrid_coordinator.py`

从现有 `HiSparseCoordinator` 拆出 hybrid 专属控制面。职责：

- 请求状态与 generation；
- 调用 page manager 建立 resident mapping；
- 调用 transfer engine mirror sealed pages；
- watermark 检查和 transition；
- 为请求申请固定 hot region；
- 构造 fused swap-in 所需静态 tensor；
- 汇总 TP completion；
- 请求结束/撤回。

不再直接保存每页 `set[int] logical_locs`，而保存连续的 request-local compressed page index。

核心方法：

```python
def publish_new_c4_rows(req_indices, compressed_positions, physical_slots): ...
def enqueue_sealed_pages(after_forward: bool): ...
def poll_transfer_completions(): ...
def reclaim_c4_blocks(pool_id: int, shortage: int) -> ReclaimResult: ...
def ensure_hot_region(req_id: int) -> bool: ...
def build_swap_metadata(forward_batch): ...
```

验证：

- randomized state-machine model test；
- request slot generation reuse；
- finish/retract at every page state；
- mixed resident/host requests in one batch；
- TP rank completion count mismatch 必须 fail closed。

### 6.3 `HybridC4PageManager`（新增）

文件建议：`python/sglang/srt/mem_cache/hybrid_c4_page_manager.py`

对应 vLLM `HiSparseResidentManager` + `HiSparseHotManager` 的 DSV4 专用等价物。它不拥有 CUDA stream，只管理页和映射。

数据结构：

```python
@dataclass
class C4PageRecord:
    request_slot: int
    generation: int
    request_page_idx: int
    physical_page_id: int
    host_page_id: int
    state: C4PageState
    transfer_id: int | None

req_to_c4_resident_locs: Tensor  # [max_reqs, max_compressed_len], int32
req_to_host_locs: Tensor         # [max_reqs, max_compressed_len], int64
```

`req_to_c4_resident_locs` 的语义：

- `>= 0`：resident physical slot，可直接给 attention；
- `-1`：不再 resident，走 hot/host；
- padding 行保持 `-1`。

这样 reuse 时可按 `(request_slot, request_page_idx)` 直接清一个连续 slice，不需要保存 logical loc set，也不需要临时 CPU tensor。

核心方法：

```python
def bind_allocated_page(...): ...
def mark_transfer_queued(transfer_id, pages): ...
def mark_transfer_complete(transfer_id): ...
def select_reclaimable_pages(shortage, protected): ...
def mark_reclaimable(page): ...
def on_page_reused(page_id, old_owner): ...
def detach_request(req_slot, generation): ...
```

验证：

- reference Python model 与 manager 随机操作逐步一致；
- page table 和 allocator owner 双向一致；
- 重复 completion、旧 generation callback、double finish 均幂等。

### 6.4 `HybridC4PageAllocator`

当前：`ReclaimablePagedTokenToKVPoolAllocator`

建议将其收敛成 DSV4 专用 `HybridC4PageAllocator`，或至少增加正式 contract：

```python
def allocate_pages(num_pages, new_owner) -> AllocationResult: ...
def release_pinned_pages(page_ids): ...
def make_reclaimable(page_ids, owners): ...
def detach_reclaimable(page_ids, owners): ...
def num_immediately_free_pages() -> int: ...
def num_reclaimable_pages() -> int: ...
```

`AllocationResult` 必须返回：

- 新分配 page ids；
- 本次实际复用的旧 owner；
- mapping invalidation 完成标志。

不要让 allocator 通过任意 Python callback 修改 coordinator。更稳妥的方式是 allocator 返回 reuse records，由 page manager 在 allocation transaction 内同步应用：

```text
pop free/reclaimable pages
  -> collect old owners
  -> invalidate old mappings
  -> install new owners
  -> allocation visible to caller
```

这让异常回滚、测试和锁顺序更明确。

验证：

- sorted/unsorted allocator；
- free、reclaimable、allocated 集合互斥；
- unpin 后数据未被覆盖；
- reuse 前 mapping 有效，reuse transaction 后无效；
- free group begin/end 与 reclaimable 页兼容。

### 6.5 `DeepSeekV4HiSparseTokenToKVPoolAllocator`

文件：`python/sglang/srt/mem_cache/allocator/hisparse.py`

需要修改：

1. 将 C4 resident 和 hot region 明确放入同一 physical allocator；
2. logical full/SWA allocator 与 C4 physical allocator 的 capacity 独立计算；
3. `_ensure_c4_pages()` 返回结构化结果，不再只返回 bool；
4. allocation 前调用 coordinator reclaim plan；
5. hot region allocation 作为一个原子 reservation；
6. expose `immediate_free + reclaimable`，但调度容量检查要区分二者；
7. clear/free_group/retract 全路径传播到 page manager。

禁止：

- allocator 在 forward 中隐式同步 host copy；
- allocation failure 后临时整请求 spill；
- 把 writable page 放入 reclaimable queue。

### 6.6 `HiSparseC4DevicePool`

文件：`python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py`

需要修改：

- 注册 request-local resident table，而不只注册 global logical mapping；
- 暴露 page geometry 和全层 pointer table；
- 提供 resident page view/hot page view；
- 支持 fused hybrid swap-in 的静态参数；
- 增加 debug owner generation tensor（仅 debug build 或环境变量开启）；
- C4 store 完成后通知 transfer engine 本 step 新产生哪些 row。

原 `full_to_hisparse_device_index_mapping` 在迁移期保留，用于 native HiSparse 和 debug cross-check；hybrid 稳定后，attention hot path 只读 request-local table。

### 6.7 `DeepSeekV4PagedHostPool`

文件：`python/sglang/srt/mem_cache/memory_pool_host.py`

需要修改：

- page-granular host allocation；
- host page owner/generation；
- transfer_id 到 host page 的引用计数；
- completion 前 host page 不可发布或复用；
- 支持 orphan completion：请求结束后 copy 仍可安全完成再释放；
- 暴露连续 page-first/layer-first geometry 给 kernel；
- host pool OOM 时返回 admission failure，而不是 assertion。

第一阶段继续使用 private pinned host pool；不接 HiCache hash/prefix publication。

### 6.8 `DSV4HiSparseTransferEngine`（新增）

文件建议：`python/sglang/srt/managers/hisparse_dsv4_transfer.py`

对应 vLLM worker connector 的 DSV4 精简版。职责：

- 独占 backup stream；
- staging buffers 和 pointer tables；
- enqueue page transfers；
- 管理 `transfer_id -> CUDA event`；
- 区分 `after_forward=False/True`；
- poll enqueued/completed；
- 统计 bytes、pages、queue latency、copy latency；
- shutdown 时 drain。

接口：

```python
def start_step(metadata): ...
def enqueue(transfers, *, after_forward): ...
def finish_forward(): ...
def take_updates() -> TransferUpdates: ...
def shutdown(): ...
```

stream ordering：

```text
compute stream writes C4
    -> producer event
backup stream waits producer event
    -> enqueue D2H kernel
    -> enqueue event: source page now stream-ordered and may enter reclaim path
    -> completion event: host page becomes durable/publishable
```

如果 allocator 复用不具备 CUDA stream-aware free 语义，则必须等 enqueue event 已被 allocator stream wait，而不是仅 CPU 知道 kernel 已 launch。

### 6.9 `DSV4PoolConfigurator`

文件：`python/sglang/srt/model_executor/pool_configurator.py`

需要统一建模：

```text
N_c4_blocks       shared C4 physical blocks
B_hot             ceil((device_buffer_size + tail_page) / c4_page_size)
R_grow             active writable tail/growth reserve
W                  max(configured_ratio * N_c4_blocks,
                       B_hot + R_grow)
H_host_pages       host pool capacity
```

请求从 resident-only 转 hybrid 必须满足：

```text
eligible_clean_pages - B_hot > 0
```

否则转换只会增加显存占用，应继续保持 resident 或直接按 native host-backed admission。

配置输出必须打印：

- C4 page bytes/page size；
- resident/hot shared block count；
- per-request hot blocks；
- watermark blocks；
- growth reserve；
- host capacity；
- 理论最大 hybrid requests。

验证：H20 141G 与 H100 80G 对同一 CLI 得到可解释且不越界的 pool report。

### 6.10 Scheduler、batch 和 PD decode

重要文件：

- `scheduler.py`；
- `schedule_batch.py`；
- `batch_result_processor.py`；
- `disaggregation/decode.py`；
- `model_runner.py`。

需要修改：

1. prefill complete 统一调用 `on_prefill_complete()`；
2. 每轮调度前 poll transfer completion；
3. capacity check 在普通 eviction 前先调用 C4 reclaim；
4. transition 需要的 hot blocks 纳入本轮 allocation budget；
5. forward metadata 携带 request state index/generation；
6. forward 完成后调用 `finish_forward()`，立即备份新 C4 rows；
7. retract/finish 统一进入 coordinator，不允许 scheduler 自行 free 一半状态；
8. PD direct-to-host import 后，host publication 与 resident tail 建立必须原子完成；
9. decode prealloc 保留 DSV4 SWA tail，不把 C128/SWA 错算为 C4 可回收量。

### 6.11 `DeepseekSparseAttnBackend`

文件：`python/sglang/srt/layers/attention/dsa_backend.py`

需要修改：

- graph-static buffer 中增加 `req_to_c4_resident_locs`/request state indices；
- anchor layer 调 fused hybrid swap-in 并记录 compact miss plan；
- shared-index follower layer 复用 plan；
- padded batch rows由 `num_real_reqs` 屏蔽；
- 不允许 attention 层出现 Python `torch.any()` 分支；
- resident-only 批次也走同一 graph-compatible kernel，可由 kernel fast path 零 copy 返回。

## 7. 必须实现或改造的重要算子

### 7.1 P0：fused DSV4 hybrid swap-in

这是对齐 vLLM 最重要的新算子。

建议在现有 `load_cache_to_device_buffer_kernel` 增加编译期 flag：

```cpp
PassThroughResidentLocs
```

Python API：

```python
load_cache_to_device_buffer_dsv4_hybrid_mla(
    top_k_tokens,                 # [B, K], compressed request-local position
    resident_locs,                # [max_reqs, max_compressed_len], int32
    device_buffer_tokens,         # hot cache tag
    host_cache_locs,
    device_buffer_locs,
    host_cache,
    device_buffer,
    top_k_device_locs,
    req_pool_indices,
    seq_lens,
    lru_slots,
    num_real_reqs,
    miss_src=None,
    miss_dst=None,
    miss_count=None,
    stats=None,
)
```

每个 top-k lane：

```text
invalid/padding     -> -1
resident_locs >= 0 -> 直接返回 resident physical slot
hot tag hit         -> 返回 hot slot，并更新 LRU
otherwise           -> 分配 LRU victim，host -> hot，返回 hot slot
```

必须保证：

- resident lane 不进入 hot hash，不占 hot slot；
- host miss count 不包含 resident/padding；
- DSV4 page-padded value/scale layout copy 正确；
- `RecordMissPlan` 与 pass-through 同时工作；
- resident-only、mixed、host-only 使用同一 kernel；
- CUDA Graph replay 时所有 shape 和 pointer 稳定。

这是 PR #35488 `PassThroughDeviceLocs` 的 DSV4 版本。当前 Python `_resolve_resident_topk + torch.where + second kernel` 只作为 oracle，算子通过后删除热路径分支。

验证：

- K 中 resident/hot/host/padding 任意混排；
- duplicate top-k；
- seq_len 小于/等于/大于 hot size；
- page 边界；
- K=2048/3072；
- CUDA Graph capture/replay 多 batch size；
- 与 Python oracle bitwise 比较 slot table 和 KV bytes；
- compute-sanitizer 无越界/race。

### 7.2 P0：DSV4 全层 page backup

当前 `transfer_cache_dsv4_mla` 和整页 `transfer_kv_all_layer_mla` 已提供大部分能力，优先复用，不应立即重写。

需要补齐的正式 wrapper：

```python
backup_dsv4_c4_pages_all_layers(
    src_layer_ptrs,
    dst_layer_ptrs,
    src_page_ids,
    dst_page_ids,
    num_pages,
)
```

要求：

- 一次 launch 覆盖所有 DSA layers；
- 输入是 page id，不在 Python 展开为 token loc；
- 处理 V4 value/scale page padding；
- 可在独立 stream 调用；
- 空 transfer no-op；
- H20/H100 都有合理 occupancy。

若现有整页 kernel 性能达到要求，只新增 wrapper、event 和测试，不新增 CUDA kernel。

验证：随机 page、随机 layer、非连续 src/dst；D2H 后逐字节比较；记录 GB/s。

### 7.3 P0：resident mapping page invalidate

采用 request-local 连续 resident table 后，大多数情况无需新 CUDA 算子：

```python
resident_locs[req_slot, page_start:page_end].fill_(-1)
```

如果一次 allocation 会复用大量 page，则新增批处理 Triton/CUDA op：

```python
invalidate_dsv4_resident_pages(
    resident_locs,
    req_slots,
    request_page_indices,
    generations,
    active_generations,
    page_size,
)
```

P0 可先用 torch slice；只有 profile 显示 scheduler launch 开销明显时才写 kernel。

### 7.4 P1：store-and-mirror C4

当前路径是：先写 C4 device page，再由 backup kernel重新读取并写 host。完整对齐可增加 dual-destination store：

```python
store_dsv4_c4_and_mirror(
    produced_c4,
    resident_slots,
    host_slots,
    resident_pool,
    host_pool,
)
```

收益：

- 少一次 GPU global-memory read；
- host mirror 与 C4 store 天然同源；
- publication ordering 更简单。

但它会侵入 DSV4 C4 生成路径，风险高。必须在 P0 correctness 和 benchmark 后决定；若 D2H copy 不在瓶颈，不写该算子。

### 7.5 P1：all-layer compact backup plan

借鉴 vLLM `hisparse_backup_layers`，把多 page、多 layer transfer plan 一次提交：

```python
backup_dsv4_c4_pages_plan(
    layer_ptrs,
    host_ptrs,
    src_page_ids,
    dst_page_ids,
    num_transfers,
)
```

仅当现有 page backup 每 step launch 数过多时需要。目标是一个 batch 一个 launch，而不是一个 page 一个 launch。

### 7.6 P1：fused metrics

在 swap-in kernel 中累加：

- resident hits；
- hot hits；
- host misses；
- H2D rows/bytes。

写 device counter，低频异步复制到 CPU。禁止每 step `.item()`。

### 7.7 不需要为对齐而新写的算子

- C128/SWA copy：沿用现有 DSV4；
- top-k indexer：沿用现有 DSA backend；
- shared-index follower copy：沿用 `copy_cache_planned_mla`；
- generic HiCache pass-through：DSV4 第一阶段不接 HiCache；
- host allocator：属于控制面，不写 CUDA kernel。

## 8. Watermark 与回收算法

### 8.1 容量定义

```text
free_now       尚未被任何 owner 引用的 page
reclaimable    clean、已入 allocator free order、旧 owner 仍可读的 page
pinned_clean   host durable 但尚未 unpin
dirty          host 不 durable
hot            固定 hot region
protected      writable tail、previous page、当前 forward source
```

调度 admission 可把 `free_now + reclaimable` 当作 eventual capacity，但真正 allocation transaction 必须先处理 reuse/invalidate。

### 8.2 触发条件

```text
W = max(
    ceil(total_c4_blocks * hybrid_reclaim_watermark),
    hot_blocks_per_request + growth_reserve_blocks,
)

pressure = free_now + reclaimable < required_blocks + W
```

### 8.3 请求选择

优先顺序：

1. 已有 hot region 且 page 已 host-valid 的请求；
2. 可释放 page 数最多的 resident-only 请求；
3. 较老请求；
4. 当前 batch/protected 请求最后或跳过。

resident-only 请求只有满足以下条件才转换：

```text
releasable_pages > hot_blocks_needed
```

转换步骤必须是一个 scheduler transaction：

1. 计算精确 shortage；
2. 预留 hot blocks；
3. 标记最少数量 clean pages reclaimable；
4. allocation transaction 复用必要页并清旧 mapping；
5. 安装 hot region；
6. 发布 request 状态为 HYBRID。

不能先丢 resident mapping，再尝试 hot allocation。

### 8.4 回收量

只标记满足 shortage 的最少 page 数，不一次 unpin 整个请求：

```text
needed = required + watermark - (free_now + reclaimable)
transition_cost = hot_blocks_needed
pages_to_release = transition_cost + needed
```

已有 hot region 的请求没有 `transition_cost`。

## 9. 异步写回与 publication

### 9.1 Prefill

正确性优先实现：

1. prefill chunk 结束后识别 newly sealed C4 pages；
2. enqueue D2H page plan；
3. 请求可以继续 prefill；
4. completion 后 page 变 `CLEAN_PINNED`；
5. 最后一个 partial page 保持 dirty/pinned；
6. prefill complete 时只补齐尾部 sealed page，而不是复制整个历史。

第一提交允许 prefill 完成后一次性 mirror 作为兼容 fallback，但最终目标必须是 chunk/page 增量。

### 9.2 Decode

DSV4 C4 每 4 个 full tokens 产生一个 compressed row。产生该 row 的 forward 完成时：

1. C4 store 已写 resident slot；
2. `finish_forward()` 将 row/page 加入 transfer plan；
3. backup stream 等 producer event；
4. copy enqueue 后 source page 可参与未来 reclaim，但 host 尚不可读；
5. completion event 后 host row durable；
6. sealed page 的所有 row durable 后，page 变 clean。

这替换当前“下一 step 备份 previous token”的一拍延迟路径。

### 9.3 Enqueued 与 completed 的区别

- `enqueued`：GPU stream 已建立先读 source、后允许 reuse 的依赖；
- `completed`：CPU host bytes 已写完，可作为 host source/prefix publication。

若 allocator 不能跨 stream 保证 reuse 顺序，则 page 只能在 `completed` 后 reclaim。先做保守版本，再通过事件/stream-aware allocator 开启 enqueue-time reclaim。

## 10. CUDA Graph 与 TP 设计

### 10.1 CUDA Graph

图内只允许：

- 静态 shape tensor；
- device scalar `num_real_reqs`；
- request state indirection；
- fused hybrid swap-in；
- graph-static miss plan buffers。

图外处理：

- host allocation；
- transfer plan 构造；
- event poll；
- watermark；
- reclaim；
- finish/retract。

禁止图内：

- `.item()`；
- `torch.any()` 控制 Python 分支；
- 创建 tensor；
- CPU list/to-list；
- 动态 host pointer 变化。

private host pool 在 init 时绑定，暂不需要 PR #35488 的 late-bound host base。未来接 HiCache 时再启用。

### 10.2 TP

每个 transfer_id 记录 expected completions：

```text
expected = 参与该 C4 layer shard 的 worker/rank 数
completed == expected 才 publication
```

控制面 transition decision 由 scheduler rank 决定并广播 page plan；各 TP rank 不独立挑 victim。避免每页 CPU all-reduce readiness。

验证：

- TP1/TP2/TP4 相同 request/page transition trace；
- 任一 rank completion 延迟时不得提前 reclaim；
- rank error 时 fail request，不允许静默使用半写 host page。

## 11. 配置与可观测性

建议配置：

```json
{
  "hybrid_mode": true,
  "top_k": 2048,
  "device_buffer_size": 4096,
  "host_to_device_ratio": 2,
  "hybrid_reclaim_watermark": 0.1,
  "hybrid_backup_mode": "page_async",
  "hybrid_reclaim_after_enqueue": false
}
```

`hybrid_reclaim_after_enqueue=false` 是第一阶段安全默认；验证 stream-aware reuse 后再开启。

必须导出的 counters/gauges/histograms：

```text
hisparse_c4_free_pages
hisparse_c4_reclaimable_pages
hisparse_c4_pinned_clean_pages
hisparse_c4_dirty_pages
hisparse_hot_pages
hisparse_resident_hits
hisparse_hot_hits
hisparse_host_misses
hisparse_d2h_pages / bytes
hisparse_h2d_rows / bytes
hisparse_transfer_queue_us
hisparse_transfer_copy_us
hisparse_reclaim_pages
hisparse_page_reuse
hisparse_transition_count
hisparse_transition_us
hisparse_host_pool_usage
hisparse_stale_generation_callbacks
```

每次测试保存 event trace：

```text
timestamp, rank, req_generation, request_page, physical_page,
event, transfer_id, stream, old_state, new_state
```

## 12. 分阶段开发计划与验证门槛

### Phase 0：冻结基线与测试框架

改动：

- 固定 baseline/native/current-MVP 启动命令；
- 保存 commit、模型 config、driver、GPU、依赖；
- 建立统一 report 目录和 JSON schema；
- 将现有 23 个 CPU tests 纳入脚本。

验证：

- 本地 CPU：23 tests 全绿；
- GPU：native DSV4 HiSparse GSM8K 可复现；
- 输出 baseline/native token hash、任务准确率、吞吐。

退出条件：基线报告可由单条 gpuq 命令重复生成。

### Phase 1：类拆分，不改数据面

改动：

- 引入 coordinator protocol；
- 新建 DSV4 hybrid coordinator/page manager/transfer engine；
- scheduler 生命周期走 protocol；
- 保持现有 copy 和 Python mixed top-k。

验证：

- 全部原单测；
- 新增 protocol/state-machine/finish-retract tests；
- GPU 与 current MVP greedy token 逐条一致；
- 性能只记录，不设提升门槛，但回退不得超过 5%。

### Phase 2：精确 page publication 与异步 transfer

改动：

- prefill sealed page 增量 mirror；
- decode `finish_forward()` 当步 mirror；
- transfer_id、enqueued/completed；
- page 粒度 host durable bitmap；
- 保守地 completion 后才 reclaim。

验证：

- 注入延迟 event，dirty/pending page 永不 reclaim；
- finish/retract 与 in-flight transfer；
- D2H 内容逐字节对比；
- 长 prefill 下 host backup 与 compute timeline 可重叠；
- native/current-MVP/Phase2 输出一致。

### Phase 3：shared pool transaction 与精确 watermark

改动：

- `HybridC4PageAllocator` transaction；
- request-local resident table；
- 最少 page reclaim；
- fixed hot reservation；
- scheduler capacity accounting。

验证：

- 随机 allocator/page-manager model test 10k+ operations；
- watermark 前零 transition；
- watermark 后回收量不低于 shortage，过量不超过一个 page；
- unpinned-not-reused 仍 resident hit；
- reuse 后同一 top-k 变 host/hot path；
- 连续压力无 OOM、double free、stale mapping。

### Phase 4：fused DSV4 hybrid swap-in

改动：

- 实现 resident pass-through flag；
- 删除 attention 热路径 Python split；
- shared-index plan 兼容；
- fused device counters。

验证：

- kernel oracle 全组合；
- compute-sanitizer；
- CUDA Graph capture/replay；
- K=2048/3072；
- resident-only kernel 开销相对直接 gather 小于 3%；
- mixed 模式 launch 数和临时 tensor 数明显下降。

### Phase 5：enqueue-time reclaim 与性能收敛

改动：

- allocator stream-aware reuse；
- `hybrid_reclaim_after_enqueue=true`；
- backup plan batch/fusion；
- 必要时 store-and-mirror。

验证：

- 高并发下无 use-after-free；
- Nsight 证明 copy/compute overlap；
- backup stream 不造成主 stream 全局同步；
- H20/H100 分别报告。

### Phase 6：可选 HiCache/radix backing

改动：

- 复用 PR #35488 backing factory/protocol；
- DSV4 expanded indexer page table；
- tree eviction hooks；
- admission ledger；
- late-bound host binding。

该阶段不应与 Phase 1–5 混在同一个 PR。

## 13. 准确度验证与报告

### 13.1 模式矩阵

| 模式 | 目的 |
|---|---|
| baseline | 不启用 HiSparse，任务准确率参考 |
| native | 当前 DSV4 native HiSparse，语义 oracle |
| hybrid-resident | 显存充足，无 page reuse |
| hybrid-unpinned | 已 unpin、尚未 reuse |
| hybrid-mixed | 同一请求 top-k 同时含 resident/host |
| hybrid-host | 历史 page 大量复用，主要走 host/hot |

### 13.2 数据集与参数

- 模型：`~/models/DeepSeek-V4-Flash-0731`；
- 数据集：`~/datasets/gsm8k`；
- greedy：temperature=0；
- 固定 prompt 顺序和 seed；
- 先 20 条 smoke，再完整 GSM8K；
- PD 分离拓扑；
- H20 与 H100 分别记录。

### 13.3 正确性门槛

主要比较 native vs hybrid：

- greedy token 序列逐请求一致率 100%；
- 首个不一致 token 必须输出 request/page/top-k tier trace；
- GSM8K exact match 不低于 native 超过 1 个样本；
- resident、unpinned、mixed、host 四条路径都有 counter 证据；
- 无 NaN、CUDA error、timeout、host OOM。

baseline vs native/hybrid 用于任务质量参考，不要求 token bitwise 一致。

### 13.4 准确度报告字段

```text
commit / mode / GPU / TP / DP / PD topology
model config hash / dataset hash / seed
num samples / exact match
native-vs-hybrid exact token match rate
first mismatch request/token/layer
resident/hot/host hit counts
page transitions/reuses
errors/timeouts
```

## 14. 性能验证与报告

### 14.1 工作负载

至少覆盖：

1. 低压：短上下文、低并发，全部 resident；
2. 临界：free pages 在 watermark 上下摆动；
3. 高压：长上下文、高并发，大量 mixed/host；
4. burst admission：同时完成多个 prefill；
5. 长 decode：验证持续 C4 mirror；
6. shared-index 模型层模式；
7. H20 141G 与 H100 80G。

### 14.2 指标

```text
request throughput
output tok/s
TTFT p50/p95/p99
TPOT/ITL p50/p95/p99
decode batch size
HBM peak/average
host pinned GB
D2H/H2D bytes and effective GB/s
resident/hot/host hit ratio
transition p50/p99
reclaim p50/p99
CUDA kernel time and launch count
copy/compute overlap ratio
CPU scheduler time
```

### 14.3 阶段性性能门槛

- hybrid-resident 相对 baseline/native decode TPOT 回退不超过 5%；
- fused swap-in 后 resident-only 附加开销目标小于 3%；
- hybrid-pressure 吞吐不低于 native HiSparse 的 95%；
- 不允许出现 scheduler-thread device synchronize；
- P99 transition 不造成超过一个 decode step 的全局停顿；
- 同负载峰值 HBM 不超过配置预算；
- host copy bytes 与产生的 durable C4 rows 一致，不重复整请求 copy。

门槛未达到时报告必须包含 Nsight Systems timeline 和 per-kernel breakdown，不能只给总吞吐。

### 14.4 报告表

| mode | GPU | input/output | concurrency | req/s | out tok/s | TTFT p99 | TPOT p99 | HBM peak | host GB | D2H GB/s | H2D GB/s | resident/hot/host % | reclaim p99 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|

## 15. 测试清单

### 15.1 CPU 单测

- allocator set invariants；
- lazy reclaim 和 allocation-time reuse；
- request generation；
- page state transitions；
- watermark/profitability；
- hot reservation rollback；
- finish/retract all states；
- transfer completion duplication/out-of-order；
- host pool capacity；
- scheduler budget。

### 15.2 CUDA kernel 测试

- DSV4 page backup bitwise；
- fused resident/hot/host swap oracle；
- padding/duplicate/boundary；
- graph capture；
- shared-index plan replay；
- TP shard pointer tables；
- compute-sanitizer。

### 15.3 GPU 集成测试

- PD prefill→decode direct path；
- low-pressure no transition；
- forced watermark；
- unpinned-not-reused；
- mixed top-k；
- full host-backed；
- concurrent finish/retract；
- host pool exhausted；
- server shutdown with transfers in flight。

## 16. 建议提交拆分

1. `refactor(hisparse): introduce coordinator protocol and DSV4 hybrid classes`
2. `feat(hisparse): add DSV4 C4 page manager and request-local resident table`
3. `feat(hisparse): add page-granular async host publication`
4. `feat(hisparse): make C4 resident and hot allocation transactional`
5. `kernel(hisparse): fuse DSV4 resident pass-through with hot/host swap-in`
6. `metrics(hisparse): add tier hit and transfer telemetry`
7. `test(hisparse): add DSV4 hybrid GPU correctness matrix`
8. `perf(hisparse): enable enqueue-time reclaim after stream-safety validation`

每个提交必须能单独运行 CPU tests；涉及 kernel 的提交必须附 kernel oracle 结果；行为切换必须有 feature flag，直到完整 GPU 报告通过。

## 17. 最终验收标准

功能完成必须同时满足：

- 每个新 C4 page/row 自动异步写 host；
- host durable 前绝不 reclaim；
- watermark 前保持 resident；
- watermark 后只回收必要 clean pages；
- unpin 后未 reuse 仍能 resident hit；
- reuse 时原子清旧 mapping；
- resident/hot/host mixed top-k 由单个 fused kernel 解析；
- CUDA Graph、TP、PD、finish/retract 全部覆盖；
- native vs hybrid greedy token 100% 一致；
- GSM8K 与性能报告完整；
- H20/H100 至少各有一次压力场景结果；
- 无 allocator leak、host leak、double free、stale callback、全局 stream synchronize。

## 18. 参考实现

- [vLLM HiSparse coordinator](https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/hisparse_coordinator.py)
- [vLLM cache managers](https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/single_type_kv_cache_manager.py)
- [vLLM HiSparse worker connector](https://github.com/vllm-project/vllm/blob/main/vllm/distributed/kv_transfer/kv_connector/v1/hisparse/worker.py)
- [vLLM HiSparse runtime](https://github.com/vllm-project/vllm/blob/main/vllm/v1/hisparse/runtime.py)
- [vLLM HiSparse CUDA kernels](https://github.com/vllm-project/vllm/blob/main/csrc/libtorch_stable/hisparse_kernels.cu)
- [SGLang PR #35488](https://github.com/sgl-project/sglang/pull/35488)：`hisparse_protocol.py`、`hisparse_hicache_coordinator.py`、`hisparse_hicache_admission.py` 和 fused pass-through kernel changes
- 当前 MVP：`my_development/dsv4_hybrid_hisparse_lazy_unpin_mvp.md`

## 19. 推荐的立即下一步

下一轮先做 Phase 1，不写新 CUDA 算子：

1. 引入 protocol；
2. 从现有 coordinator 拆出 DSV4 hybrid coordinator；
3. 新建 page manager 和 transfer engine，但内部仍调用现有 copy kernel；
4. 将 page metadata 改成 request-local page index；
5. 保持现有 Python mixed top-k 作为 correctness oracle；
6. 扩充 CPU state-machine/finish/retract tests；
7. 在 GPU 上产出 current-MVP vs refactor 的 token 等价报告。

完成这一步后再写 fused hybrid swap-in。这样算子输入、page table 和生命周期已经稳定，避免在控制面仍变化时反复改 kernel ABI。
