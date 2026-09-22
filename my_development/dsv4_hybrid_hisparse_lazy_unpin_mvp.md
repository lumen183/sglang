# DSV4 Hybrid HiSparse Lazy-Unpin MVP 实现方案

更新日期：2026-09-22

## 1. MVP 目标

本轮直接实现 vLLM hybrid HiSparse 最关键的两条语义：

1. C4 KV 一旦产生，就异步写回 pinned host memory；GPU 压力出现前仍从原始 C4 cache 读取。
2. host 副本 durable 后，低压时保持 pinned；达到 watermark 后变为 unpinned/reclaimable。unpinned page 的物理内容和 logical mapping 继续有效，直到 allocator 真正把该 page 分配给新 owner；复用发生时才使旧 owner 的 mapping 失效。

这里的“落盘”均指写入 host memory，不涉及 SSD。

本 MVP 不写新算子，继续使用已有的：

- DSV4 resident C4 读取路径；
- HiSparse host→device swap-in kernel；
- host backup kernel；
- sparse attention kernel。

为了避免扩大范围，本轮继续要求：

- DSV4 only；
- PD disaggregation；
- `--disable-radix-cache`；
- `--cuda-graph-backend-decode disabled`；
- private pinned host pool；
- 不支持 speculative decoding；
- 不处理通用 DSA/GLM compatibility。

## 2. 为什么必须补 lazy unpin

当前工作分支已经实现异步整请求 mirror 和请求级 resident→host-hot demotion，但 pressure reclaim 会直接调用 `alloc_device_buffer()`：清空整个请求的 C4 mapping、释放旧页并切换成固定 hot buffer。

这与 vLLM 有两个本质差异：

- 已写回 host 的 GPU page 在真正需要复用之前就失效，丢掉了可继续命中的 HBM cache；
- 回收粒度是整个请求，不是 page，容易产生超额回收和额外 H2D。

因此，请求级 demotion 可以作为 emergency fallback，但不能作为向 vLLM 对齐的最终 MVP 主路径。

## 3. 当前 SGLang 的缺口

`PagedTokenToKVPoolAllocator` 只有 allocated/free 两种状态。`free_page_ids()` 会直接把 page 放回 `free_pages`，但 allocator 不知道旧 owner，也没有“被下一次分配时通知旧 owner”的 callback。

当前 C4 mapping：

```text
compressed logical location
        |
        v
full_to_hisparse_device_index_mapping
        |
        v
physical C4 slot
```

如果简单把仍被 mapping 引用的 page 放入 `free_pages`，后续 allocation 会静默覆盖它，而旧 mapping 仍指向新请求的数据。因此 lazy unpin 不能只在 coordinator 中模拟，必须补 allocator 的 reuse notification。

## 4. 所有权边界

| 模块 | 职责 |
|---|---|
| scheduler | 在安全边界触发压力评估；先完成 allocation/reuse callback，再启动 forward |
| HiSparse coordinator | request/page 状态机、host publication、选择需要 unpin 的请求和页、mixed top-k 组装 |
| C4 allocator | physical page pin/unpin/free/reuse；复用前通知旧 owner；保证页不会重复进入 free queue |
| C4 pool/mapping | compressed logical→physical slot；mapping 在 unpin 后保留，在 reuse callback 中清除 |
| host pool | 每个 C4 logical page/row 的 host destination 和生命周期 |
| 现有 kernel | D2H backup、H2D miss load、sparse attention，不修改 |

核心原则：策略在 coordinator，physical page 所有权只由 allocator 修改。

## 5. 页状态机

每个 active request 的每个 C4 page 记录：

```text
DIRTY_PINNED
    GPU 有唯一有效副本，不可回收
        |
        | enqueue async D2H
        v
WRITE_PENDING
    GPU pinned；host copy 尚未 durable
        |
        | completion event observed
        v
CLEAN_PINNED
    GPU/host 都有效；低压时继续 pinned
        |
        | pressure + hot region ready
        v
CLEAN_UNPINNED
    mapping 仍指向 GPU page；page 已进入 allocator reusable queue
        |                         |
        | request/top-k 读取      | allocator 重新分配该 page
        | 仍直接使用 GPU          v
        |                     HOST_ONLY
        |                     reuse callback 清 mapping
        |                     后续 top-k 从 host/hot 获取
        |
        | request finished before reuse
        v
DETACHED
    移除 owner；page 已在 reusable queue，不得再次 free
```

新产生但尚未 sealed 的 active tail page 始终保持 `DIRTY_PINNED`。MVP 保守保护最后两个 C4 pages，避免本轮 forward 正在写或下一轮继续追加的 page 被复用。

请求状态不再用单个 `_resident_mask` 表示全部页是否 resident。只保留：

```text
NORMAL_RESIDENT       尚未分配 hot region，所有可用历史应 resident
HYBRID_HOT_READY      hot region 已分配，可存在 resident/host mixed pages
FINISHED
```

## 6. Allocator 设计

### 6.1 新增 DSV4 专用 allocator

不要改变通用 `PagedTokenToKVPoolAllocator` 的语义。增加一个仅由 DSV4 hybrid C4 使用的子类或组合包装，例如：

```python
class ReclaimablePagedTokenToKVPoolAllocator(PagedTokenToKVPoolAllocator):
    def unpin_page_ids(self, page_ids, owners): ...
    def detach_unpinned_page_ids(self, page_ids, owner): ...
    def repin_page_ids(self, page_ids, owner): ...       # MVP 可选
    def get_num_reclaimable_pages(self) -> int: ...
```

内部至少维护：

```python
reuse_owner_by_page: dict[int, PageOwner]
reclaimable_page_ids: set[int]
```

`PageOwner` 包含：

```python
req_pool_idx
request_generation
request_page_idx
compressed_logical_start
compressed_logical_end
on_reuse callback
```

`request_generation` 用来避免 req slot 被复用后，旧 callback 错误清除新请求 mapping。

### 6.2 unpin 语义

`unpin_page_ids()`：

1. 验证 page 当前不在普通 free/reclaimable 集合；
2. 注册 owner/callback；
3. 把 page 加入 allocator 可分配队列；
4. 不清 mapping；
5. 不修改 page bytes；
6. `available_size()` 将该 page 计入可分配容量。

### 6.3 reuse 语义

`alloc()`、`alloc_extend()`、`alloc_decode()` 从可分配队列取 page 后，在把 slot 返回给新 owner前：

1. 判断 page 是否有旧 reuse owner；
2. 调用旧 owner callback；
3. callback 校验 request generation；
4. 将该 page 对应的旧 logical mapping 清零；
5. coordinator 把旧 request page 标记为 `HOST_ONLY`；
6. 删除 watcher；
7. allocation 才能把 page 交给新 owner并写入新 KV。

callback 必须发生在 scheduler/allocator 路径中，并且早于下一次 model forward。MVP 不允许从 CUDA callback/thread 中直接修改 Python request state。

### 6.4 finish/free 语义

- pinned page：正常 free；
- unpinned 但尚未复用：只 detach owner，不能再次加入 free queue；
- 已复用 page：旧 request 不再拥有 physical page，不得 free；
- hot buffer page：由 request 独占，finish 时正常 free；
- `clear()`：清空 free queue、watcher、reclaimable set 和 generation state。

开启 `SGLANG_DEBUG_MEMORY_POOL` 时检查：

```text
普通 free page 不重复
reclaimable page 不重复
pinned 与 available 集合不相交
每个 watcher 指向一个 active generation
mapping 指向的 physical page 要么 pinned，要么由相同 owner unpinned
```

## 7. Host write-back 与 publication

### 7.1 Prefill

PD decode 收到完整 C4 后：

1. 保持现有 resident C4 mapping；
2. 为全部 C4 rows 分配 host slots；
3. 在 write-back stream 异步执行 D2H；
4. 将完整页标记为 `WRITE_PENDING`；
5. event 完成后标记 `CLEAN_PINNED`；
6. 最后一个非完整页和最后两个 active tail pages 保持 pinned。

现有 `_enqueue_resident_mirror()` 可以继续作为第一版批量 prefill mirror，但完成事件要发布到 page ledger，而不是只记录 request-level ready。

### 7.2 Decode

现有 `_eager_backup_previous_token()` 已经为新产生的 compressed row 异步写 host。需要补充 publication bookkeeping：

1. 每写一个 compressed row，记录它属于哪个 request/page；
2. D2H event 完成后更新该页的 durable row count；
3. page sealed 且所有有效 rows durable 后，转为 `CLEAN_PINNED`；
4. 当前 active tail pages 即使 clean 也不 unpin；
5. 不允许仅凭“copy 已 enqueue”将 page 标为 clean。

MVP 可以用一个 batch completion event 发布本轮所有 row，不要求每个 page 一个 CUDA event。

## 8. Watermark 与 hot region

watermark 保持 vLLM 语义：

```text
hot_cost_blocks = ceil(device_buffer_size / c4_page_size)
transition_watermark = max(hot_cost_blocks, floor(total_c4_blocks * 0.1))
```

压力判断使用：

```text
free_or_reclaimable_blocks < transition_watermark
```

但 hot buffer 必须在允许 resident page 丢失之前完成分配。流程：

1. scheduler boundary 检测低于 watermark；
2. 选择最老的、未 protected 的 request；
3. 为该 request 预留固定 hot region；
4. hot region 成功后将 request 标为 `HYBRID_HOT_READY`；
5. 将它的 `CLEAN_PINNED` sealed pages unpin；
6. `DIRTY_PINNED/WRITE_PENDING/active tail` 不动；
7. 重复直到可分配容量恢复到 watermark。

为避免“已经低于 hot cost，无法先分配 hot buffer”的死锁，C4 allocation 的容量检查要预留 transition reserve：

```text
required = requested_blocks + transition_watermark
```

若 admission/大块分配越过水位，allocator 在提交新分配前请求 coordinator transition。请求级 `alloc_device_buffer()` 只保留为 emergency fallback，并记录 warning；正常测试不应触发。

## 9. 不写新算子的 mixed top-k

lazy unpin 后，同一请求的 top-k 可能同时命中：

- 仍 resident 的 C4 page；
- 已复用、只能从 host 获取的 page；
- hot buffer 已缓存的位置。

MVP 用 PyTorch glue 和现有 host swap-in kernel 合并：

```python
logical_locs = resolve_topk_to_compressed_logical_locs(top_k_result)
resident_locs = mapping[logical_locs]
resident_mask = resident_locs > 0

host_topk = torch.where(
    resident_mask,
    torch.full_like(top_k_result, -1),
    top_k_result,
)
host_locs = existing_swap_in_kernel(host_topk)

output_locs = torch.where(resident_mask, resident_locs, host_locs)
```

要求现有 swap-in kernel 对 `-1` padding 保持 no-op；仓库已有 padded top-k 测试，可增加 mixed case。

这样 resident entry 零拷贝，只有 mapping 已被 reuse callback 清除的位置产生 H2D。该路径是 eager-only，性能不是本轮验收目标。

特殊情况：

- 全 resident：保持现有 fast path，不调用 swap-in；
- 全 host：保持现有 host path；
- mixed：使用上面的 mask/merge；
- invalid/out-of-range top-k：输出 `-1`。

## 10. Scheduler 安全边界

每轮顺序必须是：

```text
1. 收集上一轮 D2H completion，发布 CLEAN pages
2. 处理 finish/retract
3. 根据 watermark 分配 hot region、unpin clean pages
4. allocator 为新 token/request 分配 page
5. allocation reuse callback 清旧 mapping、更新 HOST_ONLY
6. 构建本轮 batch metadata/top-k 路径
7. launch forward
8. enqueue 新产生 C4 rows 的异步 D2H
```

禁止在 forward 已启动后到其 consumer 完成前复用本轮仍可能读取的 C4 page。当前 batch 的 req indices 要加入 protected set；active tail pages 固定 pinned。

## 11. 代码改动范围

### 必改

- `python/sglang/srt/mem_cache/allocator/hisparse.py`
  - 接入 reclaimable C4 allocator；
  - 暴露 pin/unpin/detach/reuse API；
  - 修正 available/full_available capacity 统计。
- `python/sglang/srt/mem_cache/allocator/paged.py`
  - 若采用子类，仅增加最小 protected hook，例如 allocation 取页后的 `_on_pages_allocated(page_ids)`；
  - 不改变普通 allocator 行为。
- `python/sglang/srt/managers/hisparse_coordinator.py`
  - request/page ledger；
  - D2H publication；
  - watermark transition；
  - reuse callback；
  - mixed top-k；
  - finish/retract 清理。
- `python/sglang/srt/managers/scheduler.py` 或现有 decode prepare hook
  - 增加一个明确的 pre-forward hybrid maintenance 调用。

### 可能需要小改

- `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py`
  - 增加 page/slot translation helper；不修改物理 layout。
- `python/sglang/srt/model_executor/model_runner.py`
  - 绑定 pre-forward maintenance；不修改 attention kernel。

### 不改

- `python/sglang/kernels/jit/csrc/kvcacheio/hisparse.cuh`
- sparse attention kernel；
- PR #35488 的 HiCache/radix 代码。

## 12. 建议提交顺序（一个开发轮次）

### Commit 1：reclaimable allocator

- 增加 allocation hook；
- 实现 unpin/detach/reuse watcher；
- available capacity 正确计入 reclaimable pages；
- 全部 CPU/mock 单测通过。

### Commit 2：page ledger 与异步 publication

- prefill mirror 建立 page ledger；
- decode row backup 更新 durable 状态；
- active tail protection；
- host durable 前禁止 unpin。

### Commit 3：watermark transition 与 lazy reuse

- 预留 hot region；
- clean page unpin；
- allocation callback 清 mapping；
- scheduler boundary maintenance；
- finish/retract 幂等。

### Commit 4：mixed top-k eager path

- resident mask；
- host miss mask；
- 调用现有 swap-in；
- merge physical locations；
- 不启用 CUDA Graph。

### Commit 5：GPUQ 验证和报告

- resident-only；
- unpinned-but-not-reused；
- partially-reused mixed；
- fully host-backed；
- baseline/native/hybrid 精度比较。

## 13. 单元测试门禁

### Allocator

1. unpin 后 `available_size()` 增加，但 mapping 不变；
2. unpin 后未复用，读取原 physical slot 内容不变；
3. 下一次 allocation 取得该 page 时 callback 恰好执行一次；
4. callback 后旧 mapping 清零；
5. detach unpinned page 不造成 double-free；
6. request generation 不匹配时 callback 不触碰新 request；
7. pinned/free/reclaimable 集合无交叉和重复；
8. `clear()` 后 watcher 和 page 集合为空。

### Coordinator

1. dirty/write-pending page 不能 unpin；
2. clean page 低压不 unpin；
3. watermark 后先成功分配 hot region，再 unpin；
4. unpinned page 未复用时 top-k 仍返回 resident slot；
5. 部分页复用后，同一 top-k 同时返回 resident 和 host-loaded slots；
6. protected/active-tail page 不回收；
7. host allocation 或 hot allocation 失败时状态不变；
8. finish/retract 覆盖每个页状态并完全恢复容量。

### Kernel 复用验证

不增加 kernel，但增加现有 kernel 测试：top-k 输入混有 `-1` 时不产生非法 host read；merge 后所有有效输出 slot 非负且内容与 reference gather 一致。

## 14. GPUQ 功能矩阵

先用 H100 8 卡，P/D 各 TP4：

```bash
cd /home/jovyan/whw/sglang

NUM_EXAMPLES=10 NUM_THREADS=1 MIN_SCORE=0 \
  my_scripts/hisparse_accuracy/run_gpuq.sh h100-8 hybrid-resident

NUM_EXAMPLES=4 NUM_THREADS=1 MIN_SCORE=0 NUM_SHOTS_EVICT=128 \
  my_scripts/hisparse_accuracy/run_gpuq.sh h100-8 hybrid-evict
```

需要把 coverage event 扩展为：

```text
page_write_enqueued
page_clean
page_unpinned
page_resident_hit_after_unpin
page_reused
mixed_topk
host_topk
```

四个必须独立覆盖的场景：

| 场景 | 预期 |
|---|---|
| 低压 resident | page clean 但不 unpin；无 H2D |
| 达 watermark、未复用 | page unpinned；仍从原 resident slot 读取 |
| 部分页复用 | 出现 `mixed_topk`；只为 lost positions H2D |
| 全部历史页复用 | 退化为现有 host-hot 路径，输出仍正确 |

准确度：baseline、native、hybrid 四模式使用同一 GSM8K 输入；200 题通过后再全量。hybrid 相对 native/baseline 的 exact-match 差异绝对值不超过 0.5pp，固定 greedy smoke 的 token 序列必须一致。

## 15. MVP 完成定义

同时满足以下条件才算完成：

- KV 产生后异步写 host，只有 completion 后才成为 clean；
- clean page 达 watermark 后可 unpin；
- unpin 不清 mapping、不改变 bytes；
- 未复用前 attention 继续直接读取原 GPU page；
- allocator 复用前 callback 清除旧 mapping；
- 同一请求 resident/host mixed top-k 正确；
- host durable 前绝不释放或复用；
- finish/retract 无泄漏、double-free、stale generation；
- 不增加新 kernel；
- PD GPUQ resident、unpin-not-reused、mixed、host-only 四条路径均有日志证据和正确输出。

## 16. 与后续版本的边界

本 MVP 采用 Python/PyTorch mask 合并 mixed top-k，目的是证明 cache semantics 正确。后续性能版本再做：

- fused resident/hot/host 三源 resolver；
- CUDA Graph static metadata；
- 更低开销的 reuse owner table；
- page-level LRU/clock policy；
- radix/HiCache backing；
- speculative decoding和更多 backend。

参考：

- [vLLM HiSparse coordinator](https://github.com/vllm-project/vllm/blob/main/vllm/v1/hisparse/coordinator.py)
- [vLLM block pool](https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/block_pool.py)
- [SGLang PR #35488](https://github.com/sgl-project/sglang/pull/35488)

## 17. 当前实现状态（2026-09-22）

第一版代码已经开始实现：

- 通用 paged allocator 增加无行为变化的 allocation hook；
- DSV4 C4 改用专用 `ReclaimablePagedTokenToKVPoolAllocator`；
- 支持 `unpin_page_ids`、owner detach 和 allocation-time reuse callback；
- coordinator 增加 request generation 和 page ledger；
- prefill mirror completion 发布 clean pages；
- decode backup completion 发布新增/追加 page；
- watermark reclaim 改为 hot allocation + clean page lazy-unpin；
- page 真正复用时清除旧 logical mapping；
- mixed top-k 使用已有 host swap-in kernel和 PyTorch mask 合并；
- finish/retract 区分 pinned、unpinned、host-only 和 hot pages。

已完成的本地门禁：

- Python `py_compile`；
- Ruff fatal/import 检查；
- `git diff --check`；
- hybrid policy 4 个纯 CPU 测试；
- 隔离的 reclaimable allocator smoke：确认 unpin 不触发 callback，allocation 复用时才触发。

Mac 的轻量 uv 环境缺少完整 SGLang runtime 依赖，临时下载依赖长时间无进展，因此完整 pytest 尚未执行。内网运行：

```bash
my_scripts/hisparse_accuracy/run_unit_gpuq.sh
```

单测通过后才能进入 H100 PD smoke。当前实现仍是第一版，GPU 测试前需重点审查 TP rank 决策一致性、decode active-tail publication 和 finish/retract 容量恢复。
